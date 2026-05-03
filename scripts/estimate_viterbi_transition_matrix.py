"""Estimate GeneCAD Viterbi transition probabilities from GFF3 annotations.

This script builds empirical token-level transition probabilities for the
five-state GeneCAD Viterbi model:

    0 intergenic
    1 intron
    2 five_prime_utr
    3 cds
    4 three_prime_utr

It parses canonical transcripts from one or more GFF3 files, converts each
transcript into a run-length encoded label sequence, counts adjacent-state
transitions, applies additive smoothing, and writes a row-normalized CSV.

Features:
- Progress bars show processing speed and ETA
- Parallel GFF parsing across multiple CPU cores (--num-workers)
- Auto-discovery of GFF files under fine-tuning/input_file/ directories

Recommended usage for truncated animal annotations:

    python genecad/scripts/estimate_viterbi_transition_matrix.py \
        --gff fine-tuning/input_file/animal \
        --output-csv animal_transition.csv \
        --alpha 1e-6 \
        --num-workers -1

For faster processing on multi-core systems, use --num-workers -1 to use
all available cores. This can speed up GFF parsing significantly when
processing many large files.

If your annotations are heavily truncated, run the default all-transcript
estimate first. Use --require-complete only as a comparison matrix when you
know the input contains mostly full-length transcripts.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import multiprocessing as mp
from functools import partial

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


STATE_NAMES = [
    "intergenic",
    "intron",
    "five_prime_utr",
    "cds",
    "three_prime_utr",
]
INTERGENIC, INTRON, FIVE_PRIME_UTR, CDS, THREE_PRIME_UTR = range(5)


@dataclass
class Transcript:
    transcript_id: str
    strand: str
    start: int
    end: int
    is_canonical: bool = False
    five_prime_utrs: list[tuple[int, int]] = field(default_factory=list)
    three_prime_utrs: list[tuple[int, int]] = field(default_factory=list)
    cds_parts: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Gene:
    gene_id: str
    contig: str
    start: int
    end: int
    strand: str
    transcripts: dict[str, Transcript] = field(default_factory=dict)


def open_text_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt")


def parse_attributes(attribute_field: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for item in attribute_field.split(";"):
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        attrs[key] = value
    return attrs


def parse_gff(gff_path: Path) -> dict[str, list[Gene]]:
    contig_genes: dict[str, list[Gene]] = {}
    gene_by_id: dict[str, Gene] = {}
    transcript_by_id: dict[str, Transcript] = {}

    with open_text_maybe_gzip(gff_path) as handle:
        for raw_line in handle:
            if not raw_line or raw_line.startswith("#"):
                continue
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue

            # Handle GFF lines with more than 9 columns by joining extra columns into attributes
            contig = parts[0]
            _source = parts[1]
            feature_type = parts[2]
            start_s = parts[3]
            end_s = parts[4]
            _score = parts[5]
            strand = parts[6]
            _phase = parts[7]
            attrs = "\t".join(parts[8:]) if len(parts) > 9 else parts[8]
            
            start = int(start_s) - 1
            end = int(end_s)
            attr_map = parse_attributes(attrs)

            if feature_type == "gene":
                gene_id = attr_map.get("ID") or attr_map.get("gene_id") or attr_map.get("Name")
                if gene_id is None:
                    continue
                gene = Gene(gene_id=gene_id, contig=contig, start=start, end=end, strand=strand)
                contig_genes.setdefault(contig, []).append(gene)
                gene_by_id[gene_id] = gene
                continue

            if feature_type == "mRNA":
                transcript_id = attr_map.get("ID") or attr_map.get("transcript_id") or attr_map.get("Name")
                parent_id = attr_map.get("Parent")
                if transcript_id is None or parent_id is None:
                    continue
                gene = gene_by_id.get(parent_id)
                if gene is None:
                    continue
                is_canonical = attr_map.get("canonical_transcript") == "1" or attr_map.get("tag") == "Ensembl_canonical"
                transcript = Transcript(
                    transcript_id=transcript_id,
                    strand=strand,
                    start=start,
                    end=end,
                    is_canonical=is_canonical,
                )
                gene.transcripts[transcript_id] = transcript
                transcript_by_id[transcript_id] = transcript
                continue

            parent_id = attr_map.get("Parent")
            if parent_id is None:
                continue
            transcript = transcript_by_id.get(parent_id)
            if transcript is None:
                continue

            if feature_type == "five_prime_UTR":
                transcript.five_prime_utrs.append((start, end))
            elif feature_type == "three_prime_UTR":
                transcript.three_prime_utrs.append((start, end))
            elif feature_type == "CDS":
                transcript.cds_parts.append((start, end))

    return contig_genes


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(intervals)
    if not ordered:
        return []
    merged: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def transcript_to_segments(transcript: Transcript) -> list[tuple[int, int, int]]:
    five_prime_utrs = merge_intervals(transcript.five_prime_utrs)
    three_prime_utrs = merge_intervals(transcript.three_prime_utrs)
    cds_parts = merge_intervals(transcript.cds_parts)

    if not cds_parts:
        return []

    # Exons in genomic order, but the label order within the transcript depends on strand.
    if transcript.strand == "+":
        exons = [(s, e, FIVE_PRIME_UTR) for s, e in five_prime_utrs]
        exons += [(s, e, CDS) for s, e in cds_parts]
        exons += [(s, e, THREE_PRIME_UTR) for s, e in three_prime_utrs]
    else:
        exons = [(s, e, THREE_PRIME_UTR) for s, e in three_prime_utrs]
        exons += [(s, e, CDS) for s, e in cds_parts]
        exons += [(s, e, FIVE_PRIME_UTR) for s, e in five_prime_utrs]

    exons.sort(key=lambda item: item[0])

    segments: list[tuple[int, int, int]] = []
    prev_end = transcript.start
    for start, end, state in exons:
        if start > prev_end:
            segments.append((prev_end, start, INTRON))
        segments.append((start, end, state))
        prev_end = end
    if prev_end < transcript.end:
        segments.append((prev_end, transcript.end, INTRON))

    merged: list[tuple[int, int, int]] = []
    for start, end, state in sorted(segments, key=lambda item: item[0]):
        if not merged:
            merged.append((start, end, state))
            continue
        last_start, last_end, last_state = merged[-1]
        if state == last_state and start <= last_end:
            merged[-1] = (last_start, max(last_end, end), state)
        else:
            merged.append((start, end, state))

    return merged


def transcript_to_labels(transcript: Transcript, require_complete: bool) -> np.ndarray | None:
    segments = transcript_to_segments(transcript)
    if not segments:
        return None

    has_five = any(state == FIVE_PRIME_UTR for _s, _e, state in segments)
    has_cds = any(state == CDS for _s, _e, state in segments)
    has_three = any(state == THREE_PRIME_UTR for _s, _e, state in segments)
    if require_complete and not (has_five and has_cds and has_three):
        return None

    labels: list[int] = []
    for start, end, state in segments:
        labels.extend([state] * max(0, end - start))
    if len(labels) < 2:
        return None
    return np.asarray(labels, dtype=np.int16)


def _process_gff_file(gff_path: Path, require_complete: bool) -> tuple[np.ndarray, int, int, str]:
    """Process a single GFF file and return transition counts.
    
    Used by multiprocessing to parallelize GFF parsing across multiple cores.
    Returns counts matrix, total transcripts, kept transcripts, and source file name.
    """
    counts = np.zeros((len(STATE_NAMES), len(STATE_NAMES)), dtype=np.float64)
    total_transcripts = 0
    kept_transcripts = 0
    
    contig_genes = parse_gff(gff_path)
    for genes in contig_genes.values():
        for gene in genes:
            canonical = [t for t in gene.transcripts.values() if t.is_canonical]
            if not canonical:
                canonical = list(gene.transcripts.values())
            if not canonical:
                continue

            transcript = max(canonical, key=lambda t: t.end - t.start)
            total_transcripts += 1
            labels = transcript_to_labels(transcript, require_complete=require_complete)
            if labels is None:
                continue
            kept_transcripts += 1
            for left, right in zip(labels[:-1], labels[1:]):
                counts[left, right] += 1
    
    return counts, total_transcripts, kept_transcripts, gff_path.name


def estimate_transition_counts(gff_paths: list[Path], require_complete: bool, num_workers: int = 1) -> tuple[np.ndarray, int, int]:
    """Estimate transition counts from one or more GFF files.
    
    Parameters
    ----------
    gff_paths : list[Path]
        Paths to GFF3 files to process.
    require_complete : bool
        Whether to require complete transcripts (5'UTR + CDS + 3'UTR).
    num_workers : int
        Number of parallel workers for GFF parsing (default: 1, serial processing).
        Set to -1 to use all available cores, or specify a number > 1 for parallel processing.
    
    Returns
    -------
    tuple[np.ndarray, int, int]
        Aggregated transition counts, total transcripts, kept transcripts.
    """
    counts = np.zeros((len(STATE_NAMES), len(STATE_NAMES)), dtype=np.float64)
    total_transcripts = 0
    kept_transcripts = 0

    if num_workers == 1 or len(gff_paths) == 1:
        # Serial processing with progress bar
        for gff_path in tqdm(gff_paths, desc="Processing GFF files", unit="file"):
            gff_counts, gff_total, gff_kept, fname = _process_gff_file(gff_path, require_complete)
            counts += gff_counts
            total_transcripts += gff_total
            kept_transcripts += gff_kept
            print(f"  {fname}: {gff_kept}/{gff_total} transcripts kept")
    else:
        # Parallel processing
        actual_workers = num_workers if num_workers > 1 else mp.cpu_count()
        print(f"Parallel GFF processing with {actual_workers} workers...")
        
        with mp.Pool(actual_workers) as pool:
            worker_fn = partial(_process_gff_file, require_complete=require_complete)
            results = [
                result
                for result in tqdm(
                    pool.imap_unordered(worker_fn, gff_paths),
                    total=len(gff_paths),
                    desc="Processing GFF files",
                    unit="file",
                )
            ]
        
        for gff_counts, gff_total, gff_kept, fname in results:
            counts += gff_counts
            total_transcripts += gff_total
            kept_transcripts += gff_kept
            print(f"  {fname}: {gff_kept}/{gff_total} transcripts kept")

    return counts, total_transcripts, kept_transcripts


def normalize_rows(counts: np.ndarray, alpha: float) -> np.ndarray:
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    smoothed = counts + alpha
    row_sums = smoothed.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Encountered an empty transition row even after smoothing")
    return smoothed / row_sums


def stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    eigenvals, eigenvecs = np.linalg.eig(transition_matrix.T)
    idx = np.argmin(np.abs(eigenvals - 1.0))
    dist = np.real(eigenvecs[:, idx])
    dist = dist / dist.sum()
    return dist


def write_matrix_csv(path: Path, matrix: np.ndarray, row_names: list[str], col_names: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + col_names)
        for row_name, row in zip(row_names, matrix):
            writer.writerow([row_name] + [f"{value:.12g}" for value in row])


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate GeneCAD Viterbi transition priors from GFF3")
    parser.add_argument(
        "--gff",
        required=False,
        help=(
            "Comma-separated list of GFF3 or GFF3.GZ files, directories, or globs. "
            "If omitted, the script will try to discover GFFs under "
            "fine-tuning/input_file/ and fine-tuning/ directories."),
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to write the normalized transition matrix CSV",
    )
    parser.add_argument(
        "--output-stationary-csv",
        default=None,
        help="Optional path to write the stationary distribution CSV",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-6,
        help="Additive smoothing value used before row normalization",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Keep only transcripts containing 5'UTR, CDS, and 3'UTR",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel workers for GFF parsing (default: 1, serial). "
            "Set to -1 to use all available CPU cores, or specify a number > 1 for parallel processing. "
            "Parallel processing can significantly speed up GFF parsing when processing many files."
        ),
    )
    args = parser.parse_args()

    raw_input = args.gff or ""
    candidates: list[Path] = []
    explicit_input_provided = bool(raw_input.strip())
    if explicit_input_provided:
        parts = [p.strip() for p in raw_input.split(",") if p.strip()]
        for part in parts:
            p = Path(part).expanduser()
            # directory -> collect gff files recursively
            if p.exists() and p.is_dir():
                for ext in ("**/*.gff", "**/*.gff3", "**/*.gff.gz", "**/*.gff3.gz"):
                    candidates.extend(sorted(p.glob(ext)))
                continue
            # glob pattern (contains wildcard) -> glob from cwd
            if any(ch in part for ch in "*?[]"):
                import glob as _glob

                for matched in _glob.glob(part, recursive=True):
                    candidates.append(Path(matched).expanduser())
                continue
            # file path (may not exist yet)
            candidates.append(p)

    # If no explicit input was provided, try repo-default locations
    if not explicit_input_provided:
        repo_root = Path.cwd()
        search_dirs = [repo_root / "fine-tuning" / "input_file", repo_root / "fine-tuning"]
        for d in search_dirs:
            if not d.exists():
                continue
            for ext in ("**/*.gff", "**/*.gff3", "**/*.gff.gz", "**/*.gff3.gz"):
                candidates.extend(sorted(d.glob(ext)))

    # Finalize list: keep only existing files
    gff_paths = [p for p in dict.fromkeys(candidates) if p.exists() and p.is_file()]
    if not gff_paths:
        if explicit_input_provided:
            raise SystemExit(
                "No GFF input files found from --gff. "
                "When passing a directory, files are searched recursively for "
                "*.gff, *.gff3, *.gff.gz, *.gff3.gz"
            )

        sample_search = []
        for d in (Path.cwd() / "fine-tuning" / "input_file", Path.cwd() / "fine-tuning"):
            if d.exists():
                for ext in ("**/*.gff", "**/*.gff3", "**/*.gff.gz", "**/*.gff3.gz"):
                    sample_search.extend(sorted(d.glob(ext)))
        sample_list = [str(p) for p in sample_search[:20]]
        raise SystemExit(
            "No GFF input files found. Provide --gff with files, directories, or globs. "
            f"Auto-discovered candidates (first 20): {sample_list}"
        )

    mode_desc = "--gff input" if explicit_input_provided else "auto-discovery"
    print(f"Resolved {len(gff_paths)} GFF files via {mode_desc}")

    counts, total, kept = estimate_transition_counts(
        gff_paths,
        require_complete=args.require_complete,
        num_workers=args.num_workers
    )
    transition_matrix = normalize_rows(counts, args.alpha)
    stationary = stationary_distribution(transition_matrix)

    output_csv = Path(args.output_csv).expanduser()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_matrix_csv(output_csv, transition_matrix, STATE_NAMES, STATE_NAMES)

    if args.output_stationary_csv is not None:
        stationary_path = Path(args.output_stationary_csv).expanduser()
        stationary_path.parent.mkdir(parents=True, exist_ok=True)
        with stationary_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([""] + STATE_NAMES)
            writer.writerow(["pi"] + [f"{value:.12g}" for value in stationary])

    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Total GFF files processed: {len(gff_paths)}")
    print(f"  Total transcripts examined: {total:,}")
    print(f"  Transcripts kept after filtering: {kept:,} ({100*kept/max(total,1):.1f}%)")
    print(f"  Transition matrix written to: {output_csv}")
    if args.output_stationary_csv is not None:
        print(f"  Stationary distribution written to: {args.output_stationary_csv}")
    print(f"  Alpha (smoothing): {args.alpha}")
    print(f"  Require complete transcripts: {args.require_complete}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
