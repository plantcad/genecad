"""Hybrid decoding: plain Viterbi genome-wide, frame-aware decoding only where a gene exists.

Frame-aware decoding of whole chromosomes (``frame_crf``) guarantees every CDS
is a valid ORF, but forcing a reading frame everywhere also creates new loci
made of tiny ORFs and cuts long genes into several ORFs.  Plain (5-state)
Viterbi creates neither, but leaves some genes without a valid ORF.  This
module starts from the plain decode after ``fix_orf`` and uses frame-aware
decoding only locally, on the model's own feature logits:

1. rescue: every transcript ``fix_orf`` could not repair (``partial=true``) is
   re-decoded frame-aware in a window around it.  A decoded transcript
   replaces it if it overlaps it, has more than ``min_cds`` nt of CDS and
   does not overlap a kept gene on the same strand.  Unrescued partial genes
   are dropped (or kept, flagged, with ``keep_partial``).
2. merge: consecutive same-strand genes at most ``max_gap`` apart are
   re-decoded frame-aware with intergenic forbidden between them.  If that
   yields one transcript spanning the CDS midpoints of both, it replaces the
   pair (and may be merged again with the next gene).

Frame-aware decoding only ever runs where a gene already exists, so it cannot
create new loci.  On 29 held-out maize NAM chromosomes (3 lines) this raised
exact-CDS recall from 57.6% to 58.6% and precision from 59.8% to 62.5% over
frame-aware decoding with partial transcripts dropped, and cut split
reference genes by 39% (3936 -> 2385), at the cost of 5.6% more fused genes.

Genes that are not changed are written back verbatim.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from src.atomic_io import atomic_output_path
from src.frame_crf import (
    AT_AC,
    GT_AG,
    FrameStateGraph,
    load_chromosome_codes,
    reverse_complement_codes,
)

logger = logging.getLogger(__name__)

INTERGENIC, INTRON, UTR5, CDS, UTR3 = range(5)
FEATURE_NAME = {UTR5: "five_prime_UTR", CDS: "CDS", UTR3: "three_prime_UTR"}
EXONIC_TYPES = frozenset(FEATURE_NAME.values())

Feature = tuple[str, int, int]  # (type, 1-based start, 1-based inclusive end)


class DamagedPredictions(Exception):
    """The prediction files of a sequence fail verification; predicting the sequence
    again repairs them."""

    def __init__(self, predictions_dir: str, reason: str):
        super().__init__(predictions_dir, reason)
        self.predictions_dir = predictions_dir
        self.reason = reason

    def __str__(self) -> str:
        return f"{self.predictions_dir}: {self.reason}"


@dataclass
class Gene:
    """One gene, described by its first transcript's exonic features.

    ``lines`` holds the gene's original GFF lines; a gene that keeps them is
    written back unchanged.  Genes created here have no lines.
    """

    seqid: str
    strand: str
    feats: list[Feature]
    gene_id: str
    partial: bool = False
    hybrid: str | None = None
    merged_from: tuple[str, ...] = ()
    lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.feats = sorted(self.feats, key=lambda f: (f[1], f[2]))

    @property
    def start(self) -> int:
        if not self.feats:  # no exonic features: fall back to the gene line
            return int(self.lines[0].split("\t")[3])
        return min(f[1] for f in self.feats)

    @property
    def end(self) -> int:
        if not self.feats:
            return int(self.lines[0].split("\t")[4])
        return max(f[2] for f in self.feats)

    @property
    def cds_len(self) -> int:
        return sum(e - s + 1 for t, s, e in self.feats if t == "CDS")

    @property
    def cds_mid(self) -> int:
        cds = [(s, e) for t, s, e in self.feats if t == "CDS"]
        return (min(s for s, _ in cds) + max(e for _, e in cds)) // 2


def overlaps(start: int, end: int, gene: Gene) -> bool:
    return start <= gene.end and gene.start <= end


def transcripts_from_labels(
    labels: np.ndarray, lo: int, strand: str
) -> list[list[Feature]]:
    """Split window labels into transcripts.

    ``labels`` are in genome orientation and cover 0-based positions
    ``[lo, lo + len(labels))``.  A transcript ends at intergenic, or where a
    3' UTR is followed by a new 5' UTR or CDS in transcription order.
    Transcripts without CDS are dropped.  Features are 1-based and sorted.
    """
    runs: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    order = range(len(labels)) if strand == "+" else range(len(labels) - 1, -1, -1)
    prev = INTERGENIC
    for i in order:
        label = int(labels[i])
        new_transcript = label != INTERGENIC and (
            prev == INTERGENIC or (prev == UTR3 and label in (UTR5, CDS))
        )
        if (label == INTERGENIC or new_transcript) and current:
            runs.append(current)
            current = []
        if label != INTERGENIC:
            current.append((i + lo + 1, label))
        prev = label
    if current:
        runs.append(current)

    transcripts = []
    for run in runs:
        feats: list[Feature] = []
        block: list[int] | None = None  # [label, start, end]
        for pos, label in sorted(run):
            if block and block[0] == label and block[2] == pos - 1:
                block[2] = pos
                continue
            if block and block[0] in FEATURE_NAME:
                feats.append((FEATURE_NAME[block[0]], block[1], block[2]))
            block = [label, pos, pos]
        if block and block[0] in FEATURE_NAME:
            feats.append((FEATURE_NAME[block[0]], block[1], block[2]))
        if any(t == "CDS" for t, _, _ in feats):
            transcripts.append(feats)
    return transcripts


class LocalDecoder:
    """Frame-aware decoding of arbitrary windows of one chromosome.

    Parameters
    ----------
    logits
        ``logits(strand, lo, hi)`` returns the model's feature logits for
        0-based positions ``[lo, hi)`` of ``strand`` ("+" or "-"), shape
        ``(hi - lo, 5)``, genome orientation, columns ordered as
        ``[intergenic, intron, five_prime_utr, cds, three_prime_utr]``.
    codes
        The chromosome from :func:`src.frame_crf.encode_sequence`.
    transition
        The 5x5 feature transition matrix.
    """

    def __init__(
        self,
        logits: Callable[[str, int, int], np.ndarray],
        codes: np.ndarray,
        transition: np.ndarray,
        graph: FrameStateGraph | None = None,
    ) -> None:
        self.logits = logits
        self.codes = codes
        self.transition = transition
        self.graph = graph or FrameStateGraph()

    def decode(
        self, strand: str, lo: int, hi: int, forbid: tuple[int, int] | None = None
    ) -> np.ndarray:
        """Labels for 0-based ``[lo, hi)`` in genome orientation.

        ``forbid`` is a 0-based ``[a, b)`` range where intergenic is disallowed.
        """
        logits = np.asarray(self.logits(strand, lo, hi), dtype=np.float64)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        if forbid:
            probs[max(forbid[0] - lo, 0) : max(forbid[1] - lo, 0), INTERGENIC] = 0.0
        codes = self.codes[lo:hi]
        if strand == "-":
            labels = self.graph.decode(
                np.ascontiguousarray(probs[::-1]),
                reverse_complement_codes(codes),
                self.transition,
            )
            return np.flip(labels)
        return self.graph.decode(probs, codes, self.transition)


def rescue(
    genes: list[Gene],
    decoder: LocalDecoder,
    length: int,
    flank: int,
    min_cds: int,
    keep_partial: bool = False,
) -> tuple[list[Gene], Counter]:
    """Replace partial genes by a frame-aware decode of a window around them."""
    kept = [g for g in genes if not g.partial]
    by_strand: dict[str, list[Gene]] = {"+": [], "-": []}
    for g in kept:
        by_strand[g.strand].append(g)

    out = list(kept)
    seen: set[tuple[str, tuple[Feature, ...]]] = set()
    stats: Counter = Counter()
    for g in (g for g in genes if g.partial):
        lo, hi = max(g.start - 1 - flank, 0), min(g.end + flank, length)
        found = []
        windows = (
            transcripts_from_labels(decoder.decode(g.strand, lo, hi), lo, g.strand)
            if g.cds_len
            else []
        )
        for feats in windows:
            t = Gene(g.seqid, g.strand, feats, g.gene_id, hybrid="rescued")
            if not overlaps(t.start, t.end, g):
                continue
            if t.cds_len <= min_cds:
                stats["rescue: tiny, dropped"] += 1
                continue
            if any(
                overlaps(t.start, t.end, k)
                for k in by_strand[g.strand]
                if overlaps(lo + 1, hi, k)
            ):
                stats["rescue: overlaps a kept gene, dropped"] += 1
                continue
            found.append(t)
        for n, t in enumerate(found):
            key = (t.strand, tuple(t.feats))
            if key in seen:  # neighbouring partial genes can share a window
                continue
            seen.add(key)
            if n:
                t.gene_id = f"{g.gene_id}_{n + 1}"
            out.append(t)
        if found:
            stats["partial genes rescued"] += 1
        else:
            stats["partial genes dropped"] += 1
            if keep_partial:
                out.append(g)
    return out, stats


def merge(
    genes: list[Gene],
    decoder: LocalDecoder,
    length: int,
    flank: int,
    max_gap: int,
    min_cds: int,
) -> tuple[list[Gene], Counter]:
    """Join consecutive same-strand genes that a forced frame-aware decode spans."""
    out: list[Gene] = [g for g in genes if g.cds_len == 0]
    stats: Counter = Counter()
    for strand in ("+", "-"):
        ordered = sorted(
            (g for g in genes if g.strand == strand and g.cds_len),
            key=lambda g: g.start,
            reverse=(strand == "-"),
        )
        cur: Gene | None = None
        for nxt in ordered:
            if cur is None:
                cur = nxt
                continue
            gap = (
                (nxt.start - cur.end - 1)
                if strand == "+"
                else (cur.start - nxt.end - 1)
            )
            merged = None
            if 0 < gap <= max_gap:
                stats["pairs tried"] += 1
                lo = max(min(cur.start, nxt.start) - 1 - flank, 0)
                hi = min(max(cur.end, nxt.end) + flank, length)
                forbid = (
                    (cur.end, nxt.start - 1)
                    if strand == "+"
                    else (nxt.end, cur.start - 1)
                )
                a_mid, b_mid = cur.cds_mid, nxt.cds_mid
                for feats in transcripts_from_labels(
                    decoder.decode(strand, lo, hi, forbid), lo, strand
                ):
                    t = Gene(cur.seqid, strand, feats, cur.gene_id, hybrid="merged")
                    if (
                        t.start <= min(a_mid, b_mid)
                        and t.end >= max(a_mid, b_mid)
                        and t.cds_len > min_cds
                    ):
                        t.merged_from = (cur.merged_from or (cur.gene_id,)) + (
                            nxt.merged_from or (nxt.gene_id,)
                        )
                        merged = t
                        break
            if merged is not None:
                stats["pairs merged"] += 1
                cur = merged
            else:
                out.append(cur)
                cur = nxt
        if cur is not None:
            out.append(cur)
    return out, stats


def process_sequence(
    seqid: str,
    genes: list[Gene],
    input_fasta: str,
    predictions_dir: str,
    domain: str,
    flank: int,
    max_gap: int,
    min_cds: int,
    keep_partial: bool,
    graph_options: dict | None = None,
    *,
    codes: np.ndarray | None = None,
) -> tuple[str, list[Gene], Counter]:
    """Rescue then merge the genes of one sequence, reading its FASTA record and
    prediction store.  ``graph_options`` are FrameStateGraph parameters, plus
    ``allow_u12_introns``.  ``codes`` are the encoded bases of the sequence, read
    from the FASTA if not given.  Returns ``(seqid, genes, stats)``."""
    from src.modeling import GeneClassifierConfig, token_transition_probs
    from src.prediction import merge_prediction_datasets, open_segments

    if codes is None:
        codes = load_chromosome_codes(input_fasta, seqid)
    names = GeneClassifierConfig().token_entity_names_with_background()
    try:
        segments = open_segments(predictions_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise DamagedPredictions(
            predictions_dir, f"{type(error).__name__}: {error}"
        ) from error
    if segments is None:
        # Legacy rank stores cannot be read piecewise
        ds = merge_prediction_datasets(
            predictions_dir, drop_variables=["token_predictions", "token_logits"]
        )
        logits = ds["feature_logits"].sel(feature=names)
        n_positions = logits.sizes["sequence"]
    else:
        n_positions = segments.length
    if n_positions != len(codes):
        raise ValueError(
            f"Sequence {seqid!r} has {len(codes)} bases but {n_positions} "
            f"positions were predicted; hybrid decoding requires the FASTA used for prediction"
        )
    transition = token_transition_probs(remove_incomplete_features=True, domain=domain)
    if transition.columns.tolist() != names:
        raise ValueError(
            f"Transition matrix columns {transition.columns.tolist()} != {names}"
        )

    def window(strand: str, lo: int, hi: int):
        name = "positive" if strand == "+" else "negative"
        if segments is not None:
            return segments.window(name, lo, hi, names)
        s = logits.sel(strand=name)
        return s.isel(sequence=slice(lo, hi)).transpose("sequence", "feature").values

    options = dict(graph_options or {})
    if options.pop("allow_u12_introns", False):
        options["splice_motif_groups"] = (GT_AG, AT_AC)
    decoder = LocalDecoder(window, codes, transition.values, FrameStateGraph(**options))
    rescued, stats = rescue(genes, decoder, len(codes), flank, min_cds, keep_partial)
    merged, merge_stats = merge(rescued, decoder, len(codes), flank, max_gap, min_cds)
    stats.update(merge_stats)
    logger.info(f"{seqid}: {len(genes)} genes in, {len(merged)} out; {dict(stats)}")
    return seqid, merged, stats


# -------------------------------------------------------------------------------------------------
# GFF input / output
# -------------------------------------------------------------------------------------------------


def _attributes(text: str) -> dict[str, str]:
    return dict(kv.split("=", 1) for kv in text.split(";") if "=" in kv)


def read_genes(path: str) -> tuple[list[str], dict[str, list[Gene]]]:
    """Read a GeneCAD GFF into genes (first transcript each), grouped by seqid.

    Returns the ``#`` header lines and ``{seqid: [Gene, ...]}`` in file order.
    Every line of a gene (all its transcripts) is kept in ``Gene.lines``.
    """
    header: list[str] = []
    genes: dict[str, Gene] = {}
    owner: dict[
        str, str
    ] = {}  # feature ID -> gene ID, for transcripts and their children
    first_tx: dict[str, str] = {}
    by_seqid: dict[str, list[Gene]] = {}
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                if not genes:
                    header.append(line)
                continue
            f = line.split("\t")
            attrs = _attributes(f[8])
            if f[2] == "gene":
                g = Gene(
                    f[0],
                    f[6],
                    [],
                    attrs["ID"],
                    partial=attrs.get("partial") == "true",
                    lines=[line],
                )
                genes[g.gene_id] = g
                owner[g.gene_id] = g.gene_id
                by_seqid.setdefault(f[0], []).append(g)
                continue
            parent = attrs.get("Parent", "").split(",")[0]
            gene_id = owner[parent]
            g = genes[gene_id]
            g.lines.append(line)
            if f[2] == "mRNA":
                owner[attrs["ID"]] = gene_id
                if first_tx.setdefault(gene_id, attrs["ID"]) == attrs["ID"]:
                    g.partial = attrs.get("partial") == "true"
            elif f[2] in EXONIC_TYPES and parent == first_tx.get(gene_id):
                g.feats.append((f[2], int(f[3]), int(f[4])))
            elif "ID" in attrs:
                owner[attrs["ID"]] = gene_id
    for g in genes.values():
        g.feats.sort(key=lambda feat: (feat[1], feat[2]))
    return header, by_seqid


def _gene_lines(g: Gene) -> list[str]:
    if g.lines:
        return g.lines
    gene_attrs = f"ID={g.gene_id}"
    if g.hybrid:
        gene_attrs += f";hybrid={g.hybrid}"
    if g.merged_from:
        gene_attrs += f";merged_from={','.join(g.merged_from)}"
    tx = f"{g.gene_id}.t1"
    lines = [
        f"{g.seqid}\tGeneCAD\tgene\t{g.start}\t{g.end}\t.\t{g.strand}\t.\t{gene_attrs}",
        f"{g.seqid}\tGeneCAD\tmRNA\t{g.start}\t{g.end}\t.\t{g.strand}\t.\t"
        f"ID={tx};Parent={g.gene_id};orf_status=complete",
    ]
    cds = sorted(
        (f for f in g.feats if f[0] == "CDS"),
        key=lambda f: f[1],
        reverse=(g.strand == "-"),
    )
    phase, done = {}, 0
    for f in cds:
        phase[f] = (3 - done % 3) % 3
        done += f[2] - f[1] + 1
    for i, (t, s, e) in enumerate(g.feats, 1):
        ph = phase.get((t, s, e), ".") if t == "CDS" else "."
        lines.append(
            f"{g.seqid}\tGeneCAD\t{t}\t{s}\t{e}\t.\t{g.strand}\t{ph}\tID={tx}.{t}.{i};Parent={tx}"
        )
    return lines


def write_genes(path: str, header: list[str], by_seqid: dict[str, list[Gene]]) -> None:
    """Write genes grouped by seqid (in dict order), sorted by start within each."""
    with atomic_output_path(path) as tmp, open(tmp, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        for genes in by_seqid.values():
            for g in sorted(genes, key=lambda g: (g.start, g.end)):
                for line in _gene_lines(g):
                    fh.write(line + "\n")
