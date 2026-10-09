#!/usr/bin/env python3
"""Hybrid decoding: repair a plain-Viterbi GeneCAD GFF with local frame-aware decoding.

Input is the whole-genome GFF after fix_orf (partial transcripts flagged, not
dropped) from plain (--no-frame-aware) decoding, plus the per-chromosome
prediction stores it was decoded from.  For each sequence, partial genes are
rescued and split genes merged by frame-aware decoding of local windows of the
model's logits; see src/hybrid_decode.py for the method.  Sequences without
predictions are written back unchanged.

Usage
-----
    python scripts/hybrid_decode.py --input-gff orf.gff --input-fasta genome.fa \\
        --predictions-root OUTPUT_DIR --output-gff hybrid.gff

where predictions for sequence <chrom> are in OUTPUT_DIR/<chrom>/predictions_<chrom>/.
"""

from __future__ import annotations

import argparse
import glob
import logging
import multiprocessing
import os
import sys
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from src import hybrid_decode as hd
from src.prediction_checkpoint import RUN, SUCCESS
from src.frame_crf import (
    DEFAULT_EXON_LENGTH_STRICTNESS,
    DEFAULT_MIN_CODING_RUN_LENGTH,
    DEFAULT_MIN_INTRON_LENGTH,
    iter_chromosome_codes,
)

logger = logging.getLogger(__name__)


# Exit status when prediction files fail verification; predict.sh then predicts the
# damaged sequences again and reruns this step.
DAMAGED_PREDICTIONS_EXIT = 3


def decode_sequences(jobs, input_fasta, options, workers):
    """Run `process_sequence` for each job, returning its result by sequence name,
    and the `DamagedPredictions` of the sequences whose prediction files failed
    verification.

    The FASTA is read once, in file order, and each sequence is handed to a worker
    together with its bases. Letting every worker find its own sequence would read the
    file from the start once per sequence. At most two sequences per worker are read
    ahead, which bounds the memory they take.
    """
    wanted = {seqid: (genes, pdir) for seqid, genes, pdir in jobs}
    results = {}
    damaged = []
    if workers <= 1:
        for seqid, codes in iter_chromosome_codes(input_fasta, wanted):
            genes, pdir = wanted[seqid]
            try:
                results[seqid] = hd.process_sequence(
                    seqid, genes, input_fasta, pdir, *options, codes=codes
                )
            except hd.DamagedPredictions as error:
                damaged.append(error)
        return results, damaged

    def collect(done):
        for future in done:
            try:
                seqid, genes, stats = future.result()
            except hd.DamagedPredictions as error:
                damaged.append(error)
                continue
            results[seqid] = (seqid, genes, stats)

    # spawn, not fork: the parent is already multi-threaded (torch, numba).
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        pending = set()
        for seqid, codes in iter_chromosome_codes(input_fasta, wanted):
            genes, pdir = wanted[seqid]
            pending.add(
                pool.submit(
                    hd.process_sequence,
                    seqid,
                    genes,
                    input_fasta,
                    pdir,
                    *options,
                    codes=codes,
                )
            )
            if len(pending) >= 2 * workers:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                collect(done)
        collect(wait(pending).done)
    return results, damaged


def has_predictions(predictions_dir: str) -> bool:
    """Whether a directory holds prediction files: segments or legacy rank stores."""
    return any(
        glob.glob(os.path.join(glob.escape(predictions_dir), pattern))
        for pattern in (RUN, SUCCESS, "segment.*", "predictions.*.zarr")
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(
        description="Repair plain-Viterbi GeneCAD predictions with local frame-aware decoding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-gff",
        required=True,
        help="GFF after fix_orf, partial transcripts flagged",
    )
    parser.add_argument(
        "--input-fasta", required=True, help="Genome FASTA used for prediction"
    )
    parser.add_argument(
        "--predictions-root",
        required=True,
        help="Directory holding <chrom>/predictions_<chrom>/ for each sequence",
    )
    parser.add_argument("--output-gff", required=True, help="Output GFF3 file")
    parser.add_argument(
        "--allow-missing-predictions",
        action="store_true",
        help="Leave sequences whose prediction files are missing or damaged unchanged "
        "instead of stopping with an error",
    )
    parser.add_argument("--domain", choices=["plant", "animal"], default="plant")
    parser.add_argument(
        "--flank",
        type=int,
        default=1_000,
        help="Bases added on each side of a decoding window",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=20_000,
        help="Largest gap (bp) between consecutive same-strand genes for a merge to be tried",
    )
    parser.add_argument(
        "--min-cds",
        type=int,
        default=33,
        help="A rescued or merged transcript needs more than this many nt of CDS",
    )
    parser.add_argument(
        "--keep-partial",
        action="store_true",
        help="Keep partial genes that cannot be rescued (flagged) instead of dropping them",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Sequences processed in parallel"
    )
    graph = parser.add_argument_group(
        "frame-aware decoding (as in detect_intervals.py)"
    )
    graph.add_argument(
        "--min-intron-length", type=int, default=DEFAULT_MIN_INTRON_LENGTH
    )
    graph.add_argument(
        "--min-coding-run-length", type=int, default=DEFAULT_MIN_CODING_RUN_LENGTH
    )
    graph.add_argument(
        "--exon-length-strictness", type=float, default=DEFAULT_EXON_LENGTH_STRICTNESS
    )
    graph.add_argument("--allow-u12-introns", action="store_true")
    args = parser.parse_args()

    header, by_seqid = hd.read_genes(args.input_gff)
    jobs, missing = [], []
    for seqid, genes in by_seqid.items():
        predictions_dir = os.path.join(
            args.predictions_root, seqid, f"predictions_{seqid}"
        )
        if has_predictions(predictions_dir):
            jobs.append((seqid, genes, predictions_dir))
        else:
            missing.append(seqid)
    if missing:
        message = (
            f"{len(missing)} sequence(s) have no predictions, e.g. {missing[:5]}. "
            "Without them, partial genes on these sequences cannot be rescued "
            "and split genes cannot be merged."
        )
        if not args.allow_missing_predictions:
            parser.error(
                message + " Predict them again (predict.sh does this by default), "
                "or pass --allow-missing-predictions to leave these sequences unchanged."
            )
        logger.warning(message + " They were left unchanged.")

    graph_options = {
        "min_intron_length": args.min_intron_length,
        "min_coding_run_length": args.min_coding_run_length,
        "exon_length_strictness": args.exon_length_strictness,
        "allow_u12_introns": args.allow_u12_introns,
    }
    options = (
        args.domain,
        args.flank,
        args.max_gap,
        args.min_cds,
        args.keep_partial,
        graph_options,
    )
    results, damaged = decode_sequences(jobs, args.input_fasta, options, args.workers)
    # Without its completion marker, the predict step checks every segment of a
    # sequence, discards the damaged ones and predicts their windows again.
    for error in damaged:
        logger.error(f"Prediction files failed verification: {error}")
        marker = os.path.join(error.predictions_dir, SUCCESS)
        if os.path.exists(marker):
            os.remove(marker)
    if damaged:
        message = (
            f"The prediction files of {len(damaged)} sequence(s) are damaged and their "
            "completion markers were removed."
        )
        if not args.allow_missing_predictions:
            logger.error(
                message + " Run predict.sh again to predict the damaged parts again "
                "(it does so by itself when it runs this step)."
            )
            sys.exit(DAMAGED_PREDICTIONS_EXIT)
        logger.warning(message + " These sequences were left unchanged.")
    totals: Counter = Counter()
    for seqid, _, _ in jobs:
        if seqid not in results:
            continue
        _, genes, stats = results[seqid]
        by_seqid[seqid] = genes
        totals.update(stats)

    hd.write_genes(args.output_gff, header, by_seqid)
    logger.info(f"Hybrid decoding summary: {dict(totals)}")
    logger.info(
        f"Wrote {sum(len(g) for g in by_seqid.values())} genes to {args.output_gff}"
    )


if __name__ == "__main__":
    main()
