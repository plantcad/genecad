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
import logging
import multiprocessing
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from src import hybrid_decode as hd
from src.frame_crf import (
    DEFAULT_EXON_LENGTH_STRICTNESS,
    DEFAULT_MIN_CODING_RUN_LENGTH,
    DEFAULT_MIN_INTRON_LENGTH,
)

logger = logging.getLogger(__name__)


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
        if os.path.isdir(predictions_dir):
            jobs.append((seqid, genes, predictions_dir))
        else:
            missing.append(seqid)
    if missing:
        logger.warning(
            f"{len(missing)} sequence(s) have no predictions and were left unchanged, "
            f"e.g. {missing[:5]}"
        )

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
    calls = [
        (seqid, genes, args.input_fasta, pdir, *options) for seqid, genes, pdir in jobs
    ]
    if args.workers > 1:
        # spawn, not fork: the parent is already multi-threaded (torch, numba).
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            results = list(pool.map(hd.process_sequence, *zip(*calls))) if calls else []
    else:
        results = [hd.process_sequence(*call) for call in calls]
    totals: Counter = Counter()
    for seqid, genes, stats in results:
        by_seqid[seqid] = genes
        totals.update(stats)

    hd.write_genes(args.output_gff, header, by_seqid)
    logger.info(f"Hybrid decoding summary: {dict(totals)}")
    logger.info(
        f"Wrote {sum(len(g) for g in by_seqid.values())} genes to {args.output_gff}"
    )


if __name__ == "__main__":
    main()
