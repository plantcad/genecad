"""Step 1 of MMLR: extract TIS/TTS/Donor/Acceptor junctions from a GFF3 annotation.

For each protein-coding mRNA, walks its 5'UTR/CDS/3'UTR sub-features in genomic
order to enumerate junction positions, then collapses junctions that are shared
across multiple transcripts (e.g. due to alternative splicing) into a single row.
See docs/masked_motif_logistic_regression.md for the full pipeline and output format.
"""

import argparse
from enum import Enum

import pandas as pd
from multiprocessing import Pool
import logging
from src.gff_parser import parse

logger = logging.getLogger(__name__)


class Junc(Enum):
    TIS = 0
    TTS = 1
    DONOR = 2
    ACCEPTOR = 3

    def __str__(self):
        if self == self.TIS:
            return "TIS"
        elif self == self.TTS:
            return "TTS"
        elif self == self.DONOR:
            return "Donor"
        else:
            return "Acceptor"


def get_junctions(gff, chrom_idx, gene_idx, mRNA_idx):
    """Enumerate junction positions/types for a single mRNA.

    Junctions are found by walking each feature list (5'UTR, CDS, 3'UTR) in
    genomic order and treating the gap between consecutive features of the same
    type as an intron (donor at the end of the upstream exon, acceptor at the
    start of the downstream exon). TIS/TTS fall at the first/last base of the
    CDS. Because introns can also fall *between* feature types (e.g. between the
    last 5'UTR exon and the first CDS exon), each of those boundaries is checked
    separately ("intron perfectly splits..." blocks) so that junction isn't missed.

    Orientation of pos_list/jtype_list mirrors gene.location.strand: on the plus
    strand junctions are appended 5'->3' (UTR, then TIS, CDS introns, TTS, UTR);
    on the minus strand the same walk is done in GFF/plus-strand coordinate order,
    but donor/acceptor and TIS/TTS labels are swapped since transcription runs
    the opposite direction. `pos` always points at the first base of the motif
    in the strand's own reading direction (BioPython locations are 0-based,
    half-open, so `location.end` is the base just past a feature and
    `location.start - 1` is the base just before the next one).
    """
    gene = gff[chrom_idx].features[gene_idx]
    mRNA = gene.sub_features[mRNA_idx]

    five_prime_utr = [
        feat for feat in mRNA.sub_features if feat.type == "five_prime_UTR"
    ]
    three_prime_utr = [
        feat for feat in mRNA.sub_features if feat.type == "three_prime_UTR"
    ]
    cds = [feat for feat in mRNA.sub_features if feat.type == "CDS"]

    pos_list = []
    jtype_list = []

    if gene.location.strand == 1:
        for idx in range(len(five_prime_utr) - 1):
            pos_list.append(five_prime_utr[idx].location.end)
            jtype_list.append(Junc.DONOR)

            pos_list.append(five_prime_utr[idx + 1].location.start - 1)
            jtype_list.append(Junc.ACCEPTOR)

        # case: intron perfectly splits 5' utr and cds
        if len(five_prime_utr) > 1:
            if five_prime_utr[-1].location.end != cds[0].location.start:
                pos_list.append(five_prime_utr[-1].location.end)
                jtype_list.append(Junc.DONOR)

                pos_list.append(cds[0].location.start - 1)
                jtype_list.append(Junc.ACCEPTOR)

        pos_list.append(cds[0].location.start)
        jtype_list.append(Junc.TIS)

        for idx in range(len(cds) - 1):
            pos_list.append(cds[idx].location.end)
            jtype_list.append(Junc.DONOR)

            pos_list.append(cds[idx + 1].location.start - 1)
            jtype_list.append(Junc.ACCEPTOR)

        pos_list.append(cds[-1].location.end - 1)
        jtype_list.append(Junc.TTS)

        # case: intron perfectly splits 5' utr and cds
        if len(three_prime_utr) > 1:
            if cds[-1].location.end != three_prime_utr[0].location.start:
                pos_list.append(cds[-1].location.end)
                jtype_list.append(Junc.DONOR)

                pos_list.append(three_prime_utr[0].location.start - 1)
                jtype_list.append(Junc.ACCEPTOR)

        for idx in range(len(three_prime_utr) - 1):
            pos_list.append(three_prime_utr[idx].location.end)
            jtype_list.append(Junc.DONOR)

            pos_list.append(three_prime_utr[idx + 1].location.start - 1)
            jtype_list.append(Junc.ACCEPTOR)

    else:
        # Minus strand: iterate in the same (plus-strand) coordinate order as
        # above, but donor/acceptor and TIS/TTS are swapped since the mRNA's
        # 5'->3' direction is reversed relative to genomic coordinates.
        for idx in range(len(three_prime_utr) - 1):
            pos_list.append(three_prime_utr[idx].location.end)
            jtype_list.append(Junc.ACCEPTOR)

            pos_list.append(three_prime_utr[idx + 1].location.start - 1)
            jtype_list.append(Junc.DONOR)

        # case: intron perfectly splits 5' utr and cds
        if len(three_prime_utr) > 1:
            if three_prime_utr[-1].location.end != cds[0].location.start:
                pos_list.append(three_prime_utr[-1].location.end)
                jtype_list.append(Junc.ACCEPTOR)

                pos_list.append(cds[0].location.start - 1)
                jtype_list.append(Junc.DONOR)

        pos_list.append(cds[0].location.start)
        jtype_list.append(Junc.TTS)

        for idx in range(len(cds) - 1):
            pos_list.append(cds[idx].location.end)
            jtype_list.append(Junc.ACCEPTOR)

            pos_list.append(cds[idx + 1].location.start - 1)
            jtype_list.append(Junc.DONOR)

        pos_list.append(cds[-1].location.end - 1)
        jtype_list.append(Junc.TIS)

        # case: intron perfectly splits 5' utr and cds
        if len(five_prime_utr) > 1:
            if cds[-1].location.end != five_prime_utr[0].location.start:
                pos_list.append(cds[-1].location.end)
                jtype_list.append(Junc.ACCEPTOR)

                pos_list.append(five_prime_utr[0].location.start - 1)
                jtype_list.append(Junc.DONOR)

        for idx in range(len(five_prime_utr) - 1):
            pos_list.append(five_prime_utr[idx].location.end)
            jtype_list.append(Junc.ACCEPTOR)

            pos_list.append(five_prime_utr[idx + 1].location.start - 1)
            jtype_list.append(Junc.DONOR)

    return pd.DataFrame(
        dict(
            chrom=[chrom_idx] * len(pos_list),
            gene=[gene_idx] * len(pos_list),
            mRNA=[mRNA_idx] * len(pos_list),
            pos=pos_list,
            junction=jtype_list,
        )
    )


def merge_entries(y):
    """Collapse one (chrom, gene, pos, junction) group into a single row.

    `y` is a (group_key, group_df) tuple from a pandas groupby. When multiple
    mRNAs of the same gene share an identical junction position/type (e.g. an
    intron retained in every isoform), this merges them into one row whose
    `mRNA` column is a comma-separated list of the contributing mRNA indices,
    rather than emitting redundant duplicate junctions.
    """
    if y[1].shape[0] > 1:
        z = ",".join([str(ent) for ent in list(y[1]["mRNA"])])
        return pd.DataFrame(
            dict(
                chrom=[y[0][0]],
                gene=[y[0][1]],
                mRNA=[z],
                pos=[y[0][2]],
                junction=[y[0][3]],
            )
        )
    else:
        return y[1]


def load_gff(gff_filename):
    """Parse a GFF3 into per-chromosome feature trees, keeping only
    protein-coding content: mRNAs without a CDS are dropped, each remaining
    mRNA's sub-features are restricted to CDS/UTR and sorted by position, and
    genes left with no valid mRNA are removed entirely. Also used by
    mmlr_score_junctions.py so both scripts see the same filtered gene set.
    """
    gff = [chrom for chrom in parse(gff_filename)]

    # remove transcripts without CDSs, and genes without at least one mRNA
    rem_counter = 0

    for chrom in gff:
        start_length = len(chrom.features)

        keep_genes = []

        for gene in chrom.features:
            if gene.type == "gene":
                mRNAs = [
                    mRNA
                    for mRNA in gene.sub_features
                    if any([feat.type == "CDS" for feat in mRNA.sub_features])
                ]

                for mRNA in mRNAs:
                    mRNA.sub_features = [
                        feat
                        for feat in mRNA.sub_features
                        if feat.type in ["CDS", "five_prime_UTR", "three_prime_UTR"]
                    ]
                    mRNA.sub_features.sort(key=lambda feat: feat.location.start)

                mRNAs.sort(key=lambda mRNA: mRNA.location.start)
                gene.sub_features = mRNAs

                if len(gene.sub_features) > 0:
                    keep_genes.append(gene)

        chrom.features = keep_genes

        end_length = len(chrom.features)
        rem_counter += start_length - end_length

    logger.info(
        "Removed "
        + str(rem_counter)
        + " genes without a valid protein-coding transcript"
    )

    return gff


def main():
    parser = argparse.ArgumentParser(description="Identify unique Gene Junctions")
    parser.add_argument(
        "--input-gff",
        "-i",
        type=str,
        required=True,
        help="gff with gene annotations to analyze",
    )
    parser.add_argument(
        "--output-table", "-o", type=str, required=True, help="output table path"
    )
    parser.add_argument("--num-workers", type=int, default=1, help="number of workers")

    args = parser.parse_args()

    logger.info("Loading files...")

    gff = load_gff(args.input_gff)

    logger.info("Getting junction locations")
    # One small DataFrame of junctions per mRNA; concatenated below into one table.
    junctions = [
        get_junctions(gff, chrom_idx, gene_idx, mRNA_idx)
        for chrom_idx in range(len(gff))
        for gene_idx in range(len(gff[chrom_idx].features))
        for mRNA_idx in range(len(gff[chrom_idx].features[gene_idx].sub_features))
    ]

    df = pd.concat(junctions, ignore_index=True)

    logger.info("Merging redundant entries")
    # Group identical junctions (same chrom/gene/pos/type) across alternative
    # transcripts so each unique site is only scored once downstream.
    x = iter(df.groupby(["chrom", "gene", "pos", "junction"], sort=False))
    # Launch threads
    with Pool(args.num_workers) as p:
        results = p.map(merge_entries, x)

    df2 = pd.concat(results)

    df2.to_csv(args.output_table, sep="\t", index=False)


if __name__ == "__main__":
    main()
