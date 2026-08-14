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


def main():
    parser = argparse.ArgumentParser(description="Gene Annotation Training CRF")
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

    gff = [chrom for chrom in parse(args.input_gff)]

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

    logger.info("Getting junction locations")
    junctions = [
        get_junctions(gff, chrom_idx, gene_idx, mRNA_idx)
        for chrom_idx in range(len(gff))
        for gene_idx in range(len(gff[chrom_idx].features))
        for mRNA_idx in range(len(gff[chrom_idx].features[gene_idx].sub_features))
    ]

    df = pd.concat(junctions, ignore_index=True)

    logger.info("Merging redundant entries")
    x = iter(df.groupby(["chrom", "gene", "pos", "junction"], sort=False))
    # Launch threads
    with Pool(args.num_workers) as p:
        results = p.map(merge_entries, x)

    df2 = pd.concat(results)

    df2.to_csv(args.output_table, sep="\t", index=False)


if __name__ == "__main__":
    main()
