import argparse
from enum import Enum

import gff_utils
import pandas as pd
from multiprocessing import Pool

# parse arguments
parser = argparse.ArgumentParser(description="Gene Annotation Training CRF")
parser.add_argument(
    "--gff", type=str, required=True, help="gff with gene annotations to analyze"
)
parser.add_argument("--output", type=str, required=True, help="output table path")
parser.add_argument("--num-workers", type=int, default=1, help="number of workers")
parser.add_argument(
    "--tag-canonical",
    default=False,
    action="store_true",
    help="If set, include the tag for canonical transcript",
)

args = parser.parse_args()

num_workers = args.num_workers

# from warnings import simplefilter
# simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


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
    gene = gff[chrom_idx].genes[gene_idx]
    mRNA = gene.mRNAs[mRNA_idx]

    pos_list = []
    jtype_list = []

    if gene.strand.positive():
        for idx in range(len(mRNA.five_prime_utr) - 1):
            pos_list.append(mRNA.five_prime_utr[idx][1] + 1)
            jtype_list.append(Junc.DONOR)

            pos_list.append(mRNA.five_prime_utr[idx + 1][0] - 1)
            jtype_list.append(Junc.ACCEPTOR)

        # case: intron perfectly splits 5' utr and cds
        if len(mRNA.five_prime_utr) > 1:
            if mRNA.five_prime_utr[-1][1] + 1 != mRNA.cds[0].range[0]:
                pos_list.append(mRNA.five_prime_utr[-1][1] + 1)
                jtype_list.append(Junc.DONOR)

                pos_list.append(mRNA.cds[0].range[0] - 1)
                jtype_list.append(Junc.ACCEPTOR)

        pos_list.append(mRNA.cds[0].range[0])
        jtype_list.append(Junc.TIS)

        for idx in range(len(mRNA.cds) - 1):
            pos_list.append(mRNA.cds[idx].range[1] + 1)
            jtype_list.append(Junc.DONOR)

            pos_list.append(mRNA.cds[idx + 1].range[0] - 1)
            jtype_list.append(Junc.ACCEPTOR)

        pos_list.append(mRNA.cds[-1].range[1])
        jtype_list.append(Junc.TTS)

        # case: intron perfectly splits 5' utr and cds
        if len(mRNA.three_prime_utr) > 1:
            if mRNA.cds[-1].range[1] + 1 != mRNA.three_prime_utr[0][0]:
                pos_list.append(mRNA.cds[-1].range[1] + 1)
                jtype_list.append(Junc.DONOR)

                pos_list.append(mRNA.three_prime_utr[0][0] - 1)
                jtype_list.append(Junc.ACCEPTOR)

        for idx in range(len(mRNA.three_prime_utr) - 1):
            pos_list.append(mRNA.three_prime_utr[idx][1] + 1)
            jtype_list.append(Junc.DONOR)

            pos_list.append(mRNA.three_prime_utr[idx + 1][0] - 1)
            jtype_list.append(Junc.ACCEPTOR)

    else:
        for idx in range(len(mRNA.three_prime_utr) - 1):
            pos_list.append(mRNA.three_prime_utr[idx][1] + 1)
            jtype_list.append(Junc.ACCEPTOR)

            pos_list.append(mRNA.three_prime_utr[idx + 1][0] - 1)
            jtype_list.append(Junc.DONOR)

        # case: intron perfectly splits 5' utr and cds
        if len(mRNA.three_prime_utr) > 1:
            if mRNA.three_prime_utr[-1][1] + 1 != mRNA.cds[0].range[0]:
                pos_list.append(mRNA.three_prime_utr[-1][1] + 1)
                jtype_list.append(Junc.ACCEPTOR)

                pos_list.append(mRNA.cds[0].range[0] - 1)
                jtype_list.append(Junc.DONOR)

        pos_list.append(mRNA.cds[0].range[0])
        jtype_list.append(Junc.TTS)

        for idx in range(len(mRNA.cds) - 1):
            pos_list.append(mRNA.cds[idx].range[1] + 1)
            jtype_list.append(Junc.ACCEPTOR)

            pos_list.append(mRNA.cds[idx + 1].range[0] - 1)
            jtype_list.append(Junc.DONOR)

        pos_list.append(mRNA.cds[-1].range[1])
        jtype_list.append(Junc.TIS)

        # case: intron perfectly splits 5' utr and cds
        if len(mRNA.five_prime_utr) > 1:
            if mRNA.cds[-1].range[1] + 1 != mRNA.five_prime_utr[0][0]:
                pos_list.append(mRNA.cds[-1].range[1] + 1)
                jtype_list.append(Junc.ACCEPTOR)

                pos_list.append(mRNA.five_prime_utr[0][0] - 1)
                jtype_list.append(Junc.DONOR)

        for idx in range(len(mRNA.five_prime_utr) - 1):
            pos_list.append(mRNA.five_prime_utr[idx][1] + 1)
            jtype_list.append(Junc.ACCEPTOR)

            pos_list.append(mRNA.five_prime_utr[idx + 1][0] - 1)
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


if __name__ == "__main__":
    print("loading files")

    # load gff
    gff = gff_utils.parse_gff(args.gff, not args.tag_canonical)

    chrom_list = [chrom.name for chrom in gff]

    rem_counter = 0
    # remove transcripts without CDSs, and genes without at least one mRNA
    for chrom in gff:
        start_length = len(chrom.genes)

        for gene in chrom.genes:
            gene.mRNAs = [mRNA for mRNA in gene.mRNAs if len(mRNA.cds) > 0]

            for mRNA in gene.mRNAs:
                mRNA.cds.sort(key=lambda cds: cds.range[0])
                mRNA.five_prime_utr.sort(key=lambda utr: utr[0])
                mRNA.three_prime_utr.sort(key=lambda utr: utr[0])

            gene.mRNAs.sort(key=lambda mRNA: mRNA.range[0])

        chrom.genes = [gene for gene in chrom.genes if len(gene.mRNAs) > 0]

        end_length = len(chrom.genes)
        rem_counter += start_length - end_length

    print(
        "Removed "
        + str(rem_counter)
        + " genes without a valid protein-coding transcript"
    )

    print("Getting junction locations")
    junctions = [
        get_junctions(gff, chrom_idx, gene_idx, mRNA_idx)
        for chrom_idx in range(len(gff))
        for gene_idx in range(len(gff[chrom_idx].genes))
        for mRNA_idx in range(len(gff[chrom_idx].genes[gene_idx].mRNAs))
    ]

    print("concat")
    df = pd.concat(junctions, ignore_index=True)

    print("merge redundant entries")
    x = iter(df.groupby(["chrom", "gene", "pos", "junction"], sort=False))
    # Launch threads
    with Pool(num_workers) as p:
        results = p.map(merge_entries, x)

    df2 = pd.concat(results)

    df2.to_csv(args.output, sep="\t", index=False)
