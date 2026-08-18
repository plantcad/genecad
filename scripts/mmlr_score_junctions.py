"""Step 2 of MMLR: score each junction from mmlr_prepare_junctions.py with a
PlantCAD/PlantCAD2 masked-language model.

For every junction, a window of sequence centered on it is fed to the model
with the motif bases masked out; the Masked Motif score is the mean predicted
probability the model assigns to the *actual* (reference) base at each masked
position. Requires a CUDA GPU. See docs/masked_motif_logistic_regression.md.
"""

import argparse

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader
from scripts.mmlr_prepare_junctions import load_gff, Junc
from Bio import SeqIO
import pandas as pd
from dataclasses import dataclass
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


# simple dataclass to handle multiple donor/acceptor splice site scores
@dataclass
class ScoreList:
    scores: list

    def __str__(self):
        return ",".join([str(score) for score in self.scores])


def has_canonical_tag(tags) -> bool:
    """Check whether a GFF feature's qualifier dict marks it canonical, under
    either of the two labeling schemes accepted by --tag-canonical:
    `canonical_transcript=1` or `tag=Ensembl_canonical`.
    """
    if "canonical_transcript" in tags.keys():
        return "1" in tags["canonical_transcript"]
    elif "tag" in tags.keys():
        return "Ensembl_canonical" in tags["tag"]
    else:
        return False


# convert 1/-1 to +/-
def strand_string(x):
    if x == 1:
        return "+"
    elif x == -1:
        return "-"
    else:
        return "."


def to_junc(x):
    """Inverse of Junc.__str__: parse the junction-type string written to the
    Step 1 output table back into a Junc enum member.
    """
    if x == "TIS":
        return Junc.TIS
    elif x == "TTS":
        return Junc.TTS
    elif x == "Donor":
        return Junc.DONOR
    elif x == "Acceptor":
        return Junc.ACCEPTOR
    else:
        raise TypeError("Junctions may only be one of: TIS, TTS, Donor, Acceptor")


class JunctionDataset(Dataset):
    """Produces one masked, tokenized sequence window per junction row.

    Each window is `window_size` bases centered on the junction's `pos`
    (`self.token` is the center index), padded with "N" at chromosome edges.
    The 2-3 bases making up the motif (start/stop codon, or the first/last two
    intron bases of a splice site) are replaced with the model's mask token so
    the model must predict them from context alone.
    """

    def __init__(self, fastas, gff, chrom_list, df, tokenizer, window_size):
        self.fastas = fastas  # dict of seq records
        self.chrom_list = chrom_list
        self.df = df  # df containing positions
        self.gff = gff
        self.tokenizer = tokenizer  # model tokenizer
        self.window_size = window_size  # maximum context length
        self.token = window_size // 2

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        start = self.df["pos"].iloc[idx] - self.token
        end = start + self.window_size
        chrom = self.chrom_list[self.df["chrom"].iloc[idx]]

        if start < 0:
            # Junction is near the start of the chromosome: shift the window
            # right and left-pad with N so it still spans window_size bases.
            sequence = str(self.fastas[chrom][0:end].seq.upper()).rjust(
                self.window_size, "N"
            )
        else:
            chrom_end = len(self.fastas[chrom])
            if end > chrom_end:
                end = chrom_end

            # Right-pad with N if the window runs past the chromosome end.
            sequence = str(self.fastas[chrom][start:end].seq.upper()).ljust(
                self.window_size, "N"
            )

        # Tokenize and pad to window_size
        encoding = self.tokenizer.encode_plus(
            sequence,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        input_ids = encoding["input_ids"].squeeze()

        strand = (
            self.gff[self.df["chrom"].iloc[idx]]
            .features[self.df["gene"].iloc[idx]]
            .location.strand
        )

        junction = self.df["junction"].iloc[idx]

        mask_tokens = np.zeros(self.window_size, dtype=bool)

        # `pos`/self.token is the first base of the motif in strand orientation
        # (see mmlr_prepare_junctions.get_junctions), so which offsets get
        # masked - and whether TIS/TTS or Donor/Acceptor is used - depends on
        # strand: a plus-strand TIS is a minus-strand TTS's mirror image, etc.
        if (strand == 1 and junction == Junc.TIS) or (
            strand == -1 and junction == Junc.TTS
        ):
            mask_tokens[self.token : self.token + 3] = True
        elif (strand == 1 and junction == Junc.TTS) or (
            strand == -1 and junction == Junc.TIS
        ):
            mask_tokens[self.token - 2 : self.token + 1] = True
        elif (strand == 1 and junction == Junc.DONOR) or (
            strand == -1 and junction == Junc.ACCEPTOR
        ):
            mask_tokens[self.token : self.token + 2] = True
        elif (strand == 1 and junction == Junc.ACCEPTOR) or (
            strand == -1 and junction == Junc.DONOR
        ):
            mask_tokens[self.token - 1 : self.token + 1] = True

        input_ids[mask_tokens] = (
            self.tokenizer.mask_token_id
        )  # mask the specified token index

        input_ids = input_ids.unsqueeze(0)

        return {
            "sequence": sequence,
            "input_ids": input_ids,
            "mask": mask_tokens,
            "name": idx,
        }


def get_longest_transcripts(gff, out_df):
    """Flag, per gene, the transcript with the longest total CDS length (used
    for the `longest` output column)."""
    # identify transcript with longest CDS
    longest_transcripts = [False] * out_df.shape[0]

    for chrom in gff:
        for gene in chrom.features:
            longest_transcript = np.argmax(
                [
                    np.sum(
                        [
                            (exon.location.end + 1) - exon.location.start
                            for exon in mRNA.sub_features
                        ]
                    )
                    for mRNA in gene.sub_features
                ]
            )
            transcript_name = gene.sub_features[longest_transcript].id
            longest_transcripts[out_df.index.get_loc(transcript_name)] = True

    return longest_transcripts


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Generate Masked Motif scores")
    parser.add_argument(
        "--input-gff",
        "-i",
        type=str,
        required=True,
        help="gff with gene annotations to analyze",
    )
    parser.add_argument(
        "--input-fasta", "-f", type=str, required=True, help="fasta file"
    )
    parser.add_argument(
        "--input-junctions", "-j", type=str, required=True, help="junctions file"
    )
    parser.add_argument(
        "--output-table", "-o", type=str, required=True, help="output table path"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="kuleshov-group/PlantCAD2-Medium-l48-d1024",  # pragma: allowlist secret
        help="Path to the pre-trained model. PlantCAD or PlantCAD2 models may be used. "
        "Default: kuleshov-group/PlantCAD2-Medium-l48-d1024",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for data loading. Default: 16",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device to use")
    parser.add_argument(
        "--window-size",
        type=int,
        default=8192,
        help="Size of the window for processing sequences. Must be divisible by 2. Default: 8192",
    )
    parser.add_argument(
        "--tag-canonical",
        default=False,
        action="store_true",
        help="If set, include the tag for canonical transcript",
    )

    args = parser.parse_args()

    device = "cuda:" + str(args.gpu)
    junction_df = args.input_junctions
    window_size = args.window_size
    batch_size = args.batch_size
    model_path = args.model_path
    print("loading files")

    # load gff
    gff = load_gff(args.input_gff)

    chrom_list = [chrom.id for chrom in gff]

    fastas = SeqIO.to_dict(SeqIO.parse(args.input_fasta, "fasta"))

    print("load junctions")
    df = pd.read_csv(junction_df, sep="\t")

    # convert string back into junction class
    df["junction"] = df["junction"].apply(to_junc)

    print("load model")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    ds = JunctionDataset(
        fastas=fastas,
        gff=gff,
        chrom_list=chrom_list,
        df=df,
        tokenizer=tokenizer,
        window_size=window_size,
    )
    dl = DataLoader(ds, shuffle=False, batch_size=batch_size)

    model = AutoModelForMaskedLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model.to(device)

    if args.tag_canonical:
        out_df = pd.DataFrame(
            [
                (
                    chrom.id,
                    gene.id,
                    mRNA.id,
                    mRNA.location.start,
                    mRNA.location.end - 1,
                    strand_string(gene.location.strand),
                    has_canonical_tag(mRNA.qualifiers),
                    ScoreList([]),
                    ScoreList([]),
                )
                for chrom in gff
                for gene in chrom.features
                for mRNA in gene.sub_features
            ],
            columns=[
                "chrom",
                "gene",
                "transcript",
                "start",
                "end",
                "strand",
                "canonical",
                "donor",
                "acceptor",
            ],
        )
    else:
        out_df = pd.DataFrame(
            [
                (
                    chrom.id,
                    gene.id,
                    mRNA.id,
                    mRNA.location.start,
                    mRNA.location.end - 1,
                    strand_string(gene.location.strand),
                    ScoreList([]),
                    ScoreList([]),
                )
                for chrom in gff
                for gene in chrom.features
                for mRNA in gene.sub_features
            ],
            columns=[
                "chrom",
                "gene",
                "transcript",
                "start",
                "end",
                "strand",
                "donor",
                "acceptor",
            ],
        )

    out_df.index = out_df["transcript"]

    longest_transcripts = get_longest_transcripts(gff, out_df)
    out_df["longest"] = longest_transcripts

    print("calculating zero-shot scores")
    nucleotides = list("acgt")
    nts = ["A", "C", "G", "T"]

    tis = [0.0] * out_df.shape[0]
    tts = [0.0] * out_df.shape[0]

    for batch in tqdm(dl):
        curIDs = batch["input_ids"].to(device)
        curIDs = curIDs.squeeze(1)
        names = batch["name"]
        masks = batch["mask"]
        with torch.inference_mode():
            outputs = model(input_ids=curIDs)
        all_logits = outputs.logits
        logits = all_logits[
            :, :, [tokenizer.get_vocab()[nc] for nc in nucleotides]
        ]  # get the logits for the masked token
        probs = torch.nn.functional.softmax(logits.cpu(), dim=2).numpy()

        for prob_idx in range(probs.shape[0]):
            masked_indices = np.argwhere(masks[prob_idx]).squeeze().tolist()
            ref_alleles = [
                batch["sequence"][prob_idx][token_idx] for token_idx in masked_indices
            ]
            ref_index = [nts.index(x) if x in nts else -1 for x in ref_alleles]
            # Non-ACGT reference bases (e.g. "N") have no matching model
            # probability, so they're scored as 0 rather than looked up.
            ref_probs = [
                probs[prob_idx, mask_idx, ref_idx] if ref_idx >= 0 else 0.0
                for mask_idx, ref_idx in zip(masked_indices, ref_index)
            ]

            # Masked Motif score: mean predicted probability of the *observed*
            # base at each masked position in the motif.
            avg_ref_prob = np.mean(ref_probs)

            idx = int(names[prob_idx])

            jtype = df["junction"].iloc[idx]

            # A junction row's `mRNA` column is either a single sub_features
            # index, or (after merge_entries in Step 1) a comma-separated list
            # of indices shared by multiple transcripts - resolve to mRNA IDs
            # either way so the score below is applied to every transcript
            # that shares this junction.
            if type(df["mRNA"].iloc[idx]) is not str:
                mRNA_names = [
                    gff[df["chrom"].iloc[idx]]
                    .features[df["gene"].iloc[idx]]
                    .sub_features[df["mRNA"].iloc[idx]]
                    .id
                ]
            else:
                mRNA_names = [
                    gff[df["chrom"].iloc[idx]]
                    .features[df["gene"].iloc[idx]]
                    .sub_features[int(idy)]
                    .id
                    for idy in df["mRNA"].iloc[idx].split(",")
                ]
            for mRNA_name in mRNA_names:
                if jtype == Junc.TIS:
                    # out_df["TIS"].loc[mRNA_name] = zero_shot
                    tis[out_df.index.get_loc(mRNA_name)] = avg_ref_prob
                elif jtype == Junc.TTS:
                    # out_df["TTS"].loc[mRNA_name] = zero_shot
                    tts[out_df.index.get_loc(mRNA_name)] = avg_ref_prob

                elif jtype == Junc.DONOR:
                    # A transcript may have multiple donor/acceptor sites, so
                    # scores accumulate in a ScoreList rather than overwriting.
                    out_df["donor"].loc[mRNA_name].scores.append(avg_ref_prob)
                else:
                    out_df["acceptor"].loc[mRNA_name].scores.append(avg_ref_prob)

    out_df["TIS"] = tis
    out_df["TTS"] = tts
    print("Writing")
    out_df.to_csv(args.output_table, sep="\t", index=False, header=True)


if __name__ == "__main__":
    main()
