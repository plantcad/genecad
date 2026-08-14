import argparse

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader
from scripts.gff_zero_shot_preprocess import load_gff, Junc
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


def has_protein_coding_tag(tags) -> bool:
    if "gene_biotype" in tags.keys():
        return tags["gene_biotype"] == "protein_coding"
    elif "biotype" in tags.keys():
        return tags["biotype"] == "protein_coding"
    else:
        return False


def has_canonical_tag(tags) -> bool:
    if "Ensemble_canonical" in tags.keys():
        return tags["tag"] == "Ensembl_canonical"
    elif "canonical_transcript" in tags.keys():
        return tags["canonical_transcript"] == "1"
    elif "tag" in tags.keys():
        other_tags = tags["tag"].split(",")
        return "Ensembl_canonical" in other_tags
    else:
        return False


def to_junc(x):
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
            sequence = str(self.fastas[chrom][0:end].seq.upper()).rjust(
                self.window_size, "N"
            )
        else:
            chrom_end = len(self.fastas[chrom])
            if end > chrom_end:
                end = chrom_end

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


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Gene Annotation Training CRF")
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
    parser.add_argument("--output", type=str, required=True, help="output table path")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the pre-trained model"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for data loading"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device to use")
    parser.add_argument(
        "--window-size",
        type=int,
        default=8192,
        help="Size of the window for processing sequences. Must be divisible by 4",
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

    chrom_list = [chrom.name for chrom in gff]

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
                    mRNA.location.end,
                    gene.strand,
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
                    mRNA.range[0],
                    mRNA.range[1],
                    gene.strand,
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

    # identify transcript with longest CDS
    longest_transcripts = [False] * out_df.shape[0]

    for chrom in gff:
        for gene in tqdm(chrom.features):
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
            transcript_name = gene.sub_features[longest_transcript].name
            longest_transcripts[out_df.index.get_loc(transcript_name)] = True

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
            ref_probs = [
                probs[prob_idx, mask_idx, ref_idx] if ref_idx >= 0 else 0.0
                for mask_idx, ref_idx in zip(masked_indices, ref_index)
            ]

            avg_ref_prob = np.mean(ref_probs)

            idx = int(names[prob_idx])

            jtype = df["junction"].iloc[idx]

            if type(df["mRNA"].iloc[idx]) is not str:
                mRNA_names = [
                    gff[df["chrom"].iloc[idx]]
                    .genes[df["gene"].iloc[idx]]
                    .mRNAs[df["mRNA"].iloc[idx]]
                    .name
                ]
            else:
                mRNA_names = [
                    gff[df["chrom"].iloc[idx]]
                    .genes[df["gene"].iloc[idx]]
                    .mRNAs[int(idy)]
                    .name
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
                    out_df["donor"].loc[mRNA_name].scores.append(avg_ref_prob)
                else:
                    out_df["acceptor"].loc[mRNA_name].scores.append(avg_ref_prob)

    out_df["TIS"] = tis
    out_df["TTS"] = tts
    print("Writing")
    out_df.to_csv(args.output, sep="\t", index=False, header=True)


if __name__ == "__main__":
    main()
