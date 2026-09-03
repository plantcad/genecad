"""Step 4 of MMLR: apply a trained (or default GeneCAD) classifier to score
transcripts, and optionally tag a GFF3 with a pass/fail filter based on those
scores. See docs/masked_motif_logistic_regression.md.
"""

import os.path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import json
import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Values from GeneCAD paper
default_dict = {
    "multi_intercept": -1902.2387911,
    "multi_TIS": 1257.14461103,
    "multi_TTS": 887.72821422,
    "multi_Donor": 571.2411655,
    "multi_Acceptor": 846.96884891,
    "single_intercept": -7.26727317,
    "single_TIS": 11.81083443,
    "single_TTS": 10.79526443,
}


def load_models(json_path):
    """Build the multi-exon and single-exon LogisticRegression classifiers
    from stored coefficients (from mmlr_train_classifier.py's --output-json,
    or the GeneCAD-paper defaults above) rather than by calling .fit(). The
    coef_/intercept_/classes_ attributes are set directly so the models are
    ready for .predict()/.predict_proba() without any training data.
    """
    if json_path is not None:
        with open(json_path) as file:
            models_dict = json.load(file)
    else:
        models_dict = default_dict

    multi_exon_model = LogisticRegression()

    multi_exon_model.intercept_ = np.array(models_dict["multi_intercept"])
    multi_exon_model.coef_ = np.array(
        [
            [
                models_dict["multi_TIS"],
                models_dict["multi_TTS"],
                models_dict["multi_Donor"],
                models_dict["multi_Acceptor"],
            ]
        ]
    )
    multi_exon_model.classes_ = np.array([0, 1])

    single_exon_model = LogisticRegression()
    single_exon_model.intercept_ = np.array(models_dict["single_intercept"])
    single_exon_model.coef_ = np.array(
        [[models_dict["single_TIS"], models_dict["single_TTS"]]]
    )
    single_exon_model.classes_ = np.array([0, 1])

    return multi_exon_model, single_exon_model


def to_average(x):
    """Collapse a transcript's comma-separated per-site donor/acceptor scores
    into a single mean feature value (matches mmlr_train_classifier.to_average
    so scoring uses the same features the classifier was trained on)."""
    y = [float(y) for y in x.split(",")]
    return sum(y) / len(y)


def tag_to_dict(x):
    """Parse a GFF3 attributes field (`key=val;key=val;...`) into a dict."""
    out_dict = {}

    for y in x.split(";"):
        z = y.split("=")
        out_dict[z[0]] = z[1]

    return out_dict


# score genes in table
def score_table(input_df, multi_exon_model, single_exon_model):
    """Route each transcript to the multi-exon or single-exon model based on
    whether it has a donor score, then attach the model's hard prediction
    (`filter`, 0/1) and positive-class probability (`score`) as new columns.
    """
    single_exon_df = input_df[input_df["donor"].isna()].copy()
    multi_exon_df = input_df[input_df["donor"].notna()].copy()

    multi_exon_x = np.array(
        [
            list(multi_exon_df["TIS"]),
            list(multi_exon_df["TTS"]),
            [to_average(x) for x in multi_exon_df["donor"]],
            [to_average(x) for x in multi_exon_df["acceptor"]],
        ]
    ).transpose()

    single_exon_x = np.array(
        [list(single_exon_df["TIS"]), list(single_exon_df["TTS"])]
    ).transpose()

    multi_exon_predictions = multi_exon_model.predict(multi_exon_x)
    single_exon_predictions = single_exon_model.predict(single_exon_x)

    multi_exon_proba = multi_exon_model.predict_proba(multi_exon_x)
    single_exon_proba = single_exon_model.predict_proba(single_exon_x)

    single_exon_df["filter"] = single_exon_predictions
    multi_exon_df["filter"] = multi_exon_predictions

    single_exon_df["score"] = single_exon_proba[:, 1]
    multi_exon_df["score"] = multi_exon_proba[:, 1]

    df_with_filters = pd.concat((single_exon_df, multi_exon_df), ignore_index=True)

    return df_with_filters


# Annotate GFF with model scores
def annotate_gff(df_scored, gff_path, out_dir):
    """Stream the input GFF3 line by line, writing a copy with a
    `passPlantCADFilter` tag added to each `gene` and `mRNA` feature line.
    An mRNA is tagged 1 if its own `filter` prediction is 1; a gene is tagged
    1 if *any* of its transcripts passed (np.any over the gene's group)."""
    base_name = os.path.basename(gff_path).split(".gff")[0]

    df_scored.index = df_scored["transcript"]
    x = iter(df_scored.groupby("gene", sort=False))

    gene_dict = {}

    for y in x:
        gene_dict[y[0]] = y[1]

    out_gff = out_dir + "/" + base_name + "_tagged.gff3"

    writer = open(out_gff, "w")

    with open(gff_path) as file:
        while line := file.readline():
            if line.startswith("#"):
                writer.write(line)
                continue

            line_split = line.rstrip().split("\t")

            if line_split[2] == "gene":
                tags = tag_to_dict(line_split[8])
                gene_id = tags["ID"]

                if gene_id in gene_dict.keys():
                    if np.any(gene_dict[gene_id]["filter"]):
                        tags["passPlantCADFilter"] = "1"
                    else:
                        tags["passPlantCADFilter"] = "0"
                else:
                    tags["passPlantCADFilter"] = "0"

                line_split[8] = ";".join([key + "=" + val for key, val in tags.items()])
                new_line = "\t".join(line_split)
                writer.write(new_line)
                writer.write("\n")

            elif line_split[2] == "mRNA":
                tags = tag_to_dict(line_split[8])
                transcript_id = tags["ID"]

                if transcript_id in df_scored.index:
                    if df_scored["filter"].loc[transcript_id] == 1:
                        tags["passPlantCADFilter"] = "1"
                    else:
                        tags["passPlantCADFilter"] = "0"
                else:
                    tags["passPlantCADFilter"] = "0"
                line_split[8] = ";".join([key + "=" + val for key, val in tags.items()])
                new_line = "\t".join(line_split)
                writer.write(new_line)
                writer.write("\n")

            else:
                writer.write(line)


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Classify transcripts using MMLR")
    parser.add_argument(
        "--input-table",
        "-i",
        type=str,
        required=True,
        help="tsv of Masked Motif scores for transcripts of interest. Required columns: TIS, TTS, donor, acceptor, "
        "gene, and transcript",
    )
    parser.add_argument(
        "--input-gff",
        "-g",
        type=str,
        default=None,
        help="GFF3 file to annotate. Gene and transcript names must match input table. Optional",
    )
    parser.add_argument("--output-dir", "-o", required=True, help="output directory")
    parser.add_argument(
        "--model-json",
        "-j",
        type=str,
        default=None,
        help="path to json of model weights. If none, "
        "model weights described in the GeneCAD paper are used",
    )

    args = parser.parse_args()

    logger.info("Loading models")
    multi_exon_model, single_exon_model = load_models(args.model_json)

    logger.info("Scoring genes")
    # score gene models and save scores to table
    input_df = pd.read_csv(args.input_table, sep="\t")
    input_df_scored = score_table(input_df, multi_exon_model, single_exon_model)
    base_name = os.path.basename(args.input_table).split(".tsv")[0]
    input_df_scored.to_csv(
        f"{args.output_dir}/{base_name}_scored.tsv", sep="\t", index=False
    )

    # if GFF is provided, annotate GFF
    if args.input_gff is not None:
        logger.info("Annotating GFF")
        annotate_gff(input_df_scored, args.input_gff, args.output_dir)
    else:
        logger.info("No GFF provided: skipping annotation")


if __name__ == "__main__":
    main()
