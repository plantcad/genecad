import argparse
import logging
from numpy import typing as npt
from src.sequence import (
    convert_entity_labels_to_intervals,
    viterbi_decode,
)
import torch
import torch.nn.functional as F
import numpy as np
import xarray as xr
from src.prediction import merge_prediction_datasets
from src.modeling import GeneClassifierConfig, token_transition_probs
import pandas as pd
import torch._dynamo
import json

logger = logging.getLogger(__name__)


# TODO: move to utils somewhere
def flip(sequence: npt.ArrayLike) -> npt.ArrayLike:
    """Reverse a sequence along its first axis."""
    return np.flip(sequence, axis=0)


def _detect_intervals(
    predictions: xr.Dataset,
    decode_direct: bool,
    viterbi_alpha: float | None,
    intergenic_bias: float,
    domain: str,
    remove_incomplete_features: bool,
) -> xr.Dataset:
    """Infer genomic intervals from per-token feature predictions.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments describing decoding options.
    predictions : xr.Dataset
        Dataset containing feature logits and predictions for each strand.

    Returns
    -------
    xr.Dataset
        Dataset containing inferred region intervals.
    """
    logger.info("Inferring regions from predicted labels")

    # TODO: Fetch the label properties necessary from attributes stored in the predictions
    # datasets rather than from the configuration files, or from the original model checkpoint.
    config = GeneClassifierConfig()
    region_intervals = []
    strands = predictions.strand.values.tolist()
    assert set(strands) == {"positive", "negative"}

    def _decode_intervals_viterbi(
        logits: npt.ArrayLike, remove_incomplete_features: bool
    ) -> np.ndarray:
        transition_probs = token_transition_probs(
            remove_incomplete_features=remove_incomplete_features,
            domain=domain,
        )
        if (
            transition_probs.columns.tolist()
            != config.token_entity_names_with_background()
        ):
            raise ValueError(
                f"Transition probability classes must match token entity names; expected: {config.token_entity_names_with_background()}, got: {transition_probs.columns.tolist()}"
            )
        emissions = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
        assert emissions.min() >= 0 and emissions.max() <= 1
        assert transition_probs.index.tolist() == transition_probs.columns.tolist()

        # Decoding takes ~90 seconds for 308452471 tokens on Grace CPU
        alpha = viterbi_alpha
        logger.info(f"Running viterbi decoding ({alpha=})")
        labels = viterbi_decode(
            emission_probs=emissions,
            transition_matrix=transition_probs.values,
            alpha=alpha,
        )

        assert labels.ndim == 1
        # pyrefly: ignore  # bad-argument-type
        assert len(labels) == len(logits)
        return labels

    # Penalize intergenic logits to shift the model toward predicting more
    # genic elements, compensating for class-imbalanced training data.
    # Note: this intentionally overlaps with what _create_predictions could do
    # at inference time, but we apply it here (downstream) so the bias can be
    # swept cheaply without regenerating the large prediction datasets.
    logger.info(f"Using intergenic bias: {intergenic_bias}")

    for strand in strands:
        feature_logits = predictions.sel(strand=strand).feature_logits.copy()
        feature_logits.loc[dict(feature="intergenic")] -= intergenic_bias

        if decode_direct:
            labels = feature_logits.argmax(dim="feature").values
            logger.info(f"Running direct decoding for {strand!r} strand")
            intervals = convert_entity_labels_to_intervals(
                labels=labels, class_groups=config.interval_entity_classes
            )
            region_intervals.append(intervals.assign(strand=strand, decoding="direct"))

        # Viterbi decoding (uses biased logits via softmax internally)
        else:
            logger.info(f"Running viterbi decoding for {strand!r} strand")
            logits = feature_logits.values
            if strand == "positive":
                viterbi_labels = _decode_intervals_viterbi(
                    logits=logits,
                    remove_incomplete_features=remove_incomplete_features,
                )
            else:
                viterbi_labels = flip(
                    _decode_intervals_viterbi(
                        logits=flip(logits).copy(),
                        remove_incomplete_features=remove_incomplete_features,
                    )
                )

            intervals = convert_entity_labels_to_intervals(
                # pyrefly: ignore  # bad-argument-type
                labels=viterbi_labels,
                class_groups=config.interval_entity_classes,
            )
            region_intervals.append(intervals.assign(strand=strand, decoding="viterbi"))

    region_intervals = pd.concat(region_intervals, ignore_index=True, axis=0)
    region_name_map = {
        i: config.interval_entity_name(i) for i in region_intervals["entity"].unique()
    }
    region_intervals = (
        region_intervals.rename(columns={"entity": "entity_index"})
        .assign(entity_name=lambda df: df["entity_index"].map(region_name_map))
        .rename_axis("interval", axis="index")
    )
    logger.info(f"Region intervals detected:\n{region_intervals}")
    logger.info("Region interval info:\n")
    region_intervals.info()
    region_intervals = region_intervals.to_xarray().assign_attrs(
        interval_entity_names=config.interval_entity_names
    )
    return region_intervals


def detect_intervals(
    input_dir: str,
    output: str,
    decode_direct: bool,
    viterbi_alpha: float,
    intergenic_bias: float,
    domain: str,
    remove_incomplete_features: bool,
):
    """Aggregate rank outputs and decode genomic intervals from logits.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments where ``args.input_dir`` points to
        ``predictions.*.zarr`` files produced by inference.
    """
    logger.info(
        f"Detecting intervals from rank files in {input_dir} and saving to {output}"
    )

    # Merge predictions from all ranks
    sequence_predictions = merge_prediction_datasets(
        input_dir,
        drop_variables=["token_predictions", "token_logits"],
    )

    logger.info("Detecting intervals")
    interval_predictions = _detect_intervals(
        predictions=sequence_predictions,
        decode_direct=decode_direct,
        viterbi_alpha=viterbi_alpha,
        intergenic_bias=intergenic_bias,
        domain=domain,
        remove_incomplete_features=remove_incomplete_features,
    )
    interval_predictions = interval_predictions.assign_attrs(
        # Copy attributes from sequence predictions, which have
        # been carried along from the original fasta extraction
        **sequence_predictions.attrs
    )

    logger.info("Merging sequence and interval predictions")
    result = xr.DataTree.from_dict(
        {
            "/sequences": sequence_predictions,
            "/intervals": interval_predictions,
        }
    )

    logger.info(f"Final results:\n{result}")

    logger.info(f"Saving results to output path {output}")
    result.to_zarr(output, zarr_format=2, mode="w", consolidated=True)

    logger.info("Done")


def main():
    """Convert base-level predictions to genomic intervals."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Suppress noisy HTTP traffic logs from HuggingFace Hub's internal HTTP client
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Converts base-level predictions to genomic intervals."
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default=None,
        help="Path to input dataset from predict",
    )
    parser.add_argument(
        "--output-zarr",
        "-o",
        type=str,
        default=None,
        help="Path to output zarr dataset for intervals",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Manifest json for multi-chromosome runs. Key-value pairs 'chromosome_id', 'predictions_dir' and "
        "'intervals_zarr' are required. Required if --input-dir and --output-zarr are not specified.",
    )

    parser.add_argument(
        "--viterbi-alpha",
        type=float,
        default=None,
        help="Alpha parameter for viterbi decoding (default: None)",
    )
    parser.add_argument(
        "--decode-direct",
        action="store_true",
        help="If set, decode using the direct method instead of the default viterbi method",
    )
    parser.add_argument(
        "--intergenic-bias",
        type=float,
        default=0.0,
        help=(
            "Amount to subtract from intergenic feature logits before interval "
            "decoding (default: 0.0)"
        ),
    )
    parser.add_argument(
        "--keep-incomplete-features",
        action="store_true",
        help="Keep incomplete features in the prediction",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["plant", "animal"],
        default="plant",
        help="Biological domain for Viterbi transition priors (default: plant)",
    )

    args = parser.parse_args()

    if args.manifest is None:
        if (args.input_dir is None) or (args.output_zarr is None):
            logger.error(
                "Error: one of the following must be provided:\n"
                "--manifest\n OR \n --input-dir and --output-zarr"
            )
            raise RuntimeError

        detect_intervals(
            input_dir=args.input_dir,
            output=args.output_zarr,
            decode_direct=args.decode_direct,
            viterbi_alpha=args.viterbi_alpha,
            intergenic_bias=args.intergenic_bias,
            domain=args.domain,
            remove_incomplete_features=(not args.keep_incomplete_features),
        )
    else:
        with open(args.manifest) as fh:
            entries = json.load(fh)

        for entry in entries:
            chromosome_id = entry["chromosome_id"]
            input_dir = entry["predictions_dir"]
            output_zarr = entry["intervals_zarr"]

            logger.info(f"Detecting intervals for chromosome {chromosome_id}")

            detect_intervals(
                input_dir=input_dir,
                output=output_zarr,
                decode_direct=args.decode_direct,
                viterbi_alpha=args.viterbi_alpha,
                intergenic_bias=args.intergenic_bias,
                domain=args.domain,
                remove_incomplete_features=(not args.keep_incomplete_features),
            )


if __name__ == "__main__":
    main()
