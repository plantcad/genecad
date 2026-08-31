import argparse
import logging
from numpy import typing as npt
from src.atomic_io import atomic_output_path
from src.sequence import (
    convert_entity_labels_to_intervals,
    regularize_transition_matrix,
    viterbi_decode,
)
from src.frame_crf import (
    AT_AC,
    DEFAULT_EXON_LENGTH_STRICTNESS,
    DEFAULT_MIN_CODING_RUN_LENGTH,
    DEFAULT_MIN_INTRON_LENGTH,
    DEFAULT_SPLICE_MOTIF_GROUPS,
    GT_AG,
    SpliceMotifGroup,
    encode_sequence,
    frame_aware_decode,
    reverse_complement_codes,
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
    base_codes: np.ndarray | None = None,
    min_intron_length: int = DEFAULT_MIN_INTRON_LENGTH,
    splice_motif_groups: tuple[SpliceMotifGroup, ...] = DEFAULT_SPLICE_MOTIF_GROUPS,
    min_coding_run_length: int = DEFAULT_MIN_CODING_RUN_LENGTH,
    exon_length_strictness: float = DEFAULT_EXON_LENGTH_STRICTNESS,
    include_utr_in_coding_run: bool = True,
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
        logits: npt.ArrayLike,
        remove_incomplete_features: bool,
        strand_base_codes: np.ndarray | None = None,
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

        alpha = viterbi_alpha
        matrix = transition_probs.values
        if strand_base_codes is not None:
            # Regularization has to be applied to the 5x5 feature matrix before
            # it is expanded: smoothing the expanded matrix would fill in the
            # structural zeros that carry the reading frame.
            if alpha is not None:
                matrix = regularize_transition_matrix(matrix, alpha)
            logger.info(f"Running frame-aware viterbi decoding ({alpha=})")
            labels = frame_aware_decode(
                feature_probs=emissions,
                base_codes=strand_base_codes,
                feature_transition=matrix,
                min_intron_length=min_intron_length,
                splice_motif_groups=splice_motif_groups,
                min_coding_run_length=min_coding_run_length,
                exon_length_strictness=exon_length_strictness,
                include_utr_in_coding_run=include_utr_in_coding_run,
            )
        else:
            # Decoding takes ~90 seconds for 308452471 tokens on Grace CPU
            logger.info(f"Running viterbi decoding ({alpha=})")
            labels = viterbi_decode(
                emission_probs=emissions,
                transition_matrix=matrix,
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
                    strand_base_codes=base_codes,
                )
            else:
                # The minus strand is decoded on the reversed logit array, so the
                # sequence must be reverse complemented to stay in register.
                viterbi_labels = flip(
                    _decode_intervals_viterbi(
                        logits=flip(logits).copy(),
                        remove_incomplete_features=remove_incomplete_features,
                        strand_base_codes=(
                            None
                            if base_codes is None
                            else reverse_complement_codes(base_codes)
                        ),
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


def load_chromosome_codes(fasta_path: str, chromosome_id: str) -> np.ndarray:
    """Read one chromosome from a FASTA file and encode it as base codes.

    The file is streamed a record at a time so that whole-genome FASTAs do not
    have to be held in memory.
    """
    import gzip

    opener = gzip.open if fasta_path.endswith(".gz") else open
    with opener(fasta_path, "rt") as fh:  # pyrefly: ignore[bad-argument-type]
        current: str | None = None
        chunks: list[str] = []
        for line in fh:
            if line.startswith(">"):
                if current == chromosome_id:
                    break
                current = line[1:].strip().split()[0]
                chunks = []
            elif current == chromosome_id:
                chunks.append(line.strip())
    if not chunks:
        raise ValueError(f"Sequence {chromosome_id!r} not found in {fasta_path}")
    logger.info(
        f"Loaded sequence {chromosome_id!r} ({sum(len(c) for c in chunks)} bp) "
        f"from {fasta_path}"
    )
    return encode_sequence("".join(chunks))


def detect_intervals(
    input_dir: str,
    output: str,
    decode_direct: bool,
    viterbi_alpha: float,
    intergenic_bias: float,
    domain: str,
    remove_incomplete_features: bool,
    input_fasta: str | None = None,
    min_intron_length: int = DEFAULT_MIN_INTRON_LENGTH,
    splice_motif_groups: tuple[SpliceMotifGroup, ...] = DEFAULT_SPLICE_MOTIF_GROUPS,
    min_coding_run_length: int = DEFAULT_MIN_CODING_RUN_LENGTH,
    exon_length_strictness: float = DEFAULT_EXON_LENGTH_STRICTNESS,
    include_utr_in_coding_run: bool = True,
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

    base_codes = None
    if input_fasta is not None:
        chromosome_id = sequence_predictions.attrs["chromosome_id"]
        base_codes = load_chromosome_codes(input_fasta, chromosome_id)
        n_positions = sequence_predictions.sizes["sequence"]
        if len(base_codes) != n_positions:
            raise ValueError(
                f"Sequence {chromosome_id!r} has {len(base_codes)} bases but "
                f"{n_positions} positions were predicted; frame-aware decoding "
                f"requires the FASTA used for prediction"
            )

    logger.info("Detecting intervals")
    interval_predictions = _detect_intervals(
        predictions=sequence_predictions,
        decode_direct=decode_direct,
        viterbi_alpha=viterbi_alpha,
        intergenic_bias=intergenic_bias,
        domain=domain,
        remove_incomplete_features=remove_incomplete_features,
        base_codes=base_codes,
        min_intron_length=min_intron_length,
        splice_motif_groups=splice_motif_groups,
        min_coding_run_length=min_coding_run_length,
        exon_length_strictness=exon_length_strictness,
        include_utr_in_coding_run=include_utr_in_coding_run,
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
    # Write to a .tmp path and rename onto `output` only once fully written,
    # so a killed run never leaves a partial Zarr store at the path
    # predict.sh's resume check looks for.
    with atomic_output_path(output) as tmp_output:
        result.to_zarr(tmp_output, zarr_format=2, mode="w", consolidated=True)

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
        "--input-fasta",
        "-f",
        type=str,
        default=None,
        help="Genome FASTA used for prediction. When given, decoding becomes "
        "frame-aware: the CDS is constrained to begin on ATG, end on a stop "
        "codon, stay in frame across introns, and contain no in-frame stop. "
        "Ignored when --decode-direct is set.",
    )
    parser.add_argument(
        "--min-intron-length",
        type=int,
        default=DEFAULT_MIN_INTRON_LENGTH,
        help="Shortest intron frame-aware decoding may emit. Guards against short "
        "introns being invented to step over an in-frame stop codon.",
    )
    parser.add_argument(
        "--min-coding-run-length",
        type=int,
        default=DEFAULT_MIN_CODING_RUN_LENGTH,
        help="Runs of coding sequence adjacent to an intron shorter than this are "
        "penalized (see --exon-length-strictness) rather than forbidden -- guards "
        "against an intron being invented immediately after the start codon or "
        "immediately after another intron, without also destroying genuine short "
        "exons. Pass 0 to disable the penalty entirely.",
    )
    parser.add_argument(
        "--exon-length-strictness",
        type=float,
        default=DEFAULT_EXON_LENGTH_STRICTNESS,
        help="How strongly to penalize a run of coding sequence below --min-coding-run-length: 0 "
        "removes the penalty, larger values fall off more steeply and converge on "
        "treating --min-coding-run-length as a hard floor. Strong per-base emission "
        "evidence can still outweigh the penalty at any setting above 0, which is "
        "what lets genuine short boundary exons survive.",
    )
    parser.add_argument(
        "--include-utr-in-coding-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Count the 5' UTR alongside the coding run for "
        "--min-coding-run-length purposes, so a long UTR can by itself exempt "
        "a short first coding run from the penalty. Only the start side is "
        "covered. On by default; pass --no-include-utr-in-coding-run to disable.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["plant", "animal"],
        default="plant",
        help="Biological domain for Viterbi transition priors (default: plant)",
    )
    parser.add_argument(
        "--allow-u12-introns",
        action="store_true",
        help="Also allow U12-type AT-AC introns during frame-aware decoding "
        "(default: only GT-AG/GC-AG). AT-AC introns are real but rare "
        "(~0.04%% of introns in the TAIR12 reference); enabling this roughly "
        "doubles the intron state count. Ignored unless --input-fasta is set.",
    )

    args = parser.parse_args()
    splice_motif_groups = (
        (GT_AG, AT_AC) if args.allow_u12_introns else DEFAULT_SPLICE_MOTIF_GROUPS
    )

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
            input_fasta=args.input_fasta,
            min_intron_length=args.min_intron_length,
            splice_motif_groups=splice_motif_groups,
            min_coding_run_length=args.min_coding_run_length,
            exon_length_strictness=args.exon_length_strictness,
            include_utr_in_coding_run=args.include_utr_in_coding_run,
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
                input_fasta=args.input_fasta,
                min_intron_length=args.min_intron_length,
                splice_motif_groups=splice_motif_groups,
                min_coding_run_length=args.min_coding_run_length,
                exon_length_strictness=args.exon_length_strictness,
                include_utr_in_coding_run=args.include_utr_in_coding_run,
            )


if __name__ == "__main__":
    main()
