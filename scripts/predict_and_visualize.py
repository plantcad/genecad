"""Predict gene structure from a FASTA file and visualize the results.

Pipeline:
  1. Extract FASTA  → sequences.zarr
  2. Run inference  → predictions.zarr
  3. Detect intervals → intervals.zarr
  4. Generate plots from the zarr output

Usage
-----
python scripts/predict_and_visualize.py \
    --fasta genome.fa \
    --checkpoint /path/to/model.ckpt \
    --model-path emarro/pcad2-200M-cnet-baseline \
    --species-id MySpecies \
    --chromosome Chr1 \
    --output-dir ./results

All pipeline steps are skipped when their output already exists, so you can
re-run the script cheaply to regenerate plots after tweaking visualization
options without re-running inference.
"""

import argparse
import logging
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Colour palette for entity types
# ---------------------------------------------------------------------------

ENTITY_COLORS = {
    "transcript": "#4878d0",
    "exon": "#6acc65",
    "intron": "#d65f5f",
    "cds": "#ee854a",
    "five_prime_utr": "#956cb4",
    "three_prime_utr": "#8c613c",
    "intergenic": "#cccccc",
}

STRAND_COLORS = {"positive": "#2166ac", "negative": "#d6604d"}


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _subsample_indices(size: int, max_points: int) -> np.ndarray:
    """Return indices that sample a sequence of *size* to at most *max_points*."""
    if max_points <= 0:
        raise ValueError(f"max_points must be positive, got {max_points}")
    if size <= max_points:
        return np.arange(size)
    step = (size + max_points - 1) // max_points
    return np.arange(0, size, step)


def _subsample(arr: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, values) subsampled to at most *max_points*."""
    idx = _subsample_indices(len(arr), max_points)
    return idx, arr[idx]


def plot_feature_logits(
    ax: plt.Axes,
    positions: np.ndarray,
    feature_logits: np.ndarray,
    feature_names: list[str],
    max_points: int = 4000,
    title: str = "Feature Logits",
) -> None:
    """Heatmap of entity/feature logits across genomic positions."""
    idx, _ = _subsample(positions, max_points)
    logits_sub = feature_logits[idx]  # (n_sub, n_features)
    pos_sub = positions[idx]

    im = ax.imshow(
        logits_sub.T,
        aspect="auto",
        cmap="RdYlGn",
        interpolation="nearest",
        extent=[pos_sub[0], pos_sub[-1], -0.5, len(feature_names) - 0.5],
        origin="lower",
    )
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Logit", fraction=0.02, pad=0.01)


def plot_feature_predictions(
    ax: plt.Axes,
    positions: np.ndarray,
    predictions: np.ndarray,
    feature_names: list[str],
    max_points: int = 8000,
    title: str = "Predicted Feature Labels",
) -> None:
    """Raster of predicted class index, colour-coded by entity name."""
    n_feat = len(feature_names)
    cmap = plt.cm.get_cmap("tab10", n_feat)

    idx, pred_sub = _subsample(predictions, max_points)
    pos_sub = positions[idx]

    ax.scatter(
        pos_sub,
        pred_sub,
        c=[cmap(int(p)) for p in pred_sub],
        s=1,
        linewidths=0,
        marker=",",
    )
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_ylim(-0.5, n_feat - 0.5)
    ax.set_title(title)

    legend_handles = [
        mpatches.Patch(color=cmap(i), label=feature_names[i]) for i in range(n_feat)
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=7,
        framealpha=0.6,
        ncol=2,
    )


def plot_intervals(
    ax: plt.Axes,
    intervals_df: pd.DataFrame,
    region_start: int,
    region_end: int,
    strand: str,
    title: str = "Predicted Gene Structure",
) -> None:
    """Draw interval blocks (genes, exons, CDS, UTRs) as horizontal bars."""
    df = intervals_df[
        (intervals_df["strand"] == strand)
        & (intervals_df["stop"] >= region_start)
        & (intervals_df["start"] <= region_end)
    ].copy()

    entity_order = [
        "transcript",
        "exon",
        "five_prime_utr",
        "cds",
        "three_prime_utr",
        "intron",
    ]
    entity_y = {name: i for i, name in enumerate(entity_order)}
    n_tracks = len(entity_order)

    ax.set_ylim(-0.6, n_tracks - 0.4)
    ax.set_yticks(range(n_tracks))
    ax.set_yticklabels(entity_order, fontsize=8)
    ax.set_xlim(region_start, region_end)

    for _, row in df.iterrows():
        name = row["entity_name"].lower()
        y = entity_y.get(name, n_tracks - 1)
        color = ENTITY_COLORS.get(name, "#aaaaaa")
        start = max(row["start"], region_start)
        stop = min(row["stop"], region_end)
        width = stop - start + 1
        ax.barh(y, width, left=start, height=0.6, color=color, alpha=0.75, linewidth=0)

    legend_handles = [
        mpatches.Patch(color=ENTITY_COLORS.get(n, "#aaaaaa"), label=n)
        for n in entity_order
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=7,
        framealpha=0.6,
        ncol=3,
    )
    ax.set_title(title)


def plot_strand(
    intervals_df: pd.DataFrame,
    sequences_ds: xr.Dataset,
    strand: str,
    region_start: int,
    region_end: int,
    output_path: str,
    max_logit_points: int = 4000,
) -> None:
    """Create and save a multi-panel plot for one strand."""
    chromosome_start = int(sequences_ds.sequence.isel(sequence=0).item())
    chromosome_end = int(sequences_ds.sequence.isel(sequence=-1).item())
    clipped_start = max(region_start, chromosome_start)
    clipped_end = min(region_end, chromosome_end)
    if clipped_start > clipped_end:
        raise ValueError(
            f"Region {region_start}-{region_end} does not overlap chromosome coordinates "
            f"{chromosome_start}-{chromosome_end}"
        )
    first_region_index = clipped_start - chromosome_start
    region_size = clipped_end - clipped_start + 1

    feature_names = sequences_ds.feature.values.tolist()
    logit_offsets = _subsample_indices(region_size, max_logit_points)
    logit_indices = first_region_index + logit_offsets
    logit_positions = clipped_start + logit_offsets
    feature_logits = (
        sequences_ds.feature_logits.sel(strand=strand)
        .isel(sequence=logit_indices)
        .values
    )

    pred_offsets = _subsample_indices(region_size, 8000)
    pred_indices = first_region_index + pred_offsets
    pred_positions = clipped_start + pred_offsets
    feature_preds = (
        sequences_ds.feature_predictions.sel(strand=strand)
        .isel(sequence=pred_indices)
        .values
    )

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 1, hspace=0.45)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    strand_label = "+" if strand == "positive" else "-"
    chrom = sequences_ds.attrs.get("chromosome_id", "?")
    decoding = intervals_df["decoding"].iloc[0] if len(intervals_df) > 0 else "unknown"

    plot_feature_logits(
        ax0,
        logit_positions,
        feature_logits,
        feature_names,
        max_points=max_logit_points,
        title=f"Feature Logits  |  {chrom}:{region_start}-{region_end} strand {strand_label}",
    )
    plot_feature_predictions(
        ax1,
        pred_positions,
        feature_preds,
        feature_names,
        title="Direct Argmax Feature Labels",
    )
    plot_intervals(
        ax2,
        intervals_df,
        region_start,
        region_end,
        strand=strand,
        title=f"Predicted Gene Structure  |  decoding: {decoding}  |  strand {strand_label}",
    )

    for ax in (ax0, ax1, ax2):
        ax.set_xlabel(f"Genomic position ({chrom})")

    plt.suptitle(
        f"GeneCAD Predictions  |  {chrom}  strand {strand_label}\n"
        f"region {region_start:,}–{region_end:,}",
        fontsize=13,
        y=1.01,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Saved plot: {output_path}")


# ---------------------------------------------------------------------------
# Pipeline steps (thin wrappers that call existing scripts)
# ---------------------------------------------------------------------------


def _python() -> str:
    """Return the Python interpreter to use."""
    for candidate in [
        os.path.join(os.path.dirname(sys.executable), "python"),
        sys.executable,
    ]:
        if os.path.isfile(candidate):
            return candidate
    return sys.executable


def _run(cmd: list[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        PROJECT_ROOT
        if not env.get("PYTHONPATH")
        else os.pathsep.join([PROJECT_ROOT, env["PYTHONPATH"]])
    )
    result = subprocess.run(cmd, check=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}")


def step_extract(
    fasta: str,
    species_id: str,
    chromosome: str,
    model_path: str,
    output_zarr: str,
    script_dir: str,
) -> None:
    if os.path.exists(output_zarr):
        logger.info("[1/3] sequences.zarr already exists — skipping extraction")
        return
    logger.info("[1/3] Extracting FASTA → sequences.zarr ...")
    _run(
        [
            _python(),
            os.path.join(script_dir, "extract.py"),
            "extract_fasta_file",
            "--species-id",
            species_id,
            "--fasta-file",
            fasta,
            "--chrom-map",
            f"{chromosome}:{chromosome}",
            "--tokenizer-path",
            model_path,
            "--output",
            output_zarr,
        ]
    )


def step_predict(
    sequences_zarr: str,
    species_id: str,
    chromosome: str,
    model_path: str,
    checkpoint: str,
    predictions_dir: str,
    batch_size: int,
    device: str,
    dtype: str,
    window_size: int,
    stride: int,
    script_dir: str,
) -> None:
    marker = os.path.join(predictions_dir, "predictions.0.zarr")
    if os.path.exists(marker):
        logger.info("[2/3] predictions.zarr already exists — skipping inference")
        return
    logger.info("[2/3] Running inference → predictions.zarr ...")
    _run(
        [
            _python(),
            os.path.join(script_dir, "predict.py"),
            "create_predictions",
            "--input",
            sequences_zarr,
            "--output-dir",
            predictions_dir,
            "--model-path",
            model_path,
            "--model-checkpoint",
            checkpoint,
            "--species-id",
            species_id,
            "--chromosome-id",
            chromosome,
            "--batch-size",
            str(batch_size),
            "--device",
            device,
            "--dtype",
            dtype,
            "--window-size",
            str(window_size),
            "--stride",
            str(stride),
            "--suppress-dynamo-errors",
            "yes",
        ]
    )


def step_detect_intervals(
    predictions_dir: str,
    intervals_zarr: str,
    decoding_methods: str,
    domain: str,
    script_dir: str,
) -> None:
    if os.path.exists(intervals_zarr):
        logger.info("[3/3] intervals.zarr already exists — skipping interval detection")
        return
    logger.info("[3/3] Detecting intervals → intervals.zarr ...")
    _run(
        [
            _python(),
            os.path.join(script_dir, "predict.py"),
            "detect_intervals",
            "--input-dir",
            predictions_dir,
            "--output",
            intervals_zarr,
            "--decoding-methods",
            decoding_methods,
            "--remove-incomplete-features",
            "yes",
            "--domain",
            domain,
        ]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Run GeneCAD prediction pipeline and visualize results."
    )

    # --- Required inputs ---
    parser.add_argument(
        "--fasta", required=True, help="Input genome FASTA file (.fa or .fa.gz)"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to GeneCAD classifier checkpoint (.ckpt) or HuggingFace repo ID",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to base encoder model or HuggingFace repo ID (e.g. emarro/pcad2-200M-cnet-baseline)",
    )
    parser.add_argument(
        "--species-id", required=True, help="Species label (e.g. Athaliana)"
    )
    parser.add_argument(
        "--chromosome",
        required=True,
        help="Chromosome/contig ID as it appears in the FASTA header",
    )

    # --- Output ---
    parser.add_argument(
        "--output-dir",
        default="genecad_result/visualization",
        help="Directory to write pipeline zarr files and output plots (default: genecad_result/visualization)",
    )

    # --- Genomic region to visualize ---
    parser.add_argument(
        "--region-start",
        type=int,
        default=None,
        help="0-based start position of the region to visualize (default: 0)",
    )
    parser.add_argument(
        "--region-end",
        type=int,
        default=None,
        help="0-based end position (inclusive) of the region to visualize (default: full chromosome)",
    )
    parser.add_argument(
        "--strand",
        choices=["positive", "negative", "both"],
        default="both",
        help="Strand(s) to visualize (default: both)",
    )

    # --- Model / inference options ---
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Inference batch size (default: 32)"
    )
    parser.add_argument(
        "--device", default="cuda", help="PyTorch device string (default: cuda)"
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Inference dtype (default: bfloat16)",
    )
    parser.add_argument("--window-size", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=4096)
    parser.add_argument(
        "--domain",
        choices=["plant", "animal"],
        default="plant",
        help="Biological domain for Viterbi priors (default: plant)",
    )
    parser.add_argument(
        "--decoding-methods",
        default="viterbi",
        help="Comma-separated list of decoding methods: direct,viterbi (default: viterbi)",
    )

    # --- Visualization options ---
    parser.add_argument(
        "--max-logit-points",
        type=int,
        default=4000,
        help="Max genomic positions to render in the logits heatmap (subsampled; default: 4000)",
    )

    args = parser.parse_args()

    pipeline_dir = os.path.join(args.output_dir, args.chromosome, "pipeline")
    os.makedirs(pipeline_dir, exist_ok=True)

    sequences_zarr = os.path.join(pipeline_dir, "sequences.zarr")
    predictions_dir = os.path.join(pipeline_dir, "predictions.zarr")
    intervals_zarr = os.path.join(pipeline_dir, "intervals.zarr")

    # --- Step 1: Extract ---
    step_extract(
        fasta=args.fasta,
        species_id=args.species_id,
        chromosome=args.chromosome,
        model_path=args.model_path,
        output_zarr=sequences_zarr,
        script_dir=SCRIPT_DIR,
    )

    # --- Step 2: Predict ---
    step_predict(
        sequences_zarr=sequences_zarr,
        species_id=args.species_id,
        chromosome=args.chromosome,
        model_path=args.model_path,
        checkpoint=args.checkpoint,
        predictions_dir=predictions_dir,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        window_size=args.window_size,
        stride=args.stride,
        script_dir=SCRIPT_DIR,
    )

    # --- Step 3: Detect intervals ---
    step_detect_intervals(
        predictions_dir=predictions_dir,
        intervals_zarr=intervals_zarr,
        decoding_methods=args.decoding_methods,
        domain=args.domain,
        script_dir=SCRIPT_DIR,
    )

    # --- Step 4: Load results and visualize ---
    logger.info("Loading results from %s", intervals_zarr)

    # Lazy-load — we only read the region of interest
    from src.dataset import open_datatree

    dt = open_datatree(intervals_zarr, consolidated=False)
    sequences_ds: xr.Dataset = dt["/sequences"].ds
    intervals_ds: xr.Dataset = dt["/intervals"].ds

    chromosome_start = int(sequences_ds.sequence.isel(sequence=0).item())
    chromosome_end = int(sequences_ds.sequence.isel(sequence=-1).item())
    region_start = (
        args.region_start if args.region_start is not None else chromosome_start
    )
    region_end = args.region_end if args.region_end is not None else chromosome_end
    if region_start > region_end:
        parser.error(
            f"--region-start ({region_start}) must be less than or equal to "
            f"--region-end ({region_end})"
        )

    logger.info(
        "Visualizing region %d–%d on chromosome %s",
        region_start,
        region_end,
        args.chromosome,
    )

    intervals_df = intervals_ds.to_dataframe().reset_index()

    strands_to_plot = (
        ["positive", "negative"] if args.strand == "both" else [args.strand]
    )

    plots_dir = os.path.join(args.output_dir, args.chromosome)
    os.makedirs(plots_dir, exist_ok=True)

    decoding_methods = intervals_df["decoding"].dropna().unique().tolist()
    if not decoding_methods:
        decoding_methods = ["unknown"]

    for decoding in decoding_methods:
        decoding_intervals = intervals_df[intervals_df["decoding"] == decoding]
        for strand in strands_to_plot:
            strand_label = "pos" if strand == "positive" else "neg"
            plot_path = os.path.join(
                plots_dir,
                f"predictions_{decoding}_{strand_label}_{region_start}_{region_end}.png",
            )
            plot_strand(
                intervals_df=decoding_intervals,
                sequences_ds=sequences_ds,
                strand=strand,
                region_start=region_start,
                region_end=region_end,
                output_path=plot_path,
                max_logit_points=args.max_logit_points,
            )
            print(f"Plot saved: {plot_path}")

    print("Done.")


if __name__ == "__main__":
    main()
