#!/usr/bin/env python3
"""
PC Quality Filter Experiment Analysis

This script analyzes the results from the PC quality filter experiment by:
1. Loading consolidated CSV files across all species and versions
2. Processing and filtering the data according to experimental design
3. Creating visualizations comparing model versions and ground truth filtering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import warnings

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration from experiment scripts
PIPE_DIR = (
    "/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline"
)
SPECIES_LIST = ["jregia", "pvulgaris", "carabica", "zmays", "ntabacum", "nsylvestris"]
MODEL_VERSIONS = ["1.0", "1.1", "1.2"]
GROUND_TRUTH_VARIANTS = ["0", "1"]  # 0=original, 1=pc-filtered
CHR_ID = "chr1"

# Target combinations to filter for
TARGET_COMBINATIONS = [
    ("Base", "gffcompare"),
    ("Exon", "gffcompare"),
    ("Transcript", "gffcompare"),
    ("transcript_cds", "gffeval"),
]

# Post-processing method mappings
POST_PROCESSING_MAPPING = {
    "minlen_only__with_utrs": "recall-optimized",
    "valid_only__with_utrs": "precision-optimized",
}

# Methods to ignore
IGNORE_METHODS = ["minlen_only__no_utrs", "valid_only__no_utrs"]

# Species display names
SPECIES_DISPLAY_NAMES = {
    "jregia": "J. regia",
    "pvulgaris": "P. vulgaris",
    "carabica": "C. arabica",
    "zmays": "Z. mays",
    "ntabacum": "N. tabacum",
    "nsylvestris": "N. sylvestris",
}


def construct_csv_path(species: str, version: str, variant: str) -> Path:
    """Construct the path to a consolidated CSV file."""
    return (
        Path(PIPE_DIR)
        / "predict"
        / species
        / "runs"
        / f"v{version}.{variant}"
        / CHR_ID
        / "results"
        / "gffcompare.stats.consolidated.csv"
    )


def load_all_csv_files() -> pd.DataFrame:
    """Load and combine all CSV files across species, versions, and variants."""
    data_frames = []

    for species in SPECIES_LIST:
        for version in MODEL_VERSIONS:
            for variant in GROUND_TRUTH_VARIANTS:
                csv_path = construct_csv_path(species, version, variant)

                if csv_path.exists():
                    logger.info(f"Loading {csv_path}")
                    try:
                        df = pd.read_csv(csv_path, sep="\t")
                        df["species"] = species
                        df["version"] = f"v{version}.{variant}"
                        df["major_version"] = f"v{version}"
                        df["label_filter"] = int(variant)  # 0=no filter, 1=pc-zs-filter
                        data_frames.append(df)
                    except Exception as e:
                        logger.warning(f"Failed to load {csv_path}: {e}")
                else:
                    logger.warning(f"File not found: {csv_path}")

    if not data_frames:
        raise ValueError("No CSV files were successfully loaded!")

    combined_df = pd.concat(data_frames, ignore_index=True)
    logger.info(
        f"Loaded {len(combined_df)} total records from {len(data_frames)} files"
    )
    return combined_df


def filter_and_process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and process the data according to experimental requirements."""
    logger.info("Processing data...")

    # Filter to target (level, tool) combinations
    target_mask = df[["level", "tool"]].apply(
        lambda row: (row["level"], row["tool"]) in TARGET_COMBINATIONS, axis=1
    )
    df_filtered = df[target_mask].copy()
    logger.info(f"After filtering to target combinations: {len(df_filtered)} records")

    # Remove records with methods to ignore
    df_filtered = df_filtered[~df_filtered["source"].isin(IGNORE_METHODS)]
    logger.info(f"After removing ignored methods: {len(df_filtered)} records")

    # Map post-processing methods
    df_filtered["post_processing_method"] = df_filtered["source"].map(
        POST_PROCESSING_MAPPING
    )

    # Remove records that couldn't be mapped
    unmapped_mask = df_filtered["post_processing_method"].isna()
    if unmapped_mask.any():
        unmapped_sources = df_filtered.loc[unmapped_mask, "source"].unique()
        logger.warning(f"Unmapped sources found (will be removed): {unmapped_sources}")
        df_filtered = df_filtered[~unmapped_mask]

    logger.info(f"Final processed dataset: {len(df_filtered)} records")

    # Add display names
    df_filtered["species_display"] = df_filtered["species"].map(SPECIES_DISPLAY_NAMES)
    df_filtered["label_filter_name"] = df_filtered["label_filter"].map(
        {0: "Original", 1: "PC-filtered"}
    )

    return df_filtered


def create_comprehensive_visualization(df: pd.DataFrame, output_dir: Path) -> None:
    """Create a comprehensive visualization of all results with fixed scales."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Get unique combinations for faceting
    unique_combinations = df[
        ["level", "tool", "post_processing_method"]
    ].drop_duplicates()
    n_combinations = len(unique_combinations)

    # Create subplots - arrange in a grid
    n_cols = 2
    n_rows = (n_combinations + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    # Hide extra subplots if needed
    for i in range(n_combinations, len(axes)):
        axes[i].set_visible(False)

    # Create a plot for each combination
    for idx, (_, combo) in enumerate(unique_combinations.iterrows()):
        level = combo["level"]
        tool = combo["tool"]
        method = combo["post_processing_method"]

        # Filter data for this combination
        combo_data = df[
            (df["level"] == level)
            & (df["tool"] == tool)
            & (df["post_processing_method"] == method)
        ]

        if combo_data.empty:
            logger.warning(f"No data for combination: {level}, {tool}, {method}")
            continue

        ax = axes[idx]

        # Plot for each species
        species_list = sorted(combo_data["species_display"].unique())
        colors = plt.cm.Set3(np.linspace(0, 1, len(species_list)))

        for i, species in enumerate(species_list):
            species_data = combo_data[combo_data["species_display"] == species]

            # Group by major version and label filter
            for label_filter in [0, 1]:
                filter_data = species_data[species_data["label_filter"] == label_filter]
                if filter_data.empty:
                    continue

                # Sort by major version for proper line connection
                filter_data = filter_data.sort_values("major_version")

                label_name = "Original" if label_filter == 0 else "PC-filtered"
                linestyle = "-" if label_filter == 0 else "--"
                marker = "o" if label_filter == 0 else "s"

                ax.plot(
                    filter_data["major_version"],
                    filter_data["f1_score"],
                    marker=marker,
                    linestyle=linestyle,
                    color=colors[i],
                    label=f"{species} ({label_name})",
                    linewidth=2,
                    markersize=8,
                )

        # Customize the subplot
        ax.set_xlabel("Model Version", fontsize=10, fontweight="bold")
        ax.set_ylabel("F1 Score", fontsize=10, fontweight="bold")
        ax.set_title(f"{level} - {tool} - {method}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Let each subplot have its own optimal y-scale starting just below minimum
        if not combo_data.empty:
            min_val = combo_data["f1_score"].min()
            margin = (combo_data["f1_score"].max() - min_val) * 0.05  # 5% margin
            ax.set_ylim(bottom=max(0, min_val - margin))  # Don't go below 0
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}"))

    # Collect all legend handles and labels from all subplots
    handles, labels = [], []
    for ax in axes[:n_combinations]:
        h, lab = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(lab)

    # Remove duplicates while preserving order
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    plt.tight_layout()

    # Add a single legend to the right of the entire figure
    fig.legend(
        unique_handles,
        unique_labels,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize=10,
    )

    # Save the comprehensive plot as PDF
    output_path = output_dir / "pc_quality_experiment_comprehensive.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    logger.info(f"Saved comprehensive plot: {output_path}")
    plt.close()


def create_focused_visualization(df: pd.DataFrame, output_dir: Path) -> None:
    """Create a focused visualization with only transcript-level results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Define the 4 specific combinations we want
    target_combinations = [
        ("Transcript", "gffcompare", "precision-optimized"),
        ("Transcript", "gffcompare", "recall-optimized"),
        ("transcript_cds", "gffeval", "precision-optimized"),
        ("transcript_cds", "gffeval", "recall-optimized"),
    ]

    # Create 2x2 subplot layout (narrower to give more space to legends)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    # Filter data to only include target combinations
    focused_df = df[
        df[["level", "tool", "post_processing_method"]].apply(
            lambda row: (row["level"], row["tool"], row["post_processing_method"])
            in target_combinations,
            axis=1,
        )
    ]

    # Create a plot for each target combination
    for idx, (level, tool, method) in enumerate(target_combinations):
        # Filter data for this combination
        combo_data = focused_df[
            (focused_df["level"] == level)
            & (focused_df["tool"] == tool)
            & (focused_df["post_processing_method"] == method)
        ]

        if combo_data.empty:
            logger.warning(f"No data for combination: {level}, {tool}, {method}")
            continue

        ax = axes[idx]

        # Plot for each species
        species_list = sorted(combo_data["species_display"].unique())
        colors = plt.cm.Set3(np.linspace(0, 1, len(species_list)))

        for i, species in enumerate(species_list):
            species_data = combo_data[combo_data["species_display"] == species]

            # Group by major version and label filter
            for label_filter in [0, 1]:
                filter_data = species_data[species_data["label_filter"] == label_filter]
                if filter_data.empty:
                    continue

                # Sort by major version for proper line connection
                filter_data = filter_data.sort_values("major_version")

                label_name = "Original" if label_filter == 0 else "PC-filtered"
                linestyle = "-" if label_filter == 0 else "--"
                marker = "o" if label_filter == 0 else "s"

                ax.plot(
                    filter_data["major_version"],
                    filter_data["f1_score"],
                    marker=marker,
                    linestyle=linestyle,
                    color=colors[i],
                    label=f"{species} ({label_name})",
                    linewidth=2,
                    markersize=10,
                )

        # Customize the subplot
        ax.set_xlabel("Model Version", fontsize=12, fontweight="bold")
        ax.set_ylabel("F1 Score", fontsize=12, fontweight="bold")
        ax.set_title(f"{level} - {tool} - {method}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Let each subplot have its own optimal y-scale starting just below minimum
        if not combo_data.empty:
            min_val = combo_data["f1_score"].min()
            margin = (combo_data["f1_score"].max() - min_val) * 0.05  # 5% margin
            ax.set_ylim(bottom=max(0, min_val - margin))  # Don't go below 0
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}"))

    # Collect all legend handles and labels from all subplots
    handles, labels = [], []
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(lab)

    # Remove duplicates while preserving order
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    plt.tight_layout()

    # Add a single legend to the right of the entire figure
    fig.legend(
        unique_handles,
        unique_labels,
        bbox_to_anchor=(1.02, 0.75),
        loc="upper left",
        fontsize=14,
    )

    # Add training data information
    training_info = [
        "Training Data Scale:",
        "",
        "v1.0 → 1 genome / 357M tokens (1x scale)",
        "v1.1 → 2 genomes / 1.68B tokens (4.7x scale)",
        "v1.2 → 5 genomes / 8.3B tokens (23.3x scale)",
    ]

    # Create text without background box
    textstr = "\n".join(training_info)
    fig.text(
        1.02,
        0.3,
        textstr,
        transform=fig.transFigure,
        fontsize=14,
        verticalalignment="top",
    )

    # Save the focused plot as PDF
    output_path = output_dir / "pc_quality_experiment_focused.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    logger.info(f"Saved focused plot: {output_path}")
    plt.close()


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print basic summary and pivot table to console."""

    print("\n" + "=" * 80)
    print("PC QUALITY FILTER EXPERIMENT - RESULTS")
    print("=" * 80)
    print(f"Total records: {len(df)}")
    print(f"Species: {', '.join(sorted(df['species_display'].unique()))}")
    print(f"Versions: {', '.join(sorted(df['major_version'].unique()))}")

    # Simple pivot showing F1 scores
    print("\n" + "=" * 80)
    print("F1 SCORES BY VERSION AND FILTER TYPE")
    print("=" * 80)

    # Create a combined column for version and filter type
    df["version_filter"] = df["major_version"] + "_" + df["label_filter_name"]

    # Use pivot to avoid any aggregation
    pivot = df.pivot(
        index=["species_display", "level", "tool", "post_processing_method"],
        columns="version_filter",
        values="f1_score",
    ).round(1)

    print(pivot.to_string())


def main():
    """Main analysis function."""
    logger.info("Starting PC Quality Filter Experiment Analysis")

    # Set up output directory for the single plot
    output_dir = Path("local/scratch")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load and process data
        raw_data = load_all_csv_files()
        processed_data = filter_and_process_data(raw_data)

        # Save data to CSV files as requested
        raw_data_path = output_dir / "all_evaluation_statistics.csv"
        processed_data_path = output_dir / "primary_evaluation_statistics.csv"

        raw_data.to_csv(raw_data_path, sep="\t", index=False)
        processed_data.to_csv(processed_data_path, sep="\t", index=False)

        logger.info(f"Saved raw data to: {raw_data_path}")
        logger.info(f"Saved processed data to: {processed_data_path}")

        # Create both visualizations
        create_comprehensive_visualization(processed_data, output_dir)
        create_focused_visualization(processed_data, output_dir)

        # Print summary statistics to console
        print_summary_statistics(processed_data)

        logger.info(
            f"Analysis complete! Plots saved to: {output_dir}/pc_quality_experiment_comprehensive.pdf and {output_dir}/pc_quality_experiment_focused.pdf"
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
