#!/usr/bin/env python3
"""
GeneCAD Pipeline Configuration Sweep Analysis

This script analyzes the results from the GeneCAD pipeline configuration sweep by:
1. Loading consolidated CSV files across all species and versions
2. Processing and filtering the data according to experimental design
3. Creating visualizations comparing model versions, ground-truth variants, and post-processing choices
"""

import pandas as pd
import plotnine as pn
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
MODEL_VERSIONS = ["1.0", "1.1", "1.2", "1.3", "1.4", "1.5"]
GROUND_TRUTH_VARIANTS = [
    "0",
    "1",
    "2",
    "3",
]  # 0=original/viterbi, 1=pc-filtered/viterbi, 2=original/direct, 3=pc-filtered/direct
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
                        # Map variants to ground truth and decoding method
                        if variant == "0":
                            df["ground_truth"] = "original"
                            df["decoding_method"] = "viterbi"
                        elif variant == "1":
                            df["ground_truth"] = "pc-filtered"
                            df["decoding_method"] = "viterbi"
                        elif variant == "2":
                            df["ground_truth"] = "original"
                            df["decoding_method"] = "direct"
                        elif variant == "3":
                            df["ground_truth"] = "pc-filtered"
                            df["decoding_method"] = "direct"
                        data_frames.append(df)
                    except Exception as e:
                        logger.warning(f"Failed to load {csv_path}: {e}")
                else:
                    # Warn only for these variants where sparsity in results is unexpected
                    if variant in ["0", "1"]:
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
    df_filtered["training_data"] = df_filtered["major_version"].map(
        {
            "v1.0": "1x-genome",
            "v1.1": "2x-genome",
            "v1.2": "5x-genome",
            "v1.3": "2x-genome",  # Same as v1.1 but with randomized base encoder
            "v1.4": "2x-genome",  # Same as v1.1 but with large PlantCAD base model
            "v1.5": "5x-genome",  # Same as v1.2 but starts from v1.4 checkpoint with large base model
        }
    )
    df_filtered["model_architecture"] = df_filtered["major_version"].map(
        {
            "v1.0": "plantcad+bert",
            "v1.1": "plantcad+bert",
            "v1.2": "plantcad+bert",
            "v1.3": "random_plantcad+bert",
            "v1.4": "plantcad+bert",
            "v1.5": "plantcad+bert",
        }
    )
    df_filtered["base_model"] = df_filtered["major_version"].map(
        {
            "v1.0": "PlantCAD2-Small",
            "v1.1": "PlantCAD2-Small",
            "v1.2": "PlantCAD2-Small",
            "v1.3": "PlantCAD2-Small",
            "v1.4": "PlantCAD2-Large",
            "v1.5": "PlantCAD2-Large",
        }
    )
    assert df_filtered["training_data"].notnull().all()
    assert df_filtered["model_architecture"].notnull().all()
    assert df_filtered["base_model"].notnull().all()

    # Check primary key uniqueness
    primary_key = [
        "species",
        "training_data",
        "model_architecture",
        "base_model",
        "ground_truth",
        "decoding_method",
        "level",
        "tool",
        "post_processing_method",
    ]

    # Assert no duplicates in primary key
    duplicates = df_filtered.duplicated(subset=primary_key)
    if duplicates.any():
        duplicate_rows = df_filtered[duplicates][primary_key]
        logger.error(
            f"Found {duplicates.sum()} duplicate records based on primary key:"
        )
        logger.error(f"\n{duplicate_rows.to_string()}")
        raise ValueError(
            f"Primary key violation: {duplicates.sum()} duplicate records found"
        )
    logger.info(f"Primary key validation passed: no duplicates found in {primary_key}")

    # Add primary key as dataframe attribute
    df_filtered.attrs["primary_key"] = primary_key
    df_filtered.attrs["metrics"] = ["sensitivity", "precision", "f1_score"]

    return df_filtered


def create_visualization(df: pd.DataFrame, output_dir: Path) -> None:
    """Create three focused visualizations using plotnine."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to transcript-level results only
    transcript_df = df[
        (df["level"] == "Transcript")
        & (df["tool"] == "gffcompare")
        & (df["post_processing_method"] == "precision-optimized")
    ].copy()

    # 1. Training scale impact: F1 by training_data (viterbi + original), shape by species
    plot1_data = transcript_df[
        (transcript_df["decoding_method"] == "viterbi")
        & (transcript_df["ground_truth"] == "original")
        & (transcript_df["model_architecture"] == "plantcad+bert")
        & (transcript_df["base_model"] == "PlantCAD2-Small")
    ].copy()

    plot1 = (
        pn.ggplot(
            plot1_data.assign(
                training_data=lambda df: pd.Categorical(
                    df["training_data"],
                    categories=(
                        df.groupby("training_data")["f1_score"]
                        .median()
                        .sort_values()
                        .index
                    ),
                    ordered=True,
                )
            ),
            pn.aes(x="training_data", y="f1_score"),
        )
        + pn.geom_point(mapping=pn.aes(shape="species_display"), size=3, alpha=0.8)
        + pn.geom_boxplot(
            alpha=0.2, width=0.1, position=pn.position_nudge(x=-0.2), outlier_size=0.2
        )
        + pn.labs(
            title="Training Scale Impact on Gene Annotation Performance",
            x="Training Data",
            y="Transcript F1",
            shape="Species",
        )
        + pn.theme_minimal()
        + pn.theme(axis_text_x=pn.element_text(angle=45, hjust=1))
    )

    # 2. Ground truth filtering impact: F1 by ground_truth (viterbi), shape by species, color by training_data
    plot2_data = transcript_df[
        (transcript_df["decoding_method"] == "viterbi")
        & (transcript_df["model_architecture"] == "plantcad+bert")
        & (transcript_df["base_model"] == "PlantCAD2-Small")
    ].copy()

    plot2 = (
        pn.ggplot(
            plot2_data.assign(
                ground_truth=lambda df: pd.Categorical(
                    df["ground_truth"],
                    categories=(
                        df.groupby("ground_truth")["f1_score"]
                        .median()
                        .sort_values()
                        .index
                    ),
                    ordered=True,
                )
            ),
            pn.aes(x="ground_truth", y="f1_score"),
        )
        + pn.geom_boxplot(
            alpha=0.2, width=0.1, position=pn.position_nudge(x=-0.2), outlier_size=0.2
        )
        + pn.geom_point(
            mapping=pn.aes(shape="species_display", color="training_data"),
            position=pn.position_jitter(width=0.1, height=0, random_state=42),
            size=3,
            alpha=0.8,
        )
        + pn.labs(
            title="Ground Truth Filtering Impact on Gene Annotation Performance",
            x="Ground Truth Type",
            y="Transcript F1",
            shape="Species",
            color="Training Data",
        )
        + pn.theme_minimal()
        + pn.theme(axis_text_x=pn.element_text(angle=45, hjust=1))
    )

    # 3. Decoding method impact: F1 by decoding method (original + 2x-genome), shape by species
    plot3_data = transcript_df[
        (transcript_df["ground_truth"] == "original")
        & (transcript_df["training_data"] == "2x-genome")
        & (transcript_df["model_architecture"] == "plantcad+bert")
        & (transcript_df["base_model"] == "PlantCAD2-Small")
    ].copy()

    plot3 = (
        pn.ggplot(
            plot3_data.assign(
                decoding_method=lambda df: pd.Categorical(
                    df["decoding_method"],
                    categories=(
                        df.groupby("decoding_method")["f1_score"]
                        .median()
                        .sort_values()
                        .index
                    ),
                    ordered=True,
                )
            ),
            pn.aes(x="decoding_method", y="f1_score"),
        )
        + pn.geom_boxplot(
            alpha=0.2, width=0.1, position=pn.position_nudge(x=-0.2), outlier_size=0.2
        )
        + pn.geom_point(mapping=pn.aes(shape="species_display"), size=3, alpha=0.8)
        + pn.labs(
            title="Decoding Method Impact on Gene Annotation Performance",
            x="Decoding Method",
            y="Transcript F1",
            shape="Species",
        )
        + pn.theme_minimal()
        + pn.theme(axis_text_x=pn.element_text(angle=45, hjust=1))
    )

    # 4. Model architecture impact: F1 by model_architecture (original + 2x-genome + viterbi), shape by species
    plot4_data = transcript_df[
        (transcript_df["ground_truth"] == "original")
        & (transcript_df["training_data"] == "2x-genome")
        & (transcript_df["decoding_method"] == "viterbi")
        & (transcript_df["base_model"] == "PlantCAD2-Small")
    ].copy()

    plot4 = (
        pn.ggplot(
            plot4_data.assign(
                model_architecture=lambda df: pd.Categorical(
                    df["model_architecture"],
                    categories=(
                        df.groupby("model_architecture")["f1_score"]
                        .median()
                        .sort_values()
                        .index
                    ),
                    ordered=True,
                )
            ),
            pn.aes(x="model_architecture", y="f1_score"),
        )
        + pn.geom_boxplot(
            alpha=0.2, width=0.1, position=pn.position_nudge(x=-0.2), outlier_size=0.2
        )
        + pn.geom_point(mapping=pn.aes(shape="species_display"), size=3, alpha=0.8)
        + pn.labs(
            title="Model Architecture Impact on Gene Annotation Performance",
            x="Model Architecture",
            y="Transcript F1",
            shape="Species",
        )
        + pn.theme_minimal()
        + pn.theme(axis_text_x=pn.element_text(angle=45, hjust=1))
    )

    # 5. Base model impact: F1 by base_model (original + 2x-genome + viterbi + plantcad+bert), shape by species
    plot5_data = transcript_df[
        (transcript_df["ground_truth"] == "original")
        & (transcript_df["training_data"] == "2x-genome")
        & (transcript_df["decoding_method"] == "viterbi")
        & (transcript_df["model_architecture"] == "plantcad+bert")
    ].copy()

    plot5 = (
        pn.ggplot(
            plot5_data.assign(
                base_model=lambda df: pd.Categorical(
                    df["base_model"],
                    categories=(
                        df.groupby("base_model")["f1_score"]
                        .median()
                        .sort_values()
                        .index
                    ),
                    ordered=True,
                )
            ),
            pn.aes(x="base_model", y="f1_score"),
        )
        + pn.geom_point(mapping=pn.aes(shape="species_display"), size=3, alpha=0.8)
        + pn.geom_boxplot(
            alpha=0.2, width=0.1, position=pn.position_nudge(x=-0.2), outlier_size=0.2
        )
        + pn.labs(
            title="Base Model Impact on Gene Annotation Performance",
            x="Base Model",
            y="Transcript F1",
            shape="Species",
        )
        + pn.theme_minimal()
        + pn.theme(axis_text_x=pn.element_text(angle=45, hjust=1))
    )

    # 6. Ablation results: F1 by ablation configuration, shape by species
    # Create ablation configuration labels by filtering and concatenating subsets
    config1 = df[
        (df["ground_truth"] == "original")
        & (df["decoding_method"] == "direct")
        & (df["model_architecture"] == "random_plantcad+bert")
        & (df["training_data"] == "2x-genome")
        & (df["level"] == "Transcript")
        & (df["tool"] == "gffcompare")
        & (df["post_processing_method"] == "precision-optimized")
        & (df["base_model"] == "PlantCAD2-Small")
    ].assign(ablation_config="Baseline")

    config2 = df[
        (df["ground_truth"] == "original")
        & (df["decoding_method"] == "direct")
        & (df["model_architecture"] == "plantcad+bert")
        & (df["training_data"] == "2x-genome")
        & (df["level"] == "Transcript")
        & (df["tool"] == "gffcompare")
        & (df["post_processing_method"] == "precision-optimized")
        & (df["base_model"] == "PlantCAD2-Small")
    ].assign(ablation_config="+ PlantCAD-Small")

    config3 = df[
        (df["ground_truth"] == "original")
        & (df["decoding_method"] == "direct")
        & (df["model_architecture"] == "plantcad+bert")
        & (df["training_data"] == "2x-genome")
        & (df["level"] == "Transcript")
        & (df["tool"] == "gffcompare")
        & (df["post_processing_method"] == "precision-optimized")
        & (df["base_model"] == "PlantCAD2-Large")
    ].assign(ablation_config="+ PlantCAD-Large")

    config4 = df[
        (df["ground_truth"] == "original")
        & (df["decoding_method"] == "viterbi")
        & (df["model_architecture"] == "plantcad+bert")
        & (df["training_data"] == "2x-genome")
        & (df["level"] == "Transcript")
        & (df["tool"] == "gffcompare")
        & (df["post_processing_method"] == "precision-optimized")
        & (df["base_model"] == "PlantCAD2-Large")
    ].assign(ablation_config="+ CRF")

    config5 = df[
        (df["ground_truth"] == "original")
        & (df["decoding_method"] == "viterbi")
        & (df["model_architecture"] == "plantcad+bert")
        & (df["training_data"] == "5x-genome")
        & (df["level"] == "Transcript")
        & (df["tool"] == "gffcompare")
        & (df["post_processing_method"] == "precision-optimized")
        & (df["base_model"] == "PlantCAD2-Large")
    ].assign(ablation_config="+ 5x train genomes")

    config6 = df[
        (df["ground_truth"] == "pc-filtered")
        & (df["decoding_method"] == "viterbi")
        & (df["model_architecture"] == "plantcad+bert")
        & (df["training_data"] == "5x-genome")
        & (df["level"] == "Transcript")
        & (df["tool"] == "gffcompare")
        & (df["post_processing_method"] == "precision-optimized")
        & (df["base_model"] == "PlantCAD2-Large")
    ].assign(ablation_config="+ 0-shot filter")

    config7 = df[
        (df["ground_truth"] == "pc-filtered")
        & (df["decoding_method"] == "viterbi")
        & (df["model_architecture"] == "plantcad+bert")
        & (df["training_data"] == "5x-genome")
        & (df["level"] == "transcript_cds")
        & (df["tool"] == "gffeval")
        & (df["post_processing_method"] == "precision-optimized")
        & (df["base_model"] == "PlantCAD2-Large")
    ].assign(ablation_config="+ CDS-only eval")

    # Concatenate all configurations
    plot6_data = pd.concat(
        [config1, config2, config3, config4, config5, config6, config7],
        ignore_index=True,
    )

    # Convert to ordered categorical
    ablation_order = [
        "Baseline",
        "+ PlantCAD-Small",
        "+ PlantCAD-Large",
        "+ CRF",
        "+ 5x train genomes",
        "+ 0-shot filter",
        "+ CDS-only eval",
    ]
    plot6_data["ablation_config"] = pd.Categorical(
        plot6_data["ablation_config"], categories=ablation_order, ordered=True
    )

    plot6 = (
        pn.ggplot(plot6_data, pn.aes(x="ablation_config", y="f1_score"))
        + pn.geom_boxplot(
            alpha=0.2, width=0.1, position=pn.position_nudge(x=-0.3), outlier_size=0.2
        )
        + pn.geom_point(
            mapping=pn.aes(shape="species_display"),
            position=pn.position_jitter(width=0.1, height=0, random_state=0),
            size=2,
            alpha=0.8,
        )
        + pn.labs(
            title="Ablation Results",
            x="Configuration",
            y="Transcript F1",
            shape="Species",
        )
        + pn.theme_minimal()
        + pn.theme(axis_text_x=pn.element_text(angle=25, hjust=1))
    )

    # Save individual plots
    plot1.save(output_dir / "genecad_training_scale.pdf", width=8, height=4, dpi=300)
    plot2.save(
        output_dir / "genecad_ground_truth_filter.pdf", width=8, height=4, dpi=300
    )
    plot3.save(output_dir / "genecad_decoding_method.pdf", width=8, height=4, dpi=300)
    plot4.save(
        output_dir / "genecad_model_architecture.pdf", width=8, height=4, dpi=300
    )
    plot5.save(output_dir / "genecad_base_model.pdf", width=8, height=4, dpi=300)
    plot6.save(output_dir / "genecad_ablation_results.pdf", width=8, height=4, dpi=300)

    logger.info(f"Saved 6 plots to: {output_dir}")
    logger.info("  - genecad_training_scale.pdf")
    logger.info("  - genecad_ground_truth_filter.pdf")
    logger.info("  - genecad_decoding_method.pdf")
    logger.info("  - genecad_model_architecture.pdf")
    logger.info("  - genecad_base_model.pdf")
    logger.info("  - genecad_ablation_results.pdf")


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print basic summary and pivot table to console."""

    print("\n" + "=" * 80)
    print("PC QUALITY FILTER EXPERIMENT - RESULTS")
    print("=" * 80)
    print(f"Total records: {len(df)}")
    print(f"Species: {', '.join(sorted(df['species_display'].unique()))}")
    print(f"Versions: {', '.join(sorted(df['major_version'].unique()))}")
    print(f"Ground Truth Types: {', '.join(sorted(df['ground_truth'].unique()))}")
    print(f"Decoding Methods: {', '.join(sorted(df['decoding_method'].unique()))}")

    # Simple pivot showing F1 scores
    print("\n" + "=" * 80)
    print("F1 SCORES BY VERSION AND METHOD COMBINATION")
    print("=" * 80)

    # Create version and ground truth combination for columns
    df = df.copy()
    df["version_ground_truth"] = df["major_version"] + "_" + df["ground_truth"]

    # Use pivot to avoid any aggregation
    pivot = (
        df.pivot(
            index=[
                "decoding_method",
                "species_display",
                "level",
                "tool",
                "base_model",
                "post_processing_method",
            ],
            columns="version_ground_truth",
            values="f1_score",
        )
        .sort_index(ascending=False)
        .round(1)
        .fillna("")
    )

    print(pivot.to_string())


def main():
    """Main analysis function."""
    logger.info("Starting GeneCAD Pipeline Configuration Sweep analysis")

    # Set up output directory for the single plot
    output_dir = Path("local/scratch")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load and process data
        raw_data = load_all_csv_files()
        processed_data = filter_and_process_data(raw_data)

        # Save data to CSV and parquet files as requested
        raw_data_path = output_dir / "all_evaluation_statistics.csv"
        processed_data_path = output_dir / "primary_evaluation_statistics.csv"
        processed_data_parquet_path = (
            output_dir / "primary_evaluation_statistics.parquet"
        )

        raw_data.to_csv(raw_data_path, sep="\t", index=False)
        processed_data.to_csv(processed_data_path, sep="\t", index=False)
        processed_data.to_parquet(processed_data_parquet_path, index=False)

        logger.info(f"Saved raw data to: {raw_data_path}")
        logger.info(f"Saved processed data to: {processed_data_path}")
        logger.info(f"Saved processed data to: {processed_data_parquet_path}")

        # Create visualization
        create_visualization(processed_data, output_dir)

        # Print summary statistics to console
        print_summary_statistics(processed_data)

        logger.info("Analysis complete")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
