import argparse
import logging
import xarray as xr
import pandas as pd
import numpy as np
from src.config import SpeciesConfig
from src.modeling import GeneClassifierConfig
from src.sequence import convert_entity_labels_to_intervals

# Set up logger
logger = logging.getLogger(__name__)


def summarize_mask_rates(dataset: xr.Dataset) -> pd.DataFrame:
    """Summarize masking rates by species and chromosome.

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset containing label_mask, species, and chromosome variables

    Returns
    -------
    pd.DataFrame
        DataFrame with masking rates by species and chromosome
    """
    logger.info("Calculating masking rates by species and chromosome")
    mask_rates = (
        dataset[["label_mask", "species", "chromosome"]]
        .assign(label_mask=lambda ds: ds["label_mask"].mean(dim="sequence"))
        .to_dataframe()
    )
    formatted_rates = (
        mask_rates.assign(
            chrom=lambda df: df["chromosome"].apply(
                SpeciesConfig.parse_chromosome_number
            )
        )
        .groupby(["species", "chrom"])
        .agg({"label_mask": "mean"})
        .unstack()
        .T.pipe(lambda df: df[df.notnull().sum(axis=0).sort_values().index])
        .map(lambda x: f"{x:.2%}" if not pd.isna(x) else "")
    )
    return formatted_rates


def summarize_class_frequencies(dataset: xr.Dataset) -> pd.DataFrame:
    """Summarize class frequencies by species and chromosome.

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset containing feature_labels, species, and chromosome variables

    Returns
    -------
    pd.DataFrame
        DataFrame with class frequencies by species and chromosome
    """
    logger.info("Calculating class frequencies by species and chromosome")

    config = GeneClassifierConfig()
    data = []

    # Get unique species/chromosome combinations
    combos = pd.DataFrame(
        {"species": dataset.species.values, "chromosome": dataset.chromosome.values}
    ).drop_duplicates()

    # Get token to entity name mapping from config
    token_entity_name_map = config.token_entity_name_map()

    # Process each combination
    for _, row in combos.iterrows():
        species, chromosome = row["species"], row["chromosome"]

        # Get indices for this combination
        mask = (dataset.species.values == species) & (
            dataset.chromosome.values == chromosome
        )

        # Get all feature labels for this combination
        feature_labels = dataset.feature_labels.values[mask].flatten()

        # Get frequencies
        unique_labels, counts = np.unique(feature_labels, return_counts=True)
        frequencies = counts / counts.sum()

        # Add to results
        for label, freq in zip(unique_labels, frequencies):
            data.append(
                {
                    "species": species,
                    "chromosome": chromosome,
                    "label": int(label),
                    "label_name": token_entity_name_map[int(label)],
                    "frequency": freq,
                    "chrom": SpeciesConfig.parse_chromosome_number(chromosome),
                }
            )

    if not data:
        return pd.DataFrame()

    # Create and format dataframe
    return (
        pd.DataFrame(data)
        .pivot(
            index=["species", "chrom"],
            columns=["label", "label_name"],
            values="frequency",
        )
        .map(lambda x: f"{x:.2%}")
    )


def get_interval_codons(ds: xr.Dataset) -> pd.DataFrame:
    """Extract codon information from features.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with feature_labels and input_ids variables

    Returns
    -------
    pd.DataFrame
        DataFrame with features and their associated start/stop codons
    """
    config = GeneClassifierConfig()

    # Tokenizer mapping, e.g.: https://huggingface.co/kuleshov-group/PlantCaduceus2-l24-c8192-Phyotozome-v2-proto/blob/main/tokenizer.json
    input_id_map = {3: "A", 4: "C", 5: "G", 6: "T"}

    def get_codon_summary(i: int) -> pd.DataFrame:
        # Use feature_labels directly (no conversion needed)
        labels = ds["feature_labels"].isel(sample=i).values
        assert labels.shape == (ds.sizes["sequence"],)
        mask = ds["label_mask"].isel(sample=i).values

        input_ids = ds["input_ids"].isel(sample=i).values
        assert input_ids.shape == (ds.sizes["sequence"],)
        input_seq = "".join([input_id_map.get(x, "N") for x in input_ids])

        intervals = convert_entity_labels_to_intervals(
            labels=labels,
            class_groups=config.interval_entity_classes,
            mask=mask,
        )
        intervals["sample_index"] = i
        intervals["species"] = ds["species"].isel(sample=i).item()
        intervals["chromosome"] = ds["chromosome"].isel(sample=i).item()
        intervals["entity_type"] = intervals["entity"].apply(
            config.interval_entity_name
        )
        intervals["start_codon"] = [
            None
            if row["entity_type"] != "five_prime_utr"
            else (input_seq[(row["stop"] + 1) : (row["stop"] + 4)])
            for _, row in intervals.iterrows()
        ]
        intervals["stop_codon"] = [
            None
            if row["entity_type"] != "three_prime_utr"
            else (input_seq[(row["start"] - 3) : row["start"]])
            for _, row in intervals.iterrows()
        ]
        return intervals

    intervals = pd.concat([get_codon_summary(i) for i in range(ds.sizes["sample"])])
    return intervals


def summarize_terminal_codons(
    dataset: xr.Dataset, strand: str, sample_size: int = 16_000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize start and stop (terminal) codons across a random sample of sequences.

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset with feature_labels and input_ids variables
    strand : str
        Strand to analyze ("positive" or "negative")
    sample_size : int, optional
        Number of sequences to sample, by default 16000

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of DataFrames with start and stop codon frequency tables
    """
    logger.info(
        f"Analyzing terminal codons for {strand} strand with {sample_size} random samples"
    )

    # Take a random sample of training windows
    rs = np.random.RandomState(42)
    dataset_sample = (
        # Filter to specified strand
        dataset.pipe(lambda ds: ds.sel(sample=ds.strand.values == strand)).pipe(
            lambda ds: ds.isel(
                sample=rs.randint(
                    0, ds.sizes["sample"], size=min(sample_size, ds.sizes["sample"])
                )
            )
        )
    )

    intervals = get_interval_codons(dataset_sample)

    start_codon_table = (
        intervals.groupby(["species", "chromosome", "start_codon"])
        .size()
        .unstack()
        .fillna(0)
        .astype(int)
        .pipe(lambda df: df[df.sum(axis=0).sort_values(ascending=False).index])
    )

    stop_codon_table = (
        intervals.groupby(["species", "chromosome", "stop_codon"])
        .size()
        .unstack()
        .fillna(0)
        .astype(int)
        .pipe(lambda df: df[df.sum(axis=0).sort_values(ascending=False).index])
    )

    return start_codon_table, stop_codon_table


def summarize_sample_counts(
    dataset: xr.Dataset,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Count samples by species and chromosome.

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset containing species and chromosome variables

    Returns
    -------
    tuple[pd.Series, pd.DataFrame, pd.DataFrame]
        Series with counts by species, DataFrame with counts by species+chromosome,
        and DataFrame with counts by species+chromosome+strand
    """
    logger.info("Counting samples by species and chromosome")

    df = dataset[["species", "chromosome", "strand"]].to_dataframe()

    # Count by species
    species_counts = df.groupby("species").size().sort_values(ascending=False)

    # Count by species and chromosome
    chrom_counts = (
        df.assign(
            chrom=lambda x: x["chromosome"].apply(SpeciesConfig.parse_chromosome_number)
        )
        .groupby(["species", "chrom"])
        .size()
        .unstack()
        .fillna(0)
        .astype(int)
    )

    # Count by species, chromosome, and strand
    strand_counts = (
        df.assign(
            chrom=lambda x: x["chromosome"].apply(SpeciesConfig.parse_chromosome_number)
        )
        .groupby(["species", "chrom", "strand"])
        .size()
        .unstack("strand")
        .fillna(0)
        .astype(int)
    )

    return species_counts, chrom_counts, strand_counts


def summarize_dataset(dataset_path: str, limit: int | None = None) -> None:
    """Summarize key statistics for a dataset.

    Parameters
    ----------
    dataset_path : str
        Path to dataset in Zarr format
    limit : int | None, optional
        Maximum number of samples to process, by default None
    """
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = xr.open_zarr(dataset_path)

    if limit is not None:
        logger.info(f"Limiting dataset to {limit} samples")
        dataset = dataset.isel(sample=slice(0, limit))

    logger.info(f"Dataset loaded with dimensions: {dict(dataset.sizes)}")

    # Print out the dataset
    logger.info(f"\n{dataset}")

    # ------------------------------------------------------------
    # Sample counts
    # ------------------------------------------------------------
    logger.info("Summarizing sample counts")
    species_counts, chrom_counts, strand_counts = summarize_sample_counts(dataset)
    logger.info("Samples by species:")
    logger.info("\n" + str(species_counts))
    logger.info("Samples by species and chromosome:")
    logger.info("\n" + str(chrom_counts))
    logger.info("Samples by species, chromosome, and strand:")
    logger.info("\n" + str(strand_counts))

    # ------------------------------------------------------------
    # Masking rates
    # ------------------------------------------------------------
    logger.info("Summarizing masking rates")
    mask_rates = summarize_mask_rates(dataset)
    logger.info("Masking rates by species and chromosome:")
    logger.info("\n" + str(mask_rates))

    # ------------------------------------------------------------
    # Class frequencies
    # ------------------------------------------------------------
    logger.info("Summarizing class frequencies")
    class_freqs = summarize_class_frequencies(dataset)
    logger.info("Class frequencies by species and chromosome:")
    logger.info("\n" + str(class_freqs))

    # ------------------------------------------------------------
    # Terminal codons
    # ------------------------------------------------------------
    logger.info("Summarizing terminal codons")

    # Analyze both strands separately
    for strand in ["positive", "negative"]:
        logger.info(f"Analyzing {strand} strand terminal codons")
        start_codon_table, stop_codon_table = summarize_terminal_codons(
            dataset, strand=strand
        )

        logger.info(f"{strand.capitalize()} strand start codon frequency:")
        logger.info("\n" + str(start_codon_table))

        logger.info(f"{strand.capitalize()} strand stop codon frequency:")
        logger.info("\n" + str(stop_codon_table))

    logger.info("Summary complete")


def main() -> None:
    """Parse command line arguments and execute appropriate function."""
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.expand_frame_repr", False)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Dataset Summary and Analysis Tools")
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Command to execute"
    )

    # Summarize training dataset command
    summarize_parser = subparsers.add_parser(
        "summarize_training_dataset",
        help="Summarize key statistics for a training dataset",
    )
    summarize_parser.add_argument(
        "--input",
        required=True,
        help="Path to input dataset in Zarr format (train.zarr or valid.zarr typically)",
    )
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (for debugging)",
    )

    args = parser.parse_args()

    if args.command == "summarize_training_dataset":
        summarize_dataset(args.input, args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
