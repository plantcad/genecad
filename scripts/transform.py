import pandas as pd
import argparse
import os
import logging
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor
from typing import cast as cast_type
from src.dataset import (
    DEFAULT_SEQUENCE_CHUNK_SIZE,
    open_datatree,
    list_species_contig_datatree,
    set_dimension_chunks,
    info_str,
)
import numpy as np
from src.schema import FeatureLevel, FilterReason, RegionType, GffFeatureType
from src.sampling import extract_label_dataset
from src.sequence import (
    find_overlapping_intervals,
    convert_entity_intervals_to_labels,
    expand_sequence_slice,
)
import xarray as xr

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Filter features
# -------------------------------------------------------------------------------------------------


@dataclass
class Filter:
    species_id: str
    chromosome_id: str
    gene_id: str
    strand: str
    reason: str


def create_filters(df: pd.DataFrame, reason: str) -> list[Filter]:
    """Create Filter objects resulting from a shared cause."""
    return [
        Filter(
            species_id=cast_type(str, row["species_id"]),
            chromosome_id=cast_type(str, row["chromosome_id"]),
            gene_id=cast_type(str, row["gene_id"]),
            strand=cast_type(str, row["strand"]),
            reason=reason,
        )
        for _, row in (
            df.drop_duplicates(
                subset=["species_id", "chromosome_id", "gene_id", "strand"]
            ).iterrows()
        )
    ]


def apply_filters(df: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
    """Filter a dataframe by removing rows corresponding to genes in the filters list."""
    if not filters:
        return df

    # Extract unique (species_id, chromosome_id, gene_id, strand) tuples from filters
    genes_to_filter = set(
        (f.species_id, f.chromosome_id, f.gene_id, f.strand) for f in filters
    )

    # Filter out the genes
    return df[
        ~df[["species_id", "chromosome_id", "gene_id", "strand"]]
        .apply(tuple, axis=1)
        .isin(genes_to_filter)
    ]


def filter_canonical_transcripts(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Filter]]:
    """Filter the dataframe to only keep canonical transcripts."""

    # Count canonical transcripts per gene
    canonical_counts = (
        # First, get unique transcripts (we have to only count each transcript once)
        df[
            [
                "species_id",
                "chromosome_id",
                "gene_id",
                "strand",
                "transcript_id",
                "transcript_is_canonical",
            ]
        ]
        .drop_duplicates()
        .groupby(["species_id", "chromosome_id", "gene_id", "strand"])[
            "transcript_is_canonical"
        ]
        .sum()
        .reset_index()
    )

    # Find genes with no canonical transcripts
    genes_with_no_canonical = canonical_counts[
        canonical_counts["transcript_is_canonical"] == 0
    ]
    logger.info(
        f"Found {len(genes_with_no_canonical)} genes with no canonical transcripts"
    )

    # Find genes with multiple canonical transcripts
    genes_with_multiple_canonical = canonical_counts[
        canonical_counts["transcript_is_canonical"] > 1
    ]
    logger.info(
        f"Found {len(genes_with_multiple_canonical)} genes with multiple canonical transcripts"
    )

    # Create filters for both cases
    filters = create_filters(
        genes_with_no_canonical, FilterReason.NO_CANONICAL_TRANSCRIPT.value
    ) + create_filters(
        genes_with_multiple_canonical, FilterReason.MULTIPLE_CANONICAL_TRANSCRIPTS.value
    )

    # Filter out filtered genes and keep only canonical transcripts
    df_filtered = apply_filters(df, filters)

    # Further filter to only keep canonical transcripts
    df_filtered = df_filtered[df_filtered["transcript_is_canonical"]]

    logger.info("Canonical transcript filters:")
    logger.info(f"  Original dataframe had {len(df)} rows")
    logger.info(f"  Filtered dataframe has {len(df_filtered)} rows")
    logger.info(f"  Removed {len(filters)} genes due to canonical transcript issues")

    return df_filtered, filters


def filter_incomplete_features(
    df: pd.DataFrame, remove_incomplete_features: bool = True
) -> tuple[pd.DataFrame, list[Filter]]:
    """Filter the dataframe to only keep genes with fully annotated canonical transcripts."""
    # Get list of "base" features, meaning those at the lowest level (i.e. UTRs and CDS)
    base_features = GffFeatureType.get_values(level=2)
    genes_with_incomplete_features = (
        df
        # It is crucial to do this first as significantly more
        # non-canonical transcripts are not fully annotated
        .pipe(lambda df: df[df["transcript_is_canonical"]])
        # Filter to only chosen feature types
        .pipe(lambda df: df[df["feature_type"].isin(base_features)])
        # Count distinct feature types per gene (including strand)
        .groupby(["species_id", "chromosome_id", "gene_id", "strand"])["feature_type"]
        .nunique()
        .reset_index()
        .pipe(lambda df: df[df["feature_type"] < len(base_features)])
    )

    logger.info(
        f"Found {len(genes_with_incomplete_features)} genes with incomplete canonical transcript annotations"
    )

    # Create and apply filters
    filters = create_filters(
        genes_with_incomplete_features, reason=FilterReason.INCOMPLETE_FEATURES.value
    )
    # If apply_filters is True, filter the underlying features;
    # otherwise, just return the filters which can be used later to define masks
    if remove_incomplete_features:
        df_filtered = apply_filters(df, filters)
    else:
        df_filtered = df

    logger.info("Incomplete feature filters:")
    logger.info(f"  Original dataframe had {len(df)} rows")
    logger.info(f"  Filtered dataframe has {len(df_filtered)} rows")
    logger.info(f"  Removed {len(filters)} genes due to incomplete feature annotations")
    return df_filtered, filters


def find_overlapping_genes(df: pd.DataFrame) -> list[Filter]:
    """Identify genes that overlap with other genes on the same chromosome."""
    filters = []

    # Get unique genes with their positions
    genes = df[
        ["species_id", "chromosome_id", "gene_id", "strand", "gene_start", "gene_stop"]
    ].drop_duplicates()
    total_genes = _gene_count(df)

    # Group by species, chromosome, and strand
    for (species_id, chrom, strand), group in genes.groupby(
        ["species_id", "chromosome_id", "strand"]
    ):
        # Find overlapping genes within this group
        has_overlap = find_overlapping_intervals(
            group["gene_start"], group["gene_stop"]
        )

        # Get DataFrame of overlapping genes
        overlapping_genes_df = group.loc[has_overlap]

        # Add to filters using utility function
        filters.extend(
            create_filters(overlapping_genes_df, FilterReason.OVERLAPPING_GENE.value)
        )

        logger.info(
            f"Found {len(overlapping_genes_df)} overlapping genes in {species_id}, chromosome {chrom}, strand {strand}"
        )

    logger.info(
        f"Found {len(filters)} genes with overlaps (out of {total_genes} total genes)"
    )
    return filters


def filter_overlapping_genes(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Filter]]:
    """Filter the dataframe to only keep non-overlapping genes."""
    filters = find_overlapping_genes(df)
    df_filtered = apply_filters(df, filters)

    logger.info("Overlapping gene filters:")
    logger.info(f"  Original dataframe had {len(df)} rows")
    logger.info(f"  Filtered dataframe has {len(df_filtered)} rows")
    logger.info(f"  Removed {len(filters)} genes due to overlapping genes")
    return df_filtered, filters


def find_overlapping_features(df: pd.DataFrame) -> list[Filter]:
    """Identify genes with overlapping features."""
    filters = []

    # Count total genes
    total_genes = _gene_count(df)

    # Group by species, chromosome, gene, strand, and transcript
    # (there should be only one transcript per gene after filter_canonical_transcripts)
    for (species_id, chrom, gene_id, strand, transcript_id), features_df in df.groupby(
        ["species_id", "chromosome_id", "gene_id", "strand", "transcript_id"]
    ):
        # Check for overlapping features
        has_overlap = find_overlapping_intervals(
            features_df["feature_start"], features_df["feature_stop"]
        )

        # If any features overlap, add the gene to filters
        if has_overlap.any():
            # Create a DataFrame with a single row for this gene
            gene_df = pd.DataFrame(
                [
                    {
                        "species_id": species_id,
                        "chromosome_id": chrom,
                        "gene_id": gene_id,
                        "strand": strand,
                    }
                ]
            )

            # Add to filters using utility function
            filters.extend(
                create_filters(gene_df, FilterReason.OVERLAPPING_FEATURES.value)
            )

    logger.info(
        f"Found {len(filters)} genes with overlapping features (out of {total_genes} total genes)"
    )
    return filters


def filter_overlapping_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Filter]]:
    """Filter the dataframe to only keep genes with non-overlapping features."""
    filters = find_overlapping_features(df)
    df_filtered = apply_filters(df, filters)

    logger.info("Overlapping feature filters:")
    logger.info(f"  Original dataframe had {len(df)} rows")
    logger.info(f"  Filtered dataframe has {len(df_filtered)} rows")
    logger.info(f"  Removed {len(filters)} genes due to overlapping features")
    return df_filtered, filters


def _gene_count(df: pd.DataFrame) -> int:
    """Count the number of genes in the dataframe."""
    return (
        df[["species_id", "chromosome_id", "gene_id", "strand"]]
        .drop_duplicates()
        .pipe(len)
    )


def filter_features(
    input_path: str,
    features_output_path: str,
    filters_output_path: str,
    remove_incomplete_features: bool = True,
) -> None:
    """Process GFF features by filtering to canonical transcripts and removing overlapping features.

    Parameters
    ----------
    input_path : str
        Path to input parquet file
    features_output_path : str
        Path to output parquet file for filtered features
    filters_output_path : str
        Path to output parquet file for filter definitions
    remove_incomplete_features : bool, optional
        Whether to filter out genes with incomplete feature annotations, by default True;
        note that the filter definitions will be saved regardless of whether or not the filters
        are applied to the underlying features
    """
    # Read input dataframe
    logger.info(f"Reading input from {input_path}")
    features = pd.read_parquet(input_path)

    # Create unified strand column from gene_strand
    features = features.assign(
        strand=lambda df: df["gene_strand"].map({-1: "negative", 1: "positive"})
    )
    if features["strand"].isna().any():
        invalid_strands = features[features["strand"].isna()]["gene_strand"].unique()
        raise ValueError(f"Found unmapped strand values: {invalid_strands}")

    # Sort the dataframe
    features = features.sort_values(
        [
            "species_id",
            "chromosome_id",
            "gene_start",
            "transcript_start",
            "feature_start",
        ]
    )
    total_rows = len(features)
    total_genes = _gene_count(features)

    # Filter to canonical transcripts
    logger.info("Filtering to canonical transcripts")
    filtered_features, canonical_filters = filter_canonical_transcripts(features)

    # Find overlapping genes
    logger.info("Filtering to non-overlapping genes")
    filtered_features, gene_overlap_filters = filter_overlapping_genes(
        filtered_features
    )

    # Filter to transcripts with complete features (conditional)
    logger.info(
        f"Filtering to transcripts with complete features (remove_incomplete_features={remove_incomplete_features})"
    )
    filtered_features, incomplete_features_filters = filter_incomplete_features(
        filtered_features, remove_incomplete_features=remove_incomplete_features
    )

    # Find genes with overlapping features
    logger.info("Filtering to non-overlapping features")
    filtered_features, feature_overlap_filters = filter_overlapping_features(
        filtered_features
    )

    # Create filters DataFrame for statistics
    filters = pd.DataFrame(
        [
            asdict(f)
            for f in (
                canonical_filters
                + gene_overlap_filters
                + incomplete_features_filters
                + feature_overlap_filters
            )
        ]
    )
    filtered_genes = total_genes - _gene_count(filtered_features)

    # Log final statistics
    logger.info("Final filter summary:")
    logger.info(f"  Original dataframe had {total_rows} rows")
    logger.info(f"  Filtered dataframe has {len(filtered_features)} rows")
    logger.info(
        f"  Of {total_genes} total genes, {filtered_genes} were removed ({filtered_genes / total_genes * 100:.2f}%)"
    )

    # Log filters reason statistics
    if not filters.empty:
        reason_counts = filters["reason"].value_counts()
        logger.info("Filter reason counts:")
        for reason, count in reason_counts.items():
            logger.info(f"  - {reason}: {count}")

    # Get spans associated with filtered features (for downstream masking)
    n_filters = len(filters)
    filters = filters.merge(
        features[
            [
                "species_id",
                "chromosome_id",
                "gene_id",
                "strand",
                "gene_start",
                "gene_stop",
            ]
        ].drop_duplicates(subset=["species_id", "chromosome_id", "gene_id", "strand"]),
        on=["species_id", "chromosome_id", "gene_id", "strand"],
        how="inner",
        validate="m:1",
    )
    assert len(filters) == n_filters

    # Save results
    logger.info(f"Filtered features:\n{filtered_features.head()}")
    logger.info(f"Filtered features info:\n{filtered_features.pipe(info_str)}")
    logger.info(f"Saving filtered features to {features_output_path}")
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(features_output_path), exist_ok=True)
    filtered_features.to_parquet(features_output_path, index=False)

    logger.info(f"Filter definitions:\n{filters.head()}")
    logger.info(f"Filter definitions info:\n{filters.pipe(info_str)}")
    logger.info(f"Saving filter definitions to {filters_output_path}")
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(filters_output_path), exist_ok=True)
    filters.to_parquet(filters_output_path, index=False)

    logger.info("Done")


# -------------------------------------------------------------------------------------------------
# Stack features
# -------------------------------------------------------------------------------------------------


def stack_gff_features(df: pd.DataFrame) -> pd.DataFrame:
    dfs = []
    base_key = ["species_id", "chromosome_id", "strand"]
    base_cols = ["filename", "chromosome_length"]
    for prefix, type, key, level in [
        ("gene", GffFeatureType.GENE, ["gene_id"], FeatureLevel.GENE),
        (
            "transcript",
            GffFeatureType.MRNA,
            ["gene_id", "transcript_id"],
            FeatureLevel.TRANSCRIPT,
        ),
        (
            "feature",
            None,
            ["gene_id", "transcript_id", "feature_id"],
            FeatureLevel.ANNOTATION,
        ),
    ]:
        cols = {f"{prefix}_{c}": c for c in ["id", "name", "start", "stop"]}
        if type is None:
            cols["feature_type"] = "feature_type"
        group = (
            df
            # Assign path tuple defining parent ids
            .assign(path=lambda df: df[base_key + key[:-1]].apply(tuple, axis=1))
            .drop_duplicates(subset=base_key + key)[
                base_cols + base_key + list(cols) + ["path"]
            ]
            .rename(columns=cols)
            .assign(feature_level=int(level))
        )
        if type is not None:
            group = group.assign(feature_type=type)
        dfs.append(group)
    result = pd.concat(dfs, axis=0, ignore_index=True)
    assert result.drop(columns="name").notnull().all().all()
    return result


def stack_features(input_path: str, output_path: str) -> None:
    """Convert features from wide to stacked format.

    Parameters
    ----------
    input_path : str
        Path to input parquet file with filtered features
    output_path : str
        Path to output parquet file for stacked features
    """
    logger.info(f"Loading features from {input_path}")
    features = pd.read_parquet(input_path)

    assert features["strand"].isin(["negative", "positive"]).all()

    logger.info("Stacking features")
    features = stack_gff_features(features)
    logger.info(f"Stacked features:\n{features.head()}")
    logger.info(f"Stacked features info:\n{features.pipe(info_str)}")

    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"Saving stacked features to {output_path}")
    features.to_parquet(output_path, index=False)
    logger.info("Done")


# -------------------------------------------------------------------------------------------------
# Create labels
# -------------------------------------------------------------------------------------------------


def _create_feature_intervals(group: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # Filter to lowest level features like CDS, 5' UTR, 3' UTR, etc.
    annotation_features = [
        ft
        # pyrefly: ignore  # not-iterable
        for ft in GffFeatureType
        if GffFeatureType.value_to_level()[ft] == FeatureLevel.ANNOTATION
    ]
    group = group[group["feature_type"].isin(annotation_features)]

    feature_type_to_index = {ft: i for i, ft in enumerate(annotation_features)}
    feature_coords = [GffFeatureType.value_to_slug()[ft] for ft in annotation_features]

    # Map feature types to 1-based label indices
    labels = group["feature_type"].map(feature_type_to_index) + 1
    if labels.isna().any():
        invalid_feature_types = group[labels.isna()]["feature_type"].unique()
        raise ValueError(f"Found unmapped feature types: {invalid_feature_types}")

    # Prepare intervals for conversion to arrays
    intervals = group.assign(label=labels).rename(columns={"feature_level": "level"})[
        ["start", "stop", "label", "level"]
    ]
    return intervals, feature_coords


def _create_feature_labels(
    intervals: pd.DataFrame, feature_coords: list[str], domain: tuple[int, int]
) -> tuple[np.ndarray, list[str]]:
    domain_size = domain[1] - domain[0]
    num_feature_types = len(feature_coords)

    # Generate labels at each level with errors raised for overlapping intervals
    feature_labels = convert_entity_intervals_to_labels(
        intervals, domain, num_labels=num_feature_types, on_overlap="raise"
    )
    assert feature_labels.shape == (domain_size, num_feature_types)
    return feature_labels, feature_coords


def _create_region_labels(
    group: pd.DataFrame, domain: tuple[int, int]
) -> tuple[np.ndarray, list[str]]:
    domain_size = domain[1] - domain[0]
    # pyrefly: ignore  # no-matching-overload
    region_coords: list[str] = list(RegionType)
    num_region_types = len(region_coords)

    # Helper function to create region dataframes from specific feature types
    def create_region_df(
        feature_types: list[GffFeatureType], label_index: int
    ) -> pd.DataFrame:
        return group.pipe(lambda df: df[df["feature_type"].isin(feature_types)])[
            ["path", "start", "stop"]
        ].assign(label=label_index)

    # Helper function to create aggregated region dataframes
    def create_aggregated_region_df(
        feature_types: list[GffFeatureType], label_index: int
    ) -> pd.DataFrame:
        return (
            group.pipe(lambda df: df[df["feature_type"].isin(feature_types)])
            .groupby("path")
            .agg(
                start=("start", "min"),
                stop=("stop", "max"),
            )
            .reset_index()
            .assign(label=label_index)
        )

    # Create dataframes containing intervals for each region type
    region_interval_sets = [
        # Regions defined directly by features
        create_region_df([GffFeatureType.GENE], RegionType.GENE.get_index()),
        create_region_df([GffFeatureType.MRNA], RegionType.TRANSCRIPT.get_index()),
        create_region_df(
            [
                GffFeatureType.CDS,
                GffFeatureType.FIVE_PRIME_UTR,
                GffFeatureType.THREE_PRIME_UTR,
            ],
            RegionType.EXON.get_index(),
        ),
        # Regions defined by min/max spans of consituent features
        create_aggregated_region_df([GffFeatureType.CDS], RegionType.CDS.get_index()),
        create_aggregated_region_df(
            [GffFeatureType.FIVE_PRIME_UTR], RegionType.FIVE_PRIME_UTR.get_index()
        ),
        create_aggregated_region_df(
            [GffFeatureType.THREE_PRIME_UTR], RegionType.THREE_PRIME_UTR.get_index()
        ),
        # This is not the definition of introns, but rather a region confining where they should exist;
        # they will be fully defined later by subtracting regions from this initial window
        create_aggregated_region_df(
            [
                GffFeatureType.CDS,
                GffFeatureType.FIVE_PRIME_UTR,
                GffFeatureType.THREE_PRIME_UTR,
            ],
            RegionType.INTRON.get_index(),
        ),
    ]

    # Combine all region dataframes
    region_interval_sets = pd.concat(region_interval_sets, axis=0, ignore_index=True)

    # Convert to required format for convert_entity_intervals_to_labels
    region_intervals = region_interval_sets.rename(columns={"path": "path_col"})[
        ["start", "stop", "label"]
    ]

    # Generate region labels
    region_labels = convert_entity_intervals_to_labels(
        # Convert all enum indices to 1-based indices for interval conversion
        region_intervals.assign(label=lambda df: df["label"] + 1),
        domain,
        num_labels=num_region_types,
        # Overlapping intervals are expected in this case, e.g.
        # genes and their transcripts
        on_overlap="ignore",
    )
    assert region_labels.shape == (domain_size, num_region_types)

    # Compute introns by subtracting exons from the window defined by any existing
    # transcript features, not just the bounds of the transcript itself
    exon_label = region_labels[:, RegionType.EXON.get_index()]
    intron_mask = region_labels[:, RegionType.INTRON.get_index()]
    intron_label = intron_mask.astype(bool) & ~exon_label.astype(bool)
    region_labels[:, RegionType.INTRON.get_index()] = intron_label.astype(np.int8)

    return region_labels, region_coords


def _create_boundary_intervals(
    filters_df: pd.DataFrame,
    intervals: pd.DataFrame,
    domain: tuple[int, int],
    boundary_mask_size: tuple[int, int],
) -> pd.DataFrame:
    """Create boundary intervals for a set of filters."""
    if filters_df.empty:
        return pd.DataFrame(columns=["start", "stop", "reason"])

    up_buffer, down_buffer = boundary_mask_size

    # Sort intervals for fast lookups
    interval_starts = np.sort(intervals["start"].values)
    interval_stops = np.sort(intervals["stop"].values)

    boundary_data = []
    for _, row in filters_df.iterrows():
        gene_start, gene_stop, reason = (
            cast_type(int, row["start"]),
            cast_type(int, row["stop"]),
            cast_type(str, row["reason"]),
        )

        # Upstream mask: constrain by end of nearest upstream feature
        upstream_start = gene_start - up_buffer
        idx = np.searchsorted(interval_stops, gene_start, side="left") - 1
        if idx >= 0:
            upstream_start = max(upstream_start, interval_stops[idx])
        upstream_start = max(domain[0], upstream_start)
        upstream_stop = min(domain[1], gene_start)

        # Downstream mask: constrain by start of nearest downstream feature
        downstream_stop = gene_stop + down_buffer
        idx = np.searchsorted(interval_starts, gene_stop, side="right")
        if idx < len(interval_starts):
            downstream_stop = min(downstream_stop, interval_starts[idx])
        downstream_start = max(domain[0], gene_stop)
        downstream_stop = min(domain[1], downstream_stop)

        # Add valid intervals
        if upstream_start <= upstream_stop:
            boundary_data.append(
                {"start": upstream_start, "stop": upstream_stop, "reason": reason}
            )
        if downstream_start <= downstream_stop:
            boundary_data.append(
                {"start": downstream_start, "stop": downstream_stop, "reason": reason}
            )

    return (
        pd.DataFrame(boundary_data)
        if boundary_data
        else pd.DataFrame(columns=["start", "stop", "reason"])
    )


def _generate_label_masks(
    intervals: pd.DataFrame,
    group_filters: pd.DataFrame,
    domain: tuple,
    strand: str,
    remove_incomplete_features: bool = True,
    boundary_mask_size: tuple[int, int] = (1, 1),
) -> tuple[np.ndarray, list[str]]:
    if strand not in ["positive", "negative"]:
        raise ValueError(f"Strand must be 'positive' or 'negative', got {strand}")
    if any(size < 1 for size in boundary_mask_size):
        raise ValueError(
            f"Boundary mask sizes must be at least 1, got {boundary_mask_size}"
        )

    # Assign buffers based on strand (upstream becomes downstream on negative strand)
    boundary_mask_size = (
        boundary_mask_size
        if strand == "positive"
        else cast_type(tuple[int, int], boundary_mask_size[::-1])
    )

    # Standardize group_filters: rename columns and deduplicate early
    standardized_filters = (
        group_filters[["gene_start", "gene_stop", "reason"]]
        .drop_duplicates()
        .rename(columns={"gene_start": "start", "gene_stop": "stop"})
    )

    # Configure filter reason handling based on remove_incomplete_features flag
    if remove_incomplete_features:
        # When removing incomplete features, they get both interior and boundary masks
        boundary_filter_reasons = []
        both_filter_reasons = [FilterReason.INCOMPLETE_FEATURES.value]
    else:
        # When not removing incomplete features, they get boundary masks only
        boundary_filter_reasons = [FilterReason.INCOMPLETE_FEATURES.value]
        both_filter_reasons = []

    # Split filters into three categories: boundary-only, both, and interior-only
    is_boundary_only = standardized_filters["reason"].isin(boundary_filter_reasons)
    is_both = standardized_filters["reason"].isin(both_filter_reasons)
    # pyrefly: ignore  # unsupported-operation
    is_interior_only = ~(is_boundary_only | is_both)

    boundary_only_filters = standardized_filters[is_boundary_only]
    both_filters = standardized_filters[is_both]
    interior_only_filters = standardized_filters[is_interior_only]

    intervals_list = []

    # Process interior-only filters
    if not interior_only_filters.empty:
        logger.info(
            f"Adding {len(interior_only_filters)} interior-only filters (strand: {strand}, domain: {domain})"
        )
        intervals_list.append(interior_only_filters)

    # Process both filters - add both interior and boundary intervals
    if not both_filters.empty:
        logger.info(
            f"Adding {len(both_filters)} both (interior + boundary) filters (strand: {strand}, domain: {domain})"
        )
        # Add interior intervals
        intervals_list.append(both_filters)

        # Add boundary intervals using helper function
        both_boundary_intervals = _create_boundary_intervals(
            both_filters, intervals, domain, boundary_mask_size
        )
        if not both_boundary_intervals.empty:
            intervals_list.append(both_boundary_intervals)

    # Process boundary-only filters
    if not boundary_only_filters.empty:
        logger.info(
            f"Adding {len(boundary_only_filters)} boundary-only filters (strand: {strand}, domain: {domain}, buffer: {boundary_mask_size})"
        )
        boundary_intervals = _create_boundary_intervals(
            boundary_only_filters, intervals, domain, boundary_mask_size
        )
        if not boundary_intervals.empty:
            intervals_list.append(boundary_intervals)

    # Handle empty case
    if not intervals_list:
        reasons = []
        label_masks = np.ones((domain[1] - domain[0], len(reasons)), dtype=np.int8)
        return label_masks, reasons

    # Combine all intervals and generate masks
    combined_intervals = pd.concat(intervals_list, ignore_index=True)
    # Drop duplicates that may result from downstream mask of one gene
    # perfectly overlapping the upstream mask of another gene
    combined_intervals = combined_intervals.drop_duplicates()
    reasons = combined_intervals["reason"].drop_duplicates().sort_values().to_list()
    reason_to_index = {reason: i for i, reason in enumerate(reasons)}

    label_masks = convert_entity_intervals_to_labels(
        combined_intervals.assign(
            label=lambda df: df["reason"].map(reason_to_index) + 1
        ),
        domain,
        num_labels=len(reasons),
        on_overlap="ignore",
    )

    # Labels indicate where filters are active, so invert this
    # representation to preserve positive mask convention
    label_masks = (~label_masks.astype(bool)).astype(label_masks.dtype)
    return label_masks, reasons


def create_labels(
    features_path: str,
    filters_path: str,
    output_path: str,
    chunk_size: int = DEFAULT_SEQUENCE_CHUNK_SIZE,
    remove_incomplete_features: bool = True,
    boundary_mask_size: tuple[int, int] = (1, 1),
) -> None:
    """Create labels from filtered features.

    Parameters
    ----------
    features_path : str
        Path to stacked features parquet file
    filters_path : str
        Path to filters parquet file
    output_path : str
        Path to output for Xarray dataset
    chunk_size : int, optional
        Size of chunks to write to Zarr
    remove_incomplete_features : bool, optional
        Whether incomplete features result in interior masks (yes) or upstream/downstream boundary masks (no)
    boundary_mask_size : tuple[int, int], optional
        Upstream and downstream mask sizes for boundary filters
    """
    # Load filtered features and filter definitions
    logger.info(f"Loading stacked features from {features_path}")
    features = pd.read_parquet(features_path)
    # Serialized `path` tuples in stacked features load as numpy arrays
    # and must be converted back to tuples
    assert features["path"].notnull().all()
    features["path"] = features["path"].apply(tuple)
    assert features["path"].apply(lambda x: isinstance(x, tuple)).all()
    logger.info(f"Loaded {len(features)} features:\n{features.head()}")
    logger.info(f"Stacked features info:\n{features.pipe(info_str)}")

    logger.info(f"Loading filter definitions from {filters_path}")
    filters = pd.read_parquet(filters_path)
    logger.info(f"Loaded {len(filters)} filters:\n{filters.head()}")
    logger.info(f"Filter definitions info:\n{filters.pipe(info_str)}")

    assert features["strand"].isin(["negative", "positive"]).all()
    assert filters["strand"].isin(["negative", "positive"]).all()

    # Group by species and chromosome
    chrom_groups = features.groupby(["species_id", "chromosome_id"])
    total_chroms = len(chrom_groups)
    for chrom_idx, ((species_id, chrom_id), chrom) in enumerate(chrom_groups, start=1):
        # Get chromosome information
        chrom_length = chrom["chromosome_length"].iloc[0]
        domain = (0, int(chrom_length))
        filename = chrom["filename"].iloc[0]

        # Process each strand separately
        data = []
        for strand, group in chrom.groupby("strand"):
            logger.info(
                f"Processing {species_id!r}, chromosome {chrom_id!r}, strand {strand!r} ({chrom_idx}/{total_chroms})"
            )

            # Extract feature intervals and names (for coordinate labels)
            feature_intervals, feature_coords = _create_feature_intervals(group)

            # Generate feature labels
            feature_labels, feature_coords = _create_feature_labels(
                feature_intervals, feature_coords, domain
            )
            assert np.isin(feature_labels, [0, 1]).all()

            region_labels, region_coords = _create_region_labels(group, domain)
            assert np.isin(region_labels, [0, 1]).all()

            # Generate sequence masks from filters
            group_filters = filters[
                (filters["species_id"] == species_id)
                & (filters["chromosome_id"] == chrom_id)
                & (filters["strand"] == strand)
            ]
            label_masks, reasons = _generate_label_masks(
                intervals=feature_intervals,
                group_filters=group_filters,
                domain=domain,
                strand=strand,
                remove_incomplete_features=remove_incomplete_features,
                boundary_mask_size=boundary_mask_size,
            )

            # Create Xarray dataset
            ds = xr.Dataset(
                data_vars={
                    "feature_labels": (["sequence", "feature"], feature_labels),
                    "region_labels": (["sequence", "region"], region_labels),
                    "label_masks": (["sequence", "reason"], label_masks),
                },
                coords={
                    "sequence": np.arange(chrom_length),
                    "feature": feature_coords,
                    "region": region_coords,
                    "reason": reasons,
                },
            )
            data.append(ds.expand_dims("strand").assign_coords(strand=[strand]))
        ds = xr.concat(data, dim="strand").assign_attrs(
            species_id=species_id,
            chromosome_id=chrom_id,
            filename=filename,
        )
        ds = set_dimension_chunks(ds, "sequence", chunk_size)
        logger.info(f"Saving data for group '{species_id}/{chrom_id}' to {output_path}")
        ds.to_zarr(
            output_path,
            group=f"{species_id}/{chrom_id}",
            zarr_format=2,
            consolidated=True,
            mode="w",
        )

    dt = open_datatree(output_path)
    logger.info(f"Final data tree:\n{dt}")
    logger.info("Done")


# -------------------------------------------------------------------------------------------------
# Create sequence dataset
# -------------------------------------------------------------------------------------------------


def _create_sequence_dataset_group(task):
    """Worker function to process a single group in parallel.

    Parameters
    ----------
    task : tuple
        Tuple containing (group_path, species_id, chrom_id, input_labels_path, input_tokens_path, output_path, chunk_size, task_id, total_tasks)

    Returns
    -------
    dict
        Processing statistics for this group
    """
    (
        group_path,
        species_id,
        chrom_id,
        input_labels_path,
        input_tokens_path,
        output_path,
        chunk_size,
        task_id,
        total_tasks,
    ) = task
    prefix = f"[{species_id}/{chrom_id}]"
    logger.info(f"{prefix} Starting processing (group {task_id}/{total_tasks})")

    # Load datasets for this group on demand
    logger.info(f"{prefix} Loading label dataset")
    label_ds = xr.open_zarr(input_labels_path, group=group_path)

    logger.info(f"{prefix} Loading token dataset")
    token_ds = xr.open_zarr(input_tokens_path, group=group_path)

    # Ensure that sequence tokens always span a longer range than labels
    if token_ds.sizes["sequence"] < label_ds.sizes["sequence"]:
        raise ValueError(
            f"Group {group_path}: More labels ({label_ds.sizes['sequence']}) than tokens ({token_ds.sizes['sequence']})"
        )

    # Process in chunks for memory efficiency
    output_group = f"{species_id}/{chrom_id}"
    seq_size = label_ds.sizes["sequence"]
    logger.info(
        f"{prefix} Processing {seq_size:,} base pairs in chunks of {chunk_size:,}"
    )

    for chunk_idx, start_idx in enumerate(range(0, seq_size, chunk_size)):
        end_idx = min(start_idx + chunk_size, seq_size)
        logger.info(f"{prefix} Processing chunk {start_idx:,} to {end_idx:,}")

        # Extract chunk from both datasets
        raw_label_chunk = label_ds.isel(sequence=slice(start_idx, end_idx))
        token_chunk = token_ds.isel(sequence=slice(start_idx, end_idx))

        # Extract labels with padding for BILUO context, then trim back
        window_slice, trim_slice = expand_sequence_slice(
            start_idx, end_idx, margin=1, sequence_length=seq_size
        )
        padded_labels = extract_label_dataset(
            label_ds.isel(sequence=slice(*window_slice))
        )
        label_chunk = padded_labels.isel(sequence=slice(*trim_slice))

        # Ensure trimmed chunk has same size as original window
        assert label_chunk.sizes["sequence"] == raw_label_chunk.sizes["sequence"], (
            f"Label chunk size {label_chunk.sizes['sequence']} != raw chunk size {raw_label_chunk.sizes['sequence']}"
        )

        # Combine original and computed label information
        label_chunk = xr.merge(
            [
                raw_label_chunk[["label_masks"]].rename(
                    {
                        "label_masks": "label_mask_reasons",
                    }
                ),
                label_chunk,
            ],
            combine_attrs="drop_conflicts",
            join="exact",
        )

        # Left join labels to tokens, implicitly dropping unannotated regions
        # of contigs past the last known annotation
        merged_chunk = xr.merge(
            [label_chunk, token_chunk], combine_attrs="drop_conflicts", join="left"
        )
        # Ensure that the join was complete on both sides
        assert merged_chunk.sizes["sequence"] == label_chunk.sizes["sequence"]

        # Write chunk to zarr
        # Only use append_dim for chunks after the first chunk of each group
        if chunk_idx == 0:
            # First chunk of this group - create new group
            logger.info(f"{prefix} Saving chunk to {output_path}/{output_group}")
            merged_chunk.to_zarr(
                output_path, group=output_group, zarr_format=2, consolidated=True
            )
        else:
            # Subsequent chunks of this group - append to existing group
            logger.info(f"{prefix} Appending chunk to {output_path}/{output_group}")
            merged_chunk.to_zarr(
                output_path,
                group=output_group,
                zarr_format=2,
                append_dim="sequence",
                consolidated=True,
            )

    return {
        "group": group_path,
        "processed": True,
        "species_id": species_id,
        "chrom_id": chrom_id,
        "sequence_length": seq_size,
        "chunks_processed": len(range(0, seq_size, chunk_size)),
    }


def create_sequence_dataset(
    input_labels_path: str,
    input_tokens_path: str,
    output_path: str,
    chunk_size: int = 100_000_000,
    num_workers: int | None = None,
) -> None:
    """Merge token and label datasets into a unified dataset.

    Parameters
    ----------
    input_labels_path : str
        Path to input zarr file containing labels
    input_tokens_path : str
        Path to input zarr file containing tokenized sequences
    output_path : str
        Path to output zarr file for the merged dataset
    chunk_size : int, optional
        Number of base pairs in one sequence to process per chunk for memory efficiency (default: 100M)
    num_workers : int | None, optional
        Number of processes to use for parallel processing (default: None for sequential processing)
    """
    # Get shared species/contig combinations from both datasets
    logger.info("Finding shared species/contig combinations")
    label_groups = list_species_contig_datatree(input_labels_path)
    token_groups = list_species_contig_datatree(input_tokens_path)

    # Create sets of group paths for intersection
    label_group_paths = {group["group_path"] for group in label_groups}
    token_group_paths = {group["group_path"] for group in token_groups}
    shared_group_paths = label_group_paths & token_group_paths

    # Filter to only shared groups
    shared_groups = [
        group for group in label_groups if group["group_path"] in shared_group_paths
    ]

    # Log statistics
    logger.info("Group statistics:")
    logger.info(f"  Labels dataset: {len(label_groups)} groups")
    logger.info(f"  Tokens dataset: {len(token_groups)} groups")
    logger.info(f"  Shared groups: {len(shared_groups)} groups")

    dropped_label_count = len(label_groups) - len(shared_groups)
    dropped_token_count = len(token_groups) - len(shared_groups)
    if dropped_label_count > 0:
        logger.warning(
            f"  Dropping {dropped_label_count} unshared groups from labels dataset"
        )
    if dropped_token_count > 0:
        logger.warning(
            f"  Dropping {dropped_token_count} unshared groups from tokens dataset"
        )

    # Determine number of processes to use
    if num_workers is None or num_workers == 0:
        n_processes = None
        logger.info("Using sequential processing (no multiprocessing)")
    else:
        n_processes = num_workers
        logger.info(f"Using {n_processes} workers for parallel processing")

    # Prepare arguments for worker processes
    task_args = [
        (
            group["group_path"],
            group["species_id"],
            group["chrom_id"],
            input_labels_path,
            input_tokens_path,
            output_path,
            chunk_size,
            i + 1,
            len(shared_groups),
        )
        for i, group in enumerate(shared_groups)
    ]

    # Process groups in parallel or sequentially
    logger.info("Starting group processing...")
    all_stats = []

    if n_processes is None:
        # Sequential processing
        results = map(_create_sequence_dataset_group, task_args)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            results = executor.map(_create_sequence_dataset_group, task_args)

    # Collect results
    all_stats = list(results)

    # Log processing summary
    processed_groups = [s for s in all_stats if s["processed"]]
    logger.info(
        f"Successfully processed {len(processed_groups)}/{len(shared_groups)} groups"
    )
    for stats in processed_groups:
        logger.info(
            f"  {stats['species_id']}/{stats['chrom_id']}: {stats['sequence_length']:,} bp in {stats['chunks_processed']} chunks"
        )

    logger.info("Done")


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Process GFF features and create labels"
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Command to execute"
    )

    # Filter features command
    filter_parser = subparsers.add_parser(
        "filter_features",
        help="Filter GFF features to canonical transcripts and remove overlaps",
    )
    filter_parser.add_argument(
        "--input", required=True, help="Path to input parquet file"
    )
    filter_parser.add_argument(
        "--output-features",
        required=True,
        help="Path to output parquet file for filtered features",
    )
    filter_parser.add_argument(
        "--output-filters",
        required=True,
        help="Path to output parquet file for filter definitions",
    )
    filter_parser.add_argument(
        "--remove-incomplete-features",
        choices=["yes", "no"],
        default="yes",
        help="Whether to filter out genes with incomplete feature annotations (default: yes)",
    )

    # Stack features command
    stack_parser = subparsers.add_parser(
        "stack_features", help="Convert features from wide to stacked format"
    )
    stack_parser.add_argument(
        "--input",
        required=True,
        help="Path to input parquet file with filtered features",
    )
    stack_parser.add_argument(
        "--output",
        required=True,
        help="Path to output parquet file for stacked features",
    )

    # Create labels command
    labels_parser = subparsers.add_parser(
        "create_labels", help="Create labels from filtered GFF features"
    )
    labels_parser.add_argument(
        "--input-features", required=True, help="Path to stacked features parquet file"
    )
    labels_parser.add_argument(
        "--input-filters", required=True, help="Path to filters parquet file"
    )
    labels_parser.add_argument(
        "--output", required=True, help="Path to output zarr file"
    )
    labels_parser.add_argument(
        "--remove-incomplete-features",
        choices=["yes", "no"],
        default="yes",
        help="Whether incomplete features result in interior masks (yes) or upstream/downstream boundary masks (no)",
    )
    labels_parser.add_argument(
        "--upstream-mask-size",
        type=int,
        default=944,
        help="Upstream mask size for boundary filters (default: 944 [99th percentile of feature lengths])",
    )
    labels_parser.add_argument(
        "--downstream-mask-size",
        type=int,
        default=1240,
        help="Downstream mask size for boundary filters (default: 1240 [99th percentile of feature lengths])",
    )

    # Create sequence dataset command
    sequence_parser = subparsers.add_parser(
        "create_sequence_dataset",
        help="Merge token and label datasets into a unified dataset",
    )
    sequence_parser.add_argument(
        "--input-labels",
        required=True,
        help="Path to input zarr file containing labels",
    )
    sequence_parser.add_argument(
        "--input-tokens",
        required=True,
        help="Path to input zarr file containing tokenized sequences",
    )
    sequence_parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output zarr file for the merged dataset",
    )
    sequence_parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000_000,
        help="Number of base pairs in one sequence to process per chunk for memory efficiency (default: 100M)",
    )
    sequence_parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of processes to use for parallel processing (default: None for sequential processing)",
    )

    args = parser.parse_args()

    if args.command == "filter_features":
        remove_incomplete = args.remove_incomplete_features == "yes"
        filter_features(
            args.input, args.output_features, args.output_filters, remove_incomplete
        )
    elif args.command == "stack_features":
        stack_features(args.input, args.output)
    elif args.command == "create_labels":
        remove_incomplete = args.remove_incomplete_features == "yes"
        create_labels(
            args.input_features,
            args.input_filters,
            args.output,
            remove_incomplete_features=remove_incomplete,
            boundary_mask_size=(args.upstream_mask_size, args.downstream_mask_size),
        )
    elif args.command == "create_sequence_dataset":
        create_sequence_dataset(
            args.input_labels,
            args.input_tokens,
            args.output_path,
            args.chunk_size,
            args.num_workers,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
