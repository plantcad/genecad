import pandas as pd
import argparse
import os
import logging
from dataclasses import dataclass, asdict
from src.dataset import DEFAULT_SEQUENCE_CHUNK_SIZE, open_datatree, set_dimension_chunks
from src.sequence import find_overlapping_intervals, convert_entity_intervals_to_labels, convert_to_biluo_labels, convert_biluo_entity_names
import numpy as np
from src.schema import FeatureLevel, RegionType, FeatureType, SentinelType
import xarray as xr
from src.config import get_species_config, get_species_configs

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Filter features
# -------------------------------------------------------------------------------------------------

@dataclass
class Filter:
    species_id: str
    chromosome_id: str
    gene_id: str
    reason: str

def create_filters(df: pd.DataFrame, reason: str) -> list[Filter]:
    """Create Filter objects resulting from a shared cause. """
    return [
        Filter(
            species_id=row['species_id'],
            chromosome_id=row['chromosome_id'],
            gene_id=row['gene_id'],
            reason=reason
        )
        for _, row in (
            df
            .drop_duplicates(subset=['species_id', 'chromosome_id', 'gene_id'])
            .iterrows()
        )
    ]

def apply_filters(df: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
    """Filter a dataframe by removing rows corresponding to genes in the filters list."""
    if not filters:
        return df
    
    # Extract unique (species_id, chromosome_id, gene_id) tuples from filters
    genes_to_filter = set(
        (f.species_id, f.chromosome_id, f.gene_id) 
        for f in filters
    )
    
    # Filter out the genes
    return df[
        ~df[['species_id', 'chromosome_id', 'gene_id']].apply(tuple, axis=1).isin(genes_to_filter)
    ]

def filter_canonical_transcripts(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Filter]]:
    """Filter the dataframe to only keep canonical transcripts. """
    
    # Count canonical transcripts per gene
    canonical_counts = (
        # First, get unique transcripts (we have to only count each transcript once)
        df[['species_id', 'chromosome_id', 'gene_id', 'transcript_id', 'transcript_is_canonical']]
        .drop_duplicates()
        .groupby(['species_id', 'chromosome_id', 'gene_id'])['transcript_is_canonical']
        .sum().reset_index()
    )
    
    # Find genes with no canonical transcripts
    genes_with_no_canonical = canonical_counts[canonical_counts['transcript_is_canonical'] == 0]
    logger.info(f"Found {len(genes_with_no_canonical)} genes with no canonical transcripts")
    
    # Find genes with multiple canonical transcripts
    genes_with_multiple_canonical = canonical_counts[canonical_counts['transcript_is_canonical'] > 1]
    logger.info(f"Found {len(genes_with_multiple_canonical)} genes with multiple canonical transcripts")
    
    # Create filters for both cases
    filters = (
        create_filters(genes_with_no_canonical, 'no_canonical_transcript')
        + create_filters(genes_with_multiple_canonical, 'multiple_canonical_transcripts')
    )
    
    # Filter out filtered genes and keep only canonical transcripts
    df_filtered = apply_filters(df, filters)
    
    # Further filter to only keep canonical transcripts
    df_filtered = df_filtered[df_filtered['transcript_is_canonical']]
    
    logger.info("Canonical transcript filters:")
    logger.info(f"  Original dataframe had {len(df)} rows")
    logger.info(f"  Filtered dataframe has {len(df_filtered)} rows")
    logger.info(f"  Removed {len(filters)} genes due to canonical transcript issues")
    
    return df_filtered, filters

def filter_incomplete_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Filter]]:
    """Filter the dataframe to only keep genes with fully annotated canonical transcripts. """
    # Get list of "base" features, meaning those at the lowest level (i.e. UTRs and CDS)
    base_features = FeatureType.get_values(level=2)
    genes_with_incomplete_features = (
        df
        # It is crucial to do this first as significantly more 
        # non-canonical transcripts are not fully annotated
        .pipe(lambda df: df[df['transcript_is_canonical']])
        # Filter to only chosen feature types
        .pipe(lambda df: df[df['feature_type'].isin(base_features)])
        # Count distinct feature types per gene
        .groupby(['species_id', 'chromosome_id', 'gene_id'])['feature_type']
        .nunique().reset_index()
        .pipe(lambda df: df[df['feature_type'] < len(base_features)])
    )

    logger.info(f"Found {len(genes_with_incomplete_features)} genes with incomplete canonical transcript annotations")
    
    # Create and apply filters
    filters = create_filters(genes_with_incomplete_features, reason='incomplete_features')
    df_filtered = apply_filters(df, filters)
    
    logger.info("Incomplete feature filters:")
    logger.info(f"  Original dataframe had {len(df)} rows")
    logger.info(f"  Filtered dataframe has {len(df_filtered)} rows")
    logger.info(f"  Removed {len(filters)} genes due to incomplete feature annotations")
    return df_filtered, filters


def find_overlapping_genes(df: pd.DataFrame) -> list[Filter]:
    """Identify genes that overlap with other genes on the same chromosome."""
    filters = []
    
    # Get unique genes with their positions
    genes = df[['species_id', 'chromosome_id', 'gene_id', 'gene_start', 'gene_stop']].drop_duplicates()
    total_genes = _gene_count(df)
    
    # Group by species and chromosome
    for (species_id, chrom), group in genes.groupby(['species_id', 'chromosome_id']):
        
        # Find overlapping genes within this group
        has_overlap = find_overlapping_intervals(group['gene_start'], group['gene_stop'])
        
        # Get DataFrame of overlapping genes
        overlapping_genes_df = group.loc[has_overlap]
        
        # Add to filters using utility function
        filters.extend(create_filters(overlapping_genes_df, 'overlapping_gene'))
        
        logger.info(f"Found {len(overlapping_genes_df)} overlapping genes in {species_id}, chromosome {chrom}")
    
    logger.info(f"Found {len(filters)} genes with overlaps (out of {total_genes} total genes)")
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
    
    # Group by species, chromosome, gene, and transcript
    # (there should be only one transcript per gene after filter_canonical_transcripts)
    for (species_id, chrom, gene_id, transcript_id), features_df in df.groupby(
        ['species_id', 'chromosome_id', 'gene_id', 'transcript_id']
    ):
        # Check for overlapping features
        has_overlap = find_overlapping_intervals(
            features_df['feature_start'], 
            features_df['feature_stop']
        )
        
        # If any features overlap, add the gene to filters
        if has_overlap.any():
            # Create a DataFrame with a single row for this gene
            gene_df = pd.DataFrame([{
                'species_id': species_id,
                'chromosome_id': chrom,
                'gene_id': gene_id
            }])
            
            # Add to filters using utility function
            filters.extend(create_filters(gene_df, 'overlapping_features'))
    
    logger.info(f"Found {len(filters)} genes with overlapping features (out of {total_genes} total genes)")
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
    return df[['species_id', 'chromosome_id', 'gene_id']].drop_duplicates().pipe(len)

def filter_features(input_path: str, features_output_path: str, filters_output_path: str) -> None:
    """Process GFF features by filtering to canonical transcripts and removing overlapping features.
    
    Parameters
    ----------
    input_path : str
        Path to input parquet file
    features_output_path : str
        Path to output parquet file for filtered features
    filters_output_path : str
        Path to output parquet file for filter definitions
    """
    # Read input dataframe
    logger.info(f"Reading input from {input_path}")
    features = pd.read_parquet(input_path)
    
    # Sort the dataframe
    features = features.sort_values(['species_id', 'chromosome_id', 'gene_start', 'transcript_start', 'feature_start'])
    total_rows = len(features)
    total_genes = _gene_count(features)
    
    # Filter to canonical transcripts
    filtered_features, canonical_filters = filter_canonical_transcripts(features)
    
    # Find overlapping genes
    filtered_features, gene_overlap_filters = filter_overlapping_genes(filtered_features)

    # Filter to transcripts with complete features
    filtered_features, incomplete_features_filters = filter_incomplete_features(filtered_features)
    
    # Find genes with overlapping features
    filtered_features, feature_overlap_filters = filter_overlapping_features(filtered_features)
    
    # Create filters DataFrame for statistics
    filters = pd.DataFrame([
        asdict(f)
        for f in (
            canonical_filters + 
            gene_overlap_filters + 
            incomplete_features_filters +
            feature_overlap_filters
        )
    ])
    filtered_genes = total_genes - _gene_count(filtered_features)
    
    # Log final statistics
    logger.info("Final filter summary:")
    logger.info(f"  Original dataframe had {total_rows} rows")
    logger.info(f"  Filtered dataframe has {len(filtered_features)} rows")
    logger.info(f"  Of {total_genes} total genes, {filtered_genes} were removed ({filtered_genes / total_genes * 100:.2f}%)")
    
    # Log filters reason statistics
    if not filters.empty:
        reason_counts = filters['reason'].value_counts()
        logger.info("Filter reason counts:")
        for reason, count in reason_counts.items():
            logger.info(f"  - {reason}: {count}")

    # Get spans associated with filtered features (for downstream masking)
    n_filters = len(filters)
    filters = (
        filters
        .merge(
            features[['species_id', 'chromosome_id', 'gene_id', 'gene_start', 'gene_stop']]
            .drop_duplicates(subset=['species_id', 'chromosome_id', 'gene_id']),
            on=['species_id', 'chromosome_id', 'gene_id'],
            how='inner',
            validate='m:1'
        )
    )
    assert len(filters) == n_filters
    
    # Save results
    logger.info(f"Saving filtered features to {features_output_path}")
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(features_output_path), exist_ok=True)
    filtered_features.to_parquet(features_output_path, index=False)

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
    base_key = ["species_id", "chromosome_id"]
    base_cols = ["filename", "chromosome_length"]
    for prefix, type, key, level in [
        ("gene", FeatureType.GENE, ["gene_id"], FeatureLevel.GENE),
        ("transcript", FeatureType.MRNA, ["gene_id", "transcript_id"], FeatureLevel.TRANSCRIPT),
        ("feature", None, ["gene_id", "transcript_id", "feature_id"], FeatureLevel.ANNOTATION),
    ]:
        cols = {f"{prefix}_{c}": c for c in ["id", "name", "strand", "start", "stop"]}
        if type is None:
            cols["feature_type"] = "feature_type"
        group = (
            df
            # Assign path tuple defining parent ids
            .assign(path=lambda df: df[base_key + key[:-1]].apply(tuple, axis=1))
            .drop_duplicates(subset=base_key + key)
            [base_cols + base_key + list(cols) + ["path"]]
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
    
    logger.info("Stacking features")
    features = stack_gff_features(features)
    logger.info(f"Stacked features:\n{features.head()}")
    features.info()
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Saving stacked features to {output_path}")
    features.to_parquet(output_path, index=False)
    logger.info("Done")


# -------------------------------------------------------------------------------------------------
# Create labels
# -------------------------------------------------------------------------------------------------


def _create_feature_labels(df: pd.DataFrame, domain: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    # Filter to lowest level features like CDS, 5' UTR, 3' UTR, etc.
    annotation_features = [
        ft for ft in FeatureType 
        if FeatureType.value_to_level()[ft] == FeatureLevel.ANNOTATION
    ]
    df = df[df["feature_type"].isin(annotation_features)]
    
    feature_type_to_index = {ft: i for i, ft in enumerate(annotation_features)}
    feature_coords = [FeatureType.value_to_slug()[ft] for ft in annotation_features]
    num_feature_types = len(feature_coords)
    domain_size = domain[1] - domain[0]

    # Map feature types to 1-based label indices
    labels = df["feature_type"].map(feature_type_to_index) + 1
    if labels.isna().any():
        invalid_feature_types = df[labels.isna()]["feature_type"].unique()
        raise ValueError(f"Found unmapped feature types: {invalid_feature_types}")
    
    # Prepare intervals for vectorization
    intervals = (
        df.assign(label=labels)
        .rename(columns={"feature_level": "level"})
        [["start", "stop", "label", "level"]]
    )

    # Generate labels at each level with errors raised for overlapping intervals
    feature_labels = convert_entity_intervals_to_labels(
        intervals, domain, num_labels=num_feature_types, on_overlap="raise"
    ) 
    assert feature_labels.shape == (domain_size, num_feature_types)
    return feature_labels, feature_coords

def _create_region_labels(group: pd.DataFrame, domain: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    domain_size = domain[1] - domain[0]
    num_region_types = len(RegionType)
    region_coords: list[str] = list(RegionType)

    # Helper function to create region dataframes from specific feature types
    def create_region_df(feature_types: list[FeatureType], label_index: int) -> pd.DataFrame:
        return (
            group
            .pipe(lambda df: df[df["feature_type"].isin(feature_types)])
            [["path", "start", "stop"]]
            .assign(label=label_index)
        )
    
    # Helper function to create aggregated region dataframes
    def create_aggregated_region_df(feature_types: list[FeatureType], label_index: int) -> pd.DataFrame:
        return (
            group
            .pipe(lambda df: df[df["feature_type"].isin(feature_types)])
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
        create_region_df([FeatureType.GENE], RegionType.GENE.get_index()),
        create_region_df([FeatureType.MRNA], RegionType.TRANSCRIPT.get_index()),
        create_region_df(
            [FeatureType.CDS, FeatureType.FIVE_PRIME_UTR, FeatureType.THREE_PRIME_UTR], 
            RegionType.EXON.get_index()
        ),
        
        # Regions defined by min/max spans of consituent features
        create_aggregated_region_df([FeatureType.CDS], RegionType.CODING_SEQUENCE.get_index()),
        create_aggregated_region_df([FeatureType.FIVE_PRIME_UTR], RegionType.FIVE_PRIME_UTR.get_index()),
        create_aggregated_region_df([FeatureType.THREE_PRIME_UTR], RegionType.THREE_PRIME_UTR.get_index()),

        # This is not the definition of introns, but rather a region confining where they should exist;
        # they will be fully defined later by subtracting regions from this initial window
        create_aggregated_region_df(
            [FeatureType.CDS, FeatureType.FIVE_PRIME_UTR, FeatureType.THREE_PRIME_UTR], 
            RegionType.INTRON.get_index()
        ),
    ]
    
    # Combine all region dataframes
    region_interval_sets = pd.concat(region_interval_sets, axis=0, ignore_index=True)
    
    # Convert to required format for convert_entity_intervals_to_labels
    region_intervals = region_interval_sets.rename(columns={"path": "path_col"})[["start", "stop", "label"]]
    
    # Generate region labels
    region_labels = convert_entity_intervals_to_labels(
        # Convert all enum indices to 1-based indices for labelling
        region_intervals.assign(label=lambda df: df["label"] + 1),
        domain,
        num_labels=num_region_types,
        on_overlap="ignore"
    )
    assert region_labels.shape == (domain_size, num_region_types)
    
    # Compute introns by subtracting exons from the window defined by any existing
    # transcript features, not just the bounds of the transcript itself
    exon_label = region_labels[:, RegionType.EXON.get_index()]
    intron_mask = region_labels[:, RegionType.INTRON.get_index()]
    intron_label = intron_mask.astype(bool) & ~exon_label.astype(bool)
    region_labels[:, RegionType.INTRON.get_index()] = intron_label.astype(np.int8)

    return region_labels, region_coords

def _generate_label_masks(group_filters: pd.DataFrame, domain: tuple) -> tuple[np.ndarray, list[str]]:
    reasons = group_filters["reason"].drop_duplicates().sort_values().to_list()
    reason_to_index = {reason: i for i, reason in enumerate(reasons)}
    label_masks = convert_entity_intervals_to_labels(
        group_filters[["gene_start", "gene_stop", "reason"]]
        .drop_duplicates()
        .rename(columns={"gene_start": "start", "gene_stop": "stop"})
        .assign(label=lambda df: df["reason"].map(reason_to_index) + 1),
        domain,
        num_labels=len(reasons),
        on_overlap="ignore"
    )
    # Labels indicate where filters are active, so invert this
    # representation to preserve positive mask convention
    label_masks = (~label_masks.astype(bool)).astype(label_masks.dtype)
    return label_masks, reasons


def create_labels(features_path: str, filters_path: str, output_path: str, chunk_size: int = DEFAULT_SEQUENCE_CHUNK_SIZE) -> None:
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
    features.info()
    
    logger.info(f"Loading filter definitions from {filters_path}")
    filters = pd.read_parquet(filters_path)
    logger.info(f"Loaded {len(filters)} filters:\n{filters.head()}")
    filters.info()

    # Map strand values to string labels
    features = features.assign(strand=lambda df: df["strand"].map({-1: "negative", 1: "positive"}))
    if features["strand"].isna().any():
        invalid_strands = features[features["strand"].isna()]["strand"].unique()
        raise ValueError(f"Found unmapped strand values: {invalid_strands}")

    # Group by species and chromosome
    for (species_id, chrom_id), chrom in features.groupby(["species_id", "chromosome_id"]):
        # Get chromosome information
        chrom_length = chrom["chromosome_length"].iloc[0]
        domain = (0, chrom_length)
        filename = chrom["filename"].iloc[0]

        # Process each strand separately
        data = []
        for strand, group in chrom.groupby("strand"):
            logger.info(f"Processing {species_id!r}, chromosome {chrom_id!r}, strand {strand!r}")

            # Generate feature labels
            feature_labels, feature_coords = _create_feature_labels(group, domain)
            assert np.isin(feature_labels, [0, 1]).all()

            region_labels, region_coords = _create_region_labels(group, domain)
            assert np.isin(region_labels, [0, 1]).all()
        
            # Generate sequence masks from filters
            group_filters = filters[(filters["species_id"] == species_id) & (filters["chromosome_id"] == chrom_id)]
            label_masks, reasons = _generate_label_masks(group_filters, domain)
            
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
        ds = (
            xr.concat(data, dim="strand")
            .assign_attrs(
                species_id=species_id,
                chromosome_id=chrom_id,
                filename=filename,
            )
        )
        ds = set_dimension_chunks(ds, "sequence", chunk_size)
        logger.info(f"Saving data for group '{species_id}/{chrom_id}' to {output_path}")
        ds.to_zarr(output_path, group=f"{species_id}/{chrom_id}", zarr_format=2, consolidated=True, mode="w")

    dt = open_datatree(output_path)
    logger.info(f"Final data tree:\n{dt}")
    logger.info("Done")


# -------------------------------------------------------------------------------------------------
# Create sequence dataset
# -------------------------------------------------------------------------------------------------

def create_sequence_dataset(input_labels_path: str, input_tokens_path: str, output_path: str) -> None:
    """Merge token and label datasets into a unified dataset.
    
    Parameters
    ----------
    input_labels_path : str
        Path to input zarr file containing labels
    input_tokens_path : str
        Path to input zarr file containing tokenized sequences
    output_path : str
        Path to output zarr file for the merged dataset
    """
    # Load both datasets using DataTree
    logger.info(f"Loading labels from {input_labels_path}")
    labels = open_datatree(input_labels_path)
    
    logger.info(f"Loading tokens from {input_tokens_path}")
    tokens = open_datatree(input_tokens_path)

    # Filter datasets to only include common groups
    label_groups = set(labels.groups)
    token_groups = set(tokens.groups)
    shared_groups = label_groups & token_groups
    
    # Identify shared groups and summarize those dropped
    logger.info("Group statistics:")
    logger.info(f"  Labels dataset: {len(label_groups)} groups")
    logger.info(f"  Tokens dataset: {len(token_groups)} groups")
    logger.info(f"  Shared groups: {len(shared_groups)} groups")
    dropped_label_groups = label_groups - shared_groups
    dropped_token_groups = token_groups - shared_groups
    
    if dropped_label_groups:
        logger.warning(f"  Dropping {len(dropped_label_groups)} unshared groups from labels dataset: {dropped_label_groups}")
    if dropped_token_groups:
        logger.warning(f"  Dropping {len(dropped_token_groups)} unshared groups from tokens dataset: {dropped_token_groups}")
    
    # Process each shared group individually
    for group in shared_groups:
        # Extract species_id and chrom_id from group path (handle leading /)
        parts = group.strip('/').split('/')
        if len(parts) > 2:
            raise ValueError(f"Found invalid group: {group}")
        if len(parts) < 2:
            continue
        species_id, chrom_id = parts
        logger.info(f"Processing group: {species_id}/{chrom_id}")
        
        # Get datasets for this group
        label_ds = labels[group].ds
        token_ds = tokens[group].ds
        
        # Ensure that sequence tokens always span a longer range than labels
        if token_ds.sizes["sequence"] < label_ds.sizes["sequence"]:
            raise ValueError(f"Group {group}: More labels ({label_ds.sizes['sequence']}) than tokens ({token_ds.sizes['sequence']})")
        
        # Left join labels to tokens
        merged_ds = xr.merge([label_ds, token_ds], combine_attrs="drop_conflicts", join="left")
        assert merged_ds.sizes["sequence"] == label_ds.sizes["sequence"]
        
        # Write the merged dataset to its group
        output_group = f"{species_id}/{chrom_id}"
        logger.info(f"  Saving to {output_path}/{output_group}")
        merged_ds.to_zarr(output_path, group=output_group, zarr_format=2, consolidated=True, mode="w")
    
    # Load and print the full datatree
    dt = open_datatree(output_path)
    logger.info(f"Final data tree:\n{dt}")
    logger.info("Done")


# -------------------------------------------------------------------------------------------------
# Write npz labels
# -------------------------------------------------------------------------------------------------

def get_npz_label_path(species_id: str, directory: str) -> str:
    """Get the path to the NPZ labels file for a species.
    
    Parameters
    ----------
    species_id : str
        Species ID
    directory : str
        Directory containing NPZ files
        
    Returns
    -------
    str
        Path to NPZ labels file
    """
    return os.path.join(directory, f"{species_id}_biluo_4class_labels.npz")

def extract_npz_labels(input_path: str, output_path: str, species_ids: list[str] = None) -> None:
    """Extract npz labels from Xarray DataTree.
    
    Parameters
    ----------
    input_path : str
        Path to input zarr file (output from create_labels)
    output_path : str
        Path to output npz file
    species_ids : list[str], optional
        List of species IDs to process. If None, all species will be processed
    """
    logger.info(f"Loading sequence datasets from {input_path}")
    dt = open_datatree(input_path, consolidated=False)
    datasets = [dt.ds for dt in dt.subtree if dt.is_leaf]
    logger.info(f"Loaded {len(datasets)} sequence datasets")

    # Group datasets by species to process one species at a time
    species_datasets = {}
    for ds in datasets:
        species_id = ds.attrs["species_id"]
        if species_ids and species_id not in species_ids:
            continue
        if species_id not in species_datasets:
            species_datasets[species_id] = []
        species_datasets[species_id].append(ds)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Process each species separately
    for species_id, species_ds_list in species_datasets.items():
        logger.info(f"Processing species: {species_id}")
        species_config = get_species_config(species_id)
        chromosomes = {}
        
        # Process all datasets for this species
        for i, ds in enumerate(species_ds_list):
            chrom_id = ds.attrs["chromosome_id"]
            chrom_num = species_config.get_chromosome_number(chrom_id)
            if chrom_num is None:
                raise ValueError(f"Unable to determine chromosome number for id {chrom_id}")
                
            logger.info(f"[species={species_id}] Processing {chrom_id=} ({chrom_num=})")

            # Extract non-overlapping, low-level features
            intron_labels = ds.region_labels.sel(region='intron').rename(region="feature")
            exonic_labels = ds.feature_labels.sel(feature=['five_prime_utr', 'cds', 'three_prime_utr'])
            assert intron_labels.dtype == exonic_labels.dtype
            labels = xr.concat([intron_labels, exonic_labels], dim='feature').astype(intron_labels.dtype)
            if labels.sum(dim='feature').max().item() > 1:
                raise ValueError("Found multiple features assigned to the same genomic coordinate")
            
            # Aggregate low-level features into masked, 1D labels
            label_classes = labels.feature.values.tolist()
            labels = xr.where(labels.any(dim='feature'), labels.argmax(dim='feature') + 1, 0)
            # mask=True ==> keep label, so all must be True across multiple masks
            mask = ds.label_masks.astype(bool).all(dim='reason') 
            labels = xr.where(mask, labels, -1)
            assert labels.dims == ('strand', 'sequence')

            # Convert to biluo labels (flip negative strand for labelling)
            tags_list = []
            for strand_name in labels.strand.values:
                values = labels.sel(strand=strand_name, drop=True).values
                if strand_name == 'positive':
                    tags = convert_to_biluo_labels(values)
                else:
                    tags = np.flip(convert_to_biluo_labels(np.flip(values)))
                tags_list.append(tags)
            tags = xr.DataArray(
                np.stack(tags_list), 
                dims=['strand', 'sequence'],
                coords={'strand': labels.strand.values, 'sequence': labels.sequence.values}
            )
            assert tags.shape == labels.shape

            # Show tag frequency
            tag_classes = convert_biluo_entity_names(label_classes)
            tag_class_map = {
                **{i: str(v) for i, v in SentinelType.index_to_value().items()}, 
                **pd.Series(tag_classes, index=np.arange(len(tag_classes)) + 1).to_dict()
            }
            if i == 0:
                logger.info(f"  Tag class map:\n{tag_class_map}")
            tag_frequency = tags.to_series().map(tag_class_map).fillna("Unknown").value_counts()
            logger.info(f"  Tag frequency:\n{tag_frequency}")

            # Store result
            result = (
                # Label npz convention is forward strand in first column
                # with shape (chrom_length, 2)
                tags.transpose("sequence", "strand")
                .sel(strand=["positive", "negative"])
                .to_numpy().astype(np.int8)
            )
            assert result.ndim == 2
            assert result.shape[1] == 2
            assert result.dtype == np.int8
            chromosomes[chrom_id] = dict(chrom_num=chrom_num, result=result)
        
        # Save this species to disk and free memory
        npz_path = get_npz_label_path(species_id, output_path)
        logger.info(f"[species={species_id}] Saving {npz_path}")
        # Sort chromosomes names by numerical order rather than lexical order;
        # insertion order is an essential assumption of the current downstream data loader
        chrom_ids = sorted(chromosomes.keys(), key=lambda x: chromosomes[x]["chrom_num"])
        np.savez(npz_path, **{k: chromosomes[k]["result"] for k in chrom_ids})

    logger.info("Done")


# -------------------------------------------------------------------------------------------------
# Extract npz keyfile
# -------------------------------------------------------------------------------------------------

def extract_npz_keyfile(output_path: str, species_ids: list[str], labels_dir: str, data_dir: str) -> None:
    """Generate a keyfile mapping species to their label, fasta, and window files.
    
    Parameters
    ----------
    output_path : str
        Path to save the output TSV keyfile
    species_ids : list[str]
        List of species IDs to include in the keyfile
    labels_dir : str
        Path to directory containing NPZ label files
    data_dir : str
        Path to raw data directory with subdirs like "fasta" and "windows"
    """
    # Get configurations for all specified species
    all_configs = get_species_configs(species_ids)
    
    # Filter for training species only
    training_configs = [
        config for config in all_configs
        if config.split.use_in_training
    ]
    
    if not training_configs:
        raise ValueError(f"No training species found among the {len(all_configs)} provided species.")
    logger.info(f"Found {len(training_configs)} training species:")
    for config in training_configs:
        logger.info(f"  {config.id}")

    # Collect data for each species
    data = []
    for config in training_configs:
        species_id = config.id
        
        # Check if windows.filename exists
        if not config.windows or not config.windows.filename:
            raise ValueError(f"Species {species_id} is missing required windows.filename configuration")
            
        # Construct paths
        labels_path = get_npz_label_path(species_id, labels_dir)
        fasta_path = os.path.join(data_dir, "fasta", config.fasta.filename)
        windows_path = os.path.join(data_dir, "windows", config.windows.filename)
        
        # Check if all paths exist
        paths_exist = True
        if not os.path.exists(labels_path):
            logger.warning(f"Labels path does not exist for {species_id}: {labels_path}")
            paths_exist = False
        if not os.path.exists(fasta_path):
            logger.warning(f"Fasta path does not exist for {species_id}: {fasta_path}")
            paths_exist = False
        if not os.path.exists(windows_path):
            logger.warning(f"Windows path does not exist for {species_id}: {windows_path}")
            paths_exist = False
            
        # Skip this config if any path doesn't exist
        if not paths_exist:
            logger.warning(f"Skipping {species_id} due to missing files")
            continue
            
        data.append({
            "name": species_id,
            "labels": labels_path,
            "fasta": fasta_path,
            "windows": windows_path
        })
    
    if not data:
        raise ValueError("No valid species configurations found with all required files")
    
    # Create DataFrame and save as TSV
    df = pd.DataFrame(data)
    logger.info(f"Generated keyfile with {len(df)} entries")
    
    # Set pandas to display all rows and columns
    with pd.option_context(
        'display.max_rows', None, 
        'display.max_columns', None, 
        'display.expand_frame_repr', False,
        'display.max_colwidth', None
    ):
        logger.info(f"Keyfile contents:\n{df}")
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as TSV
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved keyfile to {output_path}")


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Process GFF features and create labels")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")
    
    # Filter features command
    filter_parser = subparsers.add_parser("filter_features", help="Filter GFF features to canonical transcripts and remove overlaps")
    filter_parser.add_argument("--input", required=True, help="Path to input parquet file")
    filter_parser.add_argument("--output-features", required=True, help="Path to output parquet file for filtered features")
    filter_parser.add_argument("--output-filters", required=True, help="Path to output parquet file for filter definitions")
    
    # Stack features command
    stack_parser = subparsers.add_parser("stack_features", help="Convert features from wide to stacked format")
    stack_parser.add_argument("--input", required=True, help="Path to input parquet file with filtered features")
    stack_parser.add_argument("--output", required=True, help="Path to output parquet file for stacked features")
    
    # Create labels command
    labels_parser = subparsers.add_parser("create_labels", help="Create labels from filtered GFF features")
    labels_parser.add_argument("--input-features", required=True, help="Path to stacked features parquet file")
    labels_parser.add_argument("--input-filters", required=True, help="Path to filters parquet file")
    labels_parser.add_argument("--output", required=True, help="Path to output zarr file")

    # Create sequence dataset command
    sequence_parser = subparsers.add_parser("create_sequence_dataset", help="Merge token and label datasets into a unified dataset")
    sequence_parser.add_argument("--input-labels", required=True, help="Path to input zarr file containing labels")
    sequence_parser.add_argument("--input-tokens", required=True, help="Path to input zarr file containing tokenized sequences")
    sequence_parser.add_argument("--output-path", required=True, help="Path to output zarr file for the merged dataset")

    # Extract npz labels command
    extract_parser = subparsers.add_parser("extract_npz_labels", help="Extract npz labels from DataTree")
    extract_parser.add_argument("--input", required=True, help="Path to input zarr file")
    extract_parser.add_argument("--output", required=True, help="Path to output npz file")
    extract_parser.add_argument("--species-id", nargs="+", help="List of species IDs to process (if not specified, all species are processed)")
    
    # Extract npz keyfile command
    keyfile_parser = subparsers.add_parser("extract_npz_keyfile", help="Generate a keyfile mapping species to their label, fasta, and window files")
    keyfile_parser.add_argument("--output", required=True, help="Path to output TSV keyfile")
    keyfile_parser.add_argument("--species-id", nargs="+", required=True, help="List of species IDs to include in the keyfile")
    keyfile_parser.add_argument("--labels-dir", required=True, help="Path to directory containing NPZ label files")
    keyfile_parser.add_argument("--data-dir", required=True, help="Path to data directory with 'fasta' and 'windows' subdirs")
    
    args = parser.parse_args()
    
    if args.command == "filter_features":
        filter_features(args.input, args.output_features, args.output_filters)
    elif args.command == "stack_features":
        stack_features(args.input, args.output)
    elif args.command == "create_labels":
        create_labels(args.input_features, args.input_filters, args.output)
    elif args.command == "create_sequence_dataset":
        create_sequence_dataset(args.input_labels, args.input_tokens, args.output_path)
    elif args.command == "extract_npz_labels":
        extract_npz_labels(args.input, args.output, args.species_id)
    elif args.command == "extract_npz_keyfile":
        extract_npz_keyfile(args.output, args.species_id, args.labels_dir, args.data_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
