import argparse
import logging
import pandas as pd
import re
import os
import glob
import shutil
from src.gff_pandas import read_gff3, write_gff3
from src.gff_compare import run_gffcompare, parse_gffcompare_stats
from src.gff_eval import run_gffeval
from src.config import SPECIES_CONFIGS, get_species_configs
from src.schema import GffFeatureType, RegionType

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------------------


def load_gff(path: str, attributes_to_drop: list[str] | None = None) -> pd.DataFrame:
    """Load GFF file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to input GFF file
    attributes_to_drop : list[str] | None, optional
        List of attributes to drop from the GFF file, typically to prevent
        conflicts in normalized names
    """
    logger.info(f"Loading GFF file {path}")
    df = read_gff3(path)
    logger.info(f"Loading complete: {df.shape[0]} records found")

    if attributes_to_drop:
        # Drop parsed attributes and remove them from the original attributes as a delimited string
        # TODO: Move away from GFF for intermediate representations to avoid these terrible standards
        df = df.drop(columns=attributes_to_drop)
        df["attributes"] = [
            ";".join(
                [
                    kv
                    for kv in attrs.split(";")
                    if kv.split("=")[0] not in attributes_to_drop
                ]
            )
            for attrs in df["attributes"].fillna("")
        ]

    # Create mapping of old column names to new lcase names
    col_mapping = {}
    for col in df.columns:
        new_name = re.sub(r"\s+", "_", col).lower()
        if new_name in col_mapping.values():
            raise ValueError(
                f"Column name collision detected: multiple columns would map to '{new_name}'"
            )
        col_mapping[col] = new_name

    return df.rename(columns=col_mapping)


def save_gff(path: str, df: pd.DataFrame) -> None:
    """Save GFF dataframe to a file.

    Parameters
    ----------
    path : str
        Path to output GFF file
    df : pd.DataFrame
        DataFrame object to write, must have 'header' in attrs
    """
    # Write to file
    logger.info(f"Writing GFF to {path}")
    write_gff3(df, path)
    logger.info(f"Complete: {df.shape[0]} records written")


def remove_features_by_id(
    features: pd.DataFrame, feature_ids_to_remove: set
) -> tuple[pd.DataFrame, int, int]:
    """Remove features by ID and all their children recursively.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame with all feature data
    feature_ids_to_remove : set
        Set of feature IDs to remove

    Returns
    -------
    tuple
        (filtered features, count of removed features, count of indirect removals)
    """
    # Find all children of features to remove
    to_remove = set(feature_ids_to_remove)
    indirect_removals = set()

    # Keep adding children until no more are found
    remaining_ids = set(feature_ids_to_remove)
    while remaining_ids:
        # Find all direct children of the current set of features
        children = set(features[features["parent"].isin(remaining_ids)]["id"].dropna())
        # If no new children, we're done
        if not children - to_remove:
            break
        # Add new children to the set of features to remove
        new_children = children - to_remove
        indirect_removals.update(new_children)
        remaining_ids = new_children
        to_remove.update(new_children)

    # Convert IDs to positional indices for features with and without IDs
    indices_to_remove = set()

    # First collect indices for features that have IDs in the removal set
    has_id_indices = features[features["id"].isin(to_remove)].index
    indices_to_remove.update(has_id_indices)

    # Then collect indices for features that have parents in the removal set
    # (this covers features without IDs)
    has_parent_indices = features[features["parent"].isin(to_remove)].index
    indices_to_remove.update(has_parent_indices)

    # Filter out the features to remove by index
    original_count = features.shape[0]
    filtered_features = features.drop(index=indices_to_remove)
    removed_count = original_count - filtered_features.shape[0]

    return filtered_features, removed_count, len(indirect_removals)


# -------------------------------------------------------------------------------------------------
# CLI functions
# -------------------------------------------------------------------------------------------------


def resolve_gff_file(input_dir: str, species_id: str, output_path: str) -> None:
    """Resolve and copy a GFF file for a species using its configuration.

    Parameters
    ----------
    input_dir : str
        Directory containing input GFF files
    species_id : str
        Species ID to resolve the GFF file for
    output_path : str
        Path to copy the resolved GFF file to
    """
    logger.info(f"Resolving GFF file for species {species_id} from {input_dir}")

    # Get the species configuration
    species_configs = get_species_configs([species_id])
    if not species_configs:
        raise ValueError(f"No configuration found for species ID: {species_id}")

    config = species_configs[0]

    # Construct the input file path using the species config
    gff_filename = config.gff.filename
    input_path = os.path.join(input_dir, gff_filename)

    logger.info(f"Resolved GFF filename: {gff_filename}")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")

    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"GFF file not found: {input_path}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Copy the file
    logger.info(f"Copying {input_path} to {output_path}")
    shutil.copy2(input_path, output_path)
    logger.info("GFF file resolved and copied successfully")


def merge_gff_files(input_paths: list[str], output_path: str) -> None:
    """Merge multiple GFF files into a single file.

    Parameters
    ----------
    input_paths : list[str]
        Paths to input GFF files
    output_path : str
        Path to output GFF file
    """
    if not input_paths:
        logger.warning("No input files provided")
        return

    # Read and merge GFF files
    logger.info(f"Merging {len(input_paths)} GFF files")

    # Read the first file
    first_df = load_gff(input_paths[0])

    if len(input_paths) > 1:
        # Read and concatenate additional files
        additional_dfs = [load_gff(path) for path in input_paths[1:]]

        # Concatenate all dataframes
        merged_df = pd.concat([first_df] + additional_dfs, ignore_index=True)

        # Use attrs from the first dataframe
        merged_df.attrs = first_df.attrs
    else:
        merged_df = first_df

    # Write merged GFF
    save_gff(output_path, merged_df)


def filter_to_chromosome(
    input_path: str, output_path: str, chromosome_id: str, species_id: str | None = None
) -> None:
    """Filter GFF file to include only entries from a specific chromosome.

    Parameters
    ----------
    input_path : str
        Path to input GFF file
    output_path : str
        Path to output GFF file
    chromosome_id : str
        Chromosome ID to filter for
    species_id : str, optional
        Species ID to use for chromosome mapping
    """
    logger.info(
        f"Filtering {input_path} to chromosome {chromosome_id}"
        + (f" with species mapping for {species_id}" if species_id else "")
    )

    # Find the seq_ids to filter for based on species config if provided
    seq_ids_to_filter = [chromosome_id]  # Default if no species_id provided
    normalized_id = chromosome_id  # Default if no species_id provided
    attributes_to_drop = None

    if species_id:
        if species_id not in SPECIES_CONFIGS:
            raise ValueError(
                f"Unknown species ID: {species_id}. Available species: {list(SPECIES_CONFIGS.keys())}"
            )

        # Look up the chromosome mapping for this species
        species_config = SPECIES_CONFIGS[species_id]

        # Find all source chromosome IDs that map to our target normalized ID
        source_ids = [
            source_id
            for source_id, norm_id in species_config.chromosome_map.items()
            if norm_id == chromosome_id
        ]

        if not source_ids:
            raise ValueError(
                f"No source chromosome IDs found for normalized ID '{chromosome_id}' in species '{species_id}'"
            )

        attributes_to_drop = species_config.gff.attributes_to_drop
        seq_ids_to_filter = source_ids
        normalized_id = chromosome_id

        logger.info(f"Using source chromosome IDs: {seq_ids_to_filter}")
        logger.info(
            f"Chromosome ID will be normalized from {chromosome_id} to {normalized_id}"
        )
        logger.info(
            f"The following GFF attributes will be dropped: {attributes_to_drop}"
        )

    # Read GFF file
    features = load_gff(input_path, attributes_to_drop=attributes_to_drop)
    original_count = features.shape[0]

    # Filter to specified chromosome(s)
    features = features[features["seq_id"].isin(seq_ids_to_filter)]
    filtered_count = features.shape[0]

    # Normalize seq_id if using species mapping
    if species_id and filtered_count > 0:
        features["seq_id"] = normalized_id
        logger.info(f"Normalized seq_id to {normalized_id}")

    # Write filtered GFF
    save_gff(output_path, features)
    logger.info(f"Filter complete: {filtered_count}/{original_count} records retained")


def filter_to_strand(input_path: str, output_path: str, strand: str) -> None:
    """Filter GFF file to include only entries from a specific strand.

    Parameters
    ----------
    input_path : str
        Path to input GFF file
    output_path : str
        Path to output GFF file
    strand : str
        Strand to filter for ("positive", "negative", or "both")
    """
    if strand not in ["positive", "negative", "both"]:
        raise ValueError(
            f"Strand must be either 'positive', 'negative', or 'both', got {strand}"
        )

    strand_value = (
        "+" if strand == "positive" else "-" if strand == "negative" else None
    )
    logger.info(
        f"Filtering {input_path} to {strand} strand{'' if strand != 'both' else 's'}"
    )

    # Read GFF file
    features = load_gff(input_path)

    # Filter to specified strand
    original_count = features.shape[0]
    if strand == "both":
        # Keep only features with + or - strand
        features = features[features["strand"].isin(["+", "-"])]
    else:
        features = features[features["strand"] == strand_value]
    filtered_count = features.shape[0]

    # Write filtered GFF
    save_gff(output_path, features)
    logger.info(f"Filter complete: {filtered_count}/{original_count} records retained")


def set_source(input_path: str, output_path: str, source: str) -> None:
    """Set the source field for all records in a GFF file.

    Parameters
    ----------
    input_path : str
        Path to input GFF file
    output_path : str
        Path to output GFF file
    source : str
        Source value to set
    """
    logger.info(f"Setting source to '{source}' for all records in {input_path}")

    # Read GFF file
    features = load_gff(input_path)

    # Update source field
    original_sources = features["source"].unique()
    logger.info(f"Original sources: {original_sources}")
    features["source"] = source

    # Write updated GFF
    save_gff(output_path, features)
    logger.info(f"Source field updated for {features.shape[0]} records")


def filter_to_min_gene_length(
    input_path: str, output_path: str, min_length: int
) -> None:
    """Filter GFF file to remove genes shorter than minimum length and their children.

    Parameters
    ----------
    input_path : str
        Path to input GFF file
    output_path : str
        Path to output GFF file
    min_length : int
        Minimum gene length to retain
    """
    logger.info(f"Filtering {input_path} to remove genes shorter than {min_length} bp")

    # Read GFF file
    features = load_gff(input_path)

    original_count = features.shape[0]

    # Filter to only gene features for length checking
    genes = features[features["type"] == GffFeatureType.GENE.value]
    logger.info(f"Found {len(genes)} gene features to check for length")

    # Calculate gene lengths
    gene_lengths = genes["end"] - genes["start"] + 1

    # Identify genes to remove based on length
    too_short_mask = gene_lengths < min_length
    too_short_ids = set(genes.loc[too_short_mask, "id"].dropna())
    logger.info(f"Found {len(too_short_ids)} genes shorter than {min_length} bp")

    # Remove features and their children
    features, removed_count, indirect_count = remove_features_by_id(
        features, too_short_ids
    )

    logger.info(f"Removing {len(too_short_ids)} genes directly (too short)")
    logger.info(
        f"Removing {indirect_count} features indirectly (parent gene too short)"
    )
    logger.info(f"Total features removed: {removed_count}")

    # Write filtered GFF
    save_gff(output_path, features)
    logger.info(
        f"Filter complete: {features.shape[0]}/{original_count} records retained"
    )


def filter_to_min_feature_length(
    input_path: str, output_path: str, feature_types: list[str], min_length: int
) -> None:
    """Filter GFF file to remove small features of specified types and update gene/mRNA boundaries.

    Parameters
    ----------
    input_path : str
        Path to input GFF file
    output_path : str
        Path to output GFF file
    feature_types : list[str]
        List of feature types to filter by length
    min_length : int
        Minimum feature length to retain
    """
    logger.info(
        f"Filtering {input_path} to remove {feature_types} features shorter than {min_length} bp"
    )

    # Read GFF file
    features = load_gff(input_path)
    original_count = features.shape[0]

    # Validate feature types against schema
    # pyrefly: ignore  # no-matching-overload
    valid_types = set(GffFeatureType)
    requested_types = set(feature_types)
    invalid_types = requested_types - valid_types

    if invalid_types:
        logger.error(f"Invalid feature types requested: {sorted(invalid_types)}")
        logger.error(f"Valid feature types: {sorted(valid_types)}")
        raise ValueError(
            f"Feature types {sorted(invalid_types)} are not valid GFF feature types"
        )

    logger.info(f"Validated feature types: {sorted(requested_types)} (all valid)")

    # Calculate feature lengths
    features["length"] = features["end"] - features["start"] + 1

    # Find features to remove (small features of specified types)
    small_features_mask = features["type"].isin(feature_types) & (
        features["length"] < min_length
    )
    small_feature_ids = set(features.loc[small_features_mask, "id"].dropna())

    logger.info(f"Found {len(small_feature_ids)} small features to remove")

    # Remove small features
    features_filtered = features[~small_features_mask]

    # Group features by gene and update boundaries
    genes = features_filtered[features_filtered["type"] == GffFeatureType.GENE.value]
    mrnas = features_filtered[features_filtered["type"] == GffFeatureType.MRNA.value]

    # Track statistics for boundary updates
    genes_updated = 0
    transcripts_updated = 0
    total_genes = len(genes)
    total_transcripts = len(mrnas)

    for _, gene in genes.iterrows():
        gene_id = gene["id"]
        original_gene_start = gene["start"]
        original_gene_end = gene["end"]

        # Find all features belonging to this gene (direct children and grandchildren via mRNAs)
        gene_children = features_filtered[features_filtered["parent"] == gene_id]
        gene_mrna_ids = gene_children[
            gene_children["type"] == GffFeatureType.MRNA.value
        ]["id"].tolist()

        # First, update mRNA boundaries for this gene
        for mrna_id in gene_mrna_ids:
            mrna_features = features_filtered[features_filtered["parent"] == mrna_id]
            if not mrna_features.empty:
                original_mrna = features_filtered[
                    features_filtered["id"] == mrna_id
                ].iloc[0]
                original_mrna_start = original_mrna["start"]
                original_mrna_end = original_mrna["end"]

                mrna_start = mrna_features["start"].min()
                mrna_end = mrna_features["end"].max()

                # Update mRNA boundaries and track if changed
                if mrna_start != original_mrna_start or mrna_end != original_mrna_end:
                    transcripts_updated += 1
                    features_filtered.loc[
                        features_filtered["id"] == mrna_id, "start"
                    ] = mrna_start
                    features_filtered.loc[features_filtered["id"] == mrna_id, "end"] = (
                        mrna_end
                    )

        # Now update gene boundaries based on updated mRNA boundaries
        updated_gene_children = features_filtered[
            features_filtered["parent"] == gene_id
        ]
        updated_mrnas = updated_gene_children[
            updated_gene_children["type"] == GffFeatureType.MRNA.value
        ]

        if not updated_mrnas.empty:
            new_start = updated_mrnas["start"].min()
            new_end = updated_mrnas["end"].max()

            # Update gene boundaries and track if changed
            if new_start != original_gene_start or new_end != original_gene_end:
                genes_updated += 1
                features_filtered.loc[features_filtered["id"] == gene_id, "start"] = (
                    new_start
                )
                features_filtered.loc[features_filtered["id"] == gene_id, "end"] = (
                    new_end
                )

    # Remove temporary length column
    features_filtered = features_filtered.drop(columns=["length"])

    removed_count = original_count - len(features_filtered)

    # Log statistics
    logger.info("Boundary updates:")
    genes_percentage = (genes_updated / total_genes * 100) if total_genes > 0 else 0
    transcripts_percentage = (
        (transcripts_updated / total_transcripts * 100) if total_transcripts > 0 else 0
    )
    logger.info(
        f"  - Genes: {genes_updated}/{total_genes} ({genes_percentage:.1f}%) had boundaries updated"
    )
    logger.info(
        f"  - Transcripts: {transcripts_updated}/{total_transcripts} ({transcripts_percentage:.1f}%) had boundaries updated"
    )

    # Write filtered GFF
    save_gff(output_path, features_filtered)
    logger.info(
        f"Filter complete: {removed_count} features removed, {len(features_filtered)}/{original_count} records retained"
    )


def filter_to_representative_transcripts(
    input_path: str, output_path: str, mode: str
) -> None:
    """Filter GFF file to keep only representative transcripts for each gene.

    Parameters
    ----------
    input_path : str
        Path to input GFF file
    output_path : str
        Path to output GFF file
    mode : str
        Selection mode: "longest", "annotated", or "annotated_or_longest"
        - "longest": Always select the longest transcript for each gene
        - "annotated": Use the canonical_transcript=1 annotation, error if column missing
        - "annotated_or_longest": Use canonical_transcript if the column exists, otherwise use longest
    """
    if mode not in ["longest", "annotated", "annotated_or_longest"]:
        raise ValueError(
            f"Mode must be either 'longest', 'annotated', or 'annotated_or_longest', got {mode}"
        )

    logger.info(
        f"Filtering {input_path} to keep only representative transcripts using mode: {mode}"
    )

    # Read GFF file
    features = load_gff(input_path)

    original_count = features.shape[0]

    # Check if the canonical_transcript column exists when using annotated modes
    has_canonical_column = "canonical_transcript" in features.columns

    if mode == "annotated" and not has_canonical_column:
        error_msg = "Cannot use 'annotated' mode: 'canonical_transcript' column not found in features"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # If using annotated_or_longest and the column doesn't exist, fall back to longest
    effective_mode = mode
    if mode == "annotated_or_longest":
        if has_canonical_column:
            logger.info("Using 'annotated' mode (canonical_transcript column found)")
            effective_mode = "annotated"
        else:
            logger.info(
                "Falling back to 'longest' mode (canonical_transcript column not found)"
            )
            effective_mode = "longest"

    # Calculate feature lengths
    features["length"] = features["end"] - features["start"] + 1

    # Find all genes and their mRNA children
    genes = features[features["type"] == GffFeatureType.GENE.value]
    logger.info(f"Found {len(genes)} gene features")

    # Get gene IDs
    gene_ids = set(genes["id"].dropna())

    # Find all mRNAs
    mrnas = features[features["type"] == GffFeatureType.MRNA.value]
    logger.info(f"Found {len(mrnas)} mRNA features")

    # Track which mRNAs to keep and how they were selected
    mrnas_to_keep = set()
    selection_stats = {"annotation": 0, "length": 0}

    # For each gene, select the representative mRNA
    for gene_id in gene_ids:
        # Find all mRNAs with this gene as parent
        gene_mrnas = mrnas[mrnas["parent"] == gene_id]

        if gene_mrnas.empty:
            logger.warning(f"Gene {gene_id} has no mRNA children")
            continue

        if effective_mode == "annotated":
            # Check for proper canonical_transcript values
            canonical_values = features["canonical_transcript"].dropna().unique()
            non_valid = [v for v in canonical_values if v not in ["1", ""]]
            if non_valid:
                raise ValueError(
                    f"Found invalid canonical_transcript values: {non_valid}. Expected only '1' or ''."
                )

            # Try to find canonical transcript
            canonical_mrnas = gene_mrnas[gene_mrnas["canonical_transcript"] == "1"]

            if not canonical_mrnas.empty:
                # If multiple canonical transcripts (shouldn't happen but just in case), take the longest
                if len(canonical_mrnas) > 1:
                    logger.warning(
                        f"Gene {gene_id} has multiple canonical transcripts, selecting the longest"
                    )
                    selected_mrna = canonical_mrnas.loc[
                        canonical_mrnas["length"].idxmax()
                    ]
                else:
                    selected_mrna = canonical_mrnas.iloc[0]

                selection_stats["annotation"] += 1
            else:
                # No canonical transcript, fall back to longest
                selected_mrna = gene_mrnas.loc[gene_mrnas["length"].idxmax()]
                selection_stats["length"] += 1
        else:
            # For "longest" mode, always select the longest mRNA
            selected_mrna = gene_mrnas.loc[gene_mrnas["length"].idxmax()]
            selection_stats["length"] += 1

        # Add selected mRNA to keep set
        mrnas_to_keep.add(selected_mrna["id"])

    # Log selection statistics
    total_selected = selection_stats["annotation"] + selection_stats["length"]
    logger.info(f"Selected {total_selected} representative mRNAs:")
    logger.info(
        f"  - {selection_stats['annotation']} ({selection_stats['annotation'] / total_selected * 100:.1f}%) selected by annotation"
    )
    logger.info(
        f"  - {selection_stats['length']} ({selection_stats['length'] / total_selected * 100:.1f}%) selected by length"
    )

    # Find mRNAs to remove (those not in the keep set)
    mrnas_to_remove = set(mrnas["id"].dropna()) - mrnas_to_keep
    logger.info(f"Removing {len(mrnas_to_remove)} non-representative mRNAs")

    # Remove non-representative mRNAs and their children
    features, removed_count, indirect_count = remove_features_by_id(
        features, mrnas_to_remove
    )

    logger.info(f"Removed {len(mrnas_to_remove)} non-representative mRNAs")
    logger.info(f"Removed {indirect_count} children of non-representative mRNAs")
    logger.info(f"Total features removed: {removed_count}")

    # Remove the temporary length column
    features = features.drop(columns=["length"])

    # Write filtered GFF
    save_gff(output_path, features)
    logger.info(
        f"Filter complete: {features.shape[0]}/{original_count} records retained"
    )


def filter_to_valid_genes(
    input_path: str, output_path: str, require_utrs: bool = True
) -> None:
    """Filter GFF file to remove mRNAs without required features and genes without valid mRNAs.

    Parameters
    ----------
    input_path : str
        Path to input GFF file
    output_path : str
        Path to output GFF file
    require_utrs : bool, default True
        If True, require mRNAs to have five_prime_UTR, CDS, and three_prime_UTR.
        If False, only require CDS.
    """
    logger.info(f"Filtering {input_path} to keep only valid transcripts and genes")
    if require_utrs:
        logger.info(
            "Requiring five_prime_UTR, CDS, and three_prime_UTR for valid transcripts"
        )
    else:
        logger.info("Requiring only CDS for valid transcripts")

    # Read GFF file
    features = load_gff(input_path)
    original_count = features.shape[0]

    # Find all mRNAs
    mrnas = features[features["type"] == GffFeatureType.MRNA.value]
    logger.info(f"Found {len(mrnas)} mRNA features")

    # Create index of parent -> children
    features_by_parent = features.set_index("parent")

    # Identify which mRNAs have the required features
    valid_mrnas = set()
    for mrna_id in mrnas["id"].dropna():
        # Check if mRNA has any children
        if mrna_id in features_by_parent.index:
            # Get all children of this mRNA using the index
            children = features_by_parent.loc[[mrna_id]]

            # Check if it has all required feature types
            has_cds = (children["type"] == GffFeatureType.CDS.value).any()

            if require_utrs:
                has_five_prime = (
                    children["type"] == GffFeatureType.FIVE_PRIME_UTR.value
                ).any()
                has_three_prime = (
                    children["type"] == GffFeatureType.THREE_PRIME_UTR.value
                ).any()

                if has_five_prime and has_cds and has_three_prime:
                    valid_mrnas.add(mrna_id)
            else:
                if has_cds:
                    valid_mrnas.add(mrna_id)

    # Identify invalid mRNAs
    invalid_mrnas = set(mrnas["id"].dropna()) - valid_mrnas
    logger.info(
        f"Found {len(valid_mrnas)} valid mRNAs and {len(invalid_mrnas)} invalid mRNAs"
    )

    # Remove invalid mRNAs and their children
    if invalid_mrnas:
        features, mrna_removed_count, mrna_indirect_count = remove_features_by_id(
            features, invalid_mrnas
        )
        logger.info(
            f"Removed {len(invalid_mrnas)} invalid mRNAs, {mrna_indirect_count} child features, {mrna_removed_count} total features"
        )

    # Find all genes
    genes = features[features["type"] == GffFeatureType.GENE.value]
    logger.info(f"Found {len(genes)} gene features")

    # Identify genes with no valid mRNA children using set operations
    all_gene_ids = set(genes["id"].dropna())
    mrna_parent_ids = set(
        features[(features["type"] == GffFeatureType.MRNA.value)]["parent"].dropna()
    )
    invalid_genes = all_gene_ids - mrna_parent_ids

    logger.info(f"Found {len(invalid_genes)} genes with no valid transcripts")

    # Remove invalid genes and their children
    if invalid_genes:
        features, gene_removed_count, gene_indirect_count = remove_features_by_id(
            features, invalid_genes
        )
        logger.info(
            f"Removed {len(invalid_genes)} invalid genes, {gene_indirect_count} child features, {gene_removed_count} total features"
        )

    # Write filtered GFF
    save_gff(output_path, features)
    logger.info(
        f"Filter complete: {features.shape[0]}/{original_count} records retained"
    )


def remove_exon_utrs(input_path: str, output_path: str) -> None:
    """Remove five_prime_UTR, three_prime_UTR, and exon features from a GFF file.

    Parameters
    ----------
    input_path : str
        Path to input GFF file
    output_path : str
        Path to output GFF file
    """
    logger.info(f"Removing UTR and exon features from {input_path}")

    # Read GFF file
    features = load_gff(input_path)
    original_count = features.shape[0]

    # Find all UTR features (note: "exon" is not in GffFeatureType so we keep it as string)
    utr_types = [
        GffFeatureType.FIVE_PRIME_UTR.value,
        GffFeatureType.THREE_PRIME_UTR.value,
        RegionType.EXON.value,
    ]
    utrs = features[features["type"].isin(utr_types)]

    # Count UTRs by type
    utr_type_counts = utrs["type"].value_counts()

    logger.info(f"Found {len(utrs)} UTR/exon features:")
    for utr_type, count in utr_type_counts.items():
        logger.info(f"  - {count} {utr_type} features")

    # Remove UTR features
    filtered_features = features[~features["type"].isin(utr_types)]
    removed_count = original_count - len(filtered_features)

    # Count features by type after removal
    remaining_type_counts = filtered_features["type"].value_counts()

    logger.info("After removal, feature types remaining:")
    for feature_type, count in remaining_type_counts.items():
        percentage = (count / len(filtered_features)) * 100
        logger.info(f"  - {count} {feature_type} features ({percentage:.1f}%)")

    # Write filtered GFF
    save_gff(output_path, filtered_features)
    logger.info(
        f"UTR/exon removal complete: {removed_count} features removed, {len(filtered_features)}/{original_count} records retained ({len(filtered_features) / original_count * 100:.1f}%)"
    )


def compare_gff_files(
    reference_path: str, input_path: str, output_dir: str, gffcompare_path: str
) -> None:
    """Compare an input GFF file against a reference using gffcompare.

    Parameters
    ----------
    reference_path : str
        Path to reference GFF file
    input_path : str
        Path to input GFF file
    output_dir : str
        Directory to store comparison results
    gffcompare_path : str
        Path to gffcompare executable
    """
    logger.info(f"Comparing input GFF {input_path} against reference {reference_path}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run gffcompare
    output_prefix = run_gffcompare(
        reference_path, input_path, output_dir, gffcompare_path
    )

    # Parse the stats file
    stats_file = f"{output_prefix}.stats"

    if not os.path.exists(stats_file):
        logger.error(f"gffcompare stats file not found: {stats_file}")
        raise FileNotFoundError(
            f"gffcompare did not produce a stats file at {stats_file}"
        )

    # Show stats file content
    with open(stats_file, "r") as f:
        logger.info(f"Stats file content from gffcompare at {stats_file}:")
        logger.info("\n" + f.read())

    # Parse the stats file into dataframes
    stats_df = parse_gffcompare_stats(stats_file)

    # Save the stats dataframes to CSV
    stats_csv = os.path.join(output_dir, "gffcompare.stats.csv")

    def calculate_f1_score(row):
        """Calculate F1 score from precision and sensitivity."""
        precision = row["precision"]
        sensitivity = row["sensitivity"]
        if precision + sensitivity == 0:
            return 0
        return 2 * (precision * sensitivity) / (precision + sensitivity)

    stats_df["f1_score"] = stats_df.apply(calculate_f1_score, axis=1)

    logger.info(f"Parsed stats: \n{stats_df}")

    logger.info(f"Saving parsed stats to {stats_csv}")
    stats_df.to_csv(stats_csv, sep="\t", index=False)

    logger.info("Comparison complete")


def evaluate_gff_files(
    reference_path: str,
    input_path: str,
    output_dir: str,
    edge_tolerance: int = 0,
    ignore_unmatched: bool = True,
    as_percentage: bool = True,
) -> None:
    """Evaluate an input GFF file against a reference using internal evaluation functions.

    Parameters
    ----------
    reference_path : str
        Path to reference GFF file
    input_path : str
        Path to input GFF file
    output_dir : str
        Directory to store evaluation results
    edge_tolerance : int, default 0
        Tolerance to allow for matching transcript ends
    ignore_unmatched : bool, default True
        Whether to ignore contigs that aren't present in both GFF files
    as_percentage : bool, default True
        Whether to display values as percentages (0-100) or decimals (0.0-1.0).
        Defaults to True for consistency with gffcompare.
    """
    logger.info(f"Evaluating input GFF {input_path} against reference {reference_path}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run gffeval
    output_prefix = run_gffeval(
        reference_path,
        input_path,
        output_dir,
        edge_tolerance,
        ignore_unmatched,
        as_percentage,
    )

    # Check that the stats file was created
    stats_file = f"{output_prefix}.stats.tsv"

    if not os.path.exists(stats_file):
        logger.error(f"gffeval stats file not found: {stats_file}")
        raise FileNotFoundError(f"gffeval did not produce a stats file at {stats_file}")

    # Show stats file content
    with open(stats_file, "r") as f:
        logger.info(f"Stats file content from gffeval at {stats_file}:")
        logger.info("\n" + f.read())

    logger.info("Evaluation complete")


def summarize_gff(input_path: str, species_id: str | None = None) -> None:
    """Summarize the distribution of source and type fields in a GFF file.

    Parameters
    ----------
    input_path : str
        Path to input GFF file
    species_id : str, optional
        Species ID to use for getting attributes_to_drop configuration
    """
    logger.info(f"Summarizing GFF file: {input_path}")

    # Get attributes to drop if species_id is provided
    attributes_to_drop = None
    if species_id and species_id in SPECIES_CONFIGS:
        attributes_to_drop = SPECIES_CONFIGS[species_id].gff.attributes_to_drop

    # Read GFF file
    features = load_gff(input_path, attributes_to_drop=attributes_to_drop)

    # Count source field distribution
    source_counts = features["source"].value_counts()

    # Count type field distribution
    type_counts = features["type"].value_counts()

    # Count strand distribution
    strand_counts = features["strand"].value_counts()

    # Count sequence ID distribution
    seq_id_counts = features["seq_id"].value_counts()

    total_records = len(features)

    # Count unique combinations of source, strand, and type
    source_strand_type_combos = features.groupby(["source", "strand", "type"]).size()
    unique_combos_count = len(source_strand_type_combos)

    # Print overall summary
    print("\n=== Overall Summary ===")
    print("-" * 50)
    print(f"Total records: {total_records}")
    print(f"Total feature types: {len(type_counts)}")
    print(f"Unique source/strand/type combinations: {unique_combos_count}")

    # Print the summaries
    print("\n=== Source Distribution ===")
    print(f"{'Source':<30} {'Count':<10} {'Percentage':<10}")
    print("-" * 50)
    for source, count in source_counts.items():
        percentage = (count / total_records) * 100
        print(f"{source:<30} {count:<10} {percentage:.2f}%")

    print("\n=== Feature Type Distribution ===")
    print(f"{'Type':<30} {'Count':<10} {'Percentage':<10}")
    print("-" * 50)
    for type_name, count in type_counts.items():
        percentage = (count / total_records) * 100
        print(f"{type_name:<30} {count:<10} {percentage:.2f}%")

    print("\n=== Strand Distribution ===")
    print(f"{'Strand':<10} {'Count':<10} {'Percentage':<10}")
    print("-" * 50)
    for strand, count in strand_counts.items():
        strand_display = "+" if strand == "+" else "-" if strand == "-" else strand
        percentage = (count / total_records) * 100
        print(f"{strand_display:<10} {count:<10} {percentage:.2f}%")

    print("\n=== Sequence ID Distribution ===")
    print(f"{'Sequence ID':<30} {'Count':<10} {'Percentage':<10}")
    print("-" * 50)
    for seq_id, count in seq_id_counts.items():
        percentage = (count / total_records) * 100
        print(f"{seq_id:<30} {count:<10} {percentage:.2f}%")

    # Display source/strand/type combinations
    print("\n=== Source/Strand/Type Combinations ===")
    print(
        f"{'Source':<20} {'Strand':<10} {'Type':<20} {'Count':<10} {'Percentage':<10}"
    )
    print("-" * 75)

    # Sort combinations by count in descending order
    sorted_combos = source_strand_type_combos.sort_values(ascending=False)

    for (source, strand, type_name), count in sorted_combos.items():
        strand_display = "+" if strand == "+" else "-" if strand == "-" else strand
        percentage = (count / total_records) * 100
        print(
            f"{source:<20} {strand_display:<10} {type_name:<20} {count:<10} {percentage:.2f}%"
        )

    # Calculate transcript distribution per gene
    print("\n=== Transcripts Per Gene Distribution ===")
    print(f"{'Transcript Count':<20} {'Gene Count':<15} {'Percentage':<10}")
    print("-" * 50)

    # Find all genes
    genes = features[features["type"] == GffFeatureType.GENE.value]
    gene_ids = set(genes["id"].dropna())
    total_genes = len(gene_ids)

    # Filter for mRNAs
    mrnas = features[features["type"] == GffFeatureType.MRNA.value]

    # Count mRNAs per gene using groupby
    gene_transcript_counts = mrnas.groupby("parent").size()

    # Count how many genes have each transcript count
    transcript_distribution = gene_transcript_counts.value_counts().sort_index()

    # Handle genes with 0 transcripts
    genes_with_transcripts = set(gene_transcript_counts.index)
    genes_without_transcripts = gene_ids - genes_with_transcripts
    transcript_counts = dict(transcript_distribution)
    transcript_counts[0] = len(genes_without_transcripts)

    # Sort by transcript count and display
    if total_genes == 0:
        logger.warning(
            "No gene features found in GFF file. Skipping transcript per gene distribution."
        )
        print("No gene features found - skipping transcript distribution analysis")
    else:
        for transcript_count, gene_count in sorted(transcript_counts.items()):
            percentage = (gene_count / total_genes) * 100
            print(f"{transcript_count:<20} {gene_count:<15} {percentage:.2f}%")

    # Calculate fraction of genes with a canonical transcript
    print("\n=== Canonical Transcript Annotation ===")

    # Check if canonical_transcript column exists
    has_canonical_column = "canonical_transcript" in mrnas.columns

    if not has_canonical_column:
        print("\nNo canonical_transcript column found in the GFF file.")
        print("Skipping canonical transcript analysis.")
    else:
        # Get counts of all values for canonical_transcript
        canonical_values = mrnas["canonical_transcript"].fillna("").value_counts()

        # Print distribution of canonical_transcript values
        print("\nCanonical transcript value distribution:")
        print(f"{'Value':<20} {'Count':<10} {'Percentage':<10}")
        print("-" * 50)

        # Flag for unexpected values
        has_unexpected_values = False
        expected_values = {"1", ""}

        # Print all values and mark unexpected ones
        for value, count in canonical_values.items():
            percentage = (count / len(mrnas)) * 100
            # Mark unexpected values with '!!!'
            marker = " !!! UNEXPECTED VALUE !!!" if value not in expected_values else ""
            if marker:
                has_unexpected_values = True
            print(
                f"{value if value else '<empty>':<20} {count:<10} {percentage:.2f}%{marker}"
            )

        if has_unexpected_values:
            print("\nWARNING: Unexpected canonical_transcript values detected!")
            print("Expected values are '1' or empty/missing.")

        # Find mRNAs with canonical_transcript="1"
        canonical_mrnas = mrnas[mrnas["canonical_transcript"] == "1"]

        # Get the set of genes that have at least one canonical transcript
        genes_with_canonical = set(canonical_mrnas["parent"].unique())
        genes_with_canonical_count = len(genes_with_canonical)

        # Calculate percentage
        canonical_percentage = (
            (genes_with_canonical_count / total_genes) * 100 if total_genes > 0 else 0
        )

        print(
            f"\nGenes with canonical transcript: {genes_with_canonical_count}/{total_genes} ({canonical_percentage:.2f}%)"
        )

    # Calculate type combinations per gene
    print("\n=== Feature Type Combinations Per Transcript ===")
    print(f"{'Type Combination':<50} {'Transcript Count':<15} {'Percentage':<10}")
    print("-" * 75)

    # Get all mRNA IDs
    mrna_ids = set(
        features[features["type"] == GffFeatureType.MRNA.value]["id"].dropna()
    )

    # Filter to features that are children of mRNAs and not mRNAs themselves
    transcript_features = features[
        (features["parent"].isin(mrna_ids))
        & (features["type"] != GffFeatureType.MRNA.value)
    ]

    # Group by parent (transcript) and get unique types, then count combinations
    type_combinations = (
        transcript_features.groupby("parent")["type"]
        .unique()
        .apply(frozenset)
        .value_counts()
    )

    # Display results
    total_transcripts = len(mrna_ids)
    for combo, count in type_combinations.items():
        percentage = (count / total_transcripts) * 100
        combo_str = ", ".join(sorted(combo))
        print(f"{combo_str:<50} {count:<15} {percentage:.2f}%")


def collect_results(input_dir: str, output_path: str | None = None) -> None:
    """Collect and consolidate gffcompare.stats.csv and gffeval.stats.tsv files from subdirectories.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing subdirectories with stats files
    output_path : str, optional
        Path to save the consolidated results file. If None, defaults to
        {input_dir}/consolidated.stats.csv
    """
    logger.info(f"Collecting evaluation results from subdirectories of {input_dir}")

    # Find all stats files in subdirectories
    if output_path is None:
        output_path = os.path.join(input_dir, "consolidated.stats.csv")

    gffcompare_files = glob.glob(os.path.join(input_dir, "*/gffcompare.stats.csv"))
    gffeval_files = glob.glob(os.path.join(input_dir, "*/gffeval.stats.tsv"))

    all_files = gffcompare_files + gffeval_files

    if not all_files:
        logger.error(
            f"No gffcompare.stats.csv or gffeval.stats.tsv files found in subdirectories of {input_dir}"
        )
        return

    logger.info(
        f"Found {len(gffcompare_files)} gffcompare stats files and {len(gffeval_files)} gffeval stats files"
    )

    # Collect and process each file
    dfs = []
    for file_path in all_files:
        # Extract the directory name as source and determine tool
        source = os.path.basename(os.path.dirname(file_path))
        tool = (
            "gffcompare" if "gffcompare" in os.path.basename(file_path) else "gffeval"
        )
        logger.info(f"Processing {file_path} (source: {source}, tool: {tool})")

        # Read the file
        df = pd.read_csv(file_path, sep="\t")

        # Normalize column names for consistency between tools
        if tool == "gffeval":
            df = df.rename(columns={"recall": "sensitivity", "f1": "f1_score"})

        # Add source and tool columns
        df["source"] = source
        df["tool"] = tool

        # Ensure all dataframes have exactly the expected columns
        expected_columns = [
            "level",
            "sensitivity",
            "precision",
            "f1_score",
            "source",
            "tool",
        ]
        df = df[expected_columns]

        dfs.append(df)

    # Concatenate all dataframes
    if not dfs:
        logger.error("No valid data found to consolidate")
        return

    consolidated_df = pd.concat(dfs, ignore_index=True)

    # Sort by tool, source, and level
    consolidated_df = consolidated_df.sort_values(["tool", "source", "level"])

    # Print the consolidated dataframe
    print(consolidated_df.to_string(index=False))

    # Save to file
    logger.info(f"Saving consolidated results to {output_path}")
    consolidated_df.to_csv(output_path, sep="\t", index=False)
    logger.info("Results saved successfully")


def clean_gff(
    input_path: str,
    output_path: str,
    sort_by_start: bool = True,
    remove_invalid_interval: bool = True,
) -> None:
    """Clean a GFF file by sorting and/or removing invalid intervals.

    Parameters
    ----------
    input_path : str
        Path to input GFF file
    output_path : str
        Path to output GFF file
    sort_by_start : bool, default True
        Whether to sort records by [seq_id, start, end, type, strand] ascending
    remove_invalid_interval : bool, default True
        Whether to remove records where end < start
    """
    logger.info(f"Cleaning GFF file: {input_path}")
    logger.info(f"Sort by start: {sort_by_start}")
    logger.info(f"Remove invalid intervals: {remove_invalid_interval}")

    # Load the GFF file
    df = load_gff(input_path)
    original_count = df.shape[0]
    logger.info(f"Original record count: {original_count}")

    # Track what we're doing
    operations = []
    if sort_by_start:
        operations.append("sorting")
    if remove_invalid_interval:
        operations.append("removing invalid intervals")

    if operations:
        logger.info(f"Operations to perform: {', '.join(operations)}")
    else:
        logger.info("No operations requested, copying file as-is")

    # Remove invalid intervals if requested
    invalid_df = pd.DataFrame()  # Initialize empty dataframe
    if remove_invalid_interval:
        logger.info("Checking for invalid intervals (where end < start)...")

        # Find invalid intervals
        invalid_mask = df["end"] < df["start"]
        invalid_df = df[invalid_mask]

        if invalid_df.shape[0] > 0:
            # Display core fields for invalid records
            with pd.option_context("display.max_rows", 100):
                invalid_sample = invalid_df[
                    ["seq_id", "type", "start", "end", "strand", "id", "parent"]
                ]
                logger.warning(
                    f"Found {invalid_df.shape[0]} invalid interval(s) (first 100):\n{invalid_sample}"
                )

            # Remove invalid records
            df = df[~invalid_mask]
            final_count = df.shape[0]
            removed_count = original_count - final_count
            removal_percentage = (removed_count / original_count) * 100

            logger.info(f"Removed {removed_count} invalid interval(s)")
            logger.info(f"Remaining records: {final_count}")
            logger.info(f"Removal percentage: {removal_percentage:.2f}%")
        else:
            logger.info("No invalid intervals found")

    # Sort if requested
    if sort_by_start:
        logger.info(
            "Sorting records by [seq_id, start, end, type, strand] ascending..."
        )
        df = df.sort_values(
            ["seq_id", "start", "end", "type", "strand"], ascending=True
        )
        logger.info("Sorting complete")

    # Save the cleaned GFF
    logger.info(f"Saving cleaned GFF to {output_path}")
    save_gff(output_path, df)

    # Final summary
    final_count = df.shape[0]
    logger.info("Cleaning complete")
    logger.info(f"Final record count: {final_count}")
    if remove_invalid_interval and invalid_df.shape[0] > 0:
        logger.info(f"Total records removed: {original_count - final_count}")
        logger.info(
            f"Total removal percentage: {((original_count - final_count) / original_count) * 100:.2f}%"
        )


def filter_to_pc_quality_score_pass(
    input_path: str | None = None,
    output_path: str | None = None,
    input_dir: str | None = None,
    output_dir: str | None = None,
    species_ids: list[str] | None = None,
) -> None:
    """Filter GFF file to remove genes and transcripts with passPlantCADFilter=0.

    This function supports two interfaces:
    1. Direct file paths (for evaluation scripts)
    2. Directory + species_ids (for data preparation scripts)

    Parameters
    ----------
    input_path : str, optional
        Direct path to input GFF file
    output_path : str, optional
        Direct path to output GFF file
    input_dir : str, optional
        Directory containing input GFF files
    output_dir : str, optional
        Directory to write filtered GFF files
    species_ids : list[str], optional
        List of species IDs to determine filenames using species configuration
    """
    import os

    # Determine which interface is being used
    if input_path is not None and output_path is not None:
        # Direct file path interface
        logger.info("Using direct file path interface")
        _apply_pc_quality_filter_to_file(input_path, output_path)

    elif input_dir is not None and output_dir is not None and species_ids is not None:
        # Species configuration interface
        from src.config import get_species_config

        logger.info(
            f"Using species configuration interface for {len(species_ids)} species: {species_ids}"
        )

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Process each species
        for species_id in species_ids:
            logger.info(f"Processing species: {species_id}")

            # Get species configuration
            config = get_species_config(species_id)

            # Determine input and output paths using species configuration
            # Input: from gff_tagged directory using species config filename
            input_filename = config.gff.filename
            final_input_path = os.path.join(input_dir, input_filename)

            # Output: to gff_filtered directory using species config filename
            output_filename = config.gff.filename
            final_output_path = os.path.join(output_dir, output_filename)

            logger.info(f"  Input: {final_input_path}")
            logger.info(f"  Output: {final_output_path}")

            # Apply PC quality filter to this species
            _apply_pc_quality_filter_to_file(final_input_path, final_output_path)

        logger.info(
            f"Completed PC quality filtering for all {len(species_ids)} species"
        )

    else:
        raise ValueError(
            "Must provide either (input_path, output_path) for direct file interface "
            "or (input_dir, output_dir, species_ids) for species configuration interface"
        )


def _apply_pc_quality_filter_to_file(input_path: str, output_path: str) -> None:
    """Apply PC quality filter to a single GFF file.

    Parameters
    ----------
    input_path : str
        Path to input GFF file
    output_path : str
        Path to output GFF file
    """
    logger.info(
        f"Filtering {input_path} to keep only features with passPlantCADFilter=1"
    )
    logger.info(f"Output will be saved to {output_path}")

    # Read GFF file
    features = load_gff(input_path)
    original_count = features.shape[0]

    # Check if passPlantCADFilter column exists
    if "passplantcadfilter" not in features.columns:
        raise ValueError(
            "passPlantCADFilter column not found in GFF file. "
            "Expected column name 'passplantcadfilter' (case-insensitive)."
        )

    # Log initial statistics
    pc_filter_values = features["passplantcadfilter"].value_counts()
    logger.info("Initial passPlantCADFilter distribution:")
    for value, count in pc_filter_values.items():
        percentage = (count / original_count) * 100
        logger.info(f"  - passPlantCADFilter={value}: {count} ({percentage:.2f}%)")

    # Step 1: Remove genes where passPlantCADFilter=0
    genes = features[features["type"] == GffFeatureType.GENE.value]
    genes_with_filter = genes[genes["passplantcadfilter"].notna()]

    if genes_with_filter.empty:
        logger.warning("No gene features found with passPlantCADFilter annotations")
        genes_to_remove = set()
    else:
        bad_genes = genes_with_filter[genes_with_filter["passplantcadfilter"] == "0"]
        genes_to_remove = set(bad_genes["id"].dropna())
        logger.info(f"Found {len(genes_to_remove)} genes with passPlantCADFilter=0")

    # Step 2: Remove transcripts where passPlantCADFilter=0
    mrnas = features[features["type"] == GffFeatureType.MRNA.value]
    mrnas_with_filter = mrnas[mrnas["passplantcadfilter"].notna()]

    if mrnas_with_filter.empty:
        logger.warning("No mRNA features found with passPlantCADFilter annotations")
        transcripts_to_remove = set()
    else:
        bad_transcripts = mrnas_with_filter[
            mrnas_with_filter["passplantcadfilter"] == "0"
        ]
        transcripts_to_remove = set(bad_transcripts["id"].dropna())
        logger.info(
            f"Found {len(transcripts_to_remove)} transcripts with passPlantCADFilter=0"
        )

    # Combine all features to remove initially
    features_to_remove = genes_to_remove | transcripts_to_remove

    # Step 3: Remove features and handle cascading removals
    if features_to_remove:
        features, removed_count, indirect_count = remove_features_by_id(
            features, features_to_remove
        )

        logger.info(
            f"Removed {len(genes_to_remove)} genes directly (passPlantCADFilter=0)"
        )
        logger.info(
            f"Removed {len(transcripts_to_remove)} transcripts directly (passPlantCADFilter=0)"
        )
        logger.info(f"Removed {indirect_count} features indirectly (cascading removal)")
        logger.info(f"Total features removed: {removed_count}")
    else:
        logger.info("No features to remove based on passPlantCADFilter")

    # Step 4: Additional cascading cleanup for orphaned genes
    # Check for genes that now have no mRNA children
    remaining_genes = features[features["type"] == GffFeatureType.GENE.value]
    remaining_mrnas = features[features["type"] == GffFeatureType.MRNA.value]

    # Get gene IDs that have mRNA children
    genes_with_mrnas = set(remaining_mrnas["parent"].dropna())

    # Find genes without any mRNA children
    all_gene_ids = set(remaining_genes["id"].dropna())
    orphaned_genes = all_gene_ids - genes_with_mrnas

    if orphaned_genes:
        logger.info(
            f"Found {len(orphaned_genes)} orphaned genes (no valid transcripts)"
        )

        # Remove orphaned genes
        features, orphan_removed_count, orphan_indirect_count = remove_features_by_id(
            features, orphaned_genes
        )

        logger.info(f"Removed {len(orphaned_genes)} orphaned genes")
        logger.info(
            f"Removed {orphan_indirect_count} features indirectly (orphaned gene cleanup)"
        )
        logger.info(f"Additional features removed: {orphan_removed_count}")

    # Write filtered GFF
    save_gff(output_path, features)

    final_count = features.shape[0]
    total_removed = original_count - final_count
    removal_percentage = (total_removed / original_count) * 100

    logger.info("PC quality filter complete:")
    logger.info(f"  - Original features: {original_count}")
    logger.info(f"  - Final features: {final_count}")
    logger.info(f"  - Total removed: {total_removed} ({removal_percentage:.2f}%)")

    # Log final feature type distribution
    final_type_counts = features["type"].value_counts()
    logger.info("Final feature type distribution:")
    for feature_type, count in final_type_counts.items():
        percentage = (count / final_count) * 100 if final_count > 0 else 0
        logger.info(f"  - {feature_type}: {count} ({percentage:.2f}%)")


def main() -> None:
    """Parse command line arguments and execute the appropriate function."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Manipulate GFF files")
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Command to execute"
    )

    # Merge command
    merge_parser = subparsers.add_parser(
        "merge", help="Merge multiple GFF files into one"
    )
    merge_parser.add_argument(
        "--input", required=True, nargs="+", help="Input GFF files"
    )
    merge_parser.add_argument("--output", required=True, help="Output GFF file")

    # Resolve command
    resolve_parser = subparsers.add_parser(
        "resolve",
        help="Resolve and copy a GFF file for a species using its configuration",
    )
    resolve_parser.add_argument(
        "--input-dir", required=True, help="Directory containing input GFF files"
    )
    resolve_parser.add_argument(
        "--species-id", required=True, help="Species ID to resolve the GFF file for"
    )
    resolve_parser.add_argument(
        "--output", required=True, help="Output path to copy the resolved GFF file to"
    )

    # Filter to chromosome command
    filter_parser = subparsers.add_parser(
        "filter_to_chromosome",
        help="Filter GFF file to include only entries from a specific chromosome",
    )
    filter_parser.add_argument("--input", required=True, help="Input GFF file")
    filter_parser.add_argument("--output", required=True, help="Output GFF file")
    filter_parser.add_argument(
        "--chromosome-id", required=True, help="Chromosome ID to filter for"
    )
    filter_parser.add_argument(
        "--species-id", help="Species ID to use for chromosome mapping"
    )

    # Filter to strand command
    strand_parser = subparsers.add_parser(
        "filter_to_strand",
        help="Filter GFF file to include only entries from a specific strand",
    )
    strand_parser.add_argument("--input", required=True, help="Input GFF file")
    strand_parser.add_argument("--output", required=True, help="Output GFF file")
    strand_parser.add_argument(
        "--strand",
        required=True,
        choices=["positive", "negative", "both"],
        help="Strand to filter for",
    )

    # Set source command
    source_parser = subparsers.add_parser(
        "set_source", help="Set the source field for all records in a GFF file"
    )
    source_parser.add_argument("--input", required=True, help="Input GFF file")
    source_parser.add_argument("--output", required=True, help="Output GFF file")
    source_parser.add_argument("--source", required=True, help="Source value to set")

    # Filter to minimum length command
    length_parser = subparsers.add_parser(
        "filter_to_min_gene_length",
        help="Filter GFF file to remove genes shorter than minimum length and their children",
    )
    length_parser.add_argument("--input", required=True, help="Input GFF file")
    length_parser.add_argument("--output", required=True, help="Output GFF file")
    length_parser.add_argument(
        "--min-length", required=True, type=int, help="Minimum gene length to retain"
    )

    # Filter small features command
    small_features_parser = subparsers.add_parser(
        "filter_to_min_feature_length",
        help="Filter GFF file to remove small features of specified types and update gene/mRNA boundaries",
    )
    small_features_parser.add_argument("--input", required=True, help="Input GFF file")
    small_features_parser.add_argument(
        "--output", required=True, help="Output GFF file"
    )
    small_features_parser.add_argument(
        "--feature-types",
        required=True,
        help="Comma-separated list of feature types to filter by length",
    )
    small_features_parser.add_argument(
        "--min-length", required=True, type=int, help="Minimum feature length to retain"
    )

    # Filter to representative transcripts command
    rep_parser = subparsers.add_parser(
        "filter_to_representative_transcripts",
        help="Filter GFF file to keep only one representative transcript for each gene",
    )
    rep_parser.add_argument("--input", required=True, help="Input GFF file")
    rep_parser.add_argument("--output", required=True, help="Output GFF file")
    rep_parser.add_argument(
        "--mode",
        required=True,
        choices=["longest", "annotated", "annotated_or_longest"],
        help="Selection mode: 'longest' always uses longest transcript, 'annotated' uses canonical_transcript=1 (errors if missing), 'annotated_or_longest' tries canonical annotationfirst then falls back to longest if that annotation does not exist for even a single feature",
    )

    # Filter to valid genes command
    valid_parser = subparsers.add_parser(
        "filter_to_valid_genes",
        help="Filter GFF file to remove transcripts without required features and genes without valid transcripts",
    )
    valid_parser.add_argument("--input", required=True, help="Input GFF file")
    valid_parser.add_argument("--output", required=True, help="Output GFF file")
    valid_parser.add_argument(
        "--require-utrs",
        choices=["yes", "no"],
        default="yes",
        help="Require UTRs for valid transcripts (default: yes)",
    )

    # Remove UTRs command
    utrs_parser = subparsers.add_parser(
        "remove_exon_utrs",
        help="Remove five_prime_UTR, three_prime_UTR, and exon features from a GFF file",
    )
    utrs_parser.add_argument("--input", required=True, help="Input GFF file")
    utrs_parser.add_argument("--output", required=True, help="Output GFF file")

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare", help="Compare an input GFF file against a reference using gffcompare"
    )
    compare_parser.add_argument("--reference", required=True, help="Reference GFF file")
    compare_parser.add_argument(
        "--input", required=True, help="Input GFF file to compare"
    )
    compare_parser.add_argument(
        "--output", required=True, help="Output directory for comparison results"
    )
    compare_parser.add_argument(
        "--gffcompare-path", required=True, help="Path to gffcompare executable"
    )

    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate an input GFF file against a reference using internal evaluation functions",
    )
    evaluate_parser.add_argument(
        "--reference", required=True, help="Reference GFF file"
    )
    evaluate_parser.add_argument(
        "--input", required=True, help="Input GFF file to evaluate"
    )
    evaluate_parser.add_argument(
        "--output", required=True, help="Output directory for evaluation results"
    )
    evaluate_parser.add_argument(
        "--edge-tolerance",
        type=int,
        default=0,
        help="Tolerance to allow for matching transcript ends (default: 0)",
    )
    evaluate_parser.add_argument(
        "--ignore-unmatched",
        choices=["yes", "no"],
        default="yes",
        help="Ignore contigs that aren't present in both GFF files (default: yes)",
    )
    evaluate_parser.add_argument(
        "--as-percentage",
        choices=["yes", "no"],
        default="yes",
        help="Display values as percentages (0-100) rather than decimals (0.0-1.0) for consistency with gffcompare (default: yes)",
    )

    # Summarize command
    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize the distribution of source and type fields in a GFF file",
    )
    summarize_parser.add_argument("--input", required=True, help="Input GFF file")
    summarize_parser.add_argument(
        "--species-id",
        default=None,
        help="Species ID for getting attributes_to_drop configuration",
    )

    # Collect results command
    collect_parser = subparsers.add_parser(
        "collect_results",
        help="Collect and consolidate gffcompare.stats.csv and gffeval.stats.csv files from subdirectories",
    )
    collect_parser.add_argument(
        "--input",
        required=True,
        help="Input directory containing subdirectories with stats.csv files",
    )
    collect_parser.add_argument(
        "--output",
        help="Output file path for consolidated results (default: {input}/consolidated.stats.csv)",
    )

    # Clean command
    clean_parser = subparsers.add_parser(
        "clean", help="Clean a GFF file by sorting and/or removing invalid intervals"
    )
    clean_parser.add_argument("--input", required=True, help="Input GFF file")
    clean_parser.add_argument("--output", required=True, help="Output GFF file")
    clean_parser.add_argument(
        "--sort-by-start",
        choices=["yes", "no"],
        default="yes",
        help="Sort records by [seq_id, start, end, type, strand] ascending",
    )
    clean_parser.add_argument(
        "--remove-invalid-interval",
        choices=["yes", "no"],
        default="yes",
        help="Remove records where end < start",
    )

    # PC quality filter command
    pc_filter_parser = subparsers.add_parser(
        "filter_to_pc_quality_score_pass",
        help="Filter GFF file to remove genes and transcripts with passPlantCADFilter=0",
    )
    # Support both direct file paths and species configuration interfaces
    pc_filter_parser.add_argument(
        "--input", help="Input GFF file (direct file interface)"
    )
    pc_filter_parser.add_argument(
        "--output", help="Output GFF file (direct file interface)"
    )
    pc_filter_parser.add_argument(
        "--input-dir",
        help="Directory containing input GFF files (species config interface)",
    )
    pc_filter_parser.add_argument(
        "--output-dir",
        help="Directory to write filtered GFF files (species config interface)",
    )
    pc_filter_parser.add_argument(
        "--species-ids",
        nargs="+",
        help="List of species IDs to determine filenames (species config interface)",
    )

    args = parser.parse_args()

    if args.command == "merge":
        merge_gff_files(args.input, args.output)
    elif args.command == "resolve":
        resolve_gff_file(args.input_dir, args.species_id, args.output)
    elif args.command == "filter_to_chromosome":
        filter_to_chromosome(
            args.input, args.output, args.chromosome_id, args.species_id
        )
    elif args.command == "filter_to_strand":
        filter_to_strand(args.input, args.output, args.strand)
    elif args.command == "set_source":
        set_source(args.input, args.output, args.source)
    elif args.command == "filter_to_min_gene_length":
        filter_to_min_gene_length(args.input, args.output, args.min_length)
    elif args.command == "filter_to_min_feature_length":
        filter_to_min_feature_length(
            args.input, args.output, args.feature_types.split(","), args.min_length
        )
    elif args.command == "filter_to_representative_transcripts":
        filter_to_representative_transcripts(args.input, args.output, args.mode)
    elif args.command == "filter_to_valid_genes":
        filter_to_valid_genes(args.input, args.output, args.require_utrs == "yes")
    elif args.command == "remove_exon_utrs":
        remove_exon_utrs(args.input, args.output)
    elif args.command == "compare":
        compare_gff_files(args.reference, args.input, args.output, args.gffcompare_path)
    elif args.command == "evaluate":
        evaluate_gff_files(
            args.reference,
            args.input,
            args.output,
            args.edge_tolerance,
            args.ignore_unmatched == "yes",
            args.as_percentage == "yes",
        )
    elif args.command == "summarize":
        summarize_gff(args.input, species_id=args.species_id)
    elif args.command == "collect_results":
        collect_results(args.input, args.output)
    elif args.command == "clean":
        clean_gff(
            args.input,
            args.output,
            args.sort_by_start == "yes",
            args.remove_invalid_interval == "yes",
        )
    elif args.command == "filter_to_pc_quality_score_pass":
        filter_to_pc_quality_score_pass(
            input_path=args.input,
            output_path=args.output,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            species_ids=args.species_ids,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
