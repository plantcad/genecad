import argparse
import logging
import re
import pandas as pd
from src.gff_pandas import read_gff3, write_gff3
from src.schema import GffFeatureType

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------------------
# TODO: temp copy for testing - move to dedicated utils space


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
    has_id = "id" in features.columns
    has_parent = "parent" in features.columns

    # Find all children of features to remove
    to_remove = set(feature_ids_to_remove)
    indirect_removals = set()

    # Keep adding children until no more are found
    remaining_ids = set(feature_ids_to_remove)
    while remaining_ids and has_id and has_parent:
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
    if has_id:
        has_id_indices = features[features["id"].isin(to_remove)].index
        indices_to_remove.update(has_id_indices)

    # Then collect indices for features that have parents in the removal set
    # (this covers features without IDs)
    if has_parent:
        has_parent_indices = features[features["parent"].isin(to_remove)].index
        indices_to_remove.update(has_parent_indices)

    # Filter out the features to remove by index
    original_count = features.shape[0]
    filtered_features = features.drop(index=indices_to_remove)
    removed_count = original_count - filtered_features.shape[0]

    return filtered_features, removed_count, len(indirect_removals)


def filter_to_valid_genes(
    features: pd.DataFrame, require_utrs: bool = True
) -> pd.DataFrame:
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

    if require_utrs:
        logger.info(
            "Requiring five_prime_UTR, CDS, and three_prime_UTR for valid transcripts"
        )
    else:
        logger.info("Requiring only CDS for valid transcripts")

    # Read GFF file
    original_count = features.shape[0]

    if not {"id", "parent"}.issubset(features.columns):
        logger.warning(
            "Columns 'id' and/or 'parent' not found; skipping valid-gene filter"
        )
        return features

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
    logger.info(
        f"Filter complete: {features.shape[0]}/{original_count} records retained"
    )
    return features


def filter_to_min_gene_length(features: pd.DataFrame, min_length: int) -> pd.DataFrame:
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

    original_count = features.shape[0]

    # Filter to only gene features for length checking
    genes = features[features["type"] == GffFeatureType.GENE.value]
    logger.info(f"Found {len(genes)} gene features to check for length")

    # Calculate gene lengths
    gene_lengths = genes["end"] - genes["start"] + 1

    # Identify genes to remove based on length
    too_short_mask = gene_lengths < min_length
    if "id" in genes.columns:
        too_short_ids = set(genes.loc[too_short_mask, "id"].dropna())
    else:
        logger.warning("Column 'id' not found; cannot identify short genes by ID")
        too_short_ids = set()
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
    logger.info(
        f"Filter complete: {features.shape[0]}/{original_count} records retained"
    )
    return features


def filter_to_min_feature_length(
    features: pd.DataFrame, feature_types: list[str], min_length: int
) -> pd.DataFrame:
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

    # Read GFF file
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
    if "id" in features.columns:
        small_feature_ids = set(features.loc[small_features_mask, "id"].dropna())
        logger.info(f"Found {len(small_feature_ids)} small features to remove")
    else:
        logger.warning(
            "Column 'id' not found; counting small features by rows instead of IDs"
        )
        logger.info(f"Found {int(small_features_mask.sum())} small features to remove")

    # Remove small features
    features_filtered = features[~small_features_mask]

    # Group features by gene and update boundaries when hierarchical columns are available
    genes_updated = 0
    transcripts_updated = 0

    if {"id", "parent"}.issubset(features_filtered.columns):
        genes = features_filtered[
            features_filtered["type"] == GffFeatureType.GENE.value
        ]
        mrnas = features_filtered[
            features_filtered["type"] == GffFeatureType.MRNA.value
        ]

        # Track statistics for boundary updates
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
                mrna_features = features_filtered[
                    features_filtered["parent"] == mrna_id
                ]
                if not mrna_features.empty:
                    original_mrna = features_filtered[
                        features_filtered["id"] == mrna_id
                    ].iloc[0]
                    original_mrna_start = original_mrna["start"]
                    original_mrna_end = original_mrna["end"]

                    mrna_start = mrna_features["start"].min()
                    mrna_end = mrna_features["end"].max()

                    # Update mRNA boundaries and track if changed
                    if (
                        mrna_start != original_mrna_start
                        or mrna_end != original_mrna_end
                    ):
                        transcripts_updated += 1
                        features_filtered.loc[
                            features_filtered["id"] == mrna_id, "start"
                        ] = mrna_start
                        features_filtered.loc[
                            features_filtered["id"] == mrna_id, "end"
                        ] = mrna_end

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
                    features_filtered.loc[
                        features_filtered["id"] == gene_id, "start"
                    ] = new_start
                    features_filtered.loc[features_filtered["id"] == gene_id, "end"] = (
                        new_end
                    )
    else:
        total_genes = int(
            (features_filtered["type"] == GffFeatureType.GENE.value).sum()
        )
        total_transcripts = int(
            (features_filtered["type"] == GffFeatureType.MRNA.value).sum()
        )
        logger.warning(
            "Columns 'id' and/or 'parent' not found; skipping gene/mRNA boundary updates"
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

    logger.info(
        f"Filter complete: {removed_count} features removed, {len(features_filtered)}/{original_count} records retained"
    )

    return features_filtered


def filter_gff(
    input_path: str,
    output_path: str,
    min_feature_length: int,
    feature_types: str,
    min_gene_length: int,
    require_utrs: bool,
    keep_incomplete_models: bool,
) -> None:
    features = load_gff(input_path)

    logger.info(
        f"Filtering {input_path} to remove {feature_types} features shorter than {min_feature_length} bp"
    )
    features = filter_to_min_feature_length(
        features, feature_types.split(","), min_feature_length
    )

    logger.info(
        f"Filtering {input_path} to remove genes shorter than {min_gene_length} bp"
    )
    features = filter_to_min_gene_length(features, min_gene_length)

    if not keep_incomplete_models:
        logger.info(f"Filtering {input_path} to keep only valid transcripts and genes")
        features = filter_to_valid_genes(features, require_utrs)
    save_gff(output_path, features)


def main() -> None:
    """Parse command line arguments and execute the appropriate function."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Manipulate GFF files")

    parser.add_argument("--input-gff", "-i", required=True, help="Input GFF file")
    parser.add_argument("--output-gff", "-o", required=True, help="Output GFF file")

    # TODO: this should really be checking exon length, not feature length
    parser.add_argument(
        "--min-feature-length",
        type=int,
        default=2,
        help="minimum feature length to retain. Default 2",
    )
    parser.add_argument(
        "--feature-types",
        type=str,
        default="five_prime_UTR,three_prime_UTR,CDS",
        help="Comma-separated list of feature types to check for length. Default is five_prime_UTR,"
        "three_prime_UTR,CDS",
    )

    # TODO: this would make more sense with introns excluded. Also, as transcript length, not gene length
    parser.add_argument(
        "--min-gene-length",
        type=int,
        default=30,
        help="Minimum gene length to retain (introns included). Default 30",
    )

    parser.add_argument(
        "--require-utrs",
        action="store_true",
        help="Remove transcripts that are missing 5' or 3' "
        "UTRs. Ignored if --keep-incomplete-models True",
    )

    # Not an option I would recommend, but keeping it available for parity with earlier versions
    parser.add_argument(
        "--keep-incomplete-models",
        action="store_true",
        help="Keep incomplete feature models: mRNA "
        "transcripts that have no CDS and genes "
        "that have no mRNA transcript.",
    )
    args = parser.parse_args()

    filter_gff(
        args.input_gff,
        args.output_gff,
        args.min_feature_length,
        args.feature_types,
        args.min_gene_length,
        args.require_utrs,
        args.keep_incomplete_models,
    )


if __name__ == "__main__":
    main()
