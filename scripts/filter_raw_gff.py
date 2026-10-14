import argparse
import json
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

    # Identify which mRNAs have the required features: an mRNA has a feature type
    # when one of the rows of that type has the mRNA as its parent.
    required = [GffFeatureType.CDS.value]
    if require_utrs:
        required += [
            GffFeatureType.FIVE_PRIME_UTR.value,
            GffFeatureType.THREE_PRIME_UTR.value,
        ]
    mrna_ids = mrnas["id"].dropna()
    complete = pd.Series(True, index=mrna_ids.index)
    for feature_type in required:
        complete &= mrna_ids.isin(
            features.loc[features["type"] == feature_type, "parent"].dropna()
        )
    valid_mrnas = set(mrna_ids[complete])

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


def update_boundaries(
    features: pd.DataFrame, genes: pd.DataFrame, mrnas: pd.DataFrame
) -> tuple[pd.DataFrame, int, int]:
    """Fit each gene's mRNAs to their remaining children, then each gene to its mRNAs.

    `genes` and `mrnas` are the gene and mRNA rows of `features`. Returns the updated
    features and the number of genes and mRNAs whose boundaries changed. Computed with
    one grouping per level, so the time grows with the number of rows, not with the
    number of genes times the number of rows.
    """
    features = features.copy()

    def extents(rows: pd.DataFrame) -> pd.DataFrame:
        return rows.groupby("parent", sort=False).agg(
            start=("start", "min"), end=("end", "max")
        )

    def assign(ids, extent: pd.DataFrame) -> None:
        rows = features["id"].isin(ids)
        for column in ("start", "end"):
            features.loc[rows, column] = features.loc[rows, "id"].map(extent[column])

    # mRNAs of a gene that have children, compared with the first row of their ID
    children = extents(features)
    mrna_ids = mrnas.loc[mrnas["parent"].isin(genes["id"].dropna()), "id"]
    mrna_ids = mrna_ids[mrna_ids.isin(children.index)].drop_duplicates()
    first = features[~features["id"].duplicated()].set_index("id")
    new = children.loc[mrna_ids]
    old = first.loc[mrna_ids]
    changed = (new["start"].to_numpy() != old["start"].to_numpy()) | (
        new["end"].to_numpy() != old["end"].to_numpy()
    )
    assign(mrna_ids[changed], children)

    # Genes, compared with their own rows before any update
    spans = extents(features[features["type"] == GffFeatureType.MRNA.value])
    with_mrnas = genes[genes["id"].isin(spans.index)]
    span = spans.loc[with_mrnas["id"]]
    gene_changed = (span["start"].to_numpy() != with_mrnas["start"].to_numpy()) | (
        span["end"].to_numpy() != with_mrnas["end"].to_numpy()
    )
    assign(with_mrnas.loc[gene_changed, "id"], spans)
    return features, int(gene_changed.sum()), int(changed.sum())


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

        features_filtered, genes_updated, transcripts_updated = update_boundaries(
            features_filtered, genes, mrnas
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

    parser.add_argument(
        "--input-gff", "-i", default=None, type=str, help="Input GFF file"
    )
    parser.add_argument(
        "--output-gff", "-o", default=None, type=str, help="Output GFF file"
    )

    parser.add_argument(
        "--manifest",
        default=None,
        type=str,
        help="Manifest json for multi-chromosome runs. "
        "Key-value pairs 'chromosome_id', 'raw_gff' "
        "and 'filtered_gff' are required. Required if "
        "--input-gff and --output-gff are not specified.",
    )

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

    if args.manifest is None:
        if (args.input_gff is None) or (args.output_gff is None):
            logger.error(
                "Error: one of the following must be provided:\n"
                "--manifest\n OR \n --input-gff and --output-gff"
            )
            raise RuntimeError

        filter_gff(
            args.input_gff,
            args.output_gff,
            args.min_feature_length,
            args.feature_types,
            args.min_gene_length,
            args.require_utrs,
            args.keep_incomplete_models,
        )

    else:
        with open(args.manifest) as fh:
            entries = json.load(fh)

        for entry in entries:
            chromosome_id = entry["chromosome_id"]
            raw_gff = entry["raw_gff"]
            filtered_gff = entry["filtered_gff"]

            logger.info(f"Filtering gff for chromosome {chromosome_id}")

            filter_gff(
                raw_gff,
                filtered_gff,
                args.min_feature_length,
                args.feature_types,
                args.min_gene_length,
                args.require_utrs,
                args.keep_incomplete_models,
            )


if __name__ == "__main__":
    main()
