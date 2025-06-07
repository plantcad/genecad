import argparse
import logging
import pandas as pd
import re
import os
import glob
from src.gff_pandas import read_gff3, write_gff3
from src.gff_compare import run_gffcompare, parse_gffcompare_stats
from src.config import SPECIES_CONFIGS
from src.schema import GffFeatureType, RegionType

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
            ";".join([
                kv
                for kv in attrs.split(";")
                if kv.split("=")[0] not in attributes_to_drop
            ])
            for attrs in df["attributes"].fillna("")
        ]

    # Create mapping of old column names to new lcase names
    col_mapping = {}
    for col in df.columns:
        new_name = re.sub(r'\s+', '_', col).lower()
        if new_name in col_mapping.values():
            raise ValueError(f"Column name collision detected: multiple columns would map to '{new_name}'")
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

def remove_features_by_id(features: pd.DataFrame, feature_ids_to_remove: set) -> tuple[pd.DataFrame, int, int]:
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

def filter_to_chromosome(input_path: str, output_path: str, chromosome_id: str, species_id: str = None) -> None:
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
    logger.info(f"Filtering {input_path} to chromosome {chromosome_id}" + 
                (f" with species mapping for {species_id}" if species_id else ""))
    
    # Find the seq_ids to filter for based on species config if provided
    seq_ids_to_filter = [chromosome_id]  # Default if no species_id provided
    normalized_id = chromosome_id  # Default if no species_id provided
    attributes_to_drop = None
    
    if species_id:
        if species_id not in SPECIES_CONFIGS:
            raise ValueError(f"Unknown species ID: {species_id}. Available species: {list(SPECIES_CONFIGS.keys())}")
        
        # Look up the chromosome mapping for this species
        species_config = SPECIES_CONFIGS[species_id]
        
        # Find all source chromosome IDs that map to our target normalized ID
        source_ids = [source_id for source_id, norm_id in species_config.chromosome_map.items() 
                     if norm_id == chromosome_id]
        
        if not source_ids:
            raise ValueError(f"No source chromosome IDs found for normalized ID '{chromosome_id}' in species '{species_id}'")
        
        attributes_to_drop = species_config.gff.attributes_to_drop
        seq_ids_to_filter = source_ids
        normalized_id = chromosome_id
        
        logger.info(f"Using source chromosome IDs: {seq_ids_to_filter}")
        logger.info(f"Chromosome ID will be normalized from {chromosome_id} to {normalized_id}")
        logger.info(f"The following GFF attributes will be dropped: {attributes_to_drop}")
    
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
        raise ValueError(f"Strand must be either 'positive', 'negative', or 'both', got {strand}")
    
    strand_value = "+" if strand == "positive" else "-" if strand == "negative" else None
    logger.info(f"Filtering {input_path} to {strand} strand{'' if strand != 'both' else 's'}")
    
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

def filter_to_min_gene_length(input_path: str, output_path: str, min_length: int) -> None:
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
    features, removed_count, indirect_count = remove_features_by_id(features, too_short_ids)
    
    logger.info(f"Removing {len(too_short_ids)} genes directly (too short)")
    logger.info(f"Removing {indirect_count} features indirectly (parent gene too short)")
    logger.info(f"Total features removed: {removed_count}")
    
    # Write filtered GFF
    save_gff(output_path, features)
    logger.info(f"Filter complete: {features.shape[0]}/{original_count} records retained")

def filter_to_min_feature_length(input_path: str, output_path: str, feature_types: list[str], min_length: int) -> None:
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
    logger.info(f"Filtering {input_path} to remove {feature_types} features shorter than {min_length} bp")
    
    # Read GFF file
    features = load_gff(input_path)
    original_count = features.shape[0]
    
    # Validate feature types against schema
    valid_types = set(GffFeatureType)
    requested_types = set(feature_types)
    invalid_types = requested_types - valid_types
    
    if invalid_types:
        logger.error(f"Invalid feature types requested: {sorted(invalid_types)}")
        logger.error(f"Valid feature types: {sorted(valid_types)}")
        raise ValueError(f"Feature types {sorted(invalid_types)} are not valid GFF feature types")
    
    logger.info(f"Validated feature types: {sorted(requested_types)} (all valid)")
    
    # Calculate feature lengths
    features["length"] = features["end"] - features["start"] + 1
    
    # Find features to remove (small features of specified types)
    small_features_mask = (
        features["type"].isin(feature_types) & 
        (features["length"] < min_length)
    )
    small_feature_ids = set(features.loc[small_features_mask, "id"].dropna())
    
    logger.info(f"Found {len(small_feature_ids)} small features to remove")
    
    # Remove small features
    features_filtered = features[~features.index.isin(features[small_features_mask].index)]
    
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
        gene_mrna_ids = gene_children[gene_children["type"] == GffFeatureType.MRNA.value]["id"].tolist()
        
        # First, update mRNA boundaries for this gene
        for mrna_id in gene_mrna_ids:
            mrna_features = features_filtered[features_filtered["parent"] == mrna_id]
            if not mrna_features.empty:
                original_mrna = features_filtered[features_filtered["id"] == mrna_id].iloc[0]
                original_mrna_start = original_mrna["start"]
                original_mrna_end = original_mrna["end"]
                
                mrna_start = mrna_features["start"].min()
                mrna_end = mrna_features["end"].max()
                
                # Update mRNA boundaries and track if changed
                if mrna_start != original_mrna_start or mrna_end != original_mrna_end:
                    transcripts_updated += 1
                    features_filtered.loc[features_filtered["id"] == mrna_id, "start"] = mrna_start
                    features_filtered.loc[features_filtered["id"] == mrna_id, "end"] = mrna_end
        
        # Now update gene boundaries based on updated mRNA boundaries
        updated_gene_children = features_filtered[features_filtered["parent"] == gene_id]
        updated_mrnas = updated_gene_children[updated_gene_children["type"] == GffFeatureType.MRNA.value]
        
        if not updated_mrnas.empty:
            new_start = updated_mrnas["start"].min()
            new_end = updated_mrnas["end"].max()
            
            # Update gene boundaries and track if changed
            if new_start != original_gene_start or new_end != original_gene_end:
                genes_updated += 1
                features_filtered.loc[features_filtered["id"] == gene_id, "start"] = new_start
                features_filtered.loc[features_filtered["id"] == gene_id, "end"] = new_end
    
    # Remove temporary length column
    features_filtered = features_filtered.drop(columns=["length"])
    
    removed_count = original_count - len(features_filtered)
    
    # Log statistics
    logger.info(f"Boundary updates:")
    logger.info(f"  - Genes: {genes_updated}/{total_genes} ({genes_updated/total_genes*100:.1f}%) had boundaries updated")
    logger.info(f"  - Transcripts: {transcripts_updated}/{total_transcripts} ({transcripts_updated/total_transcripts*100:.1f}%) had boundaries updated")
    
    # Write filtered GFF
    save_gff(output_path, features_filtered)
    logger.info(f"Filter complete: {removed_count} features removed, {len(features_filtered)}/{original_count} records retained")

def filter_to_representative_transcripts(input_path: str, output_path: str, mode: str) -> None:
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
        raise ValueError(f"Mode must be either 'longest', 'annotated', or 'annotated_or_longest', got {mode}")
    
    logger.info(f"Filtering {input_path} to keep only representative transcripts using mode: {mode}")
    
    # Read GFF file
    features = load_gff(input_path)
    
    original_count = features.shape[0]
    
    # Check if the canonical_transcript column exists when using annotated modes
    has_canonical_column = 'canonical_transcript' in features.columns
    
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
            logger.info("Falling back to 'longest' mode (canonical_transcript column not found)")
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
                raise ValueError(f"Found invalid canonical_transcript values: {non_valid}. Expected only '1' or ''.")
                
            # Try to find canonical transcript
            canonical_mrnas = gene_mrnas[gene_mrnas["canonical_transcript"] == "1"]
            
            if not canonical_mrnas.empty:
                # If multiple canonical transcripts (shouldn't happen but just in case), take the longest
                if len(canonical_mrnas) > 1:
                    logger.warning(f"Gene {gene_id} has multiple canonical transcripts, selecting the longest")
                    selected_mrna = canonical_mrnas.loc[canonical_mrnas["length"].idxmax()]
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
    logger.info(f"  - {selection_stats['annotation']} ({selection_stats['annotation']/total_selected*100:.1f}%) selected by annotation")
    logger.info(f"  - {selection_stats['length']} ({selection_stats['length']/total_selected*100:.1f}%) selected by length")
    
    # Find mRNAs to remove (those not in the keep set)
    mrnas_to_remove = set(mrnas["id"].dropna()) - mrnas_to_keep
    logger.info(f"Removing {len(mrnas_to_remove)} non-representative mRNAs")
    
    # Remove non-representative mRNAs and their children
    features, removed_count, indirect_count = remove_features_by_id(features, mrnas_to_remove)
    
    logger.info(f"Removed {len(mrnas_to_remove)} non-representative mRNAs")
    logger.info(f"Removed {indirect_count} children of non-representative mRNAs")
    logger.info(f"Total features removed: {removed_count}")
    
    # Remove the temporary length column
    features = features.drop(columns=["length"])
    
    # Write filtered GFF
    save_gff(output_path, features)
    logger.info(f"Filter complete: {features.shape[0]}/{original_count} records retained")

def filter_to_valid_genes(input_path: str, output_path: str, require_utrs: bool = True) -> None:
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
        logger.info("Requiring five_prime_UTR, CDS, and three_prime_UTR for valid transcripts")
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
                has_five_prime = (children["type"] == GffFeatureType.FIVE_PRIME_UTR.value).any()
                has_three_prime = (children["type"] == GffFeatureType.THREE_PRIME_UTR.value).any()
                
                if has_five_prime and has_cds and has_three_prime:
                    valid_mrnas.add(mrna_id)
            else:
                if has_cds:
                    valid_mrnas.add(mrna_id)
    
    # Identify invalid mRNAs
    invalid_mrnas = set(mrnas["id"].dropna()) - valid_mrnas
    logger.info(f"Found {len(valid_mrnas)} valid mRNAs and {len(invalid_mrnas)} invalid mRNAs")
    
    # Remove invalid mRNAs and their children
    if invalid_mrnas:
        features, mrna_removed_count, mrna_indirect_count = remove_features_by_id(features, invalid_mrnas)
        logger.info(f"Removed {len(invalid_mrnas)} invalid mRNAs, {mrna_indirect_count} child features, {mrna_removed_count} total features")
    
    # Find all genes
    genes = features[features["type"] == GffFeatureType.GENE.value]
    logger.info(f"Found {len(genes)} gene features")
    
    # Identify genes with no valid mRNA children using set operations
    all_gene_ids = set(genes["id"].dropna())
    mrna_parent_ids = set(features[(features["type"] == GffFeatureType.MRNA.value)]["parent"].dropna())
    invalid_genes = all_gene_ids - mrna_parent_ids
    
    logger.info(f"Found {len(invalid_genes)} genes with no valid transcripts")
    
    # Remove invalid genes and their children
    if invalid_genes:
        features, gene_removed_count, gene_indirect_count = remove_features_by_id(features, invalid_genes)
        logger.info(f"Removed {len(invalid_genes)} invalid genes, {gene_indirect_count} child features, {gene_removed_count} total features")
    
    # Write filtered GFF
    save_gff(output_path, features)
    logger.info(f"Filter complete: {features.shape[0]}/{original_count} records retained")

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
    utr_types = [GffFeatureType.FIVE_PRIME_UTR.value, GffFeatureType.THREE_PRIME_UTR.value, RegionType.EXON.value]
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
    logger.info(f"UTR/exon removal complete: {removed_count} features removed, {len(filtered_features)}/{original_count} records retained ({len(filtered_features)/original_count*100:.1f}%)")

def compare_gff_files(reference_path: str, input_path: str, output_dir: str, gffcompare_path: str) -> None:
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
    output_prefix = run_gffcompare(reference_path, input_path, output_dir, gffcompare_path)
    
    # Parse the stats file
    stats_file = f"{output_prefix}.stats"
    
    if not os.path.exists(stats_file):
        logger.error(f"gffcompare stats file not found: {stats_file}")
        raise FileNotFoundError(f"gffcompare did not produce a stats file at {stats_file}")
    
    # Show stats file content 
    with open(stats_file, 'r') as f:
        logger.info(f"Stats file content from gffcompare at {stats_file}:")
        logger.info("\n" + f.read())

    # Parse the stats file into dataframes
    stats_df = parse_gffcompare_stats(stats_file)
    
    # Save the stats dataframes to CSV
    stats_csv = os.path.join(output_dir, "gffcompare.stats.csv")

    def calculate_f1_score(row):
        """Calculate F1 score from precision and sensitivity."""
        precision = row['precision']
        sensitivity = row['sensitivity']
        if precision + sensitivity == 0:
            return 0
        return 2 * (precision * sensitivity) / (precision + sensitivity)
    
    stats_df['f1_score'] = stats_df.apply(calculate_f1_score, axis=1)

    logger.info(f"Parsed stats: \n{stats_df}")
    
    logger.info(f"Saving parsed stats to {stats_csv}")
    stats_df.to_csv(stats_csv, sep='\t', index=False)
    
    logger.info("Comparison complete")

def summarize_gff(input_path: str) -> None:
    """Summarize the distribution of source and type fields in a GFF file.
    
    Parameters
    ----------
    input_path : str
        Path to input GFF file
    """
    logger.info(f"Summarizing GFF file: {input_path}")
    
    # Read GFF file
    features = load_gff(input_path)
    
    # Count source field distribution
    source_counts = features["source"].value_counts()
    
    # Count type field distribution
    type_counts = features["type"].value_counts()
    
    # Count strand distribution
    strand_counts = features["strand"].value_counts()
    
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
    
    # Display source/strand/type combinations
    print("\n=== Source/Strand/Type Combinations ===")
    print(f"{'Source':<20} {'Strand':<10} {'Type':<20} {'Count':<10} {'Percentage':<10}")
    print("-" * 75)
    
    # Sort combinations by count in descending order
    sorted_combos = source_strand_type_combos.sort_values(ascending=False)
    
    for (source, strand, type_name), count in sorted_combos.items():
        strand_display = "+" if strand == "+" else "-" if strand == "-" else strand
        percentage = (count / total_records) * 100
        print(f"{source:<20} {strand_display:<10} {type_name:<20} {count:<10} {percentage:.2f}%")
    
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
    for transcript_count, gene_count in sorted(transcript_counts.items()):
        percentage = (gene_count / total_genes) * 100
        print(f"{transcript_count:<20} {gene_count:<15} {percentage:.2f}%")
    
    # Calculate fraction of genes with a canonical transcript
    print("\n=== Canonical Transcript Annotation ===")
    
    # Check if canonical_transcript column exists
    has_canonical_column = 'canonical_transcript' in mrnas.columns
    
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
            print(f"{value if value else '<empty>':<20} {count:<10} {percentage:.2f}%{marker}")
        
        if has_unexpected_values:
            print("\nWARNING: Unexpected canonical_transcript values detected!")
            print("Expected values are '1' or empty/missing.")
        
        # Find mRNAs with canonical_transcript="1"
        canonical_mrnas = mrnas[mrnas["canonical_transcript"] == "1"]
        
        # Get the set of genes that have at least one canonical transcript
        genes_with_canonical = set(canonical_mrnas["parent"].unique())
        genes_with_canonical_count = len(genes_with_canonical)
        
        # Calculate percentage
        canonical_percentage = (genes_with_canonical_count / total_genes) * 100 if total_genes > 0 else 0
        
        print(f"\nGenes with canonical transcript: {genes_with_canonical_count}/{total_genes} ({canonical_percentage:.2f}%)")
    
    # Calculate type combinations per gene
    print("\n=== Feature Type Combinations Per Transcript ===")
    print(f"{'Type Combination':<50} {'Transcript Count':<15} {'Percentage':<10}")
    print("-" * 75)
    
    # Get all mRNA IDs
    mrna_ids = set(features[features["type"] == GffFeatureType.MRNA.value]["id"].dropna())
    
    # Filter to features that are children of mRNAs and not mRNAs themselves
    transcript_features = features[
        (features["parent"].isin(mrna_ids)) & 
        (features["type"] != GffFeatureType.MRNA.value)
    ]
    
    # Group by parent (transcript) and get unique types, then count combinations
    type_combinations = (
        transcript_features
        .groupby("parent")["type"]
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

def collect_results(input_dir: str, output_path: str = None) -> None:
    """Collect and consolidate gffcompare.stats.csv files from subdirectories.
    
    Parameters
    ----------
    input_dir : str
        Path to the directory containing subdirectories with stats.csv files
    output_path : str, optional
        Path to save the consolidated results file. If None, defaults to
        {input_dir}/gffcompare.stats.consolidated.csv
    """
    logger.info(f"Collecting gffcompare results from subdirectories of {input_dir}")
    
    # Find all gffcompare.stats.csv files in subdirectories
    if output_path is None:
        output_path = os.path.join(input_dir, "gffcompare.stats.consolidated.csv")
        
    stats_files = glob.glob(os.path.join(input_dir, "*/gffcompare.stats.csv"))
    
    if not stats_files:
        logger.error(f"No gffcompare.stats.csv files found in subdirectories of {input_dir}")
        return
    
    logger.info(f"Found {len(stats_files)} stats files")
    
    # Collect and process each file
    dfs = []
    for file_path in stats_files:
        # Extract the directory name as source
        source = os.path.basename(os.path.dirname(file_path))
        logger.info(f"Processing {file_path} (source: {source})")
        
        # Read the CSV file
        df = pd.read_csv(file_path, sep='\t')
        
        # Add source column
        df['source'] = source
        
        dfs.append(df)
    
    # Concatenate all dataframes
    if not dfs:
        logger.error("No valid data found to consolidate")
        return
    
    consolidated_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by source and level
    consolidated_df = consolidated_df.sort_values(['source', 'level'])
    
    # Print the consolidated dataframe
    print(consolidated_df.to_string(index=False))
    
    # Save to file
    logger.info(f"Saving consolidated results to {output_path}")
    consolidated_df.to_csv(output_path, sep='\t', index=False)
    logger.info("Results saved successfully")

def main() -> None:
    """Parse command line arguments and execute the appropriate function."""
    parser = argparse.ArgumentParser(description="Manipulate GFF files")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")
    
    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge multiple GFF files into one")
    merge_parser.add_argument("--input", required=True, nargs='+', help="Input GFF files")
    merge_parser.add_argument("--output", required=True, help="Output GFF file")
    
    # Filter to chromosome command
    filter_parser = subparsers.add_parser("filter_to_chromosome", help="Filter GFF file to include only entries from a specific chromosome")
    filter_parser.add_argument("--input", required=True, help="Input GFF file")
    filter_parser.add_argument("--output", required=True, help="Output GFF file")
    filter_parser.add_argument("--chromosome-id", required=True, help="Chromosome ID to filter for")
    filter_parser.add_argument("--species-id", help="Species ID to use for chromosome mapping")
    
    # Filter to strand command
    strand_parser = subparsers.add_parser("filter_to_strand", help="Filter GFF file to include only entries from a specific strand")
    strand_parser.add_argument("--input", required=True, help="Input GFF file")
    strand_parser.add_argument("--output", required=True, help="Output GFF file")
    strand_parser.add_argument("--strand", required=True, choices=["positive", "negative", "both"], help="Strand to filter for")
    
    # Set source command
    source_parser = subparsers.add_parser("set_source", help="Set the source field for all records in a GFF file")
    source_parser.add_argument("--input", required=True, help="Input GFF file")
    source_parser.add_argument("--output", required=True, help="Output GFF file")
    source_parser.add_argument("--source", required=True, help="Source value to set")
    
    # Filter to minimum length command
    length_parser = subparsers.add_parser("filter_to_min_gene_length", help="Filter GFF file to remove genes shorter than minimum length and their children")
    length_parser.add_argument("--input", required=True, help="Input GFF file")
    length_parser.add_argument("--output", required=True, help="Output GFF file")
    length_parser.add_argument("--min-length", required=True, type=int, help="Minimum gene length to retain")
    
    # Filter small features command
    small_features_parser = subparsers.add_parser("filter_to_min_feature_length", help="Filter GFF file to remove small features of specified types and update gene/mRNA boundaries")
    small_features_parser.add_argument("--input", required=True, help="Input GFF file")
    small_features_parser.add_argument("--output", required=True, help="Output GFF file")
    small_features_parser.add_argument("--feature-types", required=True, help="Comma-separated list of feature types to filter by length")
    small_features_parser.add_argument("--min-length", required=True, type=int, help="Minimum feature length to retain")
    
    # Filter to representative transcripts command
    rep_parser = subparsers.add_parser("filter_to_representative_transcripts", help="Filter GFF file to keep only one representative transcript for each gene")
    rep_parser.add_argument("--input", required=True, help="Input GFF file")
    rep_parser.add_argument("--output", required=True, help="Output GFF file")
    rep_parser.add_argument("--mode", required=True, choices=["longest", "annotated", "annotated_or_longest"], 
                         help="Selection mode: 'longest' always uses longest transcript, 'annotated' uses canonical_transcript=1 (errors if missing), 'annotated_or_longest' tries canonical annotationfirst then falls back to longest if that annotation does not exist for even a single feature")
    
    # Filter to valid genes command
    valid_parser = subparsers.add_parser("filter_to_valid_genes", help="Filter GFF file to remove transcripts without required features and genes without valid transcripts")
    valid_parser.add_argument("--input", required=True, help="Input GFF file")
    valid_parser.add_argument("--output", required=True, help="Output GFF file")
    valid_parser.add_argument("--require-utrs", choices=["yes", "no"], default="yes", help="Require UTRs for valid transcripts (default: yes)")
    
    # Remove UTRs command
    utrs_parser = subparsers.add_parser("remove_exon_utrs", help="Remove five_prime_UTR, three_prime_UTR, and exon features from a GFF file")
    utrs_parser.add_argument("--input", required=True, help="Input GFF file")
    utrs_parser.add_argument("--output", required=True, help="Output GFF file")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare an input GFF file against a reference using gffcompare")
    compare_parser.add_argument("--reference", required=True, help="Reference GFF file")
    compare_parser.add_argument("--input", required=True, help="Input GFF file to compare")
    compare_parser.add_argument("--output", required=True, help="Output directory for comparison results")
    compare_parser.add_argument("--gffcompare-path", required=True, help="Path to gffcompare executable")
    
    # Summarize command
    summarize_parser = subparsers.add_parser("summarize", help="Summarize the distribution of source and type fields in a GFF file")
    summarize_parser.add_argument("--input", required=True, help="Input GFF file")
    
    # Collect results command
    collect_parser = subparsers.add_parser("collect_results", help="Collect and consolidate gffcompare.stats.csv files from subdirectories")
    collect_parser.add_argument("--input", required=True, help="Input directory containing subdirectories with stats.csv files")
    collect_parser.add_argument("--output", help="Output file path for consolidated results (default: {input}/gffcompare.stats.consolidated.csv)")
    
    args = parser.parse_args()
    
    if args.command == "merge":
        merge_gff_files(args.input, args.output)
    elif args.command == "filter_to_chromosome":
        filter_to_chromosome(args.input, args.output, args.chromosome_id, args.species_id)
    elif args.command == "filter_to_strand":
        filter_to_strand(args.input, args.output, args.strand)
    elif args.command == "set_source":
        set_source(args.input, args.output, args.source)
    elif args.command == "filter_to_min_gene_length":
        filter_to_min_gene_length(args.input, args.output, args.min_length)
    elif args.command == "filter_to_min_feature_length":
        filter_to_min_feature_length(args.input, args.output, args.feature_types.split(','), args.min_length)
    elif args.command == "filter_to_representative_transcripts":
        filter_to_representative_transcripts(args.input, args.output, args.mode)
    elif args.command == "filter_to_valid_genes":
        filter_to_valid_genes(args.input, args.output, args.require_utrs == "yes")
    elif args.command == "remove_exon_utrs":
        remove_exon_utrs(args.input, args.output)
    elif args.command == "compare":
        compare_gff_files(args.reference, args.input, args.output, args.gffcompare_path)
    elif args.command == "summarize":
        summarize_gff(args.input)
    elif args.command == "collect_results":
        collect_results(args.input, args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
