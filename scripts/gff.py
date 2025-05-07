import argparse
import logging
import pandas as pd
import re
import os
import subprocess
from pathlib import Path
from src.gff_pandas import read_gff3, write_gff3
from src.gff_compare import run_gffcompare, parse_gffcompare_stats

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------------------

def load_gff(path: str) -> pd.DataFrame:
    """Load GFF file into a pandas DataFrame.
    
    Parameters
    ----------
    path : str
        Path to input GFF file
    """
    logger.info(f"Loading GFF file {path}")
    df = read_gff3(path)
    logger.info(f"Loading complete: {df.shape[0]} records found")

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

def filter_to_chromosome(input_path: str, output_path: str, chromosome_id: str) -> None:
    """Filter GFF file to include only entries from a specific chromosome.
    
    Parameters
    ----------
    input_path : str
        Path to input GFF file
    output_path : str
        Path to output GFF file
    chromosome_id : str
        Chromosome ID to filter for
    """
    logger.info(f"Filtering {input_path} to chromosome {chromosome_id}")
    
    # Read GFF file
    features = load_gff(input_path)
    
    # Filter to specified chromosome
    original_count = features.shape[0]
    features = features[features["seq_id"] == chromosome_id]
    filtered_count = features.shape[0]
    
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

def filter_to_min_length(input_path: str, output_path: str, min_length: int) -> None:
    """Filter GFF file to remove features shorter than minimum length and their children.
    
    Parameters
    ----------
    input_path : str
        Path to input GFF file
    output_path : str
        Path to output GFF file
    min_length : int
        Minimum feature length to retain
    """
    logger.info(f"Filtering {input_path} to remove features shorter than {min_length} bp")
    
    # Read GFF file
    features = load_gff(input_path)
    
    original_count = features.shape[0]
    
    # Calculate feature lengths
    feature_lengths = features["end"] - features["start"] + 1
    
    # Identify features to remove based on length
    too_short_mask = feature_lengths < min_length
    too_short_ids = set(features.loc[too_short_mask, "id"].dropna())
    logger.info(f"Found {len(too_short_ids)} features shorter than {min_length} bp")
    
    # Remove features and their children
    features, removed_count, indirect_count = remove_features_by_id(features, too_short_ids)
    
    logger.info(f"Removing {len(too_short_ids)} features directly (too short)")
    logger.info(f"Removing {indirect_count} features indirectly (parent too short)")
    logger.info(f"Total features removed: {removed_count}")
    
    # Write filtered GFF
    save_gff(output_path, features)
    logger.info(f"Filter complete: {features.shape[0]}/{original_count} records retained")

def filter_to_representative_transcripts(input_path: str, output_path: str, mode: str) -> None:
    """Filter GFF file to keep only representative transcripts for each gene.
    
    Parameters
    ----------
    input_path : str
        Path to input GFF file
    output_path : str
        Path to output GFF file
    mode : str
        Selection mode: "longest" or "annotated"
    """
    if mode not in ["longest", "annotated"]:
        raise ValueError(f"Mode must be either 'longest' or 'annotated', got {mode}")
    
    logger.info(f"Filtering {input_path} to keep only representative transcripts using mode: {mode}")
    
    # Read GFF file
    features = load_gff(input_path)
    
    original_count = features.shape[0]
    
    # Check for proper canonical_transcript values if using annotated mode
    if mode == "annotated":
        canonical_values = features["canonical_transcript"].dropna().unique()
        non_valid = [v for v in canonical_values if v not in ["1", ""]]
        if non_valid:
            raise ValueError(f"Found invalid canonical_transcript values: {non_valid}. Expected only '1' or ''.")
    
    # Calculate feature lengths
    features["length"] = features["end"] - features["start"] + 1
    
    # Find all genes and their mRNA children
    genes = features[features["type"] == "gene"]
    logger.info(f"Found {len(genes)} gene features")
    
    # Get gene IDs
    gene_ids = set(genes["id"].dropna())
    
    # Find all mRNAs
    mrnas = features[features["type"] == "mRNA"]
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
        
        if mode == "annotated":
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

def filter_to_valid_genes(input_path: str, output_path: str) -> None:
    """Filter GFF file to remove mRNAs without required features and genes without valid mRNAs.
    
    Parameters
    ----------
    input_path : str
        Path to input GFF file
    output_path : str
        Path to output GFF file
    """
    logger.info(f"Filtering {input_path} to keep only valid transcripts and genes")
    
    # Read GFF file
    features = load_gff(input_path)
    original_count = features.shape[0]
    
    # Find all mRNAs
    mrnas = features[features["type"] == "mRNA"]
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
            has_five_prime = (children["type"] == "five_prime_UTR").any()
            has_cds = (children["type"] == "CDS").any()
            has_three_prime = (children["type"] == "three_prime_UTR").any()
            
            if has_five_prime and has_cds and has_three_prime:
                valid_mrnas.add(mrna_id)
    
    # Identify invalid mRNAs
    invalid_mrnas = set(mrnas["id"].dropna()) - valid_mrnas
    logger.info(f"Found {len(valid_mrnas)} valid mRNAs and {len(invalid_mrnas)} invalid mRNAs")
    
    # Remove invalid mRNAs and their children
    if invalid_mrnas:
        features, mrna_removed_count, mrna_indirect_count = remove_features_by_id(features, invalid_mrnas)
        logger.info(f"Removed {len(invalid_mrnas)} invalid mRNAs, {mrna_indirect_count} child features, {mrna_removed_count} total features")
    
    # Find all genes
    genes = features[features["type"] == "gene"]
    logger.info(f"Found {len(genes)} gene features")
    
    # Identify genes with no valid mRNA children using set operations
    all_gene_ids = set(genes["id"].dropna())
    mrna_parent_ids = set(features[(features["type"] == "mRNA")]["parent"].dropna())
    invalid_genes = all_gene_ids - mrna_parent_ids
    
    logger.info(f"Found {len(invalid_genes)} genes with no valid transcripts")
    
    # Remove invalid genes and their children
    if invalid_genes:
        features, gene_removed_count, gene_indirect_count = remove_features_by_id(features, invalid_genes)
        logger.info(f"Removed {len(invalid_genes)} invalid genes, {gene_indirect_count} child features, {gene_removed_count} total features")
    
    # Write filtered GFF
    save_gff(output_path, features)
    logger.info(f"Filter complete: {features.shape[0]}/{original_count} records retained")

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
    genes = features[features["type"] == "gene"]
    gene_ids = set(genes["id"].dropna())
    total_genes = len(gene_ids)
    
    # Filter for mRNAs
    mrnas = features[features["type"] == "mRNA"]
    
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
    mrna_ids = set(features[features["type"] == "mRNA"]["id"].dropna())
    
    # Filter to features that are children of mRNAs and not mRNAs themselves
    transcript_features = features[
        (features["parent"].isin(mrna_ids)) & 
        (features["type"] != "mRNA")
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
    length_parser = subparsers.add_parser("filter_to_min_length", help="Filter GFF file to remove features and their children if shorter than minimum length")
    length_parser.add_argument("--input", required=True, help="Input GFF file")
    length_parser.add_argument("--output", required=True, help="Output GFF file")
    length_parser.add_argument("--min-length", required=True, type=int, help="Minimum feature length to retain")
    
    # Filter to representative transcripts command
    rep_parser = subparsers.add_parser("filter_to_representative_transcripts", help="Filter GFF file to keep only representative transcripts for each gene")
    rep_parser.add_argument("--input", required=True, help="Input GFF file")
    rep_parser.add_argument("--output", required=True, help="Output GFF file")
    rep_parser.add_argument("--mode", required=True, choices=["longest", "annotated"], help="Selection mode: use longest transcript or annotated canonical transcript (falling back to longest)")
    
    # Filter to valid genes command
    valid_parser = subparsers.add_parser("filter_to_valid_genes", help="Filter GFF file to remove transcripts without required features and genes without valid transcripts")
    valid_parser.add_argument("--input", required=True, help="Input GFF file")
    valid_parser.add_argument("--output", required=True, help="Output GFF file")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare an input GFF file against a reference using gffcompare")
    compare_parser.add_argument("--reference", required=True, help="Reference GFF file")
    compare_parser.add_argument("--input", required=True, help="Input GFF file to compare")
    compare_parser.add_argument("--output", required=True, help="Output directory for comparison results")
    compare_parser.add_argument("--gffcompare-path", required=True, help="Path to gffcompare executable")
    
    # Summarize command
    summarize_parser = subparsers.add_parser("summarize", help="Summarize the distribution of source and type fields in a GFF file")
    summarize_parser.add_argument("--input", required=True, help="Input GFF file")
    
    args = parser.parse_args()
    
    if args.command == "merge":
        merge_gff_files(args.input, args.output)
    elif args.command == "filter_to_chromosome":
        filter_to_chromosome(args.input, args.output, args.chromosome_id)
    elif args.command == "filter_to_strand":
        filter_to_strand(args.input, args.output, args.strand)
    elif args.command == "set_source":
        set_source(args.input, args.output, args.source)
    elif args.command == "filter_to_min_length":
        filter_to_min_length(args.input, args.output, args.min_length)
    elif args.command == "filter_to_representative_transcripts":
        filter_to_representative_transcripts(args.input, args.output, args.mode)
    elif args.command == "filter_to_valid_genes":
        filter_to_valid_genes(args.input, args.output)
    elif args.command == "compare":
        compare_gff_files(args.reference, args.input, args.output, args.gffcompare_path)
    elif args.command == "summarize":
        summarize_gff(args.input)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

