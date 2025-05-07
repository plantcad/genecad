#!/usr/bin/env python3

# Example usage:
# python scripts/misc/analyze_prediction_codons.py \
# --gff $PRED_DIR/gff/predictions__strand_positive__minlen_03__valid_only.gff \
# --fasta $DATA_DIR/testing_data/fasta/Zea_mays-B73-REFERENCE-NAM-5.0.fa.gz \
# --chrom chr1

import argparse
import os
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple
from Bio import SeqIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gzip
from src.gff_reader import read_gff3

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Sequence:
    """Container for chromosome sequence data with forward and reverse complement.
    
    Attributes
    ----------
    forward : str
        Forward strand sequence
    reverse_complement : str
        Reverse complement of the forward strand
    """
    forward: str
    reverse_complement: str

def load_cds_features(gff_file: str, chrom_id: str) -> pd.DataFrame:
    """Load GFF file and extract CDS region spans for each transcript.
    
    This function groups individual CDS features by their parent transcript
    and creates a single 'cds_region' entry that spans from the earliest
    to the latest CDS feature for each transcript.
    
    Parameters
    ----------
    gff_file : str
        Path to GFF file
    chrom_id : str
        Chromosome ID to filter
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing CDS region spans for each transcript
    """
    logger.info(f"Loading GFF file: {gff_file}")
    
    # Read GFF file using read_gff3 with attribute parsing
    df = read_gff3(gff_file, parse_attributes=True)
    
    # Create mapping of old column names to new snake_case names
    col_mapping = {}
    for col in df.columns:
        new_name = re.sub(r'\s+', '_', col).lower()
        if new_name in col_mapping.values():
            raise ValueError(f"Column name collision detected: multiple columns would map to '{new_name}'")
        col_mapping[col] = new_name
    
    # Rename columns
    df = df.rename(columns=col_mapping)
    
    # Filter for specified chromosome and CDS features
    cds_df = df[(df['seq_id'] == chrom_id) & (df['type'] == 'CDS')]
    
    if cds_df.empty:
        logger.warning(f"No CDS features found for chromosome {chrom_id}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(cds_df)} CDS features for chromosome {chrom_id}")
    
    # Ensure parent column exists
    if 'parent' not in cds_df.columns:
        raise ValueError("GFF file does not contain 'Parent' attribute for CDS features")
    
    # Convert start and end to integers
    cds_df['start'] = cds_df['start'].astype(int)
    cds_df['end'] = cds_df['end'].astype(int)
    
    # Group by parent transcript and get span
    grouped = cds_df.groupby('parent')
    
    # Create CDS regions
    cds_regions = []
    for parent, group in grouped:
        # Find minimum start and maximum end for this transcript's CDS features
        min_start = group['start'].min()
        max_end = group['end'].max()
        strand = group['strand'].iloc[0]  # All CDS features for a transcript should have same strand
        
        # Create a row for this CDS region
        cds_region = {
            'seq_id': chrom_id,
            'source': 'derived',
            'type': 'cds_region',
            'start': min_start,
            'end': max_end,
            'score': '.',
            'strand': strand,
            'phase': '.',
            'parent': parent,
            'size_bp': max_end - min_start + 1
        }
        cds_regions.append(cds_region)
    
    # Create DataFrame of CDS regions
    result_df = pd.DataFrame(cds_regions)
    
    logger.info(f"Created {len(result_df)} CDS regions from {len(cds_df)} CDS features")
    return result_df

def extract_seq_from_fasta(fasta_file: str, chrom_id: str) -> Sequence:
    """Extract the sequence for the specified chromosome from the FASTA file.
    
    Parameters
    ----------
    fasta_file : str
        Path to FASTA file (can be gzipped)
    chrom_id : str
        Chromosome ID to extract
        
    Returns
    -------
    Sequence
        Dataclass containing forward and reverse complement sequences
    """
    logger.info(f"Extracting sequence from FASTA file: {fasta_file}")
    
    # Determine file opening method based on extension
    open_func = gzip.open if fasta_file.endswith(".gz") else open
    mode = "rt" if fasta_file.endswith(".gz") else "r"
    
    # Parse FASTA file
    with open_func(fasta_file, mode) as file:
        for record in SeqIO.parse(file, "fasta"):
            if record.id == chrom_id:
                forward_seq = str(record.seq)
                logger.info(f"Found chromosome {chrom_id} in FASTA file, length: {len(forward_seq)}")
                return Sequence(
                    forward=forward_seq,
                    reverse_complement=str(record.seq.reverse_complement())
                )
    
    raise ValueError(f"Chromosome {chrom_id} not found in FASTA file")

def get_codons(mrna_features: pd.DataFrame, chrom_seq: Sequence) -> Dict[str, Counter]:
    """Extract start and stop codons for each mRNA feature, separated by strand.
    
    Parameters
    ----------
    mrna_features : pd.DataFrame
        DataFrame containing mRNA features
    chrom_seq : Sequence
        Sequence dataclass containing forward and reverse complement sequences
        
    Returns
    -------
    Dict[str, Counter]
        Dictionary with keys 'forward_start', 'forward_stop', 'reverse_start', 'reverse_stop'
        containing corresponding codon counters
    """
    # Initialize separate counters for forward and reverse strands
    forward_start_codons = Counter()
    forward_stop_codons = Counter()
    reverse_start_codons = Counter()
    reverse_stop_codons = Counter()
    
    forward_count = 0
    reverse_count = 0
    
    for _, feature in mrna_features.iterrows():
        # GFF uses 1-based coordinates, convert to 0-based for Python indexing
        start_idx = feature['start'] - 1
        end_idx = feature['end']
        
        if feature['strand'] not in ['+', '-']:
            raise ValueError(f"Invalid strand: {feature['strand']}")

        # Get the correct strand and handle accordingly
        if feature['strand'] == '+':
            forward_count += 1
            start_codon = chrom_seq.forward[start_idx:start_idx+3].upper()
            stop_codon = chrom_seq.forward[end_idx-3:end_idx].upper()
            forward_start_codons[start_codon] += 1
            forward_stop_codons[stop_codon] += 1
        else:  # '-' strand
            reverse_count += 1
            # For negative strand, use the same indexing but with reverse_complement sequence
            start_codon = chrom_seq.reverse_complement[start_idx:start_idx+3].upper()
            stop_codon = chrom_seq.reverse_complement[end_idx-3:end_idx].upper()
            reverse_start_codons[start_codon] += 1
            reverse_stop_codons[stop_codon] += 1
    
    logger.info(f"Analyzed {forward_count} forward strand and {reverse_count} reverse strand mRNA features")
    
    return {
        'forward_start': forward_start_codons,
        'forward_stop': forward_stop_codons,
        'reverse_start': reverse_start_codons,
        'reverse_stop': reverse_stop_codons
    }

def create_codon_stats_df(codon_counts):
    """Convert codon counter dict to a comprehensive DataFrame.
    
    Parameters
    ----------
    codon_counts : Dict[str, Counter]
        Dictionary of codon counters
        
    Returns
    -------
    pd.DataFrame
        DataFrame with codon frequencies and percentages for all categories
    """
    # Create combined counters
    combined_start = codon_counts['forward_start'] + codon_counts['reverse_start']
    combined_stop = codon_counts['forward_stop'] + codon_counts['reverse_stop']
    
    # All possible codons from our data
    all_codons = set()
    for counter in [*codon_counts.values(), combined_start, combined_stop]:
        all_codons.update(counter.keys())
    
    # Create a row for each codon
    rows = []
    for codon in sorted(all_codons):
        row = {'Codon': codon}
        
        # Add counts and percentages for each category
        categories = {
            'Forward Start': codon_counts['forward_start'],
            'Forward Stop': codon_counts['forward_stop'],
            'Reverse Start': codon_counts['reverse_start'],
            'Reverse Stop': codon_counts['reverse_stop'],
            'Combined Start': combined_start,
            'Combined Stop': combined_stop
        }
        
        for cat_name, counter in categories.items():
            count = counter[codon]
            total = sum(counter.values())
            pct = (count / total * 100) if total > 0 else 0
            row[f"{cat_name} Count"] = count
            row[f"{cat_name} %"] = f"{pct:.2f}%"
        
        rows.append(row)
    
    # Create DataFrame
    return pd.DataFrame(rows)

def display_size_distribution(df, label):
    """Display the distribution of feature sizes."""
    print(f"\n{label} Size Distribution (bp):")
    print("=" * 50)
    
    # Get size statistics
    stats = df['size_bp'].describe()
    
    # Print stats in a readable format
    for stat in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        print(f"{stat:8}: {stats[stat]:.1f}")
    
    # Get raw value counts and sort by size
    bins = 16 if df['size_bp'].value_counts().nunique() < 16 else None
    counts = df['size_bp'].value_counts(bins=bins).sort_index()
    
    print("\nMost common sizes (first 16):")
    print(f"{'Size (bp)':<32} | {'Count':<8} | {'Percentage':<10} | Distribution")
    print("-" * 70)
    
    # Display only the first 16 entries
    for size, count in counts.head(16).items():
        percentage = count / len(df) * 100
        bar = '#' * int(percentage)
        print(f"{str(size):<32} | {count:<8} | {percentage:6.2f}%   | {bar}")

def visualize_codon_statistics(stats_df: pd.DataFrame) -> None:
    """Create visualizations for codon statistics.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame containing codon statistics with columns for each category
    """
    # Set a clean, modern style for the plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1])
    
    # Define colors for the plots
    start_color = '#2986cc'  # Blue
    stop_color = '#cc3829'   # Red
    
    # Plot Forward Start codons (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    forward_start = stats_df[stats_df['Category'] == 'Forward Start'].sort_values('Count', ascending=False).head(8)
    ax1.bar(forward_start['Codon'], forward_start['Count'], color=start_color, alpha=0.7)
    ax1.set_title('Forward Start Codons', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12)
    # Add percentage labels on top of bars
    for i, (_, row) in enumerate(forward_start.iterrows()):
        ax1.text(i, row['Count'] + 10, row['Percentage'], ha='center', fontsize=10)
    # Add dominant codon annotation
    if not forward_start.empty and forward_start.iloc[0]['Count'] > 0:
        dominant = forward_start.iloc[0]
        ax1.annotate(f"{dominant['Codon']} ({dominant['Percentage']})",
                    xy=(0, dominant['Count']),
                    xytext=(1.5, dominant['Count'] * 0.8),
                    arrowprops=dict(arrowstyle="->", color='black'),
                    fontsize=12, fontweight='bold')
    
    # Plot Forward Stop codons (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    forward_stop = stats_df[stats_df['Category'] == 'Forward Stop'].sort_values('Count', ascending=False).head(8)
    ax2.bar(forward_stop['Codon'], forward_stop['Count'], color=stop_color, alpha=0.7)
    ax2.set_title('Forward Stop Codons', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12)
    # Add percentage labels on top of bars
    for i, (_, row) in enumerate(forward_stop.iterrows()):
        ax2.text(i, row['Count'] + 10, row['Percentage'], ha='center', fontsize=10)
    
    # Pie chart for Start Codons (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    start_data = stats_df[stats_df['Category'] == 'Forward Start'].sort_values('Count', ascending=False)
    # Group small values into "Other"
    threshold = 0.01  # 1%
    small_indices = start_data['Count'] / start_data['Count'].sum() < threshold
    if any(small_indices):
        other_count = start_data.loc[small_indices, 'Count'].sum()
        other_pct = f"{(other_count / start_data['Count'].sum() * 100):.2f}%"
        main_data = start_data.loc[~small_indices].copy()
        other_row = pd.DataFrame({'Codon': ['Other'], 'Count': [other_count], 
                                'Percentage': [other_pct], 'Category': ['Forward Start']})
        start_pie_data = pd.concat([main_data, other_row], ignore_index=True)
    else:
        start_pie_data = start_data
    
    wedges, texts, autotexts = ax3.pie(
        start_pie_data['Count'], 
        labels=start_pie_data['Codon'],
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.Blues(np.linspace(0.5, 0.8, len(start_pie_data))),
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    ax3.set_title('Start Codon Distribution', fontsize=14, fontweight='bold')
    # Style the text
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    # Pie chart for Stop Codons (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    stop_data = stats_df[stats_df['Category'] == 'Forward Stop'].sort_values('Count', ascending=False)
    # Group small values into "Other"
    small_indices = stop_data['Count'] / stop_data['Count'].sum() < threshold
    if any(small_indices):
        other_count = stop_data.loc[small_indices, 'Count'].sum()
        other_pct = f"{(other_count / stop_data['Count'].sum() * 100):.2f}%"
        main_data = stop_data.loc[~small_indices].copy()
        other_row = pd.DataFrame({'Codon': ['Other'], 'Count': [other_count], 
                                'Percentage': [other_pct], 'Category': ['Forward Stop']})
        stop_pie_data = pd.concat([main_data, other_row], ignore_index=True)
    else:
        stop_pie_data = stop_data
    
    wedges, texts, autotexts = ax4.pie(
        stop_pie_data['Count'], 
        labels=stop_pie_data['Codon'],
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.Reds(np.linspace(0.5, 0.8, len(stop_pie_data))),
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    ax4.set_title('Stop Codon Distribution', fontsize=14, fontweight='bold')
    # Style the text
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    # Add an overall title
    fig.suptitle('Codon Usage Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'prediction_codon_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_path}")
    
    # Show the plot
    plt.show()

def display_codon_statistics(stats_df: pd.DataFrame) -> None:
    """Display codon statistics in both full and summarized formats.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame containing comprehensive codon statistics
    """
    # Create list to store top codons for each category
    top_codon_dfs = []
    
    # Define the categories and their relevant columns
    categories = [
        ('Forward Start', ['Codon', 'Forward Start Count', 'Forward Start %']),
        ('Forward Stop', ['Codon', 'Forward Stop Count', 'Forward Stop %']),
        ('Reverse Start', ['Codon', 'Reverse Start Count', 'Reverse Start %']),
        ('Reverse Stop', ['Codon', 'Reverse Stop Count', 'Reverse Stop %']),
        ('Combined Start', ['Codon', 'Combined Start Count', 'Combined Start %']),
        ('Combined Stop', ['Codon', 'Combined Stop Count', 'Combined Stop %'])
    ]
    
    # Process each category
    for category_name, columns in categories:
        # Create a subset with just the relevant columns
        subset = stats_df[columns].copy()
        
        # Sort by count (descending)
        count_column = columns[1]  # The count column is always the second one
        subset = subset.sort_values(by=count_column, ascending=False)
        
        # Take the top 16 rows
        subset = subset.head(16).reset_index(drop=True)
        
        # Rename columns to standard names
        subset.columns = ['Codon', 'Count', 'Percentage']
        
        # Add category column
        subset['Category'] = category_name
        
        # Add to list
        top_codon_dfs.append(subset)
    
    # Concatenate all dataframes
    result_df = pd.concat(top_codon_dfs, ignore_index=True)
    
    # Display results
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    
    # Print full stats dataframe
    print("\nFull Codon Statistics:")
    print("=" * 80)
    print(stats_df)
    
    # Print top codons by category
    print("\n\nTop 16 Codons by Category:")
    print("=" * 80)
    print(result_df)
    
    # Create visualizations
    visualize_codon_statistics(result_df)

def main():
    parser = argparse.ArgumentParser(description="Analyze start and stop codon usage in coding sequences")
    parser.add_argument("--gff", required=True, help="Path to GFF file")
    parser.add_argument("--fasta", required=True, help="Path to FASTA file")
    parser.add_argument("--chrom", required=True, help="Chromosome ID to analyze")
    parser.add_argument("--min-length", type=int, default=0, help="Minimum CDS length in bp to retain")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.gff):
        raise FileNotFoundError(f"GFF file not found: {args.gff}")
    if not os.path.exists(args.fasta):
        raise FileNotFoundError(f"FASTA file not found: {args.fasta}")
    
    # Load CDS regions extracted from grouped CDS features
    cds_regions = load_cds_features(args.gff, args.chrom)
    
    if cds_regions.empty:
        logger.error(f"No CDS regions found for chromosome {args.chrom}")
        return
    
    # Display the distribution of feature sizes before filtering
    display_size_distribution(cds_regions, "Before Filtering")
    
    # Filter features based on size
    filtered_regions = cds_regions[cds_regions['size_bp'] >= args.min_length]
    
    if filtered_regions.empty:
        logger.error(f"No CDS regions remain after filtering for minimum length {args.min_length}")
        return
    
    # Display the distribution after filtering
    display_size_distribution(filtered_regions, "After Filtering")
    
    logger.info(f"Filtered CDS regions from {len(cds_regions)} to {len(filtered_regions)} (min length: {args.min_length} bp)")
    
    # Extract chromosome sequence
    chrom_seq = extract_seq_from_fasta(args.fasta, args.chrom)
    
    # Get start and stop codons for both strands
    codon_counts = get_codons(filtered_regions, chrom_seq)
    
    # Create comprehensive codon stats DataFrame
    stats_df = create_codon_stats_df(codon_counts)
    
    # Display codon statistics
    display_codon_statistics(stats_df)

if __name__ == "__main__":
    main() 