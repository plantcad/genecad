#!/usr/bin/env python
"""
Analyze the overlap between binary label files to see how often -1, 0, and 1 values occur
together across transcript, CDS, and intron label files.
"""

import numpy as np
import pandas as pd
import argparse


def analyze_label_overlap(transcript_file, cds_file, intron_file):
    """
    Analyze overlaps between values in the three binary label files.
    
    Args:
        transcript_file: Path to the *_binary_transcript_labels.npz file
        cds_file: Path to the *_binary_cds_labels.npz file
        intron_file: Path to the *_binary_intron_labels.npz file
    """
    # Load the label files
    transcript_data = np.load(transcript_file)
    cds_data = np.load(cds_file)
    intron_data = np.load(intron_file)
    
    # Check that all files have the same chromosome keys
    transcript_chroms = set(transcript_data.files)
    cds_chroms = set(cds_data.files)
    intron_chroms = set(intron_data.files)
    
    # Get common chromosomes
    common_chroms = transcript_chroms.intersection(cds_chroms, intron_chroms)
    print(f"Common chromosomes across all files: {', '.join(common_chroms)}")
    
    # For each chromosome
    for chrom in common_chroms:
        transcript_arr = transcript_data[chrom]
        cds_arr = cds_data[chrom]
        intron_arr = intron_data[chrom]
        
        # Check if shapes match
        if not (transcript_arr.shape == cds_arr.shape == intron_arr.shape):
            raise ValueError(
                f"Shape mismatch for chromosome {chrom}: "
                 f"transcript {transcript_arr.shape}, "
                 f"cds {cds_arr.shape}, "
                 f"intron {intron_arr.shape}"
            )
        
        # Report unique values
        print(f"\nChromosome: {chrom}")
        print(f"Transcript unique values: {np.unique(transcript_arr)}")
        print(f"CDS unique values: {np.unique(cds_arr)}")
        print(f"Intron unique values: {np.unique(intron_arr)}")
        
        # Analyze forward strand (column 0)
        print(f"\nAnalyzing forward strand (column 0) for {chrom}:")
        forward_df = analyze_strand_overlap(
            transcript_arr[:, 0], 
            cds_arr[:, 0], 
            intron_arr[:, 0],
            "Forward"
        )
        print(forward_df)
        
        # Analyze reverse strand (column 1)
        print(f"\nAnalyzing reverse strand (column 1) for {chrom}:")
        reverse_df = analyze_strand_overlap(
            transcript_arr[:, 1], 
            cds_arr[:, 1], 
            intron_arr[:, 1],
            "Reverse"
        )
        print(reverse_df)
        
        # Combined analysis (both strands)
        print(f"\nCombined analysis (both strands) for {chrom}:")
        # Flatten arrays for overall statistics
        t_flat = transcript_arr.flatten()
        c_flat = cds_arr.flatten()
        i_flat = intron_arr.flatten()
        
        combined_df = analyze_strand_overlap(t_flat, c_flat, i_flat, "Combined")
        print(combined_df)


def analyze_strand_overlap(transcript_vals, cds_vals, intron_vals, strand_name):
    """
    Analyze overlaps between values for a specific strand.
    
    Args:
        transcript_vals: 1D array of transcript values
        cds_vals: 1D array of CDS values
        intron_vals: 1D array of intron values
        strand_name: Name of the strand for reporting
    
    Returns:
        DataFrame with overlap statistics
    """
    # Create a combined array of [transcript, cds, intron] values
    combined = np.column_stack((transcript_vals, cds_vals, intron_vals))
    
    # Count unique combinations
    unique_combos, counts = np.unique(combined, axis=0, return_counts=True)
    
    # Create a DataFrame for reporting
    rows = []
    total = len(transcript_vals)
    
    for combo, count in zip(unique_combos, counts):
        t_val, c_val, i_val = combo
        rows.append({
            'Transcript': t_val,
            'CDS': c_val,
            'Intron': i_val,
            'Count': count,
            'Percentage': (count / total) * 100
        })
    
    df = pd.DataFrame(rows)
    
    # Sort by count (descending)
    df = df.sort_values('Count', ascending=False)
    
    # Format percentage column
    df['Percentage'] = df['Percentage'].map('{:.4f}%'.format)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Analyze overlaps between binary label files")
    parser.add_argument("--transcript", required=True, help="Path to the *_binary_transcript_labels.npz file")
    parser.add_argument("--cds", required=True, help="Path to the *_binary_cds_labels.npz file")
    parser.add_argument("--intron", required=True, help="Path to the *_binary_intron_labels.npz file")
    
    args = parser.parse_args()
    
    # Analyze the binary labels overlap
    analyze_label_overlap(args.transcript, args.cds, args.intron)


if __name__ == "__main__":
    main() 