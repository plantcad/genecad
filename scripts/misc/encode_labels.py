#!/usr/bin/env python
"""
Convert merged labels to BILUO format with Numba acceleration.

This script takes the output from merge_binary_labels.py (classes -1, 0, 1, 2, 3)
and converts them to BILUO (Begin, Inside, Last, Unit, Outside) formatted tags.
"""

import os
import numpy as np
import argparse
import pandas as pd
from src.sequence import convert_to_biluo_labels

def compute_run_lengths(class_indices):
    """
    Compute run lengths for each class in the array.
    
    Args:
        class_indices: numpy array of BILUO-encoded class indices
        
    Returns:
        dict: Dictionary mapping class values to their run lengths
    """
    # Find where values change
    change_points = np.where(np.diff(class_indices) != 0)[0]
    
    # Add start and end points
    run_edges = np.concatenate(([0], change_points + 1, [len(class_indices)]))
    
    # Calculate run lengths
    run_lengths = np.diff(run_edges)
    
    # Get the class value for each run
    run_values = class_indices[run_edges[:-1]]
    
    # Initialize dictionary to store lengths
    lengths_by_class = {}
    
    # Store lengths for each unique class
    for c in np.unique(class_indices):
        lengths_by_class[c] = run_lengths[run_values == c]
    
    return lengths_by_class


def main():
    parser = argparse.ArgumentParser(description="Convert merged labels to BILUO format")
    parser.add_argument("--input", required=True, help="Path to merged labels")
    parser.add_argument("--output", required=True, help="Path to save BILUO-encoded labels")
    parser.add_argument("--class-labels", nargs="+", help="List of class labels")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the merged labels
    print(f"Loading merged labels from {args.input}")
    merged_data = np.load(args.input)
    
    # Get chromosome keys
    chromosomes = list(merged_data.files)
    print(f"Processing {len(chromosomes)} chromosomes: {', '.join(chromosomes)}")
    
    # Create a dictionary to store the BILUO-encoded arrays
    biluo_arrays = {}
    
    # Dictionary to store class counts and lengths
    class_counts = {}
    class_lengths = {}  # Changed from class_length_stats
    total_positions = 0
    
    # Process each chromosome
    for chrom in chromosomes:
        print(f"Processing {chrom}...")
        merged_arr = merged_data[chrom]
        assert merged_arr.ndim == 2
        assert merged_arr.shape[1] == 2
        
        # Process each strand separately
        for s in range(merged_arr.shape[1]):
            print(f"  Converting strand {s} to BILUO format...")
            class_indices = merged_arr[:, s]
            
            # For reverse strand (s=1), process in reverse order
            if s == 1:
                class_indices = class_indices[::-1]
                
            # Convert to BILUO format
            biluo_indices = convert_to_biluo_labels(class_indices)
            
            # Compute run lengths for this strand
            strand_lengths = compute_run_lengths(biluo_indices)
            
            # Merge lengths with existing lengths
            for class_id, lengths in strand_lengths.items():
                if class_id not in class_lengths:
                    class_lengths[class_id] = lengths
                else:
                    class_lengths[class_id] = np.concatenate([class_lengths[class_id], lengths])
            
            # For reverse strand, reverse the results back
            if s == 1:
                biluo_indices = biluo_indices[::-1]
                
            # Update the array and counts
            merged_arr[:, s] = biluo_indices
            values, counts = np.unique(biluo_indices, return_counts=True)
            for value, count in zip(values, counts):
                class_counts[value] = class_counts.get(value, 0) + count
            
            total_positions += len(biluo_indices)
        
        # Store the BILUO-encoded array
        biluo_arrays[chrom] = merged_arr
        
        # Print unique values in the encoded array
        unique_vals = np.unique(merged_arr)
        print(f"  BILUO-encoded array for {chrom} contains values: {unique_vals}")
    
    # Save the BILUO-encoded arrays
    np.savez(args.output, **biluo_arrays)
    print(f"BILUO-encoded labels saved to {args.output}")
    
    # Create class name mapping if class_labels were provided
    class_name_map = {}
    if args.class_labels:
        # Special cases for mask and background
        class_name_map[-1] = "Mask"
        class_name_map[0] = "Background"
        
        # For each class (starting from 1), map to the BILUO versions
        for i, label in enumerate(args.class_labels, start=1):
            base_idx = 1 + (i-1)*4  # Formula from convert_to_biluo
            class_name_map[base_idx] = f"B-{label}"      # Begin
            class_name_map[base_idx+1] = f"I-{label}"    # Inside
            class_name_map[base_idx+2] = f"L-{label}"    # Last
            class_name_map[base_idx+3] = f"U-{label}"    # Unit
    
    # Compute final statistics across all chromosomes and strands
    summary_data = []
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_positions) * 100
        class_name = class_name_map.get(class_id, "")
        
        # Compute statistics from all lengths
        lengths = class_lengths.get(class_id, np.array([]))
        stats = pd.Series(lengths).describe()
        summary_data.append({
            "Class ID": class_id,
            "Class Name": class_name,
            "Count": count,
            "Percentage (%)": percentage,
            "N Runs": len(lengths),
            "Mean Length": f"{stats['mean']:.2f}",
            "Std Length": f"{stats['std']:.2f}",
            "Min Length": f"{stats['min']:.0f}",
            "Median Length": f"{stats['50%']:.0f}",
            "Max Length": f"{stats['max']:.0f}"
        })
    
    # Create and format the DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_df["Count"] = summary_df["Count"].apply(lambda x: f"{x:,}")
    summary_df["N Runs"] = summary_df["N Runs"].apply(lambda x: f"{x:,}")
    summary_df["Percentage (%)"] = summary_df["Percentage (%)"].apply(lambda x: f"{x:.4f}")
    
    # Print the summary
    print("\nClass Frequency and Length Statistics Summary:")
    print("--------------------------------------------")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    print(summary_df.to_string(index=False))
    
    print(f"\nTotal positions processed: {total_positions:,}")

if __name__ == "__main__":
    main()
