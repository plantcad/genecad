#!/usr/bin/env python
"""
Merge binary label files into a single file with:
- transcript_labels -> class 1
- cds_labels -> class 4
- intron_labels -> class 7

And add boundary classes:
- class 2: beginning of transcript token
- class 3: end of transcript token
- class 5: beginning of CDS token
- class 6: end of CDS token
- class 8: beginning of intron token
- class 9: end of intron token
"""

import os
import numpy as np
import argparse


def add_boundary_classes(merged, target_class, start_class, end_class):
    """
    Find boundaries for a target class and mark them with start/end classes.
    
    Args:
        merged: The array to modify
        target_class: The class to find boundaries for
        start_class: The class to mark the start positions
        end_class: The class to mark the end positions
    
    Returns:
        Modified array with boundary classes
    """
    # Create a copy to avoid modifying the original
    result = merged.copy()
    
    # Process each strand
    for s in range(merged.shape[1]):
        values = merged[:, s]
        # Preserve masked values
        mask = values == -1
        values = np.where(values == target_class, 1, 0)
        # Find starts and ends of target class
        diffs = np.diff(np.pad(values, (1, 1)))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0] - 1
        # Apply boundary classes
        result[starts, s] = start_class
        result[ends, s] = end_class
        result[mask, s] = -1
    
    return result


def merge_binary_labels(transcript_file, cds_file, intron_file, output_file):
    """
    Merge binary label files into a single file with specific class assignments.
    """
    # Load the label files
    transcript_data = np.load(transcript_file)
    cds_data = np.load(cds_file)
    intron_data = np.load(intron_file)
    
    # Check that all files have the same chromosome keys;
    # make sure to preserve order on iteration
    transcript_chroms = list(transcript_data.files)
    cds_chroms = list(cds_data.files)
    intron_chroms = list(intron_data.files)
    
    print(f"Transcript file chromosomes: {', '.join(transcript_chroms)}")
    print(f"CDS file chromosomes: {', '.join(cds_chroms)}")
    print(f"Intron file chromosomes: {', '.join(intron_chroms)}")
    
    # Check if all files have the same chromosomes
    if not (transcript_chroms == cds_chroms == intron_chroms):
        missing_in_transcript = cds_chroms.union(intron_chroms) - transcript_chroms
        missing_in_cds = transcript_chroms.union(intron_chroms) - cds_chroms
        missing_in_intron = transcript_chroms.union(cds_chroms) - intron_chroms
        
        error_msg = "Chromosome keys do not match across files:\n"
        if missing_in_transcript:
            error_msg += f"  Missing in transcript file: {', '.join(missing_in_transcript)}\n"
        if missing_in_cds:
            error_msg += f"  Missing in CDS file: {', '.join(missing_in_cds)}\n"
        if missing_in_intron:
            error_msg += f"  Missing in intron file: {', '.join(missing_in_intron)}\n"
        
        raise ValueError(error_msg)
    
    # Get chromosome keys
    chromosomes = transcript_chroms
    print(f"Processing {len(chromosomes)} chromosomes: {', '.join(chromosomes)}")
    
    # Create a dictionary to store the merged arrays
    merged_arrays = {}
    
    # Process each chromosome
    for chrom in chromosomes:
        print(f"Processing {chrom}...")
        transcript_arr = transcript_data[chrom]
        cds_arr = cds_data[chrom]
        intron_arr = intron_data[chrom]
        
        # Verify shapes match
        if not (transcript_arr.shape == cds_arr.shape == intron_arr.shape):
            raise ValueError(f"Shape mismatch for chromosome {chrom}: "
                            f"transcript {transcript_arr.shape}, "
                            f"cds {cds_arr.shape}, "
                            f"intron {intron_arr.shape}")
        
        # Report unique values in each array
        for name, arr in [("transcript", transcript_arr), ("cds", cds_arr), ("intron", intron_arr)]:
            unique_vals = np.unique(arr)
            print(f"{name} array for {chrom} contains values: {unique_vals}")
            
            # Check for unexpected values other than -1, 0, 1
            if not all(val in [-1, 0, 1] for val in unique_vals):
                unexpected = [val for val in unique_vals if val not in [-1, 0, 1]]
                raise ValueError(f"{name} array for {chrom} contains unexpected values: {unexpected}")
        
        # Convert -1 values to 0
        transcript_arr = np.where(transcript_arr == -1, 0, transcript_arr)
        cds_arr = np.where(cds_arr == -1, 0, cds_arr)
        intron_arr = np.where(intron_arr == -1, 0, intron_arr)
        
        # Create masks for valid cases
        background = (transcript_arr == 0) & (cds_arr == 0) & (intron_arr == 0)
        transcript = (transcript_arr == 1) & (cds_arr == 0) & (intron_arr == 0)
        cds = (transcript_arr == 1) & (cds_arr == 1) & (intron_arr == 0)
        intron = (transcript_arr == 1) & (cds_arr == 1) & (intron_arr == 1)
        valid_mask = background | transcript | cds | intron
        
        # Initialize merged array with -1 (invalid positions)
        merged = np.full_like(transcript_arr, -1, dtype=np.int8)
        
        # Set base classes
        merged[background] = 0  # Background
        merged[transcript] = 1  # Transcript
        merged[cds] = 2         # CDS
        merged[intron] = 3      # Intron
        
        # Store the merged array
        merged_arrays[chrom] = merged
        
        # Print percentage of valid positions
        valid_percentage = (np.sum(valid_mask) / merged.size) * 100
        print(f"  Valid positions: {valid_percentage:.2f}%")
        
        # Count each case
        print("  Case counts:")
        print(f"    (0,0,0) Background       : {np.sum(background)} positions ({np.sum(background)/merged.size*100:.2f}%)")
        print(f"    (1,0,0) Transcript-only  : {np.sum(transcript)} positions ({np.sum(transcript)/merged.size*100:.2f}%)")
        print(f"    (1,1,0) CDS in Transcript: {np.sum(cds)} positions ({np.sum(cds)/merged.size*100:.2f}%)")
        print(f"    (1,1,1) Intron in CDS    : {np.sum(intron)} positions ({np.sum(intron)/merged.size*100:.2f}%)")
    
    # Save the merged arrays 
    np.savez(output_file, **merged_arrays)
    print(f"Merged labels saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Merge binary label files with specific classes")
    parser.add_argument("--transcript", required=True, help="Path to transcript labels (class 1)")
    parser.add_argument("--cds", required=True, help="Path to CDS labels (class 4)")
    parser.add_argument("--intron", required=True, help="Path to intron labels (class 7)")
    parser.add_argument("--output", required=True, help="Path to save merged labels")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Merge the binary labels
    merge_binary_labels(args.transcript, args.cds, args.intron, args.output)


if __name__ == "__main__":
    main() 