#!/usr/bin/env python3
import sys
import argparse
import os
from pathlib import Path

try:
    import pyBigWig
except ImportError:
    print("Error: pyBigWig is not installed. Please try running with: uv run --with pyBigWig python scripts/export_bw.py ...")
    sys.exit(1)

import xarray as xr
import numpy as np
from scipy.special import softmax

# Add src to python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.prediction import merge_prediction_datasets

def main():
    parser = argparse.ArgumentParser(description="Export GeneCAD zarr predictions to BigWig format.")
    parser.add_argument("--input", required=True, help="Path to predictions.zarr directory (containing predictions.*.zarr)")
    parser.add_argument("--output-dir", required=True, help="Directory to save the .bw files")
    parser.add_argument("--feature", default=None, help="Name of the feature to export. If not specified, all features will be exported.")
    parser.add_argument("--strand", choices=["positive", "negative", "both"], default="both", help="Strand to export")
    parser.add_argument("--batch-size", type=int, default=5_000_000, help="Number of records to process at once")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading predictions from {args.input}...")
    try:
        ds = merge_prediction_datasets(args.input, drop_variables=["token_predictions", "token_logits"])
    except Exception as e:
        print(f"Error loading datasets: {e}")
        sys.exit(1)

    strands = ["positive", "negative"] if args.strand == "both" else [args.strand]
    chrom = ds.attrs.get("chromosome_id", "chr")
    sequence_len = ds.sizes["sequence"]
    features = ds.feature.values.tolist() if "feature" in ds.coords else None
    
    if not features:
        print("Error: Unable to find 'feature' coordinates in the dataset.")
        sys.exit(1)
        
    selected_features = [args.feature] if args.feature else features
    
    for strand in strands:
        if strand not in ds.strand.values:
            print(f"Warning: {strand} strand not found in dataset. Skipping.")
            continue
            
        strand_ds = ds.sel(strand=strand)

        for feat in selected_features:
            if feat not in features:
                print(f"Warning: feature '{feat}' not found in predictions. Skipping.")
                continue
                
            feat_idx = features.index(feat)
            out_file = Path(args.output_dir) / f"{chrom}_{strand}_{feat}.bw"
            print(f"Writing {feat} probabilities for {strand} strand to {out_file}...")

            bw = pyBigWig.open(str(out_file), "w")
            bw.addHeader([(chrom, sequence_len)])

            num_batches = int(np.ceil(sequence_len / args.batch_size))
            for i in range(num_batches):
                start = i * args.batch_size
                end = min((i + 1) * args.batch_size, sequence_len)
                
                # Fetch logits for chunk from zarr
                logits = strand_ds.feature_logits.isel(sequence=slice(start, end)).values
                
                # Apply softmax to convert back to probability
                probs = softmax(logits, axis=-1)
                
                # Extract the probability for the feature of interest
                val = probs[:, feat_idx].astype(np.float32)

                starts = np.arange(start, end, dtype=np.int32)
                ends = starts + 1
                
                try:
                    bw.addEntries([chrom] * len(val), starts.tolist(), ends=ends.tolist(), values=val.tolist())
                except Exception as e:
                    print(f"Error adding entries to BigWig: {e}")
                    break
                    
            bw.close()
            print(f"Finished {out_file}")

if __name__ == "__main__":
    main()
