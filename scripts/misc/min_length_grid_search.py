#!/usr/bin/env python3
import os
import argparse
import shutil
import subprocess
import matplotlib.pyplot as plt
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run grid search over min-length values and analyze GFF comparison results"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run grid search over min-length values")
    run_parser.add_argument(
        "--results-dir", 
        default="local/min_length_search", 
        help="Directory to store grid search results (default: local/min_length_search)"
    )
    run_parser.add_argument(
        "--labels-path", 
        required=True,
        help="Path to the reference labels GFF file (required)"
    )
    run_parser.add_argument(
        "--predictions-path", 
        required=True,
        help="Path to the predictions GFF file (required)"
    )
    run_parser.add_argument(
        "--gffcompare-path",
        required=True,
        help="Path to gffcompare executable (required)"
    )
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize grid search results")
    viz_parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing grid search results (required)"
    )
    viz_parser.add_argument(
        "--output-file",
        default=None,
        help="Output file for visualization (default: f1_scores.png in results directory)"
    )
    
    return parser.parse_args()

def run_grid_search(args):
    """Run grid search over min-length values."""
    # Configuration
    min_lengths = [0, 1, 3, 9, 15, 21, 30, 48, 60, 81, 120, 243, 729, 999]
    results_dir = args.results_dir
    labels_path = args.labels_path
    predictions_path = args.predictions_path
    gffcompare_path = args.gffcompare_path

    # Ensure output directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract filenames from paths for more descriptive naming
    labels_filename = os.path.basename(labels_path)
    predictions_filename = os.path.basename(predictions_path)
    
    # Copy input files to results directory
    labels_copy_path = os.path.join(results_dir, "reference_" + labels_filename)
    predictions_copy_path = os.path.join(results_dir, "original_" + predictions_filename)
    
    print(f"Copying reference labels to {labels_copy_path}")
    shutil.copy2(labels_path, labels_copy_path)
    
    print(f"Copying original predictions to {predictions_copy_path}")
    shutil.copy2(predictions_path, predictions_copy_path)

    # Run grid search over min-lengths
    results = []
    for min_length in min_lengths:
        print(f"Processing min_length = {min_length}")
        
        # Create output paths with descriptive names
        filtered_path = os.path.join(results_dir, f"predictions_min_length_{min_length}.gff")
        comparison_dir = os.path.join(results_dir, f"comparison_min_length_{min_length}")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Filter by min-length
        subprocess.run([
            "python", "scripts/gff.py", "filter_to_min_length",
            "--input", predictions_path,
            "--output", filtered_path,
            "--min-length", str(min_length)
        ], check=True)
        
        # Run comparison
        subprocess.run([
            "python", "scripts/gff.py", "compare",
            "--reference", labels_path,
            "--input", filtered_path,
            "--output", comparison_dir,
            "--gffcompare-path", gffcompare_path
        ], check=True)
        
        # Load results
        stats_df = pd.read_csv(os.path.join(comparison_dir, "gffcompare.stats.csv"), sep='\t')
        stats_df['min_length'] = min_length
        results.append(stats_df)

    # Merge results and save
    merged_df = pd.concat(results)
    
    # Save combined results
    merged_df.to_csv(os.path.join(results_dir, "merged_results.csv"), index=False)
    
    print(f"\nGrid search complete. Results saved to {os.path.join(results_dir, 'merged_results.csv')}")
    print("Use the 'visualize' command to analyze the results.")
    
    return merged_df

def visualize_results(args):
    """Visualize grid search results with a focus on F1 scores."""
    results_dir = args.results_dir
    output_file = args.output_file or os.path.join(results_dir, "f1_scores.png")
    
    # Load results
    results_file = os.path.join(results_dir, "merged_results.csv")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    df = pd.read_csv(results_file)
    
    # Print summary of key metrics by min_length
    print("\nResults Summary:")
    
    # Show each metric separately
    for metric in ['sensitivity', 'precision', 'f1_score']:
        summary = df.pivot(index='level', columns='min_length', values=metric)
        print(f"\n{metric.upper()}:")
        print(summary)
    
    # Create pivot tables for plotting
    f1_scores = df.pivot(index='min_length', columns='level', values='f1_score')
    
    # Plot F1 scores
    plt.figure(figsize=(14, 8))
    
    # Use log scale for x-axis to better visualize the range of min_length values
    for level in f1_scores.columns:
        plt.plot(f1_scores.index, f1_scores[level], marker='o', label=level)
    
    # Set up the log scale x-axis properly with custom ticks and labels
    plt.xscale('log')
    
    # Explicitly set which values to show as tick marks
    min_length_values = f1_scores.index.tolist()
    
    # Format the axis with clear labels
    plt.gca().set_xticks(min_length_values)
    plt.gca().set_xticklabels([str(x) for x in min_length_values], rotation=45)
    
    # Add vertical grid lines at each tick position
    plt.grid(True, which='major', axis='x', linestyle='-', alpha=0.3)
    
    # Ensure axis limits give enough room for labels
    plt.xlim([min(min_length_values) * 0.8, max(min_length_values) * 1.2])
    
    # Formatting and labels
    plt.xlabel('Minimum Length (log scale)')
    plt.ylabel('F1 Score')
    plt.title('F1 Scores at Different Levels vs. Minimum Length')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Add a horizontal line at y=0.5 for reference
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"F1 score visualization saved to {output_file}")
    
    # Create a table of best min_length values for each level
    best_min_lengths = {}
    for level in f1_scores.columns:
        best_idx = f1_scores[level].idxmax()
        best_min_lengths[level] = {
            'min_length': best_idx,
            'f1_score': f1_scores[level][best_idx]
        }
    
    # Print the best min_length for each level
    print("\nBest minimum length for each level:")
    for level, stats in best_min_lengths.items():
        print(f"Level {level}: min_length = {stats['min_length']}, F1 score = {stats['f1_score']:.4f}")
    
    # Create and save summary table
    summary_data = []
    for level, stats in best_min_lengths.items():
        # Get precision and sensitivity for the best min_length
        best_row = df[(df['level'] == level) & (df['min_length'] == stats['min_length'])]
        sensitivity = best_row['sensitivity'].values[0]
        precision = best_row['precision'].values[0]
        
        summary_data.append({
            'level': level,
            'best_min_length': stats['min_length'],
            'f1_score': stats['f1_score'],
            'sensitivity': sensitivity,
            'precision': precision
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(results_dir, "best_min_lengths.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary of best min_lengths saved to {summary_file}")

def main():
    args = parse_args()
    
    if args.command == "run":
        run_grid_search(args)
    elif args.command == "visualize":
        visualize_results(args)
    else:
        print(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()