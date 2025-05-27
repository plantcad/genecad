#!/usr/bin/env python
"""
Visualize BILUO-encoded labels from gene annotation data.

This script creates visualizations of randomly sampled sequences from the
BILUO-encoded labels, showing gene structure annotations on both DNA strands.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from matplotlib.patches import Rectangle
import random

from src.config import WINDOW_SIZE

# Color scheme for different features
COLOR_SCHEME = {
    'Transcript': '#2ecc71',  # Green
    'CDS': '#e74c3c',        # Red
    'Intron': '#3498db',     # Blue
    'Background': '#ecf0f1',  # Light gray
    'Mask': '#95a5a6'        # Dark gray
}

def get_class_name(class_id, class_labels):
    """Map class IDs to human-readable names."""
    if class_id == -1:
        return "Mask"
    if class_id == 0:
        return "Background"
    
    # Calculate which feature class this belongs to
    feature_idx = (class_id - 1) // 4
    if feature_idx >= len(class_labels):
        return f"Unknown ({class_id})"
    
    # Get BILUO state
    biluo_state = (class_id - 1) % 4
    biluo_prefix = ['B', 'I', 'L', 'U'][biluo_state]
    
    return f"{biluo_prefix}-{class_labels[feature_idx]}"

def get_base_class(class_name):
    """Extract base class name from BILUO-formatted class name."""
    if class_name in ["Mask", "Background"]:
        return class_name
    return class_name.split('-')[1]

def plot_sequence(ax, sequence_data, strand_idx, class_labels, window_size, show_xlabel=True):
    """Plot a single strand of sequence data."""
    y_offset = 0 if strand_idx == 0 else -1
    height = 0.8
    
    current_feature = None
    feature_start = 0
    
    for pos in range(len(sequence_data)):
        class_id = sequence_data[pos]
        class_name = get_class_name(class_id, class_labels)
        base_class = get_base_class(class_name)
        
        # Start new feature or continue current one
        if current_feature != base_class:
            if current_feature is not None:
                # Draw the previous feature
                width = pos - feature_start
                if width > 0:
                    color = COLOR_SCHEME.get(current_feature, '#7f8c8d')
                    rect = Rectangle((feature_start, y_offset), width, height,
                                  facecolor=color, edgecolor='none', alpha=0.7)
                    ax.add_patch(rect)
            
            current_feature = base_class
            feature_start = pos
    
    # Draw the last feature
    if current_feature is not None:
        width = len(sequence_data) - feature_start
        if width > 0:
            color = COLOR_SCHEME.get(current_feature, '#7f8c8d')
            rect = Rectangle((feature_start, y_offset), width, height,
                           facecolor=color, edgecolor='none', alpha=0.7)
            ax.add_patch(rect)
    
    # Set plot limits and labels
    ax.set_xlim(0, window_size)
    ax.set_ylim(-2, 1)
    ax.set_yticks([-0.6, 0.4])
    ax.set_yticklabels(['Reverse', 'Forward'])
    if show_xlabel:
        ax.set_xlabel('Position (bp)')
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)

def create_legend(fig):
    """Create a legend for the plot."""
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7, label=label)
        for label, color in COLOR_SCHEME.items()
    ]
    fig.legend(handles=legend_elements, loc='center right',
              bbox_to_anchor=(0.98, 0.5))

def visualize_random_sequences(data, class_labels, num_samples=8, window_size=1000,
                             output_dir='/tmp/label_visualizations'):
    """Create visualizations for random sequences from the data."""
    os.makedirs(output_dir, exist_ok=True)
    
    chromosomes = list(data.files)
    
    # Create a single figure with subplots
    fig = plt.figure(figsize=(15, 3 * num_samples))
    fig.suptitle('Gene Structure Visualizations\nRandom Samples from Different Chromosomes',
                y=0.98, fontsize=14)
    
    # Add some spacing between subplots
    plt.subplots_adjust(hspace=0.4)
    
    # Generate and plot random samples
    for sample_idx in range(num_samples):
        # Select random chromosome and position
        chrom = random.choice(chromosomes)
        chrom_data = data[chrom]
        max_start = max(0, chrom_data.shape[0] - window_size)
        start_pos = random.randint(0, max_start)
        end_pos = start_pos + window_size
        
        # Create subplot
        ax = plt.subplot(num_samples, 1, sample_idx + 1)
        ax.set_title(f'{chrom}:{start_pos:,}-{end_pos:,}', pad=10)
        
        # Plot both strands
        for strand in range(2):
            sequence_data = chrom_data[start_pos:end_pos, strand]
            show_xlabel = (sample_idx == num_samples - 1)  # Only show xlabel on bottom plot
            plot_sequence(ax, sequence_data, strand, class_labels, window_size, show_xlabel)
    
    # Add a single legend for the entire figure
    create_legend(fig)
    
    # Save the figure
    output_file = os.path.join(output_dir, 'sequence_visualizations.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Combined visualization saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize BILUO-encoded labels")
    parser.add_argument("--input", required=True,
                      help="Path to BILUO-encoded labels (.npz file)")
    parser.add_argument("--output-dir", default="/tmp/label_visualizations",
                      help="Directory to save visualizations")
    parser.add_argument("--num-samples", type=int, default=5,
                      help="Number of random samples to visualize")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                      help="Number of base pairs to show in each visualization")
    parser.add_argument("--class-labels", nargs="+", default=["Transcript", "CDS", "Intron"],
                      help="List of class labels in order")
    
    args = parser.parse_args()
    
    # Load the BILUO-encoded labels
    print(f"Loading BILUO-encoded labels from {args.input}")
    data = np.load(args.input)
    
    print(f"Creating visualization with {args.num_samples} samples...")
    visualize_random_sequences(
        data,
        args.class_labels,
        num_samples=args.num_samples,
        window_size=args.window_size,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 