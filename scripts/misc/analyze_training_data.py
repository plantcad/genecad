#!/usr/bin/env python3
"""
Analyze training data by sampling a random window and creating visualizations.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import logging
from src.config import WINDOW_SIZE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(333)

def sample_random_window(ds: xr.Dataset) -> tuple[xr.Dataset, dict]:
    """Sample a single random window from the dataset."""
    n_samples = ds.dims['sample']
    random_idx = np.random.randint(0, n_samples)
    
    # Sample the window
    window = ds.isel(sample=random_idx).compute()
    
    # Extract metadata
    metadata = {
        'sample_index': int(window.sample_index.values),
        'chromosome': str(window.chromosome.values),
        'species': str(window.species.values),
        'strand': str(window.strand.values),
        'position': int(window.position.values)
    }
    
    return window, metadata

def create_class_tracks(labels: np.ndarray, class_map: dict, ax: plt.Axes):
    """Create horizontal tracks for each class in the sequence."""
    # Convert string keys to integers
    class_map_int = {int(k): v for k, v in class_map.items()}
    
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    
    # Check that all classes in data are mapped
    for cls in unique_classes:
        if cls not in class_map_int:
            raise ValueError(f"Class {cls} found in data but not in class mapping. Available classes: {list(class_map_int.keys())}")
    
    # Use a prettier colormap - tab20 for up to 20 classes, viridis for fewer
    if n_classes <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    elif n_classes <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
    
    # Set up the axis
    ax.set_ylim(-0.5, n_classes - 0.5)
    ax.set_xlim(0, len(labels))
    
    # Create class name labels for y-axis
    class_names = [f"{cls}: {class_map_int[cls]}" for cls in unique_classes]
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names)
    
    # Plot each position as a vertical line in the appropriate class track
    for pos, label in enumerate(labels):
        class_idx = np.where(unique_classes == label)[0][0]
        color_idx = class_idx
        
        # Draw vertical line from class track bottom to top
        ax.axvline(x=pos, ymin=(class_idx - 0.4 + 0.5) / n_classes, 
                  ymax=(class_idx + 0.4 + 0.5) / n_classes, 
                  color=colors[color_idx], alpha=0.8, linewidth=0.5)

def create_binary_track(mask: np.ndarray, ax: plt.Axes):
    """Create binary mask visualization."""
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(0, len(mask))
    
    # Set y-axis labels
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Masked (False)', 'Unmasked (True)'])
    
    # Plot each position with prettier colors
    colors = ['#E74C3C', '#2ECC71']  # Modern red and green
    for pos, val in enumerate(mask):
        color = colors[int(val)]
        y_pos = int(val)
        
        # Draw vertical line
        ax.axvline(x=pos, ymin=(y_pos - 0.4 + 0.5) / 2, 
                  ymax=(y_pos + 0.4 + 0.5) / 2, 
                  color=color, alpha=0.8, linewidth=0.5)

def visualize_training_window(window: xr.Dataset, metadata: dict, output_path: Path, gff_df: pd.DataFrame):
    """Create comprehensive visualization of a training window."""
    # Calculate subplot heights based on number of classes
    feature_classes = window.attrs['feature_labels_classes']
    tag_classes = window.attrs['label_classes']
    
    n_feature_classes = len(feature_classes)
    n_tag_classes = len(tag_classes)
    
    # Set figure size and subplot heights
    fig = plt.figure(figsize=(20, 14))
    
    # Create subplots with different heights (made first two shorter)
    gs = fig.add_gridspec(4, 1, height_ratios=[max(1, n_feature_classes * 0.3), max(1, n_tag_classes * 0.3), 2, 3], hspace=0.3)
    
    fig.suptitle(
        f"Training Data Analysis\n"
        f"Species: {metadata['species']}, Chromosome: {metadata['chromosome']}, "
        f"Strand: {metadata['strand']}, Position: {metadata['position']:,}, "
        f"Sample: {metadata['sample_index']}", 
        fontsize=16, fontweight='bold'
    )
    
    sequence_length = len(window.input_ids)
    
    # 1. Feature Labels Track
    ax1 = fig.add_subplot(gs[0, 0])
    feature_labels = window.feature_labels.values
    create_class_tracks(feature_labels, feature_classes, ax1)
    ax1.set_title('Genomic Feature Annotations', fontsize=14, fontweight='bold')
    ax1.set_xticks([])
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Tag Labels Track (taller due to more classes)
    ax2 = fig.add_subplot(gs[1, 0])
    tag_labels = window.tag_labels.values
    create_class_tracks(tag_labels, tag_classes, ax2)
    ax2.set_title('BILUO Tag Annotations', fontsize=14, fontweight='bold')
    ax2.set_xticks([])
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Label Mask (binary)
    ax3 = fig.add_subplot(gs[2, 0])
    label_mask = window.label_mask.values
    create_binary_track(label_mask, ax3)
    ax3.set_title('Training Mask', fontsize=14, fontweight='bold')
    ax3.set_xticks([])
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Raw GFF Features
    ax4 = fig.add_subplot(gs[3, 0])
    gff_features = filter_gff_features(gff_df, metadata['species'], metadata['chromosome'], 
                                      metadata['strand'], metadata['position'])
    create_gff_track(gff_features, metadata['position'], metadata['strand'], ax4)
    ax4.set_title('Raw GFF Features', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Sequence Position', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Format x-axis for bottom plot only
    tick_positions = np.linspace(0, sequence_length, 9)
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels([f'{int(pos):,}' for pos in tick_positions])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    logger.info(f"Visualization saved to {output_path}")

def load_gff_features():
    """Load GFF features from parquet file."""
    gff_path = "/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline/transform/v0.6/intervals.parquet"
    return pd.read_parquet(gff_path)

def filter_gff_features(gff_df: pd.DataFrame, species: str, chromosome: str, strand: str, position: int) -> pd.DataFrame:
    """Filter GFF features for the specified window."""
    window_start = position
    window_end = position + WINDOW_SIZE
    
    # Filter by species, chromosome, and strand
    filtered = gff_df[
        (gff_df['species_id'] == species) &
        (gff_df['chromosome_id'] == chromosome) &
        (gff_df['strand'] == strand)
    ]
    
    # Filter by overlap with window
    overlapping = filtered[
        (filtered['start'] < window_end) & (filtered['stop'] > window_start)
    ]
    
    return overlapping

def create_gff_track(gff_features: pd.DataFrame, window_start: int, strand: str, ax: plt.Axes):
    """Create visualization of GFF features."""
    if gff_features.empty:
        ax.text(0.5, 0.5, 'No GFF features in window', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlim(0, WINDOW_SIZE)
        ax.set_ylim(-0.5, 0.5)
        return
    
    feature_types = gff_features['feature_type'].unique()
    n_types = len(feature_types)
    
    # Use consistent colors
    colors = plt.cm.tab10(np.linspace(0, 1, n_types))
    
    ax.set_ylim(-0.5, n_types - 0.5)
    ax.set_xlim(0, WINDOW_SIZE)
    
    # Set y-axis labels
    ax.set_yticks(range(n_types))
    ax.set_yticklabels(feature_types)
    
    # Plot features
    for i, feature_type in enumerate(feature_types):
        type_features = gff_features[gff_features['feature_type'] == feature_type]
        
        for _, feature in type_features.iterrows():
            # Convert to window coordinates
            start_pos = max(0, feature['start'] - window_start)
            end_pos = min(WINDOW_SIZE, feature['stop'] - window_start)
            
            # For negative strand, flip coordinates within the window
            if strand == 'negative':
                start_pos_flipped = WINDOW_SIZE - 1 - end_pos
                end_pos_flipped = WINDOW_SIZE - 1 - start_pos
                start_pos, end_pos = start_pos_flipped, end_pos_flipped
            
            if start_pos < WINDOW_SIZE and end_pos > 0 and end_pos > start_pos:
                # Draw rectangle for feature
                rect = patches.Rectangle(
                    (start_pos, i - 0.4), end_pos - start_pos, 0.8,
                    facecolor=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5
                )
                ax.add_patch(rect)

def main():
    """Main analysis function."""
    # Load dataset
    data_path = "/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline/prep/v0.6/splits/valid.zarr"
    logger.info(f"Loading dataset from {data_path}")
    ds = xr.open_zarr(data_path)
    
    # Load GFF features once
    logger.info("Loading GFF features...")
    gff_df = load_gff_features()
    
    # Create output directory
    output_dir = Path("local/scratch/training_window_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    # Generate 8 different random windows
    n_windows = 8
    logger.info(f"Generating {n_windows} random training windows...")
    
    for i in range(n_windows):
        logger.info(f"Processing window {i+1}/{n_windows}")
        
        # Sample random window
        window, metadata = sample_random_window(ds)
        logger.info(f"Window {i+1} metadata: {metadata}")
        
        # Create output filename
        output_filename = f"training_window_{i+1:02d}_sample_{metadata['sample_index']}.pdf"
        output_path = output_dir / output_filename
        
        # Create visualization
        visualize_training_window(window, metadata, output_path, gff_df)
        logger.info(f"Saved window {i+1} to {output_filename}")
    
    logger.info(f"Analysis complete! Generated {n_windows} training window visualizations in {output_dir}")

if __name__ == "__main__":
    main() 