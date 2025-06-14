#!/usr/bin/env python3
"""
Visualize genome annotation predictions with ground truth overlays.

This script creates publication-grade visualizations of model predictions
by overlaying true annotations on predicted feature probability heatmaps.

Best windows:

- python scripts/misc/figure_prediction_examples.py --seed 52 --window-size 4608 --offset 768
- python scripts/misc/figure_prediction_examples.py --seed 115 --window-size 2176
- python scripts/misc/figure_prediction_examples.py --seed 124 --window-size 4352
- python scripts/misc/figure_prediction_examples.py --seed 172 --window-size 4608
- python scripts/misc/figure_prediction_examples.py --seed 175 --window-size 8960
- python scripts/misc/figure_prediction_examples.py --seed 189 --window-size 8192 --offset 384
"""

import argparse
import logging
import random
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import softmax

from src.dataset import open_datatree
from src.gff_pandas import read_gff3

# Configure matplotlib for publication-quality plots
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data paths
TRUE_PATH = "/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline/predict/carabica/runs/v0.9.0/chr1/gff/labels.gff"
PRED_PATH = "/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline/predict/carabica/runs/v0.9.0/chr1/intervals.zarr"

def load_data() -> Tuple[pd.DataFrame, object, object]:
    """Load true annotations and prediction data."""
    logger.info("Loading true annotations...")
    true_intervals = read_gff3(TRUE_PATH)
    
    # Convert GFF coordinates from 1-based inclusive to 0-based exclusive on right
    true_intervals = true_intervals.copy()
    true_intervals['start'] = true_intervals['start'] - 1
    
    logger.info("Loading prediction data...")
    dt = open_datatree(PRED_PATH, consolidated=False)
    pred_intervals = dt["/intervals"].ds
    pred_sequences = dt["/sequences"].ds
    
    return true_intervals, pred_intervals, pred_sequences

def select_random_interval(pred_intervals, entity_name: str = "five_prime_utr", strand: str = "positive") -> dict:
    """Randomly select an interval with specified entity_name and strand."""
    # Filter for the desired entity and strand
    filtered = pred_intervals.where(
        (pred_intervals.entity_name == entity_name) & 
        (pred_intervals.strand == strand), 
        drop=True
    )
    
    if len(filtered.interval) == 0:
        raise ValueError(f"No intervals found with entity_name='{entity_name}' and strand='{strand}'")
    
    # Select a random interval
    random_idx = random.choice(filtered.interval.values)
    selected = filtered.sel(interval=random_idx).compute()
    
    interval_data = {
        'start': int(selected.start.values),
        'stop': int(selected.stop.values),
        'strand': str(selected.strand.values),
        'entity_name': str(selected.entity_name.values),
        'interval_idx': int(random_idx)
    }
    
    logger.info(f"Selected interval {interval_data['interval_idx']}: {interval_data['entity_name']} "
                f"at {interval_data['start']}-{interval_data['stop']} ({interval_data['strand']})")
    
    return interval_data

def extract_sequence_window(pred_sequences, interval_data: dict, window_size: int = 8192, offset: int = 256) -> Tuple[np.ndarray, int, int]:
    """Extract sequence window and convert logits to probabilities."""
    start_pos = interval_data['start'] - offset
    end_pos = start_pos + window_size
    
    # Ensure we don't go out of bounds
    seq_len = len(pred_sequences.sequence)
    start_pos = max(0, start_pos)
    end_pos = min(seq_len, end_pos)
    
    logger.info(f"Extracting sequence window: {start_pos}-{end_pos} (length: {end_pos - start_pos})")
    
    # Get strand index (0 for positive, 1 for negative)
    strand_idx = 0 if interval_data['strand'] == 'positive' else 1
    
    # Extract logits for the window
    logits = pred_sequences.feature_logits.sel(
        strand=pred_sequences.strand[strand_idx],
        sequence=slice(start_pos, end_pos)
    ).compute()
    
    # Convert to probabilities using softmax
    probabilities = softmax(logits.values, axis=1)  # softmax over features
    
    return probabilities, start_pos, end_pos

def filter_true_annotations(true_intervals: pd.DataFrame, start_pos: int, end_pos: int, strand: str) -> pd.DataFrame:
    """Filter true annotations to the sequence window."""
    # Convert strand notation
    gff_strand = '+' if strand == 'positive' else '-'
    
    # Filter annotations that overlap with our window, excluding gene, exon, and mRNA
    overlapping = true_intervals[
        (true_intervals['start'] < end_pos) & 
        (true_intervals['end'] > start_pos) &
        (true_intervals['strand'] == gff_strand) &
        (~true_intervals['type'].isin(['gene', 'exon', 'mRNA']))
    ].copy()
    
    # Adjust coordinates to be relative to window start
    overlapping['rel_start'] = np.maximum(0, overlapping['start'] - start_pos)
    overlapping['rel_end'] = np.minimum(end_pos - start_pos, overlapping['end'] - start_pos)
    
    logger.info(f"Found {len(overlapping)} overlapping annotations in window")
    
    return overlapping

def get_feature_colors():
    """Get consistent colors for each feature type using Plotly default colors."""
    return {
        'intergenic': '#ffffff',  # White
        'five_prime_utr': '#2ca02c',  # Green
        'three_prime_utr': '#17becf',  # Cyan
        'cds': '#9467bd',  # Purple
        'intron': '#1f77b4',  # Blue
    }

def get_feature_hatching():
    """Get hatching patterns for each feature type."""
    return {
        'intergenic': None,  # No hatching
        'five_prime_utr': '..',  # Small dots, medium density
        'three_prime_utr': 'oo',  # Small circles, medium density
        'cds': '///',  # Forward slash, high density
        'intron': '\\\\\\',  # Backslash, high density
    }

def predictions_to_intervals(probabilities: np.ndarray, features: list) -> pd.DataFrame:
    """Convert argmax predictions to non-overlapping intervals."""
    predicted = np.argmax(probabilities, axis=1)
    intervals = []
    
    if len(predicted) == 0:
        return pd.DataFrame(columns=['start', 'end', 'feature'])
    
    current_feature = predicted[0]
    start = 0
    
    for i in range(1, len(predicted)):
        if predicted[i] != current_feature:
            intervals.append({
                'start': start,
                'end': i,
                'feature': features[current_feature]
            })
            current_feature = predicted[i]
            start = i
    
    # Add final interval
    intervals.append({
        'start': start,
        'end': len(predicted),
        'feature': features[current_feature]
    })
    
    return pd.DataFrame(intervals)

def annotations_to_intervals(overlapping_annotations: pd.DataFrame, window_length: int) -> pd.DataFrame:
    """Convert true annotations to non-overlapping intervals, filling gaps with intergenic."""
    # Map annotation types to standardized names
    type_mapping = {
        'CDS': 'cds',
        'five_prime_UTR': 'five_prime_utr',
        'three_prime_UTR': 'three_prime_utr',
        'intron': 'intron',
        'intergenic': 'intergenic'
    }
    
    # Get relevant annotations and sort by start position
    annotations = overlapping_annotations[
        overlapping_annotations['type'].isin(type_mapping.keys())
    ].copy()
    annotations = annotations.sort_values('rel_start')
    
    intervals = []
    current_pos = 0
    
    for _, ann in annotations.iterrows():
        # Add intergenic region before this annotation if there's a gap
        if current_pos < ann['rel_start']:
            intervals.append({
                'start': current_pos,
                'end': int(ann['rel_start']),
                'feature': 'intergenic'
            })
        
        # Add the annotation
        intervals.append({
            'start': int(ann['rel_start']),
            'end': int(ann['rel_end']),
            'feature': type_mapping[ann['type']]
        })
        
        current_pos = max(current_pos, int(ann['rel_end']))
    
    # Add final intergenic region if there's space at the end
    if current_pos < window_length:
        intervals.append({
            'start': current_pos,
            'end': window_length,
            'feature': 'intergenic'
        })
    
    return pd.DataFrame(intervals)

def compute_discrepancies(pred_intervals: pd.DataFrame, true_intervals: pd.DataFrame, window_length: int) -> pd.DataFrame:
    """Compute discrepancies between predicted and true intervals."""
    # Create position-wise arrays
    pred_array = np.full(window_length, 'intergenic', dtype=object)
    true_array = np.full(window_length, 'intergenic', dtype=object)
    
    # Fill prediction array
    for _, interval in pred_intervals.iterrows():
        start, end = int(interval['start']), int(interval['end'])
        pred_array[start:end] = interval['feature']
    
    # Fill true annotation array
    for _, interval in true_intervals.iterrows():
        start, end = int(interval['start']), int(interval['end'])
        true_array[start:end] = interval['feature']
    
    # Find discrepancies
    discrepant = pred_array != true_array
    
    # Convert to intervals
    intervals = []
    i = 0
    while i < len(discrepant):
        if discrepant[i]:
            start = i
            while i < len(discrepant) and discrepant[i]:
                i += 1
            intervals.append({
                'start': start,
                'end': i,
                'type': 'discrepancy'
            })
        else:
            i += 1
    
    return pd.DataFrame(intervals)

def plot_interval_track(ax, intervals: pd.DataFrame, colors_dict: dict, hatching_dict: dict, track_name: str):
    """Plot a single track of intervals."""
    # Check that intervals are non-overlapping
    if len(intervals) > 1:
        sorted_intervals = intervals.sort_values('start')
        for i in range(len(sorted_intervals) - 1):
            current_end = sorted_intervals.iloc[i]['end']
            next_start = sorted_intervals.iloc[i + 1]['start']
            if current_end > next_start:
                raise ValueError(f"Overlapping intervals found in {track_name}: "
                               f"interval ending at {current_end} overlaps with interval starting at {next_start}")
    
    ax.set_xlim(0, intervals['end'].max() if len(intervals) > 0 else 1000)
    ax.set_ylim(-0.5, 0.5)
    ax.set_ylabel(track_name, rotation=0, ha='right', va='center', fontsize=14)
    
    for _, interval in intervals.iterrows():
        color = colors_dict.get(interval['feature'], '#808080')  # Default gray
        hatch = hatching_dict.get(interval['feature'], None)  # Default no hatching
        # Add very thin border to all feature rectangles
        rect = patches.Rectangle(
            (interval['start'], -0.5),
            interval['end'] - interval['start'],
            1.0,
            facecolor=color,
            edgecolor='black',
            linewidth=0.3,
            hatch=hatch,
            alpha=0.7
        )
        ax.add_patch(rect)
    
    ax.set_yticks([])
    # Add borders to interval tracks
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('black')

def plot_discrepancy_track(ax, discrepancies: pd.DataFrame, window_length: int):
    """Plot error track (black for errors, white for matches)."""
    ax.set_xlim(0, window_length)
    ax.set_ylim(-0.5, 0.5)
    ax.set_ylabel('Errors', rotation=0, ha='right', va='center', fontsize=14)
    
    for _, interval in discrepancies.iterrows():
        rect = patches.Rectangle(
            (interval['start'], -0.5),
            interval['end'] - interval['start'],
            1.0,
            facecolor='black',
            alpha=0.8,
            edgecolor='none'
        )
        ax.add_patch(rect)
    
    ax.set_yticks([])
    # Add borders to discrepancy track
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('black')

def create_visualization(probabilities: np.ndarray, features: list, overlapping_annotations: pd.DataFrame, 
                        interval_data: dict, start_pos: int, end_pos: int, output_path: Path):
    """Create publication-grade visualization with three tracks."""
    window_length = end_pos - start_pos
    
    # Define consistent ordering for features (reversed to match heatmap)
    feature_order = ['intergenic', 'five_prime_utr', 'cds', 'intron', 'three_prime_utr']
    feature_labels = {
        'intergenic': 'Intergenic',
        'intron': 'Intron', 
        'five_prime_utr': "5' UTR",
        'cds': 'CDS',
        'three_prime_utr': "3' UTR"
    }
    
    # Get colors and hatching patterns for features
    feature_colors = get_feature_colors()
    feature_hatching = get_feature_hatching()
    
    # Create intervals from predictions and annotations
    pred_intervals = predictions_to_intervals(probabilities, features)
    true_intervals = annotations_to_intervals(overlapping_annotations, window_length)
    discrepancies = compute_discrepancies(pred_intervals, true_intervals, window_length)
    
    # Sort features for heatmap
    feature_indices = []
    sorted_feature_labels = []
    for feature in feature_order:
        if feature in features:
            feature_indices.append(features.index(feature))
            sorted_feature_labels.append(feature_labels.get(feature, feature.replace('_', ' ').title()))
    
    # Reorder probabilities to match sorted features
    sorted_probabilities = probabilities[:, feature_indices]
    
    # Create figure with 4 subplots (heatmap + 3 tracks) using GridSpec for better control
    fig = plt.figure(figsize=(16, 6.4))
    gs = fig.add_gridspec(4, 2, height_ratios=[5, 1, 1, 1], width_ratios=[1, 0.03], 
                         hspace=0.1, wspace=0.02)
    
    # Heatmap subplot
    ax_heatmap = fig.add_subplot(gs[0, 0])
    
    # Create custom colormap (white to grey)
    greys_cmap = plt.cm.Greys
    colors_list = [(0.0, 'white'), (0.1, 'white'), (0.15, greys_cmap(0.3)), (1.0, greys_cmap(0.9))]
    custom_cmap = colors.LinearSegmentedColormap.from_list('white_greys', colors_list)
    
    # Plot heatmap
    im = ax_heatmap.imshow(sorted_probabilities.T, aspect='auto', cmap=custom_cmap, 
                          vmin=0, vmax=1, interpolation='nearest')
    
    ax_heatmap.set_yticks(range(len(sorted_feature_labels)))
    ax_heatmap.set_yticklabels(sorted_feature_labels, fontsize=14)
    ax_heatmap.set_title(f'C. arabica Feature Predictions\n'
                        f'Window: chr1:{start_pos:,}-{end_pos:,} ({interval_data["strand"]} strand, {window_length:,} bp)',
                        pad=20, fontsize=14)
    ax_heatmap.set_xticklabels([])
    ax_heatmap.tick_params(axis='x', which='both', length=0)
    ax_heatmap.tick_params(axis='y', which='both', length=0)
    
    # Add borders to heatmap
    for spine in ax_heatmap.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('black')
    
    # Add colorbar in the right column
    cax = fig.add_subplot(gs[0, 1])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Probability', rotation=270, labelpad=20, fontsize=14)
    
    # Create interval track subplots that span the full width of the heatmap
    ax_pred = fig.add_subplot(gs[1, 0])
    ax_true = fig.add_subplot(gs[2, 0])
    ax_disc = fig.add_subplot(gs[3, 0])
    
    # Plot tracks
    plot_interval_track(ax_pred, pred_intervals, feature_colors, feature_hatching, 'Predictions')
    plot_interval_track(ax_true, true_intervals, feature_colors, feature_hatching, 'Annotations')
    plot_discrepancy_track(ax_disc, discrepancies, window_length)
    
    # Set x-axis only on bottom plot
    for ax in [ax_pred, ax_true]:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', length=0)
    
    ax_disc.set_xlabel('Sequence Position (5\'→3\')', fontsize=14)
    
    # Add feature type legend in the right column spanning the bottom 3 tracks
    legend_ax = fig.add_subplot(gs[1:4, 1])
    legend_ax.axis('off')
    
    # Create legend patches
    legend_elements = []
    for feature, color in feature_colors.items():
        if feature in [f for f in feature_order if f in features]:  # Only show features present in data
            label = feature_labels.get(feature, feature.replace('_', ' ').title())
            hatch = feature_hatching.get(feature, None)
            legend_elements.append(patches.Patch(facecolor=color, hatch=hatch, alpha=0.7, 
                                                edgecolor='black', linewidth=0.5, label=label))
    
    legend = legend_ax.legend(handles=legend_elements, loc='center left', frameon=True, 
                             fancybox=False, shadow=False, title='Feature Types', 
                             title_fontsize=14, fontsize=12, bbox_to_anchor=(-0.3, 0.50),
                             handlelength=2.5, handleheight=1.5, handletextpad=0.5, columnspacing=1,
                             borderpad=0.5, labelspacing=0.695, markerscale=6, edgecolor='black')
    
    # Save figure in both PDF and PNG formats
    base_path = output_path.with_suffix('')  # Remove extension
    pdf_path = base_path.with_suffix('.pdf')
    png_path = base_path.with_suffix('.png')
    
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization saved to: {pdf_path} and {png_path}")

def main():
    """Main function to orchestrate the visualization process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=2024, 
                       help='Random seed for reproducibility (default: 2024)')
    parser.add_argument('--window-size', type=int, default=8192,
                       help='Size of sequence window to extract (default: 8192)')
    parser.add_argument('--offset', type=int, default=256,
                       help='Offset upstream from interval start (default: 256)')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path("local/scratch/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        true_intervals, pred_intervals, pred_sequences = load_data()
        
        # Select random interval
        interval_data = select_random_interval(pred_intervals, "five_prime_utr", "positive")
        
        # Extract sequence window and convert to probabilities
        probabilities, start_pos, end_pos = extract_sequence_window(
            pred_sequences, interval_data, window_size=args.window_size, offset=args.offset)
        
        # Get feature names
        features = pred_sequences.feature.values.tolist()
        
        # Filter true annotations for this window
        overlapping_annotations = filter_true_annotations(true_intervals, start_pos, end_pos, interval_data['strand'])
        
        # Create visualization
        output_path = output_dir / f"prediction_example_interval_{interval_data['interval_idx']}.pdf"
        create_visualization(probabilities, features, overlapping_annotations, 
                           interval_data, start_pos, end_pos, output_path)
        
        logger.info("Visualization complete!")
        
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        raise

if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------------------------
# Context
# ------------------------------------------------------------------------------------------------

# In [2]: dt
# Out[2]: 
# <xarray.DataTree>
# Group: /
# ├── Group: /intervals
# │       Dimensions:       (interval: 196478)
# │       Coordinates:
# │         * interval      (interval) int64 2MB 0 1 2 3 4 ... 196474 196475 196476 196477
# │       Data variables:
# │           decoding      (interval) object 2MB ...
# │           entity_index  (interval) int64 2MB ...
# │           entity_name   (interval) object 2MB ...
# │           start         (interval) int64 2MB ...
# │           stop          (interval) int64 2MB ...
# │           strand        (interval) object 2MB ...
# │       Attributes:
# │           interval_entity_names:  ['transcript', 'exon', 'intron', 'five_prime_utr'...
# │           species_id:             Carabica
# │           chromosome_id:          chr1
# │           model_checkpoint:       /scratch/10459/eczech/data/dna/plant_caduceus_gen...
# │           model_path:             kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b...
# └── Group: /sequences
#         Dimensions:              (feature: 5, strand: 2, sequence: 52191927, token: 17)
#         Coordinates:
#           * feature              (feature) <U15 300B 'intergenic' ... 'three_prime_utr'
#           * strand               (strand) object 16B 'positive' 'negative'
#           * sequence             (sequence) int64 418MB 0 1 2 ... 52191925 52191926
#           * token                (token) <U17 1kB 'intergenic' ... 'U-three_prime_utr'
#         Data variables:
#             feature_logits       (strand, sequence, feature) float32 2GB ...
#             feature_predictions  (strand, sequence) int64 835MB ...
#         Attributes:
#             species_id:        Carabica
#             chromosome_id:     chr1
#             model_checkpoint:  /scratch/10459/eczech/data/dna/plant_caduceus_genome_a...
#             model_path:        kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-Npn...

# In [5]: dt["/intervals"].isel(interval=slice(10)).compute()
# Out[5]: 
# <xarray.DataTree 'intervals'>
# Group: /
#     Dimensions:       (interval: 10)
#     Coordinates:
#       * interval      (interval) int64 80B 0 1 2 3 4 5 6 7 8 9
#     Data variables:
#         decoding      (interval) object 80B 'direct' 'direct' ... 'direct' 'direct'
#         entity_index  (interval) int64 80B 1 1 1 1 1 1 1 1 1 1
#         entity_name   (interval) object 80B 'transcript' ... 'transcript'
#         start         (interval) int64 80B 1353728 1365723 ... 1365789 1408278
#         stop          (interval) int64 80B 1356992 1365723 ... 1366878 1408285
#         strand        (interval) object 80B 'positive' 'positive' ... 'positive'
#     Attributes:
#         interval_entity_names:  ['transcript', 'exon', 'intron', 'five_prime_utr'...
#         species_id:             Carabica
#         chromosome_id:          chr1
#         model_checkpoint:       /scratch/10459/eczech/data/dna/plant_caduceus_gen...
#         model_path:             kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b...
# In [7]: dt["/intervals"].entity_name.to_series().value_counts()
# Out[7]: 
# entity_name
# exon               52951
# transcript         44561
# cds                34021
# intron             33613
# three_prime_utr    18861
# five_prime_utr     12471
# Name: count, dtype: int64

# In [18]: true_intervals
# Out[18]: 
#       seq_id       source             type     start       end  score strand  phase                                         attributes                              ID        Parent               product
# 0       chr1  GFF_feature       intergenic  10032205  10127987    NaN    NaN   <NA>                                 ID=intergenic_7747                 intergenic_7747           NaN                   NaN
# 1       chr1  GFF_feature       intergenic  10129398  10129453    NaN    NaN   <NA>                                 ID=intergenic_7748                 intergenic_7748           NaN                   NaN
# 2       chr1  GFF_feature       intergenic  10130399  10259842    NaN    NaN   <NA>                                 ID=intergenic_7749                 intergenic_7749           NaN                   NaN
# 3       chr1  GFF_feature       intergenic  10261250  10298527    NaN    NaN   <NA>                                 ID=intergenic_7750                 intergenic_7750           NaN                   NaN
# 4       chr1  GFF_feature       intergenic  10299146  10299221    NaN    NaN   <NA>                                 ID=intergenic_7751                 intergenic_7751           NaN                   NaN
# ...      ...          ...              ...       ...       ...    ...    ...    ...                                                ...                             ...           ...                   ...
# 46729   chr1  computomics  three_prime_UTR   9871980   9872050    NaN      +   <NA>  ID=CAG013509.t1.three_prime_UTR.1;Parent=CAG01...  CAG013509.t1.three_prime_UTR.1  CAG013509.t1                   NaN
# 46730   chr1  computomics             gene   9993335   9995761    NaN      +   <NA>                                       ID=CAG013516                       CAG013516           NaN                   NaN
# 46731   chr1  computomics             mRNA   9993335   9995761    NaN      +   <NA>  ID=CAG013516.t1;Parent=CAG013516;product=hypot...                    CAG013516.t1     CAG013516  hypothetical protein
# 46732   chr1  computomics             exon   9993335   9995761    NaN      +   <NA>         ID=CAG013516.t1.exon_1;Parent=CAG013516.t1             CAG013516.t1.exon_1  CAG013516.t1                   NaN
# 46733   chr1  computomics              CDS   9993335   9995761    NaN      +      0            ID=CAG013516.t1.cds;Parent=CAG013516.t1                CAG013516.t1.cds  CAG013516.t1                   NaN

# [46734 rows x 12 columns]