from src.schema import ModelingFeatureType as MFT
import xarray as xr
import numpy as np
import pandas as pd
from src.sequence import viterbi_decode
from src.prediction import merge_prediction_datasets
from src.modeling import token_transition_probs
from src.naming import normalize_genomic_region_label
from src.decoding import semi_markov_viterbi_decode
from src.analysis import get_sequence_modeling_labels, load_feature_length_distributions
from scipy.special import softmax
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import logging 
from sklearn.metrics import f1_score, accuracy_score
import time

logger = logging.getLogger(__name__)


def normalize_genomic_region_label(label: str, strict: bool = True) -> str | None:
    label = label.lower()
    if label in ["intergenic", "intron", "five_prime_utr", "cds", "three_prime_utr"]:
        return label
    if label == "mrna":
        return "transcript"
    if label == "coding_sequence":
        return "cds"
    if strict:
        raise ValueError(f"Invalid label: {label}")
    return None

def get_viterbi_decoded_path(emissions: np.ndarray, transitions: np.ndarray) -> np.ndarray:
    return viterbi_decode(emissions, transitions, alpha=None)


def get_semi_markov_viterbi_decoded_path(emissions: np.ndarray, transitions: np.ndarray, duration_probs: dict[int, np.ndarray]) -> np.ndarray:
    """Decode using Semi-Markov Viterbi with duration modeling."""
    return semi_markov_viterbi_decode(emissions, transitions, duration_probs)


def get_argmax_decoded_path(emissions: xr.DataArray) -> np.ndarray:
    """Decode using simple argmax of emissions."""
    return emissions.argmax(dim="feature").values


def visualize_duration_probabilities(feature_labels: list[str], raw_probs: dict[int, np.ndarray], 
                                    smoothed_probs: dict[int, np.ndarray], sigma: float, save_path: str = None):
    """Create debugging visualization of duration probabilities before and after smoothing.
    
    Parameters
    ----------
    feature_labels : list[str]
        List of feature labels
    raw_probs : dict[int, np.ndarray]
        Raw probability distributions (no smoothing)
    smoothed_probs : dict[int, np.ndarray]
        Smoothed probability distributions
    sigma : float
        Gaussian smoothing parameter used
    save_path : str, optional
        Path to save the plot (default: None, saves to local/scratch)
    """
    n_features = len(feature_labels)
    fig, axes = plt.subplots(n_features, 2, figsize=(12, 3 * n_features))
    if n_features == 1:
        axes = axes.reshape(1, -1)
    
    for i, feature in enumerate(feature_labels):
        x = np.arange(len(raw_probs[i]))
        
        # Plot raw probabilities
        axes[i, 0].plot(x, raw_probs[i], 'b-', alpha=0.7, linewidth=1, label='Raw')
        axes[i, 0].set_title(f'{feature} - Raw Probabilities')
        axes[i, 0].set_xlabel('Length (bp)')
        axes[i, 0].set_ylabel('Probability')
        if feature != 'intergenic':  # Don't limit x-axis for intergenic
            axes[i, 0].set_xlim(0, 1000)  # Focus on first 1000 bp for visibility
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot smoothed probabilities
        axes[i, 1].plot(x, smoothed_probs[i], 'r-', alpha=0.8, linewidth=2, label=f'Smoothed (σ={sigma})')
        axes[i, 1].set_title(f'{feature} - Smoothed Probabilities')
        axes[i, 1].set_xlabel('Length (bp)')
        axes[i, 1].set_ylabel('Probability')
        if feature != 'intergenic':  # Don't limit x-axis for intergenic
            axes[i, 1].set_xlim(0, 1000)  # Focus on first 1000 bp for visibility
        axes[i, 1].grid(True, alpha=0.3)
        
        # Add some statistics
        raw_max = np.max(raw_probs[i])
        smooth_max = np.max(smoothed_probs[i])
        axes[i, 0].text(0.7, 0.9, f'Max: {raw_max:.4f}', transform=axes[i, 0].transAxes)
        axes[i, 1].text(0.7, 0.9, f'Max: {smooth_max:.4f}', transform=axes[i, 1].transAxes)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = f'local/scratch/refine_comp_lengths_sigma{sigma}.pdf'
    
    plt.savefig(save_path, format='pdf', dpi=150, bbox_inches='tight')
    logger.info(f"Saved length distributions plot to {save_path}")
    plt.show()


def load_duration_probabilities(feature_labels: list[str], sigma: float = 8.0) -> dict[int, np.ndarray]:
    """Load duration probabilities using the centralized function from src.analysis.
    
    Parameters
    ----------
    feature_labels : list[str]
        List of feature labels in the order they appear in the model
    sigma : float
        Standard deviation for Gaussian kernel smoothing (default: 8.0)
        
    Returns
    -------
    dict[int, np.ndarray]
        Dictionary mapping state index to smoothed duration probability array
    """
    path = 'local/data/feature_length_distributions.parquet'
    sigma = 8.0 

    min_lengths = {
        MFT.INTERGENIC: None,
        MFT.INTRON: 3,
        MFT.FIVE_PRIME_UTR: 30,
        MFT.THREE_PRIME_UTR: 30,
        MFT.CDS: 3,
    }

    # Load raw distributions (no smoothing) for visualization
    raw_distributions = load_feature_length_distributions(
        path=path,
        feature_labels=feature_labels,
        min_length=min_lengths,
        sigma=None  # No smoothing for raw data
    )
    
    
    # Load smoothed distributions
    smoothed_distributions = load_feature_length_distributions(
        path=path,
        feature_labels=feature_labels,
        min_length=min_lengths,
        sigma=sigma
    )
    
    # Convert to the format expected by visualization and semi-Markov decoder
    raw_probs_debug = {}
    duration_probs = {}
    for i, feature in enumerate(feature_labels):
        raw_probs_debug[i] = raw_distributions[feature].values
        duration_probs[i] = smoothed_distributions[feature].values
    
    # Log information
    logger.info(f"Loaded duration probabilities for {len(duration_probs)} features (Gaussian smoothing σ={sigma})")
    for i, feature in enumerate(feature_labels):
        max_length = len(duration_probs[i])
        tail_mass = duration_probs[i][-1]
        logger.info(f"  {feature}: {max_length} duration bins, tail mass: {tail_mass:.3f}")
    
    # Create debugging visualization
    visualize_duration_probabilities(feature_labels, raw_probs_debug, duration_probs, sigma)
    
    return duration_probs


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, feature_labels: list[str], method_name: str, mask: np.ndarray = None) -> dict:
    """Compute F1 and accuracy scores for a decoding method.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray  
        Predicted labels
    feature_labels : list[str]
        Feature/class names
    method_name : str
        Name of the decoding method
    mask : np.ndarray, optional
        Boolean mask for valid positions (default: None, use all positions)
        
    Returns
    -------
    dict
        Dictionary with F1 and accuracy scores by class and overall metrics
    """
    # Apply mask if provided
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    # Compute per-class F1 scores
    f1_per_class = f1_score(y_true, y_pred, labels=range(len(feature_labels)), average=None, zero_division=0)
    
    # Compute overall F1 metrics
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Compute overall accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)
    
    # Compute per-class accuracy (fraction of correct predictions for each class)
    accuracy_per_class = []
    for i, label in enumerate(feature_labels):
        class_mask = y_true == i
        if np.any(class_mask):
            class_accuracy = accuracy_score(y_true[class_mask], y_pred[class_mask])
        else:
            class_accuracy = 0.0  # No samples for this class
        accuracy_per_class.append(class_accuracy)
    
    # Create results dictionary
    results = {
        'method': method_name,
        'f1_per_class': dict(zip(feature_labels, f1_per_class)),
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'accuracy_per_class': dict(zip(feature_labels, accuracy_per_class)),
        'accuracy_overall': overall_accuracy,
        'n_samples': len(y_true)
    }
    
    # Log results
    logger.info(f"{method_name} Metrics:")
    logger.info(f"  Samples evaluated: {len(y_true):,}")
    logger.info(f"  Overall accuracy: {overall_accuracy:.3f}")
    for i, (label, f1_score_val, acc_score_val) in enumerate(zip(feature_labels, f1_per_class, accuracy_per_class)):
        logger.info(f"  {label:>15}: F1={f1_score_val:.3f}, Acc={acc_score_val:.3f}")
    logger.info(f"  {'Macro F1':>15}: {f1_macro:.3f}")
    logger.info(f"  {'Weighted F1':>15}: {f1_weighted:.3f}")
    
    return results


def visualize_decoding_validation(emissions, labels, decode1, decode2, decode3, region_labels, mask, save_path):
    """Create visualization comparing viterbi vs argmax vs semi-markov viterbi decoding with emission probabilities."""
    true_labels = np.argmax(labels.values, axis=1)
    
    methods = [
        ('Viterbi decode', decode1, 'red'),
        ('Argmax decode', decode2, 'blue'),
        ('Semi-Markov Viterbi decode', decode3, 'green'),
        ('True labels', true_labels, 'red')
    ]
    
    fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True, 
                            gridspec_kw={'height_ratios': [0.8, 0.8, 0.8, 0.8, 0.15]})
    x_pos = np.arange(len(emissions))
    
    for i, (method_name, decode_path, color) in enumerate(methods):
        ax = axes[i]
        
        # Plot emission probabilities heatmap
        im = ax.imshow(emissions.values.T, aspect='auto', cmap='Blues', vmin=0, vmax=1)
        
        # Plot the decoding path
        ax.plot(x_pos, decode_path, color, linewidth=3, alpha=0.9, label=method_name)
        
        # Add mismatches for first three methods
        if i < 3:
            mismatches = decode_path != true_labels
            if np.any(mismatches):
                ax.scatter(x_pos[mismatches], decode_path[mismatches], 
                          c='orange', s=30, alpha=0.8, zorder=5, label='Mismatch')
        
        ax.set_ylabel('Feature')
        ax.set_yticks(range(len(region_labels)))
        ax.set_yticklabels(region_labels)
        ax.legend(loc='upper right')
        ax.set_title(f'{method_name}')
        plt.colorbar(im, ax=ax, label='Emission Probability')
    
    # Add mask subplot
    mask_ax = axes[4]
    mask_im = mask_ax.imshow(mask.values.reshape(1, -1), aspect='auto', cmap='binary', vmin=0, vmax=1)
    mask_ax.set_ylabel('Mask')
    mask_ax.set_yticks([0])
    mask_ax.set_yticklabels(['Valid'])
    mask_ax.set_title('Label Mask (1=valid, 0=ignore)')
    plt.colorbar(mask_im, ax=mask_ax, label='Mask Value')
    
    axes[-1].set_xlabel('Position (bp)')
    plt.suptitle('Viterbi vs Argmax vs Semi-Markov Viterbi Decoding Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=150, bbox_inches='tight')
    plt.show()


def load_and_align_datasets():
    """Load and align prediction and annotation datasets.
    
    Returns
    -------
    tuple
        (predictions, annotations, aligned_start, aligned_stop, feature_labels, transitions, duration_probs)
    """
    
    # Load predictions
    prediction_path = "/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline/predict/athaliana/runs/v0.5.0/chr1/predictions"
    predictions = merge_prediction_datasets(prediction_path)
    predictions = (
        predictions
        .pipe(lambda ds: ds.assign_coords(feature=[normalize_genomic_region_label(r) for r in ds.feature.values]))
    )
    logger.info(f"Loaded predictions:\n{predictions}")
    
    # Load annotations
    annotations_path = "/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline/transform/v0.4/labels.zarr/Athaliana/chr1"
    annotations = xr.open_zarr(annotations_path)
    sequence_length = annotations.sizes["sequence"]
    annotations = xr.merge([
        get_sequence_modeling_labels(annotations), 
        annotations.label_masks
    ])
    assert sequence_length == annotations.sizes["sequence"]
    logger.info(f"Loaded annotations:\n{annotations}")
    
    # Validate that predictions and annotations have compatible features/regions
    assert set(predictions.feature.values) == set(annotations.feature.values)

    # Load transitions for Viterbi decoding
    transitions = token_transition_probs()
    assert set(predictions.feature.values) == set(transitions.index.values)
    logger.info(f"Loaded transitions:\n{transitions}")
    
    feature_labels = transitions.index.to_list()
    logger.info(f"Feature labels: {feature_labels}")
    
    # Load duration probabilities for semi-Markov decoding
    duration_probs = load_duration_probabilities(feature_labels)
    
    # Find common sequence range between datasets for alignment
    pred_start, pred_stop = int(predictions.sequence.min()), int(predictions.sequence.max())
    anno_start, anno_stop = int(annotations.sequence.min()), int(annotations.sequence.max())
    
    # Use intersection of both ranges
    aligned_start = max(pred_start, anno_start)
    aligned_stop = min(pred_stop, anno_stop)
    
    logger.info(f"Predictions range: {pred_start:,} - {pred_stop:,} ({pred_stop - pred_start:,} bp)")
    logger.info(f"Annotations range: {anno_start:,} - {anno_stop:,} ({anno_stop - anno_start:,} bp)")
    logger.info(f"Aligned range: {aligned_start:,} - {aligned_stop:,} ({aligned_stop - aligned_start:,} bp)")
    
    return predictions, annotations, aligned_start, aligned_stop, feature_labels, transitions, duration_probs


def analyze_target_window(predictions, annotations, feature_labels, transitions, duration_probs,
                         start: int, stop: int, buffer: int = 5000, strand: str = "positive", 
                         window_name: str = "target"):
    """Analyze decoding performance on target validation window.
    
    Parameters
    ----------
    predictions : xarray.Dataset
        Predictions dataset
    annotations : xarray.Dataset
        Annotations dataset
    feature_labels : list[str]
        List of genomic feature labels  
    transitions : xarray.DataArray
        Transition matrix
    duration_probs : dict[int, np.ndarray]
        Duration probabilities for semi-Markov decoding
    start : int
        Start position of window
    stop : int  
        Stop position of window
    buffer : int
        Buffer around window (default: 5000)
    strand : str
        Strand to analyze (default: "positive")
    window_name : str
        Name for this window (for logging/output)
    """
    logger.info(f"Analyzing {window_name} window: {start:,} - {stop:,} (±{buffer:,} buffer) on {strand} strand")
    
    # Extract target window emissions and labels for validation
    emissions = (predictions.feature_logits
        .sel(strand=strand, sequence=slice(start - buffer, stop + buffer), feature=feature_labels)
        .transpose("sequence", "feature")
        .pipe(lambda da: da.copy(data=softmax(da.values, axis=1))))
    
    labels = (annotations.region_labels
        .sel(strand=strand, sequence=slice(start - buffer, stop + buffer), feature=feature_labels)
        .transpose("sequence", "feature"))
    
    mask = (annotations.label_masks
        .sel(strand=strand, sequence=slice(start - buffer, stop + buffer))
        .all(dim="reason"))
    
    # Compare decoding methods
    decode1 = get_viterbi_decoded_path(emissions.values, transitions.values)
    decode2 = get_argmax_decoded_path(emissions)
    
    # Time the semi-Markov Viterbi decoding
    start_time = time.time()
    decode3 = get_semi_markov_viterbi_decoded_path(emissions.values, transitions.values, duration_probs)
    semi_markov_time = time.time() - start_time
    
    window_length = emissions.shape[0]
    logger.info(f"Semi-Markov Viterbi decoding completed in {semi_markov_time:.2f}s for {window_length:,} bp window ({window_length/semi_markov_time:.0f} bp/s)")
    
    # Calculate agreement
    agreement = np.mean(decode1 == decode2)
    logger.info(f"Viterbi vs Argmax agreement in {window_name} window: {agreement:.3f}")
    
    # Calculate agreement with Semi-Markov Viterbi
    viterbi_semi_markov_agreement = np.mean(decode1 == decode3)
    argmax_semi_markov_agreement = np.mean(decode2 == decode3)
    logger.info(f"Viterbi vs Semi-Markov Viterbi agreement: {viterbi_semi_markov_agreement:.3f}")
    logger.info(f"Argmax vs Semi-Markov Viterbi agreement: {argmax_semi_markov_agreement:.3f}")
    
    # Compute F1 scores for all methods
    true_labels = np.argmax(labels.values, axis=1)
    valid_mask = mask.values.astype(bool)  # Convert to boolean mask
    
    logger.info(f"\n=== {window_name.title()} Window Performance Evaluation ===")
    viterbi_results = compute_metrics(true_labels, decode1, feature_labels, "Viterbi", valid_mask)
    argmax_results = compute_metrics(true_labels, decode2, feature_labels, "Argmax", valid_mask)
    semi_markov_results = compute_metrics(true_labels, decode3, feature_labels, "Semi-Markov Viterbi", valid_mask)
    
    # Compare methods
    logger.info(f"\n=== Method Comparison ({window_name} window) ===")
    logger.info(f"  Agreement: {agreement:.3f}")
    logger.info(f"  Viterbi F1 (macro): {viterbi_results['f1_macro']:.3f}")
    logger.info(f"  Argmax F1 (macro): {argmax_results['f1_macro']:.3f}")
    logger.info(f"  Semi-Markov Viterbi F1 (macro): {semi_markov_results['f1_macro']:.3f}")
    logger.info(f"  F1 Improvement: {argmax_results['f1_macro'] - viterbi_results['f1_macro']:+.3f}")
    logger.info(f"  Semi-Markov F1 Improvement (vs Viterbi): {semi_markov_results['f1_macro'] - viterbi_results['f1_macro']:+.3f}")
    logger.info(f"  Semi-Markov F1 Improvement (vs Argmax): {semi_markov_results['f1_macro'] - argmax_results['f1_macro']:+.3f}")
    logger.info(f"  Viterbi Accuracy (overall): {viterbi_results['accuracy_overall']:.3f}")
    logger.info(f"  Argmax Accuracy (overall): {argmax_results['accuracy_overall']:.3f}")
    logger.info(f"  Semi-Markov Viterbi Accuracy (overall): {semi_markov_results['accuracy_overall']:.3f}")
    logger.info(f"  Accuracy Improvement: {argmax_results['accuracy_overall'] - viterbi_results['accuracy_overall']:+.3f}")
    logger.info(f"  Semi-Markov Accuracy Improvement (vs Viterbi): {semi_markov_results['accuracy_overall'] - viterbi_results['accuracy_overall']:+.3f}")
    logger.info(f"  Semi-Markov Accuracy Improvement (vs Argmax): {semi_markov_results['accuracy_overall'] - argmax_results['accuracy_overall']:+.3f}")
    
    # Show per-class differences
    logger.info(f"  Per-class F1 differences (Argmax - Viterbi):")
    for label in feature_labels:
        diff = argmax_results['f1_per_class'][label] - viterbi_results['f1_per_class'][label]
        logger.info(f"    {label:>15}: {diff:+.3f}")
    
    logger.info(f"  Per-class Accuracy differences (Argmax - Viterbi):")
    for label in feature_labels:
        diff = argmax_results['accuracy_per_class'][label] - viterbi_results['accuracy_per_class'][label]
        logger.info(f"    {label:>15}: {diff:+.3f}")
    
    logger.info(f"  Per-class F1 differences (Semi-Markov - Viterbi):")
    for label in feature_labels:
        diff = semi_markov_results['f1_per_class'][label] - viterbi_results['f1_per_class'][label]
        logger.info(f"    {label:>15}: {diff:+.3f}")
    
    logger.info(f"  Per-class Accuracy differences (Semi-Markov - Viterbi):")
    for label in feature_labels:
        diff = semi_markov_results['accuracy_per_class'][label] - viterbi_results['accuracy_per_class'][label]
        logger.info(f"    {label:>15}: {diff:+.3f}")
    
    # Visualize
    save_path = f'local/scratch/refine_comp_validation_{window_name}.pdf'
    logger.info(f"Saving {window_name} window visualization to {save_path}")
    visualize_decoding_validation(emissions, labels, decode1, decode2, decode3, feature_labels, mask, save_path)


def main():
    """Main validation workflow."""
    # Configuration
    strand = "positive"  # Must be "positive" or "negative" as used in Xarray datasets
    
    # Step 1: Load and align datasets
    predictions, annotations, aligned_start, aligned_stop, feature_labels, transitions, duration_probs = load_and_align_datasets()
    
    # Step 2: Analyze validation window
    analyze_target_window(predictions, annotations, feature_labels, transitions, duration_probs,
                         21746353, 21766048, 5000, strand, "long_gene")
    # analyze_target_window(predictions, annotations, feature_labels, transitions, duration_probs,
    #                      5713794, 5722614, 5000, strand, "long_gene")
    
    # Step 3: Analyze another window for comparison (gene-rich region)
    # Use a window from the middle of the aligned data to find genes
    window_size = 21766048 - 21746353  # Same size as validation window
    comparison_window_start = aligned_start + (aligned_stop - aligned_start) // 2
    comparison_window_stop = comparison_window_start + window_size
    
    analyze_target_window(predictions, annotations, feature_labels, transitions, duration_probs,
                         comparison_window_start, comparison_window_stop, 5000, strand, "center_region")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main() 