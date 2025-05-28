import xarray as xr
import numpy as np
from src.crf import CRF, TrainingSequence, FeatureSet, GenomicFeatureSet, Adam
from src.sequence import viterbi_decode
from src.prediction import merge_prediction_datasets
from src.modeling import token_transition_probs
from src.naming import normalize_genomic_region_label
from scipy.special import softmax
import matplotlib.pyplot as plt
import logging 
from sklearn.metrics import f1_score, classification_report, accuracy_score

logger = logging.getLogger(__name__)


def get_viterbi_decoded_path(emissions: np.ndarray, transitions: np.ndarray) -> np.ndarray:
    return viterbi_decode(emissions, transitions, alpha=None)


def get_crf_decoded_path(emissions: np.ndarray, crf: CRF, epsilon: float | None = None) -> np.ndarray:
    """Decode using the trained CRF."""
    features = GenomicFeatureSet(emissions)
    return crf.decode(emissions, features)


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


def visualize_crf_validation(emissions, labels, decode1, decode2, region_labels, mask, save_path):
    """Create visualization comparing viterbi vs CRF decoding with emission probabilities."""
    true_labels = np.argmax(labels.values, axis=1)
    
    methods = [
        ('Viterbi decode', decode1, 'red'),
        ('CRF decode', decode2, 'blue'),
        ('True labels', true_labels, 'red')
    ]
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True, 
                            gridspec_kw={'height_ratios': [0.8, 0.8, 0.8, 0.15]})
    x_pos = np.arange(len(emissions))
    
    for i, (method_name, decode_path, color) in enumerate(methods):
        ax = axes[i]
        
        # Plot emission probabilities heatmap
        im = ax.imshow(emissions.values.T, aspect='auto', cmap='Blues', vmin=0, vmax=1)
        
        # Plot the decoding path
        ax.plot(x_pos, decode_path, color, linewidth=3, alpha=0.9, label=method_name)
        
        # Add mismatches for first two methods
        if i < 2:
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
    mask_ax = axes[3]
    mask_im = mask_ax.imshow(mask.values.reshape(1, -1), aspect='auto', cmap='binary', vmin=0, vmax=1)
    mask_ax.set_ylabel('Mask')
    mask_ax.set_yticks([0])
    mask_ax.set_yticklabels(['Valid'])
    mask_ax.set_title('Label Mask (1=valid, 0=ignore)')
    plt.colorbar(mask_im, ax=mask_ax, label='Mask Value')
    
    axes[-1].set_xlabel('Position (bp)')
    plt.suptitle('CRF vs Viterbi Decoding Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=150, bbox_inches='tight')
    plt.show()


def load_and_align_datasets():
    """Load and align prediction and annotation datasets.
    
    Returns
    -------
    tuple
        (predictions, annotations, aligned_start, aligned_stop, feature_labels, transitions)
    """
    
    # Load predictions (same as analyze_crf_refinements.py)
    prediction_path = "/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline/predict/athaliana/runs/v0.5.0/chr1/predictions"
    predictions = merge_prediction_datasets(prediction_path)
    predictions = (
        predictions
        .pipe(lambda ds: ds.assign_coords(feature=[normalize_genomic_region_label(r) for r in ds.feature.values]))
    )
    logger.info(f"Loaded predictions:\n{predictions}")
    
    # Load annotations (same as analyze_crf_refinements.py)
    annotations_path = "/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline/transform/v0.4/labels.zarr/Athaliana/chr1"
    annotations = xr.open_zarr(annotations_path)
    annotations = (
        annotations
        .pipe(lambda ds: ds.sel(region=list(
            e for e in ds.region.values
            if normalize_genomic_region_label(e, strict=False) is not None
        )))
        .pipe(lambda ds: (
            ds.assign_coords(region=[normalize_genomic_region_label(e) for e in ds.region.values])
        ))
    )
    logger.info(f"Loaded annotations:\n{annotations}")
    
    # Validate that predictions and annotations have compatible features/regions
    assert set(predictions.feature.values) - {"intergenic"} == set(annotations.region.values)
    
    # Load transitions from modeling.py (same as analyze_crf_refinements.py)
    transitions = token_transition_probs()
    assert set(predictions.feature.values) == set(transitions.index.values)
    logger.info(f"Loaded transitions:\n{transitions}")
    
    feature_labels = transitions.index.to_list()
    # feature_labels: ['intergenic', 'intron', 'five_prime_utr', 'cds', 'three_prime_utr']
    
    # Find common sequence range between datasets for alignment
    pred_start, pred_stop = int(predictions.sequence.min()), int(predictions.sequence.max())
    anno_start, anno_stop = int(annotations.sequence.min()), int(annotations.sequence.max())
    
    # Use intersection of both ranges
    aligned_start = max(pred_start, anno_start)
    aligned_stop = min(pred_stop, anno_stop)
    
    logger.info(f"Predictions range: {pred_start:,} - {pred_stop:,} ({pred_stop - pred_start:,} bp)")
    logger.info(f"Annotations range: {anno_start:,} - {anno_stop:,} ({anno_stop - anno_start:,} bp)")
    logger.info(f"Aligned range: {aligned_start:,} - {aligned_stop:,} ({aligned_stop - aligned_start:,} bp)")
    
    return predictions, annotations, aligned_start, aligned_stop, feature_labels, transitions


def train_crf_model(
        predictions, annotations, aligned_start, aligned_stop, 
        feature_labels, transitions, strand: str, 
        training_fraction: float = 1.0,
):
    """Train the CRF model on aligned datasets.
    
    Parameters
    ----------
    predictions : xarray.Dataset
        Predictions dataset
    annotations : xarray.Dataset  
        Annotations dataset
    aligned_start : int
        Start of aligned sequence range
    aligned_stop : int
        Stop of aligned sequence range
    feature_labels : list[str]
        List of genomic feature labels
    transitions : xarray.DataArray
        Transition matrix
    strand : str
        Strand to analyze, must be either "positive" or "negative"
    training_fraction : float
        Fraction of available data to use for training (default: 1.0).
        Will select a contiguous subset of the aligned range.
        
    Returns
    -------
    CRF
        Trained CRF model
    """
    if strand not in ["positive", "negative"]:
        raise ValueError(f"strand must be either 'positive' or 'negative', got '{strand}'")
    
    if not 0 < training_fraction <= 1.0:
        raise ValueError(f"training_fraction must be between 0 and 1, got {training_fraction}")
    
    # Calculate training slice based on fraction
    total_length = aligned_stop - aligned_start
    training_length = int(total_length * training_fraction)
    
    if training_fraction < 1.0:
        # Select contiguous subset from beginning of aligned range (avoid centromeric regions)
        start_offset = 0
        training_start = aligned_start + start_offset
        training_stop = training_start + training_length
        training_slice = slice(training_start, training_stop)
        logger.info(f"Using training fraction {training_fraction:.2f}: selecting {training_length:,} contiguous positions from {training_start:,} to {training_stop:,}")
    else:
        # Use full aligned range
        training_slice = slice(aligned_start, aligned_stop)
        logger.info(f"Using full training data: {total_length:,} contiguous positions from {aligned_start:,} to {aligned_stop:,}")
    
    # Extract training emissions (contiguous sequence)
    training_emissions = (predictions.feature_logits
        .sel(strand=strand, sequence=training_slice, feature=feature_labels)
        .transpose("sequence", "feature")
        .pipe(lambda da: da.copy(data=softmax(da.values, axis=1))))
    
    # Extract training labels (contiguous sequence)
    training_labels = (annotations.region_labels
        .pipe(lambda da: xr.concat([
            (1 - da.max(dim='region')).expand_dims("region").assign_coords(region=["intergenic"]), 
            da], dim='region'))
        .rename(region="feature") 
        .sel(strand=strand, sequence=training_slice, feature=feature_labels)
        .transpose("sequence", "feature")
        .values.argmax(axis=1))
    
    # Verify aligned shapes
    logger.info(f"Training emissions shape: {training_emissions.shape}")
    logger.info(f"Training labels shape: {training_labels.shape}")
    assert training_emissions.shape[0] == training_labels.shape[0], \
        f"Shape mismatch: emissions {training_emissions.shape[0]} vs labels {training_labels.shape[0]}"
    
    # Create training sequence
    training_seq = TrainingSequence(
        emissions=training_emissions.values,
        true_path=training_labels
    )
    
    # Create CRF with Adam optimizer
    num_states = len(feature_labels)
    
    logger.info(f"CRF dimensions: num_states={num_states}")
    
    crf = CRF(
        num_states=num_states,
        optimizer=Adam(learning_rate=0.1, beta1=0.9, beta2=0.999),
        base_transitions=np.log(transitions.values + 1e-10)
    )
    
    # Print weight info before training
    logger.info("Training CRF model with Adam optimizer...")
    
    # Train the CRF
    crf.fit(
        [training_seq], GenomicFeatureSet, 
        num_epochs=100,
        verbose=True,
        chunk_size=131_072,
        num_workers=32,  # Use multiprocessing for feature computation
        intergenic_idx=0  # intergenic is the first state
    )
    
    return crf


def analyze_target_window(predictions, annotations, crf, feature_labels, transitions, 
                         start: int, stop: int, buffer: int = 5000, strand: str = "positive", 
                         window_name: str = "target"):
    """Analyze CRF performance on target validation window.
    
    Parameters
    ----------
    predictions : xarray.Dataset
        Predictions dataset
    annotations : xarray.Dataset
        Annotations dataset
    crf : CRF
        Trained CRF model
    feature_labels : list[str]
        List of genomic feature labels  
    transitions : xarray.DataArray
        Transition matrix
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
        .pipe(lambda da: xr.concat([
            (1 - da.max(dim='region')).expand_dims("region").assign_coords(region=["intergenic"]), 
            da], dim='region'))
        .rename(region="feature")
        .sel(strand=strand, sequence=slice(start - buffer, stop + buffer), feature=feature_labels)
        .transpose("sequence", "feature"))
    
    mask = (annotations.label_masks
        .sel(strand=strand, sequence=slice(start - buffer, stop + buffer))
        .min(dim="reason"))
    
    # Compare decoding methods (using default machine epsilon)
    decode1 = get_viterbi_decoded_path(emissions.values, transitions.values)
    decode2 = get_crf_decoded_path(emissions.values, crf)
    
    # Calculate agreement
    agreement = np.mean(decode1 == decode2)
    logger.info(f"Viterbi vs CRF agreement in {window_name} window: {agreement:.3f}")
    
    # Compute F1 scores for both methods
    true_labels = np.argmax(labels.values, axis=1)
    valid_mask = mask.values.astype(bool)  # Convert to boolean mask
    
    logger.info(f"\n=== {window_name.title()} Window Performance Evaluation ===")
    viterbi_results = compute_metrics(true_labels, decode1, feature_labels, "Viterbi", valid_mask)
    crf_results = compute_metrics(true_labels, decode2, feature_labels, "CRF", valid_mask)
    
    # Compare methods
    logger.info(f"\n=== Method Comparison ({window_name} window) ===")
    logger.info(f"  Agreement: {agreement:.3f}")
    logger.info(f"  Viterbi F1 (macro): {viterbi_results['f1_macro']:.3f}")
    logger.info(f"  CRF F1 (macro): {crf_results['f1_macro']:.3f}")
    logger.info(f"  F1 Improvement: {crf_results['f1_macro'] - viterbi_results['f1_macro']:+.3f}")
    logger.info(f"  Viterbi Accuracy (overall): {viterbi_results['accuracy_overall']:.3f}")
    logger.info(f"  CRF Accuracy (overall): {crf_results['accuracy_overall']:.3f}")
    logger.info(f"  Accuracy Improvement: {crf_results['accuracy_overall'] - viterbi_results['accuracy_overall']:+.3f}")
    
    # Show per-class differences
    logger.info(f"  Per-class F1 differences (CRF - Viterbi):")
    for label in feature_labels:
        diff = crf_results['f1_per_class'][label] - viterbi_results['f1_per_class'][label]
        logger.info(f"    {label:>15}: {diff:+.3f}")
    
    logger.info(f"  Per-class Accuracy differences (CRF - Viterbi):")
    for label in feature_labels:
        diff = crf_results['accuracy_per_class'][label] - viterbi_results['accuracy_per_class'][label]
        logger.info(f"    {label:>15}: {diff:+.3f}")
    
    # Visualize
    save_path = f'local/scratch/crf_validation_comparison_{window_name}.pdf'
    logger.info(f"Saving {window_name} window visualization to {save_path}")
    visualize_crf_validation(emissions, labels, decode1, decode2, feature_labels, mask, save_path)


def main():
    """Main validation workflow."""
    # Configuration
    strand = "positive"  # Must be "positive" or "negative" as used in Xarray datasets
    
    # Step 1: Load and align datasets
    predictions, annotations, aligned_start, aligned_stop, feature_labels, transitions = load_and_align_datasets()
    
    # Step 2: Train CRF model  
    crf = train_crf_model(
        predictions=predictions, 
        annotations=annotations, 
        aligned_start=aligned_start, 
        aligned_stop=aligned_stop, 
        feature_labels=feature_labels, 
        transitions=transitions, 
        strand=strand, 
        training_fraction=0.5
    )
    
    # Step 3: Analyze validation window (original target, not used in training)
    analyze_target_window(predictions, annotations, crf, feature_labels, transitions, 
                         21746353, 21766048, 5000, strand, "validation")
    
    # Step 4: Analyze training window (gene-rich region from training data)
    # Use a window from later in the training data to find genes (avoiding start which might be centromeric)
    training_length = int((aligned_stop - aligned_start) * 0.01)
    training_window_size = 21766048 - 21746353  # Same size as validation window
    training_window_start = aligned_start + training_length // 2  # Middle of training data
    training_window_stop = training_window_start + training_window_size
    
    analyze_target_window(predictions, annotations, crf, feature_labels, transitions,
                         training_window_start, training_window_stop, 5000, strand, "training")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main() 