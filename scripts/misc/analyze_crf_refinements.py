import xarray as xr
import numpy as np
import torch
from torchcrf import CRF
from src.sequence import viterbi_decode, transition_matrix_stationary_distribution
from src.prediction import merge_prediction_datasets
from src.modeling import token_transition_probs
from src.naming import normalize_genomic_region_label
from scipy.special import softmax
import matplotlib.pyplot as plt
import logging 

logger = logging.getLogger(__name__)


def get_viterbi_decoded_path(emissions: np.ndarray, transitions: np.ndarray) -> np.ndarray:
    return viterbi_decode(emissions, transitions, alpha=None)
    

def get_torch_crf_decoded_path(emissions: np.ndarray, transitions: np.ndarray) -> np.ndarray:
    T, N = emissions.shape
    log_emissions = torch.tensor(np.log(emissions + 1e-10), dtype=torch.float32)
    start_probs = transition_matrix_stationary_distribution(transitions)

    # Prepare PyTorch CRF
    crf = CRF(N, batch_first=True)
    
    # Set transition parameters manually
    # The CRF transition matrix has different indexing than our HMM transition matrix
    # In CRF, transitions[i, j] means transitioning from tag j to tag i
    with torch.no_grad():
        # Set transition weights from the transition probabilities (transposed)
        for i in range(N):
            for j in range(N):
                crf.transitions[i, j] = torch.log(torch.tensor(transitions[j, i] + 1e-10))
        
        # Set start transition weights
        for i in range(N):
            crf.start_transitions[i] = torch.log(torch.tensor(start_probs[i] + 1e-10))
        
        # Set end transition weights (uniform)
        crf.end_transitions.fill_(0.0)
    
    # Decode with PyTorch CRF
    with torch.no_grad():
        # Add batch dimension (CRF expects batch inputs)
        emissions_batch = log_emissions.unsqueeze(0)
        mask = torch.ones(1, T, dtype=torch.bool)
        
        # Get the decoded path
        torch_path = crf.decode(emissions_batch, mask)[0]
        return np.array(torch_path, dtype=np.int64)
    

def visualize_crf_refinements(emissions, labels, decode1, decode2, region_labels, mask, save_path):
    """Create visualization comparing CRF decoding methods with emission probabilities."""
    # Convert binary labels to class indices
    true_labels = np.argmax(labels.values, axis=1)
    
    # Define the three methods to compare
    methods = [
        ('Viterbi decode', decode1, 'red'),
        ('PyTorch CRF decode', decode2, 'red'),
        ('True labels', true_labels, 'red')
    ]
    
    # Create 4 subplots with the mask subplot being smaller
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True, 
                            gridspec_kw={'height_ratios': [0.8, 0.8, 0.8, 0.15]})
    x_pos = np.arange(len(emissions))
    
    for i, (method_name, decode_path, color) in enumerate(methods):
        ax = axes[i]
        
        # Plot emission probabilities heatmap without interpolation
        im = ax.imshow(emissions.values.T, aspect='auto', cmap='Blues', 
                       vmin=0, vmax=1)
        
        # Plot the specific decoding path
        ax.plot(x_pos, decode_path, color, linewidth=3, alpha=0.9, label=method_name)
        
        # Add dots where decoded labels don't match true labels (only for first two methods)
        if i < 2:  # Only for Viterbi and PyTorch CRF decode
            mismatches = decode_path != true_labels
            if np.any(mismatches):
                ax.scatter(x_pos[mismatches], decode_path[mismatches], 
                          c='orange', s=30, alpha=0.8, zorder=5, label='Mismatch')
        
        # Formatting
        ax.set_ylabel('Feature')
        ax.set_yticks(range(len(region_labels)))
        ax.set_yticklabels(region_labels)
        ax.legend(loc='upper right')
        ax.set_title(f'{method_name}')
        plt.colorbar(im, ax=ax, label='Emission Probability')
    
    # Add mask subplot (fourth subplot)
    mask_ax = axes[3]
    mask_im = mask_ax.imshow(mask.values.reshape(1, -1), aspect='auto', cmap='binary', 
                            vmin=0, vmax=1)
    mask_ax.set_ylabel('Mask')
    mask_ax.set_yticks([0])
    mask_ax.set_yticklabels(['Valid'])
    mask_ax.set_title('Label Mask (1=valid, 0=ignore)')
    plt.colorbar(mask_im, ax=mask_ax, label='Mask Value')
    
    # Only set xlabel on bottom plot
    axes[-1].set_xlabel('Position (bp)')
    
    plt.suptitle('CRF Decoding Comparison with Emission Probabilities', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    logging.basicConfig(level=logging.INFO)
    prediction_path = "/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline/predict/athaliana/runs/v0.5.0/chr1/predictions"
    predictions = merge_prediction_datasets(prediction_path)
    predictions = (
        predictions
        .pipe(lambda ds: ds.assign_coords(feature=[normalize_genomic_region_label(r) for r in ds.feature.values]))
    )
    logger.info(f"Loaded predictions:\n{predictions}")
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
    assert set(predictions.feature.values) - {"intergenic"} == set(annotations.region.values)

    transitions = token_transition_probs()
    assert set(predictions.feature.values) == set(transitions.index.values)
    logger.info(f"Loaded transitions:\n{transitions}")

    feature_labels = transitions.index.to_list()
    # feature_labels: ['intergenic', 'intron', 'five_prime_utr', 'cds', 'three_prime_utr']

    buffer = 512
    #start, stop, strand =  6886669, 6891404, 'positive' 
    # start, stop, strand = 5659370, 5665449, 'positive'
    # start, stop, strand = 5713794, 5722614, 'positive'
    start, stop, strand = 21746353, 21766048, 'positive'


    emissions = (
        predictions.feature_logits
        .sel(
            strand=strand,
            sequence=slice(start - buffer, stop + buffer),
            feature=feature_labels
        )
        .transpose("sequence", "feature")
        .pipe(lambda da: da.copy(data=softmax(da.values, axis=1)))
    )
    logger.info(f"Extracted emissions:\n{emissions}")
    assert np.allclose(emissions.sum(dim="feature"), 1)
    assert np.all(emissions > 0)

    labels = (
        annotations.region_labels
        .pipe(lambda da: (
            xr.concat([
                (1 - da.max(dim='region'))
                .expand_dims("region")
                .assign_coords(region=["intergenic"]), 
                da
            ], dim='region')
        ))
        .rename(region="feature")
        .sel(
            strand=strand,
            sequence=slice(start - buffer, stop + buffer),
            feature=[r for r in feature_labels]
        )
        .transpose("sequence", "feature")
    )
    logger.info(f"Extracted labels:\n{labels}")
    assert emissions.feature.values.tolist() == labels.feature.values.tolist()

    mask = (
        annotations.label_masks
        .sel(
            strand=strand,
            sequence=slice(start - buffer, stop + buffer)
        )
        .min(dim="reason") # 0 indicates position to ignore
    )

    decode1 = get_viterbi_decoded_path(emissions.values, transitions.values)
    decode2 = get_torch_crf_decoded_path(emissions.values, transitions.values)

    # Visualize
    path = 'local/scratch/crf_refinements_comparison.pdf'
    logger.info(f"Saving visualization to {path}")
    visualize_crf_refinements(emissions, labels, decode1, decode2, feature_labels, mask, path)


if __name__ == "__main__":
    main()
