import logging
import numpy as np
import xarray as xr

from src.analysis import get_sequence_modeling_labels
from src.sequence import convert_to_biluo_labels
from src.schema import BILUO_TAG_CLASS_INFO, MODELING_FEATURE_CLASS_INFO, SENTINEL_MASK, SEQUENCE_MODELING_FEATURES, ModelingFeatureType as MFT

logger = logging.getLogger(__name__)


def get_tag_class_map(feature_labels: xr.DataArray) -> dict[int, str]:
    """Create mapping from tag indices to tag class names.
    
    Returns
    -------
    dict[int, str]
        Dictionary mapping tag indices to tag names
    """
    label_classes = feature_labels.feature.values.tolist()
    if (expected := SEQUENCE_MODELING_FEATURES) != (actual := label_classes):
        raise ValueError(f"Invalid label classes: {actual} != {expected}")
    
    return {e["index"]: e["name"] for e in BILUO_TAG_CLASS_INFO}

def get_feature_class_map(feature_labels: xr.DataArray) -> dict[int, str]:
    """Create mapping from feature indices to feature class names.
    
    Returns
    -------
    dict[int, str]
        Dictionary mapping feature indices to feature names
    """
    # Get label classes from feature labels
    label_classes = feature_labels.feature.values.tolist()
    if (expected := SEQUENCE_MODELING_FEATURES) != (actual := label_classes):
        raise ValueError(f"Invalid label classes: {actual} != {expected}")
    return {e["index"]: e["name"] for e in MODELING_FEATURE_CLASS_INFO}

def get_tag_stats(tag_labels: xr.DataArray, tag_class_map: dict[int, str]) -> list[dict[str, str | int | float]]:
    """Get tag frequency and count statistics.
    
    Parameters
    ----------
    tag_labels : xr.DataArray
        Tag labels array
    tag_class_map : dict[int, str]
        Mapping from tag indices to tag names
    
    Returns
    -------
    list[dict[str, str | int | float]]
        List of dictionaries with tag statistics, each containing:
        - 'tag': tag name (str)
        - 'count': number of occurrences (int)
        - 'frequency': percentage frequency (float)
    """
    # Get tag frequency with counts and percentages
    tag_values, tag_counts = np.unique(tag_labels.values, return_counts=True)
    mapped_tags = np.array([tag_class_map.get(tag, "Unknown") for tag in tag_values])
    tag_percentages = tag_counts / tag_counts.sum() * 100
    
    # Return as list of dictionaries
    return [
        {'tag': tag, 'count': int(count), 'frequency': float(freq)}
        for tag, count, freq in zip(mapped_tags, tag_counts, tag_percentages)
    ]


def extract_label_dataset(
    ds: xr.Dataset,
) -> xr.Dataset:
    """Extract BILUO labels from sequence dataset.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input sequence dataset
    
    Returns
    -------
    xr.Dataset
        Dataset containing:
        - feature_labels: The underlying feature labels used to generate BILUO tags
        - tag_labels_masked: The BILUO tags with invalid positions masked to -1
        - tag_labels: The BILUO tags without masking applied
        - label_masks: Boolean mask indicating valid label positions
    """
    # Extract non-overlapping feature indicator vectors
    feature_labels = get_sequence_modeling_labels(ds)
    feature_labels = feature_labels.transpose("strand", "sequence", "feature")
    if (actual := feature_labels.dims) != (expected := ('strand', 'sequence', 'feature')):
        raise ValueError(f"Invalid feature labels dimensions: {actual} != {expected}")

    # Flatten across features with argmax
    class_labels_unmasked = feature_labels.argmax(dim='feature')
    if (actual := class_labels_unmasked.dims) != (expected := ('strand', 'sequence')):
        raise ValueError(f"Invalid class labels dimensions: {actual} != {expected}")

    # Mask out invalid labels
    label_masks = (
        ds.label_masks.astype(bool)
        # label_masks=True ==> keep label, so all must be True across multiple masks
        .all(dim='reason')
    )
    
    # Create masked version
    class_labels_masked = xr.where(label_masks, class_labels_unmasked, SENTINEL_MASK)
    if (actual := class_labels_masked.dims) != (expected := ('strand', 'sequence')):
        raise ValueError(f"Invalid class labels dimensions: {actual} != {expected}")
    
    # Convert to BILUO labels for both masked and unmasked versions
    def convert_class_labels_to_biluo(class_labels_input):
        tags_list = []
        for strand_name in class_labels_input.strand.values:
            values = class_labels_input.sel(strand=strand_name, drop=True).values
            assert values.ndim == 1
            if strand_name == 'positive':
                tags = convert_to_biluo_labels(values)
            else:
                tags = np.flip(convert_to_biluo_labels(np.flip(values, axis=0)), axis=0)
            assert tags.ndim == 1
            tags_list.append(tags)
        
        return xr.DataArray(
            np.stack(tags_list), 
            dims=['strand', 'sequence'],
            coords={'strand': class_labels_input.strand.values, 'sequence': class_labels_input.sequence.values}
        )
    
    tag_labels_masked = convert_class_labels_to_biluo(class_labels_masked)
    tag_labels = convert_class_labels_to_biluo(class_labels_unmasked)
    assert tag_labels_masked.shape == class_labels_unmasked.shape
    assert tag_labels.shape == class_labels_unmasked.shape
    
    return xr.Dataset({
        'feature_labels': feature_labels,
        'tag_labels_masked': tag_labels_masked,
        'tag_labels': tag_labels,
        'label_masks': label_masks
    })


def select_windows(
    feature_labels: xr.DataArray, 
    seq_length: int,
    window_size: int,
    intergenic_proportion: float,
    seed: int,
    intergenic_threshold: float = 0.99
) -> tuple[list[tuple[int, int]], dict[str, int | float]]:
    """Select windows to achieve desired intergenic proportion.
    
    Parameters
    ----------
    feature_labels : xr.DataArray
        Feature labels with dimensions (strand, sequence, feature)
    seq_length : int
        Total sequence length
    window_size : int
        Size of training windows
    intergenic_proportion : float
        Desired proportion of intergenic windows relative to total windows.
        For example, 0.5 means equal numbers of intergenic and genic windows,
        0.9 means 9 intergenic windows for every 1 genic window.
    seed : int
        Random seed for window selection
    intergenic_threshold : float, optional
        Threshold for classifying windows as intergenic (default: 0.99).
        Windows with intergenic proportion > threshold are classified as intergenic.
    
    Returns
    -------
    tuple[list[tuple[int, int]], dict]
        Tuple containing:
        - List of (start, end) positions for selected windows
        - Stats dictionary with window selection information
    """
    step_size = window_size // 2
    max_steps = max(0, (seq_length // step_size) - 1)
    
    # Generate all possible windows and classify them
    intergenic_windows, genic_windows = [], []
    
    for step_num in range(max_steps):
        w_start = step_num * step_size
        w_end = w_start + window_size
        if w_end > seq_length:
            continue
            
        # Check if window is predominantly intergenic (across both strands)
        window_features = feature_labels.isel(sequence=slice(w_start, w_end))
        intergenic_prop = window_features.sel(feature=MFT.INTERGENIC).mean().item()
        
        window = (w_start, w_end)
        if intergenic_prop > intergenic_threshold:  # Highly intergenic
            intergenic_windows.append(window)
        else:
            genic_windows.append(window)
    
    # Calculate target numbers based on desired proportion;
    # Always use ALL genic windows
    genic_target = len(genic_windows)
    
    if intergenic_proportion >= 1.0:
        # Only intergenic windows
        intergenic_target = len(intergenic_windows)
        genic_target = 0
    elif intergenic_proportion <= 0.0:
        # Only genic windows
        intergenic_target = 0
    else:
        # Calculate needed intergenic windows: I/(I+G) = p => I = p*G/(1-p)
        intergenic_needed = int(genic_target * intergenic_proportion / (1 - intergenic_proportion))
        intergenic_target = min(intergenic_needed, len(intergenic_windows))
    
    # Sample windows
    rng = np.random.RandomState(seed)
    selected_intergenic = rng.choice(
        len(intergenic_windows), 
        size=min(intergenic_target, len(intergenic_windows)), 
        replace=False
    ) if intergenic_windows and intergenic_target > 0 else []
    
    selected_genic = rng.choice(
        len(genic_windows),
        size=min(genic_target, len(genic_windows)),
        replace=False
    ) if genic_windows and genic_target > 0 else []
    
    selected_windows = ([intergenic_windows[i] for i in selected_intergenic] + 
                       [genic_windows[i] for i in selected_genic])
    
    # Calculate actual proportion achieved
    total_selected = len(selected_intergenic) + len(selected_genic)
    actual_proportion = len(selected_intergenic) / total_selected if total_selected > 0 else 0.0
    
    stats = {
        'available_intergenic_windows': len(intergenic_windows),
        'available_genic_windows': len(genic_windows),
        'selected_intergenic_windows': len(selected_intergenic),
        'selected_genic_windows': len(selected_genic),
        'target_intergenic_proportion': intergenic_proportion,
        'actual_intergenic_proportion': actual_proportion
    }
    
    return selected_windows, stats 
