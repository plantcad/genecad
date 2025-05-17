import pandas as pd
import numpy as np
import xarray as xr
from numba import njit
from src.modeling import GeneClassifierConfig
from src.sequence import convert_to_entity_labels

@njit
def _transition_counts(labels: np.ndarray, classes: int) -> np.ndarray:
    """
    Calculates the transition counts matrix for labels in sequences.

    Parameters
    ----------
    labels : np.ndarray
        A 2D array of shape (N, S) where N is the number of sequences
        and S is the sequence length. Masked labels should be -1.
    classes : int
        The total number of unique classes (including the mask value if relevant,
        though transitions involving -1 are ignored).

    Returns
    -------
    np.ndarray
        A square matrix of shape (classes, classes) where element (i, j)
        is the count of transitions from class i to class j.
    """
    counts = np.zeros((classes, classes), dtype=np.float64)
    N, S = labels.shape

    for i in range(N):
        for j in range(S - 1):
            label_from = labels[i, j]
            label_to = labels[i, j + 1]
            if label_from >= 0 and label_to >= 0:
                counts[label_from, label_to] += 1

    return counts

def _convert_biluo_labels_to_entity_labels(labels: np.ndarray, sequence_length: int) -> np.ndarray:
    assert labels.shape == (sequence_length,)
    return convert_to_entity_labels(labels)

def get_transition_probs(ds: xr.Dataset, classes: list[str]) -> pd.DataFrame:
    mask = ds.soft_mask & ds.label_mask
    labels = xr.where(mask, ds.labels, -1).to_numpy()
    assert labels.shape == (ds.sizes["sample"], ds.sizes["sequence"])
    entities = np.apply_along_axis(
        _convert_biluo_labels_to_entity_labels, 
        axis=1, arr=labels, sequence_length=ds.sizes["sequence"]
    )
    counts = _transition_counts(entities, len(classes))
    row_sums = counts.sum(axis=1, keepdims=True)
    # Avoid division by zero for states with no outgoing transitions
    probs = np.divide(counts, row_sums, where=row_sums != 0, out=np.zeros_like(counts))
    return pd.DataFrame(probs, index=classes, columns=classes)


def format_transition_probs(df: pd.DataFrame, precision: int = 16) -> str:
    rows, cols = df.index.tolist(), df.columns.tolist()
    format_spec = f"{{:.{precision}e}}"

    # Calculate column widths based on formatted numbers and column headers
    col_widths = []
    for j, col_label in enumerate(cols):
        max_len = len(col_label)  # Start with header length
        for i in range(len(rows)):
            formatted_num = format_spec.format(df.iloc[i, j])
            max_len = max(max_len, len(formatted_num))
        col_widths.append(max_len)

    lines = ["np.array(["]

    # Add column header comment with padding
    padded_cols = [col.ljust(width) for col, width in zip(cols, col_widths)]
    header_comment = "# " + "  ".join(padded_cols)
    lines.append(f"    {header_comment}")

    # Add data rows with padding
    for i, row_label in enumerate(rows):
        row_values = []
        for j in range(len(cols)):
            formatted_num = format_spec.format(df.iloc[i, j])
            padded_num = formatted_num.ljust(col_widths[j])
            row_values.append(padded_num)
        
        row_str = ", ".join(row_values)
        # Add trailing comma for the row, followed by the row label comment
        lines.append(f"    [{row_str}],  # {row_label}")

    lines.append("], dtype=np.float64)") # Specify dtype for clarity
    return "\n".join(lines)


path = "/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline/prep/sequence_dataset/train.zarr"
ds = xr.open_zarr(path, consolidated=True)
config = GeneClassifierConfig()
class_names = config.token_entity_names_with_background()
probs = get_transition_probs(ds, class_names)
print(format_transition_probs(probs, precision=16))
