import numpy as np
import pandas as pd
import numpy.typing as npt
from numba import njit
from typing import Callable, Iterator, Literal
import logging
import itertools

N_BILUO_TAGS = 4

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Labeling utilities
# -------------------------------------------------------------------------------------------------

def convert_entity_intervals_to_labels(
    intervals: pd.DataFrame, 
    domain: tuple[int, int], 
    num_labels: int, 
    on_overlap: Literal["raise", "warn", "ignore"] = "raise"
) -> npt.NDArray[np.int8]:
    """Convert feature intervals to a 2D label array.
    
    Parameters
    ----------
    intervals : pd.DataFrame
        DataFrame with columns 'start', 'stop', and 'label' defining the start and stop
        positions for the intervals as well as the 1-based label index for each interval
    domain : tuple[int, int]
        The start and end positions defining the domain to generate labels for;
        domain size D is defined as domain[1] - domain[0]
    num_labels : int
        The number of label classes C, which may be greater than the number of
        unique labels present in `intervals['label']`
    on_overlap : str
        What to do if there are overlapping intervals; one of "raise", "warn", or "ignore"

    Returns
    -------
    np.ndarray
        2D array of shape (D, C) where 1 indicates presence of a label
        and 0 indicates absence.
    """
    # Define domain params
    domain_start, domain_stop = domain
    domain_size = domain_stop - domain_start

    # Handle empty intervals
    if len(intervals) == 0:
        return np.zeros((domain_size, num_labels), dtype=np.int8)

    for col in ["start", "stop", "label"]:
        if col not in intervals.columns:
            raise ValueError(f"Column {col!r} must be present in intervals")
    
    # Ensure all values are integers
    for col in ['start', 'stop', 'label']:
        if not np.issubdtype(intervals[col].dtype, np.integer):
            raise ValueError(f"Column '{col}' must contain integers")
        
    # Check for duplicated rows
    if intervals.duplicated(['start', 'stop', 'label']).any():
        raise ValueError("Duplicate intervals found")
    
    # Check for duplicated bounds
    if on_overlap != "ignore":
        for bound in ["start", "stop"]:
            duplicate_starts = intervals[intervals.duplicated(bound, keep=False)]
            if not duplicate_starts.empty:
                raise ValueError(
                    f"Duplicate {bound} positions found; "
                    "labels should not be generated from overlapping intervals. "
                    f"Examples:\n{duplicate_starts.head()}"
                )
    
    # Verify labels are in valid range
    invalid_labels = intervals[(intervals['label'] < 1) | (intervals['label'] > num_labels)]
    if not invalid_labels.empty:
        raise ValueError(f"Labels must be in [1, {num_labels}]. Invalid examples:\n{invalid_labels.head()}")
    
    # Verify that all intervals are within the domain
    invalid_intervals = intervals[
        (intervals['start'] < domain[0]) | (intervals['stop'] > domain[1])
    ]
    if not invalid_intervals.empty:
        raise ValueError(f"Intervals must be within domain {domain}. Invalid examples:\n{invalid_intervals.head()}")
    
    # Prepare arrays for numba function
    starts = intervals['start'].to_numpy()
    stops = intervals['stop'].to_numpy()
    labels = intervals['label'].to_numpy() - 1  # Convert to 0-indexed
    
    # Populate labels array
    result = np.zeros((domain_size, num_labels), dtype=np.int32)
    result = _convert_entity_intervals_to_labels(starts, stops, labels, domain_start, result)

    # Check for any labels that exceed 1
    if on_overlap != "ignore" and (result.sum(axis=1) > 1).any():
        if on_overlap == "raise":
            raise ValueError(
                "Overlapping intervals detected in generated labels. If this is "
                "expected, use `on_overlap='ignore'` to suppress this error."
            )
        elif on_overlap == "warn":
            logger.warning(
                "Overlapping intervals detected in generated labels. If this is "
                "expected, use `on_overlap='ignore'` to suppress this warning."
            )

    # Flatten to binary representation
    result = (result > 0).astype(np.int8)
    return result


@njit
def _convert_entity_intervals_to_labels(starts, stops, labels, domain_start, result):
    for i in range(len(starts)):
        start_idx = starts[i] - domain_start
        stop_idx = stops[i] - domain_start
        label_idx = labels[i]
        result[start_idx:stop_idx, label_idx] += 1
    return result

def _validate_labels(labels: np.ndarray) -> np.ndarray:
    if not isinstance(labels, np.ndarray):
        raise ValueError(f"labels must be a numpy array, got {type(labels)=}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be a 1D array, got {labels.shape=}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels must be a numpy array of integers, got {labels.dtype=}")
    if np.any(labels < -1):
        raise ValueError(f"labels must be in the set {{-1, 0, 1, 2, 3, ...}}, got {set(labels)=}")
    return labels


# -------------------------------------------------------------------------------------------------
# BILUO utilities
# -------------------------------------------------------------------------------------------------

def convert_biluo_index_to_class_name(label: int, entity_names: list[str], sentinel_names: tuple[str, str] | None = None) -> str:
    sentinel_names = _default_sentinels(sentinel_names)
    entity_name = convert_biluo_index_to_entity_name(label, entity_names, sentinel_names)
    if label < 1:
        return entity_name
    else:
        tag = "BILU"[(label - 1) % N_BILUO_TAGS]
        return f"{tag}-{entity_name}"
    

def _default_sentinels(sentinel_names: tuple[str, str] | None = None) -> tuple[str, str]:
    if sentinel_names is not None:
        if len(sentinel_names) != 2:
            raise ValueError(f"sentinel_names must be a tuple of length 2, got {sentinel_names=}")
        return sentinel_names
    return ("mask", "background")

def convert_biluo_entity_names(entity_names: list[str]) -> list[str]:
    return [
        convert_biluo_index_to_class_name(i + 1, entity_names) 
        for i in range(len(entity_names) * N_BILUO_TAGS)
    ]

def convert_biluo_index_to_entity_name(label: int, entity_names: list[str], sentinel_names: tuple[str, str] | None = None) -> str:
    sentinel_names = _default_sentinels(sentinel_names)
    if label < -1:
        raise ValueError(f"Label must be in {{-1, 0, 1, 2, ...}}, got {label}")
    if label < 1:
        return sentinel_names[label + 1]
    else:
        entity_idx = (label-1) // N_BILUO_TAGS
        return entity_names[entity_idx]


def convert_to_biluo_labels(labels: np.ndarray) -> np.ndarray:
    """Convert class labels to BILUO format

    Parameters
    ----------
        labels: 1D array of integer class indices (-1, 0, 1, 2, 3, ...) where
            -1 is a mask class and 0 is a background class
        
    Returns
    -------
        biluo_indices (np.ndarray): 1D array of integers representing BILUO tags
            where -1 and 0 classes are retained while each other class is expanded
            into 4 separate classes as follows:
        
        For class C (where C ≥ 1):
        - B-tag (Begin): 1 + (C-1)*4
        - I-tag (Inside): 2 + (C-1)*4
        - L-tag (Last): 3 + (C-1)*4
        - U-tag (Unit): 4 + (C-1)*4
        
        For example:
        - Class 1 maps to values 1, 2, 3, 4 for B, I, L, U tags
        - Class 2 maps to values 5, 6, 7, 8 for B, I, L, U tags
        - Class 3 maps to values 9, 10, 11, 12 for B, I, L, U tags
    """
    _validate_labels(labels)
    return _convert_to_biluo(labels)

@njit
def _convert_to_biluo(labels):

    n = len(labels)
    biluo_indices = np.zeros(n, dtype=np.int32)

    begin_offset = 1
    inside_offset = 2
    end_offset = 3
    unit_offset = 4
    n_tags = N_BILUO_TAGS
    
    i = 0
    while i < n:
        # Handle masked positions (-1)
        if labels[i] == -1:
            biluo_indices[i] = -1
            i += 1
            continue
            
        # Handle outside/background (0)
        if labels[i] == 0:
            biluo_indices[i] = 0
            i += 1
            continue
            
        # Determine current non-background entity class
        current_class = labels[i]
        start_idx = i
        
        # Find the end of this entity
        while i < n and labels[i] == current_class:
            i += 1
        offset_class = current_class - 1
        end_idx = i - 1
        
        # Single token entity (Unit)
        if start_idx == end_idx:
            # Unit span (U-<entity>)
            biluo_indices[start_idx] = unit_offset + offset_class * n_tags
        # Multi-token entity (Begin, Inside, End)
        else:
            # Begin span (B-<entity>)
            biluo_indices[start_idx] = begin_offset + offset_class * n_tags
            
            # Inside span (I-<entity>)
            for j in range(start_idx + 1, end_idx):
                biluo_indices[j] = inside_offset + offset_class * n_tags
                
            # End span (L-<entity>)
            biluo_indices[end_idx] = end_offset + offset_class * n_tags
    
    return biluo_indices

def convert_to_entity_labels(labels: np.ndarray) -> np.ndarray:
    """Convert BILUO class labels to entity labels

    This is the inverse of `convert_to_biluo_labels`.
    
    Parameters
    ----------
        labels: 1D array of integer class indices from `convert_to_biluo_labels`

    Returns
    -------
        entity_labels: 1D array of integers representing entity labels
    """
    _validate_labels(labels)
    entity_labels = np.where(
        np.isin(labels, [-1, 0]),
        labels,
        ((labels - 1) // N_BILUO_TAGS) + 1
    )
    return entity_labels

def convert_entity_labels_to_intervals(
    labels: np.ndarray, 
    class_groups: list[list[int]], 
    mask: np.ndarray | None = None
) -> pd.DataFrame:
    """Create entity intervals from entity labels

    Parameters
    ----------
        labels: 1D array of entity class indices, e.g. as from `convert_to_entity_labels`
        class_groups: A list of length C with each element being a list of input class
            indices to collapse into a single output class. That output class is then
            used to define resulting intervals. All class indices must be positive
            integers, and must include mappings for all possible classes in `labels`
            beyond the background class (0).
        mask: A 1D array of booleans where True indicates a valid position
            and False indicates an invalid position; if an interval contains
            any invalid positions, it is ignored.

    Returns
    -------
        intervals: A dataframe with 3 columns:
            - entity: The entity index
            - start: The start index of the interval
            - stop: The stop index of the interval
    """
    intervals = find_group_intervals(labels, class_groups=class_groups, mask=mask)
    result = []
    for entity in range(len(intervals)):
        starts, stops = intervals[entity]
        entity_intervals = list(zip(starts.tolist(), stops.tolist()))
        for start, stop in entity_intervals:
            result.append({
                'entity': int(entity + 1),  # Entity indices are 1-based
                'start': int(start),
                'stop': int(stop),
            })
    result = pd.DataFrame(result, columns=["entity", "start", "stop"])
    assert not result.duplicated().any()
    return result



# -------------------------------------------------------------------------------------------------
# Inverval utilities
# -------------------------------------------------------------------------------------------------


def find_intervals(labels: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Find the start and end indices of intervals in a sequence label array
    
    Parameters
    ----------
        labels: A 1D array of boolean sequence labels in {0, 1} defining
            where intervals exist with shape (N,)
        mask: A 1D array of booleans where True indicates a valid position
            and False indicates an invalid position; if an interval contains
            any invalid positions, it is ignored

    Returns
    -------
        intervals: An array of shape (2, M) containing the start and stop
            indices of the contiguous intervals, i.e. of sequences of 1s in the labels,
            where M is the number of intervals; start and stop indices are 0-based
            and inclusive, i.e. the interval [start, stop] includes both the start
            and stop positions.

    """
    if labels.ndim != 1:
        raise ValueError("Labels must be a 1D array")
    if len((invalid_labels := np.setdiff1d(labels, [0, 1]))) > 0:
        raise ValueError(f"Labels must be 0 or 1, not: {invalid_labels}")

    if mask is not None:
        if mask.shape != labels.shape:
            raise ValueError(f"Mask must be the same shape as labels; got {mask.shape=} and {labels.shape=}")
        if not np.issubdtype(mask.dtype, np.bool_):
            raise ValueError(f"Mask must be a boolean array; got {mask.dtype=}")

    diffs = np.diff(np.pad(labels, (1, 1)))
    starts = np.where(diffs == 1)[0]
    stops = np.where(diffs == -1)[0] - 1
    assert starts.ndim == stops.ndim == 1
    assert len(starts) == len(stops)
    
    if mask is not None and len(starts) > 0:
        valid = np.array([np.all(mask[s : e + 1]) for s, e in zip(starts, stops)])
        starts, stops = starts[valid], stops[valid]

    return np.array([starts, stops])



def find_group_intervals(labels: np.ndarray, class_groups: list[list[int]], mask: np.ndarray | None = None) -> list[np.ndarray]:
    """Find the start and end indices of intervals defined by combinations of classes in a 1D label array
    
    Parameters
    ----------
        labels: A 1D array of integer sequence labels in {0, 1, 2, 3, ...}
            with shape (N,) and values 0=background, 1=class 1, 2=class 2, etc.
            up to C classes (not including background).
        class_groups: A list of length C with each element being a list of input class
            indices to collapse into a single output class. That output class is then
            used to define resulting intervals. All class indices must be positive
            integers, and must include mappings for all possible classes in `labels`
            beyond the background class (0).
        mask: A 1D array of booleans where True indicates a valid position
            and False indicates an invalid position; if an interval contains
            any invalid positions, it is ignored.

    Returns
    -------
        intervals: A list (of length C) of arrays each with shape (2, M_i) containing the start and
            stop indices of the contiguous intervals for a given output class C_i, and number of
            intervals M_i.

    Examples
    --------
    >>> labels = np.array([0, 1, 1, 1, 0, 2, 2, 3, 0])
    >>> class_groups = [[1, 2], [3]]
    >>> find_group_intervals(labels, class_groups)
    [
        array([[1, 5], [3, 6]]), # First class has intervals [1, 3] and [5, 6]
        array([[7], [7]]),       # Second class has interval [7, 7]
    ]
    """
    # Validate class_groups
    input_classes = np.unique([c for group in class_groups for c in group])
    if len(input_classes) == 0:
        return []
    if not np.issubdtype(input_classes.dtype, np.integer):
        raise ValueError(f"class_groups must contain only integer values, got {input_classes=}")
    if (input_classes <= 0).any():
        raise ValueError(f"class_groups must contain only positive integers, got {input_classes=}")

    # Validate labels
    if labels.ndim != 1:
        raise ValueError(f"labels must be a 1D array, got {labels.shape=}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels must be a numpy array of integers, got {labels.dtype=}")
    if not np.all(np.isin(labels, np.r_[0, input_classes])):
        max_class = int(input_classes.max())
        raise ValueError(f"labels must be in the set {{0, 1, 2, ..., {max_class}}}, got {set(labels)=}")
    
    result = []
    # For each group, create a binary mask and find intervals
    for group in class_groups:
        # Create binary array where 1 indicates positions with any class from this group
        group_mask = np.isin(labels, group).astype(np.int8)
        # Find intervals of contiguous 1s
        group_intervals = find_intervals(group_mask, mask)
        result.append(group_intervals)
    
    return result


def find_overlapping_intervals(starts: pd.Series, stops: pd.Series) -> pd.Series:
    """Find intervals that overlap with any other interval."""
    if len(starts) <= 1:
        return pd.Series(False, index=starts.index)
    if not starts.is_monotonic_increasing:
        raise ValueError("Interval starts must be sorted")
    has_overlap = _find_overlapping_intervals(starts.values, stops.values)
    return pd.Series(has_overlap, index=starts.index)

@njit
def _find_overlapping_intervals(starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Fast interval overlap detection with numba.
    
    Takes sorted arrays of start and stop positions and returns a boolean array 
    indicating which intervals overlap with any other interval.
    """
    n = len(starts)
    has_overlap = np.zeros(n, dtype=np.bool_)
    
    # Edge case: 0 or 1 interval can't overlap
    if n <= 1:
        return has_overlap
    
    # Track the rightmost endpoint seen so far
    for i in range(n - 1):
        current_stop = stops[i]
        
        # If the next interval starts before the current one ends, we have an overlap
        for j in range(i + 1, n):
            if starts[j] >= current_stop:
                # Once we reach an interval that starts after the current one ends,
                # we can stop checking further intervals (they're sorted by start)
                break
            # Mark both the current interval and the overlapping one
            has_overlap[i] = True
            has_overlap[j] = True
    
    return has_overlap


# -------------------------------------------------------------------------------------------------
# Evaluation utilities
# -------------------------------------------------------------------------------------------------

def create_entity_evaluation_intervals(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_groups: list[list[int]], 
    mask: np.ndarray | None = None
) -> pd.DataFrame:
    """Compare predicted sequence labels with true intervals
    
    Parameters
    ----------
        true_labels: 1D array of entity class indices, e.g. as from `convert_to_entity_labels` 
        pred_labels: 1D array of entity class indices, e.g. as from `convert_to_entity_labels` 
        class_groups: A list of length C with each element being a list of input class
            indices to collapse into a single output class. That output class is then
            used to define resulting intervals. All class indices must be positive
            integers, and must include mappings for all possible classes in `labels`
            beyond the background class (0).
        mask: A 1D array of booleans where True indicates a valid position
            and False indicates an invalid position
            
    Returns
    -------
        intervals: DataFrame containing both labeled (true) and predicted intervals with columns:
            - entity: The entity index
            - start: The start index of the interval
            - stop: The stop index of the interval
            - predicted: True if this interval appeared in the predicted intervals, False otherwise
            - labeled: True if this interval is a ground truth (labeled) interval, False otherwise
    """
    true_intervals = convert_entity_labels_to_intervals(true_labels, class_groups, mask)
    pred_intervals = convert_entity_labels_to_intervals(pred_labels, class_groups, mask)
    key = ["entity", "start", "stop"]
    intervals = (
        pd.merge(
            true_intervals[key].astype(int).assign(labeled=True),
            pred_intervals[key].astype(int).assign(predicted=True),
            on=key,
            how="outer"
        )
        .assign(labeled=lambda df: df["labeled"].notna())
        .assign(predicted=lambda df: df["predicted"].notna())
    )
    assert intervals.notnull().all().all()
    assert not intervals[key].duplicated().any()
    return intervals

def get_evaluation_interval_metrics(intervals: pd.DataFrame) -> pd.Series:
    df = intervals
    assert df[["labeled", "predicted"]].notnull().all().all()
    true_intervals = df['labeled'].sum()
    pred_intervals = df['predicted'].sum()
    true_positives = (df['labeled'] & df['predicted']).sum()
    false_positives = (df['predicted'] & ~df['labeled']).sum()
    
    precision = true_positives / pred_intervals if pred_intervals > 0 else 0
    recall = true_positives / true_intervals if true_intervals > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return pd.Series({
        'intervals': len(intervals),
        'true_intervals': true_intervals,
        'predicted_intervals': pred_intervals,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })


# -------------------------------------------------------------------------------------------------
# Prediction utilities
# -------------------------------------------------------------------------------------------------

def create_sequence_windows(
    sequence: npt.ArrayLike,
    window_size: int,
    stride: int,
    pad_value: int = 0
) -> Iterator[tuple[npt.ArrayLike, tuple[int, int], tuple[int, int]]]:
    """Create windows for strided prediction across a sequence.
    
    Parameters
    ----------
        sequence: The sequence to create windows for
        window_size: The size of the windows to create
        stride: The stride of the windows
        pad_value: The value to pad the sequence with

    Returns
    -------
        windows: An iterator of tuples containing:
            - chunk: The sequence chunk
            - local_window: The local window boundaries describing what part of `chunk` is valid; e.g.
                all `chunk[slice(*local_window)]` subarrays are collectively exhaustive and mutually exclusive
                across the original sequence
            - global_window: The global window boundaries describing what part of `chunk` is valid
                within the context of the entire sequence; e.g. `sequence[slice(*global_window)]` is
                equivalent to `chunk[slice(*local_window)]`.
    """
    bounds = (0, len(sequence))
    pad = (window_size - len(sequence) % window_size) % window_size
    padded_sequence = np.pad(sequence, (0, pad), mode='constant', constant_values=pad_value)
    windows = create_prediction_windows(len(padded_sequence), window_size, stride)
    global_bounds, local_bounds = windows.T[:2], windows.T[2:]
    local_bounds = np.clip(local_bounds, *bounds)
    for start, stop, v_start, v_stop in zip(*global_bounds, *local_bounds):
        assert stop - start == window_size, f"Window size mismatch: {stop - start} != {window_size}"
        chunk = padded_sequence[start:stop]
        assert len(chunk) == window_size, f"Chunk size mismatch: {len(chunk)} != {window_size}"
        if v_start == v_stop:
            continue
        global_window = (v_start, v_stop)
        local_window = (v_start-start, v_stop-start) # define local window relative to global window
        assert local_window[0] >= 0 and local_window[1] <= window_size
        yield chunk, local_window, global_window

def create_prediction_windows(
    sequence_length: int,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """Create windows for strided prediction across a sequence.
    
    Parameters
    ----------
    sequence_length : int
        Total length of the sequence to process, must be even
    window_size : int
        Size of each window, must be even
    stride : int
        The step size between consecutive windows, must be in range [1, window_size].
        `sequence_length` must be divisible by `stride`.
        
    Returns
    -------
    np.ndarray
        Array of shape (n, 4) with columns: [starts, stops, v_starts, v_stops]
        where:
        - starts, stops: full window boundaries
        - v_starts, v_stops: valid region boundaries within each window
    """
    if sequence_length % 2 != 0:
        raise ValueError(f"sequence_length must be even, got {sequence_length=}")
    
    if window_size % 2 != 0:
        raise ValueError(f"window_size must be even, got {window_size=}")
    
    if not 1 <= stride <= window_size:
        raise ValueError(f"stride must be in range [1, {window_size}], got {stride=}")
    
    if sequence_length % stride != 0:
        raise ValueError(f"sequence_length must be divisible by stride, got {sequence_length} and {stride} (quotient={sequence_length / stride})")
    
    # Handle edge case with only one valid region
    if window_size >= sequence_length:
        return np.array([[0, sequence_length, 0, sequence_length]])
    
    # Calculate margin based on stride and window_size
    # Margin is the amount dropped from each side in overlapping windows to define the valid region
    # Examples:
    # - stride = window_size: margin = 0 (no overlap, valid region is full window)
    # - stride = window_size / 2: margin = window_size / 4 (25% dropped from each side)
    # - stride = window_size / 4: margin = window_size * 3/8 (37.5% dropped from each side)
    margin = int((window_size - stride) / 2)
    
    # Generate starts and stops for windows
    starts = np.arange(0, sequence_length - window_size + 1, stride)
    stops = starts + window_size
    
    # Create all valid region starts/stops
    v_starts = np.maximum(starts, starts + margin * (starts > 0))
    v_stops = np.minimum(stops, stops - margin * (stops < sequence_length))
    
    # Validate coverage: check that each valid region connects with the next
    if len(v_starts) > 1:
        gaps = v_starts[1:] - v_stops[:-1]
        if np.any(gaps != 0):
            raise AssertionError(f"Valid regions have gaps or overlaps at positions: {np.where(gaps != 0)[0]}")
    
    # Validate full coverage
    if len(v_starts) > 1 and (v_starts[0] != 0 or v_stops[-1] != sequence_length):
        raise AssertionError(f"Valid regions don't span full sequence: starts at {v_starts[0]}, ends at {v_stops[-1]}")
    
    # Stack into a (n, 4) array
    return np.column_stack((starts, stops, v_starts, v_stops))


# -------------------------------------------------------------------------------------------------
# Decoding utilities
# -------------------------------------------------------------------------------------------------

def _validate_decode_inputs(
    emission_probs: np.ndarray,
    transition_matrix: np.ndarray,
    initial_probs: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate inputs for decoding algorithms and compute initial probabilities if needed.
    
    Parameters
    ----------
    emission_probs : np.ndarray
        Matrix of shape (T, N) where T is sequence length and N is number of states.
    transition_matrix : np.ndarray
        Matrix of shape (N, N) where element (i, j) represents P(state j | state i).
    initial_probs : np.ndarray | None
        Vector of length N representing initial state probabilities.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Validated and potentially modified (emission_probs, transition_matrix, initial_probs)
    """
    # Ensure arrays are float type
    emission_probs = np.asarray(emission_probs, dtype=float)
    transition_matrix = np.asarray(transition_matrix, dtype=float)
    
    _, N = emission_probs.shape
    
    # Validate inputs
    if transition_matrix.shape != (N, N):
        raise ValueError(f"Transition matrix shape {transition_matrix.shape} doesn't match emission probabilities states {N}")
    
    if not np.allclose(transition_matrix.sum(axis=1), 1.0):
        raise ValueError("Transition matrix rows must sum to 1")
    
    if initial_probs is None:
        # Use stationary distribution of the transition matrix
        # (eigenvector with eigenvalue 1)
        eigenvals, eigenvecs = np.linalg.eig(transition_matrix.T)
        # Find closest eigenvalue to 1
        idx = np.argmin(np.abs(eigenvals - 1.0))
        # Extract the corresponding eigenvector and normalize
        initial_probs = np.real(eigenvecs[:, idx])
        initial_probs = initial_probs / initial_probs.sum()
    else:
        initial_probs = np.asarray(initial_probs, dtype=float)
        if initial_probs.shape != (N,):
            raise ValueError(f"Initial probabilities shape {initial_probs.shape} doesn't match number of states {N}")
        elif not np.isclose(initial_probs.sum(), 1.0):
            raise ValueError("Initial probabilities must sum to 1")
            
    return emission_probs, transition_matrix, initial_probs

@njit
def _viterbi_decode(
    log_emission: np.ndarray,
    log_transition: np.ndarray,
    log_initial: np.ndarray
) -> np.ndarray:
    """Numba-accelerated implementation of the Viterbi algorithm.
    
    Parameters
    ----------
    log_emission : np.ndarray
        Log emission probabilities of shape (T, N)
    log_transition : np.ndarray
        Log transition probabilities of shape (N, N)
    log_initial : np.ndarray
        Log initial state probabilities of shape (N,)
        
    Returns
    -------
    np.ndarray
        Vector of length T with most likely state indices at each position
    """
    T, N = log_emission.shape
    
    # Initialize Viterbi tables
    viterbi_prob = np.zeros((T, N))
    backpointer = np.zeros((T, N), dtype=np.int64)
    
    # Base case
    viterbi_prob[0] = log_initial + log_emission[0]
    
    # Recursive case
    for t in range(1, T):
        for j in range(N):
            # Calculate probabilities for each possible previous state
            probs = viterbi_prob[t-1] + log_transition[:, j]
            # Find most likely previous state
            backpointer[t, j] = np.argmax(probs)
            # Store probability of most likely path to this state
            viterbi_prob[t, j] = probs[backpointer[t, j]] + log_emission[t, j]
    
    # Backtrack to find optimal path
    path = np.zeros(T, dtype=np.int64)
    path[T-1] = np.argmax(viterbi_prob[T-1])
    
    for t in range(T-2, -1, -1):
        path[t] = backpointer[t+1, path[t+1]]
    
    return path

def viterbi_decode(
    emission_probs: npt.ArrayLike, 
    transition_matrix: npt.ArrayLike,
    initial_probs: npt.ArrayLike | None = None
) -> npt.NDArray[np.int64]:
    """Find most likely sequence of states using the Viterbi algorithm.
    
    Parameters
    ----------
    emission_probs : npt.ArrayLike
        Matrix of shape (T, N) where T is sequence length and N is number of states.
        Each element (t, n) represents P(observation at t | state n).
    transition_matrix : npt.ArrayLike
        Matrix of shape (N, N) where element (i, j) represents P(state j | state i).
        Rows must sum to 1.
    initial_probs : npt.ArrayLike, optional
        Vector of length N representing initial state probabilities.
        If None, the stationary distribution of the transition matrix is used.
    
    Returns
    -------
    npt.NDArray[np.int64]
        Vector of length T with most likely state indices at each position
    """
    # Validate and prepare inputs
    emission_probs, transition_matrix, initial_probs = _validate_decode_inputs(
        emission_probs, transition_matrix, initial_probs
    )
    
    # Work in log space to avoid numerical underflow
    log_emission = np.log(emission_probs + np.finfo(float).eps)
    log_transition = np.log(transition_matrix + np.finfo(float).eps)
    log_initial = np.log(initial_probs + np.finfo(float).eps)
    
    # Run Viterbi algorithm including backtracking
    return _viterbi_decode(log_emission, log_transition, log_initial)

def brute_force_decode(
    emission_probs: npt.ArrayLike, 
    transition_matrix: npt.ArrayLike,
    initial_probs: npt.ArrayLike | None = None
) -> npt.NDArray[np.int64]:
    """Find most likely sequence of states using brute force computation.
    
    This function examines all possible state sequences to find the most likely path.
    Only suitable for small examples and unit testing due to exponential complexity.
    
    Parameters
    ----------
    emission_probs : npt.ArrayLike
        Matrix of shape (T, N) where T is sequence length and N is number of states.
        Each element (t, n) represents P(observation at t | state n).
    transition_matrix : npt.ArrayLike
        Matrix of shape (N, N) where element (i, j) represents P(state j | state i).
        Rows must sum to 1.
    initial_probs : npt.ArrayLike, optional
        Vector of length N representing initial state probabilities.
        If None, the stationary distribution of the transition matrix is used.
    
    Returns
    -------
    npt.NDArray[np.int64]
        Vector of length T with most likely state indices at each position
    """
    # Validate and prepare inputs
    emission_probs, transition_matrix, initial_probs = _validate_decode_inputs(
        emission_probs, transition_matrix, initial_probs
    )
    
    T, N = emission_probs.shape
    
    # Calculate probability of each possible path with brute force
    paths = list(itertools.product(range(N), repeat=T))
    
    path_probs = []
    for path in paths:
        path_array = np.array(path)
        
        # Calculate probability of this path
        prob = initial_probs[path_array[0]] * emission_probs[0, path_array[0]]
        
        for t in range(1, T):
            prob *= transition_matrix[path_array[t-1], path_array[t]] * emission_probs[t, path_array[t]]
        
        path_probs.append((path_array, prob))
    
    # Find the path with highest probability
    best_path, _ = max(path_probs, key=lambda x: x[1])
    
    return np.array(best_path, dtype=np.int64)
