import numpy as np
from numba import njit

@njit
def _find_matches_within_tolerance(
    pred_starts: np.ndarray,
    pred_stops: np.ndarray,
    true_starts: np.ndarray,
    true_stops: np.ndarray,
    tolerance: int
) -> int:
    """
    Find number of true positive matches using binary search for efficient lookups.
    """
    # Sort both start and stop arrays with their original indices
    start_order = np.argsort(pred_starts)
    stop_order = np.argsort(pred_stops)
    sorted_starts = pred_starts[start_order]
    sorted_stops = pred_stops[stop_order]
    
    true_positives = 0
    for true_start, true_stop in zip(true_starts, true_stops):
        # Find matching indices for both start and stop positions
        start_matches = start_order[
            np.searchsorted(sorted_starts, true_start - tolerance, side='left'):
            np.searchsorted(sorted_starts, true_start + tolerance, side='right')
        ]
        stop_matches = stop_order[
            np.searchsorted(sorted_stops, true_stop - tolerance, side='left'):
            np.searchsorted(sorted_stops, true_stop + tolerance, side='right')
        ]
        
        # If any index appears in both matches, we have a true positive
        true_positives += bool(np.intersect1d(start_matches, stop_matches).size)
    
    return true_positives

def evaluate_intervals(
    pred_intervals: list[tuple[int, int]],
    true_intervals: list[tuple[int, int]],
    tolerance: int = 0
) -> dict[str, float]:
    """
    Calculate precision, recall and F1 score between predicted and true intervals.
    
    Args:
        pred_intervals: List of (start, stop) tuples for predicted intervals
        true_intervals: List of (start, stop) tuples for true intervals
        tolerance: Integer specifying how far apart interval bounds can be
                  to be considered equal. Default is 0 (exact match required).
    
    Returns:
        Dictionary containing precision, recall, and f1_score
    """
    if not pred_intervals or not true_intervals:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
    
    # Convert input lists to numpy arrays using transpose
    pred_starts, pred_stops = np.array(pred_intervals, dtype=np.int64).T
    true_starts, true_stops = np.array(true_intervals, dtype=np.int64).T
    
    # Find number of true positives
    true_positives = _find_matches_within_tolerance(
        pred_starts, pred_stops, true_starts, true_stops, tolerance
    )
    
    # Calculate metrics
    precision = true_positives / len(pred_intervals)
    recall = true_positives / len(true_intervals)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }
