import numpy as np
import numpy.typing as npt
from numba import njit
import multiprocessing as mp
from src.sequence import find_intervals
import logging
from src.sequence import transition_matrix_stationary_distribution

logger = logging.getLogger(__name__)


def _safe_log_probabilities(probs: np.ndarray, epsilon: float) -> np.ndarray:
    """Convert probabilities to log space with consistent handling of zeros.
    
    Parameters
    ----------
    probs : np.ndarray
        Probability array
    epsilon : float
        Epsilon value for numerical stability. If 0, zero probabilities become -inf.
        If > 0, zero probabilities become log(epsilon).
        
    Returns
    -------
    np.ndarray
        Log probabilities
    """
    probs = np.asarray(probs, dtype=float)
    
    if epsilon == 0:
        # Set zero probabilities to -inf (forbidden)
        log_probs = np.full_like(probs, -np.inf)
        valid_mask = probs > 0
        if np.any(valid_mask):
            log_probs[valid_mask] = np.log(probs[valid_mask])
        return log_probs
    else:
        # Use epsilon for numerical stability
        return np.log(probs + epsilon)


def _validate_semi_markov_inputs(
    emission_probs: np.ndarray,
    transition_matrix: np.ndarray,
    duration_probs: dict[int, npt.ArrayLike],
    initial_probs: np.ndarray | None,
    epsilon: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Validate inputs for semi-Markov decoding algorithms."""
    # Validate epsilon
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")
    
    # Ensure arrays are float type
    emission_probs = np.asarray(emission_probs, dtype=float)
    transition_matrix = np.asarray(transition_matrix, dtype=float)
    
    T, N = emission_probs.shape
    
    # Validate transition matrix
    if transition_matrix.shape != (N, N):
        raise ValueError(f"Transition matrix shape {transition_matrix.shape} doesn't match emission probabilities states {N}")
    if np.any(transition_matrix < 0) or np.any(transition_matrix > 1):
        raise ValueError("Transition matrix values must be between 0 and 1")
    if not np.allclose(transition_matrix.sum(axis=1), 1.0):
        raise ValueError("Transition matrix rows must sum to 1")
    
    # Validate duration probabilities
    if not isinstance(duration_probs, dict):
        raise ValueError("duration_probs must be a dictionary")
    if set(duration_probs.keys()) != set(range(N)):
        raise ValueError(f"duration_probs must have keys for all states 0 to {N-1}")
    
    max_duration = max(len(duration_probs[i]) for i in range(N))
    logger.info(f"Using inferred max duration in Semi-Markov decoding of {max_duration}")
    
    # Convert duration probs to log space and pad to uniform length
    log_duration = np.full((N, max_duration), -np.inf)
    for state, probs in duration_probs.items():
        probs_array = np.asarray(probs, dtype=float)
        if np.any(probs_array < 0) or np.any(probs_array > 1):
            raise ValueError(f"Duration probabilities for state {state} must be between 0 and 1")
        if not np.isclose(probs_array.sum(), 1.0):
            raise ValueError(f"Duration probabilities for state {state} must sum to 1")
        
        # Use shared function for consistent log probability handling
        log_probs = _safe_log_probabilities(probs_array, epsilon)
        log_duration[state, :len(probs_array)] = log_probs
    
    # Validate initial probabilities
    if initial_probs is None:
        initial_probs = transition_matrix_stationary_distribution(transition_matrix)
    else:
        initial_probs = np.asarray(initial_probs, dtype=float)
        if initial_probs.shape != (N,):
            raise ValueError(f"Initial probabilities shape {initial_probs.shape} doesn't match number of states {N}")
        if np.any(initial_probs < 0) or np.any(initial_probs > 1):
            raise ValueError("Initial probabilities must be between 0 and 1")
        if not np.isclose(initial_probs.sum(), 1.0):
            raise ValueError("Initial probabilities must sum to 1")
            
    return emission_probs, transition_matrix, initial_probs, log_duration, max_duration


def semi_markov_viterbi_decode(
    emission_probs: npt.ArrayLike,
    transition_matrix: npt.ArrayLike,
    duration_probs: dict[int, npt.ArrayLike],
    initial_probs: npt.ArrayLike | None = None,
    epsilon: float = 0.0
) -> npt.NDArray[np.int64]:
    """Find most likely sequence using exact Semi-Markov Viterbi with explicit duration modeling.
    
    This implementation uses cumulative emission precomputation for O(1) emission score lookups,
    making exhaustive duration search efficient.
    
    Parameters
    ----------
    emission_probs : npt.ArrayLike
        Matrix of shape (T, N) where T is sequence length and N is number of states
    transition_matrix : npt.ArrayLike
        Matrix of shape (N, N) where element (i, j) represents P(state j | state i)
    duration_probs : dict[int, npt.ArrayLike]
        Dictionary mapping state index to duration probability array where
        duration_probs[state][d] = P(duration = d+1 | state)
    initial_probs : npt.ArrayLike, optional
        Vector of length N representing initial state probabilities
    epsilon : float, optional
        Epsilon added to probabilities; default is 0
        
    Returns
    -------
    npt.NDArray[np.int64]
        Vector of length T with most likely state indices at each position
    """
    # Validate and prepare inputs
    emission_probs, transition_matrix, initial_probs, log_duration, max_duration = _validate_semi_markov_inputs(
        emission_probs, transition_matrix, duration_probs, initial_probs, epsilon
    )
    
    T, N = emission_probs.shape
    logger.info(f"Starting Semi-Markov Viterbi decode: T={T:,} positions, N={N} states, max_duration={max_duration}")
    
    # Work in log space using shared function for consistency
    log_emission = _safe_log_probabilities(emission_probs, epsilon)
    log_transition = _safe_log_probabilities(transition_matrix, epsilon)
    log_initial = _safe_log_probabilities(initial_probs, epsilon)
    
    result = _semi_markov_viterbi_decode(
        log_emission, log_transition, log_initial, log_duration, max_duration
    )
    
    logger.info("Semi-Markov Viterbi decode completed")
    return result


@njit
def _semi_markov_viterbi_decode(
    log_emission: np.ndarray,
    log_transition: np.ndarray,
    log_initial: np.ndarray,
    log_duration: np.ndarray,
    max_duration: int
) -> np.ndarray:
    """Numba-accelerated exact Semi-Markov Viterbi with cumulative emission optimization."""
    T, N = log_emission.shape
    
    # Precompute cumulative emission scores for efficiency
    cumulative_emission = np.zeros((T, N))
    for j in range(N):
        cumulative_emission[0, j] = log_emission[0, j]
        for t in range(1, T):
            cumulative_emission[t, j] = cumulative_emission[t-1, j] + log_emission[t, j]
    
    # Viterbi table: delta[t][j] = max log prob of ending in state j at time t
    delta = np.full((T, N), -np.inf)
    # Backpointer: (previous_state, segment_start_time)
    backpointer = np.zeros((T, N, 2), dtype=np.int64)
    
    # Initialize first position
    for j in range(N):
        if log_duration.shape[1] > 0:  # Duration 1
            delta[0, j] = log_initial[j] + log_duration[j, 0] + log_emission[0, j]
            backpointer[0, j, 0] = -1  # No previous state
            backpointer[0, j, 1] = 0   # Segment starts at 0
    
    # Forward pass with exhaustive duration search
    for t in range(1, T):
        for j in range(N):
            best_score = -np.inf
            best_prev_state = -1
            best_start_time = -1
            
            # Exhaustive search over all possible durations
            max_dur = min(max_duration, t + 1, log_duration.shape[1])
            
            # Search all durations from 1 to max_dur
            for d in range(1, max_dur + 1):
                start_time = t - d + 1
                if start_time < 0:
                    continue
                
                # Compute emission score for this segment using cumulative sums
                if start_time == 0:
                    emission_score = cumulative_emission[t, j]
                else:
                    emission_score = cumulative_emission[t, j] - cumulative_emission[start_time - 1, j]
                
                # Duration score
                if d <= log_duration.shape[1]:
                    duration_score = log_duration[j, d - 1]
                else:
                    duration_score = -np.inf
                
                if start_time == 0:
                    # First segment
                    score = log_initial[j] + duration_score + emission_score
                    if score > best_score:
                        best_score = score
                        best_prev_state = -1
                        best_start_time = start_time
                else:
                    # Not first segment - find best previous state
                    for i in range(N):
                        if delta[start_time - 1, i] == -np.inf:
                            continue
                        
                        transition_score = log_transition[i, j]
                        total_score = delta[start_time - 1, i] + transition_score + duration_score + emission_score
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_prev_state = i
                            best_start_time = start_time
            
            delta[t, j] = best_score
            backpointer[t, j, 0] = best_prev_state
            backpointer[t, j, 1] = best_start_time
    
    # Find best final state
    best_final_score = -np.inf
    best_final_state = 0
    for j in range(N):
        if delta[T - 1, j] > best_final_score:
            best_final_score = delta[T - 1, j]
            best_final_state = j
    
    # Backtrack to find path
    path = np.zeros(T, dtype=np.int64)
    current_state = best_final_state
    current_time = T - 1
    
    while current_time >= 0:
        start_time = backpointer[current_time, current_state, 1]
        
        # Fill in the segment
        for pos in range(start_time, current_time + 1):
            path[pos] = current_state
        
        # Move to previous segment
        if start_time > 0:
            prev_state = backpointer[current_time, current_state, 0]
            current_state = prev_state
            current_time = start_time - 1
        else:
            break
    
    return path


def brute_force_semi_markov_decode(
    emission_probs: npt.ArrayLike,
    transition_matrix: npt.ArrayLike,
    duration_probs: dict[int, npt.ArrayLike],
    initial_probs: npt.ArrayLike | None = None
) -> npt.NDArray[np.int64]:
    """Find most likely sequence using brute force Semi-Markov computation.
    
    This function examines all possible segmentations to find the most likely path.
    Only suitable for small examples and unit testing due to exponential complexity.
    """
    # Validate and prepare inputs
    emission_probs, transition_matrix, initial_probs, log_duration, max_duration = _validate_semi_markov_inputs(
        emission_probs, transition_matrix, duration_probs, initial_probs, 0.0
    )
    
    T, N = emission_probs.shape
    
    # Safety check to prevent memory issues
    if T > 15 or max_duration > 10:
        raise ValueError(
            f"Brute force method is only suitable for small problems. "
            f"Got T={T}, max_duration={max_duration}. "
            f"Use semi_markov_viterbi_decode for larger problems."
        )
    
    # Generate all possible segmentations
    segmentations = _generate_all_segmentations(T, N, max_duration)
    
    if not segmentations:
        return np.zeros(T, dtype=np.int64)
    
    # Work in log space using shared function for consistency
    epsilon = 0.0  # Use 0 for brute force to be consistent with validation
    log_emission = _safe_log_probabilities(emission_probs, epsilon)
    log_transition = _safe_log_probabilities(transition_matrix, epsilon)
    log_initial = _safe_log_probabilities(initial_probs, epsilon)
    
    best_score = -np.inf
    best_path = None
    
    for segmentation in segmentations:
        score = _score_segmentation(segmentation, log_emission, log_transition, log_initial, log_duration)
        if score > best_score:
            best_score = score
            best_path = segmentation
    
    # Convert segmentation to path
    path = np.zeros(T, dtype=np.int64)
    for start, end, state in best_path:
        path[start:end+1] = state
    
    return path


def _score_segmentation(
    segmentation: list[tuple[int, int, int]],
    log_emission: np.ndarray,
    log_transition: np.ndarray,
    log_initial: np.ndarray,
    log_duration: np.ndarray
) -> float:
    """Score a complete segmentation."""
    if not segmentation:
        return -np.inf
    
    score = 0.0
    
    for i, (start, end, state) in enumerate(segmentation):
        duration = end - start + 1
        
        # Initial state probability for first segment
        if i == 0:
            score += log_initial[state]
        else:
            # Transition from previous state
            prev_state = segmentation[i-1][2]
            score += log_transition[prev_state, state]
        
        # Duration probability
        if duration <= log_duration.shape[1]:
            score += log_duration[state, duration - 1]
        else:
            score += -np.inf  # Invalid duration
        
        # Emission probabilities for this segment
        score += log_emission[start:end+1, state].sum()
    
    return score


def _generate_all_segmentations(T: int, N: int, max_duration: int) -> list[list[tuple[int, int, int]]]:
    """Generate all possible segmentations of sequence of length T."""
    def generate_segments(start_pos: int) -> list[list[tuple[int, int, int]]]:
        if start_pos >= T:
            return [[]]
        
        segmentations = []
        for state in range(N):
            for duration in range(1, min(max_duration + 1, T - start_pos + 1)):
                end_pos = start_pos + duration - 1
                if end_pos >= T:
                    continue
                
                segment = (start_pos, end_pos, state)
                for rest in generate_segments(end_pos + 1):
                    segmentations.append([segment] + rest)
        
        return segmentations
    
    return generate_segments(0)


def _process_region_for_segmentation(args):
    """Helper function for parallel processing of regions in segmentation-based decoding.
    
    This function is defined at module level so it can be pickled for multiprocessing.
    """
    region_data, emission_probs, log_transition, log_initial, log_duration, max_duration, epsilon = args
    start, stop = region_data
    region_emissions = emission_probs[start:stop]
    log_region_emissions = _safe_log_probabilities(region_emissions, epsilon)
    return start, stop, _semi_markov_viterbi_decode(
        log_region_emissions, log_transition, log_initial, log_duration, max_duration
    )


def semi_markov_viterbi_segmentation(
    emission_probs: npt.ArrayLike,
    transition_matrix: npt.ArrayLike,
    duration_probs: dict[int, npt.ArrayLike],
    initial_probs: npt.ArrayLike | None = None,
    epsilon: float = 0.0,
    background_class: int = 0,
    background_class_min_length: int = 8192,
    background_class_buffer: int = 256,
    num_workers: int = 0
) -> npt.NDArray[np.int64]:
    """Find most likely sequence using segmentation-based Semi-Markov Viterbi decoding.
    
    This function identifies long runs of background class (e.g., intergenic regions) and
    processes the intervening complex regions in parallel using full Semi-Markov decoding.
    
    Parameters
    ----------
    emission_probs : npt.ArrayLike
        Matrix of shape (T, N) where T is sequence length and N is number of states
    transition_matrix : npt.ArrayLike
        Matrix of shape (N, N) where element (i, j) represents P(state j | state i)
    duration_probs : dict[int, npt.ArrayLike]
        Dictionary mapping state index to duration probability array
    initial_probs : npt.ArrayLike, optional
        Vector of length N representing initial state probabilities
    epsilon : float, optional
        Epsilon added to probabilities; default is 0
    background_class : int, optional
        State index representing background class (e.g., intergenic); default is 0
    background_class_min_length : int, optional
        Minimum length of background runs to use for segmentation; default is 8192
    background_class_buffer : int, optional
        Buffer around background segments to include in complex regions; default is 256
    num_workers : int, optional
        Number of parallel workers; 0 for sequential processing; default is 0
        
    Returns
    -------
    npt.NDArray[np.int64]
        Vector of length T with most likely state indices at each position
    """
    
    # Validate inputs once
    emission_probs, transition_matrix, initial_probs, log_duration, max_duration = _validate_semi_markov_inputs(
        emission_probs, transition_matrix, duration_probs, initial_probs, epsilon
    )
    
    T, N = emission_probs.shape
    
    # Validate background class arguments
    if not 0 <= background_class < N:
        raise ValueError(f"background_class must be in [0, {N-1}], got {background_class}")
    if background_class_buffer >= background_class_min_length // 2:
        raise ValueError(f"background_class_buffer ({background_class_buffer}) too large relative to background_class_min_length ({background_class_min_length})")
    
    logger.info(f"Starting segmentation-based Semi-Markov Viterbi: T={T:,}, background_class={background_class}, min_length={background_class_min_length:,}, buffer={background_class_buffer}")
    
    # Preprocess inputs once for efficiency
    log_transition = _safe_log_probabilities(transition_matrix, epsilon)
    log_initial = _safe_log_probabilities(initial_probs, epsilon)
    
    # Find background class runs using argmax of emissions
    argmax_labels = np.argmax(emission_probs, axis=1)
    background_mask = (argmax_labels == background_class).astype(np.int8)
    
    # Find intervals of background class
    background_intervals = find_intervals(background_mask)
    
    if background_intervals.size == 0:
        # No background intervals found, process entire sequence
        logger.info("No background intervals found, processing entire sequence")
        log_emission = _safe_log_probabilities(emission_probs, epsilon)
        return _semi_markov_viterbi_decode(
            log_emission, log_transition, log_initial, log_duration, max_duration
        )
    
    # Filter intervals by minimum length
    starts, stops = background_intervals
    lengths = stops - starts + 1
    valid_mask = lengths >= background_class_min_length
    
    if not np.any(valid_mask):
        # No sufficiently long background intervals
        logger.info("No sufficiently long background intervals found, processing entire sequence")
        log_emission = _safe_log_probabilities(emission_probs, epsilon)
        return _semi_markov_viterbi_decode(
            log_emission, log_transition, log_initial, log_duration, max_duration
        )
    
    valid_starts, valid_stops = starts[valid_mask], stops[valid_mask]
    logger.info(f"Found {len(valid_starts)} background intervals for segmentation")
    
    # Create complex regions between background intervals
    complex_regions = []
    
    # Add region before first background interval
    if valid_starts[0] > 0:
        region_start = 0
        region_stop = min(valid_starts[0] + background_class_buffer, T)
        complex_regions.append((region_start, region_stop))
    
    # Add regions between background intervals
    for i in range(len(valid_starts) - 1):
        region_start = max(0, valid_stops[i] - background_class_buffer)
        region_stop = min(valid_starts[i + 1] + background_class_buffer, T)
        if region_stop > region_start:
            complex_regions.append((region_start, region_stop))
    
    # Add region after last background interval
    if valid_stops[-1] < T - 1:
        region_start = max(0, valid_stops[-1] - background_class_buffer)
        region_stop = T
        complex_regions.append((region_start, region_stop))
    
    logger.info(f"Created {len(complex_regions)} complex regions for Semi-Markov decoding")
    
    # Initialize result with background class
    result = np.full(T, background_class, dtype=np.int64)
    
    if not complex_regions:
        return result
    
    # Prepare arguments for parallel processing
    process_args = [
        (region, emission_probs, log_transition, log_initial, log_duration, max_duration, epsilon)
        for region in complex_regions
    ]
    
    if num_workers > 0:
        # Parallel processing
        with mp.Pool(num_workers) as pool:
            region_results = pool.map(_process_region_for_segmentation, process_args)
    else:
        # Sequential processing
        region_results = [_process_region_for_segmentation(args) for args in process_args]
    
    # Stitch results back together
    for start, stop, region_path in region_results:
        # For overlapping regions (due to buffer), prioritize Semi-Markov results over background
        result[start:stop] = region_path
    
    logger.info("Segmentation-based Semi-Markov Viterbi decoding completed")
    return result

