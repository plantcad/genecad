from src.evaluation import evaluate_intervals

def test_exact_matches():
    """Test basic exact matching with no tolerance."""
    pred_intervals = [(1, 5), (10, 15), (20, 25)]
    true_intervals = [(1, 5), (10, 15), (30, 35)]
    
    metrics = evaluate_intervals(pred_intervals, true_intervals)
    
    assert metrics["precision"] == 2/3  # 2 matches out of 3 predictions
    assert metrics["recall"] == 2/3     # 2 matches out of 3 true intervals
    assert metrics["f1_score"] == 2/3   # F1 = 2 * (2/3 * 2/3) / (2/3 + 2/3)

def test_empty_inputs():
    """Test behavior with empty inputs."""
    # Empty predicted intervals
    metrics = evaluate_intervals([], [(1, 5), (10, 15)])
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    
    # Empty true intervals
    metrics = evaluate_intervals([(1, 5), (10, 15)], [])
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    
    # Both empty
    metrics = evaluate_intervals([], [])
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0

def test_multiple_matches():
    """Test exact matching with multiple identical intervals."""
    # Multiple predicted intervals match single true interval
    pred_intervals = [(1, 5), (1, 5), (10, 15)]
    true_intervals = [(1, 5), (20, 25)]
    metrics = evaluate_intervals(pred_intervals, true_intervals)
    assert metrics["precision"] == 1/3  # Only one of the (1,5) predictions counts as TP
    assert metrics["recall"] == 1/2     # Matched one out of two true intervals
    
    # Multiple true intervals match single predicted interval
    pred_intervals = [(1, 5), (10, 15)]
    true_intervals = [(1, 5), (1, 5), (20, 25)]
    metrics = evaluate_intervals(pred_intervals, true_intervals)
    assert metrics["precision"] == 1    # The (1, 5) prediction matches to both true intervals
    assert metrics["recall"] == 2/3     # Both (1, 5) true intervals count as matches
    
    # Multiple matches in both predictions and true intervals
    pred_intervals = [(1, 5), (1, 5), (10, 15), (10, 15)]
    true_intervals = [(1, 5), (1, 5), (10, 15), (20, 25)]
    metrics = evaluate_intervals(pred_intervals, true_intervals)
    assert metrics["precision"] == 3/4  # The (1,5) prediction is a TP for both true (1,5) intervals
    assert metrics["recall"] == 3/4     # All true intervals are matched except (20, 25)

def test_approximate_matches():
    """Test matching with non-zero tolerance."""
    # Test with tolerance=1
    pred_intervals = [(1, 5), (10, 16), (20, 25)]
    true_intervals = [(2, 6), (10, 15), (30, 35)]
    metrics = evaluate_intervals(pred_intervals, true_intervals, tolerance=1)
    assert metrics["precision"] == 2/3  # First two predictions match within tolerance
    assert metrics["recall"] == 2/3     # First two true intervals matched
    
    # Test with tolerance=2
    pred_intervals = [(1, 5), (11, 16)]
    true_intervals = [(3, 7), (10, 15)]
    metrics = evaluate_intervals(pred_intervals, true_intervals, tolerance=2)
    assert metrics["precision"] == 1.0  # Both predictions match within tolerance
    assert metrics["recall"] == 1.0     # Both true intervals matched
