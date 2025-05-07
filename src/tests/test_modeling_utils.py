import pytest
import torch
from src.modeling import position_boundary_indices

@pytest.mark.parametrize("input_shape, expected_output", [
    # Case 1: Typical case
    ((2, 5), torch.tensor([
        [0, 1, 1, 1, 2],
        [0, 1, 1, 1, 2]
    ], dtype=torch.long)),
    
    # Case 2: Single sequence
    ((1, 5), torch.tensor([
        [0, 1, 1, 1, 2]
    ], dtype=torch.long)),
    
    # Case 3: Sequences with only 2 items
    ((2, 2), torch.tensor([
        [0, 2],
        [0, 2]
    ], dtype=torch.long)),
    
    # Case 4: Sequences with only 1 item
    ((2, 1), torch.tensor([
        [1],
        [1]
    ], dtype=torch.long)),
])
def test_position_boundary_indices_parametrized(input_shape, expected_output):
    input_ids = torch.zeros(input_shape, dtype=torch.long)
    actual = position_boundary_indices(input_ids)
    assert torch.equal(actual, expected_output)

def test_position_boundary_indices_empty():
    # Case 5: Empty input array (sequences of length 0)
    input_ids = torch.zeros((2, 0), dtype=torch.long)
    actual = position_boundary_indices(input_ids)
    expected = torch.zeros((2, 0), dtype=torch.long)
    assert torch.equal(actual, expected)

# Test error case for non-2D input
def test_position_boundary_indices_invalid_dim():
    with pytest.raises(ValueError):
        position_boundary_indices(torch.zeros((3, 4, 5)))