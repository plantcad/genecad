import pytest
import numpy as np
import pandas as pd
from src.sequence import (
    convert_to_biluo_labels,
    convert_to_entity_labels,
    find_group_intervals,
    find_intervals,
    convert_biluo_index_to_class_name,
    convert_biluo_index_to_entity_name,
    find_overlapping_intervals,
    convert_entity_intervals_to_labels,
    convert_biluo_entity_names,
    create_prediction_windows,
    create_sequence_windows,
    viterbi_decode,
    brute_force_decode,
)


# fmt: off
@pytest.mark.parametrize("labels, expected", [
    # Test case 1: No intervals
    (np.array([0, 0, 0, 0]), np.empty((0, 2), dtype=int)),
    # Test case 2: One large interval
    (np.array([1, 1, 1, 1]), np.array([[0, 3]])),
    # Test case 3: Unit-length intervals
    (np.array([0, 1, 0, 1, 0]), np.array([[1, 1], [3, 3]])),
    # Test case 4: Empty array
    (np.array([]), np.empty((0, 2), dtype=int)),
    # Test case 5: Starting and ending with ones
    (np.array([1, 1, 0, 0, 1]), np.array([[0, 1], [4, 4]])),
    # Test case 6: Interior intervals only
    (np.array([0, 0, 1, 1, 1, 0, 1, 1, 0]), np.array([[2, 4], [6, 7]])),
    # Test case 7: Multiple intervals of varying lengths and boundary intervals on both ends
    (np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1]), np.array([[0, 0], [3, 3], [5, 6], [9, 11]])),
    # Test case 8: One value, one interval
    (np.array([1]), np.array([[0, 0]])),
    # Test case 9: One value, no intervals
    (np.array([0]), np.empty((0, 2), dtype=int)),
])
# fmt: on
def test_find_intervals(labels, expected):
    np.testing.assert_array_equal(find_intervals(labels).T, expected)

def test_find_intervals_invalid_values():
    with pytest.raises(ValueError, match="Labels must be 0 or 1"):
        find_intervals(np.array([0, 1, 2]))

def test_find_intervals_invalid_shape():
    with pytest.raises(ValueError, match="Labels must be a 1D array"):
        find_intervals(np.array([[0, 1], [1, 0]]))

@pytest.mark.parametrize("labels, mask, expected", [
    # 1) Mask always True -> same as no mask
    (np.array([1, 1, 1]), np.array([True, True, True]), np.array([[0, 2]])),
    # 2) Mask always False -> all intervals omitted
    (np.array([1, 1, 1]), np.array([False, False, False]), np.empty((0, 2), dtype=int)),
    # 3) Partial mask -> only unmasked intervals remain
    (np.array([1, 1, 0, 1]), np.array([True, False, True, True]), np.array([[3, 3]])),
    # 3) Singleton mask -> removes one unit-length interval
    (np.array([0, 0, 1, 0, 1, 1]), np.array([True, True, False, True, True, True]), np.array([[4, 5]])),
    # 4) No intervals with mask -> empty intervals
    (np.array([0, 0, 0, 0]), np.array([True, True, True, True]), np.empty((0, 2), dtype=int)),
])
def test_find_intervals_with_mask(labels, mask, expected):
    result = find_intervals(labels, mask).T
    np.testing.assert_array_equal(result, expected)



# fmt: off
@pytest.mark.parametrize("labels, class_groups, expected", [
    # Test case 1: Empty labels
    (
        np.array([], dtype=np.int32),
        [[1, 2], [3]],
        [
            np.empty((2, 0), dtype=np.int32),
            np.empty((2, 0), dtype=np.int32)
        ]
    ),
    # Test case 2: No matches (only background class)
    (
        np.array([0, 0, 0]),
        [[1], [2]],
        [
            np.empty((2, 0), dtype=np.int32),
            np.empty((2, 0), dtype=np.int32)
        ]
    ),
    # Test case 3: Single class with one interval
    (
        np.array([0, 1, 1, 0]),
        [[1]],
        [
            np.array([[1], [2]])
        ]
    ),
    # Test case 4: Multiple classes in the same group
    (
        np.array([0, 1, 2, 1, 0]),
        [[1, 2]],
        [
            np.array([[1], [3]])
        ]
    ),
    # Test case 5: Multiple groups with separate intervals
    (
        np.array([0, 1, 1, 0, 2, 2, 0]),
        [[1], [2]],
        [
            np.array([[1], [2]]),
            np.array([[4], [5]])
        ]
    ),
    # Test case 6: The example from docstring
    (
        np.array([0, 1, 1, 1, 0, 2, 2, 3, 0]),
        [[1, 2], [3]],
        [
            np.array([[1, 5], [3, 6]]),
            np.array([[7], [7]])
        ]
    ),
])
# fmt: on
def test_find_group_intervals(labels, class_groups, expected):
    result = find_group_intervals(labels, class_groups)

    # Check length matches
    assert len(result) == len(expected)

    # For each group's intervals, verify they match expected
    for i, intervals in enumerate(result):
        if len(expected) == 0:
            continue
        expected_array = expected[i]
        np.testing.assert_array_equal(intervals, expected_array)

def test_find_group_intervals_with_mask():
    labels = np.array([0, 1, 1, 1, 0, 2, 2, 0])
    mask = np.array([True, True, False, True, True, True, True, True])
    class_groups = [[1], [2]]

    result = find_group_intervals(labels, class_groups, mask)

    # The interval [1, 3] for class 1 should be filtered out because of the mask
    # Only the interval [5, 6] for class 2 should remain
    expected = [
        np.empty((2, 0), dtype=np.int32),  # No valid intervals for class 1
        np.array([[5], [6]])               # Interval for class 2
    ]

    assert len(result) == len(expected)
    for i, intervals in enumerate(result):
        np.testing.assert_array_equal(intervals, expected[i])

def test_find_group_intervals_invalid_inputs():
    # Test with non-integer class groups
    labels = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="class_groups must contain only integer values"):
        find_group_intervals(labels, [["1"], [2]])

    # Test with non-positive class groups
    with pytest.raises(ValueError, match="class_groups must contain only positive integers"):
        find_group_intervals(labels, [[0], [2]])

    # Test with invalid labels shape
    labels_2d = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="labels must be a 1D array"):
        find_group_intervals(labels_2d, [[1], [2]])

    # Test with non-integer labels
    labels_float = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="labels must be a numpy array of integers"):
        find_group_intervals(labels_float, [[1], [2]])

    # Test with labels containing values not in class_groups
    labels_invalid = np.array([0, 1, 2, 4])
    with pytest.raises(ValueError, match="labels must be in the set"):
        find_group_intervals(labels_invalid, [[1], [2]])


# fmt: off
@pytest.mark.parametrize("class_indices, expected", [
    # Test case 1: Empty array
    (np.array([], dtype=np.int32), np.array([], dtype=np.int32)),

    # Test case 2: Only background (0) and mask (-1) classes
    (np.array([-1, 0, -1, 0, 0], dtype=np.int32),
     np.array([-1, 0, -1, 0, 0], dtype=np.int32)),

    # Test case 3: Single token entities (U-tag)
    (np.array([0, 1, 0, 2, 0], dtype=np.int32),
     np.array([0, 4, 0, 8, 0], dtype=np.int32)),

    # Test case 4: Multi-token entities (B-I-L tags)
    (np.array([0, 1, 1, 1, 1, 1, 0], dtype=np.int32),
     np.array([0, 1, 2, 2, 2, 3, 0], dtype=np.int32)),

    # Test case 5: Multiple entity types in sequence
    (np.array([0, 1, 1, 0, 2, 2, 2, 0], dtype=np.int32),
     np.array([0, 1, 3, 0, 5, 6, 7, 0], dtype=np.int32)),

    # Test case 6: Adjacent entities of same type - should be merged
    (np.array([1, 1, 1], dtype=np.int32),
     np.array([1, 2, 3], dtype=np.int32)),

    # Test case 7: Adjacent entities of different types
    (np.array([1, 1, 2, 2], dtype=np.int32),
     np.array([1, 3, 5, 7], dtype=np.int32)),

    # Test case 8: Class 3 entity tags
    (np.array([0, 3, 3, 3, 0], dtype=np.int32),
     np.array([0, 9, 10, 11, 0], dtype=np.int32)),

    # Test case 9: Complex mixed scenario
    (np.array([0, 1, 0, 2, 2, 0, 3, 0, 1, 1, 1, -1], dtype=np.int32),
     np.array([0, 4, 0, 5, 7, 0, 12, 0, 1, 2, 3, -1], dtype=np.int32)),

    # Test case 10: Single token of each class
    (np.array([1, 2, 3], dtype=np.int32),
     np.array([4, 8, 12], dtype=np.int32)),

    # Test case 11: Single token
    (np.array([1], dtype=np.int32),
     np.array([4], dtype=np.int32)),

    # Test case 12: Empty array
    (np.array([], dtype=np.int32),
     np.array([], dtype=np.int32)),
])
# fmt: on
def test_convert_to_biluo(class_indices, expected):
    result = convert_to_biluo_labels(class_indices)
    np.testing.assert_array_equal(result, expected)


# fmt: off
@pytest.mark.parametrize("biluo_labels, expected", [
    # Test case 1: Basic entity conversion
    (
        np.array([0, 1, 2, 2, 2, 3, 0]),
        np.array([0, 1, 1, 1, 1, 1, 0])
    ),
    # Test case 2: Multiple entities with different types
    (
        np.array([0, 1, 2, 2, 2, 3, 0, 4, 0, 5, 6, 6, 7]),
        np.array([0, 1, 1, 1, 1, 1, 0, 1, 0, 2, 2, 2, 2])
    ),
])
# fmt: on
def test_convert_to_entity_labels(biluo_labels, expected):
    entity_labels = convert_to_entity_labels(biluo_labels)
    np.testing.assert_array_equal(entity_labels, expected)

# fmt: off
@pytest.mark.parametrize("label, entity_names, sentinel_names, expected", [
    # Test case 1: "O" tag (Outside/Background) with default sentinel names
    (0, ["GENE", "PROTEIN"], None, "background"),
    (-1, ["GENE", "PROTEIN"], None, "mask"),

    # Test case 2: Custom sentinel names
    (0, ["GENE", "PROTEIN"], ("MASKED", "BG"), "BG"),
    (-1, ["GENE", "PROTEIN"], ("MASKED", "BG"), "MASKED"),

    # Test case 3: First entity type
    (1, ["GENE", "PROTEIN"], None, "GENE"),
    (2, ["GENE", "PROTEIN"], None, "GENE"),
    (3, ["GENE", "PROTEIN"], None, "GENE"),
    (4, ["GENE", "PROTEIN"], None, "GENE"),

    # Test case 4: Second entity type
    (5, ["GENE", "PROTEIN"], None, "PROTEIN"),
    (6, ["GENE", "PROTEIN"], None, "PROTEIN"),
    (7, ["GENE", "PROTEIN"], None, "PROTEIN"),
    (8, ["GENE", "PROTEIN"], None, "PROTEIN"),

    # Test case 5: Multiple entity types
    (9, ["GENE", "PROTEIN", "DRUG"], None, "DRUG"),
    (11, ["GENE", "PROTEIN", "DRUG"], None, "DRUG"),
])
# fmt: on
def test_convert_biluo_index_to_entity_name(label, entity_names, sentinel_names, expected):
    result = convert_biluo_index_to_entity_name(label, entity_names, sentinel_names)
    assert result == expected

# Test error cases for convert_biluo_index_to_entity_name
def test_convert_biluo_index_to_entity_name_errors():
    # Test invalid label
    with pytest.raises(ValueError, match="Label must be in"):
        convert_biluo_index_to_entity_name(-2, ["O", "GENE"])

    # Test invalid sentinel_names length
    with pytest.raises(ValueError, match="sentinel_names must be a tuple of length 2"):
        convert_biluo_index_to_entity_name(0, ["O", "GENE"], ("MASK",))

# fmt: off
@pytest.mark.parametrize("label, entity_names, sentinel_names, expected", [
    # Test case 1: "O" tag (Outside/Background) with default sentinel names
    (0, ["GENE", "PROTEIN"], None, "background"),
    (-1, ["GENE", "PROTEIN"], None, "mask"),

    # Test case 2: Custom sentinel names
    (0, ["GENE", "PROTEIN"], ("MASKED", "BG"), "BG"),
    (-1, ["GENE", "PROTEIN"], ("MASKED", "BG"), "MASKED"),

    # Test case 3: BILUO tags for first entity type
    (1, ["GENE", "PROTEIN"], None, "B-GENE"),
    (2, ["GENE", "PROTEIN"], None, "I-GENE"),
    (3, ["GENE", "PROTEIN"], None, "L-GENE"),
    (4, ["GENE", "PROTEIN"], None, "U-GENE"),

    # Test case 4: BILUO tags for second entity type
    (5, ["GENE", "PROTEIN"], None, "B-PROTEIN"),
    (6, ["GENE", "PROTEIN"], None, "I-PROTEIN"),
    (7, ["GENE", "PROTEIN"], None, "L-PROTEIN"),
    (8, ["GENE", "PROTEIN"], None, "U-PROTEIN"),

    # Test case 5: BILUO tags for third entity type
    (9, ["GENE", "PROTEIN", "DRUG"], None, "B-DRUG"),
    (10, ["GENE", "PROTEIN", "DRUG"], None, "I-DRUG"),
    (11, ["GENE", "PROTEIN", "DRUG"], None, "L-DRUG"),
    (12, ["GENE", "PROTEIN", "DRUG"], None, "U-DRUG"),
])
# fmt: on
def test_convert_biluo_index_to_class_name(label, entity_names, sentinel_names, expected):
    result = convert_biluo_index_to_class_name(label, entity_names, sentinel_names)
    assert result == expected

# fmt: off
@pytest.mark.parametrize("entity_names, expected", [
    # Test case 1: Single entity list
    (
        ["GENE"],
        ["B-GENE", "I-GENE", "L-GENE", "U-GENE"]
    ),
    # Test case 2: Multiple entities
    (
        ["GENE", "PROTEIN", "DRUG"],
        [
            "B-GENE", "I-GENE", "L-GENE", "U-GENE",
            "B-PROTEIN", "I-PROTEIN", "L-PROTEIN", "U-PROTEIN",
            "B-DRUG", "I-DRUG", "L-DRUG", "U-DRUG"
        ]
    ),
    # Test case 3: Empty entity list
    ([], [])
])
# fmt: on
def test_convert_biluo_entity_names(entity_names, expected):
    result = convert_biluo_entity_names(entity_names)
    assert result == expected

# fmt: off
@pytest.mark.parametrize("starts, stops, expected", [
    # Empty case
    ([], [], pd.Series([], dtype=bool)),

    # Single interval can't overlap with itself
    ([5], [10], pd.Series([False])),

    # Non-overlapping intervals
    ([1, 5, 10, 15], [3, 8, 12, 20], pd.Series([False, False, False, False])),

    # Adjacent intervals don't count as overlapping
    ([1, 5, 8], [5, 8, 10], pd.Series([False, False, False])),

    # Basic overlapping case - first two intervals overlap
    ([1, 3, 7, 15], [5, 6, 12, 20], pd.Series([True, True, False, False])),

    # Nested intervals
    ([1, 2, 10], [8, 5, 15], pd.Series([True, True, False])),

    # All intervals form an overlapping chain
    ([1, 3, 5, 7], [4, 6, 8, 10], pd.Series([True, True, True, True])),
])
# fmt: on
def test_find_overlapping_intervals(starts, stops, expected):
    """Test find_overlapping_intervals with various interval configurations"""
    starts_series = pd.Series(starts)
    stops_series = pd.Series(stops)
    result = find_overlapping_intervals(starts_series, stops_series)

    assert isinstance(result, pd.Series)
    assert len(result) == len(expected)
    pd.testing.assert_series_equal(result, expected, check_index=False)


def test_find_overlapping_intervals_with_index():
    """Test find_overlapping_intervals preserves the index of the input Series"""
    starts = pd.Series([1, 3, 7], index=['a', 'b', 'c'])
    stops = pd.Series([5, 6, 12], index=['a', 'b', 'c'])
    result = find_overlapping_intervals(starts, stops)

    assert isinstance(result, pd.Series)
    assert list(result.index) == ['a', 'b', 'c']
    assert result['a'] and result['b']  # First two intervals overlap
    assert not result['c']  # Third interval doesn't overlap


def test_find_overlapping_intervals_unsorted():
    """Test find_overlapping_intervals with unsorted starts raises ValueError"""
    starts = pd.Series([5, 1, 10])  # Not sorted
    stops = pd.Series([8, 4, 15])
    with pytest.raises(ValueError, match="Interval starts must be sorted"):
        find_overlapping_intervals(starts, stops)


# fmt: off
@pytest.mark.parametrize("intervals, domain, num_labels, expected", [
    # Test case 1: Empty intervals - all zeros
    (
        [],
        (0, 5),
        2,
        np.zeros((5, 2), dtype=np.int8)
    ),

    # Test case 2: Single interval for label 1
    (
        [{"start": 1, "stop": 3, "label": 1}],
        (0, 5),
        2,
        np.array([
            [0, 0],
            [1, 0],
            [1, 0],
            [0, 0],
            [0, 0]
        ], dtype=np.int8)
    ),

    # Test case 3: Single interval for label 2
    (
        [{"start": 2, "stop": 4, "label": 2}],
        (0, 5),
        2,
        np.array([
            [0, 0],
            [0, 0],
            [0, 1],
            [0, 1],
            [0, 0]
        ], dtype=np.int8)
    ),

    # Test case 4: Multiple non-overlapping intervals for different labels
    (
        [
            {"start": 1, "stop": 3, "label": 1},
            {"start": 3, "stop": 5, "label": 2}
        ],
        (0, 6),
        2,
        np.array([
            [0, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 0]
        ], dtype=np.int8)
    ),

    # Test case 5: Multiple intervals for same label
    (
        [
            {"start": 0, "stop": 2, "label": 1},
            {"start": 3, "stop": 5, "label": 1}
        ],
        (0, 5),
        2,
        np.array([
            [1, 0],
            [1, 0],
            [0, 0],
            [1, 0],
            [1, 0]
        ], dtype=np.int8)
    ),

    # Test case 6: Domain with offset
    (
        [{"start": 10, "stop": 12, "label": 1}],
        (10, 15),
        2,
        np.array([
            [1, 0],
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ], dtype=np.int8)
    ),

    # Test case 7: More than 2 labels
    (
        [
            {"start": 0, "stop": 2, "label": 1},
            {"start": 2, "stop": 3, "label": 2},
            {"start": 3, "stop": 5, "label": 3},
        ],
        (0, 5),
        3,
        np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1]
        ], dtype=np.int8)
    ),

    # Test case 8: Adjacent intervals
    (
        [
            {"start": 0, "stop": 2, "label": 1},
            {"start": 2, "stop": 4, "label": 1},
            {"start": 4, "stop": 6, "label": 1}
        ],
        (0, 6),
        2,
        np.array([
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0]
        ], dtype=np.int8)
    ),
])
# fmt: on
def test_convert_entity_intervals_to_labels(intervals, domain, num_labels, expected):
    """Test successful conversion of entity intervals to label matrix."""
    # Convert input to DataFrame
    df = pd.DataFrame(intervals, columns=["start", "stop", "label"])

    # Call the function
    result = convert_entity_intervals_to_labels(
        intervals=df,
        domain=domain,
        num_labels=num_labels,
        on_overlap="raise"
    )

    # Assert the result matches expected
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.int8

# fmt: off
@pytest.mark.parametrize("intervals, domain, num_labels", [
    # Test case 1: Overlapping intervals for same label
    (
        [
            {"start": 1, "stop": 4, "label": 1},
            {"start": 3, "stop": 5, "label": 1}
        ],
        (0, 6),
        2
    ),

    # Test case 2: Overlapping intervals for different labels
    (
        [
            {"start": 1, "stop": 4, "label": 1},
            {"start": 3, "stop": 5, "label": 2}
        ],
        (0, 6),
        2
    ),

    # Test case 3: Nested intervals
    (
        [
            {"start": 1, "stop": 5, "label": 1},
            {"start": 2, "stop": 4, "label": 2}
        ],
        (0, 6),
        2
    ),
])
# fmt: on
def test_convert_entity_intervals_to_labels_overlap_errors(intervals, domain, num_labels):
    """Test that overlapping intervals raise appropriate errors."""
    # Convert input to DataFrame
    df = pd.DataFrame(intervals, columns=["start", "stop", "label"])

    # Call the function and expect it to raise an error
    with pytest.raises(ValueError, match="Overlapping intervals detected in generated labels"):
        convert_entity_intervals_to_labels(
            intervals=df,
            domain=domain,
            num_labels=num_labels,
            on_overlap="raise"
        )

# -------------------------------------------------------------------------------------------------
# Tests for create_prediction_windows
# -------------------------------------------------------------------------------------------------

def test_create_prediction_windows():
    """Test that create_prediction_windows generates correct windows."""
    # Basic test case with stride == window_size (no overlap)
    sequence_length = 16
    window_size = 8
    stride = 8  # Equivalent to stride_factor = 1.0

    # Expected result explanation:
    # With stride = window_size, there's no overlap between windows
    # Each window's valid region equals its full size (no margins)
    #
    # First row [0, 8, 0, 8]:
    #   - Window starts at 0, ends at 8
    #   - Valid region starts at 0, ends at 8 (full window)
    # Second row [8, 16, 8, 16]:
    #   - Window starts at 8, ends at 16
    #   - Valid region starts at 8, ends at 16 (full window)
    expected = np.array([
        [0, 8, 0, 8],
        [8, 16, 8, 16]
    ])

    result = create_prediction_windows(sequence_length, window_size, stride)
    np.testing.assert_array_equal(result, expected)

    # Test with stride = window_size / 2 (50% overlap)
    stride = 4 # Equivalent to stride_factor = 0.5

    # Expected result explanation:
    # With stride = window_size / 2, windows overlap by 50%
    # Each window's valid region is the middle 50% of the window
    # Margins are 25% on each side (window_size - stride)/2 = (8 - 4)/2 = 2
    #
    # First row [0, 8, 0, 6]:
    #   - Window starts at 0, ends at 8
    #   - Valid region starts at 0, ends at 6 (no left margin, but has right margin)
    # Second row [4, 12, 6, 10]:
    #   - Window starts at 4, ends at 12
    #   - Valid region starts at 6, ends at 10 (middle 50% of window)
    # Third row [8, 16, 10, 16]:
    #   - Window starts at 8, ends at 16
    #   - Valid region starts at 10, ends at 16 (has left margin, but no right margin)
    expected_half_overlap = np.array([
        [0, 8, 0, 6],
        [4, 12, 6, 10],
        [8, 16, 10, 16]
    ])

    result = create_prediction_windows(sequence_length, window_size, stride)
    np.testing.assert_array_equal(result, expected_half_overlap)

    # Test with stride = window_size / 4 (75% overlap)
    stride = 2 # Equivalent to stride_factor = 0.25

    # Expected result explanation:
    # With stride = window_size / 4, windows overlap by 75%
    # Each window's valid region is the middle 25% of the window
    # Margins are 37.5% on each side (window_size - stride)/2 = (8 - 2)/2 = 3
    #
    # Array has 7 windows with significant overlap
    expected_quarter_stride = np.array([
        [0, 8, 0, 5],
        [2, 10, 5, 7],
        [4, 12, 7, 9],
        [6, 14, 9, 11],
        [8, 16, 11, 16]
    ])

    result = create_prediction_windows(sequence_length, window_size, stride)
    np.testing.assert_array_equal(result, expected_quarter_stride)

    # Test with invalid stride (too small)
    with pytest.raises(ValueError, match="stride must be in range"):
        create_prediction_windows(16, 8, 0)

    # Test with invalid stride (too large)
    with pytest.raises(ValueError, match="stride must be in range"):
        create_prediction_windows(16, 8, 9)

    # Test with odd sequence_length
    with pytest.raises(ValueError, match="sequence_length must be even"):
        create_prediction_windows(15, 8, 4)

    # Test with odd window_size
    with pytest.raises(ValueError, match="window_size must be even"):
        create_prediction_windows(16, 7, 4)

    # Test with sequence_length not divisible by stride
    with pytest.raises(ValueError, match="sequence_length must be divisible by stride"):
        create_prediction_windows(16, 8, 3)


def test_create_representative_prediction_windows():
    sequence_length = 8192 * 3
    window_size = 8192
    stride = window_size // 2
    expected = np.array([
        [    0,  8192,     0,  6144],
        [ 4096, 12288,  6144, 10240],
        [ 8192, 16384, 10240, 14336],
        [12288, 20480, 14336, 18432],
        [16384, 24576, 18432, 24576]
    ])
    result = create_prediction_windows(sequence_length, window_size, stride)
    np.testing.assert_array_equal(result, expected)

def test_create_sequence_windows():
    chrom_lengths = {
        "Chr1": 380346,
        "Chr2": 246229,
        "Chr3": 293248,
        "Chr4": 232314,
        "Chr5": 337194,
        "ChrC": 1931,
        "ChrM": 4587,
    }
    window_size = 8192
    stride = window_size // 2
    for _, length in chrom_lengths.items():
        inputs = np.arange(1, length + 1)
        windows = create_sequence_windows(inputs, window_size, stride)
        outputs = np.concatenate([chunk[slice(*local_window)] for chunk, local_window, _ in windows])
        np.testing.assert_array_equal(outputs, inputs)


def test_viterbi_decode():
    """Test viterbi_decode with various examples comparing to known optimal paths."""

    # Test case 1: Simple alternating pattern
    # --------------------------------------
    observations = np.array([
        [0.9, 0.1],  # Strongly favors state 0
        [0.2, 0.8],  # Favors state 1
        [0.7, 0.3],  # Favors state 0
        [0.3, 0.7],  # Favors state 1
        [0.8, 0.2],  # Strongly favors state 0
    ])

    transitions = np.array([
        [0.7, 0.3],  # From state 0
        [0.3, 0.7],  # From state 1
    ])

    start_probs = np.array([0.5, 0.5])

    # Get the ground truth from brute force
    brute_path = brute_force_decode(observations, transitions, start_probs)

    # With these transitions, optimal path is all state 0
    known_optimal_path = np.array([0, 0, 0, 0, 0], dtype=np.int64)
    np.testing.assert_array_equal(
        brute_path, known_optimal_path,
        "Brute force decoder didn't match expected path"
    )

    # Verify viterbi matches the brute force result
    viterbi_path = viterbi_decode(observations, transitions, start_probs)
    np.testing.assert_array_equal(
        viterbi_path, brute_path,
        "Viterbi decoder didn't match brute force decoder"
    )

    # Test case 2: Three state system
    # -----------------------------
    observations_3state = np.array([
        [0.7, 0.2, 0.1],  # Favors state 0
        [0.1, 0.6, 0.3],  # Favors state 1
        [0.2, 0.1, 0.7],  # Favors state 2
    ])

    transitions_3state = np.array([
        [0.5, 0.3, 0.2],  # From state 0
        [0.2, 0.6, 0.2],  # From state 1
        [0.1, 0.4, 0.5],  # From state 2
    ])

    start_probs_3state = np.array([0.4, 0.3, 0.3])

    # Get the ground truth from brute force
    brute_path_3state = brute_force_decode(observations_3state, transitions_3state, start_probs_3state)

    # The optimal path should follow the highest emission probabilities
    known_optimal_path_3state = np.array([0, 1, 2], dtype=np.int64)
    np.testing.assert_array_equal(
        brute_path_3state, known_optimal_path_3state,
        "Brute force decoder didn't match expected path for 3-state system"
    )

    # Verify viterbi matches the brute force result
    viterbi_path_3state = viterbi_decode(observations_3state, transitions_3state, start_probs_3state)
    np.testing.assert_array_equal(
        viterbi_path_3state, brute_path_3state,
        "Viterbi decoder didn't match brute force decoder for 3-state system"
    )

    # Test case 3: Transitions overcome strong emissions
    # ------------------------------------------------
    observations_smoothed = np.array([
        [0.9, 0.1],  # Strongly favors state 0
        [0.9, 0.1],  # Strongly favors state 0
        [0.1, 0.9],  # Strongly favors state 1 (outlier!)
        [0.9, 0.1],  # Strongly favors state 0
        [0.9, 0.1],  # Strongly favors state 0
    ])

    # Very high probability to stay in the same state (strong smoothing)
    transitions_smoothed = np.array([
        [0.95, 0.05],  # Strong preference to stay in state 0
        [0.05, 0.95],  # Strong preference to stay in state 1
    ])

    start_probs_smoothed = np.array([0.5, 0.5])

    # Get the ground truth from brute force
    brute_path_smoothed = brute_force_decode(observations_smoothed, transitions_smoothed, start_probs_smoothed)

    # Due to strong transition constraints, the optimal path should smooth out the outlier
    known_optimal_path_smoothed = np.array([0, 0, 0, 0, 0], dtype=np.int64)
    np.testing.assert_array_equal(
        brute_path_smoothed, known_optimal_path_smoothed,
        "Brute force decoder didn't match expected smoothed path"
    )

    # Verify viterbi matches the brute force result
    viterbi_path_smoothed = viterbi_decode(observations_smoothed, transitions_smoothed, start_probs_smoothed)
    np.testing.assert_array_equal(
        viterbi_path_smoothed, brute_path_smoothed,
        "Viterbi decoder didn't match brute force decoder for smoothed path"
    )

    # Verify this path doesn't just follow strongest emissions
    strongest_emission_path = np.argmax(observations_smoothed, axis=1)
    assert not np.array_equal(viterbi_path_smoothed, strongest_emission_path), \
        "Viterbi should not simply pick the highest emission probability at each step"
