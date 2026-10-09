import numpy as np
import pytest
import xarray as xr

from src.prediction import merge_prediction_datasets, open_segments
from src.tests.segment_test_support import (
    FEATURES,
    write_rank_store,
    write_segments,
)

LENGTH = 1000
SEGMENT = 128


@pytest.fixture
def logits():
    rng = np.random.default_rng(0)
    return {
        strand: rng.normal(size=(LENGTH, len(FEATURES))).astype(np.float32)
        for strand in ("positive", "negative")
    }


@pytest.fixture
def store(tmp_path, logits):
    write_segments(tmp_path, logits, SEGMENT)
    return open_segments(str(tmp_path))


def test_blocks_are_the_segments_in_position_order(store, logits):
    for strand, expected in logits.items():
        blocks = list(store.blocks(strand))
        assert [len(b) for b in blocks] == [SEGMENT] * 7 + [LENGTH - 7 * SEGMENT]
        np.testing.assert_array_equal(np.concatenate(blocks), expected)


def test_reversed_blocks_run_from_the_last_position_backwards(store, logits):
    for strand, expected in logits.items():
        np.testing.assert_array_equal(
            np.concatenate(list(store.blocks(strand, reverse=True))), expected[::-1]
        )


def test_length_features_and_attributes_match_the_merged_dataset(tmp_path, logits):
    write_segments(tmp_path, logits, SEGMENT)
    store = open_segments(str(tmp_path))
    merged = merge_prediction_datasets(str(tmp_path))
    assert store.length == merged.sizes["sequence"] == LENGTH
    assert store.features == merged["feature"].values.tolist() == FEATURES
    assert store.attrs == merged.attrs


@pytest.mark.parametrize("strand", ["positive", "negative"])
def test_windows_read_across_segment_boundaries(store, logits, strand):
    expected = logits[strand]
    bounds = [
        (0, 1),
        (0, SEGMENT),
        (SEGMENT - 1, SEGMENT + 1),
        (100, 700),
        (7 * SEGMENT, LENGTH),
        (LENGTH - 1, LENGTH),
        (300, 300),
    ]
    for start, stop in bounds:
        np.testing.assert_array_equal(
            store.window(strand, start, stop), expected[start:stop]
        )


def test_windows_are_clipped_like_slices(store, logits):
    expected = logits["positive"]
    for start, stop in [(-20, 40), (LENGTH - 30, LENGTH + 500), (500, 400), (-5, -1)]:
        np.testing.assert_array_equal(
            store.window("positive", start, stop), expected[start:stop]
        )
    assert store.window("positive", LENGTH + 10, LENGTH + 20).shape == (
        0,
        len(FEATURES),
    )


def test_windows_can_select_and_reorder_features(store, logits):
    names = ["cds", "intergenic", "intron"]
    columns = [FEATURES.index(name) for name in names]
    np.testing.assert_array_equal(
        store.window("negative", 90, 400, names), logits["negative"][90:400][:, columns]
    )
    with pytest.raises(KeyError, match="not_a_feature"):
        store.window("negative", 0, 10, ["intergenic", "not_a_feature"])


def test_more_windows_than_cached_segments_stay_correct(store, logits):
    rng = np.random.default_rng(1)
    for _ in range(200):
        start = int(rng.integers(0, LENGTH))
        stop = int(start + rng.integers(0, 300))
        np.testing.assert_array_equal(
            store.window("positive", start, stop), logits["positive"][start:stop]
        )


def test_rank_stores_are_not_segmented(tmp_path, logits):
    write_rank_store(tmp_path, logits)
    assert open_segments(str(tmp_path)) is None


def test_unfinished_predictions_are_refused(tmp_path, logits):
    write_segments(tmp_path, logits, SEGMENT)
    (tmp_path / "_SUCCESS.json").unlink()
    with pytest.raises(FileNotFoundError):
        open_segments(str(tmp_path))


def test_a_segment_changed_after_commit_is_refused(tmp_path, logits):
    write_segments(tmp_path, logits, SEGMENT)
    first = sorted(tmp_path.glob("segment.positive.0.*.json"))[0].with_suffix(".zarr")
    second = sorted(tmp_path.glob("segment.positive.1.*.json"))[0].with_suffix(".zarr")
    swapped = xr.open_zarr(second, group="positive").load()
    swapped.to_zarr(first, group="positive", mode="w", zarr_format=2, consolidated=True)
    with pytest.raises(ValueError, match="checksum"):
        open_segments(str(tmp_path))
