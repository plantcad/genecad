import pytest
import os
import numpy as np
import xarray as xr
import tempfile
from unittest.mock import patch

pytest.importorskip("torch")
from src.dataset import XarrayDataset


def test_xarray_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create three small test datasets
        for i in range(3):
            ds = xr.Dataset(
                data_vars={
                    "input_ids": (
                        ["sample", "sequence"],
                        np.ones((3, 10), dtype=np.int32) * i,
                    ),
                    "embeddings": (
                        ["sample", "sequence"],
                        np.ones((3, 10), dtype=np.float32) * (i + 1),
                    ),
                },
                coords={
                    "sample": np.arange(3),
                    "sequence": np.arange(10),
                },
            )
            ds.to_zarr(os.path.join(tmpdir, f"dataset_{i}.zarr"), zarr_format=2)

        # Test dataset with wildcards using default transform
        dataset = XarrayDataset(os.path.join(tmpdir, "dataset_*.zarr"), chunk_size=2)

        # Check length
        assert len(dataset) == 9

        # Verify sequential access and chunk caching
        # Use mock.patch to track _load_chunk calls
        with patch.object(
            XarrayDataset, "_load_chunk", wraps=dataset._load_chunk
        ) as mock_load_chunk:
            # Access all items sequentially and verify correct values
            expected_input_ids = []
            for i in range(3):  # For each dataset
                for _ in range(3):  # For each sample in dataset
                    expected_input_ids.append(np.ones(10, dtype=np.int32) * i)

            # Check each item
            for idx in range(9):
                item = dataset[idx]

                # Verify item contents
                assert "input_ids" in item
                assert "embeddings" in item
                assert np.array_equal(item["input_ids"], expected_input_ids[idx])

            # Verify _load_chunk was called the minimum number of times (expected 6 chunks)
            # 3 datasets with 3 samples each, chunk size of 2 = 6 chunks
            assert mock_load_chunk.call_count == 6

            # Verify the specific chunks that were loaded
            expected_calls = [
                (0, 0),  # Dataset 0, sample 0 (for samples 0-1)
                (0, 2),  # Dataset 0, sample 2 (for samples 2)
                (1, 0),  # Dataset 1, sample 0 (for samples 3-4)
                (1, 2),  # Dataset 1, sample 2 (for samples 5)
                (2, 0),  # Dataset 2, sample 0 (for samples 6-7)
                (2, 2),  # Dataset 2, sample 2 (for sample 8)
            ]
            assert len(mock_load_chunk.call_args_list) == 6
            for i, call in enumerate(mock_load_chunk.call_args_list):
                assert call[0][0] == expected_calls[i][0]  # Check dataset_idx
                assert call[0][1] == expected_calls[i][1]  # Check local_dataset_idx

        # Test out-of-bounds access (negative index)
        with pytest.raises(IndexError):
            dataset[-1]

        # Test out-of-bounds access (beyond last index)
        with pytest.raises(IndexError):
            dataset[9]

        # Test with max_sample_count
        limited_dataset = XarrayDataset(
            os.path.join(tmpdir, "dataset_*.zarr"), max_sample_count=5
        )
        assert len(limited_dataset) == 5


def test_single_zarr_full_chunk():
    """Test XarrayDataset with single zarr file and chunk size equal to sequence dimension."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create single test dataset
        original_ds = xr.Dataset(
            data_vars={
                "input_ids": (["sample", "sequence"], np.arange(12).reshape(3, 4)),
                "embeddings": (["sample", "sequence"], np.arange(12, 24).reshape(3, 4)),
            },
            coords={
                "sample": np.arange(3),
                "sequence": np.arange(4),
            },
        )
        zarr_path = os.path.join(tmpdir, "single_dataset.zarr")
        original_ds.to_zarr(zarr_path, zarr_format=2)

        # Create dataset with chunk size equal to sequence dimension
        dataset = XarrayDataset(zarr_path, chunk_size=4)

        # Verify single chunk contains entire original dataset
        with patch.object(
            XarrayDataset, "_load_chunk", wraps=dataset._load_chunk
        ) as mock_load_chunk:
            # Access all samples
            for idx in range(len(dataset)):
                item = dataset[idx]
                # Verify data matches original
                assert np.array_equal(
                    item["input_ids"], original_ds.input_ids.values[idx]
                )
                assert np.array_equal(
                    item["embeddings"], original_ds.embeddings.values[idx]
                )

            # Verify only one chunk was loaded
            assert mock_load_chunk.call_count == 1
            assert mock_load_chunk.call_args_list[0][0] == (
                0,
                0,
            )  # dataset_idx=0, local_dataset_idx=0
