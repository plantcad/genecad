"""
Module containing subclasses of torch Dataset for use with training and inference

Most subclasses require fasta sequence and a tokenizer as input, though some take pre-tokenized data
"""

import io
import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
import zarr
import pandas as pd
import glob
from typing import Any, Callable
import logging

logger = logging.getLogger(__name__)


DEFAULT_SEQUENCE_CHUNK_SIZE = 10_000_000


def set_dimension_chunks(ds: xr.Dataset, dim: str, chunk_size: int) -> xr.Dataset:
    """Set chunking in one dimension of an Xarray Dataset

    This function will set the chunk size for a given dimension, while
    using the full size for all other dimensions.

    Parameters
    ----------
        ds (xarray.Dataset): The dataset to modify chunking for
        dim (str): Name of dimension to chunk
        chunk_size (int): Size of chunks for the specified dimension
    """
    # For each data variable in the dataset (including coordinates)
    for var_name, da in ds.variables.items():
        dims = da.dims
        if dim not in dims:
            continue
        # Create chunk sizes as full size for all dims except the one specified
        chunks = []
        for d in dims:
            if d == dim:
                chunks.append(chunk_size)
            else:
                chunks.append(da.sizes[d])
        ds[var_name].encoding.update({"chunks": tuple(chunks)})

    return ds


def list_species_contig_datatree(zarr_path: str) -> list[dict[str, str]]:
    """
    Convenience function to list all species/chromosome groups in a 2-level Zarr store.

    Parameters
    ----------
    zarr_path : str
        Path to the zarr store

    Returns
    -------
    list[dict[str, str]]
        List of dictionaries containing:
        - species_id: parsed species identifier
        - chrom_id: parsed chromosome identifier
        - group_path: zarr group path
        - dataset_path: full filesystem path to the dataset
    """
    # Get the raw group mappings at depth 2
    group_mappings = list_datatree(zarr_path, depth=2)

    # Parse each group path and create structured results
    results = []
    for group_path, dataset_path in group_mappings.items():
        # Parse species_id and chrom_id from group path
        parts = group_path.strip("/").split("/")
        if len(parts) != 2:
            # Skip invalid group paths (shouldn't happen at depth 2, but be defensive)
            continue
        species_id, chrom_id = parts

        results.append(
            {
                "species_id": species_id,
                "chrom_id": chrom_id,
                "group_path": group_path,
                "dataset_path": dataset_path,
            }
        )

    return results


def list_datatree(zarr_path: str, depth: int | None = None) -> dict[str, str]:
    """
    List all groups in a zarr store without loading the datasets.

    Parameters
    ----------
    zarr_path : str
        Path to the zarr store
    depth : int | None, optional
        Only include groups at this depth level (root=0, first level=1, etc.).
        If None, include all groups.

    Returns
    -------
    dict[str, str]
        Dictionary mapping group paths to filesystem paths
    """
    # Get all groups recursively using zarr API
    store = zarr.open(zarr_path, mode="r")

    # Build nested dictionary of dataset paths
    datasets_dict = {}

    def _get_dataset_paths(group, current_path="", current_depth=0):
        # Check if this group has array keys (making it a dataset)
        if len(list(group.array_keys())) > 0:
            # Only add to results if depth filter matches
            if depth is None or current_depth == depth:
                # This is a dataset, store the full path to this group
                if current_path == "":
                    # Root group
                    datasets_dict["/"] = zarr_path
                else:
                    datasets_dict[current_path] = f"{zarr_path}/{current_path}"

        # Process subgroups recursively
        for subgroup_name in group.group_keys():
            subgroup = group[subgroup_name]
            new_path = (
                f"{current_path}/{subgroup_name}" if current_path else subgroup_name
            )
            _get_dataset_paths(subgroup, new_path, current_depth + 1)

    # Start recursive discovery from the root
    _get_dataset_paths(store)

    return datasets_dict


def open_datatree(zarr_path: str, **kwargs: Any) -> xr.DataTree:
    """
    Open a zarr store as an xarray DataTree by recursively finding and loading all groups.

    This is a temporary solution for https://github.com/pydata/xarray/issues/9960, which should
    have been fixed by https://github.com/pydata/xarray/pull/10020 but has not been released yet.

    Parameters
    ----------
    zarr_path : str
        Path to the zarr store
    kwargs : Any
        Additional keyword arguments to pass to `xr.open_zarr`

    Returns
    -------
    xr.DataTree
        DataTree containing all datasets in the zarr store with matching hierarchy
    """
    # Get the structure without loading datasets
    dataset_paths = list_datatree(zarr_path)

    # Build nested dictionary of datasets by loading each one
    datasets_dict = {}
    for group_path, full_path in dataset_paths.items():
        if group_path == "/":
            # Root group - use the zarr path directly
            datasets_dict[group_path] = xr.open_zarr(full_path, **kwargs)
        else:
            # Sub-group - extract base path and group
            datasets_dict[group_path] = xr.open_zarr(
                zarr_path, group=group_path, **kwargs
            )

    return xr.DataTree.from_dict(datasets_dict)


def info_str(df: pd.DataFrame) -> str:
    """Get a string summary of pandas DataFrame.info()"""
    buf = io.StringIO()
    df.info(buf=buf)
    info_string = buf.getvalue()
    return info_string


def default_transform(ds: xr.Dataset) -> dict[str, Any]:
    sample_dict = {k: ds[k].values for k in ds.data_vars}
    return sample_dict


class XarrayDataset(Dataset):
    """Dataset for loading data from xarray"""

    def __init__(
        self,
        path: str,
        sample_transform: Callable[[xr.Dataset], dict[str, Any]] = None,
        max_sample_count: int | None = None,
        max_sequence_length: int | None = None,
        chunk_size: int = 1,
        **backend_kwargs: Any,
    ):
        self.paths = sorted(glob.glob(path))
        if not self.paths:
            raise ValueError(f"No datasets found at {path}")
        self.chunk_size = chunk_size
        self.current_chunk_id = -1
        self.current_chunk = None
        self.current_dataset_idx = -1
        self.sample_transform = sample_transform or default_transform
        self._load_datasets(max_sample_count, max_sequence_length, **backend_kwargs)

    def __len__(self):
        return self.cumulative_samples[-1] if self.cumulative_samples.size > 0 else 0

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {len(self)}"
            )
        dataset_idx, local_dataset_idx = self._locate_sample(idx)
        sample = self._get_sample(dataset_idx, local_dataset_idx)
        return self.sample_transform(sample)

    def _load_datasets(
        self,
        max_sample_count: int | None,
        max_sequence_length: int | None,
        **backend_kwargs: Any,
    ) -> None:
        """Load datasets from paths with constraints on sample count and sequence length"""
        self.datasets = []
        self.sample_counts = []
        total_samples = 0

        for p in self.paths:
            ds = xr.open_zarr(p, **backend_kwargs)
            if (
                max_sequence_length is not None
                and ds.sizes["sequence"] > max_sequence_length
            ):
                ds = ds.isel(sequence=slice(max_sequence_length))

            sample_count = ds.sizes["sample"]
            if (
                max_sample_count is not None
                and total_samples + sample_count > max_sample_count
            ):
                remaining = max_sample_count - total_samples
                if remaining > 0:
                    ds = ds.isel(sample=slice(remaining))
                    sample_count = remaining
                else:
                    break

            if sample_count > 0:
                self.datasets.append(ds)
                self.sample_counts.append(sample_count)
                total_samples += sample_count

            if max_sample_count is not None and total_samples >= max_sample_count:
                break

        self.cumulative_samples = np.cumsum(self.sample_counts)

    def _locate_sample(self, idx: int) -> tuple[int, int]:
        """Find which dataset and local index a global index corresponds to"""
        dataset_idx = np.searchsorted(self.cumulative_samples, idx, side="right")
        local_dataset_idx = idx - (
            0 if dataset_idx == 0 else self.cumulative_samples[dataset_idx - 1]
        )
        return dataset_idx, local_dataset_idx

    def _get_sample(self, dataset_idx: int, local_dataset_idx: int) -> xr.Dataset:
        """Get a sample from the appropriate chunk, loading it if necessary"""
        chunk_id = (dataset_idx, local_dataset_idx // self.chunk_size)
        if chunk_id != (self.current_dataset_idx, self.current_chunk_id):
            self._load_chunk(dataset_idx, local_dataset_idx)

        local_idx = local_dataset_idx % self.chunk_size
        return self.current_chunk.isel(sample=local_idx)

    def _load_chunk(self, dataset_idx: int, local_dataset_idx: int) -> None:
        """Load a chunk from the dataset at the specified index"""
        start = (local_dataset_idx // self.chunk_size) * self.chunk_size
        end = min(start + self.chunk_size, self.sample_counts[dataset_idx])
        self.current_chunk = (
            self.datasets[dataset_idx].isel(sample=slice(start, end)).compute()
        )
        self.current_dataset_idx, self.current_chunk_id = (
            dataset_idx,
            local_dataset_idx // self.chunk_size,
        )


class MultisampleSequenceDatasetLabeled(Dataset):
    """DataSet class with for multiple genomes, with labels"""

    def __init__(
        self,
        sequences,
        label_files,
        windows,
        species,
        contig_indices,
        tokenizer,
        window_size,
    ):
        self.species = species  # list of species names
        self.sequences = sequences  # list of lists of SeqRecords
        self.label_files = label_files  # list of file names for labels
        self.windows = windows  # array defining windows with columns: species index, chrom index, position, strand
        self.contig_indices = contig_indices  # list of lists of contig names
        self.tokenizer = tokenizer  # model tokenizer
        self.window_size = window_size  # standard length of context windows
        self.labels = None  # We will lazy-load these arrays

    # Each worker thread needs its own file handle for processing
    # Unfortunately I can't find a good way to close them programmatically
    # Though they should all be closed when the subprocess exits
    def open_label_files(self):
        self.labels = [np.load(file) for file in self.label_files]

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, idx):
        window_species = self.windows[idx, 0]
        window_chrom = self.windows[idx, 1]
        window_pos = self.windows[idx, 2]
        window_strand = self.windows[idx, 3]

        species_id = self.species[window_species]

        if window_chrom >= len(self.contig_indices[window_species]):
            raise ValueError(
                f"Contig index {window_chrom} out of range for species {species_id}"
            )
        chrom_name = self.contig_indices[window_species][window_chrom]

        if window_chrom >= len(self.sequences[window_species]):
            raise ValueError(
                f"Contig index {window_chrom} out of range for species {species_id}"
            )
        max_sequence_len = len(self.sequences[window_species][window_chrom].seq)
        sequence = self.sequences[window_species][window_chrom][
            window_pos : window_pos + self.window_size
        ].seq

        # Ensure the window does not extend beyond the end of the sequence
        if window_pos + self.window_size > max_sequence_len:
            raise ValueError(
                "Windows cannot extend beyond end of sequence; invalid window: "
                f"species={species_id}, chrom={chrom_name}, pos={window_pos}, "
                f"strand={window_strand}, window_size={self.window_size}, max_sequence_len={max_sequence_len}"
            )

        # Get the reverse complement if the strand is 1
        # for labels, [:, 1] are the labels for the reverse strand
        if window_strand == 1:
            sequence = sequence.reverse_complement()
            label = self.labels[window_species][chrom_name][
                window_pos : window_pos + self.window_size, 1
            ][::-1].copy()
        else:
            label = self.labels[window_species][chrom_name][
                window_pos : window_pos + self.window_size, 0
            ].copy()
        assert label.ndim == 1

        # If label is less than length of sequence, pad with 0.
        # This happens because the labels are based on GFFs that do not span the entire sequence,
        # and label vectors based on those annotations are often only as long as the maximum annotation position.
        # Therefore, it can safely be assumed that any positions requiring padding are intergenic (class = 0)
        if len(label) < (sequence_len := len(sequence)):
            label = np.pad(
                label,
                (0, sequence_len - len(label)),
                mode="constant",
                constant_values=0,
            )
            assert label.ndim == 1

        if len(sequence) != len(label):
            raise ValueError(
                f"Sequence and label lengths do not match for {chrom_name} at {window_pos}: {len(sequence)=} != {len(label)=}"
            )
        if len(sequence) != self.window_size:
            raise ValueError(
                f"Sequence length does not match window size for {chrom_name} at {window_pos}: {len(sequence)=} != {self.window_size=}"
            )

        # Mask out labels where label == -1 (ambiguous)
        label_mask = label >= 0
        # Soft-masking from fasta sequence defines un-usable labels
        soft_mask = np.array([char.isupper() for char in sequence])

        encoding = self.tokenizer.encode_plus(
            str(sequence),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.window_size,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        label = torch.tensor(label, dtype=torch.long)

        return {
            "sample_index": idx,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "soft_mask": soft_mask,
            "label_mask": label_mask,
            "labels": label,
            "species": self.species[window_species],
            "chromosome": chrom_name,
            "position": window_pos,
            "strand": window_strand,
        }
