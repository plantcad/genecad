"""
Module containing subclasses of torch Dataset for use with training and inference

Most subclasses require fasta sequence and a tokenizer as input, though some take pre-tokenized data
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import math
import xarray as xr
import zarr
import glob
from typing import Any, Callable
import numpy.typing as npt
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
    # Get all groups recursively using zarr API
    store = zarr.open(zarr_path, mode='r')
    
    # Build nested dictionary of datasets
    datasets_dict = {}
    
    def _get_datasets(group, current_path=""):
        # Check if this group has array keys (making it a dataset)
        if len(list(group.array_keys())) > 0:
            # This is a dataset, open it with xarray
            if current_path == "":
                # Root group
                datasets_dict["/"] = xr.open_zarr(zarr_path, **kwargs)
            else:
                datasets_dict[current_path] = xr.open_zarr(
                    zarr_path, group=current_path, **kwargs
                )
                
        # Process subgroups recursively
        for subgroup_name in group.group_keys():
            subgroup = group[subgroup_name]
            new_path = f"{current_path}/{subgroup_name}" if current_path else subgroup_name
            _get_datasets(subgroup, new_path)
    
    # Start recursive discovery from the root
    _get_datasets(store)
    
    # Convert dictionary to DataTree
    return xr.DataTree.from_dict(datasets_dict)


def default_transform(ds: xr.Dataset) -> dict[str, Any]:
    sample_dict = {k: ds[k].values for k in ds.data_vars}
    return sample_dict

class XarrayDataset(Dataset):
    """Dataset for loading data from xarray"""

    def __init__(self, 
        path: str, 
        sample_transform: Callable[[xr.Dataset], dict[str, Any]] = None, 
        max_sample_count: int | None = None, 
        max_sequence_length: int | None = None, 
        chunk_size: int = 1,
        **backend_kwargs: Any
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
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self)}")
        dataset_idx, local_dataset_idx = self._locate_sample(idx)
        sample = self._get_sample(dataset_idx, local_dataset_idx)
        return self.sample_transform(sample)
    
    def _load_datasets(self, max_sample_count: int | None, max_sequence_length: int | None, **backend_kwargs: Any) -> None:
        """Load datasets from paths with constraints on sample count and sequence length"""
        self.datasets = []
        self.sample_counts = []
        total_samples = 0
        
        for p in self.paths:
            ds = xr.open_zarr(p, **backend_kwargs)
            if max_sequence_length is not None and ds.sizes["sequence"] > max_sequence_length:
                ds = ds.isel(sequence=slice(max_sequence_length))
            
            sample_count = ds.sizes["sample"]
            if max_sample_count is not None and total_samples + sample_count > max_sample_count:
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
        dataset_idx = np.searchsorted(self.cumulative_samples, idx, side='right')
        local_dataset_idx = idx - (0 if dataset_idx == 0 else self.cumulative_samples[dataset_idx-1])
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
        self.current_chunk = self.datasets[dataset_idx].isel(sample=slice(start, end)).compute()
        self.current_dataset_idx, self.current_chunk_id = dataset_idx, local_dataset_idx // self.chunk_size



class SequenceDatasetLabeled(Dataset):
    """ DataSet class for a single genome, with labels """

    def __init__(self, sequences, labels, windows, tokenizer, max_length):
        self.sequences = sequences  # list of fasta sequences
        self.labels = labels  # list of labels for the fasta sequences
        self.windows = windows  # numpy array of training windows, with three columns: chrom index, position, strand
        self.tokenizer = tokenizer  # model tokenizer
        self.max_length = max_length  # standard length of context windows

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, idx):
        window_chrom = self.windows[idx, 0]
        window_pos = self.windows[idx, 1]
        sequence = self.sequences[window_chrom].seq[window_pos:window_pos + self.max_length]

        # Get reverse complement if the window strand is 1
        # for labels, [:, 1] are the labels for the reverse strand
        if (self.windows[idx, 2] == 1):
            sequence = sequence.reverse_complement()
            label = self.labels[window_chrom][window_pos:window_pos + self.max_length, 1][::-1].copy()
        else:
            label = self.labels[window_chrom][window_pos:window_pos + self.max_length, 0].copy()

        # Soft-masking from fasta sequence defines un-usable labels
        # Also remove any position where label == -1 (ambiguous)
        valid_labels = np.array([(char.isupper() and label[idx2] >= 0) for idx2, char in enumerate(sequence)])

        encoding = self.tokenizer.encode_plus(
            str(sequence),
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        label = torch.tensor(label, dtype=torch.long)
        mask = encoding['attention_mask'].squeeze(0)
        # For now, there is no known reason for special tokens 
        # to be included in sequences so fail on their presence
        assert (mask == 1).all()
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': mask,
            'labels': label,
            'chromosome': self.sequences[window_chrom].id,
            'position': window_pos,
            'use_indices': valid_labels
        }


class MultisampleSequenceDatasetLabeled(Dataset):
    """ DataSet class with for multiple genomes, with labels """

    def __init__(self, sequences, label_files, windows, species, contig_indices, tokenizer, max_length):
        self.species = species  # list of species names
        self.sequences = sequences  # list of lists of SeqRecords
        self.label_files = label_files  # list of file names for labels
        self.windows = windows  # array defining windows with columns: species index, chrom index, position, strand
        self.contig_indices = contig_indices  # list of lists of contig names
        self.tokenizer = tokenizer  # model tokenizer
        self.max_length = max_length  # standard length of context windows
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

        chrom_name = self.contig_indices[window_species][window_chrom]

        sequence = self.sequences[window_species][window_chrom][window_pos:window_pos + self.max_length].seq
        sequence_len = len(str(sequence))

        species_id = self.species[window_species]

        # Get the reverse complement if the strand is 1
        # for labels, [:, 1] are the labels for the reverse strand
        if (window_strand == 1):
            sequence = sequence.reverse_complement()
            label = self.labels[window_species][chrom_name][window_pos:window_pos + self.max_length, 1][::-1].copy()
        else:
            label = self.labels[window_species][chrom_name][window_pos:window_pos + self.max_length, 0].copy()
        assert label.ndim == 1

        # If label is less than length of sequence, pad with 0
        if len(label) < sequence_len:
            # This happens once at TOW:
            # WARNING:Label is less than max length for Chr2 at 19689472: 7349 < 8192
            logger.warning(f"Label is less than sequence length for {species_id}/{chrom_name} at {window_pos}: {len(label)} < {sequence_len}")
            label = np.pad(label, (0, sequence_len - len(label)), mode='constant', constant_values=-1)
            assert label.ndim == 1

        if len(sequence) != len(label):
            raise ValueError(f"Sequence and label lengths do not match for {chrom_name} at {window_pos}: {len(sequence)} != {len(label)}")

        # Mask out labels where label == -1 (ambiguous)
        label_mask = label >= 0
        # Soft-masking from fasta sequence defines un-usable labels
        soft_mask = np.array([char.isupper() for char in sequence])

        encoding = self.tokenizer.encode_plus(
            str(sequence),
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        label = torch.tensor(label, dtype=torch.long)

        return {
            'sample_index': idx,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'soft_mask': soft_mask,
            'label_mask': label_mask,
            'labels': label,
            'species': self.species[window_species],
            'chromosome': chrom_name,
            'position': window_pos,
            'strand': window_strand,
        }


class SequenceDatasetUnlabeled(Dataset):
    """ Sequence dataset without truth labels, for inference """

    def __init__(self, sequence, name, tokenizer, window_size):
        self.sequence = sequence  # one Seq object
        self.name = name  # name of contig or sequence
        self.tokenizer = tokenizer  # model tokenizer
        self.window_size = window_size  # maximum context length
        self.overlap = window_size // 2  # overlap for windows

    def __len__(self):
        return max(int(math.ceil(len(self.sequence) / self.overlap)) - 1, 1)

    def __getitem__(self, idx):
        # windows shift along the input sequence by half their length to mitigate edge effects
        end = idx * self.overlap + self.window_size
        if end > len(self.sequence):
            end = len(self.sequence)

        sequence = str(self.sequence[(idx * self.overlap):end])
        name = idx

        # Tokenize and pad to window_size
        encoding = self.tokenizer.encode_plus(
            sequence.ljust(self.window_size, 'N'),
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False
        )

        return {
            'sequence': sequence,
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'name': name
        }


class CRFDatasetLabeled(Dataset):
    """
    Dataset of labeled embeddings for CRF training and validation
    Allows for multiple genomes input
    Embeddings do not require a separate tokenization step
    """

    def __init__(self, predicted_labels, labels, windows, species, contig_indices, max_length):
        # TODO: embeddings are currently held in memory, so this won't scale past a few genomes
        self.species = species  # list of species
        self.labels = labels  # list of list of numpy arrays
        self.predicted_labels = predicted_labels  # list of arrays for predicted labels
        self.windows = windows  # array of training windows with columns: species idx, contig idx, position, strand
        self.contig_indices = contig_indices  # list of lists of contig names
        self.max_length = max_length  # context window length

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, idx):
        window_species = self.windows[idx, 0]
        window_chrom = self.windows[idx, 1]
        window_pos = self.windows[idx, 2]

        chrom_name = self.contig_indices[window_species][window_chrom]

        # Embeddings don't need to be reverse-complemented, but they do need to be reversed when strand == 1
        # for labels, [:, 1] are the labels for the reverse strand
        if (self.windows[idx, 3] == 1):
            label = self.labels[window_species][chrom_name][window_pos:window_pos + self.max_length, 1][::-1].copy()
            predicted_labels = self.predicted_labels[window_species][chrom_name][
                               window_pos:window_pos + self.max_length, :, 1][::-1].copy()
        else:
            predicted_labels = self.predicted_labels[window_species][chrom_name][
                               window_pos:window_pos + self.max_length, :, 0].copy()
            label = self.labels[window_species][chrom_name][window_pos:window_pos + self.max_length, 0].copy()

        label = torch.tensor(label, dtype=torch.int)
        predicted_labels = torch.tensor(predicted_labels)

        return {
            'input_ids': predicted_labels,
            'labels': label,
            'species': self.species[window_species],
            'chromosome': chrom_name,
            'position': window_pos,
        }


class CRFDatasetUnlabeled(Dataset):
    """
    Embeddings-only dataset for CRF inference without labels

    Again, embeddings do not need to be tokenized
    """

    def __init__(self, embedding, name, window_size):
        """
        embedding shape: [sequence_len, num_labels] or [sequence_len, num_labels, 2]
        name: a contig name or ID
        window_size: the window/chunk size used in the original inference
        """
        self.embedding = embedding
        self.name = name
        self.window_size = window_size
        self.overlap = window_size // 2

        seq_len = embedding.shape[0]
        # same logic as original inference: # of chunks
        self.num_chunks = max(int(math.ceil(seq_len / self.overlap)) - 1, 1)

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        seq_len = self.embedding.shape[0]
        chunk_start = idx * self.overlap
        chunk_end = min(chunk_start + self.window_size, seq_len)
        chunk_len = chunk_end - chunk_start

        # slice out the portion
        chunk_data = self.embedding[chunk_start:chunk_end]  # shape => [chunk_len, ...]

        # Zero-pad if chunk_len < window_size
        if chunk_len < self.window_size:
            shape_list = list(chunk_data.shape)
            shape_list[0] = self.window_size  # expand the first dim
            padded = np.zeros(shape_list, dtype=chunk_data.dtype)
            padded[:chunk_len] = chunk_data
            chunk_data = padded

        # convert to torch tensor
        chunk_data_t = torch.tensor(chunk_data, dtype=torch.float)

        return {
            'input_ids': chunk_data_t,  # shape => [window_size, num_labels] or [window_size, num_labels, 2]
            'name': torch.tensor(idx, dtype=torch.long),
            'chunk_len': torch.tensor(chunk_len, dtype=torch.long)  # how many real bases were used in this chunk
        }
