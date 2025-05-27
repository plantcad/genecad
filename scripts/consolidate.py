import os
import logging
import argparse
import time
import gzip
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from Bio import SeqIO
import lightning as L
from numba import njit
from torch.utils import data
from argparse import Namespace as Args
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from xarray import Dataset as XarrayDataset
from datasets import Dataset as HfDataset
from torch.utils.data import Dataset as TorchDataset
from src.dataset import MultisampleSequenceDatasetLabeled, set_dimension_chunks
from src.config import WINDOW_SIZE, get_species_config
logger = logging.getLogger(__name__)

def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Prepare and consolidate sequence data")
    parser.add_argument('--keyfile', type=str, required=True, help='keyfile: tab-separated file with the following headers: name, labels, fasta, windows')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory for saving output datasets')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the pre-trained model for tokenizer')
    parser.add_argument('--val-proportion', type=float, default=0.1, help='percentage of examples to use as the validation set')
    parser.add_argument('--window-size', type=int, default=WINDOW_SIZE, help='Size of the window for processing sequences')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for creating sharded sequence dataset')
    parser.add_argument('--chunk-size', type=int, default=32, help='Chunk size for saved embeddings dataset (in samples dimension)')
    parser.add_argument('--num-workers', type=int, default=16, help='Number of worker threads for data loading')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of samples to process (for testing)')
    parser.add_argument('--shuffle', type=str, default='yes', choices=['yes', 'no'], help='Whether to shuffle the data during loading')
    parser.add_argument('--remove-overlaps', type=str, default='no', choices=['yes', 'no'], help='Whether to remove overlapping windows')
    return parser.parse_args()

def worker_load_files(worker_id):
    """ Open a fresh file handle for each DataLoader worker """
    worker_info = data.get_worker_info()
    dataset = worker_info.dataset

    dataset.open_label_files()

def load_data_from_keyfile(args: Args) -> tuple[MultisampleSequenceDatasetLabeled, MultisampleSequenceDatasetLabeled]:
    logger.info(f"Loading data from keyfile {args.keyfile}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Parse keyfile
    keyfile_df = pd.read_csv(args.keyfile, sep="\t", header=0)
    num_species = keyfile_df.shape[0]
    
    species_ids = [None] * num_species
    species_fastas = [None] * num_species
    species_labels = [None] * num_species
    species_contigs = [None] * num_species
    windows = None
    
    for idx, row in keyfile_df.iterrows():
        species_id = row["name"]
        species_config = get_species_config(species_id)
        species_ids[idx] = species_id
        logger.info(f"[species_id={species_id}] Loading labels, fasta, and windows files...")

        labels_file = row["labels"]
        fasta_file = row["fasta"]
        windows_file = row["windows"]

        # ------------------------------------------------------------------------------------------------
        # Load windows
        # ------------------------------------------------------------------------------------------------
        windows_array = np.load(windows_file)
        # Columns should be: contig_id, start, strand
        assert windows_array.shape[1] == 3, f"Expected 3 columns in windows, got {windows_array.shape[1]} columns"

        # Append a new column to the 2D array as the first column
        # with a value equal to the species index
        windows_array = np.insert(windows_array, 0, idx, 1)
        assert windows_array.shape[1] == 4, f"Expected 4 columns in windows, got {windows_array.shape[1]} columns"

        if idx == 0:
            windows = windows_array
        else:
            windows = np.vstack((windows, windows_array))
        logger.info(f"[species_id={species_id}] Loaded {len(windows_array)} windows")

        windows_df = windows_to_dataframe(windows_array)

        # Define the primary chromosome id set and order that all other data must match
        chrom_id_list = species_config.sort_chromosome_ids_by_number([
            chrom_id # return normalized id like chr1, chr2, etc.
            for chrom_id in species_config.chromosome_map.values()
            if species_config.get_chromosome_number(chrom_id) is not None
        ])
        # Filter to only include chromosomes that are present in the windows
        chrom_id_list = [
            chrom_id
            for chrom_index, chrom_id in enumerate(chrom_id_list)
            if chrom_index in set(windows_df["chrom_index"])
        ]
        windows_df["chrom_id"] = windows_df["chrom_index"].map({i: v for i, v in enumerate(chrom_id_list)})
        logger.info(f"[species_id={species_id}] Found {len(chrom_id_list)} numeric chromosomes in windows: {chrom_id_list}")

        # ------------------------------------------------------------------------------------------------
        # Load labels
        # ------------------------------------------------------------------------------------------------
        with open(labels_file, "rb") as file:
            labels_npz = np.load(file)
            labels_length = {
                chrom_id: len(labels_npz[chrom_id])
                for chrom_id in labels_npz.keys()
            }
            labels_npz.close()
        logger.info(f"[species_id={species_id}] Found {len(labels_length)} chromosomes in labels: {list(labels_length.keys())}")

        # ------------------------------------------------------------------------------------------------
        # Load sequences
        # ------------------------------------------------------------------------------------------------
        fasta_records = {}
        fasta_lengths = {}
        open_func = gzip.open if fasta_file.endswith(".gz") else open
        mode = "rt" if fasta_file.endswith(".gz") else "r"
        with open_func(fasta_file, mode) as file:
            for record in SeqIO.parse(file, "fasta"):
                if record.id not in species_config.chromosome_map:
                    continue
                chrom_id = species_config.chromosome_map[record.id]
                if chrom_id not in chrom_id_list:
                    continue
                fasta_lengths[chrom_id] = len(record.seq)
                chrom_index = chrom_id_list.index(chrom_id)
                fasta_records[chrom_index] = record
        logger.info(f"[species_id={species_id}] Found {len(fasta_lengths)} chromosomes in fasta sequences: {list(fasta_lengths.keys())}")

        # ------------------------------------------------------------------------------------------------
        # Validate windows
        # ------------------------------------------------------------------------------------------------
        # Check for chromosomes in windows with no sequences
        if (missing_fasta_chrom_ids := set(chrom_id_list) - set(fasta_lengths.keys())):
            raise ValueError(f"Chromosomes {missing_fasta_chrom_ids} not found in fasta file for species {species_id}")
        # Check for chromosomes in windows with no labels
        if (missing_labels_chrom_ids := set(chrom_id_list) - set(labels_length.keys())):
            raise ValueError(f"Chromosomes {missing_labels_chrom_ids} not found in labels file for species {species_id}")

        # Validate that the labels and windows do not extend beyond the sequence length
        windows_summary = (
            windows_df
            .groupby(["chrom_index", "chrom_id"])["start"].max()
            .rename("max_window_start")
            .reset_index()
            .assign(sequence_length=lambda df: df["chrom_id"].map(fasta_lengths))
            .assign(labels_length=lambda df: df["chrom_id"].map(labels_length))
            .assign(window_margin=lambda df: df["sequence_length"] - df["max_window_start"])
            .assign(labels_margin=lambda df: df["sequence_length"] - df["labels_length"])
        )
        if windows_summary["sequence_length"].isnull().any():
            raise ValueError(f"Unable to find sequences for all chromosomes for species {species_id}:\n{windows_summary}")
        if windows_summary["labels_length"].isnull().any():
            raise ValueError(f"Unable to find labels for all chromosomes for species {species_id}:\n{windows_summary}")
        if (windows_summary["window_margin"].fillna(-1) < 0).any():
            raise ValueError(f"Found window start positions that extend beyond sequence length for species {species_id}:\n{windows_summary}")
        if (windows_summary["labels_margin"].fillna(-1) < 0).any():
            raise ValueError(f"Found labels that extend beyond sequence length for species {species_id}:\n{windows_summary}")
        with pd.option_context("display.max_rows", None):
            logger.info(f"[species_id={species_id}] Windows summary:\n{windows_summary}")

        species_labels[idx] = labels_file
        species_fastas[idx] = fasta_records
        species_contigs[idx] = chrom_id_list
    
    # If max_samples is set, limit the number of windows
    if args.max_samples is not None:
        if args.max_samples < windows.shape[0]:
            windows = windows[:args.max_samples]
            logger.info(f"Limiting to {args.max_samples} samples for testing")
    
    # Convert to DataFrame for filtering
    windows_df = windows_to_dataframe(windows)
    
    # Filter windows based on window size constraints
    windows_df = filter_windows_by_size(windows_df, species_fastas, args.window_size)
    
    # Filter windows based on overlap constraints
    if args.remove_overlaps == 'yes':
        windows_df = filter_windows_by_overlap(windows_df, args.window_size)
    
    # Convert back to numpy array for train/val split
    windows = windows_df[["species_index", "chrom_index", "start", "strand"]].values
    
    # Split into train and validation sets
    num_val = int(windows.shape[0] * args.val_proportion)
    rs = np.random.RandomState(args.seed)
    rs.shuffle(windows)

    windows_val = windows[0:num_val]
    windows_train = windows[num_val:]
    
    logger.info(f"Train samples: {len(windows_train)}, Validation samples: {len(windows_val)}")
    
    # Create datasets
    train_dataset = MultisampleSequenceDatasetLabeled(
        sequences=species_fastas,
        label_files=species_labels,
        windows=windows_train,
        species=species_ids,
        contig_indices=species_contigs,
        tokenizer=tokenizer,
        window_size=args.window_size,
    )
    
    valid_dataset = MultisampleSequenceDatasetLabeled(
        sequences=species_fastas,
        label_files=species_labels,
        windows=windows_val,
        species=species_ids,
        contig_indices=species_contigs,
        tokenizer=tokenizer,
        window_size=args.window_size,
    )
    
    return train_dataset, valid_dataset

def convert_dataset(args: Args, dataset: TorchDataset, split: str) -> XarrayDataset:
    logger.info(f"Converting {split} dataset to Xarray Dataset")
    if args.num_workers == 0:
        dataset.open_label_files()

    dataset_path = os.path.join(args.output_dir, f"{split}.zarr")
    if os.path.exists(dataset_path):
        logger.info(f"Loading existing dataset from {dataset_path}")
        return xr.open_zarr(dataset_path)
    
    # Create a dataloader for batch iteration
    logger.info(f"Creating dataloader with shuffle={args.shuffle == 'yes'}, num_workers={args.num_workers}")
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,  # Process one at a time for conversion
        num_workers=args.num_workers, 
        worker_init_fn=worker_load_files if args.num_workers > 0 else None,
        shuffle=args.shuffle == 'yes'
    )

    # Set seed before batch loop
    L.seed_everything(args.seed)

    # Convert batches to Xarray
    logger.info(f"Beginning conversion with batch_size={args.batch_size}, {dataset_path=}")
    for i, batch in enumerate(tqdm(dataloader, desc="Converting dataset")):
        # Convert lists to numpy for variables that are not already numpy arrays
        batch_arrays = HfDataset.from_dict(batch).with_format("numpy")
        data_vars = {}
        for k in batch_arrays.column_names:
            v = batch_arrays[k]
            assert v.ndim <= 2, f"Expected 1D or 2D array, got {v.ndim}D array for {k}"
            data_vars[k] = (["sample", "sequence"][:v.ndim], v)
        ds = xr.Dataset(data_vars=data_vars)
        fully_masked = (~ds["label_mask"]).all(dim="sequence").values
        assert fully_masked.shape == (ds.sizes["sample"],)
        if np.all(fully_masked):
            logger.warning("All samples have fully masked labels, skipping this batch")
            continue
        ds = ds.sel(sample=~fully_masked)
        ds = set_dimension_chunks(ds, "sample", args.chunk_size)
        ds.to_zarr(
            dataset_path, 
            zarr_format=2, 
            **(dict(append_dim="sample") if os.path.exists(dataset_path) else {})
        )
    return xr.open_zarr(dataset_path)

def log_filtering_summary(windows_df: pd.DataFrame, filtered_df: pd.DataFrame, filter_type: str) -> None:
    """Log detailed filtering summary by species and chromosome.
    
    Args:
        windows_df: Original DataFrame before filtering
        filtered_df: DataFrame after filtering
        filter_type: Description of the filter type for logging
    """
    total_windows = len(windows_df)
    total_removed = total_windows - len(filtered_df)
    
    logger.info(f"{filter_type}: {total_removed}/{total_windows} windows removed ({100 * total_removed / total_windows:.1f}%)")
    
    if total_removed == 0:
        return
    
    # Count windows by species/chromosome in original and filtered DataFrames
    original_counts = windows_df.groupby(["species_index", "chrom_index"]).size()
    filtered_counts = filtered_df.groupby(["species_index", "chrom_index"]).size() if len(filtered_df) > 0 else pd.Series(dtype=int)
    
    # Create summary statistics
    window_stats = pd.DataFrame({
        "total_windows": original_counts,
        "kept_windows": filtered_counts
    }).fillna(0).astype(int)
    
    window_stats["removed_windows"] = window_stats["total_windows"] - window_stats["kept_windows"]
    window_stats["removal_rate"] = (window_stats["removed_windows"] / window_stats["total_windows"] * 100).round(1)
    
    logger.info(f"{filter_type} by location:")
    for (species_idx, chrom_idx), stats in window_stats.iterrows():
        removed = int(stats["removed_windows"])
        total = int(stats["total_windows"])
        rate = stats["removal_rate"]
        logger.info(f"  Species {species_idx}, Chromosome {chrom_idx}: {removed}/{total} removed ({rate}%)")

def windows_to_dataframe(windows: np.ndarray) -> pd.DataFrame:
    """Convert windows array to DataFrame with proper column names."""
    return pd.DataFrame(windows, columns=["species_index", "chrom_index", "start", "strand"])

def filter_windows_by_size(windows_df: pd.DataFrame, species_fastas: list, window_size: int) -> pd.DataFrame:
    """Filter windows to ensure they fit within sequence bounds.
    
    Args:
        windows_df: DataFrame with columns [species_index, chrom_index, start, strand]
        species_fastas: List of FASTA records per species
        window_size: Size of sequence window
    """
    # Build sequence length lookup: (species_idx, chrom_idx) -> length
    seq_lengths = {
        (species_idx, chrom_idx): len(record.seq)
        for species_idx, fasta_records in enumerate(species_fastas)
        for chrom_idx, record in fasta_records.items()
    }
    
    # Add sequence length and calculate window end position
    windows_df = windows_df.copy()
    windows_df["sequence_length"] = windows_df.apply(
        lambda row: seq_lengths[(int(row["species_index"]), int(row["chrom_index"]))], 
        axis=1
    )
    
    # Validate that all sequence lengths were found
    if windows_df["sequence_length"].isnull().any():
        missing_keys = windows_df[windows_df["sequence_length"].isnull()][["species_index", "chrom_index"]].drop_duplicates()
        raise ValueError(f"Missing sequence lengths for species/chromosome combinations: {missing_keys.values.tolist()}")
    
    windows_df["window_end"] = windows_df["start"] + window_size
    
    # Filter windows that fit within sequence bounds
    valid_mask = windows_df["window_end"] <= windows_df["sequence_length"]
    windows_filtered = windows_df[valid_mask].copy()
    
    # Log filtering summary
    log_filtering_summary(windows_df, windows_filtered, "Window size filtering")
    
    return windows_filtered.drop(columns=["sequence_length", "window_end"])

@njit
def _filter_windows_by_overlap(windows: np.ndarray, window_size: int) -> np.ndarray:
    """JIT-compiled core logic for overlap filtering within a group."""
    if len(windows) <= 1:
        return np.ones(len(windows), dtype=np.bool_)
    
    valid_mask = np.ones(len(windows), dtype=np.bool_)
    
    # Sort by start position
    sort_order = np.argsort(windows[:, 2])  # Sort by start column
    
    # Find overlapping windows
    for i in range(len(sort_order)):
        if not valid_mask[sort_order[i]]:
            continue
            
        current_start = windows[sort_order[i], 2]
        current_end = current_start + window_size
        
        # Check subsequent windows for overlap
        for j in range(i + 1, len(sort_order)):
            if not valid_mask[sort_order[j]]:
                continue
                
            next_start = windows[sort_order[j], 2]
            if next_start < current_end:
                # Overlap detected - remove the later window
                valid_mask[sort_order[j]] = False
            else:
                # No more overlaps possible (sorted by start)
                break
    
    return valid_mask

def filter_windows_by_overlap(windows_df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Filter windows to remove overlaps within each species/chromosome/strand group.
    
    Args:
        windows_df: DataFrame with columns [species_index, chrom_index, start, strand]
        window_size: Size of sequence window
    """
    # Validate strand values are 0 or 1
    strands = windows_df["strand"].values
    if not np.all(np.isin(strands, [0, 1])):
        raise ValueError(f"Strand column must contain only 0 or 1, found: {np.unique(strands)}")
    
    # Process each group (species, chromosome, strand) separately
    filtered_groups = []
    total_removed = 0
    
    for (species_idx, chrom_idx, strand), group in windows_df.groupby(["species_index", "chrom_index", "strand"]):
        if len(group) == 0:
            continue
            
        # Convert group to numpy array for JIT function
        group_array = group[["species_index", "chrom_index", "start", "strand"]].values
        
        # Use JIT-compiled function for core logic
        valid_mask = _filter_windows_by_overlap(group_array, window_size)
        
        # Apply mask to group
        filtered_group = group[valid_mask].copy()
        filtered_groups.append(filtered_group)
        
        removed_count = len(group) - len(filtered_group)
        total_removed += removed_count
    
    # Combine all filtered groups
    if filtered_groups:
        windows_filtered = pd.concat(filtered_groups, ignore_index=True)
    else:
        windows_filtered = windows_df.iloc[:0].copy()  # Empty DataFrame with same structure
    
    # Log filtering summary
    log_filtering_summary(windows_df, windows_filtered, "Overlap filtering")
    
    return windows_filtered

def main() -> None:
    args = parse_args()
    
    # Load data
    train_dataset, valid_dataset = load_data_from_keyfile(args)
    
    # Convert to Xarray datasets
    logger.info("Converting train dataset...")
    train_dataset = convert_dataset(args, train_dataset, "train")
    logger.info("Converting validation dataset...")
    valid_dataset = convert_dataset(args, valid_dataset, "valid")
    
    logger.info(f"Final training dataset:\n{train_dataset}")
    logger.info(f"Final validation dataset:\n{valid_dataset}")
    logger.info("Data consolidation complete!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
