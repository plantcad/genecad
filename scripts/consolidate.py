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
from torch.utils import data
from argparse import Namespace as Args
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from xarray import Dataset as XarrayDataset
from datasets import Dataset as HfDataset
from torch.utils.data import Dataset as TorchDataset
from src.dataset import MultisampleSequenceDatasetLabeled, set_dimension_chunks
from src.config import WINDOW_SIZE
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
    
    species_names = [None] * num_species
    species_fastas = [None] * num_species
    species_labels = [None] * num_species
    species_contigs = [None] * num_species
    windows = None
    
    for idx, row in keyfile_df.iterrows():
        species_names[idx] = row["name"]

        logger.info("Loading " + species_names[idx] + "...")

        labels_file = row["labels"]
        fasta_file = row["fasta"]
        windows_file = row["windows"]

        with open(windows_file, "rb") as file:
            windows_df = np.load(file)

        windows_df = np.insert(windows_df, 0, idx, 1)

        if idx == 0:
            windows = windows_df
        else:
            windows = np.vstack((windows, windows_df))

        labels_npz = np.load(labels_file)

        chrom_list = list(labels_npz.keys())  # THIS is the official index-to-contig mapping!

        labels_npz.close()

        fasta_records = {}
        if fasta_file.endswith(".gz"):
            with gzip.open(fasta_file, "rt") as file:
                for record in SeqIO.parse(file, "fasta"):
                    if record.id in chrom_list:
                        fasta_records[chrom_list.index(record.id)] = record
        else:
            with open(fasta_file, "r") as file:
                for record in SeqIO.parse(file, "fasta"):
                    if record.id in chrom_list:
                        fasta_records[chrom_list.index(record.id)] = record


        species_labels[idx] = labels_file
        species_fastas[idx] = fasta_records
        species_contigs[idx] = chrom_list
    
    # If max_samples is set, limit the number of windows
    if args.max_samples is not None:
        if args.max_samples < windows.shape[0]:
            windows = windows[:args.max_samples]
            logger.info(f"Limiting to {args.max_samples} samples for testing")
    
    # Split into train and validation sets
    num_val = int(windows.shape[0] * args.val_proportion)
    np.random.shuffle(windows)
    
    windows_val = windows[0:num_val]
    windows_train = windows[num_val:]
    
    logger.info(f"Train samples: {len(windows_train)}, Validation samples: {len(windows_val)}")
    
    # Create datasets
    train_dataset = MultisampleSequenceDatasetLabeled(
        sequences=species_fastas,
        label_files=species_labels,
        windows=windows_train,
        species=species_names,
        contig_indices=species_contigs,
        tokenizer=tokenizer,
        max_length=args.window_size,
    )
    
    valid_dataset = MultisampleSequenceDatasetLabeled(
        sequences=species_fastas,
        label_files=species_labels,
        windows=windows_val,
        species=species_names,
        contig_indices=species_contigs,
        tokenizer=tokenizer,
        max_length=args.window_size,
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
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,  # Process one at a time for conversion
        num_workers=args.num_workers, 
        worker_init_fn=worker_load_files if args.num_workers > 0 else None
    )

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

def main() -> None:
    args = parse_args()

    # Set random seed
    L.seed_everything(args.seed)
    
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
