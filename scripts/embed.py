import os
import time
from src.dataset import set_dimension_chunks
import torch
import logging
import argparse
import xarray as xr
import numpy as np
import glob
from tqdm import tqdm
from argparse import Namespace as Args
from transformers import AutoModel, AutoConfig
from xarray import Dataset as XarrayDataset
from src.modeling import process_group

logger = logging.getLogger(__name__)

# Maintain a constant hidden state chunk size while allowing sample chunk size to vary,
# or it is common to hit buffer limits with the default Blosc codec in Zarr, e.g.:
# ValueError: Codec does not support buffers of > 2147483647 bytes
# See: https://github.com/zarr-developers/zarr-python/issues/487
DEFAULT_HIDDEN_STATE_CHUNK_SIZE = 1024

# -------------------------------------------------------------------------------------------------
# Generate embeddings
# -------------------------------------------------------------------------------------------------

def load_base_model(args: Args) -> AutoModel:
    logger.info(f"Loading base model from {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(
            args.model_path, config=config, trust_remote_code=True, dtype=torch.bfloat16
    )
    return base_model

def set_batch_chunks(batch: xr.Dataset, sample_chunk_size: int, hidden_state_chunk_size: int=DEFAULT_HIDDEN_STATE_CHUNK_SIZE) -> xr.Dataset:
    for var, da in batch.variables.items():
        chunks = []
        for d in da.dims:
            if d == "sample":
                chunks.append(sample_chunk_size)
            elif d == "hidden_state":
                chunks.append(hidden_state_chunk_size)
            else:
                chunks.append(da.sizes[d])
        batch[var].encoding.update({"chunks": tuple(chunks)})
    return batch

@torch.inference_mode()
def generate_embeddings(args: Args, dataset: XarrayDataset, base_model: AutoModel, split: str) -> XarrayDataset:
    logger.info(f"Generating embeddings for {split} dataset:\n{dataset}")

    rank, world_size = process_group()
    dataset_path = os.path.join(args.output_dir, f"{split}.{rank}.zarr")
    if os.path.exists(dataset_path):
        logger.info(f"Loading existing embeddings from {dataset_path}")
        return xr.open_zarr(dataset_path)

    batch_size = args.batch_size
    base_model = base_model.eval().to("cuda")
    
    logger.info(f"Generating embeddings with {batch_size=}, {dataset_path=}, {world_size=}, {rank=}")
    batch_ids = np.arange(dataset.sizes["sample"]) // batch_size
    n_batches = len(np.unique(batch_ids))
    batches = (
        dataset
        .assign_coords(batch_id=("sample", batch_ids))
        .pipe(lambda ds: 
            ds if world_size <= 1 else 
            ds.isel(sample=(ds.batch_id.values % world_size == rank))
        )
        .groupby("batch_id")
    )
    for batch_id, batch in batches:
        logger.info(f"Processing batch {batch_id} of {n_batches} ({rank=})")
        input_ids = torch.tensor(batch["input_ids"].values, device=base_model.device)
        outputs = base_model(input_ids=input_ids)
        embeddings = outputs.last_hidden_state.cpu().numpy()
        assert embeddings.ndim == 3, f"Expected 3D array, got {embeddings.shape=}"
        batch = batch.assign(embeddings=(
            ("sample", "sequence", "hidden_state"),
            embeddings
        ))
        batch = set_batch_chunks(batch, args.chunk_size)
        batch.to_zarr(
            dataset_path,
            zarr_format=2,
            **(dict(append_dim="sample") if os.path.exists(dataset_path) else {}),
        )
    logger.info(f"Embedding complete for {dataset_path}")
    return xr.open_zarr(dataset_path)

def generate_all_embeddings(args: Args) -> None:
    """Generate embeddings for train and validation datasets."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the already converted datasets
    logger.info("Loading train dataset...")
    train_dataset = xr.open_zarr(os.path.join(args.input_dir, "train.zarr"))
    
    # Shuffle training dataset if requested
    if args.shuffle_training_dataset:
        logger.info("Shuffling training dataset...")
        n_samples = train_dataset.sizes["sample"]
        rs = np.random.RandomState(args.seed)
        shuffled_indices = rs.permutation(n_samples)
        assert len(shuffled_indices) == n_samples
        train_dataset = train_dataset.isel(sample=shuffled_indices)
        assert train_dataset.sizes["sample"] == n_samples
    
    logger.info("Loading validation dataset...")
    valid_dataset = xr.open_zarr(os.path.join(args.input_dir, "valid.zarr"))
    
    # Load base model
    base_model = load_base_model(args)
    
    # Generate and add embeddings
    logger.info("Adding embeddings to train dataset...")
    train_dataset = generate_embeddings(args, train_dataset, base_model, "train")
    logger.info("Adding embeddings to validation dataset...")
    valid_dataset = generate_embeddings(args, valid_dataset, base_model, "valid")
    
    logger.info(f"Final training dataset with embeddings:\n{train_dataset}")
    logger.info(f"Final validation dataset with embeddings:\n{valid_dataset}")
    logger.info("Embedding generation complete")

# -------------------------------------------------------------------------------------------------
# Consolidate embeddings
# -------------------------------------------------------------------------------------------------

def consolidate_embedding_datasets(args: Args, split: str) -> None:
    """Consolidate embedding datasets from multiple ranks."""
    logger.info(f"Consolidating {split} embedding datasets...")
    
    consolidated_path = os.path.join(args.output_dir, f"{split}.zarr")
    
    if os.path.exists(consolidated_path):
        logger.info(f"Consolidated dataset already exists at {consolidated_path}")
        return
    
    rank_paths = glob.glob(os.path.join(args.input_dir, f"{split}.*.zarr"))
    logger.info(f"Found {len(rank_paths)} rank files to consolidate for {split=}")
    
    for rank_path in tqdm(rank_paths, desc=f"Consolidating {split}"):
        ds = xr.open_zarr(rank_path)
        n_samples = ds.sizes["sample"]
        
        # Load and save chunks separately, each of which will need to fit into memory
        batch_indices = list(range(0, n_samples, args.batch_size))
        for i in tqdm(batch_indices, desc=f"Consolidating {os.path.basename(rank_path)}", leave=False):
            chunk = ds.isel(sample=slice(i, i + args.batch_size))
            exists = os.path.exists(consolidated_path)
            chunk.to_zarr(
                consolidated_path,
                zarr_format=2,
                **(dict(append_dim="sample") if exists else {}),
                consolidated=True
            )
    
    logger.info(f"Consolidated {split} dataset saved to {consolidated_path}")

def consolidate_all_embeddings(args: Args) -> None:
    """Consolidate all embedding datasets."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Consolidate train and validation datasets
    consolidate_embedding_datasets(args, "train")
    consolidate_embedding_datasets(args, "valid")
    
    logger.info("Embedding datasets consolidation complete")

# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate and consolidate embeddings")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")
    
    # Generate embeddings command
    generate_parser = subparsers.add_parser("generate", help="Generate embeddings from sequence data")
    generate_parser.add_argument('--input-dir', type=str, required=True, 
                        help='Directory containing the sequence datasets produced by consolidate_data.py')
    generate_parser.add_argument('--output-dir', type=str, required=True, 
                        help='Directory for saving output embeddings')
    generate_parser.add_argument('--model-path', type=str, required=True, 
                        help='Path to the pre-trained model')
    generate_parser.add_argument('--batch-size', type=int, default=32, 
                        help='Batch size for embedding generation; a batch size of 32 uses ~18GB of GPU memory with 8192bp sequences')
    generate_parser.add_argument('--chunk-size', type=int, default=32, 
                        help='Chunk size for saved embeddings dataset (in samples dimension)')
    generate_parser.add_argument('--shuffle-training-dataset', type=bool, default=True,
                        help='Whether to shuffle the training dataset before generating embeddings')
    generate_parser.add_argument('--seed', type=int, default=42,
                        help='Seed for training dataset shuffling')
    
    # Consolidate embeddings command
    consolidate_parser = subparsers.add_parser("consolidate", help="Consolidate embedding datasets from multiple ranks")
    consolidate_parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing the embedding datasets to be consolidated')
    consolidate_parser.add_argument('--output-dir', type=str, required=True, 
                        help='Directory where consolidated embeddings will be saved')
    consolidate_parser.add_argument('--batch-size', type=int, default=128, 
                        help='Batch size for consolidating embeddings')
    
    args = parser.parse_args()
    
    if args.command == "generate":
        generate_all_embeddings(args)
    elif args.command == "consolidate":
        consolidate_all_embeddings(args)
    else:
        parser.print_help()

    logger.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")