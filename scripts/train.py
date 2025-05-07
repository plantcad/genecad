"""
Training script for LLM gene annotation classifier

This trainer is intended to take prepared datasets (created by prepare_model_lightning.py)
and fine-tune a model for a classification task.
"""
import argparse
import os
import time
import logging
import warnings
from src.dataset import XarrayDataset
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import set_float32_matmul_precision
import xarray as xr
from src.modeling import SequenceSpanClassifier, SequenceSpanClassifierConfig, ThroughputMonitor
import lightning as L
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from argparse import Namespace as Args

logger = logging.getLogger(__name__)

def parse_args() -> Args:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train genome annotation model")
    parser.add_argument('--train-dataset', type=str, required=True, help='Path to prepared training dataset (zarr)')
    parser.add_argument('--val-dataset', type=str, required=True, help='Path to prepared validation dataset (zarr)')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory for saving outputs')
    parser.add_argument('--learning-rate', type=float, default=8e-4, help='learning rate for the model')
    parser.add_argument('--learning-rate-decay', type=str, default="none", help='Whether to use learning rate decay; either "none" or "cosine"')
    parser.add_argument('--window-size', type=int, default=8192, help='Size of the window for processing sequences')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for data loading')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to resume training from')
    parser.add_argument('--checkpoint-type', type=str, default=None, help='Type of checkpoint to resume training from; one of "model", "trainer"')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')
    parser.add_argument('--log-frequency', type=int, default=5, help='Frequency of logging during training')
    parser.add_argument('--val-check-interval', type=float, default=1.0, 
                        help='how often within one training epoch to evaluate the validation set.')
    parser.add_argument('--train-eval-frequency', type=int, default=250, 
                        help='how often within one training epoch to evaluate the training set (in steps).')
    parser.add_argument('--limit-val-batches', type=float, default=1.0, 
                        help='how much of the validation dataset to check.')
    parser.add_argument('--limit-train-examples', type=int, default=None, 
                        help='how many training examples to use')
    parser.add_argument('--accumulate-grad-batches', type=int, default=16, 
                        help='batches to accumulate before optimizing')
    parser.add_argument('--num-workers', type=int, default=0, help='number of worker threads for data loading')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='batches for dataloader to prefetch')
    parser.add_argument('--project-name', default='test-project', help='wandb project name')
    parser.add_argument('--run-name', default='test-run', help='wandb run name')
    parser.add_argument('--gpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--num-nodes', type=int, default=1, help='number of nodes to use')
    parser.add_argument('--strategy', type=str, default="auto", help='training strategy')
    parser.add_argument('--checkpoint-frequency', type=int, default=1000, 
                        help="number of batches between saving checkpoints")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()

def sample_transform(ds: xr.Dataset) -> dict:
    """Custom sample transformation function for training data"""
    sample_dict = {}
    for k in ["input_ids", "label_mask", "labels", "embeddings", "sample_index"]:
        n = "inputs_embeds" if k == "embeddings" else k
        sample_dict[n] = ds[k].values
    sample_dict["inputs_embeds"] = torch.tensor(sample_dict["inputs_embeds"], dtype=torch.bfloat16)
    return sample_dict

def main() -> None:
    """Main training function"""
    args = parse_args()
    
    # Set precision and random seed
    set_float32_matmul_precision("medium")
    L.seed_everything(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    # Load preprocessed datasets
    logger.info("Loading preprocessed datasets...")
    logger.info(f"Loading training dataset from {args.train_dataset}")
    train_dataset = XarrayDataset(
        args.train_dataset, 
        max_sample_count=args.limit_train_examples, 
        max_sequence_length=args.window_size, 
        chunk_size=args.batch_size,
        sample_transform=sample_transform,
        consolidated=True,
    )
    logger.info(f"Training dataset loaded with {len(train_dataset)} samples ({len(train_dataset.datasets)} datasets)")
    logger.info(f"Loading validation dataset from {args.val_dataset}")
    val_dataset = XarrayDataset(
        args.val_dataset, 
        max_sequence_length=args.window_size, 
        chunk_size=args.batch_size,
        sample_transform=sample_transform,
        consolidated=True,
    )
    logger.info(f"Validation dataset loaded with {len(val_dataset)} samples ({len(val_dataset.datasets)} datasets)")
    
    # Create data loaders
    logger.info(f"Creating data loaders with batch_size={args.batch_size}")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=False,
        shuffle=False, # Assume the training dataset is already shuffled
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None
    )
    
    valid_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        pin_memory=False,
        shuffle=False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None
    )
    
    # Setup WandB logging
    logger.info(f"Setting up WandB logger with project={args.project_name}, run={args.run_name}")
    wandb_logger = WandbLogger(name=args.run_name, project=args.project_name, save_dir=args.output_dir)
    csv_dir = os.path.join(args.output_dir, "logs/csv")
    logger.info(f"Setting up CSV logger with save_dir={csv_dir}")
    csv_logger = CSVLogger(save_dir=csv_dir)
    loggers = [wandb_logger, csv_logger]
    
    # Initialize model
    logger.info(f"Initializing model (learning_rate={args.learning_rate}, learning_rate_decay={args.learning_rate_decay})")
    config = SequenceSpanClassifierConfig(
        max_sequence_length=args.window_size,
        train_eval_frequency=args.train_eval_frequency,
    )
    model = SequenceSpanClassifier(config, learning_rate=args.learning_rate, learning_rate_decay=args.learning_rate_decay)

    # Load checkpoint if provided
    if args.checkpoint and args.checkpoint_type == "model":
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        # Load the state dict, ignoring missing keys (like the frozen CRF parameters)
        missing_keys, unexpected_keys = model.load_state_dict(
            torch.load(args.checkpoint, weights_only=True)['state_dict'], 
            strict=False
        )
        if unexpected_keys:
            raise ValueError(f"Unexpected keys in checkpoint: {unexpected_keys}")
        if (unexpected_missing_keys := set([k for k in missing_keys if not k.startswith("crf.")])):
            raise ValueError(f"Missing keys in checkpoint: {unexpected_missing_keys}")
    
    # Setup callbacks
    logger.info("Setting up callbacks")
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=args.checkpoint_frequency,
        save_last=True,
        save_top_k=-1,
        mode="max",
        monitor="valid__entity__overall/f1",
        auto_insert_metric_name=False,
    )
    
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    throughput_monitor_callback = ThroughputMonitor()
    callbacks = [checkpoint_callback, lr_monitor_callback, throughput_monitor_callback]
    
    # Initialize trainer
    logger.info("Initializing Lightning Trainer")
    trainer = L.Trainer(
        max_epochs=args.epochs,
        default_root_dir=args.output_dir,
        log_every_n_steps=args.log_frequency,
        precision="bf16-mixed",
        devices=args.gpu,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator="gpu",
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        accumulate_grad_batches=args.accumulate_grad_batches, 
        logger=loggers, 
        callbacks=callbacks,
        profiler="simple",
        deterministic=True
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.fit(
        model=model, 
        train_dataloaders=train_loader, 
        val_dataloaders=valid_loader,
        ckpt_path=args.checkpoint if args.checkpoint_type == "trainer" else None
    )
    
    logger.info(f"Training complete (see {args.output_dir})")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Suppress the TorchMetrics warning about cpu computations when using Trainer(deterministic=True):
    # TorchMetricsUserWarning: You are trying to use a metric in deterministic mode on GPU that uses `torch.cumsum`, which is currently not supported. The tensor will be copied to the CPU memory to compute it and then copied back to GPU. Expect some slowdowns.
    warnings.filterwarnings(
        "ignore", message=".*You are trying to use a metric in deterministic mode on GPU.*"
    )
    # Also suppress this warning from ROC calculations:
    # UserWarning: No positive samples in targets, true positive value should be meaningless. Returning zero tensor in true positive score
    warnings.filterwarnings(
        "ignore", message=".*No positive samples in targets.*"
    )

    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
