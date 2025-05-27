"""
Training script for LLM gene annotation classifier

This trainer is intended to take prepared datasets (created by prepare_model_lightning.py)
and fine-tune a model for a classification task.
"""
import argparse
import os
import time
import logging
import torch
from torch.utils.data import DataLoader
from torch import set_float32_matmul_precision
import xarray as xr
import lightning as L
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from argparse import Namespace as Args
from typing import Optional
from src.dataset import XarrayDataset
from src.config import WINDOW_SIZE
from src.modeling import GeneClassifier, GeneClassifierConfig, ThroughputMonitor

logger = logging.getLogger(__name__)

def parse_args(args: Optional[list[str]] = None) -> Args:
    parser = argparse.ArgumentParser(description="Train genome annotation model")
    parser.add_argument('--train-dataset', type=str, required=True, help='Path to prepared training dataset (zarr)')
    parser.add_argument('--val-dataset', type=str, required=True, help='Path to prepared validation dataset (zarr)')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory for saving outputs')
    parser.add_argument('--learning-rate', type=float, default=8e-4, help='learning rate for the model')
    parser.add_argument('--learning-rate-decay', type=str, default="none", help='Whether to use learning rate decay; either "none" or "cosine"')
    parser.add_argument('--window-size', type=int, default=WINDOW_SIZE, help='Size of the window for processing sequences')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for data loading')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to resume training from')
    parser.add_argument('--checkpoint-type', type=str, default=None, help='Type of checkpoint to resume training from; one of "model", "trainer"')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')
    parser.add_argument('--log-frequency', type=int, default=5, help='Frequency of logging during training')
    parser.add_argument('--val-check-interval', type=float, default=1.0, help='how often within one training epoch to evaluate the validation set.')
    parser.add_argument('--train-eval-frequency', type=int, default=250, help='how often within one training epoch to evaluate the training set (in steps).')
    parser.add_argument('--limit-val-batches', type=float, default=1.0, help='how much of the validation dataset to check.')
    parser.add_argument('--limit-train-examples', type=int, default=None, help='how many training examples to use')
    parser.add_argument('--accumulate-grad-batches', type=int, default=16, help='batches to accumulate before optimizing')
    parser.add_argument('--num-workers', type=int, default=0, help='number of worker threads for data loading')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='batches for dataloader to prefetch')
    parser.add_argument('--project-name', default='test-project', help='wandb project name')
    parser.add_argument('--run-name', default='test-run', help='wandb run name')
    parser.add_argument('--run-id', default=None, help='wandb run id to be resumed, if using a checkpoint')
    parser.add_argument('--gpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--num-nodes', type=int, default=1, help='number of nodes to use')
    parser.add_argument('--strategy', type=str, default="auto", help='training strategy')
    parser.add_argument('--checkpoint-frequency', type=int, default=1000, help="number of batches between saving checkpoints")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--architecture', type=str, default="encoder-only", choices=["encoder-only", "sequence-only", "classifier-only", "all"], help='model architecture type')
    parser.add_argument('--head-encoder-layers', type=int, default=8, help='number of layers in the head encoder')
    parser.add_argument('--token-embedding-dim', type=int, default=512, help='dimension of the token embedding')
    parser.add_argument('--enable-visualization', type=str, default="yes", choices=["yes", "no"], help='Enable visualization during training')
    parser.add_argument('--base-encoder-path', type=str, default=None, help='Path to the base encoder model to use, e.g. "kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000"')
    parser.add_argument('--base-encoder-frozen', type=str, default="yes", choices=["yes", "no"], help='Freeze the base encoder during training')
    return parser.parse_args(args)

def sample_transform(ds: xr.Dataset) -> dict:
    """Custom sample transformation function for training data"""
    sample_dict = {}
    for k in ["input_ids", "label_mask", "labels", "embeddings", "sample_index"]:
        if k == "embeddings" and k not in ds:
            continue
        n = "inputs_embeds" if k == "embeddings" else k
        sample_dict[n] = ds[k].values
    if "inputs_embeds" in sample_dict:
        sample_dict["inputs_embeds"] = torch.tensor(sample_dict["inputs_embeds"], dtype=torch.bfloat16)
    return sample_dict

def train(args: Args) -> None:
    
    # Set precision and random seed
    set_float32_matmul_precision("medium")
    L.seed_everything(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    # Load preprocessed datasets
    logger.info("Loading preprocessed datasets...")
    logger.info(f"Loading training dataset from {args.train_dataset}")
    drop_variables = None
    if args.base_encoder_path:
        # Avoid deserializing large embeddings arrays when not they will not be used
        drop_variables = ["embeddings"]
    train_dataset = XarrayDataset(
        args.train_dataset, 
        max_sample_count=args.limit_train_examples, 
        max_sequence_length=args.window_size, 
        chunk_size=args.batch_size,
        sample_transform=sample_transform,
        consolidated=True,
        drop_variables=drop_variables,
    )
    logger.info(f"Training dataset loaded with {len(train_dataset)} samples ({len(train_dataset.datasets)} datasets)")
    logger.info(f"Loading validation dataset from {args.val_dataset}")
    val_dataset = XarrayDataset(
        args.val_dataset, 
        max_sequence_length=args.window_size, 
        chunk_size=args.batch_size,
        sample_transform=sample_transform,
        consolidated=True,
        drop_variables=drop_variables,
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
    logger.info(f"Setting up WandB logger with project={args.project_name}, run={args.run_name}, id={args.run_id}")
    wandb_logger = WandbLogger(
        name=args.run_name, 
        project=args.project_name, 
        save_dir=args.output_dir,
        **(dict(id=args.run_id, resume="must") if args.run_id else dict())
    )
    csv_dir = os.path.join(args.output_dir, "logs/csv")
    logger.info(f"Setting up CSV logger with save_dir={csv_dir}")
    csv_logger = CSVLogger(save_dir=csv_dir)
    loggers = [wandb_logger, csv_logger]
    
    # Initialize model
    # Load checkpoint if provided
    logger.info(f"Initializing model (learning_rate={args.learning_rate}, learning_rate_decay={args.learning_rate_decay})")
    if args.checkpoint and args.checkpoint_type == "model":
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        model = GeneClassifier.load_from_checkpoint(args.checkpoint, learning_rate=args.learning_rate, learning_rate_decay=args.learning_rate_decay)
        config = model.config
    else:
        logger.info(f"Creating new model (architecture={args.architecture}, head_encoder_layers={args.head_encoder_layers})")
        config = GeneClassifierConfig(
            architecture=args.architecture,
            max_sequence_length=args.window_size,
            token_embedding_dim=args.token_embedding_dim,
            train_eval_frequency=args.train_eval_frequency,
            head_encoder_layers=args.head_encoder_layers,
            base_encoder_path=args.base_encoder_path,
            enable_visualization=args.enable_visualization == "yes",
            base_encoder_frozen=args.base_encoder_frozen == "yes",
        )
        model = GeneClassifier(config, learning_rate=args.learning_rate, learning_rate_decay=args.learning_rate_decay)
    
    logger.info(f"Model for training:\n{model}")

    # Setup callbacks
    logger.info("Setting up callbacks")
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=args.checkpoint_frequency,
        save_on_train_epoch_end=True,
        save_last=True,
        save_top_k=-1,
        mode="max",
        monitor="valid__entity__overall/f1",
        auto_insert_metric_name=False,
    )
    
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    throughput_monitor_callback = ThroughputMonitor()
    callbacks = [checkpoint_callback, lr_monitor_callback, throughput_monitor_callback]
    
    strategy = args.strategy
    if config.use_head_encoder and strategy == "ddp":
        logger.warning("Coercing strategy from 'ddp' to 'ddp_find_unused_parameters_true' when using ModernBERT head encoder")
        strategy = "ddp_find_unused_parameters_true"


    # Initialize trainer
    logger.info("Initializing Lightning Trainer")
    trainer = L.Trainer(
        max_epochs=args.epochs,
        default_root_dir=args.output_dir,
        log_every_n_steps=args.log_frequency,
        precision="bf16-mixed",
        devices=args.gpu,
        num_nodes=args.num_nodes,
        strategy=strategy,
        accelerator="gpu",
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        accumulate_grad_batches=args.accumulate_grad_batches, 
        logger=loggers, 
        callbacks=callbacks,
        profiler="simple",
        deterministic=False
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

def main() -> None:
    args = parse_args()
    train(args)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
