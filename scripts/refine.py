"""
UNET-based refining pipeline for gene annotation predictions.

This script provides three main commands:
1. prepare - Create training datasets from predictions and labels
2. train - Train a UNET model to refine feature logits
3. apply - Apply trained model to refine predictions
"""

import argparse
import logging
import os
import sys
import numpy as np
from src.analysis import get_sequence_modeling_labels
import xarray as xr
import torch
from torch.utils.data import DataLoader
from argparse import Namespace as Args
import lightning as L
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch import set_float32_matmul_precision

from src.modeling import UnetClassifier
from src.sequence import create_sequence_windows
from src.dataset import open_datatree, XarrayDataset
from src.prediction import merge_prediction_datasets

logger = logging.getLogger(__name__)


def validate_coordinates(pred_coords: np.ndarray, label_coords: np.ndarray) -> None:
    """
    Validate that prediction and label coordinates match exactly.

    Parameters
    ----------
    pred_coords : np.ndarray
        Sequence coordinates from predictions dataset
    label_coords : np.ndarray
        Sequence coordinates from labels dataset

    Raises
    ------
    ValueError
        If coordinates don't match exactly
    """
    if not np.array_equal(pred_coords, label_coords):
        raise ValueError(
            f"Sequence coordinates don't match between predictions and labels. "
            f"Predictions: {len(pred_coords)} coords, Labels: {len(label_coords)} coords"
        )


FEATURE_COORDS = ["intergenic", "intron", "five_prime_utr", "cds", "three_prime_utr"]

# def transform_region_to_feature_labels(region_labels: xr.DataArray) -> xr.DataArray:
#     """
#     Transform region labels to feature labels compatible with predictions.

#     Parameters
#     ----------
#     region_labels : xr.DataArray
#         Region labels DataArray with dimensions (sequence, region) where region coordinates include:
#         ['intron', 'five_prime_utr', 'three_prime_utr', 'cds']

#     Returns
#     -------
#     xr.DataArray
#         Feature labels DataArray with dimensions (sequence, feature) where feature coordinates are:
#         ['intergenic', 'intron', 'five_prime_utr', 'cds', 'three_prime_utr']
#     """
#     # Create intergenic label as background (where no other labels are 1)
#     values = region_labels.sel(region=['intron', 'five_prime_utr', 'cds', 'three_prime_utr'])
#     assert values.isin([0, 1]).all()
#     intergenic = (1 - values.max(dim='region')).expand_dims("region").assign_coords(region=["intergenic"])
#     feature_labels = xr.concat([intergenic, values], dim='region').rename(region="feature")

#     # Remap CDS coordinates and validate coord order
#     assert feature_labels.feature.values.tolist() == FEATURE_COORDS
#     feature_labels = feature_labels.assign_coords(feature=coords)
#     return feature_labels


def prepare_command(args: Args) -> None:
    """
    Prepare training datasets from predictions and labels for a single species/chromosome.

    Parameters
    ----------
    args : Args
        Command line arguments
    """
    logger.info("Starting data preparation")
    logger.info(f"Processing {args.species_id}:{args.chromosome_id}")

    # Load prediction dataset using merge_prediction_datasets
    logger.info(f"Loading prediction dataset from {args.prediction_dataset}")
    pred_ds = merge_prediction_datasets(args.prediction_dataset)
    logger.info(f"Loaded prediction dataset with dimensions: {dict(pred_ds.sizes)}")

    # Load labels dataset
    logger.info(f"Loading labels dataset from {args.labels_dataset}")
    labels_ds = xr.open_zarr(args.labels_dataset)
    logger.info(f"Loaded labels dataset with dimensions: {dict(labels_ds.sizes)}")

    # Process each strand
    total_windows = 0
    sample_index = 0
    strands = pred_ds.strand.values.tolist()
    assert set(strands) == {"positive", "negative"}
    for strand in strands:
        logger.info(f"Processing strand: {strand}")

        # Extract feature logits and region labels for this strand
        feature_logits = pred_ds.sel(strand=strand).feature_logits

        # Gather feature labels for training
        feature_labels = get_sequence_modeling_labels(labels_ds.sel(strand=strand))

        # Truncate feature logits to match region labels length if necessary
        min_length = min(len(feature_logits.sequence), len(feature_labels.sequence))
        if len(feature_logits.sequence) > min_length:
            logger.info(
                f"Truncating feature logits from {len(feature_logits.sequence)} to {min_length}"
            )
            feature_logits = feature_logits.isel(sequence=slice(min_length))
        if len(feature_labels.sequence) > min_length:
            logger.info(
                f"Truncating region labels from {len(feature_labels.sequence)} to {min_length}"
            )
            feature_labels = feature_labels.isel(sequence=slice(min_length))

        # Validate coordinates match
        validate_coordinates(
            feature_logits.sequence.values, feature_labels.sequence.values
        )
        logger.info(f"Coordinate validation passed for {min_length} positions")

        # Validate that feature coordinates match between logits and labels
        logits_features = feature_logits.feature.values.tolist()
        labels_features = feature_labels.feature.values.tolist()
        if logits_features != labels_features:
            raise ValueError(
                f"Feature coordinates mismatch: "
                f"feature_logits has {logits_features}, "
                f"feature_labels has {labels_features}"
            )
        logger.info(f"Feature coordinate validation passed: {logits_features}")

        # Create windows for this strand
        sequence_length = len(feature_logits.sequence)
        feature_logits_array = feature_logits.transpose("sequence", "feature").values
        feature_labels_array = feature_labels.transpose("sequence", "feature").values
        if feature_logits_array.shape != feature_labels_array.shape:
            raise ValueError(
                f"Feature logits and labels arrays have different shapes: "
                f"feature_logits: {feature_logits_array.shape}, "
                f"feature_labels: {feature_labels_array.shape}"
            )

        # Create list of start indices
        start_indices = list(
            range(0, sequence_length - args.window_size + 1, args.window_stride)
        )

        # Shuffle indices if requested
        if args.shuffle == "yes":
            rs = np.random.RandomState(args.seed)
            start_indices = rs.permutation(start_indices).tolist()
            logger.info(
                f"Shuffled {len(start_indices)} start indices with seed {args.seed}"
            )
        else:
            logger.info(
                f"Using sequential order for {len(start_indices)} start indices"
            )

        # Split indices into train and validation sets
        rs = np.random.RandomState(args.seed)  # Use same seed for consistent splits
        n_val = int(len(start_indices) * args.val_proportion)
        val_indices_set = set(rs.choice(start_indices, size=n_val, replace=False))

        train_indices = [idx for idx in start_indices if idx not in val_indices_set]
        val_indices = [idx for idx in start_indices if idx in val_indices_set]

        logger.info(
            f"Split {len(start_indices)} indices into {len(train_indices)} train and {len(val_indices)} validation"
        )

        # Process train and validation splits separately
        for split_name, split_indices in [
            ("train", train_indices),
            ("valid", val_indices),
        ]:
            if not split_indices:
                logger.info(f"No {split_name} indices, skipping")
                continue

            logger.info(f"Processing {len(split_indices)} {split_name} windows")

            # Collect windows for this strand and split
            strand_windows = []
            num_windows = 0

            for i, start_idx in enumerate(split_indices):
                # Check window limit (apply to total across both splits)
                if args.window_limit is not None and (i + 1) >= args.window_limit:
                    logger.info(
                        f"Reached window limit of {args.window_limit} for split {split_name}, stopping window creation"
                    )
                    break

                # Extract window data using the same start_idx regardless of strand
                window_logits = feature_logits_array[
                    start_idx : (start_idx + args.window_size)
                ]
                window_labels = feature_labels_array[
                    start_idx : (start_idx + args.window_size)
                ]
                assert window_logits.shape == window_labels.shape

                # Generate 5' to 3' sequences regardless of strand, which
                # requires flipping them for the negative strand
                if strand == "negative":
                    window_logits = np.flip(window_logits, axis=0)
                    window_labels = np.flip(window_labels, axis=0)

                # Verify window has exactly the right size
                if len(window_logits) != args.window_size:
                    logger.warning(
                        f"Skipping window {start_idx} due to incorrect size: {len(window_logits)} != {args.window_size}"
                    )
                    continue

                # Store window data
                window_data = {
                    "input_logits": window_logits.astype(np.float32),
                    "target_labels": window_labels.astype(np.int8),
                    "sample_index": sample_index,
                    "species_id": args.species_id,
                    "chromosome_id": args.chromosome_id,
                    "strand": strand,
                    "position": start_idx,
                }
                strand_windows.append(window_data)
                sample_index += 1
                num_windows += 1

            logger.info(
                f"Created {num_windows} {split_name} windows for {args.species_id}:{args.chromosome_id}:{strand}"
            )

            # Create dataset for this strand and split if we have windows
            if strand_windows:
                # Stack windows into arrays
                input_logits = np.stack([w["input_logits"] for w in strand_windows])
                target_labels = np.stack([w["target_labels"] for w in strand_windows])
                sample_indices = np.array([w["sample_index"] for w in strand_windows])
                species_ids = np.array(
                    [w["species_id"] for w in strand_windows], dtype="<U64"
                )
                chromosome_ids = np.array(
                    [w["chromosome_id"] for w in strand_windows], dtype="<U64"
                )
                strands = np.array([w["strand"] for w in strand_windows], dtype="<U8")
                positions = np.array([w["position"] for w in strand_windows])

                # Create xarray dataset for this strand and split
                strand_ds = xr.Dataset(
                    data_vars={
                        "input_logits": (
                            ["sample", "sequence", "feature"],
                            input_logits,
                        ),
                        "target_labels": (
                            ["sample", "sequence", "feature"],
                            target_labels,
                        ),
                        "sample_index": (["sample"], sample_indices),
                        "species_id": (["sample"], species_ids),
                        "chromosome_id": (["sample"], chromosome_ids),
                        "strand": (["sample"], strands),
                        "position": (["sample"], positions),
                    },
                    coords={
                        "sample": np.arange(len(strand_windows)),
                        "sequence": np.arange(args.window_size),
                        "feature": FEATURE_COORDS,
                    },
                    attrs={
                        "window_size": args.window_size,
                        "window_stride": args.window_stride,
                        "species_id": args.species_id,
                        "chromosome_id": args.chromosome_id,
                        "strand": strand,
                        "split": split_name,
                    },
                )

                # Create split-specific output path
                split_output = f"{args.output}/{split_name}.zarr"
                if not os.path.exists(args.output):
                    os.makedirs(args.output, exist_ok=True)

                # Save dataset (append if file exists)
                strand_ds.to_zarr(
                    split_output,
                    zarr_format=2,
                    **(
                        dict(append_dim="sample")
                        if os.path.exists(split_output)
                        else {}
                    ),
                    consolidated=True,
                )

                total_windows += len(strand_windows)
                logger.info(
                    f"Saved {len(strand_windows)} {split_name} windows for {args.species_id}:{args.chromosome_id}:{strand} to {split_output}"
                )

    if total_windows == 0:
        raise ValueError("No valid training windows were created")

    logger.info(
        f"Data preparation complete. Created {total_windows} total training samples."
    )
    logger.info(f"Training dataset saved to {args.output}")


def train_command(args: Args) -> None:
    logger.info("Starting model training")

    # Set precision and random seed
    set_float32_matmul_precision("medium")
    L.seed_everything(args.seed)
    logger.info(f"Set random seed to {args.seed}")

    # Define sample transform function for XarrayDataset
    def sample_transform(ds: xr.Dataset) -> dict:
        """Transform xarray dataset sample to training format."""
        if ds.input_logits.dims != ("sequence", "feature"):
            raise AssertionError(
                f"Expected ('sequence', 'feature') dims for input_logits, got {ds.input_logits.dims}"
            )
        if ds.target_labels.dims != ("sequence", "feature"):
            raise AssertionError(
                f"Expected ('sequence', 'feature') dims for target_labels, got {ds.target_labels.dims}"
            )
        input_logits = torch.tensor(ds.input_logits.values)
        target_labels = torch.tensor(ds.target_labels.values)
        return {
            "input_logits": input_logits,
            "target_labels": target_labels,
            "sample_index": ds.sample_index.item(),
        }

    # Load training dataset using XarrayDataset
    logger.info(f"Loading training dataset from {args.train_dataset}")
    train_dataset = XarrayDataset(
        path=args.train_dataset,
        sample_transform=sample_transform,
        chunk_size=64,
        consolidated=True,
    )
    logger.info(f"Loaded {len(train_dataset)} training samples")

    # Load validation dataset
    logger.info(f"Loading validation dataset from {args.val_dataset}")
    val_dataset = XarrayDataset(
        path=args.val_dataset,
        sample_transform=sample_transform,
        chunk_size=64,
        consolidated=True,
    )
    logger.info(f"Loaded {len(val_dataset)} validation samples")

    # Get feature coordinates from the first sample to determine number of classes
    feature_coords = train_dataset.datasets[0].feature.values.tolist()
    num_classes = len(feature_coords)

    # Print dataset info and feature list
    logger.info(
        f"Example training dataset (1 of {len(train_dataset.datasets)}): {train_dataset.datasets[0]}"
    )
    logger.info(f"Feature coordinates: {feature_coords}")
    logger.info(f"Number of classes (features): {num_classes}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Data is already shuffled during preparation
        num_workers=2,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    # Initialize Lightning module
    logger.info("Initializing Lightning module")
    model = UnetClassifier(
        embed_dim=num_classes, num_classes=num_classes, learning_rate=args.learning_rate
    )

    # Setup WandB logging
    logger.info(
        f"Setting up WandB logger with project={args.project_name}, run={args.run_name}"
    )
    wandb_logger = WandbLogger(
        name=args.run_name, project=args.project_name, save_dir=args.output_dir
    )
    csv_dir = os.path.join(args.output_dir, "logs/csv")
    logger.info(f"Setting up CSV logger with save_dir={csv_dir}")
    csv_logger = CSVLogger(save_dir=csv_dir)
    loggers = [wandb_logger, csv_logger]

    # Setup callbacks
    logger.info("Setting up callbacks")
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=args.checkpoint_frequency,
        save_on_train_epoch_end=True,
        save_last=True,
        save_top_k=-1,
        mode="min",
        monitor="val/loss",
        auto_insert_metric_name=False,
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor_callback]

    # Initialize trainer
    logger.info("Initializing Lightning Trainer")
    trainer = L.Trainer(
        max_epochs=args.epochs,
        default_root_dir=args.output_dir,
        log_every_n_steps=args.log_frequency,
        precision="bf16-mixed",
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=loggers,
        callbacks=callbacks,
        deterministic=False,
    )

    # Start training
    logger.info("Starting training...")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    logger.info("Training completed successfully")


def apply_command(args: Args) -> None:
    """
    Apply trained model to refine predictions.

    Parameters
    ----------
    args : Args
        Command line arguments
    """
    logger.info("Starting model inference")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load trained model
    logger.info(f"Loading model from {args.model_checkpoint}")
    model = UnetClassifier.load_from_checkpoint(args.model_checkpoint)
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully:\n{model}")

    # Load input prediction dataset
    logger.info(f"Loading prediction dataset from {args.input}")
    pred_ds = merge_prediction_datasets(args.input)

    # Process each strand and save directly
    strands = pred_ds.strand.values.tolist()
    assert set(strands) == {"positive", "negative"}

    for strand in strands:
        logger.info(f"Processing strand: {strand}")

        # Extract feature logits for this strand
        feature_logits = pred_ds.feature_logits.sel(strand=strand).transpose(
            "sequence", "feature"
        )
        sequence_length = feature_logits.sizes["sequence"]
        num_features = feature_logits.sizes["feature"]

        logger.info(
            f"Processing {sequence_length} positions with {num_features} features"
        )

        # Create sliding windows
        feature_logits_array = feature_logits.values  # Shape: (sequence, feature)
        assert feature_logits_array.shape == (sequence_length, num_features)

        # Reverse processing for negative strand
        if strand == "negative":
            feature_logits_array = np.flip(feature_logits_array, axis=0)

        windows = list(
            create_sequence_windows(
                feature_logits_array,
                window_size=args.window_size,
                stride=args.window_stride,
                pad_value=0.0,
            )
        )

        logger.info(f"Created {len(windows)} windows for inference")

        # Collect local outputs for concatenation
        local_outputs = []

        # Process windows
        with torch.inference_mode():
            for window_data, local_window, _ in windows:
                # Extract the global window data for model input
                input_data = window_data  # Shape: (window_size, num_features)
                if input_data.shape != (args.window_size, num_features):
                    raise AssertionError(
                        f"Expected shape ({args.window_size}, {num_features}), got {input_data.shape}"
                    )

                # Convert to tensor with batch dimension
                input_tensor = (
                    torch.tensor(input_data, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(device)
                )

                # Apply model
                output_tensor = model(input_tensor)

                # Convert back to numpy and transpose back to (sequence, feature)
                output_array = output_tensor.squeeze(0).cpu().numpy()
                assert output_array.shape == (args.window_size, num_features), (
                    f"Expected shape ({args.window_size}, {num_features}), got {output_array.shape}"
                )

                # Extract the local window from the output (the valid, non-overlapping part)
                local_start, local_end = local_window
                local_output = output_array[local_start:local_end]

                # Collect local output
                local_outputs.append(local_output)

        # Concatenate all local outputs to form the complete refined sequence
        refined_logits = np.concatenate(local_outputs, axis=0)

        # Flip back to positive strand order
        if strand == "negative":
            refined_logits = np.flip(refined_logits, axis=0)

        if refined_logits.shape != feature_logits_array.shape:
            raise AssertionError(
                f"Shape mismatch: expected {feature_logits_array.shape}, got {refined_logits.shape}"
            )

        # Create dataset for this strand with both original and refined logits
        strand_ds = pred_ds.sel(strand=strand).copy()
        strand_ds = strand_ds.rename(
            {
                "feature_logits": "feature_logits_base",
                "feature_predictions": "feature_predictions_base",
            }
        )
        strand_ds["feature_logits"] = (["sequence", "feature"], refined_logits)
        strand_ds["feature_predictions"] = strand_ds["feature_logits"].argmax(
            dim="feature"
        )

        # Save dataset for this strand
        strand_ds.to_zarr(
            args.output,
            group=f"/{strand}",
            zarr_format=2,
            **(
                dict(append_dim="sequence")
                if os.path.exists(os.path.join(args.output, strand))
                else {}
            ),
            consolidated=True,
        )
        logger.info(f"Saved refined predictions for strand: {strand}")

    # Open and display the final result
    logger.info(f"Opening final result from {args.output}")
    final_datatree = open_datatree(args.output, consolidated=True)
    logger.info(f"Final refined predictions datatree:\n{final_datatree}")

    logger.info("Model application completed successfully")


def main():
    """Main entry point for the refining pipeline."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="UNET-based refining pipeline for gene annotation predictions"
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Command to execute"
    )

    # Prepare command
    prepare_parser = subparsers.add_parser(
        "prepare", help="Create training datasets from predictions and labels"
    )
    prepare_parser.add_argument("--species-id", required=True, help="Species ID")
    prepare_parser.add_argument("--chromosome-id", required=True, help="Chromosome ID")
    prepare_parser.add_argument(
        "--prediction-dataset", required=True, help="Path to prediction dataset"
    )
    prepare_parser.add_argument(
        "--labels-dataset", required=True, help="Path to labels dataset"
    )
    prepare_parser.add_argument(
        "--output", required=True, help="Path to output training dataset"
    )
    prepare_parser.add_argument(
        "--window-size",
        type=int,
        default=1_048_576,
        help="Window size for chunking (default: 1,048,576)",
    )
    prepare_parser.add_argument(
        "--window-stride",
        type=int,
        default=524_288,
        help="Window stride for chunking (default: 524,288)",
    )
    prepare_parser.add_argument(
        "--window-limit",
        type=int,
        default=None,
        help="Maximum number of windows to process (default: None, process all windows). Useful for debugging.",
    )
    prepare_parser.add_argument(
        "--shuffle",
        type=str,
        choices=["yes", "no"],
        default="yes",
        help="Shuffle start indices (default: yes)",
    )
    prepare_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    prepare_parser.add_argument(
        "--val-proportion",
        type=float,
        default=0.05,
        help="Proportion of windows to use for validation (default: 0.05)",
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train UNET model for refining")
    train_parser.add_argument(
        "--train-dataset", required=True, help="Path to training dataset"
    )
    train_parser.add_argument(
        "--val-dataset", required=True, help="Path to validation dataset"
    )
    train_parser.add_argument(
        "--output-dir", required=True, help="Directory for model outputs"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs (default: 10)"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: 8)"
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    train_parser.add_argument(
        "--log-frequency",
        type=int,
        default=5,
        help="Frequency of logging during training (default: 5)",
    )
    train_parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=100,
        help="Number of steps between saving checkpoints (default: 100)",
    )
    train_parser.add_argument(
        "--project-name", default="test", help="W&B project name (default: test)"
    )
    train_parser.add_argument(
        "--run-name", default="test", help="W&B run name (default: test)"
    )
    train_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    # Apply command
    apply_parser = subparsers.add_parser(
        "apply", help="Apply trained model to refine predictions"
    )
    apply_parser.add_argument(
        "--input", required=True, help="Path to prediction dataset"
    )
    apply_parser.add_argument(
        "--output", required=True, help="Path to output refined dataset"
    )
    apply_parser.add_argument(
        "--model-checkpoint", required=True, help="Path to trained model checkpoint"
    )
    apply_parser.add_argument(
        "--window-size",
        type=int,
        default=1_048_576,
        help="Window size for inference (default: 1,048,576)",
    )
    apply_parser.add_argument(
        "--window-stride",
        type=int,
        default=524_288,
        help="Window stride for inference (default: 524,288)",
    )

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "apply":
        apply_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
