import pandas as pd
import argparse
import numpy as np
import logging
import os
from argparse import Namespace as Args
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
from src.dataset import open_datatree, set_dimension_chunks
from src.sampling import (
    get_feature_class_map,
    get_tag_stats,
    extract_label_dataset,
    select_windows,
    get_tag_class_map,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Generate training windows
# -------------------------------------------------------------------------------------------------


def _generate_training_windows(task):
    """Worker function to process a single dataset in parallel.

    Parameters
    ----------
    task : tuple
        Tuple containing (args, output_path, species_id, chrom_id, task_id, total_tasks) where:
        - args: command line arguments
        - output_path: path to output zarr dataset
        - species_id: species identifier
        - chrom_id: chromosome identifier
        - task_id: 1-based task identifier
        - total_tasks: total number of tasks

    Returns
    -------
    dict
        stats_dict containing processing statistics
    """
    args, output_path, species_id, chrom_id, task_id, total_tasks = task

    # Load the dataset for this specific group
    logger.info(
        f"[{species_id}/{chrom_id}] Processing {task_id}/{total_tasks}: Loading dataset"
    )
    ds = xr.open_zarr(args.input, group=f"/{species_id}/{chrom_id}")

    seq_length = ds.sizes["sequence"]
    logger.info(f"[{species_id}/{chrom_id}] Sequence length: {seq_length} bp")

    # Extract BILUO labels once for this dataset
    ds_labels = extract_label_dataset(ds).compute()
    ds_data = ds[["sequence_input_ids", "sequence_masks"]].compute()

    # Get tag class mapping and stats data
    tag_class_map = get_tag_class_map(ds_labels.feature_labels)
    feature_class_map = get_feature_class_map(ds_labels.feature_labels)
    tag_stats = get_tag_stats(ds_labels.tag_labels_masked, tag_class_map)

    # Select windows based on intergenic proportion
    selected_windows, window_stats = select_windows(
        ds_labels.feature_labels,
        seq_length,
        window_size=args.window_size,
        intergenic_proportion=args.intergenic_proportion,
        seed=args.seed,
        intergenic_threshold=args.intergenic_threshold,
    )

    # Log window selection stats
    logger.info(
        f"[{species_id}/{chrom_id}] Windows: {window_stats['available_intergenic_windows']}/{window_stats['available_genic_windows']} available (intergenic/genic), "
        f"{window_stats['selected_intergenic_windows']}/{window_stats['selected_genic_windows']} selected, actual intergenic prop: {window_stats['actual_intergenic_proportion']:.3f} "
        f"(target: {window_stats['target_intergenic_proportion']:.3f})"
    )

    # Collect samples for this dataset
    dataset_samples = []
    local_window_counter = 0
    local_windows_processed = 0
    local_windows_kept = 0
    local_windows_skipped = 0

    # Process both strands
    if (actual := set(ds.strand.values)) != (expected := {"positive", "negative"}):
        raise ValueError(f"Expected strand values {expected}, got {actual}")

    for strand_name in ds.strand.values:
        # Extract strand data once and compute to load into memory
        strand_data = ds_data.sel(strand=strand_name)
        strand_labels = ds_labels.sel(strand=strand_name)

        for w_start, w_end in selected_windows:
            local_window_counter += 1
            local_windows_processed += 1

            # Extract window data
            window_data = strand_data.isel(sequence=slice(w_start, w_end))
            window_labels = strand_labels.isel(sequence=slice(w_start, w_end))

            # Reverse sequence dimension for negative strand
            if strand_name == "negative":
                window_data = window_data.isel(sequence=slice(None, None, -1))
                window_labels = window_labels.isel(sequence=slice(None, None, -1))

            # Check mask proportion
            window_valid_mask = window_labels.label_masks.values
            masked_count = (~window_valid_mask).sum()
            if masked_count > args.window_size * args.mask_prop:
                local_windows_skipped += 1
                continue

            local_windows_kept += 1

            # Create sample
            # Extract BILUO labels for this window
            window_tag_labels_masked = window_labels.tag_labels_masked.values
            window_tag_labels = window_labels.tag_labels.values
            # Make feature_labels 2D with argmax; check that at least one
            # feature is present at each position first to avoid ambiguity
            # w/ argmax of 0 when all features 0 vs 0 when the first feature is present
            assert (window_labels.feature_labels.sum(dim="feature") > 0).all().item()
            window_feature_labels = window_labels.feature_labels.argmax(
                dim="feature"
            ).values

            # Ensure that the mask is consistent with sentinel values in the masked labels
            assert ((window_tag_labels_masked >= 0) == window_valid_mask).all()
            sample = {
                "input_ids": window_data.sequence_input_ids.values,
                "tag_labels_masked": window_tag_labels_masked,
                "tag_labels": window_tag_labels,
                "feature_labels": window_feature_labels,
                "soft_mask": window_data.sequence_masks.values,
                "label_mask": window_valid_mask,
                "species": species_id,
                "chromosome": chrom_id,
                "strand": strand_name,
                "position": w_start,
            }

            dataset_samples.append(sample)

    # Build and save dataset if we have samples
    if dataset_samples:
        # Convert to xarray Dataset
        logger.info(
            f"[{species_id}/{chrom_id}] Creating dataset with {len(dataset_samples)} samples"
        )
        data_vars = {}
        for key in dataset_samples[0].keys():
            values_list = [sample[key] for sample in dataset_samples]

            # Handle string fields with fixed dtypes to ensure consistency across datasets
            if key == "species":
                values = np.array(values_list, dtype="<U64")
            elif key == "chromosome":
                values = np.array(values_list, dtype="<U64")
            elif key == "strand":
                values = np.array(values_list, dtype="<U8")
            else:
                values = np.stack(values_list)

            if values.ndim == 1:
                data_vars[key] = (["sample"], values)
            elif values.ndim == 2:
                data_vars[key] = (["sample", "sequence"], values)
            else:
                raise ValueError(
                    f"Unexpected number of dimensions for key '{key}': {values.ndim}. Expected 1 or 2 dimensions."
                )

        ds_out = xr.Dataset(data_vars=data_vars)
        ds_out = set_dimension_chunks(ds_out, "sample", args.chunk_size)

        # Add attributes including class mappings
        ds_out.attrs.update(
            {
                "species_id": species_id,
                "chromosome_id": chrom_id,
                "feature_labels_classes": feature_class_map,
                "label_classes": tag_class_map,
            }
        )

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to zarr
        output_group = f"{species_id}/{chrom_id}"
        logger.info(f"[{species_id}/{chrom_id}] Saving to {output_path}/{output_group}")
        ds_out.to_zarr(
            output_path, group=output_group, zarr_format=2, consolidated=True, mode="w"
        )

    stats = {
        "windows_processed": local_windows_processed,
        "windows_kept": local_windows_kept,
        "windows_skipped": local_windows_skipped,
        "species_id": species_id,
        "chrom_id": chrom_id,
        "samples_generated": len(dataset_samples),
        "tag_stats": tag_stats,
        "window_stats": window_stats,
    }

    # Print readable statistics for this dataset
    if local_windows_processed > 0:
        kept_pct = (local_windows_kept / local_windows_processed) * 100
        skipped_pct = (local_windows_skipped / local_windows_processed) * 100
        logger.info(f"[{species_id}/{chrom_id}] Completed:")
        logger.info(
            f"[{species_id}/{chrom_id}]   Windows processed: {local_windows_processed}"
        )
        logger.info(
            f"[{species_id}/{chrom_id}]   Windows kept: {local_windows_kept} ({kept_pct:.1f}%)"
        )
        logger.info(
            f"[{species_id}/{chrom_id}]   Windows skipped: {local_windows_skipped} ({skipped_pct:.1f}%)"
        )
        logger.info(
            f"[{species_id}/{chrom_id}]   Samples generated: {len(dataset_samples)}"
        )
    else:
        logger.info(f"[{species_id}/{chrom_id}] Completed: No windows processed")

    return stats


def summarize_window_stats(all_stats: list[dict]) -> None:
    """Summarize window and tag frequency statistics."""
    # Window statistics
    total_windows_processed = sum(stats["windows_processed"] for stats in all_stats)
    windows_kept = sum(stats["windows_kept"] for stats in all_stats)
    windows_skipped = sum(stats["windows_skipped"] for stats in all_stats)

    logger.info("=" * 60)
    logger.info("WINDOW STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total windows processed: {total_windows_processed}")
    logger.info(f"Windows kept: {windows_kept}")
    logger.info(f"Windows skipped: {windows_skipped}")
    if total_windows_processed > 0:
        kept_pct = (windows_kept / total_windows_processed) * 100
        skipped_pct = (windows_skipped / total_windows_processed) * 100
        logger.info(f"Keep rate: {kept_pct:.1f}%")
        logger.info(f"Skip rate: {skipped_pct:.1f}%")

    # Per-dataset statistics
    logger.info("Per-dataset statistics:")
    for stats in all_stats:
        logger.info(
            f"  {stats['species_id']}/{stats['chrom_id']}: {stats['windows_kept']} kept, {stats['windows_skipped']} skipped"
        )

    # Window selection statistics
    logger.info("\nWINDOW SELECTION STATISTICS")
    logger.info("=" * 60)

    # Build dataframe for window selection statistics
    window_selection_rows = []
    for stats in all_stats:
        window_stats = stats["window_stats"]
        window_selection_rows.append(
            {
                "Dataset": f"{stats['species_id']}/{stats['chrom_id']}",
                "Available_Intergenic": window_stats["available_intergenic_windows"],
                "Available_Genic": window_stats["available_genic_windows"],
                "Selected_Intergenic": window_stats["selected_intergenic_windows"],
                "Selected_Genic": window_stats["selected_genic_windows"],
                "Target_Intergenic_Prop": window_stats["target_intergenic_proportion"],
                "Actual_Intergenic_Prop": window_stats["actual_intergenic_proportion"],
            }
        )

    window_selection_df = pd.DataFrame(window_selection_rows)
    logger.info(
        f"Window Selection Summary:\n{window_selection_df.to_string(index=False, float_format='%.3f')}"
    )

    # Tag frequency statistics
    logger.info("\nTAG FREQUENCY STATISTICS")
    logger.info("=" * 60)

    # Build data for tables
    rows = []
    for stats in all_stats:
        dataset = f"{stats['species_id']}/{stats['chrom_id']}"
        for tag_info in stats["tag_stats"]:
            rows.append(
                {
                    "Dataset": dataset,
                    "Tag": tag_info["tag"],
                    "Count": tag_info["count"],
                    "Frequency": tag_info["frequency"],
                }
            )

    df = pd.DataFrame(rows)

    # Counts table
    counts_table = (
        df.pivot(index="Dataset", columns="Tag", values="Count").fillna(0).astype(int)
    )
    logger.info(f"Tag Counts:\n{counts_table}")

    # Percentages table
    freq_table = df.pivot(index="Dataset", columns="Tag", values="Frequency").fillna(
        0.0
    )
    logger.info(f"Tag Percentages:\n{freq_table.to_string(float_format='%.3f')}")

    logger.info(
        f"\nGenerated {sum(stats['samples_generated'] for stats in all_stats)} total samples"
    )
    logger.info("=" * 60)


def generate_training_windows(args: Args) -> None:
    """Generate training windows from sequence datasets.

    Parameters
    ----------
    args : Args
        Command line arguments containing:
        - input: Path to input Zarr dataset with sequence data
        - output: Path to output Zarr dataset
        - window_size: Size of training windows
        - mask_prop: Maximum proportion of masked sites allowed
        - intergenic_proportion: Target proportion of intergenic windows
    """

    logger.info(f"Loading sequence datasets from {args.input}")
    dt = open_datatree(args.input)

    # Get all groups (species/chromosome combinations)
    groups = list(dt.groups)
    logger.info(f"Found {len(groups)} sequence datasets")

    # Extract species_id and chrom_id from each group
    group_info = []
    for group in groups:
        # Extract species_id and chrom_id from group path (handle leading /)
        parts = group.strip("/").split("/")
        if len(parts) > 2:
            raise ValueError(f"Found invalid group: {group}")
        if len(parts) < 2:
            continue
        species_id, chrom_id = parts
        group_info.append((species_id, chrom_id))

    logger.info(f"Processing {len(group_info)} datasets:")
    for species_id, chrom_id in group_info:
        logger.info(f"  {species_id}/{chrom_id}")

    # Determine number of processes to use
    if args.num_workers is None or args.num_workers == 0:
        n_processes = None
        logger.info("Using sequential processing (no multiprocessing)")
    else:
        n_processes = args.num_workers
        logger.info(f"Using {n_processes} workers for parallel processing")

    # Prepare arguments for worker processes
    task_args = [
        (args, args.output, species_id, chrom_id, i + 1, len(group_info))
        for i, (species_id, chrom_id) in enumerate(group_info)
    ]

    # Process datasets in parallel or sequentially
    logger.info("Starting dataset processing...")
    all_stats = []

    if n_processes is None:
        # Sequential processing
        results = map(_generate_training_windows, task_args)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            results = executor.map(_generate_training_windows, task_args)

    # Collect results
    for stats in results:
        all_stats.append(stats)

    if not all_stats:
        raise ValueError("No valid windows generated")

    # Load and show the final datatree result
    dt = open_datatree(args.output)
    logger.info(f"Final data tree:\n{dt}")

    # Aggregate statistics
    summarize_window_stats(all_stats)

    logger.info("Done")


# -------------------------------------------------------------------------------------------------
# Generate training splits
# -------------------------------------------------------------------------------------------------


def generate_training_splits(args: Args) -> None:
    """Generate training splits from windowed datasets."""
    logger.info(f"Loading windowed datasets from {args.input}")
    dt = open_datatree(args.input)

    datasets = [dt.ds for dt in dt.subtree if dt.is_leaf]
    logger.info(f"Found {len(datasets)} datasets to split")

    rng = np.random.RandomState(args.seed)

    # Track global sample indices for each split
    sample_counters = {args.train_output: 0, args.valid_output: 0}

    for i, ds in enumerate(datasets):
        species_id = ds.attrs["species_id"]
        chrom_id = ds.attrs["chromosome_id"]
        logger.info(
            f"[{species_id}/{chrom_id}] Processing dataset {i + 1}/{len(datasets)}"
        )

        # Random split
        n_samples = ds.sizes["sample"]
        n_valid = int(n_samples * args.valid_proportion)
        indices = rng.permutation(n_samples)

        splits = [
            (indices[n_valid:], args.train_output, "train"),
            (indices[:n_valid], args.valid_output, "valid"),
        ]

        # Log split sizes for this dataset
        logger.info(
            f"  [{species_id}/{chrom_id}] Total samples: {n_samples}, Train: {len(splits[0][0])}, Valid: {len(splits[1][0])}"
        )

        for split_indices, output_path, split_name in splits:
            if len(split_indices) > 0:
                # Assign global sample indices
                start_idx = sample_counters[output_path]
                split_ds = ds.isel(sample=split_indices).assign(
                    sample_index=(
                        "sample",
                        np.arange(start_idx, start_idx + len(split_indices)),
                    )
                )

                # Drop all attrs then only keep the specified ones
                split_ds = split_ds.drop_attrs()
                split_ds.attrs.update(
                    {
                        k: ds.attrs[k]
                        for k in ["feature_labels_classes", "label_classes"]
                        if k in ds.attrs
                    }
                )

                # Apply chunking
                split_ds = set_dimension_chunks(split_ds, "sample", args.chunk_size)

                split_ds.to_zarr(
                    output_path,
                    zarr_format=2,
                    **(
                        dict(append_dim="sample") if os.path.exists(output_path) else {}
                    ),
                )

                # Update counter
                sample_counters[output_path] += len(split_indices)

    # Display final datasets
    for output_path, split_name in [
        (args.train_output, "train"),
        (args.valid_output, "valid"),
    ]:
        if sample_counters[output_path] > 0:
            ds = xr.open_zarr(output_path)
            logger.info(f"Final {split_name} dataset:\n{ds}")

    logger.info("Done")


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Gene Annotation Training Window Generation"
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Command to execute"
    )

    # Generate training windows command
    windows_parser = subparsers.add_parser(
        "generate_training_windows",
        help="Generate training windows from sequence datasets",
    )
    windows_parser.add_argument(
        "--input", required=True, help="Path to input Zarr dataset with sequence data"
    )
    windows_parser.add_argument(
        "--output", required=True, help="Path to output Zarr dataset"
    )
    windows_parser.add_argument(
        "--window-size", type=int, default=8192, help="Size of training windows"
    )
    windows_parser.add_argument(
        "--mask-prop",
        type=float,
        default=0.8,
        help="Maximum proportion of masked sites allowed",
    )
    windows_parser.add_argument(
        "--intergenic-proportion",
        type=float,
        default=0.8,
        help="Proportion of intergenic windows relative to total windows (e.g., 0.5 = equal numbers of intergenic and genic windows, 0.8 = 8 intergenic per 2 genic windows)",
    )
    windows_parser.add_argument(
        "--intergenic-threshold",
        type=float,
        default=0.99,
        help="Threshold for classifying windows as intergenic (windows with intergenic proportion > threshold are considered intergenic)",
    )
    windows_parser.add_argument(
        "--chunk-size", type=int, default=2048, help="Chunk size for sample dimension"
    )
    windows_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for window selection"
    )
    windows_parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of processes to use for parallel processing (each worker uses ~5-6G RAM; default: sequential processing)",
    )

    # Generate training splits command
    splits_parser = subparsers.add_parser(
        "generate_training_splits",
        help="Generate training splits from windowed datasets",
    )
    splits_parser.add_argument(
        "--input", required=True, help="Path to input windowed Zarr dataset"
    )
    splits_parser.add_argument(
        "--train-output", required=True, help="Path to training output Zarr dataset"
    )
    splits_parser.add_argument(
        "--valid-output", required=True, help="Path to validation output Zarr dataset"
    )
    splits_parser.add_argument(
        "--valid-proportion",
        type=float,
        default=0.2,
        help="Proportion of samples for validation",
    )
    splits_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting"
    )
    splits_parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Chunk size for sample dimension; align this to desired chunk sizes for training dataloader",
    )

    args = parser.parse_args()

    if args.command == "generate_training_windows":
        generate_training_windows(args)
    elif args.command == "generate_training_splits":
        generate_training_splits(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
