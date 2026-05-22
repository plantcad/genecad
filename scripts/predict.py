import argparse
import dataclasses
import logging
import os
import tqdm
from typing import Any
from numpy import typing as npt
from src.config import WINDOW_SIZE
from src.sequence import (
    create_sequence_windows,
)
import torch
import numpy as np
import xarray as xr
from argparse import Namespace as Args
from transformers import AutoModel, AutoConfig, AutoTokenizer
from src.dataset import open_datatree, set_dimension_chunks
from src.modeling import GeneClassifier, GeneClassifierConfig
from src.dist import (
    barrier,
    destroy_process_group,
    init_process_group,
    is_main_process,
    local_rank,
    process_group,
)
import torch._dynamo

logger = logging.getLogger(__name__)


def batched(input_list: list[Any], batch_size: int) -> list[list[Any]]:
    """Batches a list into sublists of a specified size."""
    return [
        input_list[i : i + batch_size] for i in range(0, len(input_list), batch_size)
    ]


def _infer_config_from_checkpoint(ckpt: dict, window_size: int) -> GeneClassifierConfig:
    """Reconstruct a GeneClassifierConfig from a checkpoint that lacks saved hyperparameters.

    Reads weight tensor shapes from the state dict to recover the key architectural
    dimensions (hidden size, number of labels, head-encoder depth) that must match
    the saved weights.  Falls back to GeneClassifierConfig defaults for anything
    that cannot be inferred.
    """
    sd = ckpt.get("state_dict", {})
    config = GeneClassifierConfig()
    config.max_sequence_length = window_size

    # hidden_size == fc1 input dim  (fc1 shape: [4*H, H])
    if "classifier.fc1.weight" in sd:
        config.token_embedding_dim = sd["classifier.fc1.weight"].shape[1]

    # num_labels == fc2 output dim  (fc2 shape: [num_labels, 4*H])
    if "classifier.fc2.weight" in sd:
        config.num_labels = sd["classifier.fc2.weight"].shape[0]

    # head_encoder depth from highest layer index present
    layer_indices = {
        int(k.split(".")[2])
        for k in sd
        if k.startswith("head_encoder.layers.") and k.split(".")[2].isdigit()
    }
    if layer_indices:
        config.head_encoder_layers = max(layer_indices) + 1

    logger.info(
        f"Inferred config from checkpoint state dict: "
        f"token_embedding_dim={config.token_embedding_dim}, "
        f"num_labels={config.num_labels}, "
        f"head_encoder_layers={config.head_encoder_layers}, "
        f"max_sequence_length={config.max_sequence_length}"
    )
    return config


def load_classifier(args: Args) -> GeneClassifier:
    logger.info(f"Loading model from {args.model_checkpoint}")
    if not hasattr(torch, args.dtype):
        raise ValueError(f"Invalid torch dtype: {args.dtype}")
    dtype = getattr(torch, args.dtype)

    # If checkpoint is not local, assume it is a Hugging Face path
    model_checkpoint = args.model_checkpoint
    if not os.path.exists(model_checkpoint):
        logger.info(
            f"Local path for classifier {model_checkpoint} not found; attempting Hugging Face download ..."
        )
        from huggingface_hub import hf_hub_download

        model_checkpoint = hf_hub_download(args.model_checkpoint, filename="model.ckpt")
        logger.info(f"Downloaded classifier to {model_checkpoint}")

    # Peek at the checkpoint to check whether hyperparameters were saved.
    # Old checkpoints (and raw Lightning trainer checkpoints) lack this key,
    # causing load_from_checkpoint to fail with a missing-argument TypeError.
    ckpt = torch.load(model_checkpoint, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters") or {}

    if hparams:
        # Prefer the serialized config object saved with the checkpoint.
        # Older checkpoints may still store config values flat, so keep a fallback.
        config = hparams.get("config")
        if isinstance(config, dict):
            config = GeneClassifierConfig(**config)
        elif not isinstance(config, GeneClassifierConfig):
            config_field_names = {
                f.name for f in dataclasses.fields(GeneClassifierConfig)
            }
            config_kwargs = {
                k: v for k, v in hparams.items() if k in config_field_names
            }
            if config_kwargs.get("max_sequence_length") is None:
                config_kwargs["max_sequence_length"] = args.window_size
            config = GeneClassifierConfig(**config_kwargs)
        if config.max_sequence_length is None:
            config.max_sequence_length = args.window_size
        config.base_encoder_path = args.model_path
        model = GeneClassifier.load_from_checkpoint(
            model_checkpoint,
            map_location=args.device,
            strict=False,
            config=config,
        )
    else:
        logger.warning(
            "Checkpoint has no saved hyperparameters (was saved before save_hyperparameters() "
            "was added). Inferring GeneClassifierConfig from state dict shapes."
        )
        config = _infer_config_from_checkpoint(ckpt, window_size=args.window_size)
        config.base_encoder_path = args.model_path
        model = GeneClassifier.load_from_checkpoint(
            model_checkpoint,
            map_location=args.device,
            strict=False,
            config=config,
        )

    model = model.eval()
    model = model.to(args.device, dtype=dtype)
    return model


def load_base_model(args: Args) -> AutoModel:
    logger.info(f"Loading base embedding model from {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    dtype = getattr(torch, args.dtype)
    base_model = AutoModel.from_pretrained(
        args.model_path, config=config, trust_remote_code=True, dtype=dtype
    )
    base_model = base_model.eval()
    base_model = base_model.to(args.device, dtype=dtype)
    return base_model


def load_tokenizer(args: Args) -> AutoTokenizer:
    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    return tokenizer


def load_models(args: Args) -> tuple[AutoModel | None, GeneClassifier, AutoTokenizer]:
    """Load the base model, classifier, and tokenizer required for inference.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments controlling checkpoint paths and device.

    Returns
    -------
    tuple[AutoModel | None, GeneClassifier, AutoTokenizer]
        Tuple containing the optional base model, the classifier, and the tokenizer.
    """
    classifier = load_classifier(args)
    base_model = None
    if classifier.config.use_precomputed_base_encodings:
        base_model = load_base_model(args)
    tokenizer = load_tokenizer(args)
    return base_model, classifier, tokenizer


def load_seq_data(args: Args) -> xr.Dataset:
    """Load the sequence dataset for a specific species and chromosome.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments describing the input path, species, and chromosome.

    Returns
    -------
    xarray.Dataset
        Dataset containing the sequence data for the requested species and chromosome.
    """
    logger.info(f"Opening input sequence datatree from {args.input_zarr}")
    sequences = open_datatree(args.input_zarr, consolidated=False)
    logger.info(f"Input sequences:\n{sequences}")

    # Check if species exists
    if args.species_id not in sequences:
        available_species = list(sequences.keys())
        raise ValueError(
            f"Species '{args.species_id}' not found in input data. Available species: {available_species}"
        )

    # Check if chromosome exists for the specified species
    if args.chromosome_id not in sequences[args.species_id]:
        available_chromosomes = list(sequences[args.species_id].keys())
        raise ValueError(
            f"Chromosome '{args.chromosome_id}' not found for species '{args.species_id}'. Available chromosomes: {available_chromosomes}"
        )

    # Select the dataset for the specified species and chromosome
    logger.info(
        f"Selecting data for species '{args.species_id}' and chromosome '{args.chromosome_id}'"
    )
    ds = sequences[args.species_id][args.chromosome_id].ds

    logger.info(f"Loaded dataset with dimensions: {dict(ds.sizes)}")
    return ds


# -------------------------------------------------------------------------------------------------
# Create predictions
# -------------------------------------------------------------------------------------------------


@torch.inference_mode()
def _create_predictions(
    args: Args,
    ds: xr.Dataset,
    base_model: AutoModel | None,
    classifier: GeneClassifier,
    tokenizer: AutoTokenizer,
) -> xr.DataTree:
    """Generate token and feature predictions in strided windows.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments configuring batch size, stride, and output paths.
    ds : xarray.Dataset
        Dataset containing the sequence input identifiers and metadata.
    base_model : AutoModel or None
        Base encoder used to compute embeddings when required by the classifier.
    classifier : GeneClassifier
        Fine-tuned classifier that produces token logits.
    tokenizer : AutoTokenizer
        Tokenizer associated with the base model.

    Returns
    -------
    xr.DataTree
        Data tree containing predictions for both forward and reverse strands.
    """
    # Get distributed processing info
    rank, world_size = process_group()
    tqdm_position = args.tqdm_position if args.tqdm_position is not None else rank
    gpu_label = str(rank)
    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_gpus:
        visible_gpu_ids = [g.strip() for g in visible_gpus.split(",") if g.strip()]
        # In torchrun, LOCAL_RANK indexes into CUDA_VISIBLE_DEVICES.
        # In single-process runs launched with one visible GPU, index 0 maps
        # to the physical GPU that was selected in predict.sh.
        gpu_index = local_rank() if world_size > 1 else 0
        if 0 <= gpu_index < len(visible_gpu_ids):
            gpu_label = visible_gpu_ids[gpu_index]

    # Construct rank-specific output path
    dataset_path = os.path.join(args.output_dir, f"predictions.{rank}.zarr")

    logger.info(
        f"Generating predictions with {args.batch_size=}, {args.window_size=}, {args.stride=} ({rank=}, {world_size=})"
    )

    # Get padding token ID from tokenizer
    pad_value = tokenizer.unk_token_id
    if pad_value is None:
        raise ValueError("Pad value from tokenizer.unk_token_id cannot be None")
    logger.info(f"Using pad_value={pad_value} (UNK token) for sequence padding")

    # Process data for each strand separately
    token_class_names = classifier.config.token_class_names
    feature_class_names = classifier.config.token_entity_names_with_background()
    num_token_classes = len(token_class_names)
    num_feature_classes = len(feature_class_names)

    strands = ds.strand.values.tolist()
    assert set(strands) == {"positive", "negative"}
    for strand in strands:
        logger.info(f"Processing strand: {strand}")
        negative_strand = strand == "negative"

        # Get sequence input ids for this strand
        sequence_input_ids = ds.sel(strand=strand).sequence_input_ids.values
        assert sequence_input_ids.ndim == 1
        sequence_coordinates = ds.sel(strand=strand).sequence.values
        assert sequence_coordinates.ndim == 1
        # While not strictly necessary, ensure that coordinates are autoincrementing,
        # 0-based integers until there is a good reason to support any other coordinates
        assert sequence_coordinates.tolist() == list(range(len(sequence_coordinates)))

        # Flip token ids on negative strand from 3'->5' to 5'->3'
        if negative_strand:
            sequence_input_ids = flip(sequence_input_ids)
            sequence_coordinates = flip(sequence_coordinates)

        # Create windows of input ids to process
        windows: list[tuple[npt.ArrayLike, tuple[int, int], tuple[int, int]]] = list(
            create_sequence_windows(
                sequence_input_ids,
                window_size=args.window_size,
                stride=args.stride,
                pad_value=pad_value,
            )
        )

        # Select windows for this rank
        # pyrefly: ignore  # bad-assignment
        windows = np.array(windows, dtype=object)
        # pyrefly: ignore  # no-matching-overload
        windows = np.array_split(windows, world_size)[rank]

        # Skip this strand if no windows were assigned to this rank
        # (can happen when the sequence is shorter than world_size windows)
        if len(windows) == 0:
            logger.warning(
                f"Rank {rank}: no windows for strand '{strand}' — skipping "
                f"(sequence may be shorter than window_size × world_size)"
            )
            continue

        # Batch windows together using ceiling division so batch_size is an
        # upper bound, not a lower bound.  e.g. 195 windows / batch 112 → 2
        # batches of 98 and 97, not 1 batch of 195.
        n_batches = max(1, (len(windows) + args.batch_size - 1) // args.batch_size)
        window_batches = np.array_split(windows, n_batches)
        logger.info(
            f"Processing {len(windows)} windows in {len(window_batches)} batches of size {args.batch_size}"
        )

        # Process batches — each rank occupies its own tqdm row so bars don't
        # overwrite each other when world_size > 1.
        for (
            batch_index,
            window_batch,
        ) in enumerate(  # pyrefly: ignore[bad-argument-type]
            tqdm.tqdm(
                window_batches,
                desc=f"[GPU {gpu_label} | {strand}]",
                position=tqdm_position,
                leave=False,
                dynamic_ncols=True,
            )
        ):
            current_batch_size = len(window_batch)

            # Get equally sized sequence windows to process for batch
            input_ids = np.array([w[0] for w in window_batch])
            input_ids = torch.tensor(input_ids, device=args.device)
            assert input_ids.shape == (current_batch_size, args.window_size)

            # Generate embeddings, if necessary
            inputs_embeds = None
            if classifier.config.use_precomputed_base_encodings:
                # pyrefly: ignore  # not-callable
                inputs_embeds = base_model(input_ids=input_ids).last_hidden_state
                assert inputs_embeds.ndim == 3
                assert inputs_embeds.shape[:2] == (current_batch_size, args.window_size)

            # Get predictions from classifier
            # pyrefly: ignore  # not-callable
            token_logits = classifier(input_ids=input_ids, inputs_embeds=inputs_embeds)
            assert token_logits.shape == (
                current_batch_size,
                args.window_size,
                num_token_classes,
            )

            # Aggregate token logits to entity/feature logits
            feature_logits = classifier.aggregate_logits(token_logits)
            assert feature_logits.shape == (
                current_batch_size,
                args.window_size,
                num_feature_classes,
            )

            token_logits = token_logits.float().cpu().numpy()
            feature_logits = feature_logits.float().cpu().numpy()

            # Extract valid regions from the processed windows
            token_logits_arrays, feature_logits_arrays, sequence_coord_arrays = (
                [],
                [],
                [],
            )
            for i in range(current_batch_size):
                _, local_window, global_window = window_batch[i]
                token_logits_window = token_logits[
                    i, local_window[0] : local_window[1], :
                ]
                feature_logits_window = feature_logits[
                    i, local_window[0] : local_window[1], :
                ]
                # pyrefly: ignore  # index-error
                sequence_coords_window = sequence_coordinates[
                    # pyrefly: ignore  # index-error
                    global_window[0] : global_window[1]
                ]
                token_logits_arrays.append(token_logits_window)
                feature_logits_arrays.append(feature_logits_window)
                sequence_coord_arrays.append(sequence_coords_window)

            # Concatenate all extracted regions
            token_logits = np.concatenate(token_logits_arrays, axis=0)
            feature_logits = np.concatenate(feature_logits_arrays, axis=0)
            sequence_coords = np.concatenate(sequence_coord_arrays, axis=0)

            # Flip back to 3'->5' if on negative strand
            if negative_strand:
                token_logits = flip(token_logits)
                feature_logits = flip(feature_logits)
                sequence_coords = flip(sequence_coords)

            # Create resulting dataset for batch
            result = xr.Dataset(
                data_vars={
                    "token_logits": (["sequence", "token"], token_logits),
                    "feature_logits": (["sequence", "feature"], feature_logits),
                },
                coords={
                    "sequence": sequence_coords,
                    "token": token_class_names,
                    "feature": feature_class_names,
                },
                attrs={
                    "strand": strand,
                    "species_id": args.species_id,
                    "chromosome_id": args.chromosome_id,
                    "model_checkpoint": args.model_checkpoint,
                    "model_path": args.model_path,
                },
            )

            # Assign predictions as max logits
            result["token_predictions"] = result.token_logits.argmax(dim="token")
            result["feature_predictions"] = result.feature_logits.argmax(dim="feature")

            # Chunk in sequence dim only and save
            result = set_dimension_chunks(result, "sequence", result.sizes["sequence"])
            os.makedirs(args.output_dir, exist_ok=True)
            # pyrefly: ignore  # no-matching-overload
            result.to_zarr(
                dataset_path,
                group=f"/{strand}",
                zarr_format=2,
                **(
                    dict(append_dim="sequence")
                    if os.path.exists(os.path.join(dataset_path, strand))
                    else {}
                ),
                consolidated=True,
            )
    logger.info(
        f"Loading completed predictions from {dataset_path} ({rank=}, {world_size=})"
    )
    result = open_datatree(dataset_path)
    return result


# TODO: move to utils somewhere
def flip(sequence: npt.ArrayLike) -> npt.ArrayLike:
    """Reverse a sequence along its first axis."""
    return np.flip(sequence, axis=0)


@torch.inference_mode()
def warmup_triton(args: Args):
    """Pre-warm the Triton autotune cache by running a dummy forward pass.

    Triton compiles and benchmarks GPU kernels the first time a new tensor
    shape is seen.  When two torchrun ranks encounter the same new shape
    simultaneously the concurrent CUDA benchmarks can collide and produce an
    illegal memory access.

    This function runs a single dummy forward pass **without** initialising the
    distributed process group, so it must be called once per GPU sequentially
    (before the multi-GPU torchrun launch).  The Triton cache persists across
    processes, so the actual torchrun run will find pre-built configs and skip
    the autotuning step entirely.

    Parameters
    ----------
    args : argparse.Namespace
        Must supply ``model_path``, ``model_checkpoint``, ``dtype``,
        ``batch_size``, and ``window_size``.
    """
    # TODO: allow other CUDA device if available
    if torch.cuda.is_available():
        args.device = "cuda:0"
    else:
        args.device = "cpu"

    logger.info(
        f"[warmup] device={args.device}  batch_size={args.batch_size}  window_size={args.window_size}"
    )

    torch.set_float32_matmul_precision("medium")
    if args.suppress_dynamo_errors == "yes":
        torch._dynamo.config.suppress_errors = True

    base_model, classifier, tokenizer = load_models(args)

    pad_value = tokenizer.unk_token_id
    if pad_value is None:
        pad_value = 0

    # Warm up with either the exact requested probe batch or a conservative
    # capped batch when no explicit probe size was provided.
    warmup_batch_size = getattr(args, "warmup_batch_size", None)
    if warmup_batch_size is None:
        warmup_batch_size = max(1, min(args.batch_size, 4))
        if warmup_batch_size != args.batch_size:
            logger.info(
                f"[warmup] Capping probe batch from {args.batch_size} to {warmup_batch_size} for safety"
            )
    else:
        warmup_batch_size = max(1, warmup_batch_size)
        logger.info(f"[warmup] Using exact probe batch size {warmup_batch_size}")

    dummy = torch.full(
        (warmup_batch_size, args.window_size),
        pad_value,
        dtype=torch.long,
        device=args.device,
    )

    if base_model is not None:
        embeds = base_model(input_ids=dummy).last_hidden_state
    else:
        embeds = None

    classifier(input_ids=dummy, inputs_embeds=embeds)  # pyrefly: ignore[not-callable]

    # Also warm up remainder-batch shape (1 sample) so the last batch in each
    # strand doesn't trigger a new autotuning run.
    if warmup_batch_size > 1:
        dummy1 = dummy[:1]
        embeds1 = (
            base_model(input_ids=dummy1).last_hidden_state
            if base_model is not None
            else None
        )
        classifier(
            input_ids=dummy1, inputs_embeds=embeds1
        )  # pyrefly: ignore[not-callable]

    logger.info("[warmup] Triton cache warm-up complete.")


def create_predictions(args: Args):
    """Run the inference pipeline to generate logits for each genomic strand.

    When launched via ``torchrun`` (multi-GPU), every rank initialises the
    distributed process group, binds to its own GPU (``LOCAL_RANK``), and
    processes a disjoint slice of the sequence windows.  All ranks write
    their own shard zarr file (``predictions.<rank>.zarr``).  After a barrier
    ensures every shard has been flushed to disk the process group is torn
    down cleanly.  The downstream ``detect_intervals`` step merges the shards
    transparently via :func:`~src.prediction.merge_prediction_datasets`.

    When launched normally (single-GPU), the function behaves exactly as
    before: rank 0, world size 1.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments controlling inputs, outputs, and runtime options.
    """
    # ---- Distributed setup ------------------------------------------------
    # No-op when not launched by torchrun (RANK env var absent).
    init_process_group()

    # Bind this process to its LOCAL_RANK GPU so tensor ops land on the
    # correct device.  torchrun sets LOCAL_RANK; single-process runs keep
    # the device that was passed via --device.
    if torch.cuda.is_available():
        lr = local_rank()
        torch.cuda.set_device(lr)
        args.device = f"cuda:{lr}"
    else:
        raise RuntimeError("No CUDA GPUs are available")

    rank, world_size = process_group()
    logger.info(
        f"create_predictions starting: {rank=}, {world_size=}, device={args.device}"
    )

    # ---- PyTorch settings -------------------------------------------------
    # Set to avoid:
    # UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication
    # available but not enabled.
    torch.set_float32_matmul_precision("medium")  # same setting as training

    # Suppress errors related to models trained with torch.compile, e.g.:
    # AssertionError: increase TRITON_MAX_BLOCK['X'] to 4096
    # https://github.com/pytorch/pytorch/issues/135028#issuecomment-2330421513
    if not args.show_dynamo_errors:
        torch._dynamo.config.suppress_errors = True

    # ---- Build work list --------------------------------------------------
    # Manifest mode: process many sequences with one model load.
    # Single mode: original behaviour (one chromosome per invocation).
    if args.manifest is not None:
        import json

        with open(args.manifest) as fh:
            entries = json.load(fh)
        logger.info(f"Manifest mode: {len(entries)} sequence(s) to process")
    else:
        entries = [
            {
                "chromosome_id": args.chromosome_id,
                "input_zarr": args.input_zarr,
                "output_dir": args.output_dir,
            }
        ]

    # ---- Inference --------------------------------------------------------
    try:
        # Load models onto this rank's device — happens ONCE regardless of
        # how many sequences are in the manifest.
        base_model, classifier, tokenizer = load_models(args)

        for i, entry in enumerate(entries):
            args.chromosome_id = entry["chromosome_id"]
            args.input_zarr = entry["input_zarr"]
            args.output_dir = entry["output_dir"]

            logger.info(
                f"[{i + 1}/{len(entries)}] Running predictions for "
                f"{args.species_id}/{args.chromosome_id}"
            )
            _create_predictions(
                args, load_seq_data(args), base_model, classifier, tokenizer
            )

            # All ranks must finish this sequence before moving to the next.
            barrier()

        if is_main_process():
            logger.info(
                f"All {len(entries)} sequence(s) done across {world_size} rank(s)."
            )
    finally:
        # Always tear down the process group even if an exception occurred so
        # that NCCL resources are released and port numbers are freed.
        destroy_process_group()

    logger.info("Done")


def main():
    """Main entry point for prediction and export functions."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Suppress noisy HTTP traffic logs from HuggingFace Hub's internal HTTP client
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Gene prediction and export tools")
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Command to execute"
    )

    # Run inference command
    inference_parser = subparsers.add_parser(
        "create_predictions",
        help="Generate token and feature logits with predicted classes",
    )
    inference_parser.add_argument(
        "--input-zarr",
        "-i",
        default=None,
        help="Path to input zarr dataset (from transform.py). "
        "Not required when --manifest is used.",
    )
    inference_parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Directory to save rank-specific output zarr datasets. "
        "Not required when --manifest is used.",
    )
    inference_parser.add_argument(
        "--manifest",
        default=None,
        help="Path to a JSON file listing sequences to process in one model-load session. "
        'Format: [{"chromosome_id": "...", "input_zarr": "...", "output_dir": "..."}, ...]. '
        "Enables processing thousands of short sequences without reloading the model each time.",
    )
    inference_parser.add_argument(
        "--model-checkpoint", required=True, help="Path to classifier checkpoint"
    )
    inference_parser.add_argument(
        "--model-path", required=True, help="Path to base embedding model"
    )

    # Selection arguments
    inference_parser.add_argument(
        "--species-id", required=True, help="Species ID to process (e.g., 'Osativa')"
    )
    inference_parser.add_argument(
        "--chromosome-id",
        default=None,
        help="Chromosome ID to process (e.g., 'Chr1'). Not required when --manifest is used.",
    )

    # Processing parameters
    inference_parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help="Window size for sequence processing",
    )
    inference_parser.add_argument(
        "--stride",
        type=int,
        default=WINDOW_SIZE // 2,
        help="Stride size for overlapping windows",
    )
    # TODO: change? default keeps crashing
    inference_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference. Default 16.",
    )
    inference_parser.add_argument(
        "--tqdm-position",
        type=int,
        default=None,
        help="Optional tqdm row to use for this process when multiple GPU jobs run in one terminal",
    )
    inference_parser.add_argument(
        "--show-dynamo-errors",
        action="store_false",
        help="Show torch dynamo errors (suppressed by default)",
    )
    # TODO: could this be determined by model config?
    inference_parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16", "float64", "double", "half"],
        help="Data type for model inference (default: bfloat16)",
    )

    # Triton warmup command — run once per GPU BEFORE multi-GPU torchrun launch
    warmup_parser = subparsers.add_parser(
        "warmup",
        help="Pre-warm Triton autotune cache with a dummy forward pass (run sequentially per GPU before multi-GPU launch)",
    )
    warmup_parser.add_argument(
        "--model-checkpoint", required=True, help="Path to classifier checkpoint"
    )
    warmup_parser.add_argument(
        "--model-path", required=True, help="Path to base embedding model"
    )
    warmup_parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help="Window size used during inference (must match actual run)",
    )
    warmup_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used during inference (must match actual run)",
    )
    warmup_parser.add_argument(
        "--warmup-batch-size",
        type=int,
        default=None,
        help="Exact batch size to probe during warmup (overrides conservative capping)",
    )
    warmup_parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16", "float64", "double", "half"],
        help="Data type matching the actual inference run",
    )
    warmup_parser.add_argument(
        "--suppress-dynamo-errors",
        type=str,
        choices=["yes", "no"],
        default="yes",
        help="Whether to suppress torch dynamo errors (default: yes)",
    )

    args = parser.parse_args()

    if args.command == "create_predictions":
        create_predictions(args)
    elif args.command == "warmup":
        warmup_triton(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
