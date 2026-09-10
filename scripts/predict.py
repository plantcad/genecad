import argparse
import dataclasses
import logging
import os
import hashlib
import gc
from time import perf_counter
from contextlib import nullcontext
import tqdm
from typing import Any
from numpy import typing as npt
from zarr.errors import GroupNotFoundError
import json
from src.atomic_io import atomic_output_path
from src.config import WINDOW_SIZE
from src.sequence import (
    create_sequence_windows,
    create_index_windows,
)
import torch
import numpy as np
import xarray as xr
from transformers import AutoModel, AutoConfig, AutoTokenizer
from src.dataset import open_datatree, set_dimension_chunks
from src.prediction_checkpoint import (
    commit_segment,
    file_hashes,
    fingerprint,
    prepare_run,
    segments,
    finish_run,
    pending_batches,
    prediction_lock,
    PredictionResumeError,
)
from src.modeling import GeneClassifier, GeneClassifierConfig
from src.dist import (
    barrier,
    destroy_process_group,
    guarded_on_main,
    init_process_group,
    is_main_process,
    local_rank,
    process_group,
)
import torch._dynamo

logger = logging.getLogger(__name__)


def prediction_model_digest(base_model, classifier, tokenizer) -> str:
    digest = hashlib.sha256()
    for label, model in (("base", base_model), ("classifier", classifier)):
        digest.update(label.encode())
        if model is None:
            continue
        config = (
            dataclasses.asdict(model.config)
            if dataclasses.is_dataclass(model.config)
            else model.config.to_dict()
        )
        digest.update(json.dumps(config, sort_keys=True, default=str).encode())
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            flat = tensor.detach().reshape(-1)
            for start in range(0, flat.numel(), 1_000_000):
                chunk = flat[start : start + 1_000_000].contiguous().cpu()
                digest.update(chunk.view(torch.uint8).numpy().tobytes())
    digest.update(fingerprint(tokenizer.get_vocab()).encode())
    digest.update(str(tokenizer.unk_token_id).encode())
    return digest.hexdigest()


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


def load_classifier(
    model_checkpoint: str,
    model_path: str | None,
    window_size: int,
    dtype: str,
    device: str,
) -> GeneClassifier:
    logger.info(f"Loading model from {model_checkpoint}")
    if not hasattr(torch, dtype):
        raise ValueError(f"Invalid torch dtype: {dtype}")
    dtype = getattr(torch, dtype)

    base_model_path = model_path

    # If checkpoint is not local, assume it is a Hugging Face path
    # TODO: as it stands, we do not attempt to infer base model from local checkpoints. This should be added.
    if not os.path.exists(model_checkpoint):
        logger.info(
            f"Local path for classifier {model_checkpoint} not found; attempting Hugging Face download ..."
        )
        from huggingface_hub import hf_hub_download

        checkpoint_local = hf_hub_download(model_checkpoint, filename="model.ckpt")

        # Infer encoder model path using config json
        if base_model_path is None:
            config_local = hf_hub_download(model_checkpoint, filename="config.json")
            with open(config_local) as file:
                config_dict = json.load(file)

            if "base_encoder_path" in config_dict.keys():
                base_model_path = config_dict["base_encoder_path"]

        logger.info(f"Downloaded classifier to {model_checkpoint}")
    else:
        checkpoint_local = model_checkpoint

    if base_model_path is None:
        logger.error(
            "Base model path cannot be inferred from classifier config. Please specify --model-path"
        )
        raise FileNotFoundError

    # Peek at the checkpoint to check whether hyperparameters were saved.
    # Old checkpoints (and raw Lightning trainer checkpoints) lack this key,
    # causing load_from_checkpoint to fail with a missing-argument TypeError.
    ckpt = torch.load(checkpoint_local, map_location="cpu", weights_only=False)
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
                config_kwargs["max_sequence_length"] = window_size
            config = GeneClassifierConfig(**config_kwargs)
        if config.max_sequence_length is None:
            config.max_sequence_length = window_size
        config.base_encoder_path = base_model_path
        model = GeneClassifier.load_from_checkpoint(
            checkpoint_local,
            map_location=device,
            strict=False,
            config=config,
        )
    else:
        logger.warning(
            "Checkpoint has no saved hyperparameters (was saved before save_hyperparameters() "
            "was added). Inferring GeneClassifierConfig from state dict shapes."
        )
        config = _infer_config_from_checkpoint(ckpt, window_size=window_size)
        config.base_encoder_path = base_model_path
        model = GeneClassifier.load_from_checkpoint(
            checkpoint_local,
            map_location=device,
            strict=False,
            config=config,
        )

    model = model.eval()
    model = model.to(device, dtype=dtype)
    return model


def load_base_model(model_path: str, dtype: str, device: str) -> AutoModel:
    """Loads PlantCAD Encoder model"""

    logger.info(f"Loading base embedding model from {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    dtype = getattr(torch, dtype)
    base_model = AutoModel.from_pretrained(
        model_path, config=config, trust_remote_code=True, dtype=dtype
    )
    base_model = base_model.eval()
    base_model = base_model.to(device, dtype=dtype)
    return base_model


def load_tokenizer(model_path) -> AutoTokenizer:
    """Loads PlantCAD tokenizer"""
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer


def load_models(
    model_checkpoint: str,
    model_path: str | None,
    window_size: int,
    dtype: str,
    device: str,
) -> tuple[AutoModel | None, GeneClassifier, AutoTokenizer]:
    """Load the base model, classifier, and tokenizer required for inference.

    Parameters
    ----------
    model_checkpoint : str
        Path (local or HuggingFace) to the GeneCAD model checkpoint
    model_path : str | None
        Path (local or HuggingFace) to the base PlantCAD Encoder model. If none, an attempt will
        be made to infer model_path from model_checkpoint config.
    window_size : int
        Size of the PlantCAD context window
    dtype: int
        Input data type
    device: str
        Device on which to run inference. Note that PlantCAD requires a CUDA GPU and cannot be run on CPU

    Returns
    -------
    tuple[AutoModel | None, GeneClassifier, AutoTokenizer]
        Tuple containing the optional base model, the classifier, and the tokenizer.
    """
    classifier = load_classifier(
        model_checkpoint=model_checkpoint,
        model_path=model_path,
        window_size=window_size,
        dtype=dtype,
        device=device,
    )
    base_model = None
    if classifier.config.use_precomputed_base_encodings:
        base_model = load_base_model(
            model_path=classifier.config.base_encoder_path, device=device, dtype=dtype
        )
    tokenizer = load_tokenizer(classifier.config.base_encoder_path)
    return base_model, classifier, tokenizer


def load_seq_data(input_zarr: str, chromosome_id: str, species_id: str) -> xr.Dataset:
    """Load the sequence dataset for a specific species and chromosome.

    Parameters
    ----------
    input_zarr: str
        Path to the input sequence data, in .zarr format
    chromosome_id: str
        Name of the chromosome in the dataset
    species_id: str
        Name of the species in the dataset

    Returns
    -------
    xarray.Dataset
        Dataset containing the sequence data for the requested species and chromosome.
    """
    try:
        logger.info(f"Opening input sequence datatree from {input_zarr}")
        sequences = open_datatree(input_zarr, consolidated=False)
        logger.info(f"Input sequences:\n{sequences}")
    except GroupNotFoundError as e:
        logger.error(
            f"File {input_zarr} is not formatted as an input sequence datatree. \n"
            f"Check that it is the output of scripts/extract_fasta.py"
        )
        raise ValueError(f"Invalid sequence store: {input_zarr}") from e

    # Check if species exists
    if species_id not in sequences:
        available_species = list(sequences.keys())
        if available_species == ["sequences", "intervals"]:
            raise ValueError(
                f"{input_zarr} is an intervals file, not a sequence file. \n"
                f"Check that it is the output of the command extract.py extract_fasta_file"
            )
        else:
            raise ValueError(
                f"Species '{species_id}' not found in input data. Available species: {available_species}"
            )

    # Check if chromosome exists for the specified species
    if chromosome_id not in sequences[species_id]:
        available_chromosomes = list(sequences[species_id].keys())
        raise ValueError(
            f"Chromosome '{chromosome_id}' not found for species '{species_id}'. Available chromosomes: "
            f"{available_chromosomes} "
        )

    # Select the dataset for the specified species and chromosome
    logger.info(
        f"Selecting data for species '{species_id}' and chromosome '{chromosome_id}'"
    )
    ds = sequences[species_id][chromosome_id].to_dataset()
    ds.set_close(sequences.close)

    logger.info(f"Loaded dataset with dimensions: {dict(ds.sizes)}")
    return ds


# -------------------------------------------------------------------------------------------------
# Create predictions
# -------------------------------------------------------------------------------------------------


@torch.inference_mode()
def _predict_window_batch(
    window_batch,
    sequence_coordinates,
    base_model,
    classifier,
    window_size,
    device,
    strand,
    species_id,
    chromosome_id,
    model_checkpoint,
    model_path,
) -> xr.Dataset:
    token_class_names = classifier.config.token_class_names
    feature_class_names = classifier.config.token_entity_names_with_background()
    num_token_classes = len(token_class_names)
    num_feature_classes = len(feature_class_names)
    negative_strand = strand == "negative"
    current_batch_size = len(window_batch)

    # Get equally sized sequence windows to process for batch
    input_ids = np.array([w[0] for w in window_batch])
    input_ids = torch.tensor(input_ids, device=device)
    assert input_ids.shape == (current_batch_size, window_size)

    # Generate embeddings, if necessary
    inputs_embeds = None
    if classifier.config.use_precomputed_base_encodings:
        # pyrefly: ignore  # not-callable
        inputs_embeds = base_model(input_ids=input_ids).last_hidden_state
        assert inputs_embeds.ndim == 3
        assert inputs_embeds.shape[:2] == (current_batch_size, window_size)

    # Get predictions from classifier
    # pyrefly: ignore  # not-callable
    token_logits = classifier(input_ids=input_ids, inputs_embeds=inputs_embeds)
    assert token_logits.shape == (
        current_batch_size,
        window_size,
        num_token_classes,
    )

    # Aggregate token logits to entity/feature logits
    feature_logits = classifier.aggregate_logits(token_logits)
    assert feature_logits.shape == (
        current_batch_size,
        window_size,
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
        token_logits_window = token_logits[i, local_window[0] : local_window[1], :]
        feature_logits_window = feature_logits[i, local_window[0] : local_window[1], :]
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
            "species_id": species_id,
            "chromosome_id": chromosome_id,
            "model_checkpoint": model_checkpoint,
            "model_path": model_path,
        },
    )

    # Assign predictions as max logits
    result["token_predictions"] = result.token_logits.argmax(dim="token")
    result["feature_predictions"] = result.feature_logits.argmax(dim="feature")

    # Chunk in sequence dim only and save
    result = set_dimension_chunks(result, "sequence", result.sizes["sequence"])
    return result


@torch.inference_mode()
def _create_predictions(
    ds: xr.Dataset,
    base_model: AutoModel | None,
    classifier: GeneClassifier,
    tokenizer: AutoTokenizer,
    species_id: str,
    chromosome_id: str,
    model_checkpoint: str,
    model_path: str,
    output_dir: str,
    batch_size: int,
    window_size: int,
    stride: int,
    device: str,
    tqdm_position: int | None,
) -> int:
    """Generate token and feature predictions in strided windows.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the sequence input identifiers and metadata.
    base_model : AutoModel or None
        Base encoder used to compute embeddings when required by the classifier.
    classifier : GeneClassifier
        Fine-tuned classifier that produces token logits.
    tokenizer : AutoTokenizer
        Tokenizer associated with the base model.
    species_id : str
        Species name
    chromosome_id : str
        Chromosome or contig name
    model_checkpoint: str
        Path (local or HuggingFace) to the GeneCAD model checkpoint
    model_path: str
        Path (local or HuggingFace) to the PlantCAD base model
    output_dir: str
        Path to the output directory
    batch_size: int
        Inference batch size
    window_size: int
        PlantCAD context window size
    stride: int
        Distance between the start position of each input window
    device: str
        Device on which to run inference
    tqdm_position: int | None
        optional tqdm row to use

    Completed batches are committed to independent Zarr segments in output_dir.
    """
    # Get distributed processing info
    rank, world_size = process_group()
    tqdm_position = tqdm_position if tqdm_position is not None else rank
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

    # Snapshot progress on every rank before any rank starts writing.
    completed = segments(output_dir, verify=False)
    barrier()

    logger.info(
        f"Generating predictions with {batch_size=}, {window_size=}, {stride=} ({rank=}, {world_size=})"
    )

    # Get padding token ID from tokenizer
    pad_value = tokenizer.unk_token_id
    if pad_value is None:
        raise ValueError("Pad value from tokenizer.unk_token_id cannot be None")
    logger.info(f"Using pad_value={pad_value} (UNK token) for sequence padding")

    # Keep an OOM-reduced cap across strands and return it for the next scaffold.
    effective_batch_size = batch_size

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
        if not np.array_equal(
            sequence_coordinates, np.arange(len(sequence_coordinates))
        ):
            raise ValueError("Sequence coordinates must be contiguous and zero-based")

        # Flip token ids on negative strand from 3'->5' to 5'->3'
        if negative_strand:
            sequence_input_ids = flip(sequence_input_ids)
            sequence_coordinates = flip(sequence_coordinates)

        # Generate window batches lazily; the padded strand remains in RAM.
        padded_length = (
            (len(sequence_input_ids) + window_size - 1) // window_size
        ) * window_size
        bounds = create_index_windows(padded_length, window_size, stride)
        total_windows = int(np.count_nonzero(bounds[:, 2] < len(sequence_input_ids)))
        windows = create_sequence_windows(
            sequence_input_ids,
            window_size=window_size,
            stride=stride,
            pad_value=pad_value,
        )
        batches = pending_batches(
            windows,
            completed,
            strand,
            rank,
            world_size,
            effective_batch_size,
            total_windows,
        )
        for window_ids, window_batch in tqdm.tqdm(
            batches,
            desc=f"[GPU {gpu_label} | {strand}]",
            position=tqdm_position,
            leave=False,
            dynamic_ncols=True,
        ):
            offset = 0
            while offset < len(window_batch):
                current = window_batch[offset : offset + effective_batch_size]
                ids = window_ids[offset : offset + effective_batch_size]
                try:
                    result = _predict_window_batch(
                        current,
                        sequence_coordinates,
                        base_model,
                        classifier,
                        window_size,
                        device,
                        strand,
                        species_id,
                        chromosome_id,
                        model_checkpoint,
                        model_path,
                    )
                except torch.cuda.OutOfMemoryError:
                    if len(current) == 1:
                        raise
                    effective_batch_size = max(1, len(current) * 4 // 5)
                    logger.warning(
                        "CUDA OOM for %s/%s (%s); retrying uncommitted windows with batch=%d, keeping models loaded",
                        species_id,
                        chromosome_id,
                        strand,
                        effective_batch_size,
                    )
                    result = None
                # Release the exception traceback before clearing GPU memory.
                if result is None:
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                commit_segment(output_dir, result, strand, ids)
                offset += len(current)
                del result
    logger.info(
        f"Finished assigned prediction windows in {output_dir} ({rank=}, batch={effective_batch_size})"
    )
    return effective_batch_size


# TODO: move to utils somewhere
def flip(sequence: npt.ArrayLike) -> npt.ArrayLike:
    """Reverse a sequence along its first axis."""
    return np.flip(sequence, axis=0)


@torch.inference_mode()
def warmup_triton(
    model_checkpoint: str,
    model_path: str | None,
    window_size: int,
    batch_size: int,
    warmup_batch_size: int | None,
    dtype: str,
    suppress_dynamo_errors: bool,
):
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
    model_checkpoint: str
        Path (local or HuggingFace) to the GeneCAD checkpoint.
    model_path: str | None
        Path (local or HuggingFace) to the base PlantCAD encoder model. Inferred from model_checkpoint if not provided
    window_size: int
        Size of the input context window
    batch_size: int
        batch size
    warmup_batch_size: int | None
        size of the warmup batch. Inferred if not provided
    dtype: str
        Data type for sequence input
    suppress_dynamo_errors : bool
        Suppress dynamo error messages in output
    """
    # TODO: allow other CUDA device if available
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    logger.info(
        f"[warmup] device={device}  batch_size={batch_size}  window_size={window_size}"
    )

    torch.set_float32_matmul_precision("medium")
    if suppress_dynamo_errors:
        torch._dynamo.config.suppress_errors = True

    base_model, classifier, tokenizer = load_models(
        model_checkpoint=model_checkpoint,
        model_path=model_path,
        window_size=window_size,
        dtype=dtype,
        device=device,
    )

    pad_value = tokenizer.unk_token_id
    if pad_value is None:
        pad_value = 0

    # Warm up with either the exact requested probe batch or a conservative
    # capped batch when no explicit probe size was provided.
    if warmup_batch_size is None:
        warmup_batch_size = max(1, min(batch_size, 4))
        if warmup_batch_size != batch_size:
            logger.info(
                f"[warmup] Capping probe batch from {batch_size} to {warmup_batch_size} for safety"
            )
    else:
        warmup_batch_size = max(1, warmup_batch_size)
        logger.info(f"[warmup] Using exact probe batch size {warmup_batch_size}")

    dummy = torch.full(
        (warmup_batch_size, window_size),
        pad_value,
        dtype=torch.long,
        device=device,
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


def create_predictions(
    manifest: str | None,
    species_id: str,
    chromosome_id: str | None,
    input_zarr: str | None,
    output_dir: str | None,
    model_checkpoint: str,
    model_path: str | None,
    window_size: int,
    stride: int,
    batch_size: int,
    dtype: str,
    tqdm_position: int | None,
    show_dynamo_errors: bool,
    batch_size_cache: str | None = None,
):
    """Run the inference pipeline to generate logits for each genomic strand.

    When launched via ``torchrun`` (multi-GPU), every rank initialises the
    distributed process group, binds to its own GPU (``LOCAL_RANK``), and
    processes a disjoint slice of the sequence windows. Ranks commit independent
    batch segments, whose window IDs survive changes to GPU count and batch
    size. After a barrier, rank zero verifies complete coverage and writes the
    chromosome completion manifest.  The downstream ``detect_intervals`` step merges the shards
    transparently via :func:`~src.prediction.merge_prediction_datasets`.

    When launched normally (single-GPU), the function behaves exactly as
    before: rank 0, world size 1.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments controlling inputs, outputs, and runtime options.
    """
    # ---- Build work list --------------------------------------------------
    # Manifest mode: process many sequences with one model load.
    # Single mode: original behaviour (one chromosome per invocation).
    if manifest is not None:
        import json

        with open(manifest) as fh:
            entries = json.load(fh)
        logger.info(f"Manifest mode: {len(entries)} sequence(s) to process")
    else:
        entries = [
            {
                "chromosome_id": chromosome_id,
                "sequence_zarr": input_zarr,
                "predictions_dir": output_dir,
            }
        ]

    # Empty manifests must not load models or probe CUDA memory.
    if not entries:
        logger.info("No scaffolds need prediction")
        return
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if batch_size_cache:
        try:
            with open(batch_size_cache) as handle:
                cached = int(handle.read().strip())
            if cached > 0:
                batch_size = min(batch_size, cached)
        except (OSError, ValueError):
            pass

    # ---- Distributed setup ------------------------------------------------
    # No-op when not launched by torchrun (RANK env var absent).
    init_process_group()

    # Bind this process to its LOCAL_RANK GPU so tensor ops land on the
    # correct device.  torchrun sets LOCAL_RANK; single-process runs keep
    # the device that was passed via --device.
    if torch.cuda.is_available():
        lr = local_rank()
        torch.cuda.set_device(lr)
        device = f"cuda:{lr}"
    else:
        raise RuntimeError("No CUDA GPUs are available")

    rank, world_size = process_group()
    logger.info(f"create_predictions starting: {rank=}, {world_size=}, device={device}")

    # ---- PyTorch settings -------------------------------------------------
    # Set to avoid:
    # UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication
    # available but not enabled.
    torch.set_float32_matmul_precision("medium")  # same setting as training

    # Suppress errors related to models trained with torch.compile, e.g.:
    # AssertionError: increase TRITON_MAX_BLOCK['X'] to 4096
    # https://github.com/pytorch/pytorch/issues/135028#issuecomment-2330421513
    if not show_dynamo_errors:
        torch._dynamo.config.suppress_errors = True

    # ---- Inference --------------------------------------------------------
    try:
        model_start = perf_counter()
        # Load models onto this rank's device — happens ONCE regardless of
        # how many sequences are in the manifest.
        base_model, classifier, tokenizer = load_models(
            model_checkpoint=model_checkpoint,
            model_path=model_path,
            window_size=window_size,
            dtype=dtype,
            device=device,
        )

        logger.info(
            "Model load completed in %.2fs; reusing this model for %d scaffold(s)",
            perf_counter() - model_start,
            len(entries),
        )
        identity_start = perf_counter()
        # Hash loaded weights once per worker, in bounded CPU chunks, to detect
        # checkpoints replaced at the same path.
        model_digest = None
        with guarded_on_main():
            if is_main_process():
                model_digest = prediction_model_digest(
                    base_model, classifier, tokenizer
                )

        logger.info(
            "Model fingerprint completed in %.2fs", perf_counter() - identity_start
        )
        for i, entry in enumerate(entries):
            scaffold_start = perf_counter()
            chromosome_id = entry["chromosome_id"]
            input_zarr = entry["sequence_zarr"]
            final_output_dir = entry["predictions_dir"]
            with (
                prediction_lock(final_output_dir)
                if is_main_process()
                else nullcontext()
            ):
                ds = load_seq_data(
                    input_zarr=input_zarr,
                    chromosome_id=chromosome_id,
                    species_id=species_id,
                )
                try:
                    with guarded_on_main():
                        if is_main_process():
                            identity = {
                                "input": fingerprint(file_hashes(input_zarr)),
                                "model": model_digest,
                                "species_id": species_id,
                                "chromosome_id": chromosome_id,
                                "window_size": window_size,
                                "stride": stride,
                                "dtype": dtype,
                                "torch": torch.__version__,
                            }
                            prepare_run(
                                final_output_dir, identity, ds.sizes["sequence"]
                            )

                    logger.info(
                        f"[{i + 1}/{len(entries)}] Running predictions for "
                        f"{species_id}/{chromosome_id}"
                    )
                    prediction_start = perf_counter()
                    batch_size = _create_predictions(
                        ds=ds,
                        base_model=base_model,
                        classifier=classifier,
                        tokenizer=tokenizer,
                        species_id=species_id,
                        chromosome_id=chromosome_id,
                        model_checkpoint=model_checkpoint,
                        model_path=model_path,
                        output_dir=final_output_dir,
                        batch_size=batch_size,
                        window_size=window_size,
                        stride=stride,
                        device=device,
                        tqdm_position=tqdm_position,
                    )

                    logger.info(
                        "Scaffold %s prediction/write time %.2fs (batch cap=%d)",
                        chromosome_id,
                        perf_counter() - prediction_start,
                        batch_size,
                    )
                    # Wait for all ranks before validating coverage and marking completion.
                    barrier()
                    with guarded_on_main():
                        if is_main_process():
                            finish_run(final_output_dir)
                            if batch_size_cache:
                                with atomic_output_path(batch_size_cache) as temporary:
                                    with open(temporary, "w") as handle:
                                        handle.write(str(batch_size) + "\n")
                    logger.info(
                        "Scaffold %s finished in %.2fs including validation",
                        chromosome_id,
                        perf_counter() - scaffold_start,
                    )
                finally:
                    ds.close()
                del ds

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

    parser = argparse.ArgumentParser(description="Gene prediction")

    parser.add_argument(
        "--input-zarr",
        "-i",
        default=None,
        help="Path to input zarr dataset (from transform.py). "
        "Not required when --manifest is used.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Directory for resumable prediction segments. "
        "Not required when --manifest is used.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to a JSON file listing sequences to process in one model-load session. "
        'Format: [{"chromosome_id": "...", "sequence_zarr": "...", "predictions_dir": "..."}, ...]. '
        "Enables processing thousands of short sequences without reloading the model each time.",
    )
    parser.add_argument(
        "--model-checkpoint", required=True, help="Path to classifier checkpoint"
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to base embedding model. If not specified, model path will be "
        "inferred from model checkpoint if possible.",
    )

    # Selection arguments
    parser.add_argument(
        "--species-id",
        default=None,
        help="Species ID to process (e.g., 'Osativa'). Required unless "
        "--triton-warmup is true",
    )
    parser.add_argument(
        "--chromosome-id",
        default=None,
        help="Chromosome ID to process (e.g., 'Chr1'). Not required when --manifest is used.",
    )

    # Processing parameters
    parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help="Window size for sequence processing",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=WINDOW_SIZE // 2,
        help="Stride size for overlapping windows",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference. Default 16.",
    )
    parser.add_argument(
        "--batch-size-cache",
        default=None,
        help="Optional file storing the last successful batch cap across worker runs.",
    )
    parser.add_argument(
        "--tqdm-position",
        type=int,
        default=None,
        help="Optional tqdm row to use for this process when multiple GPU jobs run in one terminal",
    )
    parser.add_argument(
        "--show-dynamo-errors",
        action="store_true",
        help="Show torch dynamo errors (suppressed by default)",
    )
    # TODO: could this be determined by model config?
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16", "float64", "double", "half"],
        help="Data type for model inference (default: bfloat16)",
    )
    parser.add_argument(
        "--warmup-batch-size",
        type=int,
        default=None,
        help="Exact batch size to probe during warmup (overrides conservative capping)",
    )

    parser.add_argument(
        "--triton-warmup",
        action="store_true",
        help="pre-warm the Triton autotune cache with a dummy forward pass.",
    )

    args = parser.parse_args()

    if not args.triton_warmup:
        if args.species_id is None:
            logger.error(
                "Error: --species-id is required unless --triton-warmup is set"
            )

        if args.manifest is None:
            if (
                (args.chromosome_id is None)
                or (args.input_zarr is None)
                or (args.output_dir is None)
            ):
                logger.error(
                    "Error: this is not a triton warmup run, but input data has not been provided. One of the "
                    "following is required: \n --manifest \n OR \n --chromosome-id --input-zarr and --output-dir"
                )
                raise RuntimeError

        create_predictions(
            manifest=args.manifest,
            species_id=args.species_id,
            chromosome_id=args.chromosome_id,
            input_zarr=args.input_zarr,
            output_dir=args.output_dir,
            model_checkpoint=args.model_checkpoint,
            model_path=args.model_path,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            dtype=args.dtype,
            tqdm_position=args.tqdm_position,
            show_dynamo_errors=args.show_dynamo_errors,
            batch_size_cache=args.batch_size_cache,
        )
    else:
        warmup_triton(
            model_checkpoint=args.model_checkpoint,
            model_path=args.model_path,
            window_size=args.window_size,
            batch_size=args.batch_size,
            warmup_batch_size=args.warmup_batch_size,
            dtype=args.dtype,
            suppress_dynamo_errors=args.suppress_dynamo_errors,
        )


if __name__ == "__main__":
    try:
        main()
    except PredictionResumeError as error:
        logger.error("Cannot resume predictions: %s", error)
        raise SystemExit(2) from error
