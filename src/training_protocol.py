"""Shared initialization artifacts for paired training experiments."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from src.atomic_io import atomic_output_path


def state_digest(state):
    """Hash tensor values, names, shapes and dtypes, independently of serialization."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_initial_state(model, path, seed):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to replace shared initialization: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {
        "schema_version": 1,
        "seed": seed,
        "config": asdict(model.config),
        "state_dict": state,
        "state_sha256": state_digest(state),
    }
    with atomic_output_path(path) as temporary:
        torch.save(payload, temporary)
    return payload["state_sha256"]


def read_initial_state(path, seed):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("seed") != seed:
        raise ValueError("Shared initialization format/seed does not match this run")
    if state_digest(payload["state_dict"]) != payload["state_sha256"]:
        raise ValueError("Shared initialization tensor checksum does not match")
    return payload


def restore_initial_state(model, payload):
    # These fields affect weights or the forward pass. Monitoring settings may differ.
    current = asdict(model.config)
    for key in (
        "architecture",
        "vocab_size",
        "num_labels",
        "max_sequence_length",
        "base_encoder_dim",
        "base_encoder_path",
        "base_encoder_revision",
        "base_encoder_frozen",
        "base_encoder_randomize",
        "token_embedding_dim",
        "head_encoder_layers",
        "dropout",
        "token_class_names",
        "token_class_frequencies",
    ):
        if current[key] != payload["config"][key]:
            raise ValueError(f"Shared initialization configuration mismatch: {key}")
    model.load_state_dict(payload["state_dict"], strict=True)
    actual = state_digest(model.state_dict())
    if actual != payload["state_sha256"]:
        raise ValueError("Loaded model differs from shared initialization")
    return actual
