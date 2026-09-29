"""Write synthetic predictions in the layouts the predict step leaves on disk."""

import numpy as np
import xarray as xr

from src import prediction_checkpoint as checkpoint
from src.modeling import GeneClassifierConfig

FEATURES = GeneClassifierConfig().token_entity_names_with_background()
STRANDS = ("positive", "negative")


def one_hot_logits(labels, seed=0, sharpness=8.0):
    """Logits that make each position's label the clear winner, with a little noise."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 0.5, size=(len(labels), len(FEATURES))).astype(np.float32)
    logits[np.arange(len(labels)), labels] += sharpness
    return logits


def write_segments(root, logits, segment_length, chromosome_id="chr1"):
    """Commit ``logits[strand]`` (positions x features) as consecutive segments."""
    length = len(logits["positive"])
    checkpoint.prepare_run(str(root), {"model": "test"}, length)
    for strand in STRANDS:
        values = logits[strand]
        assert len(values) == length
        for window_id, start in enumerate(range(0, length, segment_length)):
            stop = min(start + segment_length, length)
            block = values[start:stop]
            result = xr.Dataset(
                {
                    "token_logits": (
                        ["sequence", "token"],
                        np.zeros((stop - start, 2), dtype=np.float32),
                    ),
                    "token_predictions": (
                        ["sequence"],
                        np.zeros(stop - start, dtype=np.int64),
                    ),
                    "feature_logits": (["sequence", "feature"], block),
                    "feature_predictions": (["sequence"], block.argmax(axis=1)),
                },
                coords={"sequence": np.arange(start, stop), "feature": FEATURES},
                attrs={
                    "species_id": "sp",
                    "chromosome_id": chromosome_id,
                    "model_checkpoint": "test",
                    "model_path": "test",
                },
            )
            checkpoint.commit_segment(str(root), result, strand, [window_id])
    checkpoint.finish_run(str(root))


def write_rank_store(root, logits, chromosome_id="chr1"):
    """The older layout: one ``predictions.0.zarr`` store with a group per strand."""
    for strand in STRANDS:
        ds = xr.Dataset(
            {"feature_logits": (["sequence", "feature"], logits[strand])},
            coords={"sequence": np.arange(len(logits[strand])), "feature": FEATURES},
            attrs={"chromosome_id": chromosome_id},
        )
        ds.to_zarr(str(root / "predictions.0.zarr"), group=strand, zarr_format=2)
