import glob
import os
import logging
from typing import Any
import numpy as np
import xarray as xr
from src.dataset import open_datatree
from src.prediction_checkpoint import RUN, SUCCESS, completed_segments

logger = logging.getLogger(__name__)


def merge_prediction_datasets(
    input_dir: str, glob_pattern: str = "predictions.*.zarr", **kwargs: Any
) -> xr.Dataset:
    """
    Merge prediction files from multiple ranks into a single dataset.

    Parameters
    ----------
    input_dir : str
        Directory containing prediction files
    glob_pattern : str
        Glob pattern to match prediction datasets, e.g. `predictions.*.zarr`
    kwargs : Any
        Additional keyword arguments to pass to `open_datatree`; e.g. `drop_variables` can
        be useful to eliminate unused arrays (`drop_variables=["token_predictions", "token_logits"]`)

    Returns
    -------
    xr.Dataset
        Merged sequence predictions dataset
    """
    # Require verified coverage of both strands before reading segmented output.
    segmented = (
        os.path.exists(os.path.join(input_dir, RUN))
        or os.path.exists(os.path.join(input_dir, SUCCESS))
        or bool(glob.glob(os.path.join(input_dir, "segment.*")))
    )
    records = completed_segments(input_dir) if segmented else None
    rank_prediction_paths = sorted(glob.glob(os.path.join(input_dir, glob_pattern)))
    if records is None and not rank_prediction_paths:
        raise FileNotFoundError(
            f"No prediction files found matching '{glob_pattern}' in {input_dir}"
        )

    strand_datasets = []
    for strand in ["positive", "negative"]:
        logger.info(f"Processing strand: {strand}")
        rank_strand_data = []
        paths = (
            [
                os.path.join(input_dir, r["store"])
                for r in records
                if r["strand"] == strand
            ]
            if records is not None
            else rank_prediction_paths
        )
        for rank_path in paths:
            logger.debug(f"Loading strand '{strand}' from {rank_path}")
            raw_predictions = open_datatree(
                rank_path,
                consolidated=True,
                **kwargs,
            )
            # A legacy rank may own no windows for a short chromosome.
            if strand in raw_predictions.children:
                rank_strand_data.append(raw_predictions[f"/{strand}"].ds)
        logger.info(
            f"Concatenating {len(rank_strand_data)} rank datasets for {strand!r} strand along the sequence dimension."
        )
        if not rank_strand_data:
            raise ValueError(f"Missing {strand} predictions in {input_dir}")
        dataset = xr.concat(rank_strand_data, dim="sequence")
        dataset = dataset.sortby("sequence")
        if not np.array_equal(
            dataset.sequence.values, np.arange(dataset.sizes["sequence"])
        ):
            raise ValueError(f"Gaps or duplicate coordinates in {strand} predictions")
        strand_datasets.append(dataset.expand_dims(strand=[strand]))
    logger.info(
        f"Concatenating {len(strand_datasets)} datasets along the strand dimension."
    )
    sequence_predictions = xr.concat(
        strand_datasets, dim="strand", join="exact", combine_attrs="drop_conflicts"
    )
    logger.info(f"Concatenated sequence predictions dataset:\n{sequence_predictions}")

    return sequence_predictions
