import glob
import os
import logging
from bisect import bisect_right
from collections import OrderedDict
from typing import Any, Iterator
import numpy as np
import xarray as xr
from src.dataset import open_datatree
from src.prediction_checkpoint import RUN, SUCCESS, completed_segments

logger = logging.getLogger(__name__)


def segment_records(input_dir: str) -> list[dict] | None:
    """Verified prediction segments in a directory, or None for legacy rank stores.

    Requires verified coverage of both strands before any segmented output is read.
    """
    segmented = (
        os.path.exists(os.path.join(input_dir, RUN))
        or os.path.exists(os.path.join(input_dir, SUCCESS))
        or bool(glob.glob(os.path.join(input_dir, "segment.*")))
    )
    return completed_segments(input_dir) if segmented else None


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
    records = segment_records(input_dir)
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


class SegmentedPredictions:
    """Feature logits of one chromosome, read from its prediction segments as needed.

    `merge_prediction_datasets` loads every position of both strands and copies them
    while merging, which takes hundreds of gigabytes for a chromosome of a few Gb. This
    class keeps only the segment being read in memory, so decoding needs memory in
    proportion to a segment (or a window) rather than to the chromosome.

    Parameters
    ----------
    input_dir : str
        Directory with the committed segments of one chromosome
    records : list[dict]
        Verified segment receipts, as returned by `segment_records`
    """

    CACHED_SEGMENTS = 8

    def __init__(self, input_dir: str, records: list[dict]):
        self.input_dir = input_dir
        self._records = {
            strand: sorted(
                (r for r in records if r["strand"] == strand), key=lambda r: r["start"]
            )
            for strand in ("positive", "negative")
        }
        for strand, strand_records in self._records.items():
            if not strand_records:
                raise ValueError(f"Missing {strand} predictions in {input_dir}")
        self._starts = {
            strand: [r["start"] for r in strand_records]
            for strand, strand_records in self._records.items()
        }
        self._cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()

        self.length = self._records["positive"][-1]["stop"]
        if self._records["negative"][-1]["stop"] != self.length:
            raise ValueError(f"Strands differ in length in {input_dir}")

        # Same names and attributes as the merged dataset would carry
        attrs = {}
        for strand in self._records:
            with self._open(strand, 0) as ds:
                features = ds["feature"].values.tolist()
                strand_attrs = dict(ds.attrs)
            if strand == "positive":
                self.features = features
                attrs = strand_attrs
            else:
                if features != self.features:
                    raise ValueError(f"Strands differ in features in {input_dir}")
                for key, value in strand_attrs.items():
                    if key not in attrs:
                        attrs[key] = value
                    elif attrs[key] != value:
                        del attrs[key]
        self.attrs = attrs

    def _open(self, strand: str, index: int) -> xr.Dataset:
        record = self._records[strand][index]
        return xr.open_zarr(
            os.path.join(self.input_dir, record["store"]),
            group=strand,
            consolidated=True,
        )

    def _load(self, strand: str, index: int) -> np.ndarray:
        """Feature logits of one segment, shape (positions, features)."""
        record = self._records[strand][index]
        with self._open(strand, index) as ds:
            if (
                ds.sizes["sequence"] != record["stop"] - record["start"]
                or int(ds["sequence"][0]) != record["start"]
            ):
                raise ValueError(
                    f"Segment {record['store']} does not cover {record['start']}-{record['stop']}"
                )
            if ds["feature"].values.tolist() != self.features:
                raise ValueError(f"Segment {record['store']} has different features")
            return ds["feature_logits"].transpose("sequence", "feature").values

    def _cached(self, strand: str, index: int) -> np.ndarray:
        key = (strand, index)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            self._cache[key] = self._load(strand, index)
            if len(self._cache) > self.CACHED_SEGMENTS:
                self._cache.popitem(last=False)
        return self._cache[key]

    def blocks(self, strand: str, reverse: bool = False) -> Iterator[np.ndarray]:
        """Yield the feature logits of consecutive segments, shape (positions, features).

        With ``reverse`` the last segment comes first and each block is reversed, which
        is the order in which the negative strand is decoded.
        """
        indices = range(len(self._records[strand]))
        for index in reversed(indices) if reverse else indices:
            block = self._load(strand, index)
            yield block[::-1] if reverse else block

    def window(
        self, strand: str, start: int, stop: int, features: list[str] | None = None
    ) -> np.ndarray:
        """Feature logits of positions ``start`` to ``stop``, as for slicing an array.

        Parameters
        ----------
        strand : str
            Either "positive" or "negative"
        start, stop : int
            Slice bounds; out of range values are clipped
        features : list[str], optional
            Features to return, in this order; all of them if not provided

        Returns
        -------
        np.ndarray
            Array of shape (positions, features)
        """
        start, stop, _ = slice(start, stop).indices(self.length)
        stop = max(start, stop)
        if features is None:
            columns = slice(None)
        else:
            missing = [name for name in features if name not in self.features]
            if missing:
                raise KeyError(f"Features {missing} are not in the predictions")
            columns = [self.features.index(name) for name in features]
        records = self._records[strand]
        parts = []
        index = bisect_right(self._starts[strand], start) - 1
        while start < stop and index < len(records):
            record = records[index]
            if record["start"] >= stop:
                break
            block = self._cached(strand, index)
            parts.append(
                block[
                    max(start, record["start"]) - record["start"] : min(
                        stop, record["stop"]
                    )
                    - record["start"]
                ]
            )
            index += 1
        if not parts:
            return np.empty((0, len(self.features)), dtype=np.float32)[:, columns]
        return np.ascontiguousarray(np.concatenate(parts)[:, columns])


def open_segments(input_dir: str) -> SegmentedPredictions | None:
    """Open the committed segments of a directory, or None for legacy rank stores."""
    records = segment_records(input_dir)
    return None if records is None else SegmentedPredictions(input_dir, records)
