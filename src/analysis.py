import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
from src.schema import ModelingFeatureType as MFT, SEQUENCE_MODELING_FEATURES
from typing import Optional


SEQUENCE_MODELING_ANNOTATIONS = [
    MFT.FIVE_PRIME_UTR.value,
    MFT.CDS.value,
    MFT.THREE_PRIME_UTR.value,
]


def get_sequence_modeling_labels(ds: xr.Dataset) -> xr.DataArray:
    feature_labels = ds.feature_labels.sel(feature=SEQUENCE_MODELING_ANNOTATIONS)
    intron_labels = ds.region_labels.sel(region=MFT.INTRON).rename(region="feature")
    labels = xr.concat([intron_labels, feature_labels], dim="feature")
    intergenic_labels = (
        (1 - labels.max(dim="feature"))
        .expand_dims("feature", axis=1)
        .assign_coords(feature=[MFT.INTERGENIC])
    )
    labels = xr.concat([intergenic_labels, labels], dim="feature")
    assert labels.feature.values.tolist() == SEQUENCE_MODELING_FEATURES
    assert labels.isin([0, 1]).all().item()
    assert (labels.sum(dim="feature") == 1).all().item()
    return labels


def get_token_class_weights(
    path: str, split: str, class_names: list[str]
) -> tuple[xr.Dataset, pd.DataFrame]:
    path_fmt = os.path.join(path, f"{split}.{{rank}}.zarr")
    drop_variables = [v for v in xr.open_zarr(path_fmt.format(rank=0)) if v != "labels"]
    paths = glob.glob(path_fmt.format(rank="*"))
    dataset = xr.concat(
        [xr.open_zarr(p, drop_variables=drop_variables, chunks=None) for p in paths],
        dim="sample",
    )
    labels, counts = np.unique(dataset.labels.values, return_counts=True)
    label_names = {i: name for i, name in enumerate(class_names)}

    weights = (
        # Merge label names containing all classes to empirical
        # class frequencies, which may be missing certain classes
        pd.concat(
            [
                pd.Series(label_names, name="label_name"),
                pd.Series(counts, index=labels, name="label_count"),
            ],
            axis=1,
        )
        .rename_axis(index="label_index")
        .reset_index()
        # Omit masked labels
        .pipe(lambda df: df[df["label_index"] >= 0])
        # Impute missing label counts with minimum count
        .assign(label_freq=lambda df: df["label_count"].fillna(df["label_count"].min()))
        .assign(label_freq=lambda df: df["label_freq"] / df["label_freq"].sum())
        # Clip frequencies on low-side to 1/1M to avoid extreme imbalance in weights
        .assign(label_freq_clip=lambda df: df["label_freq"].clip(1e-6))
        # Set weight as normalized inverse class frequencies
        .assign(
            label_weight=lambda df: (
                df["label_freq_clip"].sum() / df["label_freq_clip"]
            )
        )
        .assign(label_weight=lambda df: df["label_weight"] / df["label_weight"].sum())
    )
    return dataset, weights


def get_normalized_inverse_class_frequencies(label_counts: np.ndarray) -> np.ndarray:
    """Numerically stable inverse-class frequency weights.

    Equivalent to inverse class frequencies normalized to sum to 1.

    See:
    - https://github.com/zongyanliu/gene_modeling/blob/003e1b380f19c902f58ae559bbf825a670331f6b/train_model_lightning.py#L219-L226
    """
    log_frequencies = np.log(label_counts)
    negative_log_frequencies = -log_frequencies
    max_value = np.max(negative_log_frequencies)
    exp_values = np.exp(negative_log_frequencies - max_value)
    softmax_values = exp_values / np.sum(exp_values)
    return softmax_values


def load_feature_length_distributions(
    path: str,
    feature_labels: list[str],
    min_length: Optional[int | dict[str, int | None]] = None,
    max_length: Optional[int] = None,
    sigma: Optional[float] = None,
) -> dict[str, pd.Series]:
    """Load feature length distributions from parquet file with optional Gaussian smoothing.

    Parameters
    ----------
    path : str
        Path to parquet file containing length distributions
    feature_labels : list[str]
        List of feature labels in the order they appear in the model
    min_length : int, dict[str, int | None], or None, optional
        Minimum allowed feature length(s). Can be:
        - int: Applied to all features
        - dict: Maps feature labels to specific min_length values (None means no constraint)
        - None: No minimum length constraint for any feature
        If provided as int or dict values, must be >= 1.
    max_length : int, optional
        Maximum feature length to use. If None, uses the maximum observed length.
        If provided, uses the lesser of this value and the maximum observed length.
        Probability mass beyond this length is truncated to the max length bin.
    sigma : float, optional
        Standard deviation for Gaussian kernel smoothing. If None, no smoothing is applied.

    Returns
    -------
    dict[str, pd.Series]
        Dictionary mapping feature name to length probability Series with length values as index
    """
    from scipy.ndimage import gaussian_filter1d

    df = pd.read_parquet(path)

    # Normalize min_length to dict format
    if isinstance(min_length, int):
        if min_length < 1:
            raise ValueError(f"min_length must be >= 1, got {min_length}")
        min_length_dict = {feature: min_length for feature in feature_labels}
    elif isinstance(min_length, dict):
        # Validate dict values
        for feature, min_val in min_length.items():
            if min_val is not None and min_val < 1:
                raise ValueError(
                    f"min_length for {feature} must be >= 1, got {min_val}"
                )
        min_length_dict = min_length
    else:  # None
        min_length_dict = {feature: None for feature in feature_labels}

    # Determine effective max length
    observed_max_length = int(df["length"].max())
    if max_length is None:
        effective_max_length = observed_max_length
    else:
        effective_max_length = min(max_length, observed_max_length)

    length_probs = {}

    for feature in feature_labels:
        feature_data = df[df["class_name"] == feature].sort_values("length")
        if len(feature_data) == 0:
            raise ValueError(f"No length data found for feature: {feature}")

        # Build empirical counts with truncation
        counts = np.zeros(effective_max_length)
        for _, row in feature_data.iterrows():
            length = int(row["length"])
            count = row["count"]

            # Truncate to effective max length
            if length > effective_max_length:
                length = effective_max_length

            length_idx = length - 1  # Convert to 0-based index
            counts[length_idx] += count

        # Apply optional Gaussian smoothing
        if sigma is not None:
            counts = gaussian_filter1d(counts, sigma=sigma, mode="constant", cval=0.0)

        # Apply feature-specific minimum length constraint
        feature_min_length = min_length_dict.get(feature)
        if feature_min_length is not None:
            counts[: feature_min_length - 1] = 0

        # Convert to probabilities
        if counts.sum() == 0:
            min_msg = (
                f" after applying min_length={feature_min_length}"
                if feature_min_length is not None
                else ""
            )
            raise ValueError(f"No valid counts for feature {feature}{min_msg}")

        probs = counts / counts.sum()

        # Create Series with true length values as index (1-based)
        length_index = np.arange(1, effective_max_length + 1)
        length_probs[feature] = pd.Series(
            probs, index=length_index, name=f"{feature}_length_prob"
        )

    return length_probs
