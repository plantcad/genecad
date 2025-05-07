import os
import glob
import numpy as np
import xarray as xr
import pandas as pd

def get_token_class_weights(path: str, split: str, class_names: list[str]) -> tuple[xr.Dataset, pd.DataFrame]:
    path_fmt = os.path.join(path, f"{split}.{{rank}}.zarr")
    drop_variables = [v for v in xr.open_zarr(path_fmt.format(rank=0)) if v != "labels"]
    paths = glob.glob(path_fmt.format(rank="*"))
    dataset = xr.concat([xr.open_zarr(p, drop_variables=drop_variables, chunks=None) for p in paths], dim="sample")
    labels, counts = np.unique(dataset.labels.values, return_counts=True)
    label_names = {i: name for i, name in enumerate(class_names)}
    
    weights = (
        # Merge label names containing all classes to empirical
        # class frequencies, which may be missing certain classes
        pd.concat([
            pd.Series(label_names, name="label_name"),
            pd.Series(counts, index=labels, name="label_count"),
        ], axis=1)
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
        .assign(label_weight=lambda df: (df["label_freq_clip"].sum() / df["label_freq_clip"]))
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