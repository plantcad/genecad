import argparse
import ast
import glob
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def _parse_label_classes(attrs: dict) -> dict[int, str]:
    value = attrs.get("label_classes")
    if value is None:
        return {}

    if isinstance(value, dict):
        parsed = value
    elif isinstance(value, str):
        parsed = ast.literal_eval(value)
    else:
        parsed = dict(value)

    out: dict[int, str] = {}
    for k, v in parsed.items():
        out[int(k)] = str(v)
    return out


def _sample_dataset_stats(ds_path: str, batch_size: int = 2048) -> dict:
    ds = xr.open_zarr(ds_path)
    required = {"tag_labels", "label_mask", "species"}
    missing = required.difference(set(ds.data_vars.keys()))
    if missing:
        raise ValueError(
            f"Dataset {ds_path} missing required variables: {sorted(missing)}"
        )

    label_classes = _parse_label_classes(ds.attrs)
    n_classes = len(label_classes)
    if n_classes == 0:
        raise ValueError(f"Dataset {ds_path} has no label_classes metadata")

    n_samples = int(ds.sizes["sample"])
    token_counts = np.zeros(n_classes, dtype=np.int64)
    species_counts: Counter[str] = Counter()
    invalid_label_count = 0
    masked_token_count = 0
    unmasked_token_count = 0

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = ds.isel(sample=slice(start, end)).compute()

        labels = batch["tag_labels"].values
        masks = batch["label_mask"].values.astype(bool)
        species = batch["species"].values

        species_counts.update(str(s) for s in species)

        valid_labels = labels[masks]
        masked_token_count += int((~masks).sum())
        unmasked_token_count += int(masks.sum())

        if valid_labels.size > 0:
            invalid = (valid_labels < 0) | (valid_labels >= n_classes)
            invalid_label_count += int(invalid.sum())
            if invalid_label_count == 0:
                token_counts += np.bincount(valid_labels, minlength=n_classes)

    ds.close()

    total_tokens = int(token_counts.sum())
    intergenic_idx = None
    for idx, name in label_classes.items():
        if name == "intergenic":
            intergenic_idx = idx
            break

    intergenic_tokens = (
        int(token_counts[intergenic_idx]) if intergenic_idx is not None else 0
    )
    non_intergenic_ratio = (
        float((total_tokens - intergenic_tokens) / total_tokens)
        if total_tokens > 0
        else 0.0
    )

    class_counts = {
        label_classes[idx]: int(token_counts[idx])
        for idx in sorted(label_classes.keys())
    }

    return {
        "dataset": ds_path,
        "n_samples": n_samples,
        "n_classes": n_classes,
        "total_unmasked_tokens": unmasked_token_count,
        "total_masked_tokens": masked_token_count,
        "invalid_label_count": invalid_label_count,
        "species_counts": dict(species_counts),
        "class_counts": class_counts,
        "non_intergenic_ratio": non_intergenic_ratio,
    }


def run_label_qc(args: argparse.Namespace) -> None:
    logger.info("Running label QC")

    train_stats = _sample_dataset_stats(args.train_dataset, batch_size=args.batch_size)
    valid_stats = _sample_dataset_stats(args.valid_dataset, batch_size=args.batch_size)

    issues: list[str] = []

    for split_name, stats in [("train", train_stats), ("valid", valid_stats)]:
        if stats["n_samples"] < args.min_samples_per_split:
            issues.append(
                f"{split_name} has too few samples: {stats['n_samples']} < {args.min_samples_per_split}"
            )
        if stats["invalid_label_count"] > 0:
            issues.append(
                f"{split_name} has invalid labels: {stats['invalid_label_count']} labels outside [0, n_classes)"
            )

    if train_stats["non_intergenic_ratio"] < args.min_non_intergenic_ratio:
        issues.append(
            "train non-intergenic ratio is too low: "
            f"{train_stats['non_intergenic_ratio']:.4f} < {args.min_non_intergenic_ratio:.4f}"
        )

    train_species = set(train_stats["species_counts"].keys())
    valid_species = set(valid_stats["species_counts"].keys())
    if len(train_species) < args.min_species_in_train:
        issues.append(
            f"train has too few species: {len(train_species)} < {args.min_species_in_train}"
        )

    missing_in_valid = sorted(train_species.difference(valid_species))
    if missing_in_valid and not args.allow_missing_species_in_valid:
        issues.append(
            "some train species are missing in valid split: "
            + ", ".join(missing_in_valid)
        )

    required_core = [
        "intergenic",
        "I-intron",
        "I-cds",
        "I-five_prime_utr",
        "I-three_prime_utr",
    ]
    for cname in required_core:
        count = train_stats["class_counts"].get(cname, 0)
        if count < args.min_core_class_tokens:
            issues.append(
                f"core class '{cname}' has too few train tokens: {count} < {args.min_core_class_tokens}"
            )

    summary = {
        "train": train_stats,
        "valid": valid_stats,
        "issues": issues,
    }
    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved label QC summary to {args.summary_json}")
    logger.info(
        "train stats: samples=%s species=%s non_intergenic_ratio=%.4f",
        train_stats["n_samples"],
        len(train_species),
        train_stats["non_intergenic_ratio"],
    )
    logger.info(
        "valid stats: samples=%s species=%s",
        valid_stats["n_samples"],
        len(valid_species),
    )

    if issues:
        for issue in issues:
            logger.error("QC issue: %s", issue)
        raise SystemExit("Label QC failed. See summary JSON for details.")

    logger.info("Label QC passed")


@dataclass
class CheckpointSelection:
    metric_name: str
    best_metric: float
    epoch: int
    step: int
    checkpoint_path: str
    reason: str


def _find_metrics_csv(output_dir: str) -> str:
    pattern = os.path.join(
        output_dir, "logs", "csv", "lightning_logs", "version_*", "metrics.csv"
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find metrics.csv with pattern: {pattern}")
    return matches[-1]


def _closest_checkpoint(
    checkpoint_dir: str, target_epoch: int, target_step: int
) -> tuple[str, str]:
    candidates = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*-step_*.ckpt")))
    if not candidates:
        last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            return last_ckpt, "fallback_to_last_ckpt"
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    def parse_ckpt(path: str) -> tuple[int, int]:
        name = os.path.basename(path)
        # epoch_00-step_001234.ckpt
        ep = int(name.split("epoch_")[1].split("-")[0])
        st = int(name.split("step_")[1].split(".")[0])
        return ep, st

    best_path = candidates[0]
    best_dist = float("inf")
    for path in candidates:
        ep, st = parse_ckpt(path)
        dist = abs(ep - target_epoch) * 1_000_000 + abs(st - target_step)
        if dist < best_dist:
            best_dist = dist
            best_path = path

    return best_path, "closest_epoch_step"


def run_select_best_checkpoint(args: argparse.Namespace) -> None:
    metrics_csv = _find_metrics_csv(args.output_dir)
    logger.info("Selecting best checkpoint using metrics from %s", metrics_csv)

    df = pd.read_csv(metrics_csv)
    metric_name = args.metric
    if metric_name not in df.columns:
        raise ValueError(f"Metric '{metric_name}' not found in {metrics_csv}")

    scored = df[["epoch", "step", metric_name]].dropna()
    if scored.empty:
        raise ValueError(
            f"Metric '{metric_name}' has no non-null rows in {metrics_csv}"
        )

    if args.mode == "max":
        best_row = scored.loc[scored[metric_name].idxmax()]
    else:
        best_row = scored.loc[scored[metric_name].idxmin()]

    epoch = int(best_row["epoch"])
    step = int(best_row["step"])
    best_metric = float(best_row[metric_name])

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    preferred_name = f"epoch_{epoch:02d}-step_{step:06d}.ckpt"
    preferred_path = os.path.join(checkpoint_dir, preferred_name)

    if os.path.exists(preferred_path):
        chosen = preferred_path
        reason = "exact_epoch_step_match"
    else:
        chosen, reason = _closest_checkpoint(checkpoint_dir, epoch, step)

    best_link = os.path.join(checkpoint_dir, "best.ckpt")
    if os.path.lexists(best_link):
        os.remove(best_link)
    os.symlink(os.path.basename(chosen), best_link)

    selection = CheckpointSelection(
        metric_name=metric_name,
        best_metric=best_metric,
        epoch=epoch,
        step=step,
        checkpoint_path=chosen,
        reason=reason,
    )

    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(selection.__dict__, f, indent=2)

    logger.info("Best metric %.6f at epoch=%d step=%d", best_metric, epoch, step)
    logger.info("Selected checkpoint: %s (%s)", chosen, reason)
    logger.info("Created symlink: %s -> %s", best_link, os.path.basename(chosen))
    logger.info("Saved checkpoint selection summary to %s", args.summary_json)


def run_feature_qc(args: argparse.Namespace) -> None:
    logger.info("Running feature-table QC")
    features = pd.read_parquet(args.features_parquet)

    issues: list[str] = []
    required_columns = [
        "species_id",
        "chromosome_id",
        "gene_id",
        "strand",
        "gene_start",
        "gene_stop",
        "feature_start",
        "feature_stop",
    ]
    missing = [c for c in required_columns if c not in features.columns]
    if missing:
        raise ValueError(f"Missing required columns in features parquet: {missing}")

    bad_feature_bounds = features[features["feature_start"] > features["feature_stop"]]
    if len(bad_feature_bounds) > 0:
        issues.append(f"feature_start > feature_stop rows: {len(bad_feature_bounds)}")

    bad_gene_bounds = features[features["gene_start"] > features["gene_stop"]]
    if len(bad_gene_bounds) > 0:
        issues.append(f"gene_start > gene_stop rows: {len(bad_gene_bounds)}")

    out_of_gene = features[
        (features["feature_start"] < features["gene_start"])
        | (features["feature_stop"] > features["gene_stop"])
    ]
    if len(out_of_gene) > 0:
        issues.append(f"feature outside gene bounds rows: {len(out_of_gene)}")

    unique_strands = sorted(str(s) for s in features["strand"].dropna().unique())
    if not set(unique_strands).issubset({"positive", "negative"}):
        issues.append(f"unexpected strand values: {unique_strands}")

    summary = {
        "rows": int(len(features)),
        "species": sorted(str(s) for s in features["species_id"].dropna().unique()),
        "unique_strands": unique_strands,
        "issues": issues,
    }

    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved feature QC summary to %s", args.summary_json)
    if issues:
        for issue in issues:
            logger.error("QC issue: %s", issue)
        raise SystemExit("Feature QC failed. See summary JSON for details.")

    logger.info("Feature-table QC passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="GeneCAD quality-control utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    p_feature = sub.add_parser("feature_qc", help="Validate transformed feature tables")
    p_feature.add_argument("--features-parquet", required=True)
    p_feature.add_argument("--summary-json", required=True)

    p_label = sub.add_parser(
        "label_qc", help="Validate generated train/valid label datasets"
    )
    p_label.add_argument("--train-dataset", required=True)
    p_label.add_argument("--valid-dataset", required=True)
    p_label.add_argument("--summary-json", required=True)
    p_label.add_argument("--batch-size", type=int, default=2048)
    p_label.add_argument("--min-samples-per-split", type=int, default=128)
    p_label.add_argument("--min-non-intergenic-ratio", type=float, default=0.01)
    p_label.add_argument("--min-species-in-train", type=int, default=2)
    p_label.add_argument(
        "--allow-missing-species-in-valid", choices=["yes", "no"], default="yes"
    )
    p_label.add_argument("--min-core-class-tokens", type=int, default=100)

    p_best = sub.add_parser(
        "select_best_checkpoint", help="Pick best checkpoint from validation metric"
    )
    p_best.add_argument("--output-dir", required=True)
    p_best.add_argument("--metric", default="valid__entity__overall/f1")
    p_best.add_argument("--mode", choices=["max", "min"], default="max")
    p_best.add_argument("--summary-json", required=True)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.command == "feature_qc":
        run_feature_qc(args)
    elif args.command == "label_qc":
        args.allow_missing_species_in_valid = (
            args.allow_missing_species_in_valid == "yes"
        )
        run_label_qc(args)
    elif args.command == "select_best_checkpoint":
        run_select_best_checkpoint(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
