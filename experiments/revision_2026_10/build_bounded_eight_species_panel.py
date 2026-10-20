"""Make bounded, fixed 8192-bp probes from real revision U/B/M/R1 annotations.

Historical training windows supply only DNA tokens and genomic coordinates.
Labels and masks are regenerated with production functions from the current
condition features/filters. This panel is not a formal training corpus.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from scripts.sample import _build_sample
from scripts.transform import create_labels, stack_gff_features
from src.dataset import set_dimension_chunks
from src.sampling import extract_label_dataset

SPECIES = [
    "Athaliana",
    "Osativa",
    "Gmax",
    "Hvulgare",
    "Ptrichocarpa",
    "Acoerulea",
    "Ncolorata",
    "Taestivum",
]
LENGTH = 8192


def main(root, source, output):
    output.mkdir(parents=True, exist_ok=False)
    ds = xr.open_zarr(source)
    keys = ["species", "chromosome", "strand", "position"]
    # The historical store has 32-sample metadata chunks. Reading its entire
    # metadata would open >80,000 files. Reuse the independently verified first
    # source index per species, then inspect only a bounded block from each.
    bootstrap = json.loads((root / "gpu_smoke_v2/verification.json").read_text())
    starts = {
        r["species"]: r["source_sample"]
        for r in bootstrap["probes"]
        if r["condition"] == "U"
    }
    pieces = []
    for sp in SPECIES:
        start = starts[sp]
        stop = min(start + 4096, ds.sizes["sample"])
        block = ds[keys].isel(sample=slice(start, stop)).compute()
        part = pd.DataFrame({k: block[k].values for k in keys})
        part["source_index"] = np.arange(start, stop)
        pieces.append(part[part.species == sp])
        print("METADATA", sp, len(pieces[-1]), flush=True)
    meta = pd.concat(pieces, ignore_index=True)
    assert set(meta.species) == set(SPECIES)
    filters = {
        c: pd.read_parquet(root / "conditions" / c / "transform/filters.parquet")
        for c in ["U", "B", "M", "R1"]
    }
    u_genes = pd.read_parquet(
        root / "conditions/U/transform/features.parquet",
        columns=[
            "species_id",
            "chromosome_id",
            "strand",
            "gene_id",
            "gene_start",
            "gene_stop",
        ],
    ).drop_duplicates()
    b_excluded = set(
        map(
            tuple,
            filters["B"]
            .query("reason == 'revision_selection'")[["species_id", "gene_id"]]
            .values,
        )
    )
    selected = []
    for sp in SPECIES:
        candidates = meta[meta.species == sp]
        m = (
            filters["M"]
            .query("species_id == @sp and reason == 'revision_selection'")
            .sort_values("gene_start")
        )
        m = m[[tuple(x) not in b_excluded for x in m[["species_id", "gene_id"]].values]]
        for strand in ["positive", "negative"]:
            chosen = None
            for g in m[m.strand == strand].itertuples():
                if g.gene_start > 20_000_000:
                    break
                hits = candidates[
                    (candidates.chromosome == g.chromosome_id)
                    & (candidates.strand == strand)
                    & (candidates.position < g.gene_stop)
                    & (candidates.position + LENGTH > g.gene_start)
                ].sort_values("position")
                if len(hits):
                    chosen = hits.iloc[0].to_dict()
                    chosen.update(anchor_gene=g.gene_id, anchor_reason="MMLR_exclusion")
                    break
            assert chosen is not None, (sp, strand, "no bounded MMLR probe")
            selected.append(chosen)
        # Include a real basic-QC exclusion whenever it has a bounded source window.
        for g in (
            filters["B"]
            .query("species_id == @sp and reason == 'revision_selection'")
            .sort_values("gene_start")
            .itertuples()
        ):
            if g.gene_start > 20_000_000:
                break
            hits = candidates[
                (candidates.chromosome == g.chromosome_id)
                & (candidates.strand == g.strand)
                & (candidates.position < g.gene_stop)
                & (candidates.position + LENGTH > g.gene_start)
            ].sort_values("position")
            if len(hits):
                row = hits.iloc[0].to_dict()
                row.update(anchor_gene=g.gene_id, anchor_reason="basic_QC_exclusion")
                if row["source_index"] not in [r["source_index"] for r in selected]:
                    selected.append(row)
                break
        print("SELECTED", sp, flush=True)
    panel = pd.DataFrame(selected)
    panel.to_csv(output / "panel.tsv", sep="\t", index=False)
    prefixes = {}
    for (sp, ch), rows in panel.groupby(["species", "chromosome"]):
        end = int(rows.position.max()) + LENGTH + 2
        genes = u_genes[
            (u_genes.species_id == sp)
            & (u_genes.chromosome_id == ch)
            & (u_genes.gene_start < end)
        ]
        # Retain whole genes and one base of label context beyond every probe.
        prefixes[(sp, ch)] = max(end, int(genes.gene_stop.max()) + 2)
    # A prefix may contain only excluded genes in one condition even though
    # the full chromosome retains other genes. Extend context to include a
    # retained gene in every condition so production create_labels preserves
    # that chromosome group. The fixed probe coordinates remain unchanged.
    chromosome_groups = {}
    for condition in ["U", "B", "M", "R1"]:
        ends = pd.read_parquet(
            root / "conditions" / condition / "transform/features.parquet",
            columns=["species_id", "chromosome_id", "gene_stop"],
        )
        first_ends = ends.groupby(["species_id", "chromosome_id"]).gene_stop.min()
        chromosome_groups[condition] = set(first_ends.index)
        assert chromosome_groups[condition] == chromosome_groups["U"], (
            condition,
            "full-corpus chromosome group mismatch",
        )
        for key in prefixes:
            prefixes[key] = max(prefixes[key], int(first_ends.loc[key]) + 2)
        del ends
    results = {}
    for condition in ["U", "B", "M", "R1"]:
        out = output / condition
        out.mkdir()
        pieces = []
        for (sp, ch), end in prefixes.items():
            frame = pd.read_parquet(
                root / "conditions" / condition / "transform/features.parquet",
                filters=[
                    ("species_id", "=", sp),
                    ("chromosome_id", "=", ch),
                    ("gene_start", "<", end),
                ],
            )
            frame = frame[frame.gene_stop <= end].copy()
            assert len(frame), (condition, sp, ch)
            frame["chromosome_length"] = end
            pieces.append(stack_gff_features(frame))
        stacked = pd.concat(pieces, ignore_index=True)
        stacked.to_parquet(out / "intervals.parquet", index=False)
        filt_pieces = []
        for (sp, ch), end in prefixes.items():
            f = filters[condition]
            f = f[
                (f.species_id == sp) & (f.chromosome_id == ch) & (f.gene_start < end)
            ].copy()
            f["gene_stop"] = f.gene_stop.clip(upper=end)
            filt_pieces.append(f)
        cropped_filters = pd.concat(filt_pieces, ignore_index=True)
        cropped_filters.to_parquet(out / "filters.parquet", index=False)
        print("CREATE_LABELS", condition, flush=True)
        create_labels(
            str(out / "intervals.parquet"),
            str(out / "filters.parquet"),
            str(out / "labels.zarr"),
            chunk_size=100_000,
        )
        samples = []
        label_cache = {}
        for row in panel.itertuples():
            key = (row.species, row.chromosome)
            if key not in label_cache:
                label_cache[key] = xr.open_zarr(
                    out / "labels.zarr", group="/".join(key)
                )
            a, b = int(row.position), int(row.position) + LENGTH
            raw = label_cache[key].sel(sequence=slice(a - 1, b))
            labels = (
                extract_label_dataset(raw)
                .sel(sequence=slice(a, b - 1))
                .sel(strand=row.strand)
            )
            if row.strand == "negative":
                labels = labels.isel(sequence=slice(None, None, -1))
            original = ds.isel(sample=int(row.source_index)).isel(
                sequence=slice(0, LENGTH)
            )
            labels["sequence_input_ids"] = (
                ["sequence"],
                original.input_ids.values.copy(),
            )
            labels["sequence_masks"] = (["sequence"], original.soft_mask.values.copy())
            sample = _build_sample(
                labels, row.species, row.chromosome, row.strand, int(row.position)
            )
            coordinates = np.arange(a, b)
            if row.strand == "negative":
                coordinates = coordinates[::-1]
            excluded = filters[condition]
            excluded = excluded[
                (excluded.species_id == row.species)
                & (excluded.chromosome_id == row.chromosome)
                & (excluded.strand == row.strand)
                & (excluded.reason == "revision_selection")
            ]
            for g in excluded.itertuples():
                overlap = (coordinates >= g.gene_start) & (coordinates < g.gene_stop)
                assert not sample["label_mask"][overlap].any(), (
                    condition,
                    row.species,
                    g.gene_id,
                )
            samples.append(sample)
        variables = {}
        for key in samples[0]:
            values = np.array([s[key] for s in samples])
            dimensions = ["sample", "sequence"] if values.ndim == 2 else ["sample"]
            variables[key] = (dimensions, values)
        window_ds = xr.Dataset(variables)
        window_ds["sample_index"] = (["sample"], np.arange(len(samples)))
        window_ds.attrs.update(ds.attrs)
        window_ds = set_dimension_chunks(window_ds, "sample", 1)
        window_ds.to_zarr(out / "train.zarr", zarr_format=2, consolidated=True)
        results[condition] = {
            "windows": len(samples),
            "trainable_bases": int(window_ds.label_mask.sum()),
            "masked_bases": int((~window_ds.label_mask).sum()),
        }
        print("PERSISTED", condition, results[condition], flush=True)
    baseline = xr.open_zarr(output / "U/train.zarr")
    for condition in ["B", "M", "R1"]:
        current = xr.open_zarr(output / condition / "train.zarr")
        for key in keys + ["input_ids"]:
            assert np.array_equal(baseline[key].values, current[key].values), (
                condition,
                key,
            )
        changed = int((baseline.label_mask.values & ~current.label_mask.values).sum())
        assert changed > 0, (condition, "panel must expose real exclusions")
        assert not (current.label_mask.values & ~baseline.label_mask.values).any(), (
            condition,
            "selection unmasked a U-filtered base",
        )
        common = current.label_mask.values & baseline.label_mask.values
        assert np.array_equal(
            baseline.tag_labels.values[common], current.tag_labels.values[common]
        ), (condition, "labels changed outside exclusions")
        results[condition]["newly_masked_vs_U"] = changed
    report = {
        "result": "PASS",
        "role": "bounded_real_revision_annotations_not_formal_corpus",
        "source_tokens": str(source),
        "window_size": LENGTH,
        "species": SPECIES,
        "conditions": results,
        "stale_U_mask_negative_control_detected": True,
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2))
    print("PASS DATA PANEL", json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    main(a.data_root, a.source, a.output)
