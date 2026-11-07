"""Regression checks for paired window universes, splits and exclusion masks."""

import importlib.util
from importlib import import_module
from argparse import Namespace
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts import sample, transform
from src.schema import SEQUENCE_MODELING_FEATURES


@pytest.fixture(autouse=True)
def synchronous_dask():
    if importlib.util.find_spec("dask") is not None:
        dask = import_module("dask")
        scheduler = dask.config.set(scheduler="synchronous")
    else:
        scheduler = nullcontext()
    with scheduler:
        yield


def sequence_dataset(strands=("positive", "negative")):
    length = 24
    mask = np.ones((2, length), dtype=bool)
    mask[:, 6:10] = False
    labels = np.zeros((2, length), dtype=np.int8)
    features = np.zeros((2, length, len(SEQUENCE_MODELING_FEATURES)), dtype=np.int8)
    features[:, :, 0] = 1
    return xr.Dataset(
        {
            "sequence_input_ids": (
                ("strand", "sequence"),
                np.stack([np.arange(length) % 13, (np.arange(length) + 3) % 13]),
            ),
            "sequence_masks": (("strand", "sequence"), np.zeros_like(mask)),
            "feature_labels": (("strand", "sequence", "feature"), features),
            "tag_labels": (("strand", "sequence"), labels),
            "tag_labels_masked": (("strand", "sequence"), np.where(mask, labels, -1)),
            "label_masks": (("strand", "sequence"), mask),
        },
        coords={
            "strand": ["positive", "negative"],
            "feature": SEQUENCE_MODELING_FEATURES,
        },
    ).sel(strand=list(strands))


def write_group(ds, root, group="Sp/chr1"):
    ds.to_zarr(str(root), group=group, mode="a", zarr_format=2, consolidated=True)


def reference_dataset(source, windows, width=4):
    tokens = []
    for strand, position in windows:
        # Invalid strand and bounds tests still need a structurally valid reference.
        values = source.sequence_input_ids.sel(
            strand=strand if strand in source.strand else "positive"
        ).values
        selected = values[max(position, 0) : max(position, 0) + width]
        selected = np.pad(selected, (0, max(0, width - len(selected))))
        tokens.append(selected[::-1] if strand == "negative" else selected)
    return xr.Dataset(
        {
            "strand": ("sample", [strand for strand, _ in windows]),
            "position": ("sample", [position for _, position in windows]),
            "input_ids": (("sample", "sequence"), np.stack(tokens)),
        }
    )


def window_args(tmp_path, source, reference, width=4):
    return Namespace(
        input=str(source),
        output=str(tmp_path / "output.zarr"),
        windows_from=str(reference),
        window_size=width,
        chunk_size=2,
        intergenic_proportion=0.8,
        num_workers=None,
    )


def generate_group(args, group="Sp/chr1"):
    species, chromosome = group.split("/")
    return sample._generate_training_windows(
        (args, args.output, group, species, chromosome, 1, 1)
    )


def test_fixed_windows_preserve_interleaved_reference_order_and_reverse_orientation(
    tmp_path,
):
    source = sequence_dataset(("negative", "positive"))
    windows = [
        ("positive", 0),
        ("negative", 4),
        ("positive", 6),  # Every base is excluded; the window must still be kept.
        ("positive", 12),
        ("negative", 16),
    ]
    reference = reference_dataset(source, windows)
    write_group(source, tmp_path / "source.zarr")
    write_group(reference, tmp_path / "reference.zarr")
    args = window_args(tmp_path, tmp_path / "source.zarr", tmp_path / "reference.zarr")
    generate_group(args)
    result = xr.open_zarr(args.output, group="Sp/chr1").load()
    assert (
        list(zip(result.strand.values.tolist(), result.position.values.tolist()))
        == windows
    )
    np.testing.assert_array_equal(result.input_ids, reference.input_ids)
    expected_masks = []
    for strand, position in windows:
        values = source.label_masks.sel(strand=strand).values[position : position + 4]
        expected_masks.append(values[::-1] if strand == "negative" else values)
    np.testing.assert_array_equal(result.label_mask, expected_masks)
    assert not result.label_mask.isel(sample=2).values.any()


@pytest.mark.parametrize(
    ("windows", "reference_width", "requested_width"),
    [
        ([("positive", 0)], 4, 8),
        ([("positive", -1)], 4, 4),
        ([("negative", 22)], 4, 4),
        ([("sideways", 0)], 4, 4),
    ],
    ids=["width-mismatch", "negative-position", "past-end", "invalid-strand"],
)
def test_invalid_reference_windows_fail_before_writing(
    tmp_path, windows, reference_width, requested_width
):
    source = sequence_dataset()
    write_group(source, tmp_path / "source.zarr")
    write_group(
        reference_dataset(source, windows, reference_width), tmp_path / "reference.zarr"
    )
    args = window_args(
        tmp_path, tmp_path / "source.zarr", tmp_path / "reference.zarr", requested_width
    )
    with pytest.raises(ValueError):
        generate_group(args)
    assert not (tmp_path / "output.zarr").exists()


def test_fixed_windows_reject_changed_dna_with_identical_coordinates(tmp_path):
    source = sequence_dataset()
    reference = reference_dataset(source, [("positive", 0)])
    source["sequence_input_ids"].values[0, 1] += 1
    write_group(source, tmp_path / "source.zarr")
    write_group(reference, tmp_path / "reference.zarr")
    args = window_args(tmp_path, tmp_path / "source.zarr", tmp_path / "reference.zarr")
    with pytest.raises(ValueError):
        generate_group(args)
    assert not (tmp_path / "output.zarr").exists()


def test_missing_reference_root_is_an_error(tmp_path):
    source = sequence_dataset()
    write_group(source, tmp_path / "source.zarr")
    args = window_args(tmp_path, tmp_path / "source.zarr", tmp_path / "absent.zarr")
    expected_errors: tuple[type[Exception], ...] = (FileNotFoundError, ValueError)
    with pytest.raises(expected_errors):
        generate_group(args)
    assert not (tmp_path / "output.zarr").exists()


def test_target_chromosome_absent_from_reference_produces_no_windows(tmp_path):
    source = sequence_dataset()
    write_group(source, tmp_path / "source.zarr")
    write_group(
        reference_dataset(source, [("positive", 0)]),
        tmp_path / "reference.zarr",
        "Sp/chr2",
    )
    args = window_args(tmp_path, tmp_path / "source.zarr", tmp_path / "reference.zarr")
    result = generate_group(args)
    assert result["samples_generated"] == 0
    assert not (tmp_path / "output.zarr").exists()


def test_missing_target_reference_chromosome_fails_before_any_output(tmp_path):
    source = sequence_dataset()
    write_group(source, tmp_path / "source.zarr")
    for group in ["Sp/chr1", "Sp/chr2"]:
        write_group(
            reference_dataset(source, [("positive", 0)]),
            tmp_path / "reference.zarr",
            group,
        )
    args = window_args(tmp_path, tmp_path / "source.zarr", tmp_path / "reference.zarr")
    with pytest.raises(ValueError):
        sample.generate_training_windows(args)
    assert not (tmp_path / "output.zarr").exists()


def test_split_is_identical_when_zarr_group_listing_order_changes(
    tmp_path, monkeypatch
):
    root = tmp_path / "windows.zarr"
    for species, chromosome, count in [
        ("B", "chr2", 7),
        ("A", "chr2", 9),
        ("A", "chr1", 11),
    ]:
        ds = xr.Dataset(
            {
                "species": ("sample", [species] * count),
                "chromosome": ("sample", [chromosome] * count),
                "position": ("sample", np.arange(count) * 4),
                "input_ids": (
                    ("sample", "sequence"),
                    np.arange(count * 4).reshape(count, 4),
                ),
            },
            attrs={
                "feature_labels_classes": {"0": "intergenic"},
                "label_classes": {"0": "intergenic"},
            },
        )
        write_group(ds, root, f"{species}/{chromosome}")
    groups = sample.list_species_contig_datatree(str(root))
    monkeypatch.setattr(
        sample,
        "get_species_config",
        lambda _: Namespace(
            split=Namespace(use_in_training=True, use_in_validation=True)
        ),
    )
    outputs = []
    for number, ordered_groups in enumerate([groups, list(reversed(groups))]):
        monkeypatch.setattr(
            sample,
            "list_species_contig_datatree",
            lambda _, order=ordered_groups: order,
        )
        args = Namespace(
            input=str(root),
            train_output=str(tmp_path / f"train{number}.zarr"),
            valid_output=str(tmp_path / f"valid{number}.zarr"),
            seed=42,
            valid_proportion=0.3,
            chunk_size=2,
        )
        sample.generate_training_splits(args)
        outputs.append(
            [
                xr.open_zarr(path).load()
                for path in [args.train_output, args.valid_output]
            ]
        )
    for first, second in zip(*outputs):
        xr.testing.assert_identical(first, second)
        np.testing.assert_array_equal(
            first.sample_index, np.arange(first.sizes["sample"])
        )
    train_keys = set(
        zip(
            outputs[0][0].species.values,
            outputs[0][0].chromosome.values,
            outputs[0][0].position.values,
        )
    )
    valid_keys = set(
        zip(
            outputs[0][1].species.values,
            outputs[0][1].chromosome.values,
            outputs[0][1].position.values,
        )
    )
    assert train_keys.isdisjoint(valid_keys)
    assert len(train_keys | valid_keys) == 27


def complete_feature_rows():
    rows = []
    for species in ["A", "B"]:
        for gene, strand, start in [("gp", 1, 4), ("gn", -1, 14)]:
            for offset, feature in enumerate(
                ["five_prime_UTR", "CDS", "three_prime_UTR"]
            ):
                rows.append(
                    {
                        "species_id": species,
                        "chromosome_id": "chr1",
                        "gene_id": gene,
                        "gene_strand": strand,
                        "gene_start": start,
                        "gene_stop": start + 6,
                        "transcript_id": f"{gene}.1",
                        "transcript_start": start,
                        "transcript_is_canonical": True,
                        "feature_type": feature,
                        "feature_start": start + 2 * offset,
                        "feature_stop": start + 2 * offset + 2,
                    }
                )
    return rows


def test_exclusion_masks_cover_forward_and_reverse_spans_without_cross_species_removal(
    tmp_path,
):
    pd.DataFrame(complete_feature_rows()).to_parquet(tmp_path / "features.parquet")
    pd.DataFrame(
        [("A", "gp"), ("A", "gp"), ("A", "gn")], columns=["species_id", "gene_id"]
    ).to_csv(tmp_path / "exclude.tsv", sep="\t", index=False)
    transform.filter_features(
        str(tmp_path / "features.parquet"),
        str(tmp_path / "retained.parquet"),
        str(tmp_path / "filters.parquet"),
        exclude_genes_path=str(tmp_path / "exclude.tsv"),
    )
    retained = pd.read_parquet(tmp_path / "retained.parquet")
    filters = pd.read_parquet(tmp_path / "filters.parquet")
    assert set(retained.species_id) == {"B"}
    assert set(retained.gene_id) == {"gp", "gn"}
    assert len(filters) == 2
    assert set(filters.reason) == {"revision_selection"}
    for strand, start in [("positive", 4), ("negative", 14)]:
        mask, reasons = transform._generate_label_masks(
            pd.DataFrame(columns=["start", "stop"]),
            filters[filters.strand == strand],
            domain=(0, 24),
            strand=strand,
        )
        assert reasons == ["revision_selection"]
        expected = np.ones(24, dtype=np.int8)
        expected[start : start + 6] = 0
        np.testing.assert_array_equal(mask[:, 0], expected)
        if strand == "negative":
            np.testing.assert_array_equal(
                mask[::-1, 0], np.r_[np.ones(4), np.zeros(6), np.ones(14)]
            )


@pytest.mark.parametrize("excluded", [None, [], [("Z", "gp")]])
def test_no_matching_exclusions_preserve_features_and_write_empty_filter_schema(
    tmp_path, excluded
):
    original = pd.DataFrame(complete_feature_rows())
    original.to_parquet(tmp_path / "features.parquet")
    selection = None
    if excluded is not None:
        selection = str(tmp_path / "exclude.tsv")
        pd.DataFrame(excluded, columns=["species_id", "gene_id"]).to_csv(
            selection, sep="\t", index=False
        )
    transform.filter_features(
        str(tmp_path / "features.parquet"),
        str(tmp_path / "retained.parquet"),
        str(tmp_path / "filters.parquet"),
        exclude_genes_path=selection,
    )
    retained = pd.read_parquet(tmp_path / "retained.parquet")
    filters = pd.read_parquet(tmp_path / "filters.parquet")
    assert len(retained) == len(original)
    assert filters.empty
    assert set(filters.columns) == {
        "species_id",
        "chromosome_id",
        "gene_id",
        "strand",
        "reason",
        "gene_start",
        "gene_stop",
    }
