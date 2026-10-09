"""Tests for FASTA sequence extraction (``scripts/extract_fasta.py``)."""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

# scripts/ is not an importable package, so load the module by path (see
# test_fix_orf.py for the same pattern).
_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "extract_fasta.py"
)
_spec = importlib.util.spec_from_file_location("extract_fasta", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
extract_fasta = importlib.util.module_from_spec(_spec)
sys.modules["extract_fasta"] = extract_fasta
_spec.loader.exec_module(extract_fasta)

from src.dataset import open_datatree  # noqa: E402

# -------------------------------------------------------------------------------------------------
# Synthetic multi-record FASTA
# -------------------------------------------------------------------------------------------------

CHROM_SEQS = {
    "chr1": "ACGTacgtNNNNACGTACGTacgtACGTACGTACGTacgtACGT",
    "chr2": "TTTTggggCCCCaaaaTTTTggggCCCC",
    "chr3": "GATTACAgattacaGATTACAgattaca",
}


@pytest.fixture
def fasta_path(tmp_path: pathlib.Path) -> str:
    path = tmp_path / "test.fasta"
    with open(path, "w") as fh:
        for chrom_id, seq in CHROM_SEQS.items():
            fh.write(f">{chrom_id} synthetic {chrom_id}\n{seq}\n")
    return str(path)


# -------------------------------------------------------------------------------------------------
# extract_fasta_manifest must match extract_fasta_file exactly
# -------------------------------------------------------------------------------------------------
#
# extract_fasta_manifest exists purely as a performance optimization over
# calling extract_fasta_file once per chromosome (single linear scan of the
# FASTA file instead of one scan per chromosome, and each matched record is
# written out immediately rather than collected first, so peak memory during
# a manifest scan is bounded by one chromosome at a time). Neither property
# is worth anything if the two code paths can silently diverge, so this
# compares their output byte-for-byte.


def test_manifest_matches_per_chromosome_extraction(
    fasta_path: str, tmp_path: pathlib.Path
) -> None:
    per_chrom_paths = {}
    for chrom_id in CHROM_SEQS:
        out = str(tmp_path / f"per_chrom_{chrom_id}.zarr")
        extract_fasta.extract_fasta_file(
            species_id="sp1",
            fasta_file=fasta_path,
            chrom_map_str=f"{chrom_id}:{chrom_id}",
            output_path=out,
            tokenizer_path=None,
        )
        per_chrom_paths[chrom_id] = out

    entries = [
        {
            "chromosome_id": chrom_id,
            "output_zarr": str(tmp_path / f"manifest_{chrom_id}.zarr"),
        }
        for chrom_id in CHROM_SEQS
    ]
    extract_fasta.extract_fasta_manifest(
        species_id="sp1",
        fasta_file=fasta_path,
        entries=entries,
        tokenizer_path=None,
    )

    for chrom_id in CHROM_SEQS:
        per_chrom_ds = open_datatree(per_chrom_paths[chrom_id])["sp1"][chrom_id]
        manifest_ds = open_datatree(str(tmp_path / f"manifest_{chrom_id}.zarr"))["sp1"][
            chrom_id
        ]
        for var in ["sequence_tokens", "sequence_masks"]:
            np.testing.assert_array_equal(
                per_chrom_ds[var].values,
                manifest_ds[var].values,
                err_msg=f"{chrom_id}/{var} differs between the two extraction paths",
            )


def test_manifest_only_writes_requested_chromosomes(
    fasta_path: str, tmp_path: pathlib.Path
) -> None:
    """A manifest naming a subset of records must not touch the others."""
    entries = [{"chromosome_id": "chr2", "output_zarr": str(tmp_path / "chr2.zarr")}]
    extract_fasta.extract_fasta_manifest(
        species_id="sp1",
        fasta_file=fasta_path,
        entries=entries,
        tokenizer_path=None,
    )

    assert (tmp_path / "chr2.zarr").exists()
    assert not (tmp_path / "chr1.zarr").exists()
    assert not (tmp_path / "chr3.zarr").exists()

    ds = open_datatree(str(tmp_path / "chr2.zarr"))["sp1"]["chr2"]
    seq = ds["sequence_tokens"].sel(strand="positive").values
    assert b"".join(seq).decode() == CHROM_SEQS["chr2"]


TOKEN_MAP = {**{b: i for i, b in enumerate("ACGTNacgt", 2)}, "[PAD]": 0, "é": 20}


def test_byte_tokenizer_matches_a_lookup_per_base() -> None:
    bases = np.frombuffer(bytes(range(256)) * 3, dtype="S1")
    encoded = {k.encode("utf-8"): v for k, v in TOKEN_MAP.items()}
    expected = np.array([encoded.get(b, -1) for b in bases], dtype=np.int64)
    tokenizer = extract_fasta.ByteTokenizer(TOKEN_MAP)
    np.testing.assert_array_equal(tokenizer(bases), expected)
    out = np.empty(len(bases), dtype=np.int64)
    assert tokenizer(bases, out=out) is out
    np.testing.assert_array_equal(out, expected)


def test_records_without_bases_are_extracted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty record gives empty arrays and does not stop the other records."""
    path = tmp_path / "empty.fasta"
    path.write_text(">chr1\nACGTNacgt\n>empty\n>chr3\nGATTACA\n")
    monkeypatch.setattr(
        extract_fasta,
        "_load_tokenizer",
        lambda tokenizer_path: extract_fasta.ByteTokenizer(TOKEN_MAP),
    )
    entries = [
        {"chromosome_id": c, "output_zarr": str(tmp_path / f"{c}.zarr")}
        for c in ("chr1", "empty", "chr3")
    ]
    extract_fasta.extract_fasta_manifest(
        species_id="sp1",
        fasta_file=str(path),
        entries=entries,
        tokenizer_path="fake",
    )
    lengths = {}
    for c in ("chr1", "empty", "chr3"):
        ds = open_datatree(str(tmp_path / f"{c}.zarr"))["sp1"][c]
        lengths[c] = ds.sizes["sequence"]
        assert ds["sequence_input_ids"].shape == (2, lengths[c])
    assert lengths == {"chr1": 9, "empty": 0, "chr3": 7}
    ds = open_datatree(str(tmp_path / "chr3.zarr"))["sp1"]["chr3"]
    np.testing.assert_array_equal(
        ds["sequence_input_ids"].sel(strand="positive").values, [4, 2, 5, 5, 2, 3, 2]
    )
