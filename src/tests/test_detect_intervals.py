import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.dataset import open_datatree
from src.tests.segment_test_support import (
    FEATURES,
    one_hot_logits,
    write_rank_store,
    write_segments,
)

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
LENGTH = 60_000
DECODING: dict[str, Any] = {
    "decode_direct": False,
    "viterbi_alpha": None,
    "intergenic_bias": 0.0,
    "domain": "plant",
    "remove_incomplete_features": True,
}


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        f"{name}_script", SCRIPTS / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


detect = load_script("detect_intervals")
export = load_script("export_gff")

IG, INTRON, UTR5, CDS, UTR3 = (
    FEATURES.index(name)
    for name in (
        "intergenic",
        "intron",
        "five_prime_utr",
        "cds",
        "three_prime_utr",
    )
)


def long_gene():
    """A 35 kb transcript with two exons, far longer than any segment used below."""
    labels = np.full(LENGTH, IG)
    labels[5_000:5_300] = UTR5
    labels[5_300:20_000] = CDS
    labels[20_000:20_400] = INTRON
    labels[20_400:40_000] = CDS
    labels[40_000:40_400] = UTR3
    return labels


@pytest.fixture
def logits():
    labels = long_gene()
    return {
        "positive": one_hot_logits(labels, seed=0),
        "negative": one_hot_logits(labels[::-1].copy(), seed=1),
    }


def decode(root, **options):
    output = root / "intervals.zarr"
    detect.detect_intervals(
        input_dir=str(root), output=str(output), **{**DECODING, **options}
    )
    return output


def intervals(zarr):
    tree = open_datatree(str(zarr), consolidated=False)
    return tree["/intervals"].ds.to_dataframe().reset_index()


def in_memory_intervals(root, **options):
    predictions = detect.merge_prediction_datasets(
        str(root), drop_variables=["token_predictions", "token_logits"]
    )
    return detect._detect_intervals(predictions=predictions, **{**DECODING, **options})


@pytest.mark.parametrize("segment_length", [257, LENGTH])
def test_a_gene_spanning_many_segments_is_one_transcript(
    tmp_path, logits, segment_length
):
    write_segments(tmp_path, logits, segment_length)
    df = intervals(decode(tmp_path))

    transcripts = df[df["entity_name"] == "transcript"]
    assert transcripts[["strand", "start", "stop"]].values.tolist() == [
        ["positive", 5_000, 40_399],
        ["negative", LENGTH - 1 - 40_399, LENGTH - 1 - 5_000],
    ]
    for strand in ("positive", "negative"):
        genic = df[df["strand"] == strand]
        assert (genic["entity_name"] == "exon").sum() == 2
        assert (genic["entity_name"] == "intron").sum() == 1
        assert (genic["entity_name"] == "cds").sum() == 2


@pytest.mark.parametrize("segment_length", [257, 4096, LENGTH])
def test_segment_size_does_not_change_the_intervals(tmp_path, logits, segment_length):
    write_segments(tmp_path, logits, segment_length)
    expected = in_memory_intervals(tmp_path).to_dataframe().reset_index()
    pd.testing.assert_frame_equal(
        intervals(decode(tmp_path)), expected, check_like=True
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(intergenic_bias=0.7),
        dict(viterbi_alpha=0.01),
        dict(intergenic_bias=1.3, remove_incomplete_features=False),
    ],
)
def test_decoding_options_give_the_same_intervals_as_the_in_memory_path(
    tmp_path, options
):
    # Weak evidence, so that the options change what is decoded
    rng = np.random.default_rng(3)
    noisy = {
        strand: rng.normal(0, 1.5, size=(20_000, len(FEATURES))).astype(np.float32)
        for strand in ("positive", "negative")
    }
    write_segments(tmp_path, noisy, 999)
    expected = in_memory_intervals(tmp_path, **options).to_dataframe().reset_index()
    assert len(expected) > 100
    pd.testing.assert_frame_equal(
        intervals(decode(tmp_path, **options)), expected, check_like=True
    )


def test_only_the_intervals_are_saved_unless_sequences_are_requested(tmp_path, logits):
    write_segments(tmp_path, logits, 1000)
    lean = decode(tmp_path)
    assert set(open_datatree(str(lean), consolidated=False).children) == {"intervals"}

    full = decode(tmp_path, save_sequences=True)
    tree = open_datatree(str(full), consolidated=False)
    assert set(tree.children) == {"intervals", "sequences"}
    assert tree["/sequences"].ds.sizes["sequence"] == LENGTH
    pd.testing.assert_frame_equal(intervals(lean), intervals(full), check_like=True)
    assert (
        open_datatree(str(lean), consolidated=False)["/intervals"].ds.attrs
        == tree["/intervals"].ds.attrs
    )


def test_the_gff_needs_nothing_but_the_intervals(tmp_path, logits):
    write_segments(tmp_path, logits, 1000)
    gffs = []
    for name, options in [("lean", {}), ("full", dict(save_sequences=True))]:
        root = tmp_path / name
        root.mkdir()
        write_segments(root, logits, 1000)
        zarr = decode(root, **options)
        gff = root / "out.gff"
        export.export_gff(str(zarr), str(gff), None, 3, 1)
        gffs.append(gff.read_text())
    assert gffs[0] == gffs[1]
    assert "\ttranscript\t" in gffs[0] or "\tmRNA\t" in gffs[0]


def test_rank_stores_are_still_decoded(tmp_path, logits):
    segments, ranks = tmp_path / "segments", tmp_path / "ranks"
    segments.mkdir()
    ranks.mkdir()
    write_segments(segments, logits, 1000)
    write_rank_store(ranks, logits)
    pd.testing.assert_frame_equal(
        intervals(decode(ranks)), intervals(decode(segments)), check_like=True
    )


def test_direct_decoding_still_works_on_segments(tmp_path, logits):
    write_segments(tmp_path, logits, 1000)
    df = intervals(decode(tmp_path, decode_direct=True))
    assert set(df["decoding"]) == {"direct"}
    assert len(df) > 0
