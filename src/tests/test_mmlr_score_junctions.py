from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from scripts.mmlr_prepare_junctions import Junc
from scripts.mmlr_score_junctions import (
    JunctionDataset,
    ScoreList,
    get_longest_transcripts,
    has_canonical_tag,
    strand_string,
    to_junc,
)


class TestScoreList:
    def test_empty(self):
        assert str(ScoreList([])) == ""

    def test_single_value(self):
        assert str(ScoreList([0.5])) == "0.5"

    def test_multiple_values_joined_with_comma(self):
        assert str(ScoreList([0.1, 0.2, 0.3])) == "0.1,0.2,0.3"


class TestHasCanonicalTag:
    def test_canonical_transcript_tag_true(self):
        assert has_canonical_tag({"canonical_transcript": "1"}) is True

    def test_canonical_transcript_tag_false(self):
        assert has_canonical_tag({"canonical_transcript": "0"}) is False

    def test_ensembl_canonical_tag(self):
        assert has_canonical_tag({"tag": "Ensembl_canonical"}) is True

    def test_tag_present_but_not_canonical(self):
        assert has_canonical_tag({"tag": "some_other_tag"}) is False

    def test_neither_key_present(self):
        assert has_canonical_tag({}) is False


class TestStrandString:
    def test_plus(self):
        assert strand_string(1) == "+"

    def test_minus(self):
        assert strand_string(-1) == "-"

    def test_other(self):
        assert strand_string(0) == "."


class TestToJunc:
    @pytest.mark.parametrize(
        "label,expected",
        [
            ("TIS", Junc.TIS),
            ("TTS", Junc.TTS),
            ("Donor", Junc.DONOR),
            ("Acceptor", Junc.ACCEPTOR),
        ],
    )
    def test_valid_labels(self, label, expected):
        assert to_junc(label) == expected

    def test_invalid_label_raises(self):
        with pytest.raises(TypeError):
            to_junc("bogus")


class FakeTokenizer:
    """Minimal stand-in for the PlantCAD AutoTokenizer: character-level
    tokenization (one token per base), just enough for JunctionDataset."""

    mask_token_id = 999
    _char_to_id = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

    def encode_plus(
        self,
        sequence,
        return_tensors="pt",
        return_attention_mask=False,
        return_token_type_ids=False,
    ):
        ids = [self._char_to_id.get(ch, 4) for ch in sequence]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def make_fake_gff(strand):
    gene = SimpleNamespace(location=SimpleNamespace(strand=strand))
    chrom = SimpleNamespace(features=[gene])
    return [chrom]


def make_fastas(chrom_id, length, base="A"):
    return {chrom_id: SeqRecord(Seq(base * length), id=chrom_id)}


class TestJunctionDataset:
    def test_len_matches_dataframe_rows(self):
        df = pd.DataFrame(
            {
                "chrom": [0, 0],
                "gene": [0, 0],
                "pos": [10, 20],
                "junction": [Junc.TIS, Junc.TTS],
            }
        )
        ds = JunctionDataset(
            fastas=make_fastas("chr1", 1000),
            gff=make_fake_gff(1),
            chrom_list=["chr1"],
            df=df,
            tokenizer=FakeTokenizer(),
            window_size=20,
        )
        assert len(ds) == 2

    @pytest.mark.parametrize(
        "strand,junction,expected_mask_range",
        [
            (1, Junc.TIS, (10, 13)),
            (-1, Junc.TTS, (10, 13)),
            (1, Junc.TTS, (8, 11)),
            (-1, Junc.TIS, (8, 11)),
            (1, Junc.DONOR, (10, 12)),
            (-1, Junc.ACCEPTOR, (10, 12)),
            (1, Junc.ACCEPTOR, (9, 11)),
            (-1, Junc.DONOR, (9, 11)),
        ],
    )
    def test_mask_offsets_by_strand_and_junction_type(
        self, strand, junction, expected_mask_range
    ):
        # pos=500 in a 1000bp chromosome, window_size=20 -> no boundary
        # padding, so mask indices land exactly where documented.
        df = pd.DataFrame(
            {"chrom": [0], "gene": [0], "pos": [500], "junction": [junction]}
        )
        ds = JunctionDataset(
            fastas=make_fastas("chr1", 1000),
            gff=make_fake_gff(strand),
            chrom_list=["chr1"],
            df=df,
            tokenizer=FakeTokenizer(),
            window_size=20,
        )

        item = ds[0]

        expected_mask = np.zeros(20, dtype=bool)
        expected_mask[expected_mask_range[0] : expected_mask_range[1]] = True
        np.testing.assert_array_equal(item["mask"], expected_mask)

        # masked positions carry the tokenizer's mask id; everything else is
        # the id for the (homogeneous "A") reference sequence.
        input_ids = item["input_ids"].squeeze().numpy()
        assert (input_ids[expected_mask] == FakeTokenizer.mask_token_id).all()
        assert (input_ids[~expected_mask] == 0).all()  # "A" -> id 0

    def test_pads_with_n_when_window_extends_before_chromosome_start(self):
        # pos=5, window_size=20, token=10 -> conceptual window starts at -5,
        # so the first 5 bases of the returned sequence should be "N".
        df = pd.DataFrame(
            {"chrom": [0], "gene": [0], "pos": [5], "junction": [Junc.TIS]}
        )
        ds = JunctionDataset(
            fastas=make_fastas("chr1", 1000),
            gff=make_fake_gff(1),
            chrom_list=["chr1"],
            df=df,
            tokenizer=FakeTokenizer(),
            window_size=20,
        )

        item = ds[0]

        assert item["sequence"][:5] == "NNNNN"
        assert item["sequence"][5:] == "A" * 15
        # pos itself (index 10) should still be correctly located within the
        # real (non-padded) region and maskable.
        assert item["mask"][10:13].all()

    def test_pads_with_n_when_window_extends_past_chromosome_end(self):
        # chromosome length 1000, pos=995, window_size=20, token=10 ->
        # conceptual window end is 1005, past the chromosome end (1000), so
        # the last 5 bases of the returned sequence should be "N".
        df = pd.DataFrame(
            {"chrom": [0], "gene": [0], "pos": [995], "junction": [Junc.TIS]}
        )
        ds = JunctionDataset(
            fastas=make_fastas("chr1", 1000),
            gff=make_fake_gff(1),
            chrom_list=["chr1"],
            df=df,
            tokenizer=FakeTokenizer(),
            window_size=20,
        )

        item = ds[0]

        assert item["sequence"][-5:] == "NNNNN"
        assert item["sequence"][:-5] == "A" * 15
        assert item["mask"][10:13].all()


def make_mrna(mrna_id, cds_ranges):
    mRNA = SeqFeature(FeatureLocation(0, 1, strand=1), type="mRNA", id=mrna_id)
    mRNA.sub_features = [
        SeqFeature(FeatureLocation(start, end, strand=1), type="CDS")
        for start, end in cds_ranges
    ]
    return mRNA


def make_gff_gene(mrnas):
    gene = SeqFeature(FeatureLocation(0, 1, strand=1), type="gene")
    gene.sub_features = mrnas
    chrom = SimpleNamespace(features=[gene])
    return [chrom]


class TestGetLongestTranscripts:
    def test_selects_transcript_with_more_total_exonic_length(self):
        mRNA_short = make_mrna("mRNA_short", [(0, 50)])
        mRNA_long = make_mrna("mRNA_long", [(0, 50), (60, 160)])
        gff = make_gff_gene([mRNA_short, mRNA_long])

        out_df = pd.DataFrame({"transcript": ["mRNA_short", "mRNA_long"]})
        out_df.index = out_df["transcript"]

        result = get_longest_transcripts(gff, out_df)

        assert dict(zip(out_df.index, result)) == {
            "mRNA_short": False,
            "mRNA_long": True,
        }

    def test_length_formula_does_not_misidentify_longest_transcript(self):
        """get_longest_transcripts sums `(exon.location.end + 1) -
        exon.location.start` per sub-feature to estimate each mRNA's total
        exonic length. BioPython FeatureLocations are 0-based half-open, so
        the correct per-exon length is `end - start`; the extra `+ 1`
        over-counts each exon by one base. For a single-exon transcript this
        is a harmless constant offset, but for multi-exon transcripts it
        inflates the total by (num_exons), which can flip the "longest"
        decision: below, mRNA_a has one exon and a true length of 100bp;
        mRNA_b has three exons totaling only 99bp. mRNA_a should be flagged
        `longest`, but the off-by-one currently makes mRNA_b's computed
        length (99 + 3 exons = 102) exceed mRNA_a's (100 + 1 exon = 101),
        so the shorter transcript wins instead.
        """
        mRNA_a = make_mrna("mRNA_a", [(0, 100)])  # true length 100
        mRNA_b = make_mrna("mRNA_b", [(0, 33), (40, 73), (80, 113)])  # true length 99
        gff = make_gff_gene([mRNA_a, mRNA_b])

        out_df = pd.DataFrame({"transcript": ["mRNA_a", "mRNA_b"]})
        out_df.index = out_df["transcript"]

        result = get_longest_transcripts(gff, out_df)

        assert dict(zip(out_df.index, result)) == {
            "mRNA_a": True,
            "mRNA_b": False,
        }
