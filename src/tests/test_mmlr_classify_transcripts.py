import json

import numpy as np
import pandas as pd
import pytest

from scripts.mmlr_classify_transcripts import (
    annotate_gff,
    default_dict,
    load_models,
    main,
    score_table,
    tag_to_dict,
    to_average,
)


class TestToAverage:
    def test_single_value(self):
        assert to_average("0.5") == 0.5

    def test_multiple_values_averaged(self):
        assert to_average("0.1,0.2,0.3") == pytest.approx(0.2)


class TestTagToDict:
    def test_parses_semicolon_separated_pairs(self):
        result = tag_to_dict("ID=gene1;Name=TestGene")
        assert result == {"ID": "gene1", "Name": "TestGene"}

    def test_single_pair(self):
        assert tag_to_dict("ID=gene1") == {"ID": "gene1"}


class TestLoadModels:
    def test_default_weights_match_genecad_paper(self):
        multi_exon_model, single_exon_model = load_models(None)

        assert multi_exon_model.intercept_ == pytest.approx(
            default_dict["multi_intercept"]
        )
        np.testing.assert_allclose(
            multi_exon_model.coef_[0],
            [
                default_dict["multi_TIS"],
                default_dict["multi_TTS"],
                default_dict["multi_Donor"],
                default_dict["multi_Acceptor"],
            ],
        )
        assert single_exon_model.intercept_ == pytest.approx(
            default_dict["single_intercept"]
        )
        np.testing.assert_allclose(
            single_exon_model.coef_[0],
            [default_dict["single_TIS"], default_dict["single_TTS"]],
        )
        # Models should be immediately usable for prediction, without .fit()
        multi_exon_model.predict(np.array([[0.9, 0.9, 0.9, 0.9]]))
        single_exon_model.predict(np.array([[0.9, 0.9]]))

    def test_loads_custom_weights_from_json(self, tmp_path):
        custom_weights = {
            "multi_intercept": -1.0,
            "multi_TIS": 1.0,
            "multi_TTS": 2.0,
            "multi_Donor": 3.0,
            "multi_Acceptor": 4.0,
            "single_intercept": -5.0,
            "single_TIS": 6.0,
            "single_TTS": 7.0,
        }
        json_path = tmp_path / "weights.json"
        with open(json_path, "w") as f:
            json.dump(custom_weights, f)

        multi_exon_model, single_exon_model = load_models(str(json_path))

        assert multi_exon_model.intercept_ == pytest.approx(-1.0)
        np.testing.assert_allclose(multi_exon_model.coef_[0], [1.0, 2.0, 3.0, 4.0])
        assert single_exon_model.intercept_ == pytest.approx(-5.0)
        np.testing.assert_allclose(single_exon_model.coef_[0], [6.0, 7.0])


class TestScoreTable:
    def make_models(self):
        # Simple weights so predictions are easy to reason about by hand:
        # multi-exon score = TIS (all other coefficients zero); single-exon
        # score = TIS as well. Decision boundary at TIS == 0.
        weights = {
            "multi_intercept": 0.0,
            "multi_TIS": 1.0,
            "multi_TTS": 0.0,
            "multi_Donor": 0.0,
            "multi_Acceptor": 0.0,
            "single_intercept": 0.0,
            "single_TIS": 1.0,
            "single_TTS": 0.0,
        }
        return load_models_from_dict(weights)

    def test_routes_by_donor_presence_and_scores_each_group(self):
        multi_exon_model, single_exon_model = self.make_models()

        input_df = pd.DataFrame(
            {
                "transcript": ["multi_pos", "multi_neg", "single_pos", "single_neg"],
                "gene": ["g1", "g1", "g2", "g2"],
                "TIS": [1.0, 0.1, 1.0, 0.1],
                "TTS": [0.0, 0.0, 0.0, 0.0],
                "donor": ["0.5,0.5", "0.5,0.5", None, None],
                "acceptor": ["0.5,0.5", "0.5,0.5", None, None],
            }
        )

        result = score_table(input_df, multi_exon_model, single_exon_model)

        result = result.set_index("transcript")
        assert result.loc["multi_pos", "filter"] == 1
        assert result.loc["multi_neg", "filter"] == 0
        assert result.loc["single_pos", "filter"] == 1
        assert result.loc["single_neg", "filter"] == 0
        # score column is the predicted probability of the positive class
        assert result.loc["multi_pos", "score"] > 0.5
        assert result.loc["multi_neg", "score"] < 0.5


def load_models_from_dict(weights_dict):
    import tempfile
    import os

    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(weights_dict, f)
        return load_models(path)
    finally:
        os.remove(path)


GFF3_ONE_GENE_TWO_TRANSCRIPTS = """##gff-version 3
chr1\ttest\tgene\t1\t400\t.\t+\t.\tID=gene1
chr1\ttest\tmRNA\t1\t400\t.\t+\t.\tID=mRNA1;Parent=gene1
chr1\ttest\tmRNA\t1\t300\t.\t+\t.\tID=mRNA2;Parent=gene1
chr1\ttest\tgene\t500\t700\t.\t+\t.\tID=gene2
chr1\ttest\tmRNA\t500\t700\t.\t+\t.\tID=mRNA3;Parent=gene2
"""


class TestAnnotateGff:
    def test_tags_genes_and_transcripts_by_filter_result(self, tmp_path):
        gff_path = tmp_path / "input.gff3"
        gff_path.write_text(GFF3_ONE_GENE_TWO_TRANSCRIPTS)

        # gene1 has one passing and one failing transcript -> gene should
        # still be tagged as passing (np.any). gene2's transcript isn't in
        # the scored table at all -> should default to failing.
        df_scored = pd.DataFrame(
            {
                "transcript": ["mRNA1", "mRNA2"],
                "gene": ["gene1", "gene1"],
                "filter": [1, 0],
            }
        )

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        annotate_gff(df_scored, str(gff_path), str(out_dir))

        out_path = out_dir / "input_tagged.gff3"
        assert out_path.exists()
        lines = out_path.read_text().splitlines()

        tags_by_id = {}
        for line in lines:
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            attrs = dict(pair.split("=") for pair in fields[8].split(";"))
            tags_by_id[attrs["ID"]] = attrs["passPlantCADFilter"]

        assert tags_by_id["gene1"] == "1"
        assert tags_by_id["mRNA1"] == "1"
        assert tags_by_id["mRNA2"] == "0"
        assert tags_by_id["gene2"] == "0"
        assert tags_by_id["mRNA3"] == "0"

    def test_preserves_comment_lines(self, tmp_path):
        gff_path = tmp_path / "input.gff3"
        gff_path.write_text(GFF3_ONE_GENE_TWO_TRANSCRIPTS)
        df_scored = pd.DataFrame(
            {
                "transcript": ["mRNA1", "mRNA2"],
                "gene": ["gene1", "gene1"],
                "filter": [1, 1],
            }
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        annotate_gff(df_scored, str(gff_path), str(out_dir))

        out_text = (out_dir / "input_tagged.gff3").read_text()
        assert out_text.startswith("##gff-version 3")


class TestMainIntegration:
    def test_end_to_end_writes_scored_table_and_tagged_gff(self, tmp_path, monkeypatch):
        gff_path = tmp_path / "input.gff3"
        gff_path.write_text(GFF3_ONE_GENE_TWO_TRANSCRIPTS)

        input_table = pd.DataFrame(
            {
                "transcript": ["mRNA1", "mRNA2", "mRNA3"],
                "gene": ["gene1", "gene1", "gene2"],
                "TIS": [0.9, 0.9, 0.9],
                "TTS": [0.9, 0.9, 0.9],
                "donor": ["0.9,0.9", None, "0.9,0.9"],
                "acceptor": ["0.9,0.9", None, "0.9,0.9"],
            }
        )
        table_path = tmp_path / "scored_junctions.tsv"
        input_table.to_csv(table_path, sep="\t", index=False)

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        monkeypatch.setattr(
            "sys.argv",
            [
                "mmlr_classify_transcripts.py",
                "--input-table",
                str(table_path),
                "--input-gff",
                str(gff_path),
                "--output-dir",
                str(out_dir),
            ],
        )

        main()

        scored_path = out_dir / "scored_junctions_scored.tsv"
        tagged_gff_path = out_dir / "input_tagged.gff3"
        assert scored_path.exists()
        assert tagged_gff_path.exists()

        scored_df = pd.read_csv(scored_path, sep="\t")
        assert "filter" in scored_df.columns
        assert "score" in scored_df.columns
        assert set(scored_df["transcript"]) == {"mRNA1", "mRNA2", "mRNA3"}
