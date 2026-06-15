import importlib
import pandas as pd

from src.tests.reelprotein_test_support import install_reelprotein_dependency_stubs

install_reelprotein_dependency_stubs()

reelprotein = importlib.import_module("src.reelprotein")


def test_parse_gff3_resolves_transcript_parent_features(tmp_path):
    """parse_gff3 should resolve CDS/UTR features whose Parent is a transcript
    ID (not a gene-ID-prefixed name) back to the correct gene."""
    gff_path = tmp_path / "input.gff3"
    gff_path.write_text(
        "\n".join(
            [
                "##gff-version 3",
                "chr1\tsrc\tgene\t1\t12\t.\t+\t.\tID=gene1",
                "chr1\tsrc\tmRNA\t1\t12\t.\t+\t.\tID=transcriptA;Parent=gene1",
                "chr1\tsrc\tfive_prime_UTR\t1\t3\t.\t+\t.\tParent=transcriptA",
                "chr1\tsrc\tCDS\t4\t9\t.\t+\t0\tParent=transcriptA",
                "chr1\tsrc\tthree_prime_UTR\t10\t12\t.\t+\t.\tParent=transcriptA",
            ]
        )
    )

    genes = reelprotein.parse_gff3(gff_path)

    gene = genes["chr1"][0]
    assert gene["feature_types"] == {"five_prime_UTR", "CDS", "three_prime_UTR"}
    assert len(gene["cds"]) == 1
    assert gene["cds"][0] == {"start": 4, "end": 9, "phase": 0}
    assert len(gene["utr5"]) == 1
    assert len(gene["utr3"]) == 1


def test_parse_gff3_still_resolves_dotted_parent_ids(tmp_path):
    """parse_gff3 should still work for the traditional gene_id.* naming."""
    gff_path = tmp_path / "input.gff3"
    gff_path.write_text(
        "\n".join(
            [
                "##gff-version 3",
                "chr1\tsrc\tgene\t1\t12\t.\t+\t.\tID=gene1",
                "chr1\tsrc\tmRNA\t1\t12\t.\t+\t.\tID=gene1.t1;Parent=gene1",
                "chr1\tsrc\tCDS\t4\t9\t.\t+\t0\tParent=gene1.t1",
                "chr1\tsrc\tfive_prime_UTR\t1\t3\t.\t+\t.\tParent=gene1.t1",
            ]
        )
    )

    genes = reelprotein.parse_gff3(gff_path)
    gene = genes["chr1"][0]
    assert "CDS" in gene["feature_types"]
    assert "five_prime_UTR" in gene["feature_types"]
    assert len(gene["cds"]) == 1


def test_read_gff_raw_resolves_transcript_parent_features(tmp_path):
    """read_gff_raw should resolve child features whose Parent is a transcript
    ID back to the correct gene entry."""
    gff_path = tmp_path / "input.gff3"
    gff_path.write_text(
        "\n".join(
            [
                "##gff-version 3",
                "chr1\tsrc\tgene\t1\t12\t.\t+\t.\tID=gene1",
                "chr1\tsrc\tmRNA\t1\t12\t.\t+\t.\tID=transcriptA;Parent=gene1",
                "chr1\tsrc\tfive_prime_UTR\t1\t3\t.\t+\t.\tParent=transcriptA",
                "chr1\tsrc\tCDS\t4\t9\t.\t+\t0\tParent=transcriptA",
                "chr1\tsrc\tthree_prime_UTR\t10\t12\t.\t+\t.\tParent=transcriptA",
            ]
        )
    )

    header, gene_entries = reelprotein.read_gff_raw(gff_path)

    # gene1 should have all subfeatures (mRNA + CDS + UTRs)
    key = ("chr1", "gene1")
    assert key in gene_entries
    subfeature_types = [sf[0][2] for sf in gene_entries[key]["subfeatures"]]
    assert "mRNA" in subfeature_types
    assert "CDS" in subfeature_types
    assert "five_prime_UTR" in subfeature_types
    assert "three_prime_UTR" in subfeature_types


def test_generate_final_gff_prefers_merged_group_output(tmp_path):
    """When both a single gene and a merged group containing it are predicted
    positive, only the merged/concatenated gene should appear in the output."""
    gff_path = tmp_path / "input.gff3"
    gff_path.write_text(
        "\n".join(
            [
                "##gff-version 3",
                "chr1\tsrc\tgene\t1\t9\t.\t+\t.\tID=gene1",
                "chr1\tsrc\tmRNA\t1\t9\t.\t+\t.\tID=gene1.t1;Parent=gene1",
                "chr1\tsrc\tCDS\t1\t9\t.\t+\t0\tID=cds1;Parent=gene1.t1",
                "chr1\tsrc\tgene\t20\t28\t.\t+\t.\tID=gene2",
                "chr1\tsrc\tmRNA\t20\t28\t.\t+\t.\tID=gene2.t1;Parent=gene2",
                "chr1\tsrc\tCDS\t20\t28\t.\t+\t0\tID=cds2;Parent=gene2.t1",
            ]
        )
    )
    output_path = tmp_path / "output.gff3"

    # Predictions include both a single gene1 and a merged gene1|gene2
    # The current ProteinID format uses ~ separator
    predictions = pd.DataFrame(
        {
            "ProteinID": ["chr1~gene1~+", "chr1~gene1|gene2~+"],
            "Predicted_Label": [1, 1],
        }
    )

    reelprotein.generate_final_gff(
        predictions, gff_path, output_path, keep_unmerged=False
    )

    output_text = output_path.read_text()
    gene_lines = [line for line in output_text.splitlines() if "\tgene\t" in line]

    # The unconcatenated gene1 should NOT appear — only the merged gene
    assert len(gene_lines) == 1
    assert "concat_gene1_gene2" in gene_lines[0]
    assert "Note=concatenated" in gene_lines[0]


def test_generate_final_gff_keeps_unmerged_genes(tmp_path):
    """With keep_unmerged=True (default), genes not part of any merge group
    should still appear in the output."""
    gff_path = tmp_path / "input.gff3"
    gff_path.write_text(
        "\n".join(
            [
                "##gff-version 3",
                "chr1\tsrc\tgene\t1\t9\t.\t+\t.\tID=gene1",
                "chr1\tsrc\tmRNA\t1\t9\t.\t+\t.\tID=gene1.t1;Parent=gene1",
                "chr1\tsrc\tCDS\t1\t9\t.\t+\t0\tID=cds1;Parent=gene1.t1",
                "chr1\tsrc\tgene\t20\t28\t.\t+\t.\tID=gene2",
                "chr1\tsrc\tmRNA\t20\t28\t.\t+\t.\tID=gene2.t1;Parent=gene2",
                "chr1\tsrc\tCDS\t20\t28\t.\t+\t0\tID=cds2;Parent=gene2.t1",
                "chr1\tsrc\tgene\t50\t60\t.\t+\t.\tID=gene3",
                "chr1\tsrc\tmRNA\t50\t60\t.\t+\t.\tID=gene3.t1;Parent=gene3",
                "chr1\tsrc\tCDS\t50\t60\t.\t+\t0\tID=cds3;Parent=gene3.t1",
            ]
        )
    )
    output_path = tmp_path / "output.gff3"

    # Only the merged group is predicted; gene3 is unrelated
    predictions = pd.DataFrame(
        {
            "ProteinID": ["chr1~gene1|gene2~+"],
            "Predicted_Label": [1],
        }
    )

    reelprotein.generate_final_gff(
        predictions, gff_path, output_path, keep_unmerged=True
    )

    output_text = output_path.read_text()
    gene_lines = [line for line in output_text.splitlines() if "\tgene\t" in line]

    # gene3 should be kept (it's unmerged), gene1/gene2 should appear as merged
    gene_ids = [line.split("\t")[-1] for line in gene_lines]
    gene_ids_str = " ".join(gene_ids)

    assert "ID=gene3" in gene_ids_str
    assert "concat_gene1_gene2" in gene_ids_str
    # gene1 and gene2 individually should NOT appear
    assert "ID=gene1\n" not in output_text or "ID=gene1;" not in output_text.replace(
        "concat_gene1_gene2", ""
    )


def test_merge_group_transcripts_recalculates_phases(tmp_path):
    # Prepare input GFF3 with two genes on forward strand, where first CDS is length 7 (10..16) and second is length 9 (30..38)
    gff_path = tmp_path / "input_fwd.gff3"
    gff_path.write_text(
        "\n".join(
            [
                "##gff-version 3",
                "chr1\tsrc\tgene\t10\t16\t.\t+\t.\tID=gene1",
                "chr1\tsrc\tmRNA\t10\t16\t.\t+\t.\tID=gene1.t1;Parent=gene1",
                "chr1\tsrc\tCDS\t10\t16\t.\t+\t0\tID=cds1;Parent=gene1.t1",
                "chr1\tsrc\tgene\t30\t38\t.\t+\t.\tID=gene2",
                "chr1\tsrc\tmRNA\t30\t38\t.\t+\t.\tID=gene2.t1;Parent=gene2",
                "chr1\tsrc\tCDS\t30\t38\t.\t+\t0\tID=cds2;Parent=gene2.t1",
            ]
        )
    )

    header, gene_entries = reelprotein.read_gff_raw(gff_path)

    # We call merge_group_transcripts for gene1 and gene2
    grp_keys = [("chr1", "gene1"), ("chr1", "gene2")]
    synthetic_mrna_id = "concat_gene1_gene2.t1"

    mrna_fields, merged_children = reelprotein.merge_group_transcripts(
        grp_keys, gene_entries, synthetic_mrna_id
    )

    # Find the CDS features in merged_children
    cds_features = [child for child, attrs in merged_children if child[2] == "CDS"]
    assert len(cds_features) == 2

    # cds1 start=10, end=16 (length 7). Phase should be 0.
    assert cds_features[0][3] == "10"
    assert cds_features[0][4] == "16"
    assert cds_features[0][7] == "0"

    # cds2 start=30, end=38 (length 9). Phase should be recalculated: (-7)%3 = 2.
    assert cds_features[1][3] == "30"
    assert cds_features[1][4] == "38"
    assert cds_features[1][7] == "2"

    # Prepare input GFF3 with two genes on reverse strand
    # First transcribed/translated CDS (higher coordinates): cds1 30..36 (length 7). Phase 0.
    # Second transcribed/translated CDS (lower coordinates): cds2 10..18 (length 9). Original phase 0.
    # Recalculated phase for cds2 should be (-7)%3 = 2.
    gff_path_rev = tmp_path / "input_rev.gff3"
    gff_path_rev.write_text(
        "\n".join(
            [
                "##gff-version 3",
                "chr1\tsrc\tgene\t30\t36\t.\t-\t.\tID=gene1",
                "chr1\tsrc\tmRNA\t30\t36\t.\t-\t.\tID=gene1.t1;Parent=gene1",
                "chr1\tsrc\tCDS\t30\t36\t.\t-\t0\tID=cds1;Parent=gene1.t1",
                "chr1\tsrc\tgene\t10\t18\t.\t-\t.\tID=gene2",
                "chr1\tsrc\tmRNA\t10\t18\t.\t-\t.\tID=gene2.t1;Parent=gene2",
                "chr1\tsrc\tCDS\t10\t18\t.\t-\t0\tID=cds2;Parent=gene2.t1",
            ]
        )
    )

    header_rev, gene_entries_rev = reelprotein.read_gff_raw(gff_path_rev)
    grp_keys_rev = [("chr1", "gene1"), ("chr1", "gene2")]

    mrna_fields_rev, merged_children_rev = reelprotein.merge_group_transcripts(
        grp_keys_rev, gene_entries_rev, synthetic_mrna_id
    )

    cds_features_rev = [
        child for child, attrs in merged_children_rev if child[2] == "CDS"
    ]
    assert len(cds_features_rev) == 2

    # On negative strand, features are sorted in descending order of coordinate
    # So 30..36 is index 0, 10..18 is index 1.
    assert cds_features_rev[0][3] == "30"
    assert cds_features_rev[0][4] == "36"
    assert cds_features_rev[0][7] == "0"

    assert cds_features_rev[1][3] == "10"
    assert cds_features_rev[1][4] == "18"
    assert cds_features_rev[1][7] == "2"
