import importlib
import pandas as pd
import pytest

from src.tests.reelprotein_test_support import install_reelprotein_dependency_stubs

install_reelprotein_dependency_stubs()

reelprotein = importlib.import_module("src.reelprotein")


def test_build_orf_gene_index_resolves_transcript_parent_features(tmp_path):
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

    genes = reelprotein.build_orf_gene_index(gff_path)

    gene = genes["chr1"][0]
    assert gene.feature_types == {"five_prime_UTR", "CDS", "three_prime_UTR"}
    assert [(feature.start, feature.end, feature.phase) for feature in gene.cds] == [
        (4, 9, 0)
    ]


def test_generate_final_gff_prefers_merged_group_output(tmp_path):
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
    predictions = pd.DataFrame(
        {
            "ProteinID": ["chr1_gene1_+", "chr1_gene1|gene2_+"],
            "Predicted_Label": [1, 1],
        }
    )

    reelprotein.generate_final_gff(predictions, gff_path, output_path)

    gene_lines = [
        line for line in output_path.read_text().splitlines() if "\tgene\t" in line
    ]
    assert gene_lines == [
        "chr1\tsrc\tgene\t1\t28\t.\t+\t.\tID=concat_gene1_gene2;Note=concatenated"
    ]


def test_build_orf_gene_index_rejects_invalid_cds_phase(tmp_path):
    gff_path = tmp_path / "invalid_phase.gff3"
    gff_path.write_text(
        "\n".join(
            [
                "##gff-version 3",
                "chr1\tsrc\tgene\t1\t12\t.\t+\t.\tID=gene1",
                "chr1\tsrc\tmRNA\t1\t12\t.\t+\t.\tID=gene1.t1;Parent=gene1",
                "chr1\tsrc\tCDS\t4\t9\t.\t+\t.\tParent=gene1.t1",
            ]
        )
    )

    with pytest.raises(ValueError, match="Invalid CDS phase"):
        reelprotein.build_orf_gene_index(gff_path)


def test_build_orf_gene_index_warns_on_malformed_gff_row(tmp_path, caplog):
    gff_path = tmp_path / "malformed.gff3"
    gff_path.write_text(
        "\n".join(
            [
                "##gff-version 3",
                "chr1\tsrc\tgene\t1\t12\t.\t+\t.\tID=gene1",
                "chr1\ttoo\tshort",
            ]
        )
    )

    with caplog.at_level("WARNING"):
        genes = reelprotein.build_orf_gene_index(gff_path)

    assert len(genes["chr1"]) == 1
    assert "Skipping malformed GFF row" in caplog.text
    assert "expected 9 tab-delimited fields, got 3" in caplog.text


def test_parsers_handle_out_of_order_gene_hierarchy(tmp_path):
    gff_path = tmp_path / "out_of_order.gff3"
    gff_path.write_text(
        "\n".join(
            [
                "##gff-version 3",
                "chr1\tsrc\tCDS\t4\t9\t.\t+\t0\tParent=tx1",
                "chr1\tsrc\tthree_prime_UTR\t10\t12\t.\t+\t.\tParent=tx1",
                "chr1\tsrc\tmRNA\t1\t12\t.\t+\t.\tID=tx1;Parent=gene1",
                "chr1\tsrc\tfive_prime_UTR\t1\t3\t.\t+\t.\tParent=tx1",
                "chr1\tsrc\tgene\t1\t12\t.\t+\t.\tID=gene1",
            ]
        )
    )

    genes = reelprotein.build_orf_gene_index(gff_path)
    _, gene_entries = reelprotein.read_gff_gene_entries(gff_path)

    assert genes["chr1"][0].feature_types == {
        "five_prime_UTR",
        "CDS",
        "three_prime_UTR",
    }
    assert [feature.feature_type for feature in gene_entries["gene1"].subfeatures] == [
        "CDS",
        "three_prime_UTR",
        "mRNA",
        "five_prime_UTR",
    ]
