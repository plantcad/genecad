import importlib
import pandas as pd

from src.tests.reelprotein_test_support import install_reelprotein_dependency_stubs

install_reelprotein_dependency_stubs()

reelprotein = importlib.import_module("src.reelprotein")
reelprotein_old = importlib.import_module("src.reelprotein_old")


def test_old_parse_gff3_drops_transcript_parent_features(tmp_path):
    # This is a valid GFF3 shape: gene -> mRNA -> CDS/UTR, with the child features
    # pointing to the transcript ID instead of a `gene_id.*` identifier.
    # The old parser only matched parents that started with the gene ID, so it
    # silently discarded all coding features for transcripts named like
    # `transcriptA`. In practice that means ReelProtein can miss real ORFs for
    # otherwise valid annotations.
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

    old_genes = reelprotein_old.parse_gff3(gff_path)
    new_genes = reelprotein.build_orf_gene_index(gff_path)

    assert old_genes["chr1"][0]["feature_types"] == set()
    assert old_genes["chr1"][0]["cds"] == []
    assert new_genes["chr1"][0].feature_types == {
        "five_prime_UTR",
        "CDS",
        "three_prime_UTR",
    }


def test_old_generate_final_gff_emits_single_and_merged_gene(tmp_path):
    # This case happens when scoring keeps both an individual gene candidate and
    # a merged multi-gene candidate covering the same locus. The old code wrote
    # both records into the final GFF, which leaves contradictory annotations
    # for one region: the original single gene and the concatenated replacement.
    # That is not just noisy output; downstream consumers can treat them as two
    # distinct models at the same coordinates.
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
    old_output_path = tmp_path / "old_output.gff3"
    new_output_path = tmp_path / "new_output.gff3"
    predictions = pd.DataFrame(
        {
            "ProteinID": ["chr1_gene1_+", "chr1_gene1|gene2_+"],
            "Predicted_Label": [1, 1],
        }
    )

    reelprotein_old.generate_final_gff(predictions, gff_path, old_output_path)
    reelprotein.generate_final_gff(predictions, gff_path, new_output_path)

    old_gene_lines = [
        line for line in old_output_path.read_text().splitlines() if "\tgene\t" in line
    ]
    new_gene_lines = [
        line for line in new_output_path.read_text().splitlines() if "\tgene\t" in line
    ]

    assert old_gene_lines == [
        "chr1\tsrc\tgene\t1\t9\t.\t+\t.\tID=gene1",
        "chr1\tsrc\tgene\t1\t28\t.\t+\t.\tID=concat_gene1_gene2;Note=concatenated",
    ]
    assert new_gene_lines == [
        "chr1\tsrc\tgene\t1\t28\t.\t+\t.\tID=concat_gene1_gene2;Note=concatenated"
    ]
