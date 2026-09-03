import pandas as pd
from Bio.SeqFeature import SeqFeature, FeatureLocation

from scripts.mmlr_prepare_junctions import Junc, get_junctions, load_gff, merge_entries


class FakeChrom:
    """Minimal stand-in for the SeqRecord objects returned by src.gff_parser.parse:
    get_junctions only ever touches `.features`."""

    def __init__(self, features):
        self.features = features


def make_gene(strand, five_prime_utr, cds, three_prime_utr, mrna_id="mRNA1"):
    """Build a gene -> mRNA -> {UTR,CDS} SeqFeature tree matching the shape
    produced by load_gff (sub_features sorted ascending by start, regardless
    of strand)."""
    sub_features = list(five_prime_utr) + list(cds) + list(three_prime_utr)
    sub_features.sort(key=lambda feat: feat.location.start)

    mRNA = SeqFeature(FeatureLocation(0, 1, strand=strand), type="mRNA", id=mrna_id)
    mRNA.sub_features = sub_features

    gene = SeqFeature(FeatureLocation(0, 1, strand=strand), type="gene")
    gene.sub_features = [mRNA]

    return gene


def feat(start, end, strand, ftype):
    return SeqFeature(FeatureLocation(start, end, strand=strand), type=ftype)


def junctions_as_pairs(df):
    """Helper: (pos, junction) pairs in emission order, easier to assert on."""
    return list(zip(df["pos"], df["junction"]))


class TestJuncEnum:
    def test_str_labels(self):
        assert str(Junc.TIS) == "TIS"
        assert str(Junc.TTS) == "TTS"
        assert str(Junc.DONOR) == "Donor"
        assert str(Junc.ACCEPTOR) == "Acceptor"


class TestGetJunctionsPlusStrand:
    def test_single_exon_no_utr(self):
        # CDS only, one exon: no introns anywhere, so only TIS/TTS emitted.
        cds = [feat(100, 200, 1, "CDS")]
        gene = make_gene(1, [], cds, [])
        gff = [FakeChrom([gene])]

        df = get_junctions(gff, 0, 0, 0)

        assert junctions_as_pairs(df) == [
            (100, Junc.TIS),
            (199, Junc.TTS),
        ]

    def test_multi_exon_cds_produces_donor_acceptor(self):
        # Two CDS exons with one intron between them (100-200, 300-400).
        cds = [feat(100, 200, 1, "CDS"), feat(300, 400, 1, "CDS")]
        gene = make_gene(1, [], cds, [])
        gff = [FakeChrom([gene])]

        df = get_junctions(gff, 0, 0, 0)

        assert junctions_as_pairs(df) == [
            (100, Junc.TIS),
            (200, Junc.DONOR),  # first base after CDS[0] ends
            (299, Junc.ACCEPTOR),  # base before CDS[1] starts
            (399, Junc.TTS),
        ]

    def test_multi_exon_utrs_produce_donor_acceptor(self):
        # 5'UTR spliced into two exons, 3'UTR spliced into two exons.
        five_utr = [
            feat(0, 50, 1, "five_prime_UTR"),
            feat(70, 100, 1, "five_prime_UTR"),
        ]
        cds = [feat(100, 200, 1, "CDS")]
        three_utr = [
            feat(200, 250, 1, "three_prime_UTR"),
            feat(300, 350, 1, "three_prime_UTR"),
        ]
        gene = make_gene(1, five_utr, cds, three_utr)
        gff = [FakeChrom([gene])]

        df = get_junctions(gff, 0, 0, 0)

        assert junctions_as_pairs(df) == [
            (50, Junc.DONOR),  # intron within 5'UTR
            (69, Junc.ACCEPTOR),
            (100, Junc.TIS),
            (199, Junc.TTS),
            (250, Junc.DONOR),  # intron within 3'UTR
            (299, Junc.ACCEPTOR),
        ]

    def test_intron_spanning_utr_cds_boundary_is_detected_when_utr_has_multiple_exons(
        self,
    ):
        # 5'UTR has 2 exons (so the "perfectly splits" check runs) and the
        # last UTR exon does NOT abut the first CDS exon: there is a genuine
        # intron between UTR and CDS which should be picked up as an extra
        # donor/acceptor pair before TIS.
        five_utr = [feat(0, 20, 1, "five_prime_UTR"), feat(30, 50, 1, "five_prime_UTR")]
        cds = [feat(100, 200, 1, "CDS")]
        gene = make_gene(1, five_utr, cds, [])
        gff = [FakeChrom([gene])]

        df = get_junctions(gff, 0, 0, 0)

        assert junctions_as_pairs(df) == [
            (20, Junc.DONOR),  # intron within 5'UTR
            (29, Junc.ACCEPTOR),
            (50, Junc.DONOR),  # intron between 5'UTR and CDS
            (99, Junc.ACCEPTOR),
            (100, Junc.TIS),
            (199, Junc.TTS),
        ]

    def test_intron_spanning_utr_cds_boundary_detected_even_with_single_exon_utr(self):
        """A genuine intron between the 5'UTR and the CDS must be reported as
        a donor/acceptor pair regardless of how many exons the UTR itself
        has. This is currently broken: get_junctions only looks for that
        intron when `len(five_prime_utr) > 1` (see the guard in the plus-
        strand branch), so a single-exon 5'UTR separated from the CDS by a
        real intron - a common gene structure - silently loses that
        junction.
        """
        five_utr = [feat(0, 20, 1, "five_prime_UTR")]  # single UTR exon
        cds = [feat(100, 200, 1, "CDS")]  # gap of 80bp before CDS: real intron
        gene = make_gene(1, five_utr, cds, [])
        gff = [FakeChrom([gene])]

        df = get_junctions(gff, 0, 0, 0)

        assert junctions_as_pairs(df) == [
            (20, Junc.DONOR),  # intron between 5'UTR and CDS
            (99, Junc.ACCEPTOR),
            (100, Junc.TIS),
            (199, Junc.TTS),
        ]


class TestGetJunctionsMinusStrand:
    def test_single_exon_no_utr(self):
        cds = [feat(100, 200, -1, "CDS")]
        gene = make_gene(-1, [], cds, [])
        gff = [FakeChrom([gene])]

        df = get_junctions(gff, 0, 0, 0)

        # Labels are swapped relative to plus strand: first CDS base (in
        # genomic order) is TTS, last is TIS.
        assert junctions_as_pairs(df) == [
            (100, Junc.TTS),
            (199, Junc.TIS),
        ]

    def test_multi_exon_cds_produces_donor_acceptor_swapped(self):
        cds = [feat(100, 200, -1, "CDS"), feat(300, 400, -1, "CDS")]
        gene = make_gene(-1, [], cds, [])
        gff = [FakeChrom([gene])]

        df = get_junctions(gff, 0, 0, 0)

        assert junctions_as_pairs(df) == [
            (100, Junc.TTS),
            (200, Junc.ACCEPTOR),  # swapped relative to plus strand
            (299, Junc.DONOR),
            (399, Junc.TIS),
        ]

    def test_multi_exon_utrs_produce_donor_acceptor_swapped(self):
        # On the minus strand, the 3'UTR sits at lower coordinates (it's
        # transcribed first, but in reverse) and the 5'UTR at higher ones.
        three_utr = [
            feat(0, 50, -1, "three_prime_UTR"),
            feat(70, 100, -1, "three_prime_UTR"),
        ]
        cds = [feat(100, 200, -1, "CDS")]
        five_utr = [
            feat(200, 250, -1, "five_prime_UTR"),
            feat(300, 350, -1, "five_prime_UTR"),
        ]
        gene = make_gene(-1, five_utr, cds, three_utr)
        gff = [FakeChrom([gene])]

        df = get_junctions(gff, 0, 0, 0)

        assert junctions_as_pairs(df) == [
            (50, Junc.ACCEPTOR),
            (69, Junc.DONOR),
            (100, Junc.TTS),
            (199, Junc.TIS),
            (250, Junc.ACCEPTOR),
            (299, Junc.DONOR),
        ]


class TestMergeEntries:
    def test_single_row_passes_through_unchanged(self):
        df = pd.DataFrame(
            {
                "chrom": [0],
                "gene": [0],
                "mRNA": [0],
                "pos": [100],
                "junction": [Junc.DONOR],
            }
        )
        group = ((0, 0, 100, Junc.DONOR), df)

        result = merge_entries(group)

        pd.testing.assert_frame_equal(result, df)

    def test_multiple_rows_merge_mrna_into_comma_list(self):
        df = pd.DataFrame(
            {
                "chrom": [0, 0],
                "gene": [0, 0],
                "mRNA": [0, 1],
                "pos": [100, 100],
                "junction": [Junc.DONOR, Junc.DONOR],
            }
        )
        group = ((0, 0, 100, Junc.DONOR), df)

        result = merge_entries(group)

        assert result.shape[0] == 1
        assert result.iloc[0]["mRNA"] == "0,1"
        assert result.iloc[0]["chrom"] == 0
        assert result.iloc[0]["gene"] == 0
        assert result.iloc[0]["pos"] == 100
        assert result.iloc[0]["junction"] == Junc.DONOR


GFF3_TWO_GENES = """##gff-version 3
chr1\ttest\tgene\t1\t400\t.\t+\t.\tID=gene1
chr1\ttest\tmRNA\t1\t400\t.\t+\t.\tID=mRNA1;Parent=gene1
chr1\ttest\tfive_prime_UTR\t1\t50\t.\t+\t.\tID=utr1;Parent=mRNA1
chr1\ttest\tthree_prime_UTR\t351\t400\t.\t+\t.\tID=utr2;Parent=mRNA1
chr1\ttest\tCDS\t51\t350\t.\t+\t.\tID=cds1;Parent=mRNA1
chr1\ttest\tgene\t500\t700\t.\t+\t.\tID=gene2
chr1\ttest\tncRNA\t500\t700\t.\t+\t.\tID=ncrna1;Parent=gene2
"""


class TestLoadGff:
    def test_drops_genes_without_protein_coding_transcript(self, tmp_path):
        gff_path = tmp_path / "test.gff3"
        gff_path.write_text(GFF3_TWO_GENES)

        gff = load_gff(str(gff_path))

        assert len(gff) == 1  # one chromosome
        chrom = gff[0]
        # gene2 (ncRNA only) should have been removed
        assert len(chrom.features) == 1
        assert chrom.features[0].id == "gene1"

    def test_mrna_sub_features_restricted_and_sorted(self, tmp_path):
        gff_path = tmp_path / "test.gff3"
        gff_path.write_text(GFF3_TWO_GENES)

        gff = load_gff(str(gff_path))
        gene = gff[0].features[0]
        mRNA = gene.sub_features[0]

        types = [feat.type for feat in mRNA.sub_features]
        assert types == ["five_prime_UTR", "CDS", "three_prime_UTR"]
        starts = [feat.location.start for feat in mRNA.sub_features]
        assert starts == sorted(starts)


class TestMainIntegration:
    def test_end_to_end_writes_expected_table(self, tmp_path, monkeypatch):
        import scripts.mmlr_prepare_junctions as mod

        gff_path = tmp_path / "test.gff3"
        gff_path.write_text(GFF3_TWO_GENES)
        out_path = tmp_path / "junctions.tsv"

        monkeypatch.setattr(
            "sys.argv",
            [
                "mmlr_prepare_junctions.py",
                "--input-gff",
                str(gff_path),
                "--output-table",
                str(out_path),
                "--num-workers",
                "1",
            ],
        )

        mod.main()

        assert out_path.exists()
        df = pd.read_csv(out_path, sep="\t")
        assert set(df.columns) == {"chrom", "gene", "mRNA", "pos", "junction"}
        # gene1 is single-exon (no introns): only TIS/TTS rows expected
        assert set(df["junction"]) == {"TIS", "TTS"}
        assert len(df) == 2
