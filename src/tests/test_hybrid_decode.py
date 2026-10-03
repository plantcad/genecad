"""Tests for hybrid decoding: plain Viterbi genome-wide plus local frame-aware repair."""

from pathlib import Path

import numpy as np
import pytest

from src import frame_crf as fh

pytest.importorskip("torch")
from src import hybrid_decode as hd  # noqa: E402
from src.modeling import token_transition_probs  # noqa: E402
from src.tests.segment_test_support import write_segments  # noqa: E402

IG, IN, U5, CDS, U3 = range(5)


# -------------------------------------------------------------------------------------------------
# Synthetic locus
# -------------------------------------------------------------------------------------------------
#
# Same frozen background as test_frame_crf.py.  One two-exon gene:
#
#   [0, 2000)     intergenic
#   [2000, 2200)  5' UTR
#   exon 1 CDS    VALID_CODING[:50]
#   intron        GT ... AG (200 nt)
#   exon 2 CDS    VALID_CODING[50:]
#   200 nt        3' UTR
#   2000 nt       intergenic

_BACKGROUND = (
    (Path(__file__).parent / "fixtures" / "frame_crf_background.txt")
    .read_text()
    .strip()
)
VALID_CODING = "ATG" + "GCT" * 28 + "TAA"  # 90 nt, 29 residues
SPLIT, INTRON_LEN = 50, 200

# 1-based features of that gene on the plus strand
UTR5 = ("five_prime_UTR", 2001, 2200)
CDS1 = ("CDS", 2201, 2250)
CDS2 = ("CDS", 2451, 2490)
UTR3 = ("three_prime_UTR", 2491, 2690)
PLUS_GENE = [UTR5, CDS1, CDS2, UTR3]


def build_locus():
    exon1, exon2 = VALID_CODING[:SPLIT], VALID_CODING[SPLIT:]
    intron = "GT" + _BACKGROUND[2200 : 2200 + INTRON_LEN - 4] + "AG"
    sequence = _BACKGROUND[0:2200] + exon1 + intron + exon2 + _BACKGROUND[2396:4596]
    labels = np.array(
        [IG] * 2000
        + [U5] * 200
        + [CDS] * len(exon1)
        + [IN] * INTRON_LEN
        + [CDS] * len(exon2)
        + [U3] * 200
        + [IG] * 2000
    )
    assert len(sequence) == len(labels)
    return sequence, labels


LENGTH = len(build_locus()[0])


def mirror(feats):
    """Features of the same gene after the chromosome is reverse complemented."""
    return sorted(
        ((t, LENGTH - e + 1, LENGTH - s + 1) for t, s, e in feats), key=lambda f: f[1]
    )


MINUS_GENE = mirror(PLUS_GENE)


def emissions(labels, confidence=0.9):
    probs = np.full((len(labels), 5), (1.0 - confidence) / 4.0)
    probs[np.arange(len(labels)), labels] = confidence
    return probs


def make_decoder(strand, labels):
    """A LocalDecoder over the synthetic chromosome, with the gene on `strand`.

    Logits for the other strand are all-intergenic."""
    sequence, _ = build_locus()
    if strand == "-":
        sequence = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        labels = labels[::-1].copy()
    logits = {
        strand: np.log(emissions(labels)),
        ("-" if strand == "+" else "+"): np.log(emissions(np.full(len(labels), IG))),
    }
    matrix = token_transition_probs(
        remove_incomplete_features=True, domain="plant"
    ).values
    return hd.LocalDecoder(
        lambda s, lo, hi: logits[s][lo:hi], fh.encode_sequence(sequence), matrix
    )


def gene(strand, feats, gene_id, partial=False):
    return hd.Gene(
        seqid="chr1", strand=strand, feats=list(feats), gene_id=gene_id, partial=partial
    )


# -------------------------------------------------------------------------------------------------
# transcripts_from_labels
# -------------------------------------------------------------------------------------------------


def test_labels_become_one_transcript_with_1_based_features():
    labels = np.array([IG, U5, U5, CDS, CDS, IN, CDS, U3, IG])
    assert hd.transcripts_from_labels(labels, lo=10, strand="+") == [
        [
            ("five_prime_UTR", 12, 13),
            ("CDS", 14, 15),
            ("CDS", 17, 17),
            ("three_prime_UTR", 18, 18),
        ]
    ]


def test_a_utr3_followed_by_a_new_start_begins_a_new_transcript():
    labels = np.array([CDS, U3, CDS])
    assert hd.transcripts_from_labels(labels, lo=0, strand="+") == [
        [("CDS", 1, 1), ("three_prime_UTR", 2, 2)],
        [("CDS", 3, 3)],
    ]


def test_minus_strand_transcripts_are_split_in_transcription_order():
    # Read right to left: CDS, 3'UTR, then a new CDS.  Transcripts come out
    # upstream first, each with its features in genome order.
    labels = np.array([CDS, U3, CDS])
    assert hd.transcripts_from_labels(labels, lo=0, strand="-") == [
        [("three_prime_UTR", 2, 2), ("CDS", 3, 3)],
        [("CDS", 1, 1)],
    ]


def test_transcripts_without_cds_are_dropped():
    labels = np.array([U5, U5, IG, U5, CDS])
    assert hd.transcripts_from_labels(labels, lo=0, strand="+") == [
        [("five_prime_UTR", 4, 4), ("CDS", 5, 5)]
    ]


# -------------------------------------------------------------------------------------------------
# LocalDecoder
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("strand,expected", [("+", PLUS_GENE), ("-", MINUS_GENE)])
def test_decoder_recovers_the_gene_in_genome_coordinates(strand, expected):
    _, labels = build_locus()
    decoder = make_decoder(strand, labels)
    lo, hi = 1500, 3200
    window = decoder.decode(strand, lo, hi)
    assert hd.transcripts_from_labels(window, lo, strand) == [expected]


def test_forbidding_intergenic_joins_two_halves_of_a_gene():
    """The intron is predicted intergenic, so a free decode cannot give one
    transcript; forbidding intergenic across the gap recovers the real gene."""
    _, labels = build_locus()
    labels[2250:2450] = IG
    decoder = make_decoder("+", labels)
    lo, hi = 1500, 3200

    free = hd.transcripts_from_labels(decoder.decode("+", lo, hi), lo, "+")
    assert PLUS_GENE not in free

    forced = decoder.decode("+", lo, hi, forbid=(2250, 2450))
    assert hd.transcripts_from_labels(forced, lo, "+") == [PLUS_GENE]


# -------------------------------------------------------------------------------------------------
# rescue
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("strand,expected", [("+", PLUS_GENE), ("-", MINUS_GENE)])
def test_rescue_replaces_a_partial_gene_with_the_local_frame_aware_decode(
    strand, expected
):
    _, labels = build_locus()
    decoder = make_decoder(strand, labels)
    broken = [("CDS", expected[1][1], expected[2][2])]  # one CDS over both exons
    genes, stats = hd.rescue(
        [gene(strand, broken, "g7", partial=True)],
        decoder,
        LENGTH,
        flank=1000,
        min_cds=33,
    )

    assert stats["partial genes rescued"] == 1
    assert len(genes) == 1
    assert genes[0].feats == expected
    assert genes[0].gene_id == "g7"
    assert genes[0].hybrid == "rescued"
    assert not genes[0].partial


def test_rescue_leaves_complete_genes_untouched():
    _, labels = build_locus()
    decoder = make_decoder("+", labels)
    complete = gene("+", [("CDS", 10, 21)], "g1")
    genes, stats = hd.rescue([complete], decoder, LENGTH, flank=1000, min_cds=33)
    assert genes == [complete]
    assert stats["partial genes rescued"] == 0


def test_an_unrescuable_partial_gene_is_dropped_or_kept_on_request():
    _, labels = build_locus()
    decoder = make_decoder("+", labels)
    nowhere = gene("+", [("CDS", 100, 400)], "g2", partial=True)  # intergenic region

    genes, stats = hd.rescue([nowhere], decoder, LENGTH, flank=50, min_cds=33)
    assert genes == []
    assert stats["partial genes dropped"] == 1

    genes, _ = hd.rescue(
        [nowhere], decoder, LENGTH, flank=50, min_cds=33, keep_partial=True
    )
    assert genes == [nowhere]


def test_a_rescued_gene_may_not_overlap_a_kept_gene_on_the_same_strand():
    _, labels = build_locus()
    decoder = make_decoder("+", labels)
    kept = gene("+", [("CDS", 2600, 2700)], "g1")
    partial = gene("+", [("CDS", 2201, 2490)], "g2", partial=True)

    genes, stats = hd.rescue([kept, partial], decoder, LENGTH, flank=1000, min_cds=33)
    assert genes == [kept]
    assert stats["rescue: overlaps a kept gene, dropped"] == 1


def test_a_rescued_gene_with_too_little_cds_is_dropped():
    _, labels = build_locus()
    decoder = make_decoder("+", labels)
    partial = gene("+", [("CDS", 2201, 2490)], "g2", partial=True)

    genes, stats = hd.rescue([partial], decoder, LENGTH, flank=1000, min_cds=90)
    assert genes == []
    assert stats["rescue: tiny, dropped"] == 1


# -------------------------------------------------------------------------------------------------
# merge
# -------------------------------------------------------------------------------------------------


def split_halves(strand):
    """The gene as two fragments, the intron predicted intergenic."""
    _, labels = build_locus()
    labels[2250:2450] = IG
    feats = PLUS_GENE if strand == "+" else MINUS_GENE
    left, right = feats[:2], feats[2:]
    if strand == "+":
        return labels, gene("+", left, "gA"), gene("+", right, "gB")
    return labels, gene("-", right, "gA"), gene("-", left, "gB")  # gA is upstream on -


@pytest.mark.parametrize("strand,expected", [("+", PLUS_GENE), ("-", MINUS_GENE)])
def test_merge_joins_two_fragments_of_one_gene(strand, expected):
    labels, a, b = split_halves(strand)
    decoder = make_decoder(strand, labels)

    genes, stats = hd.merge(
        [a, b], decoder, LENGTH, flank=1000, max_gap=20_000, min_cds=33
    )

    assert stats["pairs tried"] == 1
    assert stats["pairs merged"] == 1
    assert len(genes) == 1
    assert genes[0].feats == expected
    assert genes[0].gene_id == "gA"
    assert genes[0].hybrid == "merged"
    assert genes[0].merged_from == ("gA", "gB")


def test_merge_does_not_try_pairs_further_apart_than_max_gap():
    labels, a, b = split_halves("+")
    decoder = make_decoder("+", labels)
    genes, stats = hd.merge(
        [a, b], decoder, LENGTH, flank=1000, max_gap=100, min_cds=33
    )
    assert genes == [a, b]
    assert stats["pairs tried"] == 0


def test_merge_keeps_pairs_the_forced_decode_cannot_span():
    """Two confident, separate genes: forbidding intergenic between them does
    not produce one transcript covering both CDS midpoints."""
    _, labels = build_locus()
    decoder = make_decoder("+", labels)
    a = gene("+", PLUS_GENE, "gA")
    b = gene("+", [("CDS", 3500, 3600)], "gB")  # confidently intergenic region

    genes, stats = hd.merge(
        [a, b], decoder, LENGTH, flank=500, max_gap=20_000, min_cds=33
    )
    assert stats["pairs tried"] == 1
    assert stats["pairs merged"] == 0
    assert genes == [a, b]


def test_merge_never_joins_genes_on_opposite_strands():
    labels, a, _ = split_halves("+")
    decoder = make_decoder("+", labels)
    other = gene("-", [("CDS", 2451, 2490)], "gM")
    genes, stats = hd.merge(
        [a, other], decoder, LENGTH, flank=1000, max_gap=20_000, min_cds=33
    )
    assert stats["pairs tried"] == 0
    assert sorted(g.gene_id for g in genes) == ["gA", "gM"]


# -------------------------------------------------------------------------------------------------
# GFF input / output
# -------------------------------------------------------------------------------------------------

GFF = """##gff-version 3
chr1\tGeneCAD\tgene\t101\t200\t.\t+\t.\tID=chr1_gene_1
chr1\tGeneCAD\tmRNA\t101\t200\t.\t+\t.\tID=chr1_gene_1.t1;Parent=chr1_gene_1;orf_status=complete
chr1\tGeneCAD\tCDS\t101\t200\t.\t+\t0\tID=x;Parent=chr1_gene_1.t1
chr1\tGeneCAD\tgene\t301\t400\t.\t-\t.\tID=chr1_gene_2;partial=true
chr1\tGeneCAD\tmRNA\t301\t400\t.\t-\t.\tID=chr1_gene_2.t1;Parent=chr1_gene_2;partial=true
chr1\tGeneCAD\tfive_prime_UTR\t381\t400\t.\t-\t.\tParent=chr1_gene_2.t1
chr1\tGeneCAD\tCDS\t301\t380\t.\t-\t0\tParent=chr1_gene_2.t1
chr2\tGeneCAD\tgene\t5\t40\t.\t+\t.\tID=chr2_gene_1
chr2\tGeneCAD\tmRNA\t5\t40\t.\t+\t.\tID=chr2_gene_1.t1;Parent=chr2_gene_1
chr2\tGeneCAD\tthree_prime_UTR\t5\t40\t.\t+\t.\tParent=chr2_gene_1.t1
"""


def test_read_genes_groups_the_first_transcript_by_sequence(tmp_path):
    path = tmp_path / "in.gff"
    path.write_text(GFF)
    header, by_seqid = hd.read_genes(str(path))

    assert header == ["##gff-version 3"]
    assert list(by_seqid) == ["chr1", "chr2"]
    g1, g2 = by_seqid["chr1"]
    assert (g1.gene_id, g1.strand, g1.partial, g1.feats) == (
        "chr1_gene_1",
        "+",
        False,
        [("CDS", 101, 200)],
    )
    assert (g2.gene_id, g2.strand, g2.partial) == ("chr1_gene_2", "-", True)
    assert g2.feats == [("CDS", 301, 380), ("five_prime_UTR", 381, 400)]
    # a gene with no CDS is still read, so it can be passed through
    assert by_seqid["chr2"][0].feats == [("three_prime_UTR", 5, 40)]


def test_unchanged_genes_are_written_back_verbatim(tmp_path):
    path, out = tmp_path / "in.gff", tmp_path / "out.gff"
    path.write_text(GFF)
    header, by_seqid = hd.read_genes(str(path))
    hd.write_genes(str(out), header, by_seqid)
    assert out.read_text() == GFF


def test_new_genes_are_written_with_phases_and_hybrid_attributes(tmp_path):
    out = tmp_path / "out.gff"
    merged = hd.Gene(
        seqid="chr1",
        strand="-",
        feats=[
            ("three_prime_UTR", 90, 99),
            ("CDS", 100, 110),
            ("CDS", 200, 203),
            ("five_prime_UTR", 204, 210),
        ],
        gene_id="chr1_gene_5",
        hybrid="merged",
        merged_from=("chr1_gene_5", "chr1_gene_4"),
    )
    hd.write_genes(str(out), ["##gff-version 3"], {"chr1": [merged]})
    lines = [line.split("\t") for line in out.read_text().splitlines()[1:]]

    assert [f[2] for f in lines] == [
        "gene",
        "mRNA",
        "three_prime_UTR",
        "CDS",
        "CDS",
        "five_prime_UTR",
    ]
    assert (
        lines[0][8]
        == "ID=chr1_gene_5;hybrid=merged;merged_from=chr1_gene_5,chr1_gene_4"
    )
    assert lines[1][8] == "ID=chr1_gene_5.t1;Parent=chr1_gene_5;orf_status=complete"
    assert (lines[0][3], lines[0][4]) == ("90", "210")
    # Minus strand: the 4 nt CDS at 200-203 comes first in coding order (phase 0);
    # the 11 nt CDS at 100-110 starts 4 nt in, so its phase is (3 - 4 % 3) % 3 = 2.
    assert [f[7] for f in lines if f[2] == "CDS"] == ["2", "0"]
    assert all(f[8].endswith("Parent=chr1_gene_5.t1") for f in lines[2:])


# -------------------------------------------------------------------------------------------------
# scripts/hybrid_decode.py
# -------------------------------------------------------------------------------------------------


def load_script():
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[2] / "scripts" / "hybrid_decode.py"
    spec = importlib.util.spec_from_file_location("hybrid_decode_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["hybrid_decode_script"] = module
    spec.loader.exec_module(module)
    return module


def write_predictions(root, chrom, labels):
    """Predictions in pipeline layout: <root>/<chrom>/predictions_<chrom>/."""
    import xarray as xr

    from src.modeling import GeneClassifierConfig

    names = GeneClassifierConfig().token_entity_names_with_background()
    store = root / chrom / f"predictions_{chrom}" / "predictions.0.zarr"
    for strand, strand_labels in (
        ("positive", labels),
        ("negative", np.full(len(labels), IG)),
    ):
        ds = xr.Dataset(
            {
                "feature_logits": (
                    ["sequence", "feature"],
                    np.log(emissions(strand_labels)).astype(np.float32),
                )
            },
            coords={"sequence": np.arange(len(labels)), "feature": names},
            attrs={"chromosome_id": chrom},
        )
        ds.to_zarr(str(store), group=strand, zarr_format=2)


@pytest.mark.parametrize("workers", ["1", "2"])
def test_cli_rescues_a_partial_gene_and_passes_other_sequences_through(
    tmp_path, monkeypatch, workers
):
    sequence, labels = build_locus()
    (tmp_path / "genome.fa").write_text(f">chr1\n{sequence}\n>chr2\n{'A' * 100}\n")
    write_predictions(tmp_path, "chr1", labels)
    (tmp_path / "in.gff").write_text(
        "##gff-version 3\n"
        "chr1\tGeneCAD\tgene\t2201\t2490\t.\t+\t.\tID=chr1_gene_1;partial=true\n"
        "chr1\tGeneCAD\tmRNA\t2201\t2490\t.\t+\t.\tID=chr1_gene_1.t1;Parent=chr1_gene_1;partial=true\n"
        "chr1\tGeneCAD\tCDS\t2201\t2490\t.\t+\t0\tParent=chr1_gene_1.t1\n"
        "chr2\tGeneCAD\tgene\t11\t40\t.\t+\t.\tID=chr2_gene_1\n"
        "chr2\tGeneCAD\tmRNA\t11\t40\t.\t+\t.\tID=chr2_gene_1.t1;Parent=chr2_gene_1\n"
        "chr2\tGeneCAD\tCDS\t11\t40\t.\t+\t0\tParent=chr2_gene_1.t1\n"
    )
    script = load_script()
    monkeypatch.setattr(
        "sys.argv",
        [
            "hybrid_decode.py",
            "--input-gff",
            str(tmp_path / "in.gff"),
            "--input-fasta",
            str(tmp_path / "genome.fa"),
            "--predictions-root",
            str(tmp_path),
            "--output-gff",
            str(tmp_path / "out.gff"),
            "--workers",
            workers,
        ],
    )
    script.main()

    _, by_seqid = hd.read_genes(str(tmp_path / "out.gff"))
    (rescued,) = by_seqid["chr1"]
    assert rescued.gene_id == "chr1_gene_1"
    assert rescued.feats == PLUS_GENE
    assert not rescued.partial
    # chr2 has no predictions: its genes are written back unchanged
    assert [g.gene_id for g in by_seqid["chr2"]] == ["chr2_gene_1"]
    assert (
        "chr2\tGeneCAD\tCDS\t11\t40\t.\t+\t0\tParent=chr2_gene_1.t1"
        in (tmp_path / "out.gff").read_text()
    )


def write_segmented_predictions(root, chrom, labels, segment_length):
    """The same logits as write_predictions, in the layout the predict step writes."""
    logits = {
        "positive": np.log(emissions(labels)).astype(np.float32),
        "negative": np.log(emissions(np.full(len(labels), IG))).astype(np.float32),
    }
    write_segments(root / chrom / f"predictions_{chrom}", logits, segment_length, chrom)


@pytest.mark.parametrize("segment_length", [64, 1000, 10_000])
def test_cli_result_does_not_depend_on_how_predictions_are_stored(
    tmp_path, monkeypatch, segment_length
):
    sequence, labels = build_locus()
    (tmp_path / "genome.fa").write_text(f">chr1\n{sequence}\n")
    (tmp_path / "in.gff").write_text(
        "##gff-version 3\n"
        "chr1\tGeneCAD\tgene\t2201\t2490\t.\t+\t.\tID=chr1_gene_1;partial=true\n"
        "chr1\tGeneCAD\tmRNA\t2201\t2490\t.\t+\t.\tID=chr1_gene_1.t1;Parent=chr1_gene_1;partial=true\n"
        "chr1\tGeneCAD\tCDS\t2201\t2490\t.\t+\t0\tParent=chr1_gene_1.t1\n"
    )
    ranks, segments = tmp_path / "ranks", tmp_path / "segments"
    write_predictions(ranks, "chr1", labels)
    write_segmented_predictions(segments, "chr1", labels, segment_length)

    script = load_script()
    results = {}
    for name, root in [("ranks", ranks), ("segments", segments)]:
        output = tmp_path / f"{name}.gff"
        monkeypatch.setattr(
            "sys.argv",
            [
                "hybrid_decode.py",
                "--input-gff",
                str(tmp_path / "in.gff"),
                "--input-fasta",
                str(tmp_path / "genome.fa"),
                "--predictions-root",
                str(root),
                "--output-gff",
                str(output),
            ],
        )
        script.main()
        results[name] = output.read_text()

    assert results["segments"] == results["ranks"]
    _, by_seqid = hd.read_genes(str(tmp_path / "segments.gff"))
    (rescued,) = by_seqid["chr1"]
    assert rescued.feats == PLUS_GENE
    assert not rescued.partial


def partial_gene_gff(chroms):
    lines = ["##gff-version 3"]
    for chrom in chroms:
        gene = f"{chrom}_gene_1"
        lines += [
            f"{chrom}\tGeneCAD\tgene\t2201\t2490\t.\t+\t.\tID={gene};partial=true",
            f"{chrom}\tGeneCAD\tmRNA\t2201\t2490\t.\t+\t.\tID={gene}.t1;Parent={gene};partial=true",
            f"{chrom}\tGeneCAD\tCDS\t2201\t2490\t.\t+\t0\tParent={gene}.t1",
        ]
    return "\n".join(lines) + "\n"


def run_hybrid(script, monkeypatch, tmp_path, output, workers):
    monkeypatch.setattr(
        "sys.argv",
        [
            "hybrid_decode.py",
            "--input-gff",
            str(tmp_path / "in.gff"),
            "--input-fasta",
            str(tmp_path / "genome.fa"),
            "--predictions-root",
            str(tmp_path),
            "--output-gff",
            str(output),
            "--workers",
            str(workers),
        ],
    )
    script.main()
    return output.read_text()


def test_cli_reads_a_fasta_with_many_sequences_once_and_repairs_them_all(
    tmp_path, monkeypatch
):
    # More sequences than the workers read ahead, listed in the FASTA in the
    # reverse of the order of the GFF, with a record that has no genes at all
    sequence, labels = build_locus()
    chroms = [f"chr{i}" for i in range(1, 8)]
    records = [f">{name}\n{sequence}\n" for name in reversed(chroms)]
    records.insert(3, ">unannotated\nACGTACGT\n")
    (tmp_path / "genome.fa").write_text("".join(records))
    (tmp_path / "in.gff").write_text(partial_gene_gff(chroms))
    for name in chroms:
        write_predictions(tmp_path, name, labels)

    script = load_script()
    outputs = {
        workers: run_hybrid(
            script, monkeypatch, tmp_path, tmp_path / f"out{workers}.gff", workers
        )
        for workers in (1, 2, 3)
    }
    assert outputs[2] == outputs[1]
    assert outputs[3] == outputs[1]

    _, by_seqid = hd.read_genes(str(tmp_path / "out1.gff"))
    assert list(by_seqid) == chroms
    for name in chroms:
        (rescued,) = by_seqid[name]
        assert rescued.feats == PLUS_GENE
        assert not rescued.partial


def test_cli_stops_when_a_sequence_with_predictions_is_not_in_the_fasta(
    tmp_path, monkeypatch
):
    sequence, labels = build_locus()
    (tmp_path / "genome.fa").write_text(f">chr1\n{sequence}\n")
    (tmp_path / "in.gff").write_text(partial_gene_gff(["chr1", "chr2"]))
    for name in ("chr1", "chr2"):
        write_predictions(tmp_path, name, labels)

    with pytest.raises(ValueError, match="'chr2' not found"):
        run_hybrid(load_script(), monkeypatch, tmp_path, tmp_path / "out.gff", 1)


def test_cli_decoding_options_reach_the_frame_aware_graph(tmp_path, monkeypatch):
    """The synthetic gene's intron is 200 nt, so requiring 300 nt introns must
    stop the rescue from rebuilding it."""
    sequence, labels = build_locus()
    (tmp_path / "genome.fa").write_text(f">chr1\n{sequence}\n")
    write_predictions(tmp_path, "chr1", labels)
    (tmp_path / "in.gff").write_text(
        "##gff-version 3\n"
        "chr1\tGeneCAD\tgene\t2201\t2490\t.\t+\t.\tID=chr1_gene_1;partial=true\n"
        "chr1\tGeneCAD\tmRNA\t2201\t2490\t.\t+\t.\tID=chr1_gene_1.t1;Parent=chr1_gene_1;partial=true\n"
        "chr1\tGeneCAD\tCDS\t2201\t2490\t.\t+\t0\tParent=chr1_gene_1.t1\n"
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "hybrid_decode.py",
            "--input-gff",
            str(tmp_path / "in.gff"),
            "--input-fasta",
            str(tmp_path / "genome.fa"),
            "--predictions-root",
            str(tmp_path),
            "--output-gff",
            str(tmp_path / "out.gff"),
            "--min-intron-length",
            "300",
        ],
    )
    load_script().main()

    _, by_seqid = hd.read_genes(str(tmp_path / "out.gff"))
    assert all(g.feats != PLUS_GENE for g in by_seqid.get("chr1", []))
