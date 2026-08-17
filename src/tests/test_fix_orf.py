"""Tests for the ORF-aware CDS boundary repair step (``scripts/fix_orf.py``)."""

import importlib.util
import pathlib
import sys

import pytest

# scripts/ is not an importable package, so load the module by path.  It must be
# registered in sys.modules before execution, otherwise @dataclass cannot
# resolve the module namespace while processing the class bodies.
_MODULE_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "fix_orf.py"
_spec = importlib.util.spec_from_file_location("fix_orf", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
fix_orf = importlib.util.module_from_spec(_spec)
sys.modules["fix_orf"] = fix_orf
_spec.loader.exec_module(fix_orf)


# -------------------------------------------------------------------------------------------------
# Synthetic locus
# -------------------------------------------------------------------------------------------------
#
# A two-exon transcript with a known, unique ORF.  The spliced mRNA is 120 nt:
#
#     t[0, 10)    5' UTR    (contains no ATG, so the ORF start is unambiguous)
#     t[10, 100)  CDS       ATG + 28 x GCT + TAA  =  90 nt, 29 residues
#     t[100, 120) 3' UTR
#
# Exon 1 covers t[0, 60) and exon 2 covers t[60, 120), separated by a 60 nt
# intron.  Splitting the CDS 50 / 40 across the two exons makes the second CDS
# feature's GFF3 phase non-zero, which exercises the phase computation.

FIVE_PRIME_UTR_SEQ = "CAAACCCGGG"  # 10 nt, no ATG
ORF_SEQ = "ATG" + "GCT" * 28 + "TAA"  # 90 nt
THREE_PRIME_UTR_SEQ = "CCCAAACCCAAACCCAAACC"  # 20 nt, no ATG
MRNA_SEQ = FIVE_PRIME_UTR_SEQ + ORF_SEQ + THREE_PRIME_UTR_SEQ

CANONICAL_INTRON = "GT" + "T" * 56 + "AG"
NONCANONICAL_INTRON = "AA" + "T" * 56 + "CC"

FLANK = "T" * 100

# Genomic layout: FLANK | exon1 (101-160) | intron (161-220) | exon2 (221-280) | FLANK
EXON1 = (101, 160)
EXON2 = (221, 280)

# Feature blocks of the correct annotation, as (start, end, type, phase).
PLUS_CORRECT = [
    (101, 110, "five_prime_UTR", "."),
    (111, 160, "CDS", "0"),
    (221, 260, "CDS", "1"),
    (261, 280, "three_prime_UTR", "."),
]
MINUS_CORRECT = [
    (101, 120, "three_prime_UTR", "."),
    (121, 160, "CDS", "1"),
    (221, 270, "CDS", "0"),
    (271, 280, "five_prime_UTR", "."),
]

# The same locus with the CDS wrongly extended over the entire transcript, so
# it neither starts on ATG nor ends on a stop codon.
PLUS_BROKEN = [(101, 160, "CDS", "0"), (221, 280, "CDS", "0")]
MINUS_BROKEN = [(101, 160, "CDS", "0"), (221, 280, "CDS", "0")]


def build_chromosome(strand: str, intron: str = CANONICAL_INTRON) -> str:
    locus = MRNA_SEQ[:60] + intron + MRNA_SEQ[60:]
    assert len(locus) == 180
    if strand == "-":
        locus = fix_orf.revcomp(locus)
    return FLANK + locus + FLANK


def build_gff(blocks, strand: str) -> str:
    lines = [
        "##gff-version 3",
        f"chr1\ttest\tgene\t101\t280\t.\t{strand}\t.\tID=g1",
        f"chr1\ttest\tmRNA\t101\t280\t.\t{strand}\t.\tID=g1.t1;Parent=g1",
    ]
    for start, end, feature_type, phase in blocks:
        lines.append(
            f"chr1\ttest\t{feature_type}\t{start}\t{end}\t.\t{strand}\t{phase}\t"
            f"Parent=g1.t1"
        )
    return "\n".join(lines) + "\n"


def run(tmp_path, blocks, strand, intron=CANONICAL_INTRON, **kwargs):
    """Run fix_orf over a synthetic locus and return (stats, parsed records)."""
    gff = tmp_path / "in.gff"
    fasta = tmp_path / "genome.fa"
    out = tmp_path / "out.gff"
    gff.write_text(build_gff(blocks, strand))
    fasta.write_text(">chr1\n" + build_chromosome(strand, intron) + "\n")

    options: dict[str, object] = {
        "max_shift": 300,
        "min_protein_length": 10,
        "require_canonical": True,
        "report_path": None,
    }
    options.update(kwargs)
    stats = fix_orf.fix_orf(
        input_gff=str(gff),
        input_fasta=str(fasta),
        output_gff=str(out),
        **options,  # pyrefly: ignore[bad-argument-type]
    )
    _, records = fix_orf.read_gff(str(out))
    return stats, records


def exonic(records):
    return sorted(
        (r.start, r.end, r.type, r.phase)
        for r in records
        if r.type in fix_orf.EXONIC_TYPES
    )


def mrna_attributes(records):
    return next(r.attributes for r in records if r.type == "mRNA")


# -------------------------------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("strand,correct", [("+", PLUS_CORRECT), ("-", MINUS_CORRECT)])
def test_valid_orf_is_recognised_and_left_alone(tmp_path, strand, correct):
    """A transcript that already encodes a complete ORF must not be modified."""
    stats, records = run(tmp_path, correct, strand)

    assert stats["complete"] == 1
    assert stats["repaired"] == 0
    assert exonic(records) == sorted(correct)
    assert mrna_attributes(records)["orf_status"] == "complete"
    assert "partial" not in mrna_attributes(records)


@pytest.mark.parametrize(
    "strand,broken,correct",
    [("+", PLUS_BROKEN, PLUS_CORRECT), ("-", MINUS_BROKEN, MINUS_CORRECT)],
)
def test_broken_cds_is_repaired_to_the_true_orf(tmp_path, strand, broken, correct):
    """Moving only the TIS and TTS within the transcript recovers the real ORF,
    and the recovered CDS carries correct GFF3 phases."""
    stats, records = run(tmp_path, broken, strand)

    assert stats["repaired"] == 1
    assert exonic(records) == sorted(correct)

    attributes = mrna_attributes(records)
    assert attributes["orf_status"] == "repaired"
    assert attributes["orf_shift_5"] == "10"  # CDS start moved 10 nt downstream
    assert attributes["orf_shift_3"] == "-20"  # CDS end moved 20 nt upstream


def test_minus_strand_splicing_preserves_exon_order(tmp_path):
    """Regression: the spliced sequence must be assembled in ascending genomic
    order before reverse complementing.  Reverse complementing exons that are
    already in coding order silently reverses them, which yields a scrambled
    transcript and bogus 'repairs'."""
    _, records = run(tmp_path, MINUS_CORRECT, "-")
    transcript = fix_orf.Transcript(
        mrna=next(r for r in records if r.type == "mRNA"),
        children=[r for r in records if r.type in fix_orf.EXONIC_TYPES],
        seqid="chr1",
        strand="-",
    )
    transcript.build_exons()
    assert transcript.spliced_sequence(build_chromosome("-")) == MRNA_SEQ


@pytest.mark.parametrize("strand", ["+", "-"])
def test_exon_structure_and_transcript_span_are_never_changed(tmp_path, strand):
    """A repair may only relabel exonic sequence: exon blocks, and therefore
    every splice junction, must survive untouched."""
    broken = PLUS_BROKEN if strand == "+" else MINUS_BROKEN
    _, records = run(tmp_path, broken, strand)

    def exon_blocks(blocks):
        merged: list[list[int]] = []
        for start, end in sorted((b[0], b[1]) for b in blocks):
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [tuple(b) for b in merged]

    assert exon_blocks(exonic(records)) == [EXON1, EXON2]
    mrna = next(r for r in records if r.type == "mRNA")
    assert (mrna.start, mrna.end) == (101, 280)
    gene = next(r for r in records if r.type == "gene")
    assert (gene.start, gene.end) == (101, 280)


def test_noncanonical_introns_block_repair_by_default(tmp_path):
    """If the model's own splice calls are not canonical, no ORF is built on
    top of them; the transcript is flagged instead."""
    stats, records = run(tmp_path, PLUS_BROKEN, "+", intron=NONCANONICAL_INTRON)

    assert stats["repaired"] == 0
    assert stats["partial"] == 1
    assert exonic(records) == sorted(PLUS_BROKEN)

    attributes = mrna_attributes(records)
    assert attributes["orf_issue"] == "noncanonical_intron"
    assert attributes["partial"] == "true"
    assert attributes["start_range"] == ".,101"
    assert attributes["end_range"] == "280,."
    gene = next(r for r in records if r.type == "gene")
    assert gene.attributes["partial"] == "true"


def test_noncanonical_introns_can_be_opted_into(tmp_path):
    stats, records = run(
        tmp_path, PLUS_BROKEN, "+", intron=NONCANONICAL_INTRON, require_canonical=False
    )

    assert stats["repaired"] == 1
    assert exonic(records) == sorted(PLUS_CORRECT)


def test_max_shift_bounds_how_far_a_boundary_may_move(tmp_path):
    """The true ORF needs a 20 nt move at the 3' end, so a 5 nt cap must reject
    it rather than settle for some other ORF."""
    stats, records = run(tmp_path, PLUS_BROKEN, "+", max_shift=5)

    assert stats["repaired"] == 0
    assert stats["partial"] == 1
    assert exonic(records) == sorted(PLUS_BROKEN)
    assert mrna_attributes(records)["orf_issue"] == "orf_outside_window"


def test_transcript_with_no_possible_orf_is_reported_separately(tmp_path):
    """A transcript whose sequence contains no ATG at all is distinguished from
    one whose ORF merely lies outside the search window."""
    blocks = [(101, 160, "CDS", "0"), (221, 280, "CDS", "0")]
    gff = tmp_path / "in.gff"
    fasta = tmp_path / "genome.fa"
    out = tmp_path / "out.gff"
    gff.write_text(build_gff(blocks, "+"))
    # Exonic sequence with no ATG anywhere, intron left canonical
    locus = ("CCC" * 20) + CANONICAL_INTRON + ("CCC" * 20)
    fasta.write_text(">chr1\n" + FLANK + locus + FLANK + "\n")

    stats = fix_orf.fix_orf(
        input_gff=str(gff),
        input_fasta=str(fasta),
        output_gff=str(out),
        max_shift=300,
        min_protein_length=10,
        require_canonical=True,
        report_path=None,
    )

    assert stats["partial"] == 1
    assert stats["issue:no_orf_in_transcript"] == 1


def test_ambiguous_bases_are_not_repaired(tmp_path):
    """An ORF cannot be verified across an assembly gap, so N-containing
    transcripts are flagged rather than guessed at."""
    gff = tmp_path / "in.gff"
    fasta = tmp_path / "genome.fa"
    out = tmp_path / "out.gff"
    gff.write_text(build_gff(PLUS_BROKEN, "+"))
    chromosome = build_chromosome("+")
    # Introduce an N inside exon 2, away from the CDS boundaries
    chromosome = chromosome[:250] + "N" + chromosome[251:]
    fasta.write_text(">chr1\n" + chromosome + "\n")

    stats = fix_orf.fix_orf(
        input_gff=str(gff),
        input_fasta=str(fasta),
        output_gff=str(out),
        max_shift=300,
        min_protein_length=10,
        require_canonical=True,
        report_path=None,
    )

    assert stats["repaired"] == 0
    assert stats["issue:ambiguous_bases"] == 1


def test_sequences_absent_from_the_fasta_are_passed_through(tmp_path):
    gff = tmp_path / "in.gff"
    fasta = tmp_path / "genome.fa"
    out = tmp_path / "out.gff"
    gff.write_text(build_gff(PLUS_BROKEN, "+"))
    fasta.write_text(">other_contig\n" + "A" * 100 + "\n")

    stats = fix_orf.fix_orf(
        input_gff=str(gff),
        input_fasta=str(fasta),
        output_gff=str(out),
        max_shift=300,
        min_protein_length=10,
        require_canonical=True,
        report_path=None,
    )

    assert stats["skipped"] == 1
    _, records = fix_orf.read_gff(str(out))
    assert exonic(records) == sorted(PLUS_BROKEN)
    # An untouched transcript must not be labelled partial: we simply don't know
    assert "partial" not in mrna_attributes(records)


def test_first_in_frame_stop_terminates_the_orf():
    """A repaired ORF must stop at the first in-frame stop codon, never read
    through one."""
    seq = "AAA" + "ATG" + "GCT" * 5 + "TAA" + "GCT" * 5 + "TAG"
    next_stop = fix_orf.build_next_in_frame_stop(seq)
    assert next_stop[3] == 3 + 3 + 15  # the first TAA, not the trailing TAG

    orf = fix_orf.find_best_orf(
        seq, cds_begin=3, cds_stop=len(seq), max_shift=300, min_protein_length=1
    )
    assert orf == (3, 3 + 3 + 15 + 3)


def test_kozak_score_returns_none_outside_sequence_bounds():
    seq = "A" * 5 + "ATG" + "A" * 5  # 13 nt total
    # ATG is at offset 5; the real window needs 6 nt upstream, which isn't there
    assert fix_orf.kozak_score(seq, 5) is None


def test_kozak_score_returns_none_for_ambiguous_bases():
    seq = (
        "N" * fix_orf.KOZAK_WINDOW_UPSTREAM
        + "ATG"
        + "A" * fix_orf.KOZAK_WINDOW_DOWNSTREAM
    )
    assert fix_orf.kozak_score(seq, fix_orf.KOZAK_WINDOW_UPSTREAM) is None


def test_kozak_score_is_deterministic_and_finite():
    seq = "CGCGCG" + "ATG" + "GCGCGC"
    score = fix_orf.kozak_score(seq, 6)
    assert score is not None
    assert score == fix_orf.kozak_score(seq, 6)  # deterministic
    assert -1000 < score < 1000  # sane magnitude, not NaN/inf
