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
        "fix_weak_starts": True,
        "weak_start_threshold": 9,
        "kozak_margin": 3.0,
        "weak_kozak_threshold": 5.0,
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


# -------------------------------------------------------------------------------------------------
# first_exon_length and find_best_start_by_kozak
# -------------------------------------------------------------------------------------------------


def test_first_exon_length_single_exon_cds():
    """CDS entirely within the transcript's last (only) exon: length is the
    full CDS span."""
    transcript = fix_orf.Transcript(
        mrna=None,
        children=[],
        seqid="chr1",
        strand="+",
        exon_offsets=[0],
        exons=[(101, 200)],
    )
    assert fix_orf.first_exon_length(transcript, begin=10, stop=90) == 80


def test_first_exon_length_cds_spanning_a_splice_junction():
    """CDS starts in the first exon and continues into the second: length is
    only the portion up to the first splice junction."""
    transcript = fix_orf.Transcript(
        mrna=None,
        children=[],
        seqid="chr1",
        strand="+",
        exon_offsets=[0, 10],
        exons=[(101, 110), (151, 200)],
    )
    assert fix_orf.first_exon_length(transcript, begin=4, stop=22) == 6


@pytest.fixture
def tiny_kozak_pwm(monkeypatch):
    """A 2-up/2-down window PWM that strongly prefers C at position -2 and
    position +2, and is neutral everywhere else -- gives every test in this
    file a predictable, hand-computable Kozak score."""
    monkeypatch.setattr(fix_orf, "KOZAK_WINDOW_UPSTREAM", 2)
    monkeypatch.setattr(fix_orf, "KOZAK_WINDOW_DOWNSTREAM", 2)
    strong = (-4.0, 4.0, -4.0, -4.0)  # prefers C
    neutral = (0.0, 0.0, 0.0, 0.0)
    monkeypatch.setattr(
        fix_orf,
        "KOZAK_PWM_LOG_ODDS",
        (strong, neutral, neutral, neutral, neutral, neutral, strong),
    )
    monkeypatch.setattr(fix_orf, "KOZAK_BACKGROUND", (0.25, 0.25, 0.25, 0.25))


# Shared 26 nt spliced sequence for find_best_start_by_kozak and
# repair_transcript tests below. Two in-frame ATGs, 9 nt (3 codons) apart so
# they share a reading frame and a stop codon:
#   t[0:4)    leading (no ATG)
#   t[4:7)    "weak" candidate ATG
#   t[7:10)   codon (Lys)
#   t[10:13)  codon (Ala) -- also carries the "strong" candidate's -2/-1 context
#   t[13:16)  "strong" candidate ATG (also codon 3 of the weak-start ORF)
#   t[16:19)  codon (Ser/Leu) -- also carries the "strong" candidate's +1/+2 context
#   t[19:22)  TAA stop, shared by both candidates (same frame)
#   t[22:26)  trailing (no ATG)
# "WEAK_STRONG": weak candidate has poor Kozak context, strong candidate has
# good Kozak context -- this is the sequence used by the "should repair" test.
WEAK_STRONG_SEQ = "CCAAATGAAAGCGATGTCATAACCCC"
# "STRONG_WEAK": flip which candidate has the good context -- the original
# (short) start is well-supported and must be left alone.
STRONG_WEAK_SEQ = "CCCAATGACAGAGATGTTATAACCCC"
# "WEAK_WEAK": neither candidate has good context -- nothing to switch to,
# should be flagged instead.
WEAK_WEAK_SEQ = "CCAAATGAAAGAGATGTTATAACCCC"


def test_find_best_start_by_kozak_prefers_the_better_scoring_candidate(tiny_kozak_pwm):
    result = fix_orf.find_best_start_by_kozak(
        WEAK_STRONG_SEQ, cds_begin=4, cds_stop=22, max_shift=300, min_protein_length=1
    )
    assert result is not None
    begin, stop, score = result
    assert (begin, stop) == (13, 22)
    assert score == pytest.approx(8.0)


def test_find_best_start_by_kozak_can_return_the_original_when_it_scores_best(
    tiny_kozak_pwm,
):
    result = fix_orf.find_best_start_by_kozak(
        STRONG_WEAK_SEQ, cds_begin=4, cds_stop=22, max_shift=300, min_protein_length=1
    )
    assert result is not None
    begin, stop, score = result
    assert (begin, stop) == (4, 22)
    assert score == pytest.approx(8.0)


def test_find_best_start_by_kozak_returns_none_outside_the_shift_window(tiny_kozak_pwm):
    """The strong candidate is 9 nt from cds_begin; a 3 nt window excludes it,
    so only the (weak) original is a valid candidate."""
    result = fix_orf.find_best_start_by_kozak(
        WEAK_STRONG_SEQ, cds_begin=4, cds_stop=22, max_shift=3, min_protein_length=1
    )
    assert result is not None
    begin, stop, score = result
    assert (begin, stop) == (4, 22)
    assert score == pytest.approx(-8.0)


def test_find_best_start_by_kozak_never_returns_an_out_of_frame_candidate(
    tiny_kozak_pwm,
):
    """Regression for the frame-changing switch bug: an out-of-frame ATG with
    a strong Kozak context must lose to an in-frame ATG with an equally
    strong context, even though the out-of-frame candidate is found first
    and would win under a pure score comparison.

    Layout (offsets in a 45 nt sequence, all non-codon positions filled with
    neutral 'C'):
      offset 6   cds_begin's own ATG, weak context (non-C at -2/+2) -> -8.0
      offset 13  out-of-frame ATG (13 - 6 = 7, not a multiple of 3), strong
                 context (default 'C' filler at -2/+2) -> +8.0, with its own
                 downstream in-frame (class 1 mod 3) TAA stop at offset 22
      offset 30  in-frame ATG (30 - 6 = 24, a multiple of 3), strong context
                 -> +8.0, sharing cds_begin's downstream in-frame (class 0
                 mod 3) TAA stop at offset 39
    Without the frame filter, offset 13 is encountered first during the scan
    and ties offset 30's score, so it (wrongly) wins.  With the filter,
    offset 13 is skipped entirely and offset 30 is the correct winner.
    """
    seq = list("C" * 45)

    def set_codon(i: int, codon: str) -> None:
        seq[i : i + 3] = list(codon)

    cds_begin = 6
    set_codon(cds_begin, "ATG")
    seq[cds_begin - 2] = "A"  # weak: non-C at -2
    seq[cds_begin + 4] = "A"  # weak: non-C at +2

    out_of_frame = cds_begin + 7  # 13, not a multiple of 3 from cds_begin
    set_codon(out_of_frame, "ATG")

    in_frame_alt = cds_begin + 24  # 30, a multiple of 3 from cds_begin
    set_codon(in_frame_alt, "ATG")

    set_codon(22, "TAA")  # first in-frame stop for out_of_frame's own class
    set_codon(39, "TAA")  # first in-frame stop for cds_begin/in_frame_alt's class

    seq = "".join(seq)

    result = fix_orf.find_best_start_by_kozak(
        seq, cds_begin=cds_begin, cds_stop=42, max_shift=300, min_protein_length=1
    )

    assert result is not None
    begin, stop, score = result
    assert (begin - cds_begin) % 3 == 0, (
        f"find_best_start_by_kozak returned an out-of-frame candidate: "
        f"begin={begin}, cds_begin={cds_begin}"
    )
    assert (begin, stop) == (in_frame_alt, 42)
    assert score == pytest.approx(8.0)
    # The frame-preserving property this fix guarantees: a same-frame switch
    # can never move the stop codon.
    assert stop == 42


def test_find_best_start_by_kozak_rejects_a_disjoint_upstream_orf(tiny_kozak_pwm):
    """Regression, found on real Oropetium thomaeum production data: a
    same-frame ATG is not automatically part of the *same* ORF. An ATG
    upstream of cds_begin, still inside the exonic (e.g. 5' UTR) sequence,
    can have its own short in-frame stop before ever reaching cds_begin --
    a disjoint little ORF (a real uORF, or just incidental sequence) that
    happens to score well on Kozak context alone, since kozak_score() only
    looks at the +/-6 nt immediately around the ATG. Switching to it would
    silently replace the gene's actual protein with this unrelated
    fragment. find_best_start_by_kozak must reject any candidate whose own
    forced stop is not cds_stop itself, even when it shares cds_begin's
    reading frame and out-scores the real start.

    Layout (40 nt, 'C' filler elsewhere):
      offset 8   disjoint upstream ATG, strong context (default 'C' filler
                 at -2/+2) -> +8.0, but its own first in-frame stop is at
                 offset 14 (TAA) -- 9 nt later, nowhere near cds_stop.
      offset 20  cds_begin, weak context (non-C at -2/+2) -> -8.0, whose
                 real forced stop is the TAA at offset 32 (cds_stop = 35).
    Offset 8 is in cds_begin's reading frame (20 - 8 = 12) and scores
    higher, so without the stop-preserving filter it would wrongly win.
    """
    seq = list("C" * 40)

    def set_codon(i: int, codon: str) -> None:
        seq[i : i + 3] = list(codon)

    set_codon(8, "ATG")  # disjoint upstream ATG, strong context
    set_codon(14, "TAA")  # its own forced stop -- not cds_stop

    cds_begin = 20
    set_codon(cds_begin, "ATG")
    seq[cds_begin - 2] = "A"  # weak: non-C at -2
    seq[cds_begin + 4] = "A"  # weak: non-C at +2
    set_codon(32, "TAA")  # cds_begin's real forced stop
    cds_stop = 35

    seq = "".join(seq)

    result = fix_orf.find_best_start_by_kozak(
        seq, cds_begin=cds_begin, cds_stop=cds_stop, max_shift=300, min_protein_length=1
    )

    assert result is not None
    begin, stop, score = result
    assert stop == cds_stop, (
        f"find_best_start_by_kozak returned a candidate whose forced stop "
        f"({stop}) is not the original cds_stop ({cds_stop}) -- it switched "
        f"to a disjoint ORF instead of a different start for the same one."
    )
    # With the disjoint candidate correctly excluded, only the (weak)
    # original remains a valid candidate.
    assert (begin, stop, score) == (cds_begin, cds_stop, pytest.approx(-8.0))


# -------------------------------------------------------------------------------------------------
# repair_transcript weak-start end-to-end wiring
# -------------------------------------------------------------------------------------------------

# Genomic layout for the weak-start tests below (plus strand):
#   FLANK(100) | exon1 (101-110, 10nt) | intron (111-150, 40nt canonical) |
#   exon2 (151-166, 16nt) | FLANK(100)
# Spliced transcript = exon1 + exon2 = the same 26 nt sequences used above.
WEAK_START_INTRON = "GT" + "T" * 36 + "AG"  # 40 nt, canonical
WEAK_START_NONCANONICAL_INTRON = "AA" + "T" * 36 + "CC"  # 40 nt, non-canonical

# Original (input) annotation: CDS = genomic (105,110)+(151,162), a 6 nt
# first coding exon -- already a *complete*, valid ORF (matches
# WEAK_STRONG_SEQ / STRONG_WEAK_SEQ / WEAK_WEAK_SEQ's t[4:22) ORF), so it
# will hit the new is_complete_orf branch rather than the existing repair
# path.
WEAK_START_BLOCKS = [
    (101, 104, "five_prime_UTR", "."),
    (105, 110, "CDS", "0"),
    (151, 162, "CDS", "0"),
    (163, 166, "three_prime_UTR", "."),
]


def build_weak_start_chromosome(
    spliced_seq: str, intron: str = WEAK_START_INTRON
) -> str:
    exon1, exon2 = spliced_seq[:10], spliced_seq[10:]
    locus = exon1 + intron + exon2
    assert len(locus) == 10 + 40 + 16
    return FLANK + locus + FLANK


def run_weak_start(tmp_path, spliced_seq, intron=WEAK_START_INTRON, **kwargs):
    gff = tmp_path / "in.gff"
    fasta = tmp_path / "genome.fa"
    out = tmp_path / "out.gff"
    gff.write_text(build_gff(WEAK_START_BLOCKS, "+"))
    fasta.write_text(
        ">chr1\n" + build_weak_start_chromosome(spliced_seq, intron) + "\n"
    )

    options: dict[str, object] = {
        "max_shift": 300,
        "min_protein_length": 1,
        "require_canonical": True,
        "report_path": None,
        "fix_weak_starts": True,
        "weak_start_threshold": 9,
        "kozak_margin": 3.0,
        "weak_kozak_threshold": 5.0,
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


def test_weak_start_with_a_better_candidate_is_repaired(tiny_kozak_pwm, tmp_path):
    stats, records = run_weak_start(tmp_path, WEAK_STRONG_SEQ)

    assert stats["repaired"] == 1
    assert exonic(records) == sorted(
        [
            (101, 110, "five_prime_UTR", "."),
            (151, 153, "five_prime_UTR", "."),
            (154, 162, "CDS", "0"),
            (163, 166, "three_prime_UTR", "."),
        ]
    )
    attributes = mrna_attributes(records)
    assert attributes["orf_status"] == "repaired"
    assert attributes["orf_issue"] == "weak_start_kozak"


def test_weak_start_noncanonical_intron_blocks_switch_by_default(
    tiny_kozak_pwm, tmp_path
):
    """The same sequence/candidate that gets switched in
    test_weak_start_with_a_better_candidate_is_repaired must NOT be switched
    when the transcript's intron is non-canonical: an active repair needs
    trustworthy splice calls, same as the pre-existing invalid-ORF repair
    path. The transcript is left as its already-valid original ORF."""
    stats, records = run_weak_start(
        tmp_path, WEAK_STRONG_SEQ, intron=WEAK_START_NONCANONICAL_INTRON
    )

    assert stats["repaired"] == 0
    assert stats["complete"] == 1
    assert exonic(records) == sorted(WEAK_START_BLOCKS)
    attributes = mrna_attributes(records)
    assert attributes["orf_status"] == "complete"
    assert "orf_issue" not in attributes


def test_weak_start_noncanonical_intron_switch_can_be_opted_into(
    tiny_kozak_pwm, tmp_path
):
    """The same non-canonical-intron transcript IS switched when
    require_canonical=False, mirroring
    test_noncanonical_introns_can_be_opted_into for the existing repair
    path."""
    stats, records = run_weak_start(
        tmp_path,
        WEAK_STRONG_SEQ,
        intron=WEAK_START_NONCANONICAL_INTRON,
        require_canonical=False,
    )

    assert stats["repaired"] == 1
    assert exonic(records) == sorted(
        [
            (101, 110, "five_prime_UTR", "."),
            (151, 153, "five_prime_UTR", "."),
            (154, 162, "CDS", "0"),
            (163, 166, "three_prime_UTR", "."),
        ]
    )
    attributes = mrna_attributes(records)
    assert attributes["orf_status"] == "repaired"
    assert attributes["orf_issue"] == "weak_start_kozak"


def test_genuinely_well_supported_short_first_exon_is_left_alone(
    tiny_kozak_pwm, tmp_path
):
    stats, records = run_weak_start(tmp_path, STRONG_WEAK_SEQ)

    assert stats["complete"] == 1
    assert stats["repaired"] == 0
    assert exonic(records) == sorted(WEAK_START_BLOCKS)
    attributes = mrna_attributes(records)
    assert attributes["orf_status"] == "complete"
    assert "orf_issue" not in attributes


def test_weak_start_with_no_good_candidate_is_flagged_not_repaired(
    tiny_kozak_pwm, tmp_path
):
    stats, records = run_weak_start(tmp_path, WEAK_WEAK_SEQ)

    assert stats["complete"] == 1
    assert stats["repaired"] == 0
    assert exonic(records) == sorted(WEAK_START_BLOCKS)
    attributes = mrna_attributes(records)
    assert attributes["orf_status"] == "complete"
    assert attributes["orf_issue"] == "weak_kozak_support"


def test_fix_weak_starts_can_be_opted_out(tiny_kozak_pwm, tmp_path):
    """The same sequence that gets repaired by default must be left
    completely alone with --no-fix-weak-starts (fix_weak_starts=False)."""
    stats, records = run_weak_start(tmp_path, WEAK_STRONG_SEQ, fix_weak_starts=False)

    assert stats["repaired"] == 0
    assert stats["complete"] == 1
    assert exonic(records) == sorted(WEAK_START_BLOCKS)
    assert "orf_issue" not in mrna_attributes(records)


def test_weak_start_search_does_not_trigger_for_a_long_first_exon(
    tiny_kozak_pwm, tmp_path
):
    """A transcript whose first exon is already >= weak_start_threshold must
    not be touched, even with a weak Kozak score -- this only reuses
    WEAK_STRONG_SEQ's sequence bytes but treats the whole thing as single
    long exon by not splitting it across a splice junction, so
    first_exon_length is the full 18 nt CDS length, well above the
    threshold."""
    gff = tmp_path / "in.gff"
    fasta = tmp_path / "genome.fa"
    out = tmp_path / "out.gff"
    # Single-exon layout: the whole spliced sequence in one genomic block.
    blocks = [
        (101, 104, "five_prime_UTR", "."),
        (105, 122, "CDS", "0"),
        (123, 126, "three_prime_UTR", "."),
    ]
    gff.write_text(build_gff(blocks, "+"))
    fasta.write_text(">chr1\n" + FLANK + WEAK_STRONG_SEQ + FLANK + "\n")
    stats = fix_orf.fix_orf(
        input_gff=str(gff),
        input_fasta=str(fasta),
        output_gff=str(out),
        max_shift=300,
        min_protein_length=1,
        require_canonical=True,
        report_path=None,
        fix_weak_starts=True,
        weak_start_threshold=9,
        kozak_margin=3.0,
        weak_kozak_threshold=0.0,
    )
    assert stats["repaired"] == 0
    _, records = fix_orf.read_gff(str(out))
    assert exonic(records) == sorted(blocks)


# -------------------------------------------------------------------------------------------------
# calibrate_kozak_margin
# -------------------------------------------------------------------------------------------------


def test_calibrate_kozak_margin_falls_back_when_too_few_confident_transcripts(
    tmp_path,
):
    """An empty (or near-empty) genome has nothing to calibrate against --
    calibrate_kozak_margin must return the default unchanged rather than
    over-fitting a margin to a handful of transcripts."""
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\n" + FLANK + "\n")
    margin = fix_orf.calibrate_kozak_margin(
        by_seqid={},
        input_fasta=str(fasta),
        max_shift=300,
        min_protein_length=1,
        weak_start_threshold=9,
        default_margin=3.0,
    )
    assert margin == 3.0


def build_confident_loci_chromosome(n: int) -> tuple[str, str]:
    """n single-exon transcripts on chr1, each using WEAK_STRONG_SEQ's ORF: a
    poor-Kozak-context start that is nonetheless the correct one, because the
    exon it sits in is long enough that first_exon_length never dips below a
    realistic weak_start_threshold. Every one of these is exactly the kind of
    already-known-correct, unambiguous start calibrate_kozak_margin measures
    against -- and because WEAK_STRONG_SEQ's alternate (offset 13) ATG always
    outscores the true start, every one registers as a transcript the
    uncalibrated default margin would wrongly switch away from.

    Returns (gff_text, fasta_text).
    """
    locus_len = len(WEAK_STRONG_SEQ)
    gap = 20
    lines = ["##gff-version 3"]
    body = []
    for i in range(n):
        start = 101 + i * (locus_len + gap)
        gid, tid = f"g{i}", f"g{i}.t1"
        lines.append(
            f"chr1\ttest\tgene\t{start}\t{start + locus_len - 1}\t.\t+\t.\tID={gid}"
        )
        lines.append(
            f"chr1\ttest\tmRNA\t{start}\t{start + locus_len - 1}\t.\t+\t.\tID={tid};Parent={gid}"
        )
        lines.append(
            f"chr1\ttest\tfive_prime_UTR\t{start}\t{start + 3}\t.\t+\t.\tParent={tid}"
        )
        lines.append(
            f"chr1\ttest\tCDS\t{start + 4}\t{start + 21}\t.\t+\t0\tParent={tid}"
        )
        lines.append(
            f"chr1\ttest\tthree_prime_UTR\t{start + 22}\t{start + locus_len - 1}\t.\t+\t.\tParent={tid}"
        )
        body.append(WEAK_STRONG_SEQ + "T" * gap)
    gff = "\n".join(lines) + "\n"
    fasta = ">chr1\n" + FLANK + "".join(body) + FLANK + "\n"
    return gff, fasta


def run_calibration_scenario(tmp_path, n_confident, calibrate_margin):
    """n_confident WEAK_STRONG_SEQ loci on chr1 (see
    build_confident_loci_chromosome), plus one genuinely ambiguous
    short-first-exon WEAK_STRONG_SEQ transcript on chr2 (same shape as
    test_weak_start_with_a_better_candidate_is_repaired). Returns the
    ambiguous transcript's mRNA attributes."""
    confident_gff, confident_fasta = build_confident_loci_chromosome(n_confident)
    # g1/g1.t1 collides with build_confident_loci_chromosome's own i=1 locus
    # (Parent lookups are by ID regardless of seqid) -- rename to a unique ID.
    ambiguous_gff = (
        build_gff(WEAK_START_BLOCKS, "+")
        .replace("chr1", "chr2")
        .replace("Parent=g1.t1", "Parent=gamb.t1")
        .replace("ID=g1.t1;Parent=g1", "ID=gamb.t1;Parent=gamb")
        .replace("ID=g1\n", "ID=gamb\n")
    )
    ambiguous_fasta = ">chr2\n" + build_weak_start_chromosome(WEAK_STRONG_SEQ) + "\n"

    gff = tmp_path / "in.gff"
    fasta = tmp_path / "genome.fa"
    out = tmp_path / "out.gff"
    gff.write_text(confident_gff + "\n".join(ambiguous_gff.splitlines()[1:]) + "\n")
    fasta.write_text(confident_fasta + ambiguous_fasta)

    fix_orf.fix_orf(
        input_gff=str(gff),
        input_fasta=str(fasta),
        output_gff=str(out),
        max_shift=300,
        min_protein_length=1,
        require_canonical=True,
        report_path=None,
        fix_weak_starts=True,
        weak_start_threshold=9,
        kozak_margin=3.0,
        weak_kozak_threshold=0.0,
        calibrate_margin=calibrate_margin,
    )
    _, records = fix_orf.read_gff(str(out))
    return next(r.attributes for r in records if r.type == "mRNA" and r.seqid == "chr2")


def test_calibrated_margin_blocks_a_switch_the_default_margin_would_make(
    tiny_kozak_pwm, tmp_path
):
    """250 confident transcripts all show the default margin would switch
    away from a known-correct start (WEAK_STRONG_SEQ's alternate always
    outscores its true start) -- calibration should pick this up and raise
    the margin enough that the same switch no longer fires on a genuinely
    ambiguous transcript with an identical score gap."""
    attributes = run_calibration_scenario(tmp_path, 250, calibrate_margin=True)
    assert attributes["orf_status"] == "complete"
    assert (
        "orf_issue" not in attributes or attributes["orf_issue"] != "weak_start_kozak"
    )


def test_calibration_can_be_turned_off(tiny_kozak_pwm, tmp_path):
    """The same scenario with calibration disabled falls back to the fixed
    default margin, which switches -- confirming the previous test's
    "left alone" result comes from calibration, not from something else in
    the fixture blocking the switch."""
    attributes = run_calibration_scenario(tmp_path, 250, calibrate_margin=False)
    assert attributes["orf_status"] == "repaired"
    assert attributes["orf_issue"] == "weak_start_kozak"
