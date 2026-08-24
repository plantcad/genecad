#!/usr/bin/env python3
"""ORF-aware repair of CDS boundaries in GeneCAD predictions.

GeneCAD decodes per-base feature labels (intergenic / intron / 5'UTR / CDS /
3'UTR) and never reads the underlying nucleotide sequence, so a predicted CDS
is not guaranteed to be a translatable ORF: it may not start on ATG, may not
end on a stop codon, may not be a multiple of three, or may carry an in-frame
stop.  This step repairs those models using the genome sequence.

The repair is deliberately constrained so that it cannot invent gene structure:

1.  Exon structure is frozen.  Exons are derived by merging *contiguous*
    predicted features (CDS / 5'UTR / 3'UTR); every remaining gap is an intron
    and is never moved, split, merged or created.  Consequently every splice
    junction in the output is a junction the model itself predicted.
2.  Only the TIS and the TTS move, and they move *within the spliced mRNA* —
    i.e. only across sequence the model already called exonic.  A repair can
    therefore only relabel predicted UTR as CDS or predicted CDS as UTR; it can
    never pull in intronic or intergenic sequence.
3.  Internal introns must be canonical (GT-AG / GC-AG / AT-AC) for a repair to
    be attempted.  If the model's own splice calls are not trustworthy, neither
    is any ORF built on top of them.  (Relax with --allow-noncanonical-introns.)
    This applies to rule 6's weak-start switch too: a non-canonical intron
    blocks the switch (but not the weak_kozak_support flag, which changes no
    gene structure).
4.  The repaired ORF must be fully valid: ATG start, length % 3 == 0, no
    in-frame stop, stop codon end.
5.  Among valid solutions the one closest to the original prediction wins
    (minimum total boundary movement), and movement is capped by --max-shift.
    Truncating a long CDS down to a short ORF is therefore rejected: it would
    require a large 3' shift.
6.  Even when the predicted CDS is already a valid ORF, if its first *coding*
    exon (i.e. the CDS portion of the first exon it overlaps, not counting any
    5'UTR on that exon) is shorter than --weak-start-threshold, alternative
    start codons within the same already-predicted exonic sequence are
    considered and the one with the strongest Kozak-context support is
    preferred, provided it beats the original by --kozak-margin. --kozak-margin
    is a floor, not a fixed value: by default it is raised per genome by
    calibrate_kozak_margin, which checks the same switch logic against this
    genome's own unambiguous start codons and raises the margin only as far
    as needed to keep the measured rate of switching away from a known-correct
    start at or below 2% (see calibrate_kozak_margin's docstring for the
    cross-species numbers behind that -- one held-out species had a
    quiet-but-real 8-9% false-positive rate at the fixed default that this
    catches). Disable with --no-calibrate-kozak-margin. Candidates
    are restricted to the original start's reading frame *and* to those
    whose own forced stop is the original stop codon unchanged, so a switch
    can only move the TIS -- it can never change the stop codon or the
    encoded protein downstream of the new start. This still only relabels predicted
    exonic sequence (rule 2 still applies) -- it never invents new gene
    structure. Transcripts whose best candidate is still weak
    (--weak-kozak-threshold) are flagged (orf_issue=weak_kozak_support)
    rather than forced. Disable with --no-fix-weak-starts.

Transcripts that cannot be repaired under these rules are *not* forced into an
ORF.  They are passed through unchanged and flagged (partial=true, orf_issue=…,
plus GFF3 start_range / end_range) so downstream steps can filter them.

Usage
-----
    python scripts/fix_orf.py -i raw.gff -f genome.fa -o fixed.gff \\
        [--report status.tsv] [--max-shift 300]
"""

from __future__ import annotations

import argparse
import gzip
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

STOP_CODONS = frozenset(("TAA", "TAG", "TGA"))
START_CODON = "ATG"

# Canonical intron donor/acceptor dinucleotide pairs, read 5'->3' on the
# transcribed strand.  GT-AG and GC-AG are U2-type; AT-AC is U12-type.
CANONICAL_SPLICE_PAIRS = frozenset((("GT", "AG"), ("GC", "AG"), ("AT", "AC")))

# Log2-odds Kozak-context PWM: fit once, offline, from real confirmed start
# codons pooled across multiple Phytozome plant species. Deliberately
# excludes Oropetium thomaeum, the species used to validate this whole
# change -- fitting and validating on the same species would be circular.
# Window is [-KOZAK_WINDOW_UPSTREAM, ATG, +KOZAK_WINDOW_DOWNSTREAM] in
# coding (5'->3') orientation, columns A/C/G/T.
KOZAK_WINDOW_UPSTREAM = 6
KOZAK_WINDOW_DOWNSTREAM = 6
KOZAK_PWM_LOG_ODDS: tuple[tuple[float, float, float, float], ...] = (
    (0.0136, -0.0756, 0.3939, -0.2537),
    (-0.0890, 0.5300, -0.0483, -0.2820),
    (0.3170, 0.0057, 0.2045, -0.5907),
    (0.4690, -0.5523, 0.5644, -0.9349),
    (0.2857, 0.7449, -0.6555, -0.7332),
    (0.2614, 0.4346, 0.3240, -1.1425),
    (1.6575, -19.6135, -19.6135, -20.4058),
    (-20.4059, -19.6135, -19.6135, 1.6576),
    (-20.4059, -19.6135, 2.4499, -20.4058),
    (-0.4964, -0.9217, 1.5662, -1.2134),
    (-0.0608, 0.9640, -0.0891, -0.9244),
    (-0.4841, -0.3068, 0.8155, -0.0636),
    (0.0288, -0.1188, 0.6590, -0.5329),
    (-0.1478, 0.6393, 0.0373, -0.3954),
    (-0.3735, 0.2646, 0.4629, -0.1626),
)
# Background base composition the PWM above was fit against. Not read by
# kozak_score(): KOZAK_PWM_LOG_ODDS is already log-odds (foreground/background),
# so the background is already baked into those numbers. Kept only as fitting
# provenance/documentation -- do not wire it into the scoring math.
KOZAK_BACKGROUND: tuple[float, float, float, float] = (0.3170, 0.1830, 0.1830, 0.3170)

# Consensus: AAAAAAATGGCGAAT  (positions -6..ATG..+6)
# Fit from 438262 confirmed start codons pooled across 11 species: Athaliana,
# Bstricta, Crubella, Esalsugineum, Fvesca, Csativus, Cclementina,
# Mtruncatula, Ljaponicus, Dcarota, Osativa.


def kozak_score(seq: str, atg_offset: int) -> float | None:
    """Log2-odds Kozak-context score for the ATG at transcript offset atg_offset.

    None if the window falls outside seq, or contains a base other than
    A/C/G/T (an assembly gap in the window makes the score meaningless).
    """
    lo = atg_offset - KOZAK_WINDOW_UPSTREAM
    hi = atg_offset + 3 + KOZAK_WINDOW_DOWNSTREAM
    if lo < 0 or hi > len(seq):
        return None
    window = seq[lo:hi]
    score = 0.0
    for i, base in enumerate(window):
        if base not in "ACGT":
            return None
        score += KOZAK_PWM_LOG_ODDS[i]["ACGT".index(base)]
    return score


CDS = "CDS"
FIVE_PRIME_UTR = "five_prime_UTR"
THREE_PRIME_UTR = "three_prime_UTR"
MRNA = "mRNA"
GENE = "gene"

EXONIC_TYPES = (CDS, FIVE_PRIME_UTR, THREE_PRIME_UTR)

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


# -------------------------------------------------------------------------------------------------
# GFF records
# -------------------------------------------------------------------------------------------------


@dataclass
class Record:
    """A single GFF3 feature line, kept in a mutable form."""

    seqid: str
    source: str
    type: str
    start: int  # 1-based inclusive
    end: int  # 1-based inclusive
    score: str
    strand: str
    phase: str
    attributes: dict[str, str]
    order: int  # original line number, used to preserve output ordering

    @property
    def id(self) -> str | None:
        return self.attributes.get("ID")

    @property
    def parent(self) -> str | None:
        return self.attributes.get("Parent")

    def to_line(self) -> str:
        attrs = ";".join(f"{k}={v}" for k, v in self.attributes.items())
        return "\t".join(
            [
                self.seqid,
                self.source,
                self.type,
                str(self.start),
                str(self.end),
                self.score,
                self.strand,
                self.phase,
                attrs if attrs else ".",
            ]
        )


def parse_attributes(text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    if text in (".", ""):
        return attrs
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        key, sep, value = part.partition("=")
        if sep:
            attrs[key] = value
    return attrs


def read_gff(path: str) -> tuple[list[str], list[Record]]:
    """Read a GFF3 file into a header block and a list of records."""
    header: list[str] = []
    records: list[Record] = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:  # pyrefly: ignore[bad-argument-type]
        for order, line in enumerate(fh):
            if line.startswith("#"):
                # Only keep header comments that precede the first feature
                if not records:
                    header.append(line.rstrip("\n"))
                continue
            line = line.rstrip("\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) != 9:
                logger.warning(f"Skipping malformed line {order + 1}: {line[:80]!r}")
                continue
            records.append(
                Record(
                    seqid=fields[0],
                    source=fields[1],
                    type=fields[2],
                    start=int(fields[3]),
                    end=int(fields[4]),
                    score=fields[5],
                    strand=fields[6],
                    phase=fields[7],
                    attributes=parse_attributes(fields[8]),
                    order=order,
                )
            )
    if not header:
        header = ["##gff-version 3"]
    return header, records


# -------------------------------------------------------------------------------------------------
# Transcript model
# -------------------------------------------------------------------------------------------------


@dataclass
class Transcript:
    """An mRNA plus its exonic children, indexed for coordinate conversion."""

    mrna: Record
    children: list[Record]
    seqid: str
    strand: str
    # Exons in 5'->3' coding order as 1-based inclusive genomic (start, end)
    exons: list[tuple[int, int]] = field(default_factory=list)
    # Transcript-coordinate offset at which each exon begins
    exon_offsets: list[int] = field(default_factory=list)
    length: int = 0

    def build_exons(self) -> None:
        """Merge contiguous exonic features into exons, ordered 5'->3'."""
        blocks = sorted(
            (r.start, r.end) for r in self.children if r.type in EXONIC_TYPES
        )
        merged: list[tuple[int, int]] = []
        for start, end in blocks:
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        if self.strand == "-":
            merged.reverse()
        self.exons = merged
        offsets: list[int] = []
        total = 0
        for start, end in merged:
            offsets.append(total)
            total += end - start + 1
        self.exon_offsets = offsets
        self.length = total

    def genomic_to_transcript(self, pos: int) -> int | None:
        """Convert a 1-based genomic position to a 0-based transcript offset."""
        for (start, end), offset in zip(self.exons, self.exon_offsets):
            if start <= pos <= end:
                return offset + (pos - start if self.strand == "+" else end - pos)
        return None

    def transcript_to_blocks(self, begin: int, stop: int) -> list[tuple[int, int]]:
        """Map a half-open transcript interval to 1-based inclusive genomic blocks.

        Blocks are returned in ascending genomic order.
        """
        blocks: list[tuple[int, int]] = []
        if begin >= stop:
            return blocks
        for (gstart, gend), offset in zip(self.exons, self.exon_offsets):
            exon_len = gend - gstart + 1
            lo = max(begin, offset)
            hi = min(stop, offset + exon_len)
            if lo >= hi:
                continue
            a, b = lo - offset, hi - offset  # offsets within the exon, half-open
            if self.strand == "+":
                blocks.append((gstart + a, gstart + b - 1))
            else:
                blocks.append((gend - b + 1, gend - a))
        blocks.sort()
        return blocks

    def spliced_sequence(self, chrom: str) -> str:
        """Extract the spliced mRNA sequence in coding orientation.

        ``self.exons`` is stored in 5'->3' coding order, so on the minus strand
        it runs in descending genomic order.  Splicing must be done in ascending
        genomic order before reverse complementing, since
        ``revcomp(a + b) == revcomp(b) + revcomp(a)`` would otherwise reverse the
        exon order a second time.
        """
        ordered = self.exons if self.strand == "+" else self.exons[::-1]
        seq = "".join(chrom[start - 1 : end] for start, end in ordered).upper()
        return seq if self.strand == "+" else revcomp(seq)

    def introns(self) -> list[tuple[int, int]]:
        """Introns as 1-based inclusive genomic (start, end), 5'->3' coding order."""
        result: list[tuple[int, int]] = []
        for first, second in zip(self.exons, self.exons[1:]):
            if self.strand == "+":
                result.append((first[1] + 1, second[0] - 1))
            else:
                result.append((second[1] + 1, first[0] - 1))
        return result


# -------------------------------------------------------------------------------------------------
# ORF logic
# -------------------------------------------------------------------------------------------------


def has_internal_stop(seq: str, begin: int, stop: int) -> bool:
    """True if seq[begin:stop] contains an in-frame stop codon before its last codon."""
    return any(seq[i : i + 3] in STOP_CODONS for i in range(begin, stop - 3, 3))


def is_complete_orf(seq: str, begin: int, stop: int, min_protein_length: int) -> bool:
    length = stop - begin
    if length % 3 != 0 or length < 3 * (min_protein_length + 1):
        return False
    if seq[begin : begin + 3] != START_CODON:
        return False
    if seq[stop - 3 : stop] not in STOP_CODONS:
        return False
    return not has_internal_stop(seq, begin, stop)


def build_next_in_frame_stop(seq: str) -> list[int]:
    """For each offset i, the offset of the first in-frame stop codon at or after i.

    Returns -1 where no in-frame stop exists.  Computed in O(len(seq)) by
    scanning backwards within each of the three reading frames.
    """
    n = len(seq)
    next_stop = [-1] * (n + 1)
    for start in range(max(n - 2, 0), -1, -1):
        nxt = next_stop[start + 3] if start + 3 <= n else -1
        next_stop[start] = start if seq[start : start + 3] in STOP_CODONS else nxt
    return next_stop


def find_best_orf(
    seq: str,
    cds_begin: int,
    cds_stop: int,
    max_shift: int | None,
    min_protein_length: int,
) -> tuple[int, int] | None:
    """Find the valid ORF whose boundaries move least from (cds_begin, cds_stop).

    Given a start offset the ORF end is forced: it must terminate at the *first*
    in-frame stop, otherwise the ORF would contain an internal stop.  The search
    is therefore one-dimensional over candidate ATG positions.

    Parameters
    ----------
    seq
        Spliced mRNA sequence in coding orientation.
    cds_begin, cds_stop
        Half-open transcript interval of the predicted CDS.
    max_shift
        Maximum allowed movement of either boundary, or None for unbounded
        (used to distinguish "no ORF at all" from "ORF outside the window").
    min_protein_length
        Minimum protein length in residues, excluding the stop codon.

    Returns
    -------
    tuple[int, int] | None
        Half-open transcript interval of the best ORF, or None.
    """
    next_stop = build_next_in_frame_stop(seq)
    if max_shift is None:
        lo, hi = 0, len(seq)
    else:
        lo = max(0, cds_begin - max_shift)
        hi = min(len(seq), cds_begin + max_shift + 1)

    best: tuple[int, int] | None = None
    best_key: tuple[int, int, int] | None = None
    for begin in range(lo, hi):
        if seq[begin : begin + 3] != START_CODON:
            continue
        stop_offset = next_stop[begin]
        if stop_offset < 0:
            continue
        stop = stop_offset + 3
        if stop - begin < 3 * (min_protein_length + 1):
            continue
        if max_shift is not None and abs(stop - cds_stop) > max_shift:
            continue
        # Prefer the smallest total boundary movement; break ties toward the
        # longer ORF, then toward the smaller start offset, for determinism.
        cost = abs(begin - cds_begin) + abs(stop - cds_stop)
        key = (cost, -(stop - begin), begin)
        if best_key is None or key < best_key:
            best_key, best = key, (begin, stop)
    return best


def first_exon_length(transcript: Transcript, begin: int, stop: int) -> int:
    """Length, in transcript coordinates, of the CDS's first coding exon.

    The CDS is already known contiguous within the spliced transcript (the
    caller has already ruled out ``discontiguous_cds``), so the first splice
    junction after ``begin`` -- the smallest exon offset strictly greater
    than ``begin`` -- caps the first coding exon, or ``stop`` does if
    ``begin`` falls in the CDS's last exon.
    """
    later_offsets = [o for o in transcript.exon_offsets if o > begin]
    first_exon_end = min(later_offsets) if later_offsets else stop
    return min(first_exon_end, stop) - begin


def find_best_start_by_kozak(
    seq: str,
    cds_begin: int,
    cds_stop: int,
    max_shift: int,
    min_protein_length: int,
) -> tuple[int, int, float] | None:
    """Best-Kozak-scoring valid ORF within max_shift of (cds_begin, cds_stop),
    among candidates that preserve both cds_begin's reading frame and
    cds_stop exactly.

    Same candidate space as find_best_orf -- every in-frame ATG within the
    window, each with its forced downstream in-frame stop -- but ranked by
    kozak_score() instead of boundary movement, and restricted to
    candidates that (a) share cds_begin's reading frame and (b) whose
    forced downstream stop is cds_stop itself, unchanged. This repair only
    re-picks *which* start codon within an already-valid ORF is used; it
    must never change the reading frame or the stop codon, either of which
    would silently replace the protein with an unrelated one chosen on 5'
    evidence alone.

    Both restrictions are necessary, not just the frame one: a same-frame
    candidate upstream of cds_begin (still within the exonic sequence, e.g.
    in the 5' UTR) is not guaranteed to reach cds_stop without hitting an
    earlier in-frame stop first -- that region was never covered by the
    original ORF's own validity check, so it may harbor a short, unrelated
    upstream ORF (a real uORF, or just incidental sequence) that happens to
    have a strong Kozak context purely because Kozak scoring only looks at
    +/-6 nt around the ATG, independent of what follows. Requiring
    stop == cds_stop explicitly rules those out, rather than relying on it
    holding "naturally" -- confirmed against real genome-scale data that it
    does not hold naturally for every same-frame candidate in the window.
    Candidates whose Kozak window falls outside the sequence (kozak_score
    returns None) are skipped: there is nothing to compare.

    Returns
    -------
    tuple[int, int, float] | None
        (begin, stop, kozak_score) for the winner, or None if no valid
        candidate in the window has a computable Kozak score.
    """
    next_stop = build_next_in_frame_stop(seq)
    lo = max(0, cds_begin - max_shift)
    hi = min(len(seq), cds_begin + max_shift + 1)

    best: tuple[int, int, float] | None = None
    for begin in range(lo, hi):
        if (begin - cds_begin) % 3 != 0:
            continue
        if seq[begin : begin + 3] != START_CODON:
            continue
        stop_offset = next_stop[begin]
        if stop_offset < 0:
            continue
        stop = stop_offset + 3
        if stop != cds_stop:
            # The candidate's own downstream path hits an earlier (or
            # later) in-frame stop than the original -- it is not a
            # different start for the same ORF, it is a different ORF.
            continue
        if stop - begin < 3 * (min_protein_length + 1):
            continue
        if abs(stop - cds_stop) > max_shift:
            continue
        score = kozak_score(seq, begin)
        if score is None:
            continue
        if best is None or score > best[2]:
            best = (begin, stop, score)
    return best


# -------------------------------------------------------------------------------------------------
# Repair
# -------------------------------------------------------------------------------------------------


@dataclass
class Result:
    status: str
    issue: str = ""
    shift5: int = 0
    shift3: int = 0
    missing_start: bool = False
    missing_stop: bool = False


def rebuild_children(
    transcript: Transcript, begin: int, stop: int, source_records: list[Record]
) -> list[Record]:
    """Regenerate CDS and UTR records for a new CDS interval in transcript coords."""
    mrna_id = transcript.mrna.id or ""
    template = source_records[0]
    order = min(r.order for r in source_records)

    spans = [
        (FIVE_PRIME_UTR, transcript.transcript_to_blocks(0, begin)),
        (CDS, transcript.transcript_to_blocks(begin, stop)),
        (THREE_PRIME_UTR, transcript.transcript_to_blocks(stop, transcript.length)),
    ]

    # GFF3 phase: number of bases to remove from the start of a CDS feature to
    # reach the first complete codon, accumulated in 5'->3' coding order.
    cds_blocks = spans[1][1]
    coding_order = sorted(cds_blocks, reverse=(transcript.strand == "-"))
    phases: dict[tuple[int, int], int] = {}
    cumulative = 0
    for block in coding_order:
        phases[block] = (-cumulative) % 3
        cumulative += block[1] - block[0] + 1

    records: list[Record] = []
    for feature_type, blocks in spans:
        for index, (start, end) in enumerate(blocks, start=1):
            records.append(
                Record(
                    seqid=transcript.seqid,
                    source=template.source,
                    type=feature_type,
                    start=start,
                    end=end,
                    score=template.score,
                    strand=transcript.strand,
                    phase=str(phases[(start, end)]) if feature_type == CDS else ".",
                    attributes={
                        "ID": f"{mrna_id}.{feature_type}.{index}",
                        "Parent": mrna_id,
                    },
                    order=order,
                )
            )
    records.sort(key=lambda r: (r.start, r.end))
    return records


def has_noncanonical_introns(transcript: Transcript, chrom: str) -> bool:
    """True if any of the transcript's introns has a non-canonical donor/acceptor pair."""
    for intron_start, intron_end in transcript.introns():
        if intron_end < intron_start:
            return True
        donor = chrom[intron_start - 1 : intron_start + 1].upper()
        acceptor = chrom[intron_end - 2 : intron_end].upper()
        if transcript.strand == "-":
            donor, acceptor = revcomp(acceptor), revcomp(donor)
        if (donor, acceptor) not in CANONICAL_SPLICE_PAIRS:
            return True
    return False


def repair_transcript(
    transcript: Transcript,
    chrom: str,
    max_shift: int,
    min_protein_length: int,
    require_canonical: bool,
    fix_weak_starts: bool,
    weak_start_threshold: int,
    kozak_margin: float,
    weak_kozak_threshold: float,
) -> tuple[Result, list[Record] | None]:
    """Evaluate and, if needed and possible, repair one transcript's CDS."""
    located = locate_cds(transcript, chrom)
    if isinstance(located, Result):
        return located, None
    seq, begin, stop = located

    missing_start = seq[begin : begin + 3] != START_CODON
    missing_stop = seq[stop - 3 : stop] not in STOP_CODONS
    ambiguous = "N" in seq[begin:stop]

    if not ambiguous and is_complete_orf(seq, begin, stop, min_protein_length):
        if (
            fix_weak_starts
            and first_exon_length(transcript, begin, stop) < weak_start_threshold
        ):
            original_score = kozak_score(seq, begin)
            alt = find_best_start_by_kozak(
                seq, begin, stop, max_shift, min_protein_length
            )
            # An active repair still needs trustworthy splice calls to build
            # on, exactly like the invalid-ORF repair path below -- but a
            # non-canonical intron only blocks the *switch*, not the
            # informational weak_kozak_support flag, since flagging never
            # changes gene structure.
            blocked_by_noncanonical_introns = require_canonical and (
                has_noncanonical_introns(transcript, chrom)
            )
            if (
                not blocked_by_noncanonical_introns
                and alt is not None
                and original_score is not None
                and alt[2] > original_score + kozak_margin
            ):
                new_begin, new_stop, _ = alt
                records = rebuild_children(
                    transcript, new_begin, new_stop, transcript.children
                )
                return (
                    Result(
                        "repaired",
                        issue="weak_start_kozak",
                        shift5=new_begin - begin,
                        shift3=new_stop - stop,
                    ),
                    records,
                )
            candidate_scores = [
                s for s in (original_score, alt[2] if alt else None) if s is not None
            ]
            if candidate_scores and max(candidate_scores) < weak_kozak_threshold:
                return Result("complete", issue="weak_kozak_support"), None
        return Result("complete"), None

    if "N" in seq:
        # An ORF cannot be verified across an assembly gap, and picking a start
        # or stop codon adjacent to one would be guesswork.
        return (
            Result(
                "partial",
                "ambiguous_bases",
                missing_start=missing_start,
                missing_stop=missing_stop,
            ),
            None,
        )

    # Introns must be trustworthy before an ORF is built on top of them.
    if has_noncanonical_introns(transcript, chrom) and require_canonical:
        return (
            Result(
                "partial",
                "noncanonical_intron",
                missing_start=missing_start,
                missing_stop=missing_stop,
            ),
            None,
        )

    orf = find_best_orf(seq, begin, stop, max_shift, min_protein_length)
    if orf is None:
        # Distinguish "structurally impossible" from "outside the search window"
        # so the two failure modes can be triaged separately.
        unbounded = find_best_orf(seq, begin, stop, None, min_protein_length)
        issue = "no_orf_in_transcript" if unbounded is None else "orf_outside_window"
        return (
            Result(
                "partial",
                issue,
                missing_start=missing_start,
                missing_stop=missing_stop,
            ),
            None,
        )

    new_begin, new_stop = orf
    records = rebuild_children(transcript, new_begin, new_stop, transcript.children)
    return Result("repaired", shift5=new_begin - begin, shift3=new_stop - stop), records


# -------------------------------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------------------------------


def iter_fasta(path: str):
    """Yield (seqid, sequence) pairs, one chromosome at a time."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:  # pyrefly: ignore[bad-argument-type]
        seqid: str | None = None
        chunks: list[str] = []
        for line in fh:
            if line.startswith(">"):
                if seqid is not None:
                    yield seqid, "".join(chunks)
                seqid = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if seqid is not None:
            yield seqid, "".join(chunks)


def locate_cds(transcript: Transcript, chrom: str) -> tuple[str, int, int] | Result:
    """(spliced_seq, begin, stop) for a transcript's predicted CDS in transcript
    coordinates, or a Result describing why it couldn't be located. stop is
    exclusive. Shared setup between repair_transcript and
    calibrate_kozak_margin -- the latter only cares whether this succeeded,
    not why it failed, and skips the transcript on any Result return."""
    cds_records = [r for r in transcript.children if r.type == CDS]
    if not cds_records:
        return Result("skipped", "no_cds")
    transcript.build_exons()
    if transcript.length == 0:
        return Result("skipped", "no_exons")
    seq = transcript.spliced_sequence(chrom)
    if len(seq) != transcript.length:
        return Result("skipped", "sequence_length_mismatch")
    coding_first = min(r.start for r in cds_records)
    coding_last = max(r.end for r in cds_records)
    if transcript.strand == "+":
        begin = transcript.genomic_to_transcript(coding_first)
        stop_inclusive = transcript.genomic_to_transcript(coding_last)
    else:
        begin = transcript.genomic_to_transcript(coding_last)
        stop_inclusive = transcript.genomic_to_transcript(coding_first)
    if begin is None or stop_inclusive is None:
        return Result("skipped", "cds_outside_exons")
    stop = stop_inclusive + 1
    cds_length = sum(r.end - r.start + 1 for r in cds_records)
    if stop - begin != cds_length:
        # The predicted CDS is not contiguous within the spliced transcript,
        # which means UTR was called between two CDS blocks.  Repairing that
        # would require changing the model's own feature layout.
        return Result("partial", "discontiguous_cds")
    return seq, begin, stop


# Target false-positive rate for calibrate_kozak_margin: the fraction of a
# genome's own already-correct start codons the calibrated margin is allowed
# to risk switching away from. 2% was picked the same way exon_length
# strictness and the Kozak margin/threshold defaults were: bracketed against
# real cross-species validation, not guessed. See calibrate_kozak_margin.
KOZAK_MARGIN_TARGET_FP_RATE = 0.02


def calibrate_kozak_margin(
    by_seqid: dict[str, list[Transcript]],
    input_fasta: str,
    max_shift: int,
    min_protein_length: int,
    weak_start_threshold: int,
    default_margin: float,
    min_confident: int = 200,
) -> float:
    """Pick --kozak-margin for this genome instead of trusting one fixed
    value for every species.

    A single cross-species Kozak PWM does not carry the same reliability
    into every genome it is applied to -- the model's own weak-start
    candidates are ambiguous by construction, so there is no direct way to
    check the margin against them, but every genome already has thousands of
    transcripts whose start is *not* ambiguous (first coding exon well clear
    of weak_start_threshold, complete ORF, canonical introns). Those are
    known-correct starts. Running the same find_best_start_by_kozak logic
    against them, at the default margin, measures directly how often this
    genome's Kozak scores would talk the rule into switching away from a
    start already known to be right -- a per-genome reliability estimate
    that needs no assumption about what a "universal" Kozak signal looks
    like, and no borrowed cross-species data.

    The margin is raised only as far as needed to keep that measured
    false-positive rate at or below KOZAK_MARGIN_TARGET_FP_RATE (the
    (1 - KOZAK_MARGIN_TARGET_FP_RATE) quantile of the score-gap distribution
    over known-correct starts), never lowered below default_margin.

    Cross-species validation (Othomaeum, Rcommunis, Cquinoa, Oeuropaea;
    reference-annotated, held out from PWM fitting): at the fixed default
    margin, aggregate precision on switches that actually changed TIS
    correctness was 66.3% (63 correct / 32 wrong), but Cquinoa alone was
    net negative (9 correct / 12 wrong) -- its own confident starts showed
    an 8-9% false-positive rate against the default margin, far above the
    other three species. Per-genome calibration raised Cquinoa's margin
    enough to stop triggering wrong switches there (0/0) while leaving the
    other three species' margins close to the default, moving aggregate
    precision to 78.3% (18/23) with no species left net negative.

    Falls back to default_margin if there are too few confident transcripts
    (under 200) to calibrate against reliably -- e.g. a single small
    scaffold processed on its own.
    """
    gaps: list[float] = []
    n_confident = 0

    for seqid, chrom in iter_fasta(input_fasta):
        if seqid not in by_seqid:
            continue
        for transcript in by_seqid[seqid]:
            located = locate_cds(transcript, chrom)
            if isinstance(located, Result):
                continue
            seq, begin, stop = located
            if "N" in seq[begin:stop]:
                continue
            if not is_complete_orf(seq, begin, stop, min_protein_length):
                continue
            if first_exon_length(transcript, begin, stop) < weak_start_threshold:
                continue  # the ambiguous population itself -- must not leak into calibration
            if has_noncanonical_introns(transcript, chrom):
                continue
            original_score = kozak_score(seq, begin)
            if original_score is None:
                continue
            alt = find_best_start_by_kozak(
                seq, begin, stop, max_shift, min_protein_length
            )
            n_confident += 1
            if alt is not None:
                gaps.append(alt[2] - original_score)

    if n_confident < min_confident:
        logger.warning(
            f"Only {n_confident} confident in-genome start codons found; too few "
            f"to calibrate --kozak-margin, keeping default {default_margin}"
        )
        return default_margin

    gaps.sort()
    n_would_switch = sum(1 for g in gaps if g > default_margin)
    if gaps:
        k = min(int(len(gaps) * (1 - KOZAK_MARGIN_TARGET_FP_RATE)), len(gaps) - 1)
        calibrated = max(default_margin, gaps[k])
    else:
        calibrated = default_margin
    logger.info(
        f"Calibrated --kozak-margin {default_margin} -> {calibrated:.2f} from "
        f"{n_confident} confident in-genome start codons ({n_would_switch}/"
        f"{n_confident} = {100 * n_would_switch / n_confident:.2f}% would have "
        f"been wrongly switched at the default margin)"
    )
    return calibrated


def fix_orf(
    input_gff: str,
    input_fasta: str,
    output_gff: str,
    max_shift: int,
    min_protein_length: int,
    require_canonical: bool,
    report_path: str | None,
    fix_weak_starts: bool = True,
    weak_start_threshold: int = 9,
    kozak_margin: float = 3.0,
    weak_kozak_threshold: float = 5.0,
    calibrate_margin: bool = True,
) -> Counter:
    logger.info(f"Reading GFF {input_gff}")
    header, records = read_gff(input_gff)
    logger.info(f"Read {len(records)} records")

    by_id = {r.id: r for r in records if r.id}
    children_by_parent: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        if record.parent:
            children_by_parent[record.parent].append(record)

    transcripts: dict[str, Transcript] = {}
    for record in records:
        if record.type != MRNA or not record.id:
            continue
        transcripts[record.id] = Transcript(
            mrna=record,
            children=children_by_parent.get(record.id, []),
            seqid=record.seqid,
            strand=record.strand,
        )
    logger.info(f"Found {len(transcripts)} transcripts")

    by_seqid: dict[str, list[Transcript]] = defaultdict(list)
    for transcript in transcripts.values():
        by_seqid[transcript.seqid].append(transcript)

    if fix_weak_starts and calibrate_margin:
        kozak_margin = calibrate_kozak_margin(
            by_seqid,
            input_fasta,
            max_shift,
            min_protein_length,
            weak_start_threshold,
            kozak_margin,
        )

    stats: Counter = Counter()
    shift_hist: Counter = Counter()
    results: dict[str, Result] = {}
    replacements: dict[str, list[Record]] = {}
    seen_seqids: set[str] = set()

    for seqid, chrom in iter_fasta(input_fasta):
        if seqid not in by_seqid:
            continue
        seen_seqids.add(seqid)
        logger.info(f"Processing {seqid} ({len(by_seqid[seqid])} transcripts)")
        for transcript in by_seqid[seqid]:
            result, new_records = repair_transcript(
                transcript,
                chrom,
                max_shift=max_shift,
                min_protein_length=min_protein_length,
                require_canonical=require_canonical,
                fix_weak_starts=fix_weak_starts,
                weak_start_threshold=weak_start_threshold,
                kozak_margin=kozak_margin,
                weak_kozak_threshold=weak_kozak_threshold,
            )
            mrna_id = transcript.mrna.id or ""
            results[mrna_id] = result
            stats[result.status] += 1
            if result.issue:
                stats[f"issue:{result.issue}"] += 1
            if new_records is not None:
                replacements[mrna_id] = new_records
                shift_hist[(result.shift5, result.shift3)] += 1

    missing = set(by_seqid) - seen_seqids
    if missing:
        logger.warning(
            f"{len(missing)} sequence(s) in the GFF were absent from the FASTA and "
            f"were left untouched, e.g. {sorted(missing)[:5]}"
        )
        for seqid in missing:
            for transcript in by_seqid[seqid]:
                results[transcript.mrna.id or ""] = Result("skipped", "no_sequence")
                stats["skipped"] += 1
                stats["issue:no_sequence"] += 1

    # ---------------------------------------------------------------------
    # Annotate and write
    # ---------------------------------------------------------------------
    for mrna_id, result in results.items():
        mrna = transcripts[mrna_id].mrna
        mrna.attributes["orf_status"] = result.status
        if result.status == "repaired":
            mrna.attributes["orf_shift_5"] = str(result.shift5)
            mrna.attributes["orf_shift_3"] = str(result.shift3)
        if result.issue:
            mrna.attributes["orf_issue"] = result.issue
        if result.status == "partial":
            mrna.attributes["partial"] = "true"
            if result.missing_start:
                mrna.attributes["start_range"] = f".,{mrna.start}"
            if result.missing_stop:
                mrna.attributes["end_range"] = f"{mrna.end},."
            parent = by_id.get(mrna.parent or "")
            if parent is not None and parent.type == GENE:
                parent.attributes["partial"] = "true"

    output: list[Record] = []
    replaced_parents = set(replacements)
    for record in records:
        if record.parent in replaced_parents and record.type in EXONIC_TYPES:
            continue  # superseded by regenerated features
        output.append(record)
    for mrna_id, new_records in replacements.items():
        output.extend(new_records)

    # Preserve the input's gene-by-gene layout: sort by original position, and
    # place regenerated children immediately after the mRNA they belong to.
    output.sort(key=lambda r: (r.order, r.start, r.end, r.type))

    logger.info(f"Writing {len(output)} records to {output_gff}")
    with open(output_gff, "w") as fh:
        for line in header:
            fh.write(line + "\n")
        for record in output:
            fh.write(record.to_line() + "\n")

    if report_path:
        logger.info(f"Writing per-transcript report to {report_path}")
        with open(report_path, "w") as fh:
            fh.write("transcript_id\tseqid\tstrand\tstatus\tissue\tshift_5\tshift_3\n")
            for mrna_id, result in sorted(results.items()):
                transcript = transcripts[mrna_id]
                fh.write(
                    f"{mrna_id}\t{transcript.seqid}\t{transcript.strand}\t"
                    f"{result.status}\t{result.issue}\t{result.shift5}\t{result.shift3}\n"
                )

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    total = sum(stats[k] for k in ("complete", "repaired", "partial", "skipped"))
    logger.info("=" * 68)
    logger.info(f"ORF repair summary ({total} transcripts)")
    for status in ("complete", "repaired", "partial", "skipped"):
        count = stats[status]
        logger.info(f"  {status:10s} {count:8d}  {100 * count / max(total, 1):5.1f}%")
    valid = stats["complete"] + stats["repaired"]
    logger.info(
        f"  complete ORFs: {stats['complete']} -> {valid} "
        f"({100 * stats['complete'] / max(total, 1):.1f}% -> "
        f"{100 * valid / max(total, 1):.1f}%)"
    )
    issues = {k[6:]: v for k, v in stats.items() if k.startswith("issue:")}
    if issues:
        # Not all issues mean "unrepaired": weak_start_kozak is a repair
        # that happened (status=repaired) but is still worth surfacing here;
        # the others (noncanonical_intron, ambiguous_bases, ...) are not.
        logger.info("  by issue:")
        for reason, count in sorted(issues.items(), key=lambda kv: -kv[1]):
            logger.info(f"    {reason:24s} {count:8d}")
    if shift_hist:
        shifts = [abs(a) + abs(b) for (a, b), n in shift_hist.items() for _ in range(n)]
        shifts.sort()
        logger.info(
            f"  repair boundary movement (nt): median={shifts[len(shifts) // 2]}, "
            f"p90={shifts[int(len(shifts) * 0.9)]}, max={shifts[-1]}"
        )
    logger.info("=" * 68)
    return stats


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(
        description="Repair CDS boundaries in GeneCAD predictions so they form valid ORFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-gff", "-i", required=True, help="Input GFF3 file")
    parser.add_argument("--input-fasta", "-f", required=True, help="Genome FASTA file")
    parser.add_argument("--output-gff", "-o", required=True, help="Output GFF3 file")
    parser.add_argument(
        "--max-shift",
        type=int,
        default=300,
        help="Maximum movement (nt, in spliced transcript coordinates) allowed for "
        "the TIS and for the TTS. Bounds how far a repair may depart from the "
        "model's prediction, and prevents long CDS being truncated to short ORFs.",
    )
    parser.add_argument(
        "--min-protein-length",
        type=int,
        default=10,
        help="Minimum protein length in residues (excluding the stop codon) for a repair",
    )
    parser.add_argument(
        "--allow-noncanonical-introns",
        action="store_true",
        help="Attempt repair even when the transcript has non-canonical introns. "
        "Off by default: an ORF built on untrustworthy splice calls is not trustworthy.",
    )
    parser.add_argument(
        "--no-fix-weak-starts",
        dest="fix_weak_starts",
        action="store_false",
        help="Disable Kozak-context re-ranking of already-valid but "
        "suspiciously short first exons. On by default.",
    )
    parser.add_argument(
        "--weak-start-threshold",
        type=int,
        default=9,
        help="First coding exon length (nt) below which alternative start "
        "codons are considered, when --no-fix-weak-starts is not set.",
    )
    parser.add_argument(
        "--kozak-margin",
        type=float,
        default=3.0,
        help="Minimum Kozak log2-odds advantage an alternative start codon "
        "must have over the original to trigger a switch. Used as the floor "
        "value calibrate_kozak_margin raises from per genome, unless "
        "--no-calibrate-kozak-margin is set, in which case it is used as-is.",
    )
    parser.add_argument(
        "--no-calibrate-kozak-margin",
        dest="calibrate_margin",
        action="store_false",
        help="Use --kozak-margin as a fixed value for every genome instead of "
        "raising it per genome from that genome's own confident (unambiguous) "
        "start codons. On by default.",
    )
    parser.add_argument(
        "--weak-kozak-threshold",
        type=float,
        default=5.0,
        help="Kozak log2-odds score below which even the best candidate "
        "start is flagged (orf_issue=weak_kozak_support) rather than kept "
        "silently.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional TSV path for per-transcript status output",
    )
    args = parser.parse_args()

    fix_orf(
        input_gff=args.input_gff,
        input_fasta=args.input_fasta,
        output_gff=args.output_gff,
        max_shift=args.max_shift,
        min_protein_length=args.min_protein_length,
        require_canonical=not args.allow_noncanonical_introns,
        report_path=args.report,
        fix_weak_starts=args.fix_weak_starts,
        weak_start_threshold=args.weak_start_threshold,
        kozak_margin=args.kozak_margin,
        weak_kozak_threshold=args.weak_kozak_threshold,
        calibrate_margin=args.calibrate_margin,
    )


if __name__ == "__main__":
    main()
