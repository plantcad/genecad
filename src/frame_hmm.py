"""Frame-aware, sequence-constrained HMM decoding of per-base feature predictions.

The default decoder in :mod:`scripts.detect_intervals` runs Viterbi over the five
modelling features (intergenic / intron / 5' UTR / CDS / 3' UTR) and never looks
at the genome.  It therefore has no way to express the constraints that make a
CDS translatable, and routinely emits coding regions that are not a multiple of
three, do not begin on ATG, do not end on a stop codon, or contain an in-frame
stop.

This module replaces that 5-state chain with an expanded state space in which
those constraints are structural: any path the decoder is *able* to take yields
a valid ORF.  Each expanded state carries

  * the modelling feature it emits (so decoding output stays a 0-4 feature label
    array and the rest of the pipeline is unchanged),
  * its codon position, so reading frame is tracked across introns, and
  * a hard constraint on the base at that position.

Because every constraint is a per-base test, it stays correct when an intron
interrupts a codon: the frame is carried by the state, not by genomic adjacency.

State space
-----------
Stop codons are excluded by tracking the codon prefix.  A stop codon is TAA, TAG
or TGA, so only a leading ``T`` can begin one.  Codon position 0 splits on
"is this base a T", position 1 splits on which stop prefix (``TA`` / ``TG``) is
live, and position 2 then forbids exactly the bases that would complete a stop.
Introns are likewise split by which coding state they must resume into, so a
codon may be interrupted at any of its three positions without losing frame.

The start codon and the terminal stop codon get dedicated states, which is what
lets the decoder require ATG at the beginning of the CDS and a stop codon at its
end.  Introns are not permitted to interrupt the start or the stop codon itself;
this occurs in real genes but is rare, and supporting it would roughly double the
state count.

Introns additionally carry their donor and acceptor dinucleotides as states, so
every emitted intron matches one of the configured splice motifs (GT-AG and
GC-AG by default; see :class:`SpliceMotifGroup`).  That constraint is load
bearing rather than decorative: given only the frame constraints, the cheapest
way for the decoder to escape an in-frame stop codon is to invent a two-base
intron across it, and measured on real predictions it did so readily.

Transition probabilities are derived mechanically from the same 5x5 feature
matrix the existing decoder uses (:func:`src.modeling.token_transition_probs`),
so the learned feature-level statistics are preserved and only the structure is
new.

A canonical donor is not by itself enough, because GT...AG turns up by chance
every few hundred bases, so each intron also carries a run of mandatory body
states that impose a minimum length.  These forced states have a single
predecessor each, so they cost states but no back-pointer memory (see
:class:`PredecessorIndex`); per-position memory is set by the number of
*branching* states, which does not change with the minimum length.

Known limitations
-----------------
* U12-type AT-AC introns are off by default. They are real but rare (~0.04%
  of introns in the TAIR12 reference, against >99.9% GT-AG/GC-AG), and
  supporting them roughly doubles the intron state count -- each configured
  :class:`SpliceMotifGroup` gets its own donor, body and acceptor sub-chain,
  since a donor from one group must never resume through another group's
  acceptor. Pass ``splice_motif_groups=(GT_AG, AT_AC)`` to
  :func:`build_states`/:func:`build_edges`/:func:`frame_aware_decode` to
  enable them. Motifs beyond GT-AG, GC-AG and AT-AC are not offered: every
  other dinucleotide pairing combined accounts for under 0.02% of introns in
  that reference, consistent with annotation noise rather than real splicing.
* Introns are not permitted to interrupt the start or the stop codon.
* Genes must have both a 5' and a 3' UTR, as in the unconstrained decoder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numba import njit
from numpy import typing as npt

logger = logging.getLogger(__name__)

# Feature indices, matching src.schema.SEQUENCE_MODELING_FEATURES
IG, IN, U5, CDS, U3 = range(5)
N_FEATURES = 5

# Base codes produced by `encode_sequence`
BASE_A, BASE_C, BASE_G, BASE_T, BASE_OTHER = range(5)
N_BASE_CODES = 5

# Per-base constraint codes
MASK_ANY = 0
MASK_IS_A = 1
MASK_IS_G = 2
MASK_IS_T = 3
MASK_NOT_T = 4
MASK_NOT_A = 5
MASK_NOT_AG = 6
MASK_IS_AG = 7
MASK_IS_TC = 8
MASK_IS_C = 9
N_MASKS = 10

# A large finite stand-in for -inf, so that adding two of them cannot produce NaN
NEG_INF = -1e30


def build_mask_table() -> np.ndarray:
    """Boolean table of shape (N_MASKS, N_BASE_CODES): is this base allowed?

    An ambiguous base satisfies negative constraints ("not A") but never a
    positive one ("is A"), so an N can sit inside a CDS but can never be taken
    as part of a start or stop codon, and never forces an in-frame stop.
    """
    table = np.zeros((N_MASKS, N_BASE_CODES), dtype=np.bool_)
    table[MASK_ANY, :] = True
    table[MASK_IS_A, BASE_A] = True
    table[MASK_IS_G, BASE_G] = True
    table[MASK_IS_T, BASE_T] = True
    table[MASK_NOT_T, :] = True
    table[MASK_NOT_T, BASE_T] = False
    table[MASK_NOT_A, :] = True
    table[MASK_NOT_A, BASE_A] = False
    table[MASK_NOT_AG, :] = True
    table[MASK_NOT_AG, BASE_A] = False
    table[MASK_NOT_AG, BASE_G] = False
    table[MASK_IS_AG, BASE_A] = True
    table[MASK_IS_AG, BASE_G] = True
    table[MASK_IS_TC, BASE_T] = True
    table[MASK_IS_TC, BASE_C] = True
    table[MASK_IS_C, BASE_C] = True
    return table


@dataclass(frozen=True)
class State:
    name: str
    feature: int
    mask: int


# fmt: off
CORE_STATES: list[State] = [
    State("intergenic",      IG,  MASK_ANY),
    State("utr5",            U5,  MASK_ANY),

    # Start codon: A-T-G
    State("start_a",         CDS, MASK_IS_A),
    State("start_t",         CDS, MASK_IS_T),
    State("start_g",         CDS, MASK_IS_G),

    # CDS body.  Codon position 0 splits on "could this begin a stop codon".
    State("cds_p0_t",        CDS, MASK_IS_T),     # prefix so far: T
    State("cds_p0_x",        CDS, MASK_NOT_T),    # prefix so far: not a stop
    # Codon position 1: which stop prefix, if any, is still live.
    State("cds_p1_ta",       CDS, MASK_IS_A),     # TA_
    State("cds_p1_tg",       CDS, MASK_IS_G),     # TG_
    State("cds_p1_safe_t",   CDS, MASK_NOT_AG),   # T but not TA/TG
    State("cds_p1_safe_x",   CDS, MASK_ANY),      # did not start with T
    # Codon position 2: forbid exactly the bases completing a stop codon.
    State("cds_p2_ta",       CDS, MASK_NOT_AG),   # forbids TAA, TAG
    State("cds_p2_tg",       CDS, MASK_NOT_A),    # forbids TGA
    State("cds_p2_safe",     CDS, MASK_ANY),

    # Terminal stop codon: T-[AG]-x, completing TAA / TAG / TGA
    State("stop_t",          CDS, MASK_IS_T),
    State("stop_a",          CDS, MASK_IS_A),     # TA_
    State("stop_g",          CDS, MASK_IS_G),     # TG_
    State("stop_end_a",      CDS, MASK_IS_AG),    # TAA / TAG
    State("stop_end_g",      CDS, MASK_IS_A),     # TGA

    State("utr3",            U3,  MASK_ANY),
]
# fmt: on

# Introns, one class per coding state they must resume into so that the reading
# frame survives them.  Each class becomes a chain of states (see
# `intron_chain_parts`) that forces a canonical donor and acceptor and a minimum
# length.  This is not cosmetic: without it the decoder dodges an in-frame stop
# codon by inventing a short "intron" across it, and it does so readily.
INTRON_CLASSES: list[tuple[str, tuple[str, ...]]] = [
    ("utr5", ("utr5", "start_a")),
    ("p0t", ("cds_p1_ta", "cds_p1_tg", "cds_p1_safe_t")),
    ("p0x", ("cds_p1_safe_x",)),
    ("p1ta", ("cds_p2_ta",)),
    ("p1tg", ("cds_p2_tg",)),
    ("p1safe", ("cds_p2_safe",)),
    ("p2", ("cds_p0_t", "cds_p0_x", "stop_t")),
    ("utr3", ("utr3",)),
]


@dataclass(frozen=True)
class SpliceMotifGroup:
    """Donor dinucleotides that share a single acceptor dinucleotide.

    Pairing, not the donor alone, is what a spliceosome enforces: the major
    (U2) spliceosome accepts either GT or GC at the 5' splice site but always
    requires AG at the 3', while the minor (U12) spliceosome is strict and
    requires AT...AC.  Grouping by shared acceptor is what lets GT and GC
    share one acceptor sub-chain while AT gets its own -- a donor from one
    group can never resume through another group's acceptor.
    """

    name: str
    donor_masks: tuple[int, int]
    acceptor_masks: tuple[int, int]


GT_AG = SpliceMotifGroup("gtag", (MASK_IS_G, MASK_IS_TC), (MASK_IS_A, MASK_IS_G))
AT_AC = SpliceMotifGroup("atac", (MASK_IS_A, MASK_IS_T), (MASK_IS_A, MASK_IS_C))

# GT-AG and GC-AG account for >99.9% of introns in the TAIR12 reference (chr5:
# 45,058 GT-AG + 484 GC-AG of 45,597 total). AT-AC (U12-type) is real but rare
# there (20 introns, 0.044%) and every other dinucleotide pairing combined
# accounts for under 0.02% -- almost certainly annotation noise rather than
# alternative splicing -- so only AT-AC is offered as an opt-in extra. Adding
# a group roughly doubles the intron state count (each gets its own donor,
# body and acceptor sub-chain, see `intron_chain_parts`), so it is off by
# default; pass ``splice_motif_groups=(GT_AG, AT_AC)`` to enable it.
DEFAULT_SPLICE_MOTIF_GROUPS: tuple[SpliceMotifGroup, ...] = (GT_AG,)

# The donor and acceptor account for four bases; a minimum-length intron also
# needs at least one body base, so this is the shortest intron representable.
STRUCTURAL_MIN_INTRON_LENGTH = 5

# Twenty is deliberately below the ~40-70 nt often quoted as a minimum for
# spliceosomal lariat formation. Checked across 64 Phytozome angiosperm
# reference annotations (~14.5M introns total, not just Arabidopsis): only
# 0.13% of introns are shorter than 20 nt, but raising the floor to 40 would
# exclude 0.43% more -- and 37 of the 64 species have at least one genuine
# sub-20nt intron, with a handful (e.g. Lens ervoides, Brassica oleracea,
# Arachis hypogaea) north of 1%. Meanwhile an unconstrained frame-aware
# decoder emits ~0.4% of introns below 20 nt on Arabidopsis -- the excess
# being short introns invented to step over an in-frame stop codon. Twenty
# keeps essentially every real intron across these genomes while still
# closing that gap. This is only a default: pass a larger value (e.g. via
# --min-intron-length in scripts/detect_intervals.py) to enforce a stricter,
# more textbook-canonical minimum for organisms where that's preferred.
DEFAULT_MIN_INTRON_LENGTH = 20


def intron_chain_parts(
    min_intron_length: int,
    splice_motif_groups: tuple[SpliceMotifGroup, ...] = DEFAULT_SPLICE_MOTIF_GROUPS,
) -> dict[str, list[tuple[str, int]]]:
    """Per-group (suffix, base constraint) lists for the states of one intron.

    Keyed by :attr:`SpliceMotifGroup.name`. Within a group's chain, ``b1..bk``
    are mandatory body positions that impose the minimum length; the looping
    ``body`` state that follows is what actually absorbs intron length. Groups
    never merge -- each gets its own donor, body and acceptor states -- so an
    intron entered through one group's donor can only leave through that same
    group's acceptor.
    """
    if min_intron_length < STRUCTURAL_MIN_INTRON_LENGTH:
        raise ValueError(
            f"min_intron_length must be at least {STRUCTURAL_MIN_INTRON_LENGTH}; "
            f"got {min_intron_length}"
        )
    mandatory = min_intron_length - STRUCTURAL_MIN_INTRON_LENGTH
    body_parts = [(f"b{i}", MASK_ANY) for i in range(1, mandatory + 1)] + [
        ("body", MASK_ANY)
    ]
    return {
        group.name: (
            [("d1", group.donor_masks[0]), ("d2", group.donor_masks[1])]
            + body_parts
            + [("a1", group.acceptor_masks[0]), ("a2", group.acceptor_masks[1])]
        )
        for group in splice_motif_groups
    }


def intron_state(intron_class: str, group_name: str, part: str) -> str:
    return f"intron_{intron_class}_{group_name}_{part}"


def build_states(
    min_intron_length: int = DEFAULT_MIN_INTRON_LENGTH,
    splice_motif_groups: tuple[SpliceMotifGroup, ...] = DEFAULT_SPLICE_MOTIF_GROUPS,
) -> list[State]:
    states = list(CORE_STATES)
    chain_parts = intron_chain_parts(min_intron_length, splice_motif_groups)
    for intron_class, _ in INTRON_CLASSES:
        for group_name, parts in chain_parts.items():
            for part, mask in parts:
                states.append(
                    State(intron_state(intron_class, group_name, part), IN, mask)
                )
    return states


MIN_INTRON_LENGTH = DEFAULT_MIN_INTRON_LENGTH
STATES: list[State] = build_states()
STATE_INDEX = {state.name: i for i, state in enumerate(STATES)}
N_STATES = len(STATES)


def build_edges(
    feature_probs: np.ndarray,
    min_intron_length: int = DEFAULT_MIN_INTRON_LENGTH,
    splice_motif_groups: tuple[SpliceMotifGroup, ...] = DEFAULT_SPLICE_MOTIF_GROUPS,
) -> list[tuple[int, int, float]]:
    """Expand the 5x5 feature transition matrix into the frame-aware state graph.

    Returns a list of ``(source, destination, probability)`` edges, indexed
    against ``build_states(min_intron_length)``.

    Weights are taken directly from the feature matrix.  Where several expanded
    successors share a feature *and* are mutually exclusive under their base
    constraints -- for instance ``cds_p0_t`` and ``cds_p0_x`` -- each receives
    the full feature weight rather than a share of it, because which one applies
    is determined by the sequence, not by chance.  :func:`validate_edges` checks
    that this never lets the feasible weight out of a state exceed 1.
    """
    p = np.asarray(feature_probs, dtype=float)
    if p.shape != (N_FEATURES, N_FEATURES):
        raise ValueError(f"Expected a {N_FEATURES}x{N_FEATURES} matrix; got {p.shape}")

    chain_parts = intron_chain_parts(min_intron_length, splice_motif_groups)
    s = {
        state.name: i
        for i, state in enumerate(build_states(min_intron_length, splice_motif_groups))
    }
    edges: list[tuple[int, int, float]] = []

    def add(source: str, destination: str, weight: float) -> None:
        if weight > 0:
            edges.append((s[source], s[destination], float(weight)))

    cds_cont, cds_intron, cds_end = p[CDS][CDS], p[CDS][IN], p[CDS][U3]
    intron_stay = p[IN][IN]
    intron_exit = 1.0 - intron_stay

    # -- intergenic and 5' UTR ------------------------------------------------
    add("intergenic", "intergenic", p[IG][IG])
    add("intergenic", "utr5", p[IG][U5])

    def enter_intron(source: str, intron_class: str, weight: float) -> None:
        # Each configured motif group's donor mask is mutually exclusive with
        # every other group's (G-first vs A-first), so -- as with cds_p0_t /
        # cds_p0_x above -- each gets the full weight rather than a share of
        # it; only one can ever be feasible for a given base.
        for group in splice_motif_groups:
            add(source, intron_state(intron_class, group.name, "d1"), weight)

    add("utr5", "utr5", p[U5][U5])
    enter_intron("utr5", "utr5", p[U5][IN])
    add("utr5", "start_a", p[U5][CDS])

    # -- start codon ----------------------------------------------------------
    add("start_a", "start_t", 1.0)
    add("start_t", "start_g", 1.0)

    def leave_codon_boundary(source: str) -> None:
        """Successors of a CDS base that completes a codon.

        The next base either opens an intron, continues the coding sequence, or
        begins the terminal stop codon.  Reaching the 3' UTR is only possible
        through that stop codon, so the feature matrix's CDS -> 3' UTR mass is
        what drives entry into it.
        """
        enter_intron(source, "p2", cds_intron)
        add(source, "cds_p0_t", cds_cont)
        add(source, "cds_p0_x", cds_cont)
        add(source, "stop_t", cds_end)

    leave_codon_boundary("start_g")

    # -- CDS body -------------------------------------------------------------
    # Mid-codon there is nowhere to go but onward or into an intron, so the
    # CDS -> 3' UTR mass folds into continuing.
    cds_onward = cds_cont + cds_end

    enter_intron("cds_p0_t", "p0t", cds_intron)
    add("cds_p0_t", "cds_p1_ta", cds_onward)
    add("cds_p0_t", "cds_p1_tg", cds_onward)
    add("cds_p0_t", "cds_p1_safe_t", cds_onward)

    enter_intron("cds_p0_x", "p0x", cds_intron)
    add("cds_p0_x", "cds_p1_safe_x", cds_onward)

    enter_intron("cds_p1_ta", "p1ta", cds_intron)
    add("cds_p1_ta", "cds_p2_ta", cds_onward)

    enter_intron("cds_p1_tg", "p1tg", cds_intron)
    add("cds_p1_tg", "cds_p2_tg", cds_onward)

    for source in ("cds_p1_safe_t", "cds_p1_safe_x"):
        enter_intron(source, "p1safe", cds_intron)
        add(source, "cds_p2_safe", cds_onward)

    for source in ("cds_p2_ta", "cds_p2_tg", "cds_p2_safe"):
        leave_codon_boundary(source)

    # -- terminal stop codon and 3' UTR --------------------------------------
    add("stop_t", "stop_a", 1.0)
    add("stop_t", "stop_g", 1.0)
    add("stop_a", "stop_end_a", 1.0)
    add("stop_g", "stop_end_g", 1.0)
    add("stop_end_a", "utr3", 1.0)
    add("stop_end_g", "utr3", 1.0)

    add("utr3", "utr3", p[U3][U3])
    enter_intron("utr3", "utr3", p[U3][IN])
    add("utr3", "intergenic", p[U3][IG])

    # -- intron chains --------------------------------------------------------
    # Each intron runs donor -> b1..bk -> body* -> acceptor before resuming,
    # separately per configured motif group (e.g. G -> T/C -> ... -> A -> G for
    # GT_AG). The b-states are mandatory, which is what imposes the minimum
    # intron length; the decision to end the intron is taken on leaving the
    # looping body state, and the acceptor states that follow are forced, so
    # the exit weight sits on that one edge.
    resume_weights = {
        # A 5' UTR intron may resume into more UTR or straight into the start
        # codon, and those two are not mutually exclusive under their masks, so
        # they share the exit mass.
        "utr5": {"utr5": 0.5, "start_a": 0.5},
        # At a codon boundary the intron resumes into the next codon or into the
        # terminal stop codon, in the same ratio a coding base would.
        "p2": {
            "cds_p0_t": cds_cont / (cds_cont + cds_end),
            "cds_p0_x": cds_cont / (cds_cont + cds_end),
            "stop_t": cds_end / (cds_cont + cds_end),
        },
    }
    for intron_class, destinations in INTRON_CLASSES:
        for group_name, parts in chain_parts.items():
            chain = [intron_state(intron_class, group_name, part) for part, _ in parts]
            body, acceptor_1, acceptor_2 = chain[-3], chain[-2], chain[-1]
            # Donor and the mandatory body positions are a forced march
            for source, destination in zip(chain, chain[1:-2]):
                add(source, destination, 1.0)
            add(body, body, intron_stay)
            add(body, acceptor_1, intron_exit)
            add(acceptor_1, acceptor_2, 1.0)
            weights = resume_weights.get(intron_class, {})
            for destination in destinations:
                add(acceptor_2, destination, weights.get(destination, 1.0))

    return edges


def validate_edges(
    edges: list[tuple[int, int, float]],
    mask_table: np.ndarray,
    states: list[State] | None = None,
) -> None:
    """Check that no state can emit more than unit probability at any base.

    Expanded successors that share a feature weight are only legitimate because
    their base constraints are mutually exclusive.  This asserts that property
    directly: for every state and every possible next base, the weights of the
    successors actually reachable at that base must not exceed 1.  It also
    checks that the graph is fully connected, since an unreachable state would
    silently disable whatever constraint it carries.
    """
    if states is None:
        states = STATES

    outgoing: dict[int, list[tuple[int, float]]] = {}
    for source, destination, weight in edges:
        outgoing.setdefault(source, []).append((destination, weight))

    for source, successors in outgoing.items():
        for base in range(N_BASE_CODES):
            total = sum(
                weight
                for destination, weight in successors
                if mask_table[states[destination].mask, base]
            )
            if total > 1.0 + 1e-9:
                raise ValueError(
                    f"State {states[source].name!r} has feasible outgoing probability "
                    f"{total:.6f} > 1 for base code {base}; expanded successors that "
                    f"share a feature weight must be mutually exclusive"
                )

    all_states = set(range(len(states)))
    orphans = all_states - {destination for _, destination, _ in edges}
    dead_ends = all_states - set(outgoing)
    if orphans or dead_ends:
        raise ValueError(
            f"Disconnected states: no incoming {[states[i].name for i in sorted(orphans)]}, "
            f"no outgoing {[states[i].name for i in sorted(dead_ends)]}"
        )


class PredecessorIndex(NamedTuple):
    """Edges indexed by destination, plus the back-pointer layout.

    Most states in the expanded graph -- every donor, mandatory-body and
    acceptor position of every intron -- have exactly one predecessor, so the
    path into them is not a choice and needs no back-pointer.  Recording
    back-pointers only for the states that actually branch keeps the table
    narrow no matter how long the intron chains get, which is what makes a
    realistic minimum intron length affordable: the state count grows but the
    per-position memory does not.
    """

    indptr: np.ndarray
    sources: np.ndarray
    log_weights: np.ndarray
    sole_predecessor: np.ndarray  # -1 where the state branches
    branch_column: np.ndarray  # -1 where the state does not branch
    n_branch_columns: int


def build_predecessor_csr(
    edges: list[tuple[int, int, float]], n_states: int = -1
) -> PredecessorIndex:
    if n_states < 0:
        n_states = N_STATES
    by_destination: dict[int, list[tuple[int, float]]] = {
        j: [] for j in range(n_states)
    }
    for source, destination, weight in edges:
        by_destination[destination].append((source, weight))

    indptr = np.zeros(n_states + 1, dtype=np.int64)
    sources: list[int] = []
    log_weights: list[float] = []
    sole_predecessor = np.full(n_states, -1, dtype=np.int16)
    branch_column = np.full(n_states, -1, dtype=np.int16)
    n_branch_columns = 0

    for destination in range(n_states):
        predecessors = sorted(by_destination[destination])
        for source, weight in predecessors:
            sources.append(source)
            log_weights.append(np.log(weight))
        indptr[destination + 1] = len(sources)
        if len(predecessors) == 1:
            sole_predecessor[destination] = predecessors[0][0]
        elif len(predecessors) > 1:
            branch_column[destination] = n_branch_columns
            n_branch_columns += 1

    return PredecessorIndex(
        indptr,
        np.asarray(sources, dtype=np.int64),
        np.asarray(log_weights, dtype=np.float64),
        sole_predecessor,
        branch_column,
        n_branch_columns,
    )


_BASE_LOOKUP = np.full(256, BASE_OTHER, dtype=np.uint8)
for _char, _code in (("A", BASE_A), ("C", BASE_C), ("G", BASE_G), ("T", BASE_T)):
    _BASE_LOOKUP[ord(_char)] = _code
    _BASE_LOOKUP[ord(_char.lower())] = _code

_COMPLEMENT_CODE = np.array(
    [BASE_T, BASE_G, BASE_C, BASE_A, BASE_OTHER], dtype=np.uint8
)


def encode_sequence(sequence: str) -> np.ndarray:
    """Encode a nucleotide string as base codes, case-insensitively."""
    raw = np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)
    return _BASE_LOOKUP[raw]


def reverse_complement_codes(codes: np.ndarray) -> np.ndarray:
    """Reverse complement an encoded sequence.

    Predictions for the minus strand are decoded on the reversed logit array, so
    the base at reversed index ``i`` is the complement of the base at genomic
    index ``len - 1 - i``.  The result is materialised contiguously rather than
    left as a reverse-strided view, which the decoding kernel reads far faster.
    """
    return np.ascontiguousarray(_COMPLEMENT_CODE[codes][::-1])


@njit(cache=True)
def _masked_viterbi(
    log_emission: np.ndarray,
    base_codes: np.ndarray,
    state_feature: np.ndarray,
    state_mask: np.ndarray,
    mask_table: np.ndarray,
    pred_indptr: np.ndarray,
    pred_sources: np.ndarray,
    pred_log_weights: np.ndarray,
    sole_predecessor: np.ndarray,
    branch_column: np.ndarray,
    n_branch_columns: int,
    log_initial: np.ndarray,
) -> np.ndarray:
    """Viterbi over a sparse, per-position-masked state graph.

    Only the previous score row is retained, and back-pointers are stored only
    for states that have more than one predecessor, so per-position memory is
    two bytes per branching state regardless of how many forced chain states the
    graph contains.
    """
    n_positions = log_emission.shape[0]
    n_states = state_feature.shape[0]

    scores = np.full((2, n_states), NEG_INF)
    backpointer = np.full((n_positions, n_branch_columns), -1, dtype=np.int16)

    base = base_codes[0]
    for j in range(n_states):
        if mask_table[state_mask[j], base]:
            scores[0, j] = log_initial[j] + log_emission[0, state_feature[j]]

    for t in range(1, n_positions):
        previous = (t - 1) % 2
        current = t % 2
        base = base_codes[t]
        for j in range(n_states):
            if not mask_table[state_mask[j], base]:
                scores[current, j] = NEG_INF
                continue
            best = NEG_INF
            best_source = -1
            for k in range(pred_indptr[j], pred_indptr[j + 1]):
                i = pred_sources[k]
                value = scores[previous, i] + pred_log_weights[k]
                if value > best:
                    best = value
                    best_source = i
            if best_source < 0:
                scores[current, j] = NEG_INF
            else:
                scores[current, j] = best + log_emission[t, state_feature[j]]
                column = branch_column[j]
                if column >= 0:
                    backpointer[t, column] = best_source

    last = (n_positions - 1) % 2
    best_state = 0
    best_score = scores[last, 0]
    for j in range(1, n_states):
        if scores[last, j] > best_score:
            best_score = scores[last, j]
            best_state = j

    path = np.empty(n_positions, dtype=np.int16)
    path[n_positions - 1] = best_state
    for t in range(n_positions - 1, 0, -1):
        column = branch_column[best_state]
        if column < 0:
            source = sole_predecessor[best_state]
        else:
            source = backpointer[t, column]
        if source < 0:
            # Defensive: a severed path can only arise from a fully masked
            # column, which cannot happen while the intergenic state is
            # unconstrained.  Fall back to intergenic rather than crash.
            source = 0
        best_state = source
        path[t - 1] = best_state

    return path


def frame_aware_decode(
    feature_probs: npt.ArrayLike,
    base_codes: np.ndarray,
    feature_transition: npt.ArrayLike,
    epsilon: float | None = None,
    return_states: bool = False,
    min_intron_length: int = DEFAULT_MIN_INTRON_LENGTH,
    splice_motif_groups: tuple[SpliceMotifGroup, ...] = DEFAULT_SPLICE_MOTIF_GROUPS,
) -> np.ndarray:
    """Decode per-base feature probabilities under frame and codon constraints.

    Parameters
    ----------
    feature_probs
        Array of shape ``(T, 5)`` of per-base feature probabilities, ordered as
        ``[intergenic, intron, five_prime_utr, cds, three_prime_utr]``.
    base_codes
        Array of shape ``(T,)`` from :func:`encode_sequence`, in the same
        orientation as ``feature_probs``.
    feature_transition
        The ``5x5`` feature transition matrix, already regularized if desired.
        Regularization must be applied here rather than to the expanded matrix:
        smoothing the expanded transitions would destroy the structural zeros
        that enforce the reading frame.
    epsilon
        Floor added to probabilities before taking logs.
    return_states
        Return expanded state indices instead of feature labels.  Intended for
        tests and diagnostics.  Indices refer to
        ``build_states(min_intron_length)``.
    min_intron_length
        Shortest intron the decoder may emit.  Introns below this length are
        overwhelmingly artefacts of stepping over an in-frame stop codon rather
        than real splicing.
    splice_motif_groups
        Which donor/acceptor dinucleotide pairings an intron may use. Defaults
        to GT-AG/GC-AG only; pass ``(GT_AG, AT_AC)`` to also allow U12-type
        AT-AC introns, at the cost of roughly doubling the intron state count.

    Returns
    -------
    np.ndarray
        Feature labels of shape ``(T,)`` with values in ``0..4``, or expanded
        state indices when ``return_states`` is set.
    """
    feature_probs = np.asarray(feature_probs)
    if feature_probs.ndim != 2 or feature_probs.shape[1] != N_FEATURES:
        raise ValueError(
            f"Expected feature probs of shape (T, 5); got {feature_probs.shape}"
        )
    base_codes = np.ascontiguousarray(base_codes, dtype=np.uint8)
    if base_codes.shape[0] != feature_probs.shape[0]:
        raise ValueError(
            f"Sequence length {base_codes.shape[0]} does not match the number of "
            f"predicted positions {feature_probs.shape[0]}"
        )
    if epsilon is None:
        epsilon = float(np.finfo(np.float32).tiny)

    states = build_states(min_intron_length, splice_motif_groups)
    n_states = len(states)
    mask_table = build_mask_table()
    edges = build_edges(
        np.asarray(feature_transition, dtype=float),
        min_intron_length,
        splice_motif_groups,
    )
    validate_edges(edges, mask_table, states)
    index = build_predecessor_csr(edges, n_states)

    state_feature = np.array([state.feature for state in states], dtype=np.int64)
    state_mask = np.array([state.mask for state in states], dtype=np.int64)

    # The initial distribution is deliberately weak: chromosomes begin in
    # intergenic sequence, and over millions of positions the choice is
    # immaterial to the decoded path.
    initial = np.full(n_states, 1.0 / (10 * n_states))
    initial[[state.name for state in states].index("intergenic")] = 1.0
    log_initial = np.log(initial / initial.sum())

    # Build the log-emission array with a single allocation.  At one row per
    # base this is gigabytes for a large chromosome, so the naive
    # `np.log(probs + eps)` -- which materialises two more full-size temporaries
    # -- is worth avoiding.
    log_emission = np.array(feature_probs, dtype=np.float32, copy=True)
    np.add(log_emission, np.float32(epsilon), out=log_emission)
    np.log(log_emission, out=log_emission)

    logger.info(
        f"Frame-aware decoding of {log_emission.shape[0]} positions over "
        f"{n_states} states ({len(edges)} transitions, "
        f"{index.n_branch_columns} with back-pointers, "
        f"min_intron_length={min_intron_length}, "
        f"splice_motif_groups={[g.name for g in splice_motif_groups]})"
    )
    path = _masked_viterbi(
        log_emission,
        base_codes,
        state_feature,
        state_mask,
        mask_table,
        index.indptr,
        index.sources,
        index.log_weights,
        index.sole_predecessor,
        index.branch_column,
        index.n_branch_columns,
        log_initial,
    )
    if return_states:
        return path.astype(np.int64)
    return state_feature[path.astype(np.int64)]
