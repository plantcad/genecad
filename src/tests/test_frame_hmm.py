"""Tests for frame-aware, sequence-constrained HMM decoding."""

import numpy as np
import pytest

from src import frame_hmm as fh
from src.modeling import token_transition_probs

IG, IN, U5, CDS, U3 = range(5)
STOPS = {"TAA", "TAG", "TGA"}


def plant_matrix() -> np.ndarray:
    return token_transition_probs(
        remove_incomplete_features=True, domain="plant"
    ).values


def random_sequence(rng: np.random.Generator, n: int) -> str:
    return "".join(rng.choice(list("ACGT"), n))


def emissions_from(labels: np.ndarray, confidence: float) -> np.ndarray:
    """Per-base feature probabilities that put `confidence` on the given labels."""
    probs = np.full((len(labels), 5), (1.0 - confidence) / 4.0)
    probs[np.arange(len(labels)), labels] = confidence
    return probs


def coding_sequences(sequence: str, labels: np.ndarray) -> list[str]:
    """Concatenate CDS-labelled bases per gene, splitting genes on intergenic runs."""
    genes: list[str] = []
    current: list[str] = []
    in_gene = False
    for base, label in zip(sequence, labels):
        if label == IG:
            if in_gene:
                genes.append("".join(current))
                current, in_gene = [], False
            continue
        in_gene = True
        if label == CDS:
            current.append(base)
    if in_gene:
        genes.append("".join(current))
    return [gene for gene in genes if gene]


def is_valid_orf(seq: str) -> bool:
    return (
        len(seq) >= 6
        and len(seq) % 3 == 0
        and seq.startswith("ATG")
        and seq[-3:] in STOPS
        and not any(seq[i : i + 3] in STOPS for i in range(3, len(seq) - 3, 3))
    )


# -------------------------------------------------------------------------------------------------
# Synthetic locus
# -------------------------------------------------------------------------------------------------


def build_locus(
    rng: np.random.Generator, coding: str, split: int, intron_len: int = 200
):
    """Lay a coding sequence out over two exons and return (sequence, labels)."""
    exon1, exon2 = coding[:split], coding[split:]
    intron = "GT" + random_sequence(rng, intron_len - 4) + "AG"
    sequence = (
        random_sequence(rng, 2000)
        + random_sequence(rng, 200)
        + exon1
        + intron
        + exon2
        + random_sequence(rng, 200)
        + random_sequence(rng, 2000)
    )
    labels = np.array(
        [IG] * 2000
        + [U5] * 200
        + [CDS] * len(exon1)
        + [IN] * intron_len
        + [CDS] * len(exon2)
        + [U3] * 200
        + [IG] * 2000
    )
    assert len(sequence) == len(labels)
    return sequence, labels


VALID_CODING = "ATG" + "GCT" * 28 + "TAA"  # 90 nt, 29 residues


# -------------------------------------------------------------------------------------------------
# State graph
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("domain", ["plant", "animal"])
@pytest.mark.parametrize("remove_incomplete", [True, False])
def test_expanded_graph_is_a_valid_probability_model(domain, remove_incomplete):
    """Expanded successors that share a feature weight must be mutually exclusive,
    so that no state can emit more than unit probability at any base."""
    matrix = token_transition_probs(
        remove_incomplete_features=remove_incomplete, domain=domain
    ).values
    edges = fh.build_edges(matrix)
    fh.validate_edges(edges, fh.build_mask_table())  # raises if violated

    mask_table = fh.build_mask_table()
    outgoing: dict[int, list[tuple[int, float]]] = {}
    for source, destination, weight in edges:
        outgoing.setdefault(source, []).append((destination, weight))
    for source, successors in outgoing.items():
        for base in range(fh.N_BASE_CODES):
            total = sum(
                weight
                for destination, weight in successors
                if mask_table[fh.STATES[destination].mask, base]
            )
            assert total <= 1.0 + 1e-9, fh.STATES[source].name


def test_every_state_is_connected():
    edges = fh.build_edges(plant_matrix())
    sources = {source for source, _, _ in edges}
    destinations = {destination for _, destination, _ in edges}
    assert sources == set(range(fh.N_STATES))
    assert destinations == set(range(fh.N_STATES))


def test_mask_table_treats_ambiguous_bases_conservatively():
    table = fh.build_mask_table()
    # An N can never be read as a specific base ...
    assert not table[fh.MASK_IS_A, fh.BASE_OTHER]
    assert not table[fh.MASK_IS_T, fh.BASE_OTHER]
    assert not table[fh.MASK_IS_G, fh.BASE_OTHER]
    assert not table[fh.MASK_IS_AG, fh.BASE_OTHER]
    # ... but it also never completes a stop codon
    assert table[fh.MASK_NOT_A, fh.BASE_OTHER]
    assert table[fh.MASK_NOT_AG, fh.BASE_OTHER]


def test_sequence_encoding_is_case_insensitive_and_reversible():
    sequence = "ACGTNacgtn"
    codes = fh.encode_sequence(sequence)
    assert list(codes[:5]) == list(codes[5:])
    assert list(codes[:5]) == [
        fh.BASE_A,
        fh.BASE_C,
        fh.BASE_G,
        fh.BASE_T,
        fh.BASE_OTHER,
    ]

    rng = np.random.default_rng(7)
    sequence = random_sequence(rng, 500)
    complement = str.maketrans("ACGT", "TGCA")
    expected = fh.encode_sequence(sequence.translate(complement)[::-1])
    assert np.array_equal(
        fh.reverse_complement_codes(fh.encode_sequence(sequence)), expected
    )


# -------------------------------------------------------------------------------------------------
# Decoding
# -------------------------------------------------------------------------------------------------


def test_confident_predictions_are_recovered_exactly():
    rng = np.random.default_rng(0)
    sequence, labels = build_locus(rng, VALID_CODING, split=50)
    decoded = fh.frame_aware_decode(
        emissions_from(labels, 0.9), fh.encode_sequence(sequence), plant_matrix()
    )
    assert np.array_equal(decoded, labels)


def test_frame_is_carried_across_an_intron():
    """The exon boundary falls mid-codon, so recovering the ORF requires the
    reading frame to survive the intron."""
    rng = np.random.default_rng(1)
    for split in (49, 50, 51):  # each of the three codon positions
        sequence, labels = build_locus(rng, VALID_CODING, split=split)
        decoded = fh.frame_aware_decode(
            emissions_from(labels, 0.9), fh.encode_sequence(sequence), plant_matrix()
        )
        assert coding_sequences(sequence, decoded) == [VALID_CODING], f"{split=}"


def test_sloppy_cds_boundaries_snap_back_to_the_true_orf():
    """The characteristic failure of the unconstrained decoder is a CDS that
    bleeds a base or two into the flanking UTRs."""
    rng = np.random.default_rng(2)
    sequence, labels = build_locus(rng, VALID_CODING, split=50)
    sloppy = labels.copy()
    sloppy[2198:2200] = CDS  # eats into the 5' UTR
    sloppy[2290:2292] = CDS  # eats into the 3' UTR

    decoded = fh.frame_aware_decode(
        emissions_from(sloppy, 0.9), fh.encode_sequence(sequence), plant_matrix()
    )
    assert coding_sequences(sequence, decoded) == [VALID_CODING]


def test_unconstrained_decoder_does_not_survive_the_same_input():
    """Guards the premise of this module: the 5-state decoder really does emit
    an untranslatable CDS for the input above."""
    from src.sequence import viterbi_decode

    rng = np.random.default_rng(2)
    sequence, labels = build_locus(rng, VALID_CODING, split=50)
    sloppy = labels.copy()
    sloppy[2198:2200] = CDS
    sloppy[2290:2292] = CDS

    decoded = viterbi_decode(
        emission_probs=emissions_from(sloppy, 0.9), transition_matrix=plant_matrix()
    )
    assert not all(is_valid_orf(seq) for seq in coding_sequences(sequence, decoded))


def test_in_frame_stop_codons_are_never_emitted():
    """A CDS carrying an in-frame stop must not be decodable as coding, even
    when the emissions insist on it."""
    rng = np.random.default_rng(3)
    # TAA sits in frame, 12 nt into the coding sequence
    poisoned = "ATG" + "GCT" * 3 + "TAA" + "GCT" * 24 + "TGA"
    sequence, labels = build_locus(rng, poisoned, split=50)

    decoded = fh.frame_aware_decode(
        emissions_from(labels, 0.9), fh.encode_sequence(sequence), plant_matrix()
    )
    for coding in coding_sequences(sequence, decoded):
        assert is_valid_orf(coding)
        assert coding != poisoned


@pytest.mark.parametrize("seed", range(6))
def test_every_decoded_cds_is_a_valid_orf(seed):
    """The structural guarantee: whatever the emissions say, any coding sequence
    the decoder emits translates cleanly."""
    rng = np.random.default_rng(100 + seed)
    coding = "ATG" + "".join(
        rng.choice(["GCT", "AAA", "TTT", "CCG", "GGA", "TCA"]) for _ in range(40)
    )
    coding += "TAG"
    sequence, labels = build_locus(rng, coding, split=int(rng.integers(20, 100)))

    # Degrade the emissions: low confidence plus noise, so the decoder is not
    # simply reading off a clean answer.
    probs = emissions_from(labels, 0.55) + rng.random((len(labels), 5)) * 0.2
    probs /= probs.sum(axis=1, keepdims=True)

    decoded = fh.frame_aware_decode(probs, fh.encode_sequence(sequence), plant_matrix())
    genes = coding_sequences(sequence, decoded)
    assert genes, "expected at least one coding region"
    for gene in genes:
        assert is_valid_orf(gene), f"invalid ORF decoded: {gene[:30]}...{gene[-9:]}"


def test_minus_strand_decoding_recovers_a_reverse_strand_gene():
    """Exercise the transformation detect_intervals applies to the minus strand:
    the logits are decoded reversed, so the sequence must be reverse
    complemented to stay in register.  A gene on the minus strand must come back
    out at the right coordinates."""
    rng = np.random.default_rng(4)
    locus, locus_labels = build_locus(rng, VALID_CODING, split=50)

    # The same gene, now lying on the minus strand of the chromosome
    complement = str.maketrans("ACGT", "TGCA")
    chromosome = locus.translate(complement)[::-1]
    chromosome_labels = locus_labels[::-1].copy()

    codes = fh.encode_sequence(chromosome)
    probs = emissions_from(chromosome_labels, 0.9)

    # `np.flip` mirrors exactly what detect_intervals does for the minus strand
    decoded = np.flip(
        fh.frame_aware_decode(
            np.flip(probs, axis=0),
            fh.reverse_complement_codes(codes),
            plant_matrix(),
        ),
        axis=0,
    )

    assert np.array_equal(decoded, chromosome_labels)
    # and the coding sequence read off the minus strand is the true ORF
    coding = "".join(
        base for base, label in zip(chromosome, decoded) if label == CDS
    ).translate(complement)[::-1]
    assert coding == VALID_CODING


def decoded_introns(sequence: str, labels: np.ndarray) -> list[str]:
    """Every maximal run of intron-labelled bases, as a sequence."""
    runs: list[str] = []
    current: list[str] = []
    for base, label in zip(sequence, labels):
        if label == IN:
            current.append(base)
        elif current:
            runs.append("".join(current))
            current = []
    if current:
        runs.append("".join(current))
    return runs


@pytest.mark.parametrize("seed", range(6))
def test_decoded_introns_are_always_canonical(seed):
    """Without a splice-site constraint the decoder escapes an in-frame stop by
    inventing a two-base intron.  Every emitted intron must carry a real donor
    and acceptor."""
    rng = np.random.default_rng(200 + seed)
    coding = (
        "ATG"
        + "".join(rng.choice(["GCT", "AAA", "TTT", "CCG"]) for _ in range(40))
        + "TAG"
    )
    sequence, labels = build_locus(rng, coding, split=int(rng.integers(20, 100)))
    probs = emissions_from(labels, 0.55) + rng.random((len(labels), 5)) * 0.2
    probs /= probs.sum(axis=1, keepdims=True)

    decoded = fh.frame_aware_decode(probs, fh.encode_sequence(sequence), plant_matrix())
    introns = decoded_introns(sequence, decoded)
    assert introns, "expected at least one intron"
    for intron in introns:
        assert len(intron) >= fh.MIN_INTRON_LENGTH
        assert (intron[:2], intron[-2:]) in (("GT", "AG"), ("GC", "AG")), intron[:8]


def test_minimum_intron_length_is_enforced():
    """A canonical donor alone is not enough: GT...AG occurs by chance every few
    hundred bases, so a short intron must also be structurally unreachable."""
    rng = np.random.default_rng(9)
    sequence, labels = build_locus(rng, VALID_CODING, split=50, intron_len=12)
    probs = emissions_from(labels, 0.9)
    codes = fh.encode_sequence(sequence)

    # Permitted when the minimum is relaxed to the structural floor
    relaxed = fh.frame_aware_decode(
        probs, codes, plant_matrix(), min_intron_length=fh.STRUCTURAL_MIN_INTRON_LENGTH
    )
    assert [len(i) for i in decoded_introns(sequence, relaxed)] == [12]

    # Excluded at the default minimum, even though the emissions ask for it
    strict = fh.frame_aware_decode(probs, codes, plant_matrix())
    for intron in decoded_introns(sequence, strict):
        assert len(intron) >= fh.DEFAULT_MIN_INTRON_LENGTH
    for coding in coding_sequences(sequence, strict):
        assert is_valid_orf(coding)


@pytest.mark.parametrize("min_intron_length", [5, 20, 40])
def test_backpointer_memory_does_not_grow_with_minimum_intron_length(min_intron_length):
    """The mandatory body states have one predecessor each, so raising the
    minimum intron length must cost states but not per-position memory."""
    states = fh.build_states(min_intron_length)
    edges = fh.build_edges(plant_matrix(), min_intron_length)
    fh.validate_edges(edges, fh.build_mask_table(), states)
    index = fh.build_predecessor_csr(edges, len(states))

    assert len(states) == 20 + 8 * min_intron_length
    assert index.n_branch_columns == 24
    # every non-branching state must know its unique predecessor
    for j in range(len(states)):
        assert (index.branch_column[j] >= 0) != (index.sole_predecessor[j] >= 0)


def test_a_noncanonical_intron_is_not_decoded_as_an_intron():
    rng = np.random.default_rng(8)
    sequence, labels = build_locus(rng, VALID_CODING, split=50)
    # Break the donor dinucleotide of the only intron
    intron_start = 2000 + 200 + 50
    sequence = sequence[:intron_start] + "AA" + sequence[intron_start + 2 :]

    decoded = fh.frame_aware_decode(
        emissions_from(labels, 0.9), fh.encode_sequence(sequence), plant_matrix()
    )
    assert decoded[intron_start] != IN
    for intron in decoded_introns(sequence, decoded):
        assert (intron[:2], intron[-2:]) in (("GT", "AG"), ("GC", "AG"))


def test_mismatched_sequence_length_is_rejected():
    rng = np.random.default_rng(5)
    sequence, labels = build_locus(rng, VALID_CODING, split=50)
    with pytest.raises(ValueError, match="does not match"):
        fh.frame_aware_decode(
            emissions_from(labels, 0.9),
            fh.encode_sequence(sequence[:-10]),
            plant_matrix(),
        )


def test_returning_expanded_states_maps_back_to_features():
    rng = np.random.default_rng(6)
    sequence, labels = build_locus(rng, VALID_CODING, split=50)
    codes = fh.encode_sequence(sequence)
    features = fh.frame_aware_decode(emissions_from(labels, 0.9), codes, plant_matrix())
    states = fh.frame_aware_decode(
        emissions_from(labels, 0.9), codes, plant_matrix(), return_states=True
    )
    state_feature = np.array([state.feature for state in fh.STATES])
    assert np.array_equal(state_feature[states], features)

    # The start codon really is decoded through the dedicated start states
    coding_start = int(np.argmax(features == CDS))
    assert [fh.STATES[s].name for s in states[coding_start : coding_start + 3]] == [
        "start_a",
        "start_t",
        "start_g",
    ]
    coding_end = len(features) - 1 - int(np.argmax((features == CDS)[::-1]))
    assert fh.STATES[states[coding_end]].name in ("stop_end_a", "stop_end_g")
