"""Interval evaluation functions from https://github.com/maize-genetics/reelAnnote-dev/pull/2"""

import numpy as np
from Bio import SeqFeature
from numba import njit
import src.portion as P
from dataclasses import dataclass
import itertools


@dataclass
class Intervals:
    """
    Representation of a transcript model as a set of intervals.
    """

    cds: P.Interval  # all CDS features
    utr: P.Interval  # all UTR features
    intron: P.Interval  # all introns, inferred from space between CDS and UTR intervals
    intron_cds: P.Interval  # all introns that are fully contained within CDS
    exon: P.Interval  # all exons (union of CDS and UTR)


@dataclass
class ConfusionCounts:
    """
    Counts of occurrences in a confusion matrix. False negatives omitted..
    """

    tp: int = 0  # true positives
    fp: int = 0  # false positives
    fn: int = 0  # false negatives

    def __post_init__(self):
        """Validate that all counts are non-negative."""
        assert self.tp >= 0, f"True positives must be non-negative, got {self.tp}"
        assert self.fp >= 0, f"False positives must be non-negative, got {self.fp}"
        assert self.fn >= 0, f"False negatives must be non-negative, got {self.fn}"

    def __add__(self, other):
        return ConfusionCounts(
            self.tp + other.tp, self.fp + other.fp, self.fn + other.fn
        )

    def precision(self):
        if (self.tp + self.fp) > 0:
            return self.tp / (self.tp + self.fp)
        else:
            return 0.0

    def recall(self):
        if (self.tp + self.fn) > 0:
            return self.tp / (self.tp + self.fn)
        else:
            return 0.0

    def f1(self):
        if (self.tp + self.fp + self.fn) > 0:
            return 2 * self.tp / (2 * self.tp + self.fp + self.fn)
        else:
            return 0.0

    def to_score_string(self, as_percentage: bool = True):
        precision = self.precision()
        recall = self.recall()
        f1 = self.f1()

        if as_percentage:
            precision *= 100
            recall *= 100
            f1 *= 100

        return str(precision) + "\t" + str(recall) + "\t" + str(f1) + "\n"


@dataclass
class Stats:
    """
    This class keeps track of gffcompare statistics at several levels.
    All statistics are represented as ConfusionCounts
    """

    transcript_cds: ConfusionCounts
    transcript_intron: ConfusionCounts
    transcript: ConfusionCounts
    exon_cds: ConfusionCounts
    exon: ConfusionCounts
    intron_cds: ConfusionCounts
    intron: ConfusionCounts
    exon_cds_longest: ConfusionCounts
    exon_longest: ConfusionCounts
    intron_cds_longest: ConfusionCounts
    intron_longest: ConfusionCounts
    base_cds: ConfusionCounts
    base_utr: ConfusionCounts
    base_exon: ConfusionCounts

    def __add__(self, other):
        return Stats(
            self.transcript_cds + other.transcript_cds,
            self.transcript_intron + other.transcript_intron,
            self.transcript + other.transcript,
            self.exon_cds + other.exon_cds,
            self.exon + other.exon,
            self.intron_cds + other.intron_cds,
            self.intron + other.intron,
            self.exon_cds_longest + other.exon_cds_longest,
            self.exon_longest + other.exon_longest,
            self.intron_cds_longest + other.intron_cds_longest,
            self.intron_longest + other.intron_longest,
            self.base_cds + other.base_cds,
            self.base_utr + other.base_utr,
            self.base_exon + other.base_exon,
        )


def blank_stats():
    """
    Initialize a blank Stats object
    """
    return Stats(
        ConfusionCounts(),
        ConfusionCounts(),
        ConfusionCounts(),
        ConfusionCounts(),
        ConfusionCounts(),
        ConfusionCounts(),
        ConfusionCounts(),
        ConfusionCounts(),
        ConfusionCounts(),
        ConfusionCounts(),
        ConfusionCounts(),
        ConfusionCounts(),
        ConfusionCounts(),
        ConfusionCounts(),
    )


@njit
def _find_matches_within_tolerance(
    pred_starts: np.ndarray,
    pred_stops: np.ndarray,
    true_starts: np.ndarray,
    true_stops: np.ndarray,
    tolerance: int,
) -> tuple[int, int]:
    """
    Find number of predicted and true intervals that have at least one match.

    Returns:
        tuple: (matched_pred_count, matched_true_count)
    """
    # Sort both start and stop arrays with their original indices
    start_order = np.argsort(pred_starts)
    stop_order = np.argsort(pred_stops)
    sorted_starts = pred_starts[start_order]
    sorted_stops = pred_stops[stop_order]

    # Track which intervals have been matched
    matched_pred = np.zeros(len(pred_starts), dtype=np.bool_)
    matched_true = np.zeros(len(true_starts), dtype=np.bool_)

    for i, (true_start, true_stop) in enumerate(zip(true_starts, true_stops)):
        # Find matching indices for both start and stop positions
        start_matches = start_order[
            np.searchsorted(
                sorted_starts, true_start - tolerance, side="left"
            ) : np.searchsorted(sorted_starts, true_start + tolerance, side="right")
        ]
        stop_matches = stop_order[
            np.searchsorted(
                sorted_stops, true_stop - tolerance, side="left"
            ) : np.searchsorted(sorted_stops, true_stop + tolerance, side="right")
        ]

        # Mark any predicted intervals that match both start and stop
        matching_indices = np.intersect1d(start_matches, stop_matches)
        if matching_indices.size > 0:
            matched_pred[matching_indices] = True
            matched_true[i] = True

    return np.sum(matched_pred), np.sum(matched_true)


def overlaps(x: tuple[int, int], y: tuple[int, int]) -> bool:
    """
    Check if two ranges overlap
    Assumes ranges are closedopen format [)

    Args:
        x: tuple describing first range
        y: tuple describing second range

    Returns:
        boolean: true if ranges overlap
    """
    return x[0] < y[1] and x[1] > y[0]


def to_interval(feat: SeqFeature) -> tuple[int, int]:
    """
    convert a SeqFeature object to a tuple describing its range

    Args:
        feat: SeqFeature

    Returns:
        tuple (start, end)
    """
    return int(feat.location.start), int(feat.location.end)


def get_longest_transcript_index(gene: SeqFeature) -> int:
    """
    get the index of the longest transcript in a gene

    Args:
        gene: SeqFeature of a gene

    Returns:
        int: index of longest transcript
    """
    return np.argmax(
        [
            int(
                np.sum(
                    [
                        feature.location.end - feature.location.start
                        for feature in transcript.sub_features
                        if (
                            feature.type == "CDS"
                            or feature.type == "five_prime_UTR"
                            or feature.type == "three_prime_UTR"
                        )
                    ]
                )
            )
            for transcript in gene.sub_features
        ]
    )


def get_next_overlap_set(
    pred_features: list[SeqFeature],
    true_features: list[SeqFeature],
    start_pidx: int,
    start_tidx: int,
    strand: int = 1,
):
    """
    Get the next set of overlapping genes from a given start point. An overlapping set contains all protein-coding
    genes that overlap the first gene on the same strand, and all genes that overlap the genes that overlapped the
    first gene, and so on recursively.

    Args:
        pred_features: list of predicted SeqFeatures for a contig. Only "gene" types will be considered, others
        will be ignored (e.g. "ncRNA", "miRNA", "TE")
        true_features: list of true SeqFeatures for a contig. Only "gene" types will be considered, others
        will be ignored (e.g. "ncRNA", "miRNA", "TE")
        start_pidx: index of the leftmost predicted feature to start with
        start_tidx: index of the leftmost true feature to start with
        strand: strand to check. Default is 1 (positive strand)

    Returns:
        pred_overlaps: list of integers. For an index idx, pred_overlaps[idx] overlaps true_overlaps[idx]
        true_overlaps: list of integers. For an index idx, pred_overlaps[idx] overlaps true_overlaps[idx]
        next_pidx: starting predicted feature index for the next set of overlapping genes
        next_tidx: starting true feature index for the next set of overlapping genes
    """

    pidx = start_pidx
    tidx = start_tidx

    # get the next valid predicted gene
    while pidx < len(pred_features):
        if (
            pred_features[pidx].type != "gene"
            or pred_features[pidx].location.strand != strand
        ):
            pidx += 1
        else:
            break

    # get the next valid true gene
    while tidx < len(true_features):
        if (
            true_features[tidx].type != "gene"
            or true_features[tidx].location.strand != strand
        ):
            tidx += 1
        else:
            break

    # check if end of a contig was reached
    if pidx >= len(pred_features) and tidx < len(true_features):
        return [], [tidx], pidx, tidx + 1
    elif tidx >= len(true_features) and pidx < len(pred_features):
        return [pidx], [], pidx + 1, tidx
    elif pidx == len(pred_features) and tidx >= len(true_features):
        return [], [], pidx, tidx

    # check if the predicted and true genes overlap
    # if they don't, we make a single group with the first one
    if not overlaps(to_interval(pred_features[pidx]), to_interval(true_features[tidx])):
        if pred_features[pidx].location.start < true_features[tidx].location.start:
            return [pidx], [], pidx + 1, tidx
        else:
            return [], [tidx], pidx, tidx + 1

    # Note: if there are genes in one list that span multiple genes in another those genes will appear more than once
    # so the lengths of these two sets should be equal (except in the case of no overlaps above)
    pidx_overlaps = []
    tidx_overlaps = []

    # tstart is the first known feature to overlap the previous gene
    # tend is the last known feature to overlap the previous gene
    tstart = tidx
    tend = tidx

    # now we have guaranteed that at least the first set overlap
    # from here we walk along each list, checking for overlaps as we go
    while pidx < len(pred_features):
        # ensure that the strand and type of feature is consistent
        if (
            pred_features[pidx].type != "gene"
            or pred_features[pidx].location.strand != strand
        ):
            pidx += 1
            continue

        # get start and end of predicted gene to match
        p_interval = to_interval(pred_features[pidx])

        # set start point
        tidx = tstart
        update_tstart = True

        while tidx < len(true_features):
            # get next valid true feature to compare
            if (
                true_features[tidx].type != "gene"
                or true_features[tidx].location.strand != strand
            ):
                tidx += 1
                if update_tstart:
                    tstart = tidx
                continue

            # get start and end of true gene to match
            t_interval = to_interval(true_features[tidx])

            if overlaps(p_interval, t_interval):
                pidx_overlaps.append(pidx)
                tidx_overlaps.append(tidx)

                update_tstart = (
                    False  # we have definitely reached at least the first overlap
                )
                tidx += 1
            else:  # there was not an overlap
                # if we have not yet reached tend, then we need to continue checking
                tidx += 1
                if update_tstart and p_interval[0] > t_interval[1]:
                    tstart = tidx

                if tidx > tend:
                    # there was no overlap, and we have exhausted all true genes that overlapped with the previous
                    # predicted gene
                    # new tend is the largest index that had a confirmed match
                    tend = max(tend, tidx_overlaps[-1])
                    break

        # we have just finished finding overlaps for pidx

        if tstart > tend:
            break
        else:
            pidx += 1

    return (
        pidx_overlaps,
        tidx_overlaps,
        max(pidx_overlaps[-1] + 1, pidx),
        max(tidx_overlaps[-1] + 1, tidx),
    )


def do_genes_match(
    pred_gene: Intervals, true_gene: Intervals, tolerance: int = 0
) -> dict[str, bool]:
    """
    Get whether a pair of transcripts match on the cds level, intron level, and full length, with tolerance allowed
    only for the transcript start/stop.

    Args:
        pred_gene: Intervals object for predicted transcript
        true_gene: Intervals object for true transcript
        tolerance: allowed tolerance on transcript start/stop to be considered a full match

    Returns:
        Dictionary with boolean values for three levels of match:
        - 'cds': All CDS intervals match
        - 'intron': All intron intervals match
        - 'full': All exons match. First and last exon may match within the given tolerance
    """
    if pred_gene.cds == true_gene.cds:
        if pred_gene.intron == true_gene.intron:
            # finally, we check if tsStart and tsStop are within allowed tolerance
            pl = pred_gene.exon.lower
            pu = pred_gene.exon.upper
            tl = true_gene.exon.lower
            tu = true_gene.exon.upper

            if (tl - tolerance <= pl <= tl + tolerance) and (
                tu - tolerance <= pu <= tu + tolerance
            ):
                return {"cds": True, "intron": True, "full": True}
            else:  # transcription ends are off but everything else is good
                return {"cds": True, "intron": True, "full": False}
        else:
            # cds's equal, but utr's not
            return {"cds": True, "intron": False, "full": False}
    else:
        # cds's not equal, genes not equal on any level
        return {"cds": False, "intron": False, "full": False}


def evaluate_transcript_matches(
    pred_intervals: dict,
    true_intervals: dict,
    pindices: list[int],
    tindices: list[int],
    edge_tolerance: int = 0,
) -> tuple[ConfusionCounts, ConfusionCounts, ConfusionCounts]:
    """
    Evaluate transcript-level matches between predicted and true intervals.

    Args:
        pred_intervals: Dictionary mapping gene indices to lists of Intervals objects
        true_intervals: Dictionary mapping gene indices to lists of Intervals objects
        pindices: List of predicted gene indices that overlap with true genes
        tindices: List of true gene indices that overlap with predicted genes
        edge_tolerance: Allowed tolerance on transcript start/stop for full matches

    Returns:
        Tuple of ConfusionCounts for (cds_matches, intron_matches, full_matches)
    """
    match_types = ["cds", "intron", "full"]

    # Create boolean tracking dictionaries for matches
    matched_pred = {
        match_type: {idx: False for idx in pred_intervals.keys()}
        for match_type in match_types
    }
    matched_true = {
        match_type: {idx: False for idx in true_intervals.keys()}
        for match_type in match_types
    }

    if len(pindices) != len(tindices):
        raise ValueError(
            f"Predicted and true indices must have the same length; got {len(pindices)=} and {len(tindices)=}"
        )

    for pidx, tidx in zip(pindices, tindices):
        match_results = [
            do_genes_match(pred_transcript, true_transcript, edge_tolerance)
            for pred_transcript, true_transcript in itertools.product(
                pred_intervals[pidx], true_intervals[tidx]
            )
        ]

        # Validate that all match results have the expected keys
        for result in match_results:
            if set(result.keys()) != set(match_types):
                raise ValueError(
                    f"Match result keys {set(result.keys())} do not match expected match_types {set(match_types)}"
                )

        # Mark matches for each type if *any* predicted transcript matches *any* true transcript
        for match_type in match_types:
            if any(result[match_type] for result in match_results):
                matched_pred[match_type][pidx] = True
                matched_true[match_type][tidx] = True

    # collect the match counts into ConfusionCounts objects
    tcds = ConfusionCounts(
        sum(matched_pred["cds"].values()),
        len(pred_intervals) - sum(matched_pred["cds"].values()),
        len(true_intervals) - sum(matched_true["cds"].values()),
    )
    tintron = ConfusionCounts(
        sum(matched_pred["intron"].values()),
        len(pred_intervals) - sum(matched_pred["intron"].values()),
        len(true_intervals) - sum(matched_true["intron"].values()),
    )
    tfull = ConfusionCounts(
        sum(matched_pred["full"].values()),
        len(pred_intervals) - sum(matched_pred["full"].values()),
        len(true_intervals) - sum(matched_true["full"].values()),
    )

    return tcds, tintron, tfull


def overlap_stats(
    pred_features: list[SeqFeature],
    true_features: list[SeqFeature],
    pindices: list[int],
    tindices: list[int],
    edge_tolerance: int = 0,
) -> Stats:
    """
    For a set of predicted and true genes, generate a set of stats at the transcript, exon, intron, and base levels.

    Args:
        pred_features: list of predicted genes as SeqFeature objects
        true_featyres: list of true genes as SeqFeature objects
        pindices: list of indices at which a predicted gene overlaps a true gene. If one predicted gene overlaps
        multiple true genes, the index should appear multiple times
        tindices: list of indices at which a true gene overlaps a predicted gene. If one true gene overlaps
        multiple predicted genes, the index should appear multiple times
        edge_tolerance: int specifying how far apart transcription start/stop site can be from true to be
        considered equal

    Returns:
        Stats object
    """

    # hold transcripts as interval sets
    pred_intervals = {}
    true_intervals = {}

    # keep track of the longest transcript for some of the stats
    pred_longest_index = {}
    true_longest_index = {}

    # convert SeqFeatures into groups of intervals
    for pidx in np.unique(pindices):
        pred_intervals[pidx] = [
            transcript_to_intervals(transcript)
            for transcript in pred_features[pidx].sub_features
        ]
        pred_longest_index[pidx] = get_longest_transcript_index(pred_features[pidx])

    for tidx in np.unique(tindices):
        true_intervals[tidx] = [
            transcript_to_intervals(transcript)
            for transcript in true_features[tidx].sub_features
        ]
        true_longest_index[tidx] = get_longest_transcript_index(true_features[tidx])

    if len(pindices) == 0:  # only one true feature, no predicted features
        return Stats(
            transcript_cds=ConfusionCounts(fn=1),
            transcript_intron=ConfusionCounts(fn=1),
            transcript=ConfusionCounts(fn=1),
            exon_cds=ConfusionCounts(
                fn=sum([len(transcript.cds) for transcript in true_intervals[tidx]])
            ),
            exon=ConfusionCounts(
                fn=sum([len(transcript.exon) for transcript in true_intervals[tidx]])
            ),
            intron_cds=ConfusionCounts(
                fn=sum(
                    [len(transcript.intron_cds) for transcript in true_intervals[tidx]]
                )
            ),
            intron=ConfusionCounts(
                fn=sum([len(transcript.intron) for transcript in true_intervals[tidx]])
            ),
            exon_cds_longest=ConfusionCounts(
                fn=len(true_intervals[tidx][true_longest_index[tidx]].cds)
            ),
            exon_longest=ConfusionCounts(
                fn=len(true_intervals[tidx][true_longest_index[tidx]].exon)
            ),
            intron_cds_longest=ConfusionCounts(
                fn=len(true_intervals[tidx][true_longest_index[tidx]].intron_cds)
            ),
            intron_longest=ConfusionCounts(
                fn=len(true_intervals[tidx][true_longest_index[tidx]].intron)
            ),
            base_cds=ConfusionCounts(
                fn=sum(
                    [
                        x.upper - x.lower
                        for x in true_intervals[tidx][true_longest_index[tidx]].cds
                    ]
                )
            ),
            base_utr=ConfusionCounts(
                fn=sum(
                    [
                        x.upper - x.lower
                        for x in true_intervals[tidx][true_longest_index[tidx]].utr
                    ]
                )
            ),
            base_exon=ConfusionCounts(
                fn=sum(
                    [
                        x.upper - x.lower
                        for x in true_intervals[tidx][true_longest_index[tidx]].exon
                    ]
                )
            ),
        )
    elif len(tindices) == 0:  # only one predicted feature, no true features
        return Stats(
            transcript_cds=ConfusionCounts(fp=1),
            transcript_intron=ConfusionCounts(fp=1),
            transcript=ConfusionCounts(fp=1),
            exon_cds=ConfusionCounts(
                fp=sum([len(transcript.cds) for transcript in pred_intervals[pidx]])
            ),
            exon=ConfusionCounts(
                fp=sum([len(transcript.exon) for transcript in pred_intervals[pidx]])
            ),
            intron_cds=ConfusionCounts(
                fp=sum(
                    [len(transcript.intron_cds) for transcript in pred_intervals[pidx]]
                )
            ),
            intron=ConfusionCounts(
                fp=sum([len(transcript.intron) for transcript in pred_intervals[pidx]])
            ),
            exon_cds_longest=ConfusionCounts(
                fp=len(pred_intervals[pidx][pred_longest_index[pidx]].cds)
            ),
            exon_longest=ConfusionCounts(
                fp=len(pred_intervals[pidx][pred_longest_index[pidx]].exon)
            ),
            intron_cds_longest=ConfusionCounts(
                fp=len(pred_intervals[pidx][pred_longest_index[pidx]].intron_cds)
            ),
            intron_longest=ConfusionCounts(
                fp=len(pred_intervals[pidx][pred_longest_index[pidx]].intron)
            ),
            base_cds=ConfusionCounts(
                fp=sum(
                    [
                        x.upper - x.lower
                        for x in pred_intervals[pidx][pred_longest_index[pidx]].cds
                    ]
                )
            ),
            base_utr=ConfusionCounts(
                fp=sum(
                    [
                        x.upper - x.lower
                        for x in pred_intervals[pidx][pred_longest_index[pidx]].utr
                    ]
                )
            ),
            base_exon=ConfusionCounts(
                fp=sum(
                    [
                        x.upper - x.lower
                        for x in pred_intervals[pidx][pred_longest_index[pidx]].exon
                    ]
                )
            ),
        )

    # base level metrics
    pred_cds_interval = P.empty()
    pred_utr_interval = P.empty()
    for pidx in pred_intervals.keys():
        pred_cds_interval = (
            pred_cds_interval | pred_intervals[pidx][pred_longest_index[pidx]].cds
        )
        pred_utr_interval = (
            pred_utr_interval | pred_intervals[pidx][pred_longest_index[pidx]].utr
        )
    pred_exon_interval = pred_cds_interval | pred_utr_interval

    true_cds_interval = P.empty()
    true_utr_interval = P.empty()
    for tidx in true_intervals.keys():
        true_cds_interval = (
            true_cds_interval | true_intervals[tidx][true_longest_index[tidx]].cds
        )
        true_utr_interval = (
            true_utr_interval | true_intervals[tidx][true_longest_index[tidx]].utr
        )

    # exons are the union of cds and utr
    true_exon_interval = true_cds_interval | true_utr_interval

    base_cds = evaluate_bases(pred_cds_interval, true_cds_interval)
    base_utr = evaluate_bases(pred_utr_interval, true_utr_interval)
    base_exon = evaluate_bases(pred_exon_interval, true_exon_interval)

    # transcript level metrics
    tcds, tintron, tfull = evaluate_transcript_matches(
        pred_intervals, true_intervals, pindices, tindices, edge_tolerance
    )

    # exon and intron level metrics
    # TODO we could move building the intervals sets to a separate function for clarity and code reuse
    # for all transcripts, use set to avoid double-counting exons/introns shared between transcripts
    pred_exon_cds_intervals = set()
    pred_exon_intervals = set()
    pred_intron_cds_intervals = set()
    pred_intron_intervals = set()

    # if we just use the longest transcript we don't need to worry about that
    pred_exon_cds_intervals_longest = []
    pred_exon_intervals_longest = []
    pred_intron_cds_intervals_longest = []
    pred_intron_intervals_longest = []

    # build up the range sets
    for pname, pinterval in pred_intervals.items():
        for ptidx, ptranscript in enumerate(pinterval):
            if ptidx == pred_longest_index[pname]:
                pred_exon_cds_intervals_longest.extend(
                    interval_to_tuples(ptranscript.cds)
                )
                pred_exon_intervals_longest.extend(interval_to_tuples(ptranscript.exon))
                pred_intron_cds_intervals_longest.extend(
                    interval_to_tuples(ptranscript.intron_cds)
                )
                pred_intron_intervals_longest.extend(
                    interval_to_tuples(ptranscript.intron)
                )

            pred_exon_cds_intervals.update(interval_to_tuples(ptranscript.cds))
            pred_exon_intervals.update(interval_to_tuples(ptranscript.exon))
            pred_intron_cds_intervals.update(interval_to_tuples(ptranscript.intron_cds))
            pred_intron_intervals.update(interval_to_tuples(ptranscript.intron))

    # convert back to list for compatibility reasons
    pred_exon_cds_intervals = list(pred_exon_cds_intervals)
    pred_exon_intervals = list(pred_exon_intervals)
    pred_intron_cds_intervals = list(pred_intron_cds_intervals)
    pred_intron_intervals = list(pred_intron_intervals)

    # for all transcripts, use set to avoid double-counting exons/introns shared between transcripts
    true_exon_cds_intervals = set()
    true_exon_intervals = set()
    true_intron_cds_intervals = set()
    true_intron_intervals = set()

    # if we just use the longest transcript we don't need to worry about that
    true_exon_cds_intervals_longest = []
    true_exon_intervals_longest = []
    true_intron_cds_intervals_longest = []
    true_intron_intervals_longest = []

    # build up the range sets
    for tname, tinterval in true_intervals.items():
        for ttidx, ttranscript in enumerate(tinterval):
            if ttidx == true_longest_index[tname]:
                true_exon_cds_intervals_longest.extend(
                    interval_to_tuples(ttranscript.cds)
                )
                true_exon_intervals_longest.extend(interval_to_tuples(ttranscript.exon))
                true_intron_cds_intervals_longest.extend(
                    interval_to_tuples(ttranscript.intron_cds)
                )
                true_intron_intervals_longest.extend(
                    interval_to_tuples(ttranscript.intron)
                )

            true_exon_cds_intervals.update(interval_to_tuples(ttranscript.cds))
            true_exon_intervals.update(interval_to_tuples(ttranscript.exon))
            true_intron_cds_intervals.update(interval_to_tuples(ttranscript.intron_cds))
            true_intron_intervals.update(interval_to_tuples(ttranscript.intron))

    # convert back to list for compatibility reasons
    true_exon_cds_intervals = list(true_exon_cds_intervals)
    true_exon_intervals = list(true_exon_intervals)
    true_intron_cds_intervals = list(true_intron_cds_intervals)
    true_intron_intervals = list(true_intron_intervals)

    # evaluate all the intervals
    exon_cds_counts_longest = evaluate_intervals(
        pred_exon_cds_intervals_longest, true_exon_cds_intervals_longest
    )
    exon_counts_longest = evaluate_intervals(
        pred_exon_intervals_longest, true_exon_intervals_longest
    )
    intron_cds_counts_longest = evaluate_intervals(
        pred_intron_cds_intervals_longest, true_intron_cds_intervals_longest
    )
    intron_counts_longest = evaluate_intervals(
        pred_intron_intervals_longest, true_intron_intervals_longest
    )

    exon_cds_counts = evaluate_intervals(
        pred_exon_cds_intervals, true_exon_cds_intervals
    )
    exon_counts = evaluate_intervals(pred_exon_intervals, true_exon_intervals)
    intron_cds_counts = evaluate_intervals(
        pred_intron_cds_intervals, true_intron_cds_intervals
    )
    intron_counts = evaluate_intervals(pred_intron_intervals, true_intron_intervals)

    return Stats(
        tcds,
        tintron,
        tfull,
        exon_cds_counts,
        exon_counts,
        intron_cds_counts,
        intron_counts,
        exon_cds_counts_longest,
        exon_counts_longest,
        intron_cds_counts_longest,
        intron_counts_longest,
        base_cds,
        base_utr,
        base_exon,
    )


def interval_to_tuples(interval: P.Interval):
    """
    Convert an integer object into a tuple

    Args:
        interval: Interval object

    Returns:
        tuple (start, end)
    """
    return [(atom.lower, atom.upper) for atom in interval]


def transcript_to_intervals(transcript: SeqFeature):
    """
    Convert a SeqFeature of a transcript or mRNA into a set of intervals

    Args:
        transcript: SeqFeature for a transcript or mRNA

    Returns:
        Intervals object
    """
    t_interval = P.closedopen(
        int(transcript.location.start), int(transcript.location.end)
    )

    cds_interval = P.empty()
    utr_interval = P.empty()

    for feature in transcript.sub_features:
        if feature.type == "five_prime_UTR" or feature.type == "three_prime_UTR":
            utr_interval = utr_interval | P.closedopen(
                int(feature.location.start), int(feature.location.end)
            )
        elif feature.type == "CDS":
            cds_interval = cds_interval | P.closedopen(
                int(feature.location.start), int(feature.location.end)
            )

    intron_interval = t_interval - cds_interval - utr_interval
    intron_cds_interval = (
        P.closedopen(cds_interval.lower, cds_interval.upper) - cds_interval
    )
    exon_interval = cds_interval | utr_interval

    return Intervals(
        cds=cds_interval,
        utr=utr_interval,
        intron=intron_interval,
        intron_cds=intron_cds_interval,
        exon=exon_interval,
    )


def evaluate_bases(
    pred_intervals: P.Interval, true_intervals: P.Interval
) -> ConfusionCounts:
    """
    Calculate precision, recall and F1 score between predicted and true intervals with individual base as the unit.

    Args:
        pred_intervals: Interval object for predicted intervals
        true_intervals: Integer object for true intervals

    Returns:
        ConfusionCounts object
    """
    true_positives = int(
        np.sum([x.upper - x.lower for x in (true_intervals & pred_intervals)])
    )
    false_negatives = int(
        np.sum([x.upper - x.lower for x in (true_intervals - pred_intervals)])
    )
    false_positives = int(
        np.sum([x.upper - x.lower for x in (pred_intervals - true_intervals)])
    )

    return ConfusionCounts(true_positives, false_positives, false_negatives)


def evaluate_intervals(
    pred_intervals: list[tuple[int, int]],
    true_intervals: list[tuple[int, int]],
    tolerance: int = 0,
) -> ConfusionCounts:
    """
    Calculate true positive, false positive, and false negative counts between predicted and true intervals.

    Args:
        pred_intervals: List of (start, stop) tuples for predicted intervals
        true_intervals: List of (start, stop) tuples for true intervals
        tolerance: Integer specifying how far apart interval bounds can be
                  to be considered equal. Default is 0 (exact match required).

    Returns:
        ConfusionCounts object
    """
    if not pred_intervals or not true_intervals:
        return ConfusionCounts(0, len(pred_intervals), len(true_intervals))

    # Convert input lists to numpy arrays using transpose
    pred_starts, pred_stops = np.array(pred_intervals, dtype=np.int64).T
    true_starts, true_stops = np.array(true_intervals, dtype=np.int64).T

    # Find number of matched predicted and true intervals
    matched_pred_count, matched_true_count = _find_matches_within_tolerance(
        pred_starts, pred_stops, true_starts, true_stops, tolerance
    )
    assert matched_pred_count <= len(pred_intervals)
    assert matched_true_count <= len(true_intervals)

    return ConfusionCounts(
        matched_pred_count,  # TP: predicted intervals with matches
        len(pred_intervals)
        - matched_pred_count,  # FP: predicted intervals without matches
        len(true_intervals) - matched_true_count,  # FN: true intervals without matches
    )
