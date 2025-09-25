import src.evaluation as ev
from Bio.SeqFeature import SeqFeature, FeatureLocation


def test_exact_matches():
    """Test basic exact matching with no tolerance."""
    pred_intervals = [(1, 5), (10, 15), (20, 25)]
    true_intervals = [(1, 5), (10, 15), (30, 35)]

    metrics = ev.evaluate_intervals(pred_intervals, true_intervals)

    assert metrics.tp == 2  # 2 matches out of 3 predictions
    assert metrics.fp == 1  # 2 matches out of 3 true intervals
    assert metrics.fn == 1  # F1 = 2 * (2/3 * 2/3) / (2/3 + 2/3)


def test_empty_inputs():
    """Test behavior with empty inputs."""
    # Empty predicted intervals
    metrics = ev.evaluate_intervals([], [(1, 5), (10, 15)])
    assert metrics.tp == 0
    assert metrics.fp == 0
    assert metrics.fn == 2

    # Empty true intervals
    metrics = ev.evaluate_intervals([(1, 5), (10, 15)], [])
    assert metrics.tp == 0
    assert metrics.fp == 2
    assert metrics.fn == 0

    # Both empty
    metrics = ev.evaluate_intervals([], [])
    assert metrics.tp == 0
    assert metrics.fp == 0
    assert metrics.fn == 0


def test_multiple_matches():
    """Test exact matching with multiple identical intervals."""
    # Multiple predicted intervals match single true interval
    pred_intervals = [(1, 5), (1, 5), (10, 15)]
    true_intervals = [(1, 5), (20, 25)]
    metrics = ev.evaluate_intervals(pred_intervals, true_intervals)
    assert metrics.tp == 2  # Both (1,5) predicted intervals have matches
    assert metrics.fp == 1  # (10,15) predicted interval has no match
    assert metrics.fn == 1  # (20,25) true interval has no match

    # Multiple true intervals match single predicted interval
    pred_intervals = [(1, 5), (10, 15)]
    true_intervals = [(1, 5), (1, 5), (20, 25)]
    metrics = ev.evaluate_intervals(pred_intervals, true_intervals)
    assert metrics.tp == 1  # (1,5) predicted interval has matches
    assert metrics.fp == 1  # (10,15) predicted interval has no match
    assert metrics.fn == 1  # (20,25) true interval has no match

    # Multiple matches in both predictions and true intervals
    pred_intervals = [(1, 5), (1, 5), (10, 15), (10, 15)]
    true_intervals = [(1, 5), (1, 5), (10, 15), (20, 25)]
    metrics = ev.evaluate_intervals(pred_intervals, true_intervals)
    assert metrics.tp == 4  # All predicted intervals have matches
    assert metrics.fp == 0  # No predicted intervals without matches
    assert metrics.fn == 1  # (20,25) true interval has no match


def test_approximate_matches():
    """Test matching with non-zero tolerance."""
    # Test with tolerance=1
    pred_intervals = [(1, 5), (10, 16), (20, 25)]
    true_intervals = [(2, 6), (10, 15), (30, 35)]
    metrics = ev.evaluate_intervals(pred_intervals, true_intervals, tolerance=1)
    assert metrics.tp == 2  # First two predictions match within tolerance
    assert metrics.fp == 1  # First two true intervals matched
    assert metrics.fn == 1

    # Test with tolerance=2
    pred_intervals = [(1, 5), (11, 16)]
    true_intervals = [(3, 7), (10, 15)]
    metrics = ev.evaluate_intervals(pred_intervals, true_intervals, tolerance=2)
    assert metrics.tp == 2  # Both predictions match within tolerance
    assert metrics.fp == 0  # Both true intervals matched
    assert metrics.fn == 0


def test_nested_matches():
    """Test nested exact matching with no tolerance."""
    pred_intervals = [(1, 5), (8, 10), (12, 16)]
    true_intervals = [(1, 10), (12, 16)]

    metrics = ev.evaluate_intervals(pred_intervals, true_intervals)

    assert metrics.tp == 1
    assert metrics.fp == 2
    assert metrics.fn == 1


def test_confusion_count_addition():
    x = ev.ConfusionCounts(1, 15, 98)
    y = ev.ConfusionCounts(254, 82, 215)

    z = x + y

    assert z.tp == 1 + 254
    assert z.fp == 15 + 82
    assert z.fn == 98 + 215


def test_confusion_count_pr():
    counts = ev.ConfusionCounts(50, 20, 10)

    assert counts.precision() == 5 / 7
    assert counts.recall() == 5 / 6
    assert counts.f1() == 10 / 13


def test_confusion_count_zero():
    counts = ev.ConfusionCounts()

    assert counts.precision() == 0
    assert counts.recall() == 0
    assert counts.f1() == 0


def test_overlaps():
    a = (5, 20)
    b = (10, 15)
    c = (5, 15)
    d = (10, 20)
    e = (15, 20)
    f = (5, 15)
    g = (5, 10)

    assert ev.overlaps(a, b)
    assert ev.overlaps(b, a)
    assert ev.overlaps(c, d)
    assert ev.overlaps(d, c)
    assert ev.overlaps(a, c)
    assert ev.overlaps(c, a)
    assert ev.overlaps(a, d)
    assert ev.overlaps(d, a)

    assert not ev.overlaps(e, f)
    assert not ev.overlaps(f, e)
    assert not ev.overlaps(e, g)
    assert not ev.overlaps(g, e)


def test_feature_to_interval():
    a = SeqFeature(FeatureLocation(10, 20, strand=1))
    b = SeqFeature(FeatureLocation(46, 874, strand=-1))

    assert ev.to_interval(a) == (10, 20)
    assert ev.to_interval(b) == (46, 874)


def test_get_longest_feature():
    # actual longest transcript
    transcript_b = SeqFeature(FeatureLocation(30, 160, strand=1), type="mRNA")
    transcript_b.sub_features = [
        SeqFeature(FeatureLocation(30, 60, strand=1), type="five_prime_UTR"),
        SeqFeature(FeatureLocation(60, 70, strand=1), type="CDS"),
        SeqFeature(FeatureLocation(80, 90, strand=1), type="CDS"),
        SeqFeature(FeatureLocation(100, 110, strand=1), type="CDS"),
        SeqFeature(FeatureLocation(110, 130, strand=1), type="three_prime_UTR"),
        SeqFeature(FeatureLocation(140, 160, strand=1), type="three_prime_UTR"),
    ]

    # longest if you double-count exons
    transcript_a = SeqFeature(FeatureLocation(50, 120, strand=1), type="mRNA")
    transcript_a.sub_features = [
        SeqFeature(FeatureLocation(50, 80, strand=1), type="five_prime_UTR"),
        SeqFeature(FeatureLocation(80, 90, strand=1), type="CDS"),
        SeqFeature(FeatureLocation(90, 120, strand=1), type="three_prime_UTR"),
        SeqFeature(FeatureLocation(50, 120, strand=1), type="exon"),
    ]

    # longest if you include introns
    transcript_c = SeqFeature(FeatureLocation(10, 170, strand=1), type="mRNA")
    transcript_c.sub_features = [
        SeqFeature(FeatureLocation(10, 20, strand=1), type="five_prime_UTR"),
        SeqFeature(FeatureLocation(30, 40, strand=1), type="CDS"),
        SeqFeature(FeatureLocation(110, 130, strand=1), type="CDS"),
        SeqFeature(FeatureLocation(150, 170, strand=1), type="three_prime_UTR"),
    ]

    gene = SeqFeature(FeatureLocation(10, 170, strand=1), type="gene")
    gene.sub_features = [transcript_a, transcript_b, transcript_c]

    assert ev.get_longest_transcript_index(gene) == 1


def test_overlap_set_empty():
    pindices, tindices, pnext, tnext = ev.get_next_overlap_set([], [], 0, 0)

    assert pindices == []
    assert tindices == []
    assert pnext == 0
    assert tnext == 0

    # sets contain something but index is past the end of the list

    pindices, tindices, pnext, tnext = ev.get_next_overlap_set(
        [SeqFeature(FeatureLocation(0, 10, strand=1), type="gene")],
        [SeqFeature(FeatureLocation(0, 10, strand=1), type="gene")],
        1,
        1,
    )

    assert pindices == []
    assert tindices == []
    assert pnext == 1
    assert tnext == 1


def test_overlap_set_singletons():
    p_features = [
        SeqFeature(FeatureLocation(20, 50, strand=1), type="gene"),
        SeqFeature(FeatureLocation(100, 150, strand=1), type="gene"),
    ]
    t_features = [SeqFeature(FeatureLocation(60, 120, strand=1), type="gene")]

    # simple no overlap
    pindices, tindices, pnext, tnext = ev.get_next_overlap_set(
        p_features, t_features, 0, 0
    )
    assert pindices == [0]
    assert tindices == []
    assert pnext == 1
    assert tnext == 0

    # and on the negative strand
    p_features = [
        SeqFeature(FeatureLocation(20, 50, strand=-1), type="gene"),
        SeqFeature(FeatureLocation(100, 150, strand=-1), type="gene"),
    ]
    t_features = [SeqFeature(FeatureLocation(60, 120, strand=-1), type="gene")]

    # simple no overlap
    pindices, tindices, pnext, tnext = ev.get_next_overlap_set(
        p_features, t_features, 0, 0, -1
    )
    assert pindices == [0]
    assert tindices == []
    assert pnext == 1
    assert tnext == 0

    # overlap with ncRNA, but not gene
    p_features = [
        SeqFeature(FeatureLocation(100, 130, strand=1), type="ncRNA"),
        SeqFeature(FeatureLocation(140, 150, strand=1), type="gene"),
    ]
    t_features = [SeqFeature(FeatureLocation(60, 120, strand=1), type="gene")]

    pindices, tindices, pnext, tnext = ev.get_next_overlap_set(
        p_features, t_features, 0, 0
    )
    assert pindices == []
    assert tindices == [0]
    assert pnext == 1
    assert tnext == 1

    # overlap on the wrong strand
    p_features = [
        SeqFeature(FeatureLocation(20, 70, strand=1), type="gene"),
        SeqFeature(FeatureLocation(100, 150, strand=1), type="gene"),
    ]
    t_features = [SeqFeature(FeatureLocation(60, 120, strand=-1), type="gene")]

    pindices, tindices, pnext, tnext = ev.get_next_overlap_set(
        p_features, t_features, 0, 0
    )
    assert pindices == [0]
    assert tindices == []
    assert pnext == 1
    assert tnext == 1

    pindices, tindices, pnext, tnext = ev.get_next_overlap_set(
        p_features, t_features, 0, 0, -1
    )
    assert pindices == []
    assert tindices == [0]
    assert pnext == 2
    assert tnext == 1


def test_overlap_set():
    p_features = [
        SeqFeature(FeatureLocation(20, 70, strand=1), type="ncRNA"),
        SeqFeature(FeatureLocation(100, 150, strand=1), type="gene"),
        SeqFeature(FeatureLocation(180, 240, strand=1), type="gene"),
        SeqFeature(FeatureLocation(250, 350, strand=1), type="gene"),
        SeqFeature(FeatureLocation(380, 450, strand=1), type="gene"),
        SeqFeature(FeatureLocation(700, 900, strand=1), type="gene"),
        SeqFeature(FeatureLocation(950, 1030, strand=1), type="gene"),
    ]

    t_features = [
        SeqFeature(FeatureLocation(30, 80, strand=1), type="ncRNA"),
        SeqFeature(FeatureLocation(120, 270, strand=1), type="gene"),
        SeqFeature(FeatureLocation(290, 300, strand=1), type="lnc_rna"),
        SeqFeature(FeatureLocation(320, 390, strand=1), type="gene"),
        SeqFeature(FeatureLocation(410, 430, strand=-1), type="gene"),
        SeqFeature(FeatureLocation(440, 460, strand=1), type="gene"),
        SeqFeature(FeatureLocation(680, 730, strand=1), type="gene"),
    ]

    # positive strand
    pindices, tindices, pnext, tnext = ev.get_next_overlap_set(
        p_features, t_features, 0, 0
    )
    assert pindices == [1, 2, 3, 3, 4, 4]
    assert tindices == [1, 1, 1, 3, 3, 5]
    assert pnext == 5
    assert tnext == 6

    pindices, tindices, pnext, tnext = ev.get_next_overlap_set(
        p_features, t_features, 5, 6
    )
    assert pindices == [5]
    assert tindices == [6]
    assert pnext == 6
    assert tnext == 7

    # same thing but invert the strands
    p_features = [
        SeqFeature(FeatureLocation(20, 70, strand=-1), type="ncRNA"),
        SeqFeature(FeatureLocation(100, 150, strand=-1), type="gene"),
        SeqFeature(FeatureLocation(180, 240, strand=-1), type="gene"),
        SeqFeature(FeatureLocation(250, 350, strand=-1), type="gene"),
        SeqFeature(FeatureLocation(380, 450, strand=-1), type="gene"),
        SeqFeature(FeatureLocation(480, 500, strand=-1), type="ncRNA"),
        SeqFeature(FeatureLocation(600, 630, strand=1), type="gene"),
        SeqFeature(FeatureLocation(700, 900, strand=-1), type="gene"),
    ]

    t_features = [
        SeqFeature(FeatureLocation(30, 80, strand=-1), type="ncRNA"),
        SeqFeature(FeatureLocation(120, 270, strand=-1), type="gene"),
        SeqFeature(FeatureLocation(290, 300, strand=-1), type="lnc_rna"),
        SeqFeature(FeatureLocation(320, 390, strand=-1), type="gene"),
        SeqFeature(FeatureLocation(410, 430, strand=1), type="gene"),
        SeqFeature(FeatureLocation(440, 460, strand=-1), type="gene"),
        SeqFeature(FeatureLocation(680, 730, strand=-1), type="gene"),
    ]

    # positive strand
    pindices, tindices, pnext, tnext = ev.get_next_overlap_set(
        p_features, t_features, 0, 0, -1
    )
    assert pindices == [1, 2, 3, 3, 4, 4]
    assert tindices == [1, 1, 1, 3, 3, 5]
    assert pnext == 7
    assert tnext == 6

    pindices, tindices, pnext, tnext = ev.get_next_overlap_set(
        p_features, t_features, 5, 6, -1
    )
    assert pindices == [7]
    assert tindices == [6]
    assert pnext == 8
    assert tnext == 7


def test_overlap_set_nested():
    p_features = [
        SeqFeature(FeatureLocation(30, 80, strand=1), type="gene"),
        SeqFeature(FeatureLocation(40, 60, strand=1), type="gene"),
        SeqFeature(FeatureLocation(100, 150, strand=1), type="gene"),
        SeqFeature(FeatureLocation(120, 220, strand=1), type="ncRNA"),
        SeqFeature(FeatureLocation(200, 300, strand=1), type="gene"),
        SeqFeature(FeatureLocation(240, 360, strand=-1), type="gene"),
        SeqFeature(FeatureLocation(600, 650, strand=1), type="gene"),
    ]

    t_features = [
        SeqFeature(FeatureLocation(20, 70, strand=1), type="gene"),
        SeqFeature(FeatureLocation(35, 85, strand=-1), type="gene"),
        SeqFeature(FeatureLocation(75, 110, strand=1), type="gene"),
        SeqFeature(FeatureLocation(105, 110, strand=1), type="gene"),
        SeqFeature(FeatureLocation(120, 310, strand=1), type="gene"),
        SeqFeature(FeatureLocation(160, 170, strand=1), type="miRNA"),
        SeqFeature(FeatureLocation(500, 700, strand=1), type="gene"),
        SeqFeature(FeatureLocation(620, 650, strand=1), type="gene"),
    ]
    pindices, tindices, pnext, tnext = ev.get_next_overlap_set(
        p_features, t_features, 0, 0
    )
    assert pindices == [0, 0, 1, 2, 2, 2, 4]
    assert tindices == [0, 2, 0, 2, 3, 4, 4]
    assert pnext == 6
    assert tnext == 5

    pindices, tindices, pnext, tnext = ev.get_next_overlap_set(
        p_features, t_features, 6, 5
    )
    assert pindices == [6, 6]
    assert tindices == [6, 7]
    assert pnext == 7
    assert tnext == 8


def test_transcript_to_intervals():
    transcript = SeqFeature(FeatureLocation(30, 160, strand=1), type="mRNA")
    transcript.sub_features = [
        SeqFeature(FeatureLocation(30, 60, strand=1), type="five_prime_UTR"),
        SeqFeature(FeatureLocation(60, 70, strand=1), type="CDS"),
        SeqFeature(FeatureLocation(60, 70, strand=1), type="exon"),
        SeqFeature(FeatureLocation(80, 90, strand=1), type="CDS"),
        SeqFeature(FeatureLocation(90, 100, strand=1), type="intron"),
        SeqFeature(FeatureLocation(100, 110, strand=1), type="CDS"),
        SeqFeature(FeatureLocation(110, 130, strand=1), type="three_prime_UTR"),
        SeqFeature(FeatureLocation(140, 160, strand=1), type="three_prime_UTR"),
    ]

    intervals = ev.transcript_to_intervals(transcript)

    # with UTRs
    assert intervals.cds == {(60, 70), (80, 90), (100, 110)}
    assert intervals.utr == {(30, 60), (110, 130), (140, 160)}
    assert intervals.intron == {(70, 80), (90, 100), (130, 140)}
    assert intervals.intron_cds == {(70, 80), (90, 100)}
    assert intervals.exon == {(30, 70), (80, 90), (100, 130), (140, 160)}


def test_transcript_to_intervals_missing_utr():
    # without UTRs

    transcript = SeqFeature(FeatureLocation(60, 110, strand=1), type="mRNA")
    transcript.sub_features = [
        SeqFeature(FeatureLocation(60, 70, strand=1), type="CDS"),
        SeqFeature(FeatureLocation(60, 70, strand=1), type="exon"),
        SeqFeature(FeatureLocation(80, 90, strand=1), type="CDS"),
        SeqFeature(FeatureLocation(90, 100, strand=1), type="intron"),
        SeqFeature(FeatureLocation(100, 110, strand=1), type="CDS"),
    ]

    intervals = ev.transcript_to_intervals(transcript)

    assert intervals.cds == {(60, 70), (80, 90), (100, 110)}
    assert intervals.utr == set()
    assert intervals.intron == {(70, 80), (90, 100)}
    assert intervals.intron_cds == {(70, 80), (90, 100)}
    assert intervals.exon == {(60, 70), (80, 90), (100, 110)}


def test_gene_match():
    true_gene = ev.Intervals(
        {(40, 50), (60, 80)},
        {(10, 30), (80, 90), (110, 130)},
        {(30, 40), (50, 60), (90, 110)},
        {(50, 60)},
        {(10, 30), (40, 50), (60, 90), (110, 130)},
    )
    match_w_tolerance = ev.Intervals(
        {(40, 50), (60, 80)},
        {(15, 30), (80, 90), (110, 138)},
        {(30, 40), (50, 60), (90, 110)},
        {(50, 60)},
        {(15, 30), (40, 50), (60, 90), (110, 138)},
    )
    cds_match = ev.Intervals(
        {(40, 50), (60, 80)},
        {(28, 30), (85, 92), (110, 138)},
        {(30, 40), (50, 60), (80, 85), (92, 110)},
        {(50, 60)},
        {(28, 30), (40, 50), (60, 80), (85, 92), (110, 138)},
    )
    nonmatch = ev.Intervals(
        {(40, 50), (60, 81)},
        {(10, 30), (81, 90), (110, 130)},
        {(30, 40), (50, 60), (90, 110)},
        {(50, 60)},
        {(10, 30), (40, 50), (60, 90), (110, 130)},
    )

    assert ev.do_genes_match(match_w_tolerance, true_gene, 10) == {
        "cds": True,
        "intron": True,
        "full": True,
    }
    assert ev.do_genes_match(match_w_tolerance, true_gene, 5) == {
        "cds": True,
        "intron": True,
        "full": False,
    }
    assert ev.do_genes_match(cds_match, true_gene, 10) == {
        "cds": True,
        "intron": False,
        "full": False,
    }
    assert ev.do_genes_match(nonmatch, true_gene, 10) == {
        "cds": False,
        "intron": False,
        "full": False,
    }


def test_duplicated_true_intervals():
    """Test behavior with duplicated true intervals and unique predicted intervals."""
    pred_intervals = [(10, 20), (30, 40), (50, 60)]
    true_intervals = [(10, 20), (10, 20), (30, 40), (70, 80)]  # 2 duplicates of (10,20)

    metrics = ev.evaluate_intervals(pred_intervals, true_intervals, tolerance=0)

    # Expected: TP=2 (pred intervals (10,20) and (30,40) have matches)
    #          FP=1 (pred interval (50,60) has no match)
    #          FN=1 (true interval (70,80) has no match)
    assert metrics.tp == 2
    assert metrics.fp == 1
    assert metrics.fn == 1


def test_duplicated_pred_intervals():
    """Test behavior with duplicated predicted intervals and unique true intervals."""
    pred_intervals = [(10, 20), (10, 20), (30, 40), (50, 60)]  # 2 duplicates of (10,20)
    true_intervals = [(10, 20), (30, 40), (70, 80)]

    metrics = ev.evaluate_intervals(pred_intervals, true_intervals, tolerance=0)

    # Expected: TP=3 (all pred intervals that match: 2×(10,20) + 1×(30,40))
    #          FP=1 (pred interval (50,60) has no match)
    #          FN=1 (true interval (70,80) has no match)
    assert metrics.tp == 3
    assert metrics.fp == 1
    assert metrics.fn == 1


def test_duplicated_both_intervals():
    """Test behavior with duplicated intervals in both predicted and true."""
    pred_intervals = [(10, 20), (10, 20), (30, 40)]  # 2 duplicates of (10,20)
    true_intervals = [(10, 20), (10, 20), (10, 20), (50, 60)]  # 3 duplicates of (10,20)

    metrics = ev.evaluate_intervals(pred_intervals, true_intervals, tolerance=0)

    # Expected: TP=2 (both pred intervals (10,20) have matches, (30,40) doesn't)
    #          FP=1 (pred interval (30,40) has no match)
    #          FN=1 (true interval (50,60) has no match)
    assert metrics.tp == 2
    assert metrics.fp == 1
    assert metrics.fn == 1


def test_tolerance_multiple_matches():
    """Test behavior with tolerance=1 causing multiple matches."""
    pred_intervals = [(10, 20), (30, 40)]
    true_intervals = [(11, 21), (29, 39), (31, 41), (50, 60)]

    metrics = ev.evaluate_intervals(pred_intervals, true_intervals, tolerance=1)

    # With tolerance=1:
    # - (11,21) matches (10,20) ✓
    # - (29,39) matches (30,40) ✓
    # - (31,41) matches (30,40) ✓
    # - (50,60) matches nothing
    # Expected: TP=2 (both pred intervals have matches)
    #          FP=0 (no pred intervals are unmatched)
    #          FN=1 (true interval (50,60) has no match)
    assert metrics.tp == 2
    assert metrics.fp == 0
    assert metrics.fn == 1
