# Short first CDS: empirical reanalysis

Date: 2026-09-15. This report updates [the initial analysis](training_vs_decoder_analysis.md) with new computations on existing artifacts and cached predictions. No model training or production predictions were changed.

## Revised conclusion

**The earlier local re-decoding results are not a reliable basis for choosing training over decoder improvements.** The failures are strongly associated with negative-strand cache-coordinate handling. On all 11 Arabidopsis cohort loci, coordinate-aware replay reproduces the original chains; deliberately treating storage row numbers as genomic coordinates instead reproduces all 11 historical experimental chains, including their negative-strand failures.

The 17-tag boundary hypothesis also now has a small empirical check. Existing tag evidence is informative, but a simple conditional boundary score does not outperform the existing five-feature intron score in the tested paired comparison. There is no evidence yet that exposing the tags alone solves the problem.

Recommended order: keep the current production behavior, finish the
coordinate-correct baseline, and use the results to prepare an error-focused
training set. Run the next training comparison when an updated base model is
available. Do not implement the proposed decoder roadmap from these results.

## 1. Recomputed strand-specific results

The source table contains 374 unique loci, each with three local decoder variants. They are a selected short-first-CDS/top-decile-intron cohort, not a representative genome sample. Raw/reference chain agreement gives 117 correct loci and 257 errors under the table's chosen reference policy.

For the historical `run0` local baseline:

| Strand | Loci | Exact replay of original chain | Empty decoded chain | Correct→wrong |
| --- | ---: | ---: | ---: | ---: |
| Positive | 174 | 174 | 0 | 0 |
| Negative | 200 | 7 | 162 | 67 |

Of the 67 harms, 63 are empty-chain outputs. All three corrections attributed to the CDS-only penalty occur on the positive strand. These counts were independently recomputed from the chain intervals rather than copied from the existing summary.

Sources: [audit script](../../pipelines/experiments/short_first_cds_reanalysis/audit_existing.py), [summary](../../pipelines/experiments/short_first_cds_reanalysis/audit_summary.json), [per-locus audit](../../pipelines/experiments/short_first_cds_reanalysis/locus_audit.tsv).

## 2. Controlled replay identifies a reproducible coordinate failure mode

### Cache storage order is not genomic order

In the original Arabidopsis chromosome-4 cache, the first three `sequence` coordinates are:

| Rank store | Positive strand | Negative strand |
| --- | --- | --- |
| `predictions.0.zarr` | 0, 1, 2 | 25,525,153; 25,525,154; 25,525,155 |
| `predictions.1.zarr` | 12,888,064; 12,888,065; 12,888,066 | 12,639,137; 12,639,138; 12,639,139 |

Thus array row 0 is not necessarily genomic coordinate 0. Reversing the logits for negative-strand decoding does not repair an earlier lookup at the wrong genomic position.

### New replay, using the existing cache and current graph

The experiment uses all 11 Arabidopsis loci from the historical table, their original windows, the original `Athaliana_TAIR12_chr4_finetuned` cache (not the later rerun cache), the matching local FASTA, current plant transition matrix, no short-run penalty, and no added intergenic bias/alpha. It selects a decoded gene by maximum CDS-base overlap with the original chain. The diagnostic asserts complete and nonduplicated coordinate coverage for each crop.

| Cache lookup | Positive: replay original | Negative: replay original | Match historical experiment |
| --- | ---: | ---: | ---: |
| Actual `sequence` coordinate lookup | 5/5 | 6/6 | 5/11 |
| Deliberately wrong: concatenate ranks and slice by row number | 5/5 | 0/6 | 11/11 |

Coordinate-aware replay yields 9/11 reference-exact chains, preserving the same two original errors. It does not fix the model. It restores a valid experimental baseline. Simple orientation mistakes tested separately did not reproduce the historical chains on these six negative loci; wrong positional indexing did.

This is strong mechanistic evidence for a cache-coordinate problem in the old experiment. The original generating script was not located, so it is not a line-level diagnosis of that script, and full-cohort replay has not yet established the cause of all 67 harms. Nonetheless, those harms cannot credibly be presented as evidence that a new short-CDS penalty itself damages predictions.

The current production [cache merge](../../src/prediction.py) sorts by `sequence` and checks complete coordinate coverage; [negative-strand decoding](../../scripts/detect_intervals.py), lines 160–174, reverses logits, reverse-complements bases, and restores output order. The reproduced failure belongs to the offline experiment; it does not demonstrate a bug in those current production steps.

Sources: [coordinate-aware replay script](../../pipelines/experiments/short_first_cds_reanalysis/replay.py), [results](../../pipelines/experiments/short_first_cds_reanalysis/replay_results.json), [deliberately wrong positional control](../../pipelines/experiments/short_first_cds_reanalysis/positional_replay.py), [control results](../../pipelines/experiments/short_first_cds_reanalysis/positional_results.json).

## 3. What the error set actually contains

Comparing CDS intervals in transcription order partitions the 257 reference mismatches as follows:

| Geometric discrepancy | Loci |
| --- | ---: |
| Only TIS boundary within first CDS segment differs | 2 |
| Can remove leading CDS segments and move start into a retained segment, preserving downstream CDS | 48 |
| First CDS segment differs beyond its TIS; remaining segments identical | 30 |
| Multiple prefix differences, but at least one exact downstream CDS segment shared | 90 |
| No exact terminal CDS segment shared | 87 |

Only 50/257 (19.5%) fit these two simple start-change geometries. The remaining 207/257 (80.5%) require other CDS interval changes. This helps explain why a same-transcript Kozak repair has limited scope.

These are coordinate classifications, not biological feasibility tests. They do not check ATG, reading frame, matching stop, physical exon boundaries, or reference certainty. Some of the 50 would fail actual `fix_orf` restrictions; some other start relocations could involve sequence outside the original CDS. The 87 terminal-segment mismatches do not necessarily have different stop positions: a different terminal splice boundary also changes that interval. The whole cohort should not be described as isolated first-exon errors.

## 4. New sparse comparison of existing boundary scores

We read cached 17-tag and five-feature logits by their actual genomic coordinates, without neural inference. Among the cohort, 213 loci have distinct predicted/reference first donor positions and at least two CDS segments in both chains; all 213 paired scores were found across six species. The control group contains 117 exact-chain loci, of which 108 had scores at the conventional six-species cache paths; nine Arabidopsis controls were missing from that particular lookup. Arabidopsis replay above uses its separately named cache.

At the first intronic base in transcript orientation, compare:

- Existing intron feature probability, `P(intron)`.
- Conditional beginning-of-intron tag probability, `P(B-intron | intron)`.
- Joint beginning-of-intron probability, `P(B-intron)`.

| Score | Reference donor scores higher | Predicted donor scores higher |
| --- | ---: | ---: |
| Existing five-feature intron score | 165/213 (77.5%) | 48/213 |
| Conditional B-intron score | 136/213 (63.8%) | 77/213 |
| Joint B-intron score | 132/213 (62.0%) | 81/213 |

The conditional-score median log advantage is only about 0.0058. The small per-species sample sizes and selected, correlated loci preclude treating these fractions as general donor accuracy. Exact-chain controls have zero paired differences because they compare the same coordinate; that is a lookup consistency check, not a measure of specificity.

**Interpretation:** existing emissions often already favor the reference donor at that single base. This makes full-path score analysis worthwhile, but does not show that the correct complete path should win: its TIS, other junctions, long intron body, and transition contributions can differ. The conditional tag score is not an obvious replacement for the existing signal. Neither ranking experiment evaluates successful gene repair, and no weights were trained or tuned.

These “reference first donor” pairs need not be alternative choices for the same physical junction when the structures have different leading exons. They are exploratory first-donor comparisons. Joint tag scores also use the same model evidence as the feature score; naively adding both risks double-counting.

Stored feature logits and a float64 logsumexp reconstruction of the saved token logits differ by up to about 0.0194 on inspected positions. Feature aggregation occurs before conversion of the tensors to float32 in [prediction code](../../scripts/predict.py), lines 379–395, so arithmetic precision is a possible explanation; the exact historical arithmetic was not reproduced. Do not claim bitwise equivalence.

Sources: [boundary script](../../pipelines/experiments/short_first_cds_reanalysis/boundary_scores.py), [paired scores](../../pipelines/experiments/short_first_cds_reanalysis/per_boundary.tsv), [summary and cache provenance](../../pipelines/experiments/short_first_cds_reanalysis/boundary_summary.json).

## 5. Previous oracle scores still cannot attribute the cause

All 117 original/reference-exact CDS chains have nonzero recorded oracle-minus-raw scores. All 374 records have a positive transition component. CDS equality does not imply UTR/state-path equality, so these observations alone do not prove a scoring bug. They do show that the comparison contains more than the CDS discrepancy of interest, or uses inconsistent construction.

Given the reproduced coordinate problem, rebuild those scores using correctly aligned logits and the same interval, boundary conditions, and allowed graph. An identical full state path must have zero score difference. Only then use emission-versus-transition differences to discuss training versus decoder causes. The earlier positive-oracle counts should not support either side of that debate.

## 6. Revised decision and next experiment

1. Treat the old local negative-strand results as invalid for causal comparison until coordinate-correct full-cohort replay is complete. Retain historical files as provenance.
2. Recompute run0 and short-CDS penalty variants on the same correctly aligned inputs, with matching locus extraction. Verify baseline agreement separately by strand; extend the 11-locus replay to the full selected cohort and ordinary-gene controls.
3. Rebuild reference feasibility and full-path score decomposition; separate TIS-only, first-junction, and broader structural mismatches.
4. Use the corrected results to choose examples, labels, and context for the
   next training run. Record any reference paths that the decoder cannot
   represent; those are separate decoder issues that retraining cannot fix.
5. Reserve untouched data for the final comparison; these loci are now development data. Keep true short CDS, exact TIS/junction/chain accuracy, and newly introduced errors in the decision criteria.

There is still no matched retraining run, so this report does not measure how
much retraining will help. It also gives no support for adding more penalties
or decoder terms. Its main result is that the earlier local-decoding comparison
was affected by a reproducible coordinate error.

Reproduction commands and input hashes are in the [experiment directory](../../pipelines/experiments/short_first_cds_reanalysis/README.md). The original cached datasets remain local dependencies; this report does not claim a fully portable data bundle.
