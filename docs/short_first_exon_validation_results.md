# Short-first-CDS repair validation

Date: 2026-08-26

## Scope

Offline paired validation was performed using the existing GeneCAD raw
predictions and reference annotations. Production decoder code, model weights,
and prediction outputs were not modified.

The complete six-species comparison used Cquinoa, Othomaeum, Rcommunis,
Oeuropaea, Lervoides, and Lsativa. Additional checks used Arabidopsis TAIR12
chromosomes 4 and 5 and Taestivum chromosomes 1D, 4B, and 5A.

## Findings

The current calibrated Kozak repair had mixed results. In the two confirmatory
species it produced four wrong-to-correct and one correct-to-wrong CDS-chain
transitions under the all-reference policy. In the Taestivum chromosome check,
it produced no correction and two correct-to-wrong transitions. It is therefore
not supported as a generally safe repair for this architecture class.

The fixed 3.0 Kozak margin, joint-feature classifier, local re-decoding, wider
same-transcript ATG search, direct pre-ORF path replacement, constrained-prefix
decoding, and broad top-k prefix replacement all failed their safety or recall
criteria. Local re-decoding was particularly harmful, producing three
wrong-to-correct versus 67 correct-to-wrong CDS-chain transitions in the tested
top-decile following-intron cohort.

Explicit prefix candidate enumeration found the exact reference chain for only
17 of 257 errors with top-1 boundaries, 24 with top-10 boundaries, and 40 with
top-50 boundaries. Correct paths are frequently absent from the candidate set;
reranking alone cannot recover them.

One conservative decoder-anchor micro-repair showed positive retrospective
results. It required a downstream start within 120 spliced nucleotides, a
complete same-stop ORF, a Kozak improvement greater than 3, and a
species-relative top-10% following intron. In that stratum it produced four
wrong-to-correct and zero correct-to-wrong CDS-chain transitions, and five
wrong-to-correct and zero correct-to-wrong TIS transitions. Exact-chain error
recall was approximately 1.6%, and the Arabidopsis and Taestivum checks made no
action. This is insufficient evidence for production integration.

## Decision

Retain the raw GeneCAD prediction as the production baseline. Do not merge the
tested repair prototypes. A future repair should default to preserving the raw
prediction, require independent agreement among start, splice, ORF, and
long-intron evidence, and be validated on additional held-out species before
production use.
