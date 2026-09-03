# Ab initio avoidance of false short first-CDS-exon + long-intron calls

## Problem definition

This error comes in at least two forms, and a single Kozak rule can't cover both:

1. **TIS is wrong but splice structure is right**: the first short CDS segment should be 5' UTR, and the real ATG sits in a downstream exon. `fix_orf.py`'s Kozak switch handles part of this, but it requires matching frame and stop, and keeps the original intron.
2. **The first exon/donor/intron itself is fake**: post-hoc ATG relocation isn't enough here, because a "new gene path starting downstream" or some other splice path should win in the global decode instead.

GeneCAD already has a solid starting point: `FrameStateGraph` enforces legal ATG, stop, reading frame, and canonical splice motifs, and applies a soft penalty to short coding runs via
`(length / 9) ** strictness`; testing in this repo already shows a hard 9-nt floor hurts real short exons. The current limits are:

- the decoder still runs a single Viterbi path over five-class per-base emissions plus feature transitions;
- splice boundaries only check that the dinucleotide is legal; nothing in the transition directly scores how real a given donor/acceptor looks;
- the intron body loop implicitly gives something close to a geometric duration, and the short-exon prior only covers `<9 nt`;
- `include_utr_in_coding_run=True` conflates physical first-exon length with first **CDS**-segment length into one quantity;
- Kozak only enters at the `fix_orf.py` stage, so it can't compete directly against an alternative gene/splice path.

So the goal shouldn't be "ban short exons" or "ban long introns." It should be keeping a short exon only when the start, splice, coding, duration, and whole-ORF evidence are all strong enough together.

## Literature-backed intrinsic approaches

### 1. Explicit-duration / semi-Markov length models

GENSCAN's generalized HMM models transcription/translation/splice signals together with exon/intron composition and length distributions; GeneMark-ES's HSMM goes further, building separate three-periodic coding models and data-smoothed state-duration distributions for initial, internal, terminal, and single-exon coding states. This argues that the **initial CDS exon should be its own state/type**, not just share a generic CDS→intron transition. [Burge & Karlin 1997](https://pubmed.ncbi.nlm.nih.gov/9149143/), [Lomsadze et al. 2005](https://academic.oup.com/nar/article/33/20/6494/1082033)

AUGUSTUS's key warning: a plain HMM self-loop can only produce a shifted-geometric intron length, which is the wrong shape for short introns and over-penalizes real long ones. AUGUSTUS uses an empirical explicit distribution for the short range and only switches to a geometric state for the tail. [Stanke & Waack 2003](https://gobics.de/mario/papers/GenePred2003.pdf)

What this means for GeneCAD:

- Build an empirical/smoothed log prior for `initial CDS length`, with enough resolution over the error-prone 3–60 nt range; beyond the cap, fall into a tail state.
- Intron duration can also use an "empirical head + geometric tail," but **long introns can't just get penalized across the board**; that would break real plant genes that legitimately have long introns.
- A regularized interaction term like `log P(L_initial, L_intron)` or `log P(L_initial | intron-bin)` is worth evaluating, but only adopt it once held-out data confirms the short+long combination is actually anomalous. Sparse 2D counts need to shrink toward the two marginals, or rare real genes end up assigned near-zero probability.

### 2. Score splice sites instead of just checking GT/GC–AG

GeneID scores start, donor, and acceptor separately with log-likelihood-ratio PWMs, adds the defining-site scores to frame-specific coding log-likelihood to get an exon score, then runs dynamic programming to find the highest-scoring gene structure. [Parra, Blanco & Guigó 2000](https://genome.crg.es/courses/Lisbon04/papers/paper3.pdf)

Plant-specific results back using more than the dinucleotide, too: SplicePredictor in maize/Arabidopsis uses splice-site sequence quality plus the U and GC composition contrast on either side of the junction, and the higher-scoring group shows better specificity. [Brendel et al. 1998](https://academic.oup.com/nar/article/26/20/4748/2902399)

The most direct improvement for GeneCAD is adding a calibrated local LLR (or the logit from a small plant donor/acceptor classifier) on the edges entering the donor chain and leaving the acceptor chain. A genuine 3–8 nt initial CDS exon is barely long enough to show coding periodicity at all, so a strong start plus strong donor plus strong acceptor matters more for keeping it.

### 3. Kozak/TIS belongs in joint decoding, but shouldn't decide alone

Plant AUG context isn't a universal constant: a survey of 5,074 plant genes found purines common at −3 and +4, with the consensus differing between monocots and dicots. [Joshi et al. 1997](https://pubmed.ncbi.nlm.nih.gov/9426620/)

So `kozak_score()` could become one term on the `5'UTR -> start_a` edge, letting these two full paths compete directly:

```text
path A: upstream ATG -> 3–8 nt CDS -> long intron -> downstream CDS -> same stop
path B: upstream UTR/intergenic -> downstream ATG -> downstream CDS -> same stop
```

Compare full path scores, not `alternative_kozak - original_kozak > 3`:

```text
S(path) = neural emission + grammar transition
        + w_start  * TIS-context score
        + w_splice * (donor + acceptor scores)
        + duration priors
        + w_coding * frame-specific coding LLR
```

This also covers whole-ORF consistency: ATG, stop, no internal stop, phase across the intron, and the three-periodic coding potential of the initial segment together with the rest of the CDS all get decided jointly. AUGUSTUS itself jointly models the translation-initiation motif, initial-exon content, splice models, coding content, and duration, rather than picking the exon first and then overturning that call with a start motif on its own. [Stanke & Waack 2003](https://gobics.de/mario/papers/GenePred2003.pdf)

### 4. An additional coding periodicity/content score

GeneID trains separate order-5 Markov models for coding and intron sequence, splits the coding model by the three codon positions, and folds the frame-specific coding/noncoding LLR into the exon score. [Parra et al. 2000](https://genome.crg.es/courses/Lisbon04/papers/paper3.pdf)

GeneCAD's transformer emissions probably already learn this signal, but a short initial segment gives too few samples for it to be stable. Treat explicit coding LLR as a diagnostic feature first; only add it to the path score with a learned weight if held-out ablation shows a gain. Don't assume raw neural log-probability, Kozak log2-odds, and Markov natural-log LLR sit on the same scale.

### 5. Posterior/n-best: keep the uncertainty when the evidence is thin

Right now `_masked_viterbi()` only returns the best path. AUGUSTUS outputs posterior probabilities for exons, introns, and transcripts (an exon's posterior depends on compatible neighboring exons, not just the exon itself) and can sample alternative transcripts from them. Its own documentation warns that model posteriors can be overconfident and need recalibration. [AUGUSTUS official README](https://github.com/nextgenusfs/augustus/blob/master/README.TXT#L2783-L2886)

A cheaper first step for GeneCAD is a constrained two-best diagnostic:

- `S_short`: best path forced to include the suspect short-first-CDS boundary;
- `S_alt`: best path with that boundary forbidden (or a downstream start forced instead);
- use `Δ = S_short - S_alt` to decide accept/repair/flag, instead of looking only at the top-1 label.

Sparse forward-backward or full n-best can come later. For a 3-nt exon the intrinsic sequence genuinely can't resolve, an honest `ambiguous_short_first_exon` label preserves sensitivity better than forcing a repair.

### 6. Species/genome calibration

SNAP shows gene prediction is sensitive to species-specific parameters, and even a close relative doesn't guarantee the most compatible parameters; GeneMark-ES instead self-trains coding, noncoding, site, and duration parameters purely from anonymous genomic DNA via constrained iterative Viterbi training, and was tested on Arabidopsis among others. [Korf 2004](https://link.springer.com/article/10.1186/1471-2105-5-59), [Lomsadze et al. 2005](https://academic.oup.com/nar/article/33/20/6494/1082033)

GeneCAD's existing per-genome Kozak-margin calibration could be extended this way, but "GeneCAD's own long-first-exon predictions" aren't known-correct truth; they're pseudo-labels. The safer approach:

- iterate the estimate using predictions with high posterior/a large two-best margin, a complete ORF, and strong splice sites at both ends;
- put clade priors and shrinkage on length/site/start priors to prevent self-training drift;
- calibrate all thresholds/weights on held-out **species**, not just random genes, to avoid same-species leakage.

## Suggested design and order for GeneCAD

1. **Add observability first, without touching predictions**: for every suspect locus, output the original/alternative TIS, Kozak, donor, acceptor, first-CDS length, physical-first-exon length, intron length, coding LLR, and top-2 path gap. This settles whether the error is in the TIS or the splice path before anything else changes.
2. **Split the two lengths apart**: keep the `physical first exon = 5' UTR + CDS` splicing prior, and add a separate `first CDS segment` prior. Don't let a long 5' UTR automatically cancel every short-CDS warning.
3. **Add boundary scores**: weight edges with a plant/clade-specific donor/acceptor local LLR; run the ablation on held-out species first.
4. **Move the TIS score into the frame decoder**: keep `fix_orf` as a validation/safety net, but let the upstream-short-exon path and the downstream-start path compete in the same objective.
5. **Replace the hand-tuned power penalty with a capped semi-Markov duration**: start with just the initial-CDS empirical head; only touch the intron head/tail once that shows a benefit. Skip the 2D `short initial × long intron` feature unless it shows a stable gain across species.
6. **Add an uncertainty policy last**: auto-select only above a high `Δ`; flag/keep n-best in the middle range; keep the original prediction below a low `Δ`. Calibrate against short-first-exon precision/recall, exact-locus F1, and TIS accuracy together, stratified by first-CDS and intron-length bins.

This path is compatible with the existing `FrameStateGraph`: forced chains already show state expansion works, so it can be extended with capped duration states without throwing out the neural encoder. Modern Helixer also pairs a "sequence-only deep model + structured HMM" to produce a complete gene model, which supports this kind of hybrid architecture. [Holst et al. 2025](https://www.nature.com/articles/s41592-025-02939-1)

## Main pitfalls

- **A hard minimum exon length**: deletes real 3–8 nt exons outright; the repo's own chromosome-level results already show locus F1 dropping.
- **Penalizing every long intron**: breaks genes; AUGUSTUS's empirical-head/geometric-tail approach exists specifically to fix this.
- **Kozak alone**: weak-context real starts, leaky scanning, and monocot/dicot differences will all produce errors; Kozak should be one piece of joint evidence, not the decider.
- **Double-counting**: the transformer emission is already derived from the same DNA context, so an extra PWM/Markov score risks double-counting; weights need to come from held-out calibration.
- **Treating a canonical motif as a real splice site**: GT…AG shows up by chance often enough; it needs full junction context and competing paths.
- **Treating pseudo-labels as truth**: the "confident starts" in the existing per-genome calibration are still model predictions. Self-training needs bounded updates, shrinkage, and a stopping condition.
- **Judging only by overall F1**: this failure mode is rare, so it needs its own tracking across `<9 nt`, intron-length deciles, monocot/dicot, and UTR-present/absent strata.

## Conclusion

Another hard filter is not the highest-value next step. **Plant-specific splice boundary scores, an initial-CDS duration prior, joint Kozak/ORF path comparison, and calibrated two-best uncertainty** are. These four suppress weak-evidence false short-first-exon paths while keeping real short exons that have a strong start, strong splice sites, and consistent coding evidence. Just raising `exon_length_strictness` or penalizing long introns can't reliably hit that precision/sensitivity balance.
