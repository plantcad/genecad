# Short first CDS errors: training versus decoding

Analysis date: 2026-09-15. Inspected checkout: `506848e1be1333263d2925bd516fffa2ddc361e1`.

**Empirical update:** [The subsequent reanalysis](short_first_cds_reanalysis.md) reproduces the negative-strand experimental failure through incorrect cache-coordinate indexing, replays all 11 Arabidopsis loci correctly, and measures existing boundary scores at 213 donor pairs. Read it before interpreting the initial hypotheses below.

## Recommendation

Plan to retrain when an updated base model is available. Before that run, make
the baseline reproducible and use the existing error set to check labels, data
splits, context, and evaluation. The tested repairs should not be deployed.
The 17-class boundary scores already produced by the model can help show which
examples need more attention during training.

This report combines source inspection with existing local experimental artifacts. It does not report new inference, retraining, biological validation, or a fresh benchmark. The current checkout and historical experimental code are different revisions. The live PR page could not be retrieved; the reviewer interpretation relies on the supplied comment.

## 1. What the pipeline actually learns and uses

```text
DNA encoder → optional projection/token embedding → ModernBERT head → classifier
            → 17 BILUO logits
            → logsumexp aggregation to 5 feature logits
            → optional intergenic bias + softmax
            → frame-aware Viterbi using feature transitions and sequence masks
            → export/filter/merge → fix_orf
```

- **Training already includes boundary labels.** Four features have B/I/L/U tags (beginning, interior, last, unit-length), plus intergenic: 17 classes. These do not separately encode initial versus internal CDS segments. Architecture components are configurable. Sources: [model configuration](../../src/modeling.py), lines 136–145 and 350–418.
- **The training objective is token cross entropy**, not the score of a complete gene path. `_compute_loss` flattens tokens and applies the training mask. `--auto-class-weights` initializes/resets the learned bias using frequencies; it does not make the CE loss class-weighted. Sources: [modeling](../../src/modeling.py), lines 420–436 and 665–671; [trainer](../../scripts/train.py), lines 306–329 and 393–414. PyTorch documents the default unweighted mean reduction in [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
- **Boundary identity is collapsed before production decoding.** `aggregate_logits` sums probability mass within each feature via logsumexp. This preserves total feature evidence but not the relative B/I/L/U probabilities. The frame decoder consumes only the five feature probabilities. Sources: [aggregation](../../src/modeling.py), lines 713–729; [decoder call](../../scripts/detect_intervals.py), lines 87–120.
- **Both representations are saved.** Prediction output includes `token_logits` and `feature_logits`; therefore existing caches may support boundary diagnostics without GPU inference. Presence and provenance must be checked per genome; this audit confirmed a cached tag array in the local test-region output, not coverage for every cohort. Source: [prediction](../../scripts/predict.py), lines 379–431.
- **Transitions are supplied separately from token training.** `token_transition_probs` selects plant/animal constant matrices; the frame graph expands these into constrained states. They are not jointly optimized by the inspected CE loss. Sources: [transition selection](../../src/modeling.py), line 888 onward; [frame graph](../../src/frame_crf.py).

### Why the 17-to-5 conversion matters

As an illustration, a true donor and false donor can have the same total intron probability, yet very different probabilities of being the **beginning** of an intron. Their intron emission score is then identical after aggregation. Neighboring feature scores can still distinguish them: aggregation does not prove a failure, but it can hide a useful distinction.

Inspect the existing boundary scores at true and false junctions first. If they already distinguish the two, reusing that signal deserves a controlled experiment before a new splice classifier or expensive base-model fine-tuning. Tag scores need orientation- and coordinate-correct mapping; a CDS beginning can also be an internal exon, so it is not automatically a TIS score. Do not simply add raw tag logits to feature logits, which would count related evidence twice.

## 2. What the existing results establish

### Tested repairs have limited or harmful effects

The [committed validation note](short_first_exon_validation_results.md) reports:

| Experiment | Reported result | Supported interpretation |
| --- | --- | --- |
| Calibrated Kozak, two confirmatory species | 4 wrong→correct; 1 correct→wrong CDS chains | Small benefit in that cohort |
| Calibrated Kozak, tested wheat chromosomes | 0 corrections; 2 correct→wrong | Benefit is not consistent across cohorts |
| Restricted prefix enumeration, top 1/10/50 boundaries | 17/24/40 of 257 errors contain the exact reference candidate | Candidate recall limits that particular reranker |
| Conservative micro-repair | 4 chain corrections; 0 harms; about 1.6% error recall | Too little evidence for general deployment |

The top-50 figure is about 15.6%. It is **not** an upper bound on the original full Viterbi graph, and it does not prove that the neural representation lacks the information. Window limits, frozen suffixes, anchors, and candidate construction may exclude the answer. The top-k counts above come from the committed summary; their generating artifacts were not independently reproduced in this audit.

### Important correction: 3 repairs versus 67 harms is confounded

The existing cross-species local re-decoding experiment has these aggregate, all-reference rows:

| Local decoding variant | Wrong→correct versus raw GFF | Correct→wrong versus raw GFF |
| --- | ---: | ---: |
| `run0`, no short coding-run penalty | 0 | 67 |
| `run9_s16_utr` | 1 | 67 |
| `run9_s16_cds` | 3 | 67 |

Against the **local run0** baseline, the CDS-only penalty gives 3 chain corrections and 0 chain regressions; TIS changes are 3 corrections and 2 regressions. Thus the 67 harms cannot be assigned to the added short-run penalty. They already arise in the local reconstruction baseline.

Sources: local [metrics](../../genecad_result/experiments/short_first_exon_validation/cross_species_decoder_redecode/metrics.tsv), [incremental metrics](../../genecad_result/experiments/short_first_exon_validation/cross_species_decoder_redecode/incremental_metrics.tsv). These ignored artifacts exist in this workspace but are not part of the tracked report distribution.

This still argues against deploying that local repair procedure. It does **not** establish that decoder changes generally fail. First explain why local re-decoding changes the original result: context/initialization, selected transcript, graph settings, and export/filter/reconstruction differences are candidate explanations, not yet proven causes.

### Existing decoder results also show a metric tradeoff

The historical [frame comparison](../../genecad_result/frame_aware_eval/accuracy.tsv) reports Arabidopsis chr4 locus F1 increasing from 0.8128 to 0.8545, while intron F1 changes from 0.9232 to 0.9221. Other plant rows likewise show some whole-CDS gains with splice-level losses. These are historical pipeline comparisons, not a fresh test of today's defaults, but they show why “better” must specify both complete structure and junction accuracy.

Structural validity alone is insufficient: a graph can produce a legal ORF at the wrong biological locus. A hybrid neural/structured architecture is reasonable in principle—[Helixer's primary paper](https://www.nature.com/articles/s41592-025-02939-1) describes such a combination—but that is not evidence for any particular extra GeneCAD scoring term.

## 3. Training is a plausible target, with prerequisites

### Objective mismatch is plausible, not yet measured

Unweighted token CE gives each eligible token a contribution; long regions contribute many tokens, while a rare short CDS contributes few. A lower average loss therefore need not improve exact start or first-junction accuracy. However, BILUO already supervises boundaries: do not claim that boundary supervision is absent. Measure initial-CDS-specific class counts, confusion, and margins before selecting weighting or sampling changes.

### Evaluation must distinguish improvement from easier validation

1. **Overlapping windows can cross the train/validation split.** Sampling uses half-window stride, then the split randomly assigns windows within a contig if that species participates in both sets. This permits shared sequence. Actual overlap and historical checkpoint contamination were not measured. Split genomic blocks before window creation, ensure neither strand/context crosses the split, and audit homology when evaluating cross-species generalization. Sources: [sampling](../../src/sampling.py), lines 201–239; [split](../../scripts/sample.py), lines 616–642.
2. **Training and validation masks differ.** Training uses `label_mask`; validation replaces it with all ones. Report trusted-label and excluded/ambiguous-label results separately before selecting a checkpoint. Source: [batch preparation](../../src/modeling.py), lines 810–815.
3. **Current checkpoint selection is not the production objective.** Entity validation aggregates tags and takes local argmax; the wrapper selects by `valid__entity__overall/f1`. Add fixed genomic-set evaluation after the actual decoder and downstream pipeline. Sources: [entity evaluation](../../src/modeling.py), lines 744–746; [training wrapper](../../train.sh), lines 1091–1096.
4. **Reference isoforms require an explicit policy.** Training extraction selects the longest transcript as canonical and transformation retains canonical transcripts. A different supported isoform must not automatically be labeled a biological error. Sources: [extraction](../../scripts/extract_train.py), lines 216–217; [transformation](../../scripts/transform.py), lines 124–128.

These are implementation observations, not proof that any particular released checkpoint was trained with a problematic split or label set. Freeze the checkpoint config and data manifest before causal attribution.

### “Better training” has several scopes

Head-only adaptation, supervised fine-tuning of the DNA encoder, changing labels/sampling/loss, and foundation-model pretraining are different interventions. Current `train.sh` presets unfreeze the base encoder, whereas direct `scripts/train.py` defaults differ. The first training experiment should adapt an existing checkpoint to a diagnosed weakness, not start foundation-model pretraining from scratch. Sources: [wrapper](../../train.sh), lines 142–162; [trainer](../../scripts/train.py), lines 212–216; [optimizer](../../src/modeling.py), lines 537–553.

The default context is 8192 positions. If evidence lies outside the same model window, simply adding epochs does not make it jointly visible. Check actual checkpoint context and position relative to inference windows. Also, a larger shell `--window-size` currently reaches sampling but not the wrapper's training invocation; the trainer defaults to 8192 and the dataset can truncate larger samples. A context experiment must verify all stages. Sources: [wrapper](../../train.sh), lines 1014 and 1055–1082; [trainer](../../scripts/train.py), lines 85–88 and 430–439; [dataset](../../src/dataset.py), lines 241–245.

## 4. The diagnostic experiment to run first

### A. Freeze provenance and reproduce the baseline

Record source revision, checkpoint/hash/config, genome and reference versions, logits-cache provenance, context/window settings, strand, bias, alpha, graph parameters, and downstream filter/ORF options. The six-species manifest records hashes and older git revisions; its listed raw GFF, FASTA, and reference paths were present at inspection time. Presence was checked; file hashes were not recomputed.

Reproduce an unchanged result from the same cache and complete settings. If using cropped loci, carry compatible boundary conditions and sufficient flanks, and first demonstrate agreement with the original whole-contig output. Otherwise a local decoder comparison is testing context/reconstruction changes as well as the intended intervention. Preserve outputs immediately after decoding and after each downstream step to localize where a structure changes.

### B. Build a fixed reference-based cohort

Reuse prior loci as exploratory data; they have already informed many proposals. Reserve new untouched loci/species for final confirmation. Include:

- Wrong TIS with correct physical splice structure.
- Wrong first exon/donor/acceptor or gene boundary.
- Genuine short initial CDS segments, with and without a long UTR.
- Ordinary genes and genuine long introns as regression controls.
- Uncertain/alternative-isoform cases reported separately.

Use separate fields for first physical transcript exon, physical exon containing the first CDS, and first CDS segment. They differ when there are upstream UTR-only exons. Keep cohorts fixed from baseline/reference; do not reselect only the short exons surviving each intervention. Prespecify first-CDS bins (for example 3–8, 9–30, 31–60, >60 nt), species/clade, UTR availability, intron length, and window-edge status. Compute intron percentiles from a fixed development/reference distribution.

### C. Test reference-path feasibility before scoring

Determine whether the supported reference structure has any legal state path under the actual graph. Record specific exclusions: motif family, intron minimum, split start/stop codon, required UTR structure, sequence ambiguity, or imposed crop/anchor/suffix restrictions. The graph's documented limitations include start/stop codons that cannot be interrupted by introns and required UTR states ([frame decoder](../../src/frame_crf.py), lines 64–81).

If a correct path is excluded, retraining its emissions cannot make that path reachable. If a restricted candidate generator misses it while the full graph allows it, improve/evaluate candidate construction separately from model training.

### D. Compare complete, compatible path scores

For legal paths over the same genomic interval and compatible outer states:

```text
S(path) = initial-state score
        + sum of log feature emissions
        + sum of expanded-graph log transition weights
Delta = S(best supported reference-compatible path) - S(predicted path)
```

Use the actual graph implementation, including forced edges and penalties; do not substitute a generic length formula. Reference CDS alone may allow multiple UTR/state paths: constrain the supported annotation and maximize over unspecified compatible states. Do not invent a single UTR extension and interpret its score as the reference truth.

The current kernel performs Viterbi maximization over allowed states and returns a path ([kernel](../../src/frame_crf.py), lines 1054–1122). `return_states=True` is available, but score decomposition and constrained-reference decoding need diagnostic implementation. Under identical conditions its best path should not have a materially lower score than a feasible reference path. A positive Delta warrants checking scoring, constraints, context, or output reconstruction before attributing the difference to training.

Report emission and transition contributions separately, alongside 17-tag boundary margins. If emissions favor the reference but transition contributions overturn it, prioritize calibration/decoder experiments. If emissions favor the error, inspect labels, context, and fine-tuning. This decomposition identifies intervention candidates; it does not prove a unique cause, since training can compensate for priors and calibration can compensate for emissions.

### E. Audit already available boundary evidence

Compare strand-correct intron B/L scores and feature margins around supported versus false sites, controlling for species, intron length, and coding-run length. If cached tag arrays are missing, re-infer only selected loci with verified context. Keep this diagnostic separate from using annotations at production inference: references are for evaluation, not extra runtime evidence.

## 5. Minimal comparison after diagnosis

| Arm | Model | Decoder | Purpose |
| --- | --- | --- | --- |
| A | Existing checkpoint | Existing settings | Reproducible baseline |
| B | Targeted fine-tuning | Same settings as A | Effect of training |
| C | Same checkpoint as A | One diagnosed change | Effect of decoding/calibration |
| D | Same fine-tuned checkpoint as B | Same change as C | Interaction |

Choose B from observed failure evidence: trusted hard-example sampling with real-short controls, boundary-aware weighting, or better labels/context. Choose C from evidence: calibration, use of existing boundary information, or a demonstrated representational restriction. Do not bundle new PWM, coding LLR, duration model, and n-best policy in one first experiment. An unchanged-recipe fine-tuning control is useful if attributing gains to a new training recipe rather than simply additional updates. Repeat promising training arms across seeds when gains are small.

Use an equal declared tuning budget, development-only tuning, and untouched final evaluation. First hold the decoder fixed to isolate training effects; then separately compare fully calibrated pipelines so a changed logit scale does not unfairly penalize a new model.

### Decision measures

- Paired locus counts: wrong→correct, correct→wrong, wrong→different-wrong, unchanged; include gene gains/losses and merge/split errors.
- Error-recovery rate: corrected baseline errors / baseline errors in the fixed cohort.
- Damage rate: newly incorrect baseline-correct loci / baseline-correct loci.
- Exact TIS, first donor/acceptor, whole CDS-chain accuracy, and genuine short-CDS retention.
- Whole-chromosome complete-structure and splice precision/recall, including ordinary genes.
- Per-species results, macro summaries, and uncertainty from resampling independent loci/blocks, not bases; limited species counts constrain cross-species claims.
- Training cost, inference cost, and maintenance burden.

An error detector's AUPRC is not a repair success rate. Zero observed harms among a handful of actions is not evidence of negligible harm probability. Set acceptable damage and whole-genome regression limits on development data before seeing the final test. Prefer the simplest intervention that meets those limits with a useful correction rate. Existing [evaluation code](../../scripts/evaluate.py) covers several CDS/splice/site metrics; fixed-cohort transition accounting and reference-policy checks need additional analysis.

## 6. Decision

Use this work to prepare for retraining with the next base-model update. Keep
the production decoder and post-processing defaults unchanged for now. The
score checks above are useful for choosing training examples and finding label
or context problems; they are not a proposal to add more scoring terms to the
decoder.

After retraining, compare the new model against the current pipeline on fixed
held-out species. Only revisit the decoder if the comparison shows that a
correct reference path is blocked by the graph or consistently loses despite
stronger model evidence.
