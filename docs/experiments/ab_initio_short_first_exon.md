# Short first-CDS exon errors

This note records what we learned from the short first-CDS exon work and what
to revisit when a new base model is available. It is an experiment note, not a
description of current GeneCAD behavior.

## Problem

The errors fall into two broad groups:

1. The splice structure is right, but the predicted translation start is
   wrong. Sequence before the real start should be 5' UTR.
2. The first exon or first intron is wrong. Moving the start codon within the
   same transcript cannot fix this case.

The current decoder keeps gene models structurally valid, but it cannot tell us
which part of the model caused a bad prediction. In particular:

- splice boundaries must use a supported dinucleotide, but that check does not
  score the full donor or acceptor context;
- the intron state has a simple length behavior, and the short coding-run
  penalty only covers runs shorter than 9 nt;
- `include_utr_in_coding_run=True` counts the 5' UTR and the first CDS segment
  together;
- Kozak scoring is used later by `fix_orf.py`, after the gene path has already
  been chosen.

These points are useful for diagnosis, but adding a separate rule for each one
would make the pipeline harder to train, test, and maintain.

## Decision

Do not add another short-exon filter or expand the Kozak repair. The validation
results did not show a safe general repair, and several tested changes fixed
few loci or introduced new errors.

The next main attempt should be retraining when an updated base model is
available. The model should learn start, splice, coding, and surrounding
sequence evidence together instead of relying on a growing set of fixes after
prediction.

The decoder should continue to enforce basic structural rules such as valid
start and stop codons, reading frame, and supported splice motifs. A decoder
change should only be considered if a reference path is impossible under the
current graph. Retraining cannot recover a path that the graph does not allow.

## Data for the next training run

Build a fixed set that contains both errors and controls:

- wrong translation starts with otherwise correct splice structure;
- wrong first donors, acceptors, or exon boundaries;
- real short first CDS segments, including genes with a long 5' UTR;
- real long introns;
- ordinary genes from each evaluation species;
- uncertain or alternative-isoform cases, reported separately.

Keep physical first-exon length and first-CDS-segment length as separate
fields. Split data by genomic region or species before creating overlapping
windows so nearby sequence cannot appear in both training and validation.

Before changing sampling or loss weights, measure how often these cases occur
and how the current model scores their starts and splice boundaries. Use the
results to decide whether the next run needs more hard examples, different
labels, more context, or a change to the training objective.

## Evaluation

Compare the new model with the current production pipeline using the same
decoder settings. Report at least:

- exact translation-start accuracy;
- first donor and acceptor accuracy;
- complete CDS-chain accuracy;
- retained real short first CDS segments;
- newly broken predictions that were correct before retraining;
- whole-genome CDS and splice precision and recall for each species.

Use development species to choose the training setup and keep separate species
for the final check. The short-first-CDS loci used in the earlier experiments
are now development data and should not be the final test set.

## Conclusion

The short-first-CDS problem should be addressed through better training with a
future base-model update. The existing experiments remain useful for building
the training set and evaluation, but they do not support another production
filter or post-processing rule.
