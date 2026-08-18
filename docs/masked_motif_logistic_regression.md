# Masked Motif Logistic Regression

Masked Motif Logistic Regression (MMLR) is a technique to score gene models and identify
low-confidence gene models that may be pseudogenes or otherwise mis-annotated. It leverages the
observation that PlantCAD and PlantCAD2 detect strong signals around intron and CDS boundaries,
and are able to distinguish canonical motifs in boundaries from the same sequence in other
contexts.

This document outlines the steps required to calculate MMLR scores and offers advice for fine-tuning
scores to specific datasets.

> [NOTE!]
> `mmlr_score_junctions.py` requires a CUDA GPU

## Step 1: Prepare Junctions

MMLR scores four primary junction types: Translation Initiation Site (TIS), Translation Termination Site (TTS),
Donor Splice Site (Donor) and Acceptor Splice Site (Acceptor). The script `mmlr_prepare_junctions.py` identifies
junctions from a GFF annotation file, categorizes them, and removes redundancy between shared sites (i.e. due to
alternative splicing).

```
python mmlr_prepare_junctions.py \
--input-gff gene_annotations.gff3 \
--output-table junctions.tsv \
--num-workers 5
```

### Parameters

* `--input-gff`, `-i` - path to the input gff3 file. Required.
* `--output-table`, `-o` - path to the output table of junctions. Required.
* `--num-workers` - Number of worker threads to use when removing redundant junctions. Default is 1 (single-threaded)

### Output

The output of this script is a tab-separated table of unique junctions labeling their position and type. The table
has the following columns:
* chrom - index of the chromosome/contig, as determined by the order in the GFF file
* gene - index of the gene, sorted by start position and excluding non-protein-coding genes
* mRNA - comma-separated list of indices of the mRNA in which this junction appears
* pos - position of the first base pair in the motif, specific to strand. E.g. the "A" in a start codon's ATG, or the
"T" in the CAT codon on the negative strand.
* junction - junction type. One of: TIS, TTS, Donor, Acceptor

## Step 2: Score Junctions

This script calculates the Masked Motif scores for all junctions identified in step 1. The Masked Motif
score is the mean PlantCAD probability for the bases in the motif: the three bases comprising the start/stop codon for
TIS/TTS, and the first/last two bases of the intron for donor and acceptor splice sites. Note that this is slightly different
from the zero-shot scores described in the original PlantCAD paper (Zhai et al. 2025), as it uses the raw probabilities of
each base of the motif, rather than the difference in log-likelihood between two variants. The Masked Motif probability score
is calculated based on the observed fasta sequence and does not enforce the presence of canonical motifs: rare and/or non-
canonical codons/splice site motifs are accepted.

```
python mmlr_score_junctions.py \
--input-gff gene_annotations.gff3 \
--input-fasta assembly.fa \
--input-juctions junctions.tsv \
--model-path kuleshov-group/PlantCAD2-Medium-l48-d1024 \
--output scored_junctions.tsv
```

### Parameters

* `--input-gff`, `-i` - path to the input gff3 file. Required.
* `--input-fasta`, `-f` - path to the input fasta assembly. Required.
* `--input-junctions`, `-j` - path to the input junctions table from step 1. Required.
* `--model-path` - local or HuggingFace path to the desired PlantCAD model. Either PlantCAD or PlantCAD2 models may be used.
Default: `kuleshov-group/PlantCAD2-Medium-l48-d1024`
* `--output-table`, `-o` - path to the output table of scored gene junctions. Required.
* `--batch-size` - PlantCAD batch size. Default 16
* `--gpu` - GPU device index to use. Default 0
* `--window-size` - Size of the PlantCAD context window. Note that PlantCAD (v1) models require a window size of 512,
while PlantCAD2 models require a window size of 2048 - 8192. Window size must be even. Default 8192
* `--tag-canonical` - Flag. If set, the output table will contain an extra column detailing whether each gene model
was labeled as a canonical transcript in the original gff3 file. Accepted labeling schemes are: `tag=Ensembl_canonical` and `canonical_transcript=1`

### Output

The output is a tab-separated table contains one line for each protein-coding transcript in the GFF file. The columns are as follows:
* chrom - chromosome/contig name
* gene - gene ID
* transcript - transcript ID
* start - transcript start position
* end - transcript end position
* canonical - (optional) True if transcript is labeled canonical in the GFF file, false otherwise
* donor - comma-separated list of donor splice site Masked Motif scores
* acceptor - comma-separated list of acceptor splice site Masked Motif scores
* longest - True if this transcript is the longest for that gene (exons only)
* TIS - start codon Masked Motif score
* TTS - stop codon Masked Motif score

## Step 3: (Optional) Train MMLR Classifier

The MMLR score is an L2-regularized logistic regression model fit to predict whether a gene model is
truly protein-coding, given the Masked Motif scores of its major junctions. The GeneCAD paper outlines
a set of model weights based on well-studied classical maize genes. However, users may wish to fit
their own model weights, especially if they are working with genomes outside the scope of the original
PlantCAD and GeneCAD papers.

This script trains an MMLR Classifier using a positive-unlabeled learning method adapted
from [hkiyomaru/pu-learning](https://github.com/hkiyomaru/pu-learning). Training and validation data
are produced by the Step 2 script `mmlr_score_junctions.py`, but require one more column, labeled `validated`.
This column should contain true/false values indicated whether each transcript has experimental evidence
supporting its designation as a true protein-coding gene (e.g. proteomic data, functional studies, etc.). You may
use whatever method of validation best suits your dataset, but we recommend using a stringent cutoff. `false` values
do not necessarily indicate a pseudogene or other mis-annotation under this positive-unlabeled learning scheme: instead
it is a neutral label.

This script trains two models: one for genes with multiple exons, and one for single-exon genes. The reason for
this is that single-exon genes do not contain donor and acceptor splice sites to score. Validated example transcripts
must be labeled for both groups.

```
python mmlr_train_classifier.py \
--training-table scored_junctions_with_validation.tsv \
--output-json model_weights.json
```

### Parameters

* `--training-table`, `-t` - path to a tsv containing training data. Required columns: TIS, TTS, donor, acceptor, and validated. See above for details. Required.
* `--estimated-positive-rate`, `-p` - Estimated proportion of the full dataset that are true protein-coding genes, independent of whether they are labeled as `validated`. Default 0.75
* `--test-proportion` - proportion of training data to reserve for testing. Default 0.25
* `--validation-table`, `-v` - path to a tsv containing extra validation data, e.g. from another species. Uses the same format as the training table. Optional.
* `--output-json`, `-o` - path to the output json file containing the classifier model weights. Required.
* `--seed` - Seed to control randomization. Optional.


## Step 4: Score Transcripts

With individual Masked Motif scores and a trained classifier, we can now score individual transcripts. This
script can be used to produce an annotated table in the same format as Step 2, and it can also annotate a GFF
file with the `passPlantCADFilter` tag. A value of 1 for a transcript indicates that the transcript had an
MMLR score over 0.5. A value of 1 for a gene indicates that at least one of that gene's transcripts had an MMLR score
over 0.5

```
python mmlr_classify_transcripts.py \
--input-table scored_junctions.tsv \
--input-gff gene_annotations.gff3 \
--model-json model_weights.json \
--output-dir path/to/output
```

### Parameters

* `--input-table`, `-i` - path to a tsv containing transcripts with Masked Motif scores. Use the output from Step 2. Required.
* `--input-gff`, `-g` - path to a gff file to be annotated. Optional.
* `--model-json`, `-j` - path to a json file containing model weights. Output from step 3. Optional - default uses weights from GeneCAD paper
* `--output-dir`, `-o` - output directory. Required.
