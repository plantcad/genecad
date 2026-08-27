# Detailed documentation for detect_intervals.py

Step [3/8] of the prediction pipeline converts base-level classifications to genomic regions,
such as CDS and UTR exons using the viterbi algorithm and domain-specific
 transition probabilities.

```
python detect_intervals.py \
--input-dir predictions_Chr1/
--output-zarr intervals_Chr1.zarr
```

### Parameters

* `--input-dir`, `-i` - The base-level predictions, output from `predict.py`
* `--output-zarr`, `-o` - Output zarr file containing interval/region data.
* `--manifest` - Json file containing input and output paths for a set of chromosomes/samples.
This can be used in place of specifying `--input-dir` and `--output-zarr`. Required key for each
sample are: chromosome_id, predictions_dir, and intervals_zarr.
* `--viterbi-alpha` - Float between 0 and 1. Increases the transition probability for all state transitions.
Higher transition probabilities are more sensitive to detecting genes, but also increase the likelihood of
finding pseudogenes and/or creating ill-formed gene models. Default None.
* `--decode-direct` - Flag. By default, this script uses the viterbi algorithm and a
preset transition matrix to adhere to valid gene structure (e.g. introns must be bounded by exons, 3'
UTR must follow CDS, etc.). The direct method creates intervals directly from the base-level predictions,
 with no enforcement of gene model structure.
* `--intergenic-bias` - Float. Penalizes predictions of intergenic bases, increasing the model's sensitivity towards
predicting genic regions (CDS, 3' UTR, 5' UTR, or intron). Note that this value is applied before
softmax, and so can be greater than 1.
* `--keep-incomplete-features` - Flag. If set, gene models missing UTRs are allowed.
* `--input-fasta`, `-f` - Genome FASTA used for prediction. When given, decoding becomes frame-aware: the CDS is
constrained to begin on ATG, end on a stop codon, stay in frame across introns, and contain no in-frame stop.
Ignored when `--decode-direct` is set.
* `--min-intron-length` - Shortest intron frame-aware decoding may emit. Guards against short introns being invented
to step over an in-frame stop codon. Default: 20.
* `--min-coding-run-length` - Runs of coding sequence adjacent to an intron shorter than this are penalized, not
forbidden (see `--exon-length-strictness`). Guards against an intron being invented immediately after the start codon
or immediately after another intron, without also destroying genuine short exons. Use 0 to disable the penalty
entirely. Default: 9.
* `--exon-length-strictness` - How strongly to penalize a run of coding sequence below `--min-coding-run-length`. 0 removes the
penalty; larger values fall off more steeply and converge on treating `--min-coding-run-length` as a hard floor. Strong
per-base emission evidence can still outweigh the penalty at any setting above 0, which is what lets genuine short
boundary exons survive. Default: 16.
* `--include-utr-in-coding-run` - Flag. Counts the 5' UTR alongside the coding run for
`--min-coding-run-length` purposes, so a long UTR can by itself exempt a short first coding run from the penalty.
Only the start side is covered. On by default; pass `--no-include-utr-in-coding-run` to disable.
* `--domain` - Domain sets the transition probabilities between states based on empirical observation from
different types of organisms. Default: plant. Options: plant, animal.
* `--allow-u12-introns` - Flag. Also allow U12-type AT-AC introns during frame-aware decoding, in addition to the
default GT-AG/GC-AG. These are real but rare; enabling this roughly doubles the intron state count and slows
decoding accordingly. Ignored unless `--input-fasta` is set.

### Next Step

`python export_gff.py` [Export GFF Documentation](export_gff.md)
