# Detailed documentation for detect_intervals.py

This script converts base-level classifications to genomic regions,
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
* `--alpha-viterbi` - Float between 0 and 1. Increases the transition probability for all state transitions.
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
* `--domain` - Domain sets the transition probabilities between states based on empirical observation from
different types of organisms. Default: plant. Options: plant, animal.

### Next Step

`python export_gff.py` [Export GFF Documentation](export_gff.md)
