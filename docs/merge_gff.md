# Detailed documentation for merge_gff.py

Step [6/7] of the prediction pipeline merges a list of single-chromosome GFF files into one GFF file.

```
python merge_gff.py \
--input-gff genecad_raw.gff \
--output-gff genecad_filtered.gff
```

### Parameters

* `--input-gff`, `-i` - Input GFF file. Output from `export_gff.py`. Required if `--manifest` is not specified.
* `--output-gff`, `-o` - Output GFF file. Required if `--manifest` is not specified.
* `--manifest` - Json file containing input and output paths for a set of chromosomes/samples.
This can be used in place of specifying `--input-gff` and `--output-gff`. Required key for each
sample are: chromosome_id, raw_gff, and filtered_gff.
* `--min-feature-length`- Minimum length for features such as CDS or UTR. Default 2.
* `--feature-types` - A comma-separated list of features to filter for minimum length.
Features must be one of: gene, mRNA, three_prime_UTR, five_prime_UTR, CDS. Default is three_prime_UTR,five_prime_UTR,CDS
* `--min-gene-length` - Minimum length for genes. Includes length of introns. Default 30.
* `--require-utrs` - If set, this flag removes all transcripts that are missing the 5' or 3' UTR.
This parameter is ignored if `--keep-incomplete-models` is True.
* `--keep-incomplete-models` - If set, this flag retains genes with no mRNA transcript and mRNA
transcripts with no CDS. The default behavior is to remove such models.

### Next Step

`python refine.py` [Refine Documentation](refine.md)
