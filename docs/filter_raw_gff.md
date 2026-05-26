# Detailed documentation for filter_raw_gff.py

This script applies a basic set of filters to raw GeneCAD output, removing
gene models and features that are abnormally short or that lack key components, such
as coding sequences or UTRs.

This script is intended for use within the GeneCAD pipeline and may not function as intended
with other GFF files.

```
python filter_raw_gff.py \
--input-gff genecad_raw.gff \
--output-gff genecad_filtered.gff
```

### Parameters

* `--input-gff`, `-i` - Input GFF file (required). Output from `export_gff.py`.
* `--output-gff`, `-o` - Output GFF file (required).
* `--min-feature-length`- Minimum length for features such as CDS or UTR. Default 2.
* `--feature-types` - A comma-separated list of features to filter for minimum length.
Features must be one of: gene, mRNA, three_prime_UTR, five_prime_UTR, CDS. Default is three_prime_UTR,five_prime_UTR,CDS
* `--min-gene-length` - Minimum length for genes. Includes length of introns. Default 30.
* `--require-utrs` - If set, this flag removes all transcripts that are missing the 5' or 3' UTR.
This parameter is ignored if `--keep-incomplete-models` is True.
* `--keep-incomplete-models` - If set, this flag retains genes with no mRNA transcript and mRNA
transcripts with no CDS. The default behavior is to remove such models.


### A Note on Order of Operations

The above filters are applied in the following fixed order:

1. Features shorter than the minimum feature length are removed
2. Genes shorter than the minimum gene length are removed
3. If `--keep-incomplete-models`, stop here. Otherwise continue.
4. Transcripts missing required components are removed: CDS, plus UTRs if `--require-utrs`
5. Genes missing transcripts are removed.

The removal of sub-features such as CDS and UTRs may impact the total gene length
or cause a transcript to be missing required components during later filtering steps.
This is intentional, as the order above ensures that all filter parameters are satisfied.
