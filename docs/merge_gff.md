# Detailed documentation for merge_gff.py

Step [6/7] of the prediction pipeline merges a list of single-chromosome GFF files into one GFF file.

```
python merge_gff.py \
--input-gffs genecad_chr1.gff genecad_chr2.gff genecad_chr3.gff \
--output-gff genecad_raw.gff
```

### Parameters

* `--input-gffs`, `-i` - One or more input GFF files. Required if `--manifest` is not specified.
* `--output-gff`, `-o` - Merged output GFF file.
* `--manifest` - Json file containing input files for a set of chromosomes. GFF files must be listed
under the key `filtered_gff`, or `raw_gff` if `--use-raw-gffs` flag is set.
* `--use-raw-gffs` - Flag. If set, the input gff list will be drawn from `raw_gff` in the manifest.json
instead of the default `filtered_gff`. This flag does nothing if `input-gffs` are
specified directly.

### Next Step

`python refine.py` [Refine Documentation](refine.md)
