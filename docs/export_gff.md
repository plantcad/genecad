# Detailed documentation for export_gff.py

This script converts predicted intervals into a GFF3 formatted file.

```
python export_gff.py \
--input-zarr intervals_Chr1.zarr \
--output-gff Chr1_raw.gff
```

### Parameters

* `--input-zarr`, `-i` - Input intervals zarr file. This is the output from
`detect_intervals.py`.
* `--output-gff`, `-o` - Output gff file name.
* `--min-transcript-length` - Remove any transcripts that are shorter than this (including introns).
Default 0
* `--tqdm-position` - Index of the tqdm row to use for export when multiple jobs are run in one terminal.
Optional, default: None
* `--cpu-workers` - Number of CPU worker processes for transcript grouping. Default: 1
