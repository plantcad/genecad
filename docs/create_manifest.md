# Detailed documentation for create_manifest.py

Optional step between [1/7] and [2/7] of the prediction pipeline creates a keyfile that
coordinates processing multiple contigs at the same time.

```
python create_manifest.py \
--input-zarr path/to/output/sequence.zarr \
--output-json path/to/output/manifest.json \
--species-id species_name \
--intermediate-dir path/to/output
```

### Parameters

* `--input-zarr` `-i-` - sequences.zarr file from `extract_fasta.py`
* `--output-json` `-o` - output manifest.json file
* `--species-id` - The name of the species or sample
* `--intermediate-dir`- Path to a directory which will store intermediate GeneCAD files.
By default, this is the parent directory of `input-zarr`


### Manifest Format
GeneCAD manifest files use human-readable JSON format to list the parameters
specific to each chromosome while using the prediction pipeline. The following
parameters are required:
* `chromosome_id` - name of the chromosome
* `sequence_zarr` - sequences.zarr file from `extract_fasta.py`
* `predictions_dir` - predictions directory from `predict.py`
* `intervals_zarr` - intervals.zarr file from `detect_intervals.py`
* `raw_gff` - unfiltered gff file from `export_gff.py`
* `filtered_gff` - post-filter gff file from `filter_raw_gff.py`

An example of the formatting for the `manifest` file
```
[{"chromosome_id": "Chr1", "sequence_zarr": "sequence.zarr", "predictions_dir": "predictions_Chr1/", "intervals_zarr": "intervals_Chr1.zarr", "raw_gff": "predictions_raw_Chr1.gff", "filtered_gff": "predictions_filtered_Chr1.gff"},
{"chromosome_id": "Chr2", "sequence_zarr": "sequence.zarr", "predictions_dir":"predictions_Chr2/", "intervals_zarr": "intervals_Chr2.zarr", "raw_gff": "predictions_raw_Chr2.gff", "filtered_gff": "predictions_filtered_Chr2.gff"},
{"chromosome_id": "Chr3", "sequence_zarr": "sequence.zarr", "predictions_dir":"predictions_Chr3/", "intervals_zarr": "intervals_Chr3.zarr", "raw_gff": "predictions_raw_Chr3.gff", "filtered_gff": "predictions_filtered_Chr3.gff"}]

```

### Next Step

`python predict.py` [Predict Documentation](predict.md)
