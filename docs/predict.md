# Detailed documenation for predict.py

## create_predictions

Performs multiclass classification on each base pair in the input sequence.
[NOTE!] This command requires a CUDA GPU
```
python predict.py create_predictions \
--manifest manifest.json \
--checkpoint plantcad/genecad_plant \
--model-path emarro/pcad2-200M-cnet-baseline \
--species-id species_name
```
Note: This step can be run on either a single chromosome, or a set of chromosomes.
If using single-chromosome mode, the parameters `--chromosome-id`, `--input-zarr`, and
`--output-dir` are required. If using multi-chromosome mode, the parameter
`--manifest` is required. The manifest json file contains the required input and output
information for each chromosome. An example of the formatting is shown below.

### Parameters

* `--manifest` - A JSON file containing input/output parameters for each chromosome to be processed. Required if `chromosome-id`, `input`, and `output-dir` are not specified.
* `--chromosome-id` - name of the chromosome to be processed. Required if `manifest` is not specified
* `--input-zarr`, `-i` - path to the input zarr file. Required if `manifest` is not specified
* `--output-dir`, `-o` - path to the zarr output file. Required if `manifest` is not specified
* `--model-checkpoint` - GeneCAD model checkpoint, which can be loaded from a local directory or HuggingFace. See [[Available GeneCAD models]]. Required
* `--model-path` - base PlantCAD model, which can be loaded from a local directory or HuggingFace. See [[Available GeneCAD models]]. Required
* `--species-id` - the name of the species or sample to process
* `--window-size` - context length for the model. Must be between 2048 and 8192. Default 8192.
* `--stride` - the distance between start position for each window the model sees. It is recommended to set stride to `window-size / 2` to avoid edge effects at window ends. Default 4096
* `--batch-size` - number of windows to process simultaneously. Default 16.
* `--tqdm-position` - tqdm row to use for this process when multiple GPU jobs run in one terminal. Optional
* `--show-dynamo-errors` - Flag to show torch dynamo errors, which are suppressed by default.
* `--dtype` - model inference data type. Options: float32, float16, bfloat16(default), float64, double, half


An example of the formatting for the `manifest` file
```
[{"chromosome_id": "Chr1", "input_zarr": "sequence.zarr", "output_dir":"predictions_Chr1/"},
{"chromosome_id": "Chr2", "input_zarr": "sequence.zarr", "output_dir":"predictions_Chr2/"},
{"chromosome_id": "Chr3", "input_zarr": "sequence.zarr", "output_dir":"predictions_Chr3/"}]

```

## warmup

TODO

## export_gff

TODO - move to new file
