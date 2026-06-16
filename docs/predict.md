# Detailed documenation for predict.py

Step [2/7] of the prediction pipeline performs multiclass classification on each base pair in the input sequence.

> [NOTE!]
> This script requires a CUDA GPU

```
python predict.py \
--manifest manifest.json \
--model-checkpoint plantcad/genecad_plant \
--model-path emarro/pcad2-200M-cnet-baseline \
--species-id species_name
```
> [!NOTE]
> This step can be run on either a single chromosome, or a set of chromosomes.
> If using single-chromosome mode, the parameters `--chromosome-id`, `--input-zarr`, and
> `--output-dir` are required. If using multi-chromosome mode, the parameter
> `--manifest` is required. The manifest json file contains the required input and output
> information for each chromosome, and can be generated using `create_manifest.py` ([Documentation](create_manifest.md))

### Parameters

* `--manifest` - A JSON file containing input/output parameters for each chromosome to be processed. Required if `chromosome-id`, `input`, and `output-dir` are not specified.
* `--chromosome-id` - name of the chromosome to be processed. Required if `manifest` is not specified
* `--input-zarr`, `-i` - path to the input zarr file. Required if `manifest` is not specified
* `--output-dir`, `-o` - path to the zarr output file. Required if `manifest` is not specified
* `--model-checkpoint` - GeneCAD head model checkpoint, which can be loaded from a local directory or HuggingFace. See [Available GeneCAD models](../README.md#available-models). Required
* `--model-path` - base PlantCAD model, which can be loaded from a local directory or HuggingFace. If not
specified, the script will attempt to infer the base model from the model checkpoint. See [Available GeneCAD models](../README.md#available-models) for model pairs.
* `--species-id` - the name of the species or sample to process
* `--window-size` - context length for the model. Must be between 2048 and 8192. Default 8192.
* `--stride` - the distance between start position for each window the model sees. It is recommended to set stride to `window-size / 2` to avoid edge effects at window ends. Default 4096
* `--batch-size` - number of windows to process simultaneously. Default 16.
* `--tqdm-position` - tqdm row to use for this process when multiple GPU jobs run in one terminal. Optional
* `--show-dynamo-errors` - Flag to show torch dynamo errors, which are suppressed by default.
* `--dtype` - model inference data type. Options: float32, float16, bfloat16(default), float64, double, half
* `--warmup-batch-size` - Size of the warmup batches. Only used if `--triton-warmup is set`
* `--triton-warmup` - Pre-warms the Triton autotune cache with dummy data. If set, input data is ignored and
output will not be written to file.



### Next Step

`python detect_intervals.py` [Detect Intervals Documentation](detect_intervals.md)
