# Detailed documenation for predict.py

Step [2/8] of the prediction pipeline performs multiclass classification on each base pair in the input sequence.

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
* `--batch-size-cache` - optional file for the successful batch cap. A cached value
can lower the requested cap, but never raise it. The shell sets a separate path per GPU.
* `--tqdm-position` - tqdm row to use for this process when multiple GPU jobs run in one terminal. Optional
* `--show-dynamo-errors` - Flag to show torch dynamo errors, which are suppressed by default.
* `--dtype` - model inference data type. Options: float32, float16, bfloat16(default), float64, double, half
* `--warmup-batch-size` - Size of the warmup batches. Only used if `--triton-warmup is set`
* `--triton-warmup` - Pre-warms the Triton autotune cache with dummy data. If set, input data is ignored and
output will not be written to file.



### Processing many scaffolds

`predict.sh` groups unfinished scaffolds into manifests under `<output-dir>/.state/`.
Each active GPU runs one Python worker; DDP uses one distributed launch for its
manifest. Each worker loads and fingerprints the model once, then processes its
scaffolds sequentially. Sequence data and intermediate predictions are released
between scaffolds. The model stays on the GPU. CPU processing and output merging
start after all prediction workers succeed.

On CUDA OOM, the worker reduces the batch cap by 20% and retries the uncommitted
windows. It releases temporary GPU tensors before retrying and keeps the lower cap
for subsequent strands and scaffolds. An OOM at one window, a model-load failure,
or another error stops the worker. Restarting a killed process requires loading
the model again; completed segments are retained.

Logs report model loading, fingerprinting, prediction/write time, and total time
per scaffold, including validation. GPU timings and peak memory must be measured
on the intended hardware and input.

### Resuming interrupted predictions

Run the same command with the same output directory to resume. Each batch writes
an independent `segment.*.zarr`, then an atomic JSON receipt recording its strand,
window range, genomic coordinates, and SHA-256 file hashes. Restart removes
incomplete or corrupt segments and recomputes their windows.

Window IDs are independent of batch size and GPU count, so either can change on
restart. Window context and strand orientation are preserved. GPU floating-point
results may differ slightly with different batching or hardware.

The run identity includes the input store's content, loaded model weights and
configuration, tokenizer vocabulary, species/chromosome, window size, stride,
dtype, and PyTorch version. A mismatch stops the run without deleting its outputs;
use a new output directory for changed inputs or models. A rank-zero filesystem
lock limits each chromosome output directory to one writer job.

After all ranks finish, the worker checks file hashes and complete coverage of
both strands before writing `_SUCCESS.json`. Downstream readers require this
manifest and verify the segments. Legacy `predictions.<rank>.zarr` stores remain
readable by interval detection but cannot be resumed without progress receipts.
Use a new output directory to regenerate legacy predictions. Old chromosome-wide
`.tmp` directories are not imported.

File hashing uses 4 MiB blocks. Windows are generated in batches, but the current
sequence and padded strand remain in RAM, so long chromosomes can still require
substantial memory. Checksums and segment files add I/O. Other pipeline stages
retain their atomic output writes and stage-level resume checks. The shell skips
scaffolds whose final filtered GFF already exists.

### Next Step

`python detect_intervals.py` [Detect Intervals Documentation](detect_intervals.md)
