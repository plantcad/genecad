# Troubleshooting: Common Problems

Please see our [Issues](https://github.com/plantcad/genecad/issues) page for other user
issues and solutions.

## Table of Contents
* [Installation Errors](#installation-errors)
* [Import Errors](#import-errors)
* [HuggingFace/Network Errors](#hugging-face--network-errors)
* [Runtime Errors](#runtime-errors)
* [SkyPilot Errors](#skypilot-errors)
* [Other Problems](#other-problems)

## Installation Errors

### `mamba-ssm` or `causal-conv1d` build fails

These packages compile CUDA kernels from source and are sensitive to the CUDA/GCC/PyTorch combination.

```
RuntimeError: Error building extension 'causal_conv1d_cuda'
```

- Confirm your CUDA version: `nvcc --version` and `nvidia-smi`. GeneCAD requires CUDA ≥ 12.4 (12.8 recommended).
- Confirm GCC is available: `gcc --version`. GCC 11 or 12 is the most reliable with CUDA 12.x.
- The fastest fix is to switch to Docker, which ships with all compiled extensions pre-built:
  ```bash
  docker pull ghcr.io/plantcad/genecad_v1:latest
  docker run --rm --gpus all -v $(pwd):/workspace -w /workspace \
    ghcr.io/plantcad/genecad_v1:latest bash predict.sh -i genome.fa -s MySpecies
  ```

### `flash-attn` build fails

Same root cause as above. Docker is the safest path if you cannot match the exact CUDA/GCC versions.

### `genecad: command not found`

The most common cause is forgetting to activate the virtual environment after opening a new terminal:

```bash
cd genecad
source .venv/bin/activate
genecad predict ...
```

To avoid doing this every session, add the activation to your shell profile:

```bash
echo 'source /path/to/genecad/.venv/bin/activate' >> ~/.bashrc
source ~/.bashrc
```

If you installed via the release wheel and activation is not the issue, the `genecad` executable may be in `~/.local/bin`, which may not be on your `PATH`:

> [!NOTE] All `genecad` subcommands can alternatively be invoked with their corresponding bash
> or python scripts (see main [README](../README.md) for details).

```bash
export PATH="$HOME/.local/bin:$PATH"
# Make it permanent:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Python version issues

GeneCAD requires Python 3.12 exactly. Check with `python --version`. If you are using `uv`, the correct version is managed automatically via `uv venv`.

## Import Errors

### `ModuleNotFoundError: No module named 'mamba_ssm'`

```bash
bash scripts/install_release.sh   # re-runs the mamba install step
```

If that fails, switch to Docker (see above).

### `ModuleNotFoundError: No module named 'flash_attn'`

Same fix — re-run `install_release.sh`. `flash-attn` builds from source and can fail silently on mismatched environments.

### `ModuleNotFoundError: No module named 'src'`

Run `genecad` from inside the cloned repository directory, or ensure `PYTHONPATH` includes the repo root:

```bash
cd /path/to/genecad
genecad predict ...
```

## Hugging Face / Network Errors

### `GatedRepoError` or `401 Unauthorized` when downloading models

The default GeneCAD models are public, so login is usually not required. If you use a private or gated checkpoint, or Hugging Face rate-limits anonymous downloads, log in once:

```bash
huggingface-cli login
```

### Slow or failed model download

Model weights (~5 GB) are cached in `~/.cache/huggingface` after the first download. If you are on a cluster without internet access, download the weights on a login node first, then run inference on a compute node — the cache is reused automatically.

### Training data download fails (`huggingface-cli download` error)

```bash
huggingface-cli login   # ensure you are authenticated
genecad train --domain plant   # re-run; completed steps are skipped
```

## Runtime Errors

### No CUDA GPUs
```
RuntimeError: No CUDA GPUs are available
```
The main model requires a CUDA GPU to be run for both inference
and training - it cannot be used on a CPU alone. This problem is
often caused by incompatibility between CUDA and pytorch versions, or
by pytorch installing without CUDA drivers. Run the following script
in python:

```
import torch
print(torch.version.cuda)
print(torch.cuda.is_available())
```

If `torch.version.cuda` is `None` or an empty string, you have installed a version of torch
that does not support GPU acceleration. This can happen if the GPUs are not
visible to `pip` during the installation process.

If `torch.version.cuda` returns a version number but `torch.cuda.is_available()` is false,
there is either an incompatibility between the driver and pytorch, or you do not have access to
your machine's GPU. On HPCs using SLURM, this can happen if you request a GPU node with `salloc`. You can call `srun` from within `salloc` : `srun --pty bash -i` for
an interactive session. Alternatively, your pytorch CUDA version may be incompatible with the
driver installed. Check this using the command `nvidia-smi` or `nvcc --version`. The version should
be the same or a minor version higher than `torch.version.cuda`.

In most situations, the solution is to remove the current environment and retry installation.
If you are using an HPC with multiple nodes, you must run the installation process on a node with a GPU.

### GPU / CUDA Errors

**`CUDA out of memory` (OOM) during inference**

```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```


Reduce the batch size. GeneCAD auto-retries at decreasing sizes, but you can skip the probing and pin a safe value directly:

```bash
genecad predict -i genome.fa -s MySpecies --batch-size 4
# If still OOM, try 2 or 1
genecad predict -i genome.fa -s MySpecies --batch-size 1
```

**`CUDA out of memory` during training**

```bash
genecad train --domain plant -b 2   # reduce per-GPU batch size
```

Gradient accumulation is adjusted automatically to maintain the effective batch size.

### Assertion Error during detect_intervals.py

```
Traceback (most recent call last):
  File "/local/workdir/ahb232/genecad/scripts/predict.py", line 1541, in <module>
    main()
  File "/local/workdir/ahb232/genecad/scripts/predict.py", line 1533, in main
    detect_intervals(args)
  File "/local/workdir/ahb232/genecad/scripts/predict.py", line 830, in detect_intervals
    sequence_predictions = merge_prediction_datasets(
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/local/workdir/ahb232/genecad/.venv/lib/python3.12/site-packages/src/prediction.py", line 65, in merge_prediction_datasets
    assert np.array_equal(
           ^^^^^^^^^^^^^^^
AssertionError
```
This can be caused if there was an interruption in the earlier predict.py step
and that step was restarted. Certain data may have been written to file twice. Remove the
affected directory and rerun predict.py. This may only affect one chromosome in a multi-chromosome
run, in which case only the affected chromosome needs to be redone.

### Run was interrupted (job timeout, manual Ctrl-C, OOM)

Re-run the exact same command. GeneCAD checks which steps have already written their output and skips them — only the remaining work runs.

> [!WARNING] If the run was interrupted during step 2/5 - Prediction, there may be a partially-formed
> predictions subdirectory in the directory of the chromosome that was being processed: e.g.
> output_dir/ChrN/predictions_ChrN. Delete this subdirectory and its contents before restarting the run.

```bash
genecad predict -i genome.fa -s MySpecies   # safe to re-run
```

To force a chromosome to restart from scratch, delete its directory:

```bash
rm -rf genecad_result/MySpecies_predictions/Chr1/
```

## SkyPilot Errors

### `sky api start` fails with `RuntimeError: Failed to start SkyPilot API server`

The SkyPilot API server starts in the background and `sky api start` may time out on its own health check even though the server is running fine. Check if it actually started:

```bash
curl http://127.0.0.1:46580/api/health
```

If you see `{"status": "ok"}`, the server is up — ignore the error and proceed normally. If the server is not running:

```bash
sky api stop
sleep 2
sky api start
```

### `sky check lambda` shows `Lambda: disabled`

You need to configure your Lambda API key. Generate one at [https://cloud.lambdalabs.com/api-keys](https://cloud.lambdalabs.com/api-keys), then:

```bash
mkdir -p ~/.lambda_cloud
echo "api_key = YOUR_API_KEY_HERE" > ~/.lambda_cloud/lambda_keys
sky check lambda   # should now show: Lambda: enabled ✓
```

### `sky launch` fails with connection errors on an HPC compute node

HPC compute nodes typically have no outbound internet access and cannot reach cloud provider APIs. SkyPilot must be run from your **local machine** or the cluster's **login node** (which usually has internet). Check with your sysadmin if you are unsure.

### `bash predict.sh: No such file or directory` during manual debugging

If you use `ssh genecad` to manually interact with the node, you will land in the home directory (`~`). SkyPilot syncs your local repo to `~/sky_workdir/` — you must `cd` there before running Docker or local scripts:

```bash
ssh genecad
cd ~/sky_workdir   # ← the repo and predict.sh are here
docker run --rm --gpus all \
  -v $(pwd):/workspace -w /workspace \
  ghcr.io/plantcad/genecad_v1:latest \
  bash predict.sh
```

### Custom input FASTA is "not found" during `sky exec`

If you placed your custom FASTA inside the `genecad/data/` directory, SkyPilot did not upload it. SkyPilot automatically respects the repository's `.gitignore` file, which excludes the `data/` folder to prevent uploading massive files.

**Fix:** Move your FASTA to the root of the repository (e.g., `genecad/my_genome.fa`) so it isn't ignored, and re-run `sky launch`.

### SkyPilot conflicts with GeneCAD dependencies

Install SkyPilot in a separate virtual environment, not inside the GeneCAD `.venv`:

```bash
python3 -m venv ~/skypilot-env
source ~/skypilot-env/bin/activate
pip install "skypilot[lambda]>=0.10.3"
```

## Other Problems

### No GPU Available

If you do not have access to a local GPU, you have two options:

**Option 1 — Cloud GPU via SkyPilot**:
Follow the complete end-to-end instructions in the [Cloud Provisioning](#cloud-provisioning-skypilot) section to deploy a node, run the workload, and securely download your results.

**Option 2 — HPC cluster via SLURM** (see [HPC Clusters](#hpc-clusters-slurm)):

```bash
salloc --partition=gpu-queue --nodes=1 --gres=gpu:1
genecad predict -i genome.fa -s MySpecies
```

### GPU utilization is low / inference is slow

The auto batch size starts conservatively. After a successful run, GeneCAD prints the batch size it settled on — pin that value in future runs to skip probing:

```bash
# Example: the run printed "Succeeded with batch size 24"
genecad predict -i genome.fa -s MySpecies --batch-size 24
```

**`CUDA error: no kernel image is available for execution on the device`**

Your PyTorch build was compiled for a different CUDA version than what is installed. Check:

```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

The two versions must match. Install the correct PyTorch build or use Docker.

**`torch.cuda.is_available()` returns `False`**

- On a cluster, you may need to request a GPU node (`salloc --gres=gpu:1 ...`) before running.
- Check that `CUDA_VISIBLE_DEVICES` is not set to an empty string or `-1`.
- Verify your NVIDIA driver is loaded: `nvidia-smi`.

### Assembly has thousands of small scaffolds and the run is very slow

Use `--top-n-contigs` to process only the longest sequences, which contain most of the genes:

```bash
genecad predict -i genome.fa -s MySpecies --top-n-contigs 50
```

### Which `-m` mode should I use?

| Organism type | Flag |
|---------------|------|
| Land plants (angiosperms, gymnosperms, mosses, ferns) | `-m plant` (default) |
| Algae | `-m plant` |
| Vertebrates (fish, amphibians, reptiles, birds, mammals) | `-m animal` |
| Invertebrates | `-m animal` (experimental) |

---
