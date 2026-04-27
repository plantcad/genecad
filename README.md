<div align="center">

# GeneCAD: Foundation Model Genome Annotation

![](https://img.shields.io/badge/version-1.1.0-blue)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/plantcad/genecad/actions/workflows/ci.yaml/badge.svg)](https://github.com/plantcad/genecad/actions/workflows/ci.yaml)
[![DOI](https://zenodo.org/badge/DOI/10.1101/2025.10.31.685877.svg)](https://doi.org/10.1101/2025.10.31.685877)
[![Quick Start](https://img.shields.io/badge/Quick%20Start-1%20script-blue)](#quick-start)
[![Web UI](https://img.shields.io/badge/Web%20UI-No--code-green)](#web-interface-no-code)
[![Release Wheel](https://img.shields.io/badge/Install-GitHub%20Release%20Wheel-orange)](#using-github-release-wheel)
[![GeneCAD Downloads](https://img.shields.io/github/downloads/plantcad/genecad/total?label=GitHub%20downloads)](https://github.com/plantcad/genecad/releases)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg?style=flat)](https://huggingface.co/collections/plantcad/genecad-68c686ccf14312bf6de356de)

GeneCAD is an end-to-end genome annotation pipeline for plants and animals, powered by the DNA foundation model [PlantCAD2](https://doi.org/10.1101/2025.10.31.685877).

*No species-specific training data. No external alignments. Out-of-the-box multi-GPU scalability.*

</div>

Unlike traditional annotation tools that rely on hand-crafted features or splice-site grammars, GeneCAD learns gene structure directly from sequence using a pretrained transformer encoder followed by a Viterbi decoder and protein-level refinement via [ReelProtein](https://onlinelibrary.wiley.com/doi/10.1111/tpj.70483).

GeneCAD natively supports both plant and animal genome annotation; use `-m plant` (default) or `-m animal` to select the model family.

**Recommended install:** run the quick start below. It installs from the tracked GitHub release wheel, so the GitHub download counter is updated.

## Contents

- [Quick Start](#quick-start)
- [Web Interface (No-Code)](#web-interface-no-code)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [Using GitHub Release wheel](#using-github-release-wheel)
  - [Using uv](#using-uv)
  - [Using Docker](#using-docker)
  - [Using SLURM](#using-slurm)
  - [Using SkyPilot](#using-skypilot)
- [Inference](#inference)
  - [Available models](#available-models)
  - [Running the pipeline](#running-the-pipeline)
  - [Common run recipes](#common-run-recipes)
  - [Multi-GPU acceleration](#multi-gpu-acceleration)
  - [Pipeline steps](#pipeline-steps)
  - [Outputs](#outputs)
  - [GFF3 format notes](#gff3-format-notes)
  - [Resuming an interrupted run](#resuming-an-interrupted-run)
  - [Throughput](#throughput)
  - [Troubleshooting](#troubleshooting)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Development](#development)
  - [Docker](#docker)
  - [Reproduction](#reproduction)

---

## Quick Start

Annotate a full plant genome in a few commands. No configuration required — the example *Arabidopsis thaliana* TAIR12 sequence is downloaded automatically.

```bash
# 1. Create and activate a virtual environment
uv venv
source .venv/bin/activate

# 2. Install GeneCAD from tracked GitHub release wheel
bash scripts/install_release.sh

# 3. Run the full prediction pipeline
genecad predict
```

> [!TIP]
> Two GFF3 files are written to `genecad_result/Athaliana_predictions/`:
>
> | File | Description |
> |------|-------------|
> | `Athaliana_GeneCAD_raw.gff` | Raw model predictions (all chromosomes merged) |
> | `Athaliana_GeneCAD_final.gff` | Protein-refined, publication-ready annotations |

To annotate your own genome, pass your FASTA file and a few labels:

```bash
genecad predict \
  -i /path/to/my_genome.fa \
  -o /path/to/output_dir \
  -s MySpecies \
  -m plant   # or: -m animal  for vertebrate genomes
```

**Use all available GPUs** to run chromosomes in parallel (significantly faster on large genomes):

```bash
genecad predict --gpus all
```

> [!NOTE]
> Prefer cloning the repo? See [Using uv](#using-uv) in the Setup section — it gives the same result without the download count.

---

## Web Interface (No-Code)

If you are not comfortable with the command line or are SSH'd into a remote computing cluster, GeneCAD provides a simple graphical web interface to run annotations directly from your browser.

1. Install GeneCAD first by running the quick start install command:
  ```bash
  bash scripts/install_release.sh
  ```
2. Launch the Web UI. If you are running this on a remote cluster without GUI access, use the `--share` flag to generate a secure, temporary public link:
   ```bash
   genecad ui --share
   ```
3. Look for a link like `https://xxxxx.gradio.live` in your terminal. Click it to open the GeneCAD UI on your local laptop.
4. The web UI lets you set the same main options as the command line, including:
  - input FASTA path or file upload
  - output directory
  - species name
  - model family (`plant` or `animal`)
  - top contigs to process
  - minimum transcript length
  - CPU workers
  - batch size
  - GPUs (`all` or a comma-separated list)
5. Click "Run GeneCAD Pipeline" to start annotation.

---

## Prerequisites

| Requirement | Minimum | Recommended | Notes |
|-------------|---------|-------------|-------|
| OS | Linux (x86-64) | — | macOS is not supported (no CUDA) |
| GPU | NVIDIA ≥ 16 GB VRAM | A100 / H100 | e.g. RTX 3090/4090 for development |
| CUDA | 12.4 | 12.8 | Must match PyTorch 2.7.1 build |
| Python | 3.11 | 3.12 | Managed automatically by `uv` |
| Disk | ~20 GB free | — | Model weights cached in `~/.cache/huggingface` |

Model weights are downloaded automatically from Hugging Face on first run. Internet access is required unless weights are pre-cached.

---

## Setup

Quick Start is the recommended path for most users because it is the simplest and it tracks GitHub downloads.

If you need an advanced setup, use one of the sections below.

### Using GitHub Release wheel

Use the **Quick Start** install commands above.

That path is the recommended install and is the most reliable setup we currently support.

If needed, you can override the version in the installer script:

```bash
GENECAD_VERSION=1.1.0 bash scripts/install_release.sh
```

For development or contributing, clone and sync dependencies instead:

```bash
git clone https://github.com/plantcad/genecad.git
cd genecad
uv sync --extra torch --extra mamba
```

### Using uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that handles the virtual environment and all dependencies in one command.

```bash
# Clone the repository
git clone https://github.com/plantcad/genecad && cd genecad

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Make uv available immediately (or open a new terminal)
export PATH="$HOME/.local/bin:$PATH"

# Install all dependencies
# mamba and causal-conv1d build from source — this can take 3–30 minutes
uv sync --extra torch --extra mamba

# Run the pipeline
bash predict.sh
```

PyTorch is pinned to 2.7.1 (CUDA 12.8) to ensure compatibility with `mamba` (2.2.4) and `causal-conv1d` (1.5.0.post8). Newer combinations may work but are not officially tested.

`mamba` and `causal-conv1d` build from source without build isolation for reliability. To cache the built wheels and speed up future installs, add these entries to `pyproject.toml`:

```toml
[tool.uv.sources]
mamba-ssm = { path = "path/to/mamba_ssm-2.2.4-*.whl" }
causal-conv1d = { path = "path/to/causal_conv1d-1.5.0.post8-*.whl" }
# Or use a remote URL:
# mamba-ssm = { url = "https://.../mamba_ssm-2.2.4-*.whl" }
```

### Using Docker

The Docker image at `ghcr.io/plantcad/genecad_v1` bundles the complete runtime. Source code is **mounted at run time**, so changes you make to the cloned repository take effect immediately — no rebuild needed.

> [!IMPORTANT]
> If your environment requires a specific Docker alias (e.g., `docker1`), replace `docker` with your local command in the examples below.

```bash
# Clone the repository
git clone https://github.com/plantcad/genecad && cd genecad

# Pull the image
docker pull ghcr.io/plantcad/genecad_v1:latest

# Run on the bundled Arabidopsis example (auto-downloads FASTA)
docker run --rm --gpus all \
  -v $(pwd):/workspace -w /workspace \
  ghcr.io/plantcad/genecad_v1:latest \
  bash predict.sh

# Annotate a custom plant genome
docker run --rm --gpus all \
  -v $(pwd):/workspace -w /workspace \
  ghcr.io/plantcad/genecad_v1:latest \
  bash predict.sh \
    -i /workspace/data/my_plant.fa \
    -o /workspace/output \
    -s Zmays \
    -m plant

# Annotate an animal genome
docker run --rm --gpus all \
  -v $(pwd):/workspace -w /workspace \
  ghcr.io/plantcad/genecad_v1:latest \
  bash predict.sh \
    -i /workspace/data/my_animal.fa \
    -o /workspace/output \
    -s Hsapiens \
    -m animal
```

### Using SLURM

Follow the [uv](#using-uv) instructions above to create a virtual environment on your cluster. Load the appropriate modules for your cluster (e.g. CUDA 12.8, Python 3.12). The minimum tested combination is CUDA 12.4, Python 3.11, and PyTorch 2.5.1.

**Interactive single-node job:**

```bash
salloc --partition=gpu-queue --nodes=1 --ntasks=1 --gres=gpu:1

cd /path/to/genecad
bash predict.sh \
  -i /path/to/input.fa \
  -o /path/to/output \
  -s MySpecies \
  -m plant
```

**Batch job (`sbatch`):**

```bash
cat << 'EOF' > run_genecad.slurm
#!/bin/bash
#SBATCH --partition=gpu-queue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=genecad

cd /path/to/genecad
bash predict.sh \
  -i /path/to/input.fa \
  -o /path/to/output \
  -s MySpecies \
  -m plant
EOF

sbatch run_genecad.slurm
```

> [!TIP]
> **Multi-GPU on a Single Node:** To use all GPUs on a node, request them with `--gres=gpu:4` (or however many are available) and pass `--gpus all` to `predict.sh`. If there are many chromosomes, they will be distributed across GPUs. If there are fewer chromosomes than GPUs, GeneCAD will automatically split the longest sequences into parallel windows across the GPUs.
> 
> **Multi-Node Distributed Inference (e.g., TACC):** GeneCAD natively detects multi-node SLURM topologies. If you allocate multiple nodes (e.g., `#SBATCH --nodes=4`), you can use `srun` to seamlessly process enormous single contigs across the entire cluster. `predict.sh` will automatically bypass local launchers and allow PyTorch Lightning to route the distributed process windows.
> 
> ```bash
> # Request multiple nodes/GPUs and launch via srun
> #SBATCH --nodes=2
> #SBATCH --ntasks=8
> #SBATCH --gres=gpu:4
> srun bash predict.sh -i /path/to/genome.fa -s MySpecies --gpus all
> ```
>
> **Custom launcher:** If your cluster requires a non-standard Python entrypoint (e.g. `srun python` instead of `torchrun`), use `--launcher` or the `LAUNCHER` environment variable to override automatic detection:
>
> ```bash
> # Via flag
> bash predict.sh -i genome.fa -s MySpecies --launcher 'srun python'
>
> # Via environment variable
> LAUNCHER='srun python' bash predict.sh -i genome.fa -s MySpecies
> ```

### Using SkyPilot

[SkyPilot](https://docs.skypilot.co/en/latest/docs/index.html) lets you provision on-demand cloud GPUs across many providers (AWS, GCP, Lambda, RunPod, etc.) without managing infrastructure manually. This is useful if you do not have access to a local GPU or HPC cluster.

```bash
# Install SkyPilot (adjust the cloud extra as needed)
pip install "skypilot[lambda]>=0.10.3"

# Clear local SkyPilot state, if any
sky api stop; [ -d ~/.sky ] && rm -rf ~/.sky

# Deploy a single GPU node (no setup — we use Docker instead of uv sync)
sky launch --num-nodes 1 --yes --no-setup \
  --cluster genecad examples/configs/cluster.sky.yaml

# SSH into the node, pull the pre-built image, and run
ssh genecad
docker pull ghcr.io/plantcad/genecad_v1:latest
docker run --rm --gpus all \
  -v $(pwd):/workspace -w /workspace \
  ghcr.io/plantcad/genecad_v1:latest \
  bash predict.sh

# Terminate the node when done
sky down genecad
```

> [!IMPORTANT]
> **Why Docker instead of `uv sync`?**
> `flash-attn`, `mamba-ssm`, and `causal-conv1d` must all compile from source, which can take **30–60 minutes** on a fresh cloud node and may fail if the CUDA/GCC versions mismatch. The pre-built Docker image includes all compiled packages, making cluster setup take seconds instead of hours.

See the [Throughput](#throughput) section for GPU cost comparisons across providers.

<details><summary>SkyPilot GPU rates</summary>

```
> sky show-gpus L4:1   # low-end development GPU
GPU  CLOUD   INSTANCE_TYPE  DEVICE_MEM  HOURLY_PRICE  REGION
L4   RunPod  1x_L4_SECURE   24GB        $ 0.440       CA
L4   GCP     g2-standard-4  24GB        $ 0.705       us-east4
L4   AWS     g6.xlarge      22GB        $ 0.805       us-east-2

> sky show-gpus H100:1  # high-end production GPU
GPU   CLOUD       INSTANCE_TYPE          DEVICE_MEM  HOURLY_PRICE  REGION
H100  Hyperbolic  1x-H100-75-722         80GB        $ 1.290       default
H100  Lambda      gpu_1x_h100_pcie       80GB        $ 2.490       europe-central-1
H100  GCP         a3-highgpu-1g          80GB        $ 5.383       us-central1
```

</details>

---

## Inference

### Available models

GeneCAD provides two pre-trained models for different taxonomic groups. Both are downloaded automatically from Hugging Face on first run.

| Mode (`-m`) | Organism type | Base model | GeneCAD head |
|-------------|--------------|------------|--------------|
| `plant` *(default)* | Plants | [`plantcad/pcad2-200M-cnet-baseline`](https://huggingface.co/plantcad/pcad2-200M-cnet-baseline) | [`Zong-Yan/genecad_plant`](https://huggingface.co/Zong-Yan/genecad_plant) |
| `animal` | Animals / vertebrates | [`emarro/vcad2_small_experimental`](https://huggingface.co/emarro/vcad2_small_experimental) | [`Zong-Yan/genecad_vert`](https://huggingface.co/Zong-Yan/genecad_vert) |

### Running the pipeline

The primary entry point is `predict.sh`. It discovers all chromosomes in the input FASTA automatically and runs the complete pipeline on each one.

```
Usage: predict.sh [OPTIONS]

Options:
  -i, --input PATH      Genome FASTA file to annotate
                        Default: downloads Arabidopsis thaliana TAIR12 example
  -o, --output DIR      Output directory  (default: genecad_result/Athaliana_predictions)
  -s, --species NAME    Species label prefixed on output filenames  (default: Athaliana)
  -m, --mode MODE       Model to use: plant | animal  (default: plant)
  -n, --top-n-contigs N Predict only the N longest FASTA sequences
                        (default: all sequences)
  -b, --batch-size N    Inference batch size per GPU  (default: auto — scaled to GPU VRAM)
  -g, --gpus LIST       GPU IDs to use: comma-separated list or 'all'  (default: 0)
  --launcher CMD        Custom entrypoint to launch predict.py (e.g. 'srun python').
                        Overrides automatic DDP/SLURM detection.
                        Can also be set via the LAUNCHER environment variable.
  -h, --help            Show this help message
```

> [!IMPORTANT]
> `predict.sh` is the **only script you need to run**. Files under `scripts/` (`scripts/predict.py`, `scripts/gff.py`, etc.) are internal pipeline modules called automatically by `predict.sh` — do not run them directly.

**Default example (Arabidopsis TAIR12, auto-downloaded):**

```bash
bash predict.sh
```

For common real-world commands (custom genome, top-N contigs, multi-GPU), see [Common run recipes](#common-run-recipes).

### Common run recipes

**Custom plant genome:**

```bash
bash predict.sh \
  -i /path/to/Zmays.fa \
  -o /path/to/output \
  -s Zmays \
  -m plant
```

**Custom animal genome:**

```bash
bash predict.sh \
  -i /path/to/genome.fa \
  -o /path/to/output \
  -s Hsapiens \
  -m animal
```

**Run only top N longest contigs/scaffolds** (useful for very fragmented assemblies):

```bash
bash predict.sh \
  -i /path/to/genome.fa \
  -o /path/to/output \
  -s MySpecies \
  -m plant \
  --top-n-contigs 100
```

When `--top-n-contigs` is used, GeneCAD selects the N longest sequences by length, then processes them in the same order they appear in the input FASTA headers. This keeps output ordering stable and easier to compare with the source assembly.

**Use all GPUs with automatic per-GPU batch sizing:**

```bash
bash predict.sh -i /path/to/genome.fa -s MySpecies --gpus all
```

**Pin a known-safe batch size for reproducibility:**

```bash
bash predict.sh -i /path/to/genome.fa -s MySpecies --gpus all --batch-size 4
```

### Multi-GPU acceleration

By default, `predict.sh` uses GPU `0` and processes chromosomes sequentially. Pass `--gpus` to distribute chromosomes across multiple GPUs and process them **in parallel** — ideal for large, multi-chromosome genomes.

```bash
# Use all available GPUs (auto-detected)
bash predict.sh -i genome.fa -s MySpecies --gpus all

# Use specific GPUs (e.g. 0, 1, 2, 3)
bash predict.sh -i genome.fa -s MySpecies --gpus 0,1,2,3

# Use 2 GPUs with a fixed batch size
bash predict.sh -i genome.fa -s MySpecies --gpus 0,1 -b 32
```

Chromosomes are assigned round-robin across the specified GPUs. The batch size is auto-scaled independently per GPU (so mixed fleets — e.g. an A100 and a V100 — work correctly).

Progress bars are shown per running worker in the terminal, and labels reflect the physical GPU selected for each worker. The pipeline is fully resumable in both single- and multi-GPU modes.

### Pipeline steps

The script runs the following steps for every chromosome in the input FASTA:

| Step | Script | Description |
|------|--------|-------------|
| 1. Extract | `scripts/extract.py` | Parse FASTA and tokenize sequences into Zarr format |
| 2. Predict | `scripts/predict.py` | Windowed inference with the GeneCAD classifier (GPU) |
| 3. Decode | `scripts/predict.py` | Viterbi decoding of per-token logits into genomic intervals |
| 4. Export | `scripts/predict.py` | Convert decoded intervals to raw GFF3 |
| 5. Filter | `scripts/gff.py` | Remove short or structurally invalid gene models |
| 6. Merge | `scripts/merge_gff.py` | Concatenate per-chromosome GFFs into genome-wide files |
| 7. Refine | `scripts/refine.py` | ReelProtein: protein embedding + XGBoost scoring for final filtering |

### Outputs

After running, the output directory contains:

```
<OUTPUT_DIR>/
├── <SPECIES_ID>_GeneCAD_raw.gff      ← all predicted gene models (pre-refinement)
├── <SPECIES_ID>_GeneCAD_final.gff    ← final, protein-validated annotations
└── <CHR_ID>/                         ← per-chromosome intermediates (for debugging)
    ├── predictions_recall.gff
    └── pipeline/
        ├── sequences.zarr
        ├── predictions.zarr
        └── intervals.zarr
```

The two top-level GFF3 files are the primary outputs. Intermediate files are preserved for debugging or downstream analysis.

### GFF3 format notes

GeneCAD outputs standard 9-column GFF3 records and includes a `##gff-version 3` header. The final files contain hierarchical feature relationships via `ID` and `Parent` attributes, including:

- `gene`
- `mRNA`
- `five_prime_UTR`
- `CDS`
- `three_prime_UTR`

Coordinates are 1-based and inclusive, consistent with GFF3 conventions.

### Resuming an interrupted run

`predict.sh` is fully **resumable** — if a run is interrupted (e.g. job timeout or OOM), simply re-run the same command. Each step checks whether its output file already exists and skips it if so. Only the remaining work will run.

To **force a step to re-run**, delete its output:

```bash
# Re-run predictions for one chromosome from scratch
rm -rf genecad_result/<OUTPUT_DIR>/<CHR_ID>/

# Re-run only the token prediction step (Step 2)
rm -rf genecad_result/<OUTPUT_DIR>/<CHR_ID>/pipeline/predictions.zarr
```

### Throughput

Observed GeneCAD inference throughput on different GPUs. Datacenter-class GPUs (A100/H100) are 3–6× more cost-efficient per megabase than consumer development GPUs.

| Provider | Instance | Throughput (bp/s) | Cost ($/hr) | Cost per Mbp ($) |
|----------|----------|-------------------|-------------|------------------|
| GCE | g2-standard-32 / L4 | 4,063 | 1.7344 | 0.1186 |
| Lambda | gpu_1x_a100_sxm4 | 17,687 | 1.2900 | 0.0202 |
| Lambda | gpu_1x_h100_pcie | 22,020 | 2.4900 | 0.0314 |
| Lambda | gpu_2x_h100_sxm5 | 34,290 | 6.3800 | 0.0517 |

Estimated cost for common reference plant genomes (using Lambda A100 at $0.0202/Mbp):

| Genome | Length (Mbp) | GPU Hours | Cost ($) |
|--------|-------------|-----------|----------|
| *Arabidopsis thaliana* | 120 | 2.12 | 2.42 |
| *Zea mays* (corn) | 2,182 | 34.27 | 44.04 |
| *Hordeum vulgare* (barley) | 4,224 | 66.34 | 85.32 |
| *Triticum aestivum* (wheat) | 14,577 | 227.00 | 294.46 |

### Troubleshooting

| Symptom | Likely cause | What to do |
|--------|--------------|------------|
| GPU utilization is low | Batch size is conservative for stability | Re-run with a higher fixed `--batch-size` (for example 8, 16, 32) and keep the highest stable value |
| Auto batch size starts too high | Free VRAM sampled before model load | Keep auto mode and allow retry shrink, or pin a stable manual `--batch-size` |
| Interrupted run | Job timeout / manual stop | Re-run the same command; completed steps are skipped automatically |

---

## Training

This section is for researchers who want to **fine-tune GeneCAD on new species** or reproduce the published models. If you only want to annotate a genome, skip to [Inference](#inference) — no training is needed.

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| ≥ 2 × NVIDIA GPUs | Training uses PyTorch DDP. Single-GPU works but is very slow. |
| HuggingFace account | Run `huggingface-cli login` to download training data. |
| WandB (optional) | Metrics are logged to WandB. Disable with `WANDB_MODE=disabled`. |

### Quick start

```bash
# Login to HuggingFace (one-time)
huggingface-cli login

# Fine-tune on the 5-species plant dataset (downloads data automatically)
bash train.sh --domain plant

# Fine-tune on the vertebrate animal dataset (downloads data automatically)
bash train.sh --domain animal

# Use all detected GPUs
bash train.sh --domain animal -g all

# Use a specific subset of GPUs
bash train.sh --domain animal -g 0,2,3

# Fine-tune with custom settings
bash train.sh \
  --domain plant \
  -g 4 \              # number of GPUs
  -b 4 \              # per-GPU batch size
  --epochs 3 \        # training epochs
  -l 1e-4 \           # learning rate
  -o results/my_run \ # output directory
  -r my-run-name      # WandB run name
```

### Options

```
Options:
  -m, --domain MODE     Training domain: plant | animal (default: plant)
  -o, --output DIR      Output directory for checkpoints and logs
                        (default: genecad_result/training)
  -r, --run-name NAME   WandB run name  (default: genecad-plant-multispecies)
  -p, --project NAME    WandB project   (default: genecad)
  -g, --gpus SPEC       GPU selection: count, comma list, or 'all' (default: 1)
                         examples: 2, 0,1,3, all
  -b, --batch-size N    Per-GPU batch size (default: 4)
  --epochs N            Number of training epochs (default: 1)
  -e, --effective N     Effective batch size (default: 384)
  -l, --lr RATE         Learning rate   (default: 2e-4)
  --base-model ID       Override base encoder model ID
  --animal-input-dir DIR  Override animal FASTA source directory
  --animal-gff-dir DIR    Override animal GFF source directory
  -h, --help            Show this message
```

Notes:
- `-g all` requires `nvidia-smi` to auto-detect GPU IDs.
- When using a list (for example `-g 0,2,3`), training is restricted to that subset via `CUDA_VISIBLE_DEVICES`.

### Pipeline steps

`train.sh` is fully resumable — each step checks for its output and skips if already done.

| Step | Description |
|------|-------------|
| 0. Download | Fetch training GFF3 and FASTA files from HuggingFace (plant and animal) |
| 1. Link | Create symlinks with naming conventions expected by the config |
| 2. Extract GFF | Parse gene annotations into a flat feature table (Parquet) |
| 3. Tokenize | Tokenize genome sequences into Zarr format |
| 4. Filter | Remove incomplete or invalid gene features |
| 5. Stack | Convert features into genomic interval windows |
| 6. Labels | Assign per-token class labels from the filtered intervals |
| 7. Splits | Sample training windows and split into train/validation sets |
| 8. Train | Fine-tune GeneCAD with PyTorch DDP + PyTorch Lightning |

### Outputs

```
<OUTPUT_DIR>/
├── training.log                        ← full training log (tee'd to stdout)
├── checkpoints/
│   └── *.ckpt                          ← Lightning checkpoints
└── pipeline/                           ← intermediate files (can be large)
    ├── extract/
    │   ├── raw_features.parquet        ← parsed gene annotations
    │   └── tokens.zarr                 ← tokenized genome sequences
    ├── transform/
    │   ├── features.parquet            ← filtered features
    │   ├── intervals.parquet           ← stacked genomic intervals
    │   ├── labels.zarr                 ← per-token class labels
    │   ├── sequences.zarr              ← sequence + label dataset
    │   └── windows.zarr                ← sampled training windows
    └── prep/
        └── splits/
            ├── train.zarr              ← training split
            └── valid.zarr              ← validation split
```

> [!WARNING]
> The `pipeline/` directory can be **several hundred GB** depending on genome sizes and number of species. It is safe to delete after training — checkpoints are self-contained. The pipeline is resumable, so intermediate files are preserved by default to avoid recomputation.

To use a trained checkpoint for inference, pass it to `predict.sh`:

```bash
bash predict.sh \
  -i genome.fa \
  -s MySpecies \
  -m plant \
  --model-checkpoint results/my_run/checkpoints/epoch=0-step=5000.ckpt
```

### Summarizing training data

To inspect the class balances, masking rates, and terminal codon frequencies of your training or validation splits, you can use the built-in summary tool on your compiled `zarr` datasets:

```bash
uv run python scripts/summarize.py summarize_training_dataset \
  --input genecad_result/training/plant/pipeline/prep/splits/train.zarr
```
This utility provides key statistics to ensure your model gradients remain stable across custom datasets and species variations.

---

## Evaluation

GeneCAD includes a built-in evaluation tool, `scripts/evaluate.py`, that produces a structured five-section report comparing predicted annotations against a reference. It is self-contained for Sections 1, 2, 3, and 5. Section 4 uses BUSCO if you enable it, but BUSCO is optional.

### Usage

```bash
uv run python scripts/evaluate.py \
  --ref   /path/to/reference.gff3 \
  --pred  /path/to/Athaliana_GeneCAD_final.gff \
  --fasta /path/to/genome.fa \
  --output report.txt
```

Most users only need these arguments:

| Option | Description |
|--------|-------------|
| `--ref` | Reference GFF3 annotation (required) |
| `--pred` | Predicted GFF3 annotation (required) |
| `--fasta` | Genome FASTA; enables Sections 3 and 4 |
| `--output` | Write the report to a file instead of stdout |
| `--skip-busco` | Skip Section 4 for a faster, fully portable run |

### BUSCO setup

If BUSCO is available on your system, GeneCAD can pick it up automatically. If not, choose the option that matches your environment:

| Environment | Recommended setting |
|-------------|---------------------|
| Local Conda / Mamba | `export BUSCO_CMD='conda run -n busco-5.5.0 busco'` |
| HPC with a site activate script | `export BUSCO_ACTIVATE_SCRIPT=/programs/miniconda3/bin/activate` |
| Already activated BUSCO env | Run `busco --version` first, then invoke evaluation normally |
| No BUSCO required | Add `--skip-busco` |

```bash
# Portable full evaluation with an explicit BUSCO command
export BUSCO_CMD='conda run -n busco-5.5.0 busco'
uv run python scripts/evaluate.py \
  --ref /path/to/reference.gff3 \
  --pred /path/to/prediction.gff3 \
  --fasta /path/to/genome.fa \
  --output report.txt
```

If your cluster uses a pre-installed BUSCO module or activate script, set the script path once and keep using the same evaluation command:

```bash
export BUSCO_ACTIVATE_SCRIPT=/programs/miniconda3/bin/activate
uv run python scripts/evaluate.py \
  --ref /path/to/reference.gff3 \
  --pred /path/to/prediction.gff3 \
  --fasta /path/to/genome.fa \
  --output report.txt
```

For a quick debug run, skip BUSCO entirely:

```bash
uv run python scripts/evaluate.py \
  --ref /path/to/reference.gff3 \
  --pred /path/to/prediction.gff \
  --fasta /path/to/genome.fa \
  --skip-busco \
  --output report.txt
```

If you want GeneCAD to try to install BUSCO automatically, add `--auto-install-busco` and optionally change the environment name with `--busco-env`.

> [!TIP]
> If BUSCO is already on your `PATH`, GeneCAD will use it automatically after a quick self-check. Otherwise, it falls back to the command or activate script you provide.

If you run BUSCO with `--augustus`, point `AUGUSTUS_CONFIG_PATH` to a writable directory:

```bash
mkdir -p "$HOME/augustus_config"
cp -r /programs/miniconda3/envs/busco-5.5.0/config "$HOME/augustus_config"
export AUGUSTUS_CONFIG_PATH="$HOME/augustus_config/config"
```

Other evaluation options:
- `--lineage` (default: `embryophyta_odb10`)
- `--cpu` (default: `32`)
- `--busco-env` (default: `busco-5.5.0`)
- `--busco-cmd` for an explicit BUSCO launcher
- `--busco-activate-script` for cluster-specific activation scripts

### Example output (*Arabidopsis thaliana*)

```
==============================================================
SECTION 1 – CDS-Based Evaluation (UTRs ignored)
  Gene is correct if its CDS chain matches ANY ref isoform.
==============================================================

  Reference : 26867 loci | 32588 transcripts | 145570 unique CDS exons
  Predicted : 25267 loci | 25267 transcripts | 137742 unique CDS exons

--- CDS-exon-level ---
  Precision : 0.9086   Recall : 0.8597   F1 : 0.8835

--- Locus-level ---
  Precision : 0.7571   Recall : 0.7120   F1 : 0.7338

--- Transcript-level ---
  Precision : 0.7571   Recall : 0.5870   F1 : 0.6613


==============================================================
SECTION 2 – Full Exon Evaluation (includes UTRs)
  Intron chain / Locus[IC]: multi-exon transcripts only.
==============================================================

--- Base level (nucleotides) ---
  Precision : 0.9907   Recall : 0.6801   F1 : 0.8065

--- Intron level (splice junctions) ---
  Precision : 0.9173   Recall : 0.7959   F1 : 0.8523

--- Intron chain level ---
  Precision : 0.6179   Recall : 0.3677   F1 : 0.4610


==============================================================
SECTION 3 – Splice Site Analysis
==============================================================

  Total introns analysed : 112475
  GT-AG (canonical)      : 110788  (98.50%)
  GC-AG (semi-canonical) :   1687  ( 1.50%)
  Other (non-canonical)  :      0  ( 0.00%)


==============================================================
SECTION 4 – BUSCO Evaluation
==============================================================

  C:99.1%[S:98.2%,D:0.9%],F:0.5%,M:0.4%,n:2326
  2306  Complete BUSCOs (C)
    12  Fragmented BUSCOs (F)
     8  Missing BUSCOs (M)


==============================================================
SECTION 5 – Site-Level Error Breakdown
  TIS = Translation Initiation Site
  TTS = Translation Termination Site
==============================================================

  Site             Ref    Pred      TP     FP     FN    Prec     Rec      F1
  TIS            28301   25267   23006   2261   5295  0.9105  0.8129  0.8589
  TTS            28340   25267   22998   2269   5342  0.9102  0.8115  0.8580
  Junc.Donor    125574  112475  106403   6072  19171  0.9460  0.8473  0.8940
  Junc.Acc.     126788  112475  106654   5821  20134  0.9482  0.8412  0.8915
```

### Metric guide

| Section | Metric | Description |
|---------|--------|-------------|
| 1 | CDS-exon | Exact coding exon boundary match (UTRs ignored) |
| 1 | Locus | Gene correct if any isoform's CDS chain matches a reference isoform |
| 1 | Transcript | Exact full-isoform CDS chain match |
| 2 | Base | Nucleotide-level overlap of predicted vs. reference exons |
| 2 | Intron | Exact splice junction pair match |
| 2 | Intron chain | Complete intron chain match (= full transcript interior) |
| 3 | Splice sites | GT-AG / GC-AG canonical intron dinucleotide frequency |
| 4 | BUSCO | Benchmarked Universal Single-Copy Orthologs completeness |
| 5 | TIS / TTS | Translation start / stop site precision and recall |
| 5 | Donor / Acceptor | Individual 5′ and 3′ splice site accuracy |

> [!NOTE]
> Sections 3 and 4 require `--fasta`. For BUSCO, prefer `--busco-cmd` for reproducibility, or use `--auto-install-busco` for first-run setup.

---

## Citation

If you use GeneCAD in your research, please cite:

```bibtex
@article{liu2025genecad,
  title   = {GeneCAD: Plant Genome Annotation with a DNA Foundation Model},
  author  = {Liu, Zong-Yan and Berthel, Ana and Czech, Eric and Stitzer, Michelle
             and Hsu, Sheng-Kai and Pennell, Matt and Buckler, Edward S. and Zhai, Jingjing},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.10.31.685877},
  url     = {https://doi.org/10.1101/2025.10.31.685877}
}
```

---

## Development

### Docker

```bash
# Build the image (requires Linux with Docker and NVIDIA drivers)
sudo usermod -aG docker ubuntu && newgrp docker
docker build --progress=plain --no-cache -t genecad:v1.0.1 .

# Test the build — runs the full pipeline on the Arabidopsis example
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace genecad:v1.0.1 \
  bash predict.sh

# Publish to GitHub Container Registry
# Requires a personal access token with "write:packages" stored in GHCR_TOKEN
IMAGE=ghcr.io/plantcad/genecad_v1
docker tag genecad:v1.0.1 $IMAGE:v1.0.1
docker tag genecad:v1.0.1 $IMAGE:latest
echo $GHCR_TOKEN | docker login ghcr.io -u <github-username> --password-stdin
docker push $IMAGE:v1.0.1
docker push $IMAGE:latest
```

### Reproduction

To reproduce the published GeneCAD results for *Juglans regia* (Walnut) chromosome 1:

```bash
mkdir -p data results

# Download FASTA and reference GFF from Hugging Face
uv run huggingface-cli download plantcad/genecad-dev \
  data/plant/fasta/evaluation/Juglans_regia_chr1.fa.gz \
  --repo-type dataset --local-dir .
uv run huggingface-cli download plantcad/genecad-dev \
  data/plant/gff/evaluation/Juglans_regia_chr1.gff3 \
  --repo-type dataset --local-dir .

# Run the full GeneCAD pipeline
docker run --rm --gpus all \
  -v $(pwd):/workspace -w /workspace \
  ghcr.io/plantcad/genecad_v1:latest \
  bash predict.sh \
    -i data/plant/fasta/evaluation/Juglans_regia_chr1.fa.gz \
    -o results \
    -s Jregia \
    -m plant

# Evaluate against the reference annotation
docker run --rm \
  -v $(pwd):/workspace -w /workspace \
  ghcr.io/plantcad/genecad_v1:latest \
  python scripts/evaluate.py \
    --ref   data/plant/gff/evaluation/Juglans_regia_chr1.gff3 \
    --pred  results/Jregia_GeneCAD_final.gff \
    --fasta data/plant/fasta/evaluation/Juglans_regia_chr1.fa.gz \
    --output results/Jregia_eval_report.txt
cat results/Jregia_eval_report.txt
```

Expected results (Section 1 CDS-based, Locus-level):

```
--- Locus-level ---
  Precision : 0.7571
  Recall    : 0.7120
  F1        : 0.7338
```
