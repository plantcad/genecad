<div align="center">

# GeneCAD: Plant Genome Annotation with a DNA Foundation Model

![](https://img.shields.io/badge/version-0.1.0-blue)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/plantcad/genecad/actions/workflows/ci.yaml/badge.svg)](https://github.com/plantcad/genecad/actions/workflows/ci.yaml)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.1101/2025.10.31.685877-b31b1b.svg)](https://doi.org/10.1101/2025.10.31.685877)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?logo=python&logoColor=white)](https://www.python.org)
[![Release Wheel](https://img.shields.io/badge/Install-GitHub%20Release%20Wheel-orange)](#step-1-download-and-install)
[![Docker](https://img.shields.io/badge/Install-Docker-blue?logo=docker)](#using-docker)
[![GeneCAD Downloads](https://img.shields.io/github/downloads/plantcad/genecad/total?label=GitHub%20downloads)](https://github.com/plantcad/genecad/releases)
[![HF Downloads](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fplantcad%2Fgenecad_plant&query=%24.downloads&label=HF%20downloads&color=yellow&logo=huggingface)](https://huggingface.co/plantcad/genecad_plant)

GeneCAD is an end-to-end genome annotation pipeline for plants, powered by the DNA foundation model [PlantCAD2](https://doi.org/10.1101/2025.10.31.685877). Give it a genome FASTA and it returns a publication-ready GFF3 annotation. No species-specific training data, no external alignments, no configuration required.

*Any species. Any genome size. Out-of-the-box multi-GPU scalability.*

</div>

Unlike traditional annotation tools that rely on hand-crafted features or splice-site grammars, GeneCAD learns gene structure directly from sequence using a pretrained transformer encoder followed by a Viterbi decoder and protein-level refinement via [ReelProtein](https://onlinelibrary.wiley.com/doi/10.1111/tpj.70483). Both plant (`-m plant`, default) and vertebrate (`-m animal`) genomes are supported.

## Contents

- [🚀 Quick Start](#-quick-start)
- [Inference Guide](#inference-guide)
  - [CLI Reference](#cli-reference)
  - [Common run recipes](#common-run-recipes)
  - [Multi-GPU acceleration](#multi-gpu-acceleration)
  - [Outputs](#outputs)
  - [Throughput](#throughput)
- [Advanced Setup & Execution](#advanced-setup--execution)
  - [Containers (Singularity / Apptainer)](#containers-singularity--apptainer)
  - [Install from Source (Using uv)](#install-from-source-using-uv)
  - [Containers (Docker)](#containers-docker)
  - [HPC Clusters (SLURM)](#hpc-clusters-slurm)
  - [Cloud Provisioning (SkyPilot)](#cloud-provisioning-skypilot)
- [Model Weights & Data](#model-weights--data)
- [Advanced Usage](#advanced-usage)
  - [Evaluation](#evaluation)
  - [Training](#training)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Development](#development)

---

## 🚀 Quick Start

We provide a **Command Line Interface (CLI)** for automated pipelines and local usage.

### Prerequisites

| Requirement | Minimum | Recommended | Notes |
|-------------|---------|-------------|-------|
| OS | Linux (x86-64) | — | macOS is not supported (no CUDA) |
| GPU | NVIDIA ≥ 16 GB VRAM | A100 / H100 | e.g. RTX 3090/4090 for development |
| CUDA | 12.4 | 12.8 | Must match PyTorch 2.7.1 build |
| Python | 3.11 | 3.12 | Managed automatically by `uv` |
| Disk | ~20 GB free | — | Model weights cached in `~/.cache/huggingface` |

### Step 1: Download and Install

Open your terminal application and run these commands to download and install GeneCAD. You can copy and paste the entire block:

```bash
# 1. Install 'uv' (a fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Download the GeneCAD repository
git clone https://github.com/plantcad/genecad.git
cd genecad

# 3. Create a virtual environment and install GeneCAD
uv venv
source .venv/bin/activate   # run this every time you open a new terminal
bash scripts/install_release.sh
```

> [!NOTE]
> `source .venv/bin/activate` must be run **every time you open a new terminal** before using `genecad`. If you see `genecad: command not found`, this is almost always the cause. See [Troubleshooting](#troubleshooting) for a permanent fix.

### Step 2: Run an annotation

**Run a test example (Arabidopsis):**
No configuration required — the example *Arabidopsis thaliana* TAIR12 sequence is downloaded automatically.
```bash
genecad predict
```

**Annotate your own genome:**

```bash
genecad predict \
  -i /path/to/my_genome.fa \
  -o /path/to/output_dir \
  -s MySpecies \
  -m plant
```

> [!NOTE]
> * **`-s MySpecies`**: Sets the prefix for your output files (e.g., `MySpecies_GeneCAD_final.gff`).
> * **`-m plant`**: Sets the model to use. Use `-m animal` for vertebrates.


**Use all available GPUs** to speed up processing (significantly faster on large genomes):
```bash
genecad predict --gpus all
```

> [!TIP]
> **Output Files**
> GeneCAD produces two **GFF3** annotation files in your output directory (e.g., `genecad_result/Athaliana_predictions/`):
>
> | File | Description |
> |------|-------------|
> | `[Species]_GeneCAD_raw.gff` | Raw model predictions (all chromosomes merged) |
> | `[Species]_GeneCAD_final.gff` | Protein-refined, publication-ready annotations — **use this one** |
>
> Both files follow the standard GFF3 format and can be loaded directly into **IGV**, **JBrowse2**, or **Apollo** alongside your genome FASTA.

---

## Inference Guide

### CLI Reference

The primary entry point is the command line tool `genecad predict`. It discovers all chromosomes in the input FASTA automatically and runs the complete pipeline on each one.

```
Usage: genecad predict [OPTIONS]

Options:
  -i, --input PATH      Whole-genome nucleotide FASTA (.fa / .fa.gz) to annotate.
                        Must contain chromosome or scaffold sequences — not proteins or reads.
                        Default: downloads Arabidopsis thaliana TAIR12 example
  -o, --output DIR      Output directory  (default: genecad_result/Athaliana_predictions)
  -s, --species NAME    A label for this run (default: Athaliana). Used as the output
                        file prefix (NAME_GeneCAD_raw.gff, NAME_GeneCAD_final.gff) and
                        as an internal data key in intermediate files. Any consistent
                        label works — it does not affect the model or annotation results.
  -m, --mode MODE       Model to use: plant | animal  (default: plant)
  -n, --top-n-contigs N Predict only the N longest FASTA sequences
                        (default: all sequences)
  -l, --min-transcript-length N  Minimum transcript length in bp (default: 3)
  -c, --cpu-workers N   CPU worker processes for GFF export (default: 1)
  -b, --batch-size N    Inference batch size per GPU  (default: auto — scaled to GPU VRAM)
  -g, --gpus LIST       GPU IDs to use: comma-separated list or 'all'  (default: 0)
  --launcher CMD        Custom entrypoint to launch predict.py (e.g. 'srun python').
                        Overrides automatic DDP/SLURM detection.
                        Can also be set via the LAUNCHER environment variable.
  -h, --help            Show this help message
```

> [!IMPORTANT]
> `genecad predict` internally manages the pipeline. Files under `scripts/` (`scripts/predict.py`, `scripts/gff.py`, etc.) are internal pipeline modules called automatically — do not run them directly.

### Common run recipes

**Custom plant genome:**

```bash
genecad predict \
  -i /path/to/Zmays.fa \
  -o /path/to/output \
  -s Zmays \
  -m plant
```

**Custom animal genome:**

```bash
genecad predict \
  -i /path/to/genome.fa \
  -o /path/to/output \
  -s Hsapiens \
  -m animal
```

**Run only top N longest contigs/scaffolds** (useful for very fragmented assemblies):

```bash
genecad predict \
  -i /path/to/genome.fa \
  -o /path/to/output \
  -s MySpecies \
  -m plant \
  --top-n-contigs 100
```

When `--top-n-contigs` is used, GeneCAD selects the N longest sequences by length, then processes them in the same order they appear in the input FASTA headers. This keeps output ordering stable and easier to compare with the source assembly.

**Pin a known-safe batch size for reproducibility:**

```bash
genecad predict -i /path/to/genome.fa -s MySpecies --gpus all --batch-size 4
```

### Multi-GPU acceleration

By default, `genecad predict` uses GPU `0` and processes chromosomes sequentially. Pass `--gpus` to distribute chromosomes across multiple GPUs and process them **in parallel** — ideal for large, multi-chromosome genomes.

```bash
# Use all available GPUs (auto-detected)
genecad predict -i genome.fa -s MySpecies --gpus all

# Use specific GPUs (e.g. 0, 1, 2, 3)
genecad predict -i genome.fa -s MySpecies --gpus 0,1,2,3

# Use 2 GPUs with a fixed batch size
genecad predict -i genome.fa -s MySpecies --gpus 0,1 -b 32
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

`genecad predict` is fully **resumable** — if a run is interrupted (e.g. job timeout or OOM), simply re-run the same command. Each step checks whether its output file already exists and skips it if so. Only the remaining work will run.

To **force a step to re-run**, delete its output:

```bash
# Re-run predictions for one chromosome from scratch
rm -rf genecad_result/MySpecies_predictions/Chr1/

# Re-run only the token prediction step (Step 2)
rm -rf genecad_result/MySpecies_predictions/Chr1/pipeline/predictions.zarr
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

---

## Advanced Setup & Execution

### Containers (Singularity / Apptainer)

For HPC environments where Docker is not available, you can use Singularity (or Apptainer) to pull and run the Docker image directly. The `--nv` flag is required to enable GPU support.

> [!NOTE]
> **Understanding File Paths in Containers**
> The argument `--bind $(pwd):/workspace` connects your current folder on the host machine to the `/workspace` folder inside the container.
>
> If you run the command from `/home/user/my_project`, then inside the container, `/workspace` actually points to `/home/user/my_project`. Always place your input FASTA files inside your current directory so the container can see them!

```bash
# Clone the repository
git clone https://github.com/plantcad/genecad && cd genecad

# Pull the Docker image and convert it to a Singularity image
singularity pull genecad.sif docker://ghcr.io/plantcad/genecad_v1:latest

# Run on the bundled Arabidopsis example (auto-downloads FASTA)
singularity exec --nv \
  --bind $(pwd):/workspace \
  --pwd /workspace \
  genecad.sif \
  /usr/local/bin/genecad bash predict.sh

# Annotate a custom plant genome
singularity exec --nv \
  --bind $(pwd):/workspace \
  --pwd /workspace \
  genecad.sif \
  /usr/local/bin/genecad bash predict.sh \
    -i /workspace/data/my_plant.fa \
    -o /workspace/output \
    -s Zmays \
    -m plant

# Annotate an animal genome
singularity exec --nv \
  --bind $(pwd):/workspace \
  --pwd /workspace \
  genecad.sif \
  /usr/local/bin/genecad bash predict.sh \
    -i /workspace/data/my_animal.fa \
    -o /workspace/output \
    -s Hsapiens \
    -m animal
```

### Install from Source (Using uv)

If you are a developer and want to install GeneCAD from the source code instead of using the pre-built wheel, use [uv](https://docs.astral.sh/uv/):

```bash
# Clone the repository
git clone https://github.com/plantcad/genecad && cd genecad

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install all dependencies
# mamba and causal-conv1d build from source — this can take 3–30 minutes
uv sync --extra torch --extra mamba

# Activate the virtual environment
source .venv/bin/activate

# Run the pipeline
genecad predict
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

### Containers (Docker)

The Docker image at `ghcr.io/plantcad/genecad_v1` bundles the complete runtime. Source code is **mounted at run time**, so changes you make to the cloned repository take effect immediately — no rebuild needed.

> [!NOTE]
> **Understanding File Paths in Containers**
> The argument `-v $(pwd):/workspace` connects your current folder on the host machine to the `/workspace` folder inside the container. Always place your input FASTA files inside your current directory so the container can see them!

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

### HPC Clusters (SLURM)

SLURM is a job scheduler — you still need to install GeneCAD first using the [Quick Start](#step-1-download-and-install) or [from source](#install-from-source-using-uv) on your login node, then submit jobs with `sbatch` or `salloc`.

Load the appropriate modules for your cluster before installing (e.g. CUDA 12.8, Python 3.12). The minimum tested combination is CUDA 12.4, Python 3.11, and PyTorch 2.5.1.

**Interactive single-node job:**

```bash
salloc --partition=gpu-queue --nodes=1 --ntasks=1 --gres=gpu:1

cd /path/to/genecad
# Activate your environment if you installed via uv or venv
source .venv/bin/activate

genecad predict \
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
# Activate your environment if you installed via uv or venv
source .venv/bin/activate

genecad predict \
  -i /path/to/input.fa \
  -o /path/to/output \
  -s MySpecies \
  -m plant
EOF

sbatch run_genecad.slurm
```

> [!TIP]
> **Multi-GPU on a Single Node:** To use all GPUs on a node, request them with `--gres=gpu:4` (or however many are available) and pass `--gpus all` to `genecad predict`. If there are many chromosomes, they will be distributed across GPUs. If there are fewer chromosomes than GPUs, GeneCAD will automatically split the longest sequences into parallel windows across the GPUs.
>
> **Multi-Node Distributed Inference (e.g., TACC):** GeneCAD natively detects multi-node SLURM topologies. If you allocate multiple nodes (e.g., `#SBATCH --nodes=4`), you can use `srun` to seamlessly process enormous single contigs across the entire cluster. `genecad predict` will automatically bypass local launchers and allow PyTorch Lightning to route the distributed process windows.
>
> ```bash
> # Request multiple nodes/GPUs and launch via srun
> #SBATCH --nodes=2
> #SBATCH --ntasks=8
> #SBATCH --gres=gpu:4
> srun genecad predict -i /path/to/genome.fa -s MySpecies --gpus all
> ```
>
> **Custom launcher:** If your cluster requires a non-standard Python entrypoint (e.g. `srun python` instead of `torchrun`), use `--launcher` or the `LAUNCHER` environment variable to override automatic detection:
>
> ```bash
> # Via flag
> genecad predict -i genome.fa -s MySpecies --launcher 'srun python'
>
> # Via environment variable
> LAUNCHER='srun python' genecad predict -i genome.fa -s MySpecies
> ```

### Cloud Provisioning (SkyPilot)

[SkyPilot](https://docs.skypilot.co/en/latest/docs/index.html) lets you provision on-demand cloud GPUs across many providers (AWS, GCP, Lambda, RunPod, etc.) without managing infrastructure manually. This is useful if you do not have access to a local GPU or HPC cluster.

> [!IMPORTANT]
> Run these commands from your **local machine or a login node with internet access** — not from an HPC compute node, which typically has no outbound internet and cannot reach cloud provider APIs.

**Step 1 — Install SkyPilot** (in a dedicated environment, separate from the GeneCAD venv to avoid dependency conflicts):

```bash
python3 -m venv ~/skypilot-env
source ~/skypilot-env/bin/activate
pip install "skypilot[lambda]>=0.10.3"
```

**Step 2 — Start the API server and configure credentials:**

```bash
# Start the SkyPilot API server (runs as a background process)
sky api start

# Verify credentials for your cloud provider
sky check lambda
```

If `sky check lambda` shows `Lambda: disabled`, configure your API key:
1. Generate a key at [https://cloud.lambdalabs.com/api-keys](https://cloud.lambdalabs.com/api-keys)
2. Add it to `~/.lambda_cloud/lambda_keys`:

```bash
mkdir -p ~/.lambda_cloud
echo "api_key = YOUR_API_KEY_HERE" > ~/.lambda_cloud/lambda_keys
sky check lambda   # should now show: Lambda: enabled ✓
```

**Step 3 — Launch a GPU node and run:**

*(Make sure your terminal is currently in the root of the cloned `genecad` repository)*

> [!NOTE]
> **Annotating your own genome:** When you are ready to annotate your own data, place your FASTA file inside the `genecad` directory (e.g., `genecad/my_genome.fa`) before running `sky launch`. This ensures SkyPilot automatically uploads it to the cloud. You would then append `-i my_genome.fa -s MySpecies` to the `bash predict.sh` command below.

```bash
# 1. Deploy the GPU node (--no-setup: Docker replaces uv sync)
sky launch --num-nodes 1 --yes --no-setup \
  --cluster genecad examples/configs/cluster.sky.yaml

# 2. Run the workload (this example runs the built-in Arabidopsis test)
sky exec genecad 'docker pull ghcr.io/plantcad/genecad_v1:latest && \
                  docker run --rm --gpus all \
                  -v $(pwd):/workspace -w /workspace \
                  ghcr.io/plantcad/genecad_v1:latest \
                  bash predict.sh'

# 3. Download the results back to your local machine
rsync -avz genecad:~/sky_workdir/genecad_result/ ./genecad_result/

# 4. Terminate the node to stop billing
sky down genecad --yes
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

## Model Weights & Data

### Available models

GeneCAD provides two pre-trained models for different taxonomic groups. Both are downloaded automatically from Hugging Face on first run and cached in `~/.cache/huggingface`. Internet access is required on the first run.

| Mode (`-m`) | Organism type | Base model | GeneCAD head |
|-------------|--------------|------------|--------------|
| `plant` *(default)* | Plants | [`emarro/pcad2-200M-cnet-baseline`](https://huggingface.co/emarro/pcad2-200M-cnet-baseline) | [`plantcad/genecad_plant`](https://huggingface.co/plantcad/genecad_plant) |
| `animal` | Animals / vertebrates | [`emarro/pcad2_vert_small`](https://huggingface.co/emarro/pcad2_vert_small) | [`plantcad/genecad_vert`](https://huggingface.co/plantcad/genecad_vert) |

**Pre-downloading models** (recommended for clusters without internet on compute nodes):

```bash
# Log in to Hugging Face once (required to access the models)
huggingface-cli login

# Download the plant model (~5 GB, cached in ~/.cache/huggingface)
huggingface-cli download plantcad/genecad_plant
huggingface-cli download emarro/pcad2-200M-cnet-baseline

# Download the animal model (~3 GB)
huggingface-cli download plantcad/genecad_vert
huggingface-cli download emarro/pcad2_vert_small
```

Run these on the login node (which has internet access). When you then run `genecad predict` on a compute node, the weights are loaded from the local cache without any download.

---

## Advanced Usage

### Evaluation

GeneCAD includes a built-in evaluation tool that produces a structured five-section report comparing predicted annotations against a reference. It is self-contained for Sections 1, 2, 3, and 5. Section 4 uses BUSCO if you enable it, but BUSCO is optional.

**Native installation (uv / pip)**

```bash
genecad evaluate \
  --ref   /path/to/reference.gff3 \
  --pred  /path/to/Athaliana_GeneCAD_final.gff \
  --fasta /path/to/genome.fa \
  --output report.txt
```

**Using Docker**

Docker automatically uses the environment wrapper. Since the `genecad` command might not be pre-installed in the container's path, call the Python script directly. Note how the paths use `/workspace`, which is mapped to your current directory:

```bash
docker run --rm \
  -v $(pwd):/workspace -w /workspace \
  ghcr.io/plantcad/genecad_v1:latest \
  python src/cli.py evaluate \
    --ref /workspace/reference.gff3 \
    --pred /workspace/Athaliana_GeneCAD_final.gff \
    --fasta /workspace/genome.fa \
    --output report.txt
```

**Using Singularity**

Singularity requires you to invoke the wrapper `/usr/local/bin/genecad` first. Since the `genecad` command might not be pre-installed in the container's path, call the Python script directly. Note how the paths use `/workspace`, which is mapped to your current directory:

```bash
singularity exec \
  --bind $(pwd):/workspace --pwd /workspace \
  genecad.sif \
  /usr/local/bin/genecad python src/cli.py evaluate \
    --ref /workspace/reference.gff3 \
    --pred /workspace/Athaliana_GeneCAD_final.gff \
    --fasta /workspace/genome.fa \
    --output report.txt
```

Most users only need these arguments:

| Option | Description |
|--------|-------------|
| `--ref` | Reference GFF3 annotation (required) |
| `--pred` | Predicted GFF3 annotation (required) |
| `--fasta` | Genome FASTA; enables Sections 3 and 4 |
| `--output` | Write the report to a file instead of stdout |
| `--lineage` | BUSCO lineage dataset (default: `embryophyta_odb10`) |
| `--cpu` | Number of CPU threads for BUSCO (default: `32`) |
| `--skip-busco` | Skip Section 4 for a faster, fully portable run |
| `--busco-out` | BUSCO output directory name (default: `busco_eval`) |
| `--fix-busco-env` | Auto-repair a broken conda env and retry once |

**BUSCO setup**

If BUSCO is available on your system, GeneCAD can pick it up automatically. If not, choose the option that matches your environment:

| Environment | Recommended setting |
|-------------|---------------------|
| Local Conda / Mamba | `export BUSCO_CMD='conda run -n busco-5.5.0 busco'` |
| HPC with a site activate script | `export BUSCO_ACTIVATE_SCRIPT=/programs/miniconda3/bin/activate` |
| Already activated BUSCO env | Run `busco --version` first, then invoke evaluation normally |
| No BUSCO required | Add `--skip-busco` |

For finer control, the following flags can be passed directly to `genecad evaluate`:

| Option | Description |
|--------|-------------|
| `--busco-env` | Conda environment name for BUSCO (default: `busco-5.5.0`) |
| `--busco-cmd` | Explicit BUSCO launcher command (overrides environment detection) |
| `--busco-activate-script` | Path to a cluster-specific conda activation script |

```bash
# Portable full evaluation with an explicit BUSCO command
export BUSCO_CMD='conda run -n busco-5.5.0 busco'
genecad evaluate \
  --ref /path/to/reference.gff3 \
  --pred /path/to/prediction.gff3 \
  --fasta /path/to/genome.fa \
  --output report.txt
```

If your cluster uses a pre-installed BUSCO module or activate script, set the script path once and keep using the same evaluation command:

```bash
export BUSCO_ACTIVATE_SCRIPT=/programs/miniconda3/bin/activate
genecad evaluate \
  --ref /path/to/reference.gff3 \
  --pred /path/to/prediction.gff3 \
  --fasta /path/to/genome.fa \
  --output report.txt
```

For a quick debug run, skip BUSCO entirely:

```bash
genecad evaluate \
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

**Example output (*Arabidopsis thaliana*)**

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

**Metric guide**

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

### Training

This section is for researchers who want to **fine-tune GeneCAD on new species** or reproduce the published models. If you only want to annotate a genome, skip to [Inference Guide](#inference-guide) — no training is needed.

**Prerequisites**

| Requirement | Notes |
|-------------|-------|
| ≥ 2 × NVIDIA GPUs | Training uses PyTorch DDP. Single-GPU works but is very slow. |
| HuggingFace account | Run `huggingface-cli login` to download training data. |
| WandB (optional) | Metrics are logged to WandB. Disable with `WANDB_MODE=disabled`. |

**Native installation (uv / pip)**

```bash
# Login to HuggingFace (one-time)
huggingface-cli login

# Fine-tune on the 5-species plant dataset (downloads data automatically)
genecad train --domain plant

# Fine-tune on the vertebrate animal dataset (downloads data automatically)
genecad train --domain animal

# Use all detected GPUs
genecad train --domain animal -g all

# Use a specific subset of GPUs
genecad train --domain animal -g 0,2,3

# Fine-tune with custom settings
#   -g 4                number of GPUs
#   -b 4                per-GPU batch size
#   --epochs 3          training epochs
#   -l 1e-4             learning rate
#   -o results/my_run   output directory
#   -r my-run-name      WandB run name
genecad train \
  --domain plant \
  -g 4 \
  -b 4 \
  --epochs 3 \
  -l 1e-4 \
  -o results/my_run \
  -r my-run-name
```

**Using Docker**

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace -w /workspace \
  ghcr.io/plantcad/genecad_v1:latest \
  bash train.sh --domain plant -g all
```

**Using Singularity**

```bash
singularity exec --nv \
  --bind $(pwd):/workspace --pwd /workspace \
  genecad.sif \
  /usr/local/bin/genecad bash train.sh --domain plant -g all
```

**Options**

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
  --gff-parallel N        Parallel workers for Step 2 GFF extraction (default: 1)
  --window-size N         Training window size for sample generation (default: 8192)
  --intergenic-proportion P  Intergenic window ratio in [0,1] for sample generation (default: 0.15)
  --base-frozen yes|no    Freeze base encoder during training (default: domain preset)
  --auto-class-weights yes|no Compute class weights from train set before training (default: yes)
  --label-qc yes|no       Run split/label QC before training (default: yes)
  --min-non-intergenic-ratio P  Fail if train non-intergenic token ratio < P (default: 0.01)
  --min-core-class-tokens N  Fail if core classes have too few tokens (default: 100)
  --hf-upload-repo ID     Upload artifacts to Hugging Face repo (optional)
  --hf-upload-type TYPE   Model or dataset repo type (default: model)
  --hf-upload-path PATH   Path inside HF repo (default: run name)
  -h, --help            Show this message
```

Notes:
- `-g all` requires `nvidia-smi` to auto-detect GPU IDs.
- When using a list (for example `-g 0,2,3`), training is restricted to that subset via `CUDA_VISIBLE_DEVICES`.

**Pipeline steps**

`genecad train` is fully resumable — each step checks for its output and skips if already done.

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

**Outputs**

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

To use a trained checkpoint for inference, pass it to `genecad predict`:

```bash
genecad predict \
  -i genome.fa \
  -s MySpecies \
  -m plant \
  --model-checkpoint results/my_run/checkpoints/epoch=0-step=5000.ckpt
```

**Summarizing training data**

To inspect the class balances, masking rates, and terminal codon frequencies of your training or validation splits, you can use the built-in summary tool on your compiled `zarr` datasets:

```bash
genecad summarize summarize_training_dataset \
  --input genecad_result/training/plant/pipeline/prep/splits/train.zarr
```
This utility provides key statistics to ensure your model gradients remain stable across custom datasets and species variations.

---

## Troubleshooting

### Installation

**`mamba-ssm` or `causal-conv1d` build fails**

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

**`flash-attn` build fails**

Same root cause as above. Docker is the safest path if you cannot match the exact CUDA/GCC versions.

**`genecad: command not found`**

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

```bash
export PATH="$HOME/.local/bin:$PATH"
# Make it permanent:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Wrong Python version**

GeneCAD requires Python 3.12 exactly. Check with `python --version`. If you are using `uv`, the correct version is managed automatically via `uv venv`.

---

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

**GPU utilization is low / inference is slow**

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

---

### Import / Module Errors

**`ModuleNotFoundError: No module named 'mamba_ssm'`**

```bash
bash scripts/install_release.sh   # re-runs the mamba install step
```

If that fails, switch to Docker (see above).

**`ModuleNotFoundError: No module named 'flash_attn'`**

Same fix — re-run `install_release.sh`. `flash-attn` builds from source and can fail silently on mismatched environments.

**`ModuleNotFoundError: No module named 'src'`**

Run `genecad` from inside the cloned repository directory, or ensure `PYTHONPATH` includes the repo root:

```bash
cd /path/to/genecad
genecad predict ...
```

---

### Inference

**Run was interrupted (job timeout, manual Ctrl-C, OOM)**

Re-run the exact same command. GeneCAD checks which steps have already written their output and skips them — only the remaining work runs.

```bash
genecad predict -i genome.fa -s MySpecies   # safe to re-run
```

To force a chromosome to restart from scratch, delete its directory:

```bash
rm -rf genecad_result/MySpecies_predictions/Chr1/
```

**Assembly has thousands of small scaffolds and the run is very slow**

Use `--top-n-contigs` to process only the longest sequences, which contain most of the genes:

```bash
genecad predict -i genome.fa -s MySpecies --top-n-contigs 50
```

**Which `-m` mode should I use?**

| Organism type | Flag |
|---------------|------|
| Land plants (angiosperms, gymnosperms, mosses, ferns) | `-m plant` (default) |
| Algae | `-m plant` |
| Vertebrates (fish, amphibians, reptiles, birds, mammals) | `-m animal` |
| Invertebrates | `-m animal` (experimental) |

---

### HuggingFace / Network

**`GatedRepoError` or `401 Unauthorized` when downloading models**

The model weights require a HuggingFace account. Log in once:

```bash
huggingface-cli login
```

**Slow or failed model download**

Model weights (~5 GB) are cached in `~/.cache/huggingface` after the first download. If you are on a cluster without internet access, download the weights on a login node first, then run inference on a compute node — the cache is reused automatically.

**Training data download fails (`huggingface-cli download` error)**

```bash
huggingface-cli login   # ensure you are authenticated
genecad train --domain plant   # re-run; completed steps are skipped
```

---

### SkyPilot

**`sky api start` fails with `RuntimeError: Failed to start SkyPilot API server`**

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

**`sky check lambda` shows `Lambda: disabled`**

You need to configure your Lambda API key. Generate one at [https://cloud.lambdalabs.com/api-keys](https://cloud.lambdalabs.com/api-keys), then:

```bash
mkdir -p ~/.lambda_cloud
echo "api_key = YOUR_API_KEY_HERE" > ~/.lambda_cloud/lambda_keys
sky check lambda   # should now show: Lambda: enabled ✓
```

**`sky launch` fails with connection errors on an HPC compute node**

HPC compute nodes typically have no outbound internet access and cannot reach cloud provider APIs. SkyPilot must be run from your **local machine** or the cluster's **login node** (which usually has internet). Check with your sysadmin if you are unsure.

**`bash predict.sh: No such file or directory` during manual debugging**

If you use `ssh genecad` to manually interact with the node, you will land in the home directory (`~`). SkyPilot syncs your local repo to `~/sky_workdir/` — you must `cd` there before running Docker or local scripts:

```bash
ssh genecad
cd ~/sky_workdir   # ← the repo and predict.sh are here
docker run --rm --gpus all \
  -v $(pwd):/workspace -w /workspace \
  ghcr.io/plantcad/genecad_v1:latest \
  bash predict.sh
```

**Custom input FASTA is "not found" during `sky exec`**

If you placed your custom FASTA inside the `genecad/data/` directory, SkyPilot did not upload it. SkyPilot automatically respects the repository's `.gitignore` file, which excludes the `data/` folder to prevent uploading massive files.

**Fix:** Move your FASTA to the root of the repository (e.g., `genecad/my_genome.fa`) so it isn't ignored, and re-run `sky launch`.

**SkyPilot conflicts with GeneCAD dependencies**

Install SkyPilot in a separate virtual environment, not inside the GeneCAD `.venv`:

```bash
python3 -m venv ~/skypilot-env
source ~/skypilot-env/bin/activate
pip install "skypilot[lambda]>=0.10.3"
```

---

### No GPU Available

If you do not have access to a local GPU, you have two options:

**Option 1 — Cloud GPU via SkyPilot**:
Follow the complete end-to-end instructions in the [Cloud Provisioning](#cloud-provisioning-skypilot) section to deploy a node, run the workload, and securely download your results.

**Option 2 — HPC cluster via SLURM** (see [HPC Clusters](#hpc-clusters-slurm)):

```bash
salloc --partition=gpu-queue --nodes=1 --gres=gpu:1
genecad predict -i genome.fa -s MySpecies
```

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
docker build --progress=plain --no-cache -t genecad:v0.1.0 .

# Test the build — runs the full pipeline on the Arabidopsis example
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace genecad:v0.1.0 \
  bash predict.sh

# Publish to GitHub Container Registry
# Requires a personal access token with "write:packages" stored in GHCR_TOKEN
IMAGE=ghcr.io/plantcad/genecad_v1
docker tag genecad:v0.1.0 $IMAGE:v0.1.0
docker tag genecad:v0.1.0 $IMAGE:latest
echo $GHCR_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
docker push $IMAGE:v0.1.0
docker push $IMAGE:latest
```

### Reproduction

To reproduce the published GeneCAD results for *Juglans regia* (Walnut) chromosome 1:

```bash
mkdir -p data results

# Download FASTA and reference GFF from Hugging Face
huggingface-cli download plantcad/genecad-dev \
  data/plant/fasta/evaluation/Juglans_regia_chr1.fa.gz \
  --repo-type dataset --local-dir .
huggingface-cli download plantcad/genecad-dev \
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
  genecad evaluate \
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
