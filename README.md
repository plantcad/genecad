![](https://img.shields.io/badge/version-1.0.0-blue)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/plantcad/genecad/actions/workflows/ci.yaml/badge.svg)](https://github.com/plantcad/genecad/actions/workflows/ci.yaml)
[![DOI](https://zenodo.org/badge/DOI/10.1101/2025.10.31.685877.svg)](https://doi.org/10.1101/2025.10.31.685877)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg?style=flat)](https://huggingface.co/collections/plantcad/genecad-68c686ccf14312bf6de356de)
[![Container](https://img.shields.io/badge/container-ghcr.io%2Fplantcad%2Fgenecad-blue?logo=github)](https://github.com/orgs/plantcad/packages/container/package/genecad)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)

# GeneCAD

GeneCAD is an end-to-end genome annotation pipeline for plants and animals, powered by the DNA foundation model [PlantCAD2](https://doi.org/10.1101/2025.10.31.685877). Unlike traditional annotation tools that rely on hand-crafted features or splice-site grammars, GeneCAD learns gene structure directly from sequence using a pretrained transformer encoder followed by a Viterbi decoder and protein-level refinement via [ReelProtein](https://onlinelibrary.wiley.com/doi/10.1111/tpj.70483). It runs on a single GPU and requires no species-specific training data or external alignments.

## Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [Using Docker](#using-docker)
  - [Using uv](#using-uv)
  - [Using SLURM](#using-slurm)
  - [Using SkyPilot](#using-skypilot)
- [Inference](#inference)
  - [Available models](#available-models)
  - [Running the pipeline](#running-the-pipeline)
  - [Pipeline steps](#pipeline-steps)
  - [Outputs](#outputs)
  - [Throughput](#throughput)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Development](#development)
  - [Docker](#docker)
  - [Reproduction](#reproduction)

---

## Quick Start

Annotate a full plant genome in four commands. No configuration required — the example *Arabidopsis thaliana* TAIR12 sequence is downloaded automatically.

```bash
# 1. Clone the repository
git clone https://github.com/plantcad/genecad && cd genecad

# 2. Install uv, then reload your shell PATH
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 3. Install Python dependencies (mamba build takes 3–30 min)
uv sync --extra torch --extra mamba

# 4. Run the full prediction pipeline
bash predict.sh
```

> **Output:** Two GFF3 files are created in `genecad_result/Athaliana_predictions/`:
> | File | Description |
> |------|-------------|
> | `Athaliana_GeneCAD_raw.gff` | Raw model predictions |
> | `Athaliana_GeneCAD_final.gff` | Protein-refined, publication-ready annotations |

To annotate your own genome, pass your FASTA file, output directory, species name, and model mode:

```bash
bash predict.sh \
  -i /path/to/my_genome.fa \
  -o /path/to/output_dir \
  -s MySpecies \
  -m plant   # or: -m animal  for vertebrate genomes
```

---

## Prerequisites

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| OS | Linux (x86-64) | macOS is not supported (no CUDA) |
| GPU | NVIDIA GPU with ≥ 16 GB VRAM | e.g. A100, H100, RTX 3090/4090 |
| CUDA | 12.4 | 12.8 recommended; matches PyTorch 2.7.1 |
| Python | 3.11 | 3.12 recommended |
| Disk | ~20 GB free | Models are cached in `~/.cache/huggingface` |

Model weights (~800 MB per model) are downloaded automatically from Hugging Face on first run. Internet access is required unless weights are pre-cached.

---

## Setup

Choose the installation method that best fits your environment:

| Method | Best for |
|--------|----------|
| [Docker](#using-docker) | Reproducible runs, no environment management |
| [uv](#using-uv) | Local development, interactive use |
| [SLURM](#using-slurm) | HPC/supercomputer clusters |
| [SkyPilot](#using-skypilot) | On-demand cloud GPUs |

### Using Docker

The Docker image at `ghcr.io/plantcad/genecad` contains the full runtime environment. Source code is **mounted at run time**, so no rebuild is needed when you update the repository.

```bash
# Clone the GeneCAD repository
git clone https://github.com/plantcad/genecad && cd genecad

# Pull the image
docker pull ghcr.io/plantcad/genecad:latest

# Run the full prediction pipeline on Arabidopsis (auto-downloads example FASTA)
docker run --rm --gpus all \
  -v $(pwd):/workspace -w /workspace \
  ghcr.io/plantcad/genecad:latest \
  bash predict.sh

# Annotate a custom plant genome
docker run --rm --gpus all \
  -v $(pwd):/workspace -w /workspace \
  ghcr.io/plantcad/genecad:latest \
  bash predict.sh \
    -i /workspace/data/my_plant.fa \
    -o /workspace/output \
    -s Zmays \
    -m plant

# Annotate an animal genome
docker run --rm --gpus all \
  -v $(pwd):/workspace -w /workspace \
  ghcr.io/plantcad/genecad:latest \
  bash predict.sh \
    -i /workspace/data/my_animal.fa \
    -o /workspace/output \
    -s Hsapiens \
    -m animal
```

### Using uv

```bash
# Clone the repository
git clone https://github.com/plantcad/genecad && cd genecad

# Install uv, then reload your shell PATH
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install all dependencies
# mamba and causal-conv1d build from source — this can take 3–30 minutes
uv sync --extra torch --extra mamba

# Run the pipeline
bash predict.sh
```

PyTorch is pinned to 2.7.1 (CUDA 12.8) for compatibility with `mamba` (2.2.4) and `causal-conv1d` (1.5.0.post8). Newer combinations may work but are not officially tested.

`mamba` and `causal-conv1d` build from source without build isolation for reliability. To cache the built wheels and speed up future installs, add these entries to `pyproject.toml`:

```toml
[tool.uv.sources]
mamba-ssm = { path = "path/to/mamba_ssm-2.2.4-*.whl" }
causal-conv1d = { path = "path/to/causal_conv1d-1.5.0.post8-*.whl" }
# OR use a remote URL:
# mamba-ssm = { url = "https://.../mamba_ssm-2.2.4-*.whl" }
```

### Using SLURM

Follow the [uv](#using-uv) instructions to create a virtual environment on your cluster first. Load the appropriate modules for CUDA 12.8 and Python 3.12 (minimum tested: CUDA 12.4, Python 3.11, PyTorch 2.5.1).

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

> **Note:** Only one GPU per node is currently supported. For very large genomes, each chromosome is processed sequentially. To parallelise across chromosomes, split your FASTA by chromosome and submit one job per chromosome.

### Using SkyPilot

[SkyPilot](https://docs.skypilot.co/en/latest/docs/index.html) allows you to provision on-demand cloud GPUs across many providers (AWS, GCP, Lambda, RunPod, etc.) without managing infrastructure manually. This is useful if you do not have access to a local GPU or HPC cluster.

```bash
# Install SkyPilot (adjust the cloud extra as needed: lambda, aws, gcp, etc.)
pip install "skypilot[lambda]>=0.10.3"

# Clear local SkyPilot state, if any
sky api stop; [ -d ~/.sky ] && rm -rf ~/.sky

# Deploy a single GPU node
sky launch --num-nodes 1 --yes --no-setup \
  --cluster genecad examples/configs/cluster.sky.yaml

# SSH into the node, then install and run
ssh genecad
git clone https://github.com/plantcad/genecad && cd genecad
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv sync --extra torch --extra mamba
bash predict.sh

# Terminate the node when done
sky down genecad
```

See the [Throughput](#throughput) section for GPU cost comparisons across providers.

<details><summary>SkyPilot GPU rates</summary>

```
> sky show-gpus L4:1 # low-end development GPU
GPU  QTY  CLOUD   INSTANCE_TYPE  DEVICE_MEM  vCPUs  HOST_MEM  HOURLY_PRICE  REGION
L4   1.0  RunPod  1x_L4_SECURE   -           4      24GB      $ 0.440       CA
L4   1.0  GCP     g2-standard-4  24GB        4      16GB      $ 0.705       us-east4
L4   1.0  AWS     g6.xlarge      22GB        4      16GB      $ 0.805       us-east-2

> sky show-gpus H100:1 # high-end production GPU
GPU   QTY  CLOUD       INSTANCE_TYPE             DEVICE_MEM  HOURLY_PRICE  REGION
H100  1.0  Hyperbolic  1x-H100-75-722            80GB        $ 1.290       default
H100  1.0  Lambda      gpu_1x_h100_pcie          80GB        $ 2.490       europe-central-1
H100  1.0  GCP         a3-highgpu-1g             80GB        $ 5.383       us-central1
```

</details>

---

## Inference

### Available models

GeneCAD provides two pre-trained models for different taxonomic groups. Both are downloaded automatically from Hugging Face on first run.

| Mode (`-m`) | Organism type | Base model | GeneCAD head |
|-------------|--------------|------------|--------------|
| `plant` *(default)* | Plants | [`plantcad/pcad2-200M-cnet-baseline`](https://huggingface.co/plantcad/pcad2-200M-cnet-baseline) | [`Zong-Yan/genecad_plant`](https://huggingface.co/Zong-Yan/genecad_plant) |
| `animal` | Animals / vertebrates | [`emarro/pcad2_vert_small`](https://huggingface.co/emarro/pcad2_vert_small) | [`Zong-Yan/genecad_vert`](https://huggingface.co/Zong-Yan/genecad_vert) |

### Running the pipeline

The primary entry point is `predict.sh`. It discovers all chromosomes in your FASTA file automatically and runs the complete pipeline on each one.

```
Usage: predict.sh [OPTIONS]

Options:
  -i, --input PATH    Genome FASTA file to annotate
                      Default: downloads Arabidopsis thaliana TAIR12 example
  -o, --output DIR    Output directory  (default: genecad_result/Athaliana_predictions)
  -s, --species NAME  Species label prefixed on output filenames  (default: Athaliana)
  -m, --mode MODE     Model: plant | animal  (default: plant)
  -h, --help          Show this help message
```

**Plant genome (auto-downloads Arabidopsis TAIR12 example):**
```bash
bash predict.sh
```

**Custom plant genome:**
```bash
bash predict.sh \
  -i /path/to/Zmays.fa \
  -o /path/to/output \
  -s Zmays \
  -m plant
```

**Animal genome:**
```bash
bash predict.sh \
  -i /path/to/genome.fa \
  -o /path/to/output \
  -s Hsapiens \
  -m animal
```

### Pipeline steps

The script automatically runs the following steps for every chromosome in the input FASTA:

| Step | Tool | Description |
|------|------|-------------|
| 1. Extract | `scripts/extract.py` | Parse FASTA and tokenize sequences into Zarr format |
| 2. Predict | `scripts/predict.py` | Windowed inference with the GeneCAD classifier (GPU) |
| 3. Decode | `scripts/predict.py` | Viterbi decoding of per-token logits into genomic intervals |
| 4. Export | `scripts/predict.py` | Convert intervals to GFF3 |
| 5. Filter | `scripts/gff.py` | Remove short or structurally invalid gene models |
| 6. Merge | `scripts/merge_gff.py` | Concatenate per-chromosome GFFs; prefix gene IDs with chromosome name |
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

### Throughput

Observed GeneCAD inference throughput on different GPUs. Datacenter-class GPUs (A100/H100) are 3–6× more cost-efficient per megabase than development GPUs.

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

## Evaluation

GeneCAD includes a built-in evaluation tool (`src/evaluate.py`) that produces a structured five-section report. It does not require `gffcompare` or any other external tool beyond an optional BUSCO install for Section 4.

### Usage

```bash
uv run python src/evaluate.py \
  --ref   /path/to/reference.gff3 \
  --pred  /path/to/Athaliana_GeneCAD_final.gff \
  --fasta /path/to/genome.fa \
  --output report.txt
```

```
Options:
  --ref        Reference GFF3 annotation               (required)
  --pred       Predicted GFF3 annotation               (required)
  --fasta      Genome FASTA — enables sections 3 and 4
  --lineage    BUSCO lineage dataset  (default: embryophyta_odb10)
  --cpu        CPU threads for BUSCO (default: 32)
  --output     Write report to file instead of stdout
```

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
| 1 | Locus | Gene correct if any isoform's CDS chain matches a reference |
| 1 | Transcript | Exact full-isoform CDS chain match |
| 2 | Base | Nucleotide-level overlap of predicted vs. reference exons |
| 2 | Intron | Exact splice junction pair match |
| 2 | Intron chain | Complete intron chain match (= transcript interior) |
| 3 | Splice sites | GT-AG / GC-AG canonical intron dinucleotide frequency |
| 4 | BUSCO | Benchmarked Universal Single-Copy Orthologs completeness |
| 5 | TIS / TTS | Translation start / stop site accuracy |
| 5 | Donor / Acceptor | Individual splice site accuracy |

> **Note:** Sections 3 and 4 require `--fasta`. BUSCO additionally requires `busco` in PATH, or a conda environment named `busco-5.5.0`.

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
# Build the image (on a Linux machine with Docker and NVIDIA drivers)
sudo usermod -aG docker ubuntu && newgrp docker
docker build --progress=plain --no-cache -t genecad:v1.0.1 .

# Test the build — run the full pipeline on the Arabidopsis example
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace genecad:v1.0.1 \
  bash predict.sh

# Publish to GitHub Container Registry
IMAGE=ghcr.io/plantcad/genecad
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
hf download plantcad/genecad-dev \
  data/fasta/evaluation/Juglans_regia_chr1.fa.gz \
  --repo-type dataset --local-dir .
hf download plantcad/genecad-dev \
  data/gff/evaluation/Juglans_regia_chr1.gff3 \
  --repo-type dataset --local-dir .

# Run the full GeneCAD pipeline
docker run --rm --gpus all \
  -v $(pwd):/workspace -w /workspace \
  ghcr.io/plantcad/genecad:latest \
  bash predict.sh \
    -i data/fasta/evaluation/Juglans_regia_chr1.fa.gz \
    -o results \
    -s Jregia \
    -m plant

# Evaluate against the reference annotation (CPU-only)
docker run --rm \
  -v $(pwd):/workspace -w /workspace \
  ghcr.io/plantcad/genecad:latest \
  uv run python src/evaluate.py \
    --ref  data/gff/evaluation/Juglans_regia_chr1.gff3 \
    --pred results/Jregia_GeneCAD_final.gff \
    --fasta data/fasta/evaluation/Juglans_regia_chr1.fa.gz \
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
