![](https://img.shields.io/badge/version-1.0.0-blue)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/plantcad/genecad/actions/workflows/ci.yaml/badge.svg)](https://github.com/plantcad/genecad/actions/workflows/ci.yaml)
[![DOI](https://zenodo.org/badge/DOI/10.1101/2025.08.27.672609.svg)](https://doi.org/10.1101/2025.08.27.672609)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg?style=flat)](https://huggingface.co/collections/plantcad/genecad-68c686ccf14312bf6de356de)
[![Container](https://img.shields.io/badge/container-ghcr.io%2Fplantcad%2Fgenecad-blue?logo=github)](https://github.com/orgs/plantcad/packages/container/package/genecad)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)

# GeneCAD

GeneCAD is an in silico genome annotation pipeline for plants built on top of the DNA foundation model [PlantCAD2 (Zhai et al. 2025)](https://doi.org/10.1101/2025.08.27.672609).

## Contents

- [Setup](#setup)
  - [Using Docker](#using-docker)
  - [Using uv](#using-uv)
  - [Using SkyPilot](#using-skypilot)
  - [Using SLURM](#using-slurm)
- [Inference](#inference)
  - [Steps](#steps)
  - [Dry run](#dry-run)
  - [Outputs](#outputs)
  - [Throughput](#throughput)
  - [Refinement](#refinement)
- [Evaluation](#evaluation)
- [Development](#development)
  - [Environment](#environment)
  - [Hugging Face](#hugging-face)
  - [Docker](#docker)
  - [Reproduction](#reproduction)

## Setup

GeneCAD can be installed one of two ways:

1. Using a pre-built Docker image
2. Using a virtual environment via `uv`

We also provide instructions for executing it one of two ways:

1. Using an ephemeral cloud cluster via SkyPilot
2. Using an existing SLURM/HPC cluster

See the sections below for more details.

### Using Docker

This example demonstrates how to run the GeneCAD inference pipeline for a single chromosome via Docker using the image published at `ghcr.io/plantcad/genecad`:

```bash
# Clone the GeneCAD repository
git clone --single-branch https://github.com/plantcad/genecad && cd genecad
# Optionally, checkout a release tag for greater reproducibility
# git checkout v0.0.11

# Pull the image
docker pull ghcr.io/plantcad/genecad:latest

# Run inference on Arabidopsis chromosome 4
docker run --rm --gpus all \
  -v $(pwd):/workspace -w /workspace \
  ghcr.io/plantcad/genecad:latest \
  bash examples/scripts/run_inference.sh

# Run inference on Arabidopsis chromosome 4 using large base PlantCAD2 model
docker run --rm --gpus all \
  -v $(pwd):/workspace -w /workspace \
  -e HEAD_MODEL_PATH=plantcad/GeneCAD-l8-d768-PC2-Large \
  -e BASE_MODEL_PATH=kuleshov-group/PlantCAD2-Large-l48-d1536 \
  ghcr.io/plantcad/genecad:latest \
  bash examples/scripts/run_inference.sh
```

See [examples/scripts/run_inference.sh](examples/scripts/run_inference.sh) for more information on what this does. Note that the docker container only contains the environment, not source code.  Source code is mounted directly so that any changes in the cloned GeneCAD repository take effect immediately.

For convenience, see [genecad_inference_pipeline.log](docs/logs/genecad_inference_pipeline.log) for logs from a complete run of this example so that you know what to expect.

### Using uv

Requirements for installing GeneCAD include CUDA, Linux and [uv](https://docs.astral.sh/uv/). Use these steps to create a compatible environment:

```bash
# Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment (run in project root)
uv venv # Initializes Python 3.12 environment

# Install dependencies
# The mamba/causal-conv1d dependencies can take ~3-30 minutes
# to build from source, depending on CPUs available
uv sync --extra torch --extra mamba
```

PyTorch is included as an optional dependency that can be left out if you wish to install it yourself.  This option is provided because the managed `torch` version in this project is pinned to 2.7.1 built with CUDA 12.8, which is restrictive but necessary to ensure compatibility with the frozen `mamba` (2.2.4) and `causal-conv1d` (1.5.0.post8) dependencies.  Newer CUDA and PyTorch versions may work, but are not tested or officially supported.

Note also that `mamba` and `causal-conv1d` are currently forced to build from source (without build isolation).  This can be slow, but it is more reliable than relying on pre-built wheels.  Those wheels, for Mamba 2.2.4 at least, often install correctly and then fail with inscrutable errors later.  This is probably because the [wheels](https://github.com/state-spaces/mamba/releases/tag/v2.2.4) are only specific to major CUDA versions.  Either way, installing from source is safer and it is always possible to cache the wheels if you wish.  You could add the following in the `pyproject.toml` for this project to do so:

```
[tool.uv.sources]
mamba-ssm = { path = "path/to/mamba_ssm-2.2.4-*.whl" }
causal-conv1d = { path = "path/to/causal_conv1d-1.5.0.post8-*.whl" }
# OR upload the wheels somewhere and use a remote URL:
mamba-ssm = { url = "https://.../mamba_ssm-2.2.4-*.whl" }
causal-conv1d = { url = "https://.../causal_conv1d-1.5.0.post8-*.whl" }
```

### Using SkyPilot

If you do not already have a system with an NVIDIA GPU, or are interested in convenient ways to get one (or several), we recommend using [SkyPilot](https://docs.skypilot.co/en/latest/docs/index.html).  This provides an interface for [optimizing costs](https://docs.skypilot.co/en/latest/overview.html#skypilot-s-cost-and-capacity-optimization) and resource utilization across many cloud providers.

Here are steps you can take from any Mac, Linux or Windows local workstation after installing [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv pip install "skypilot[lambda]>=0.10.3"

# Clear local SkyPilot state, if any
sky api stop; [ -d ~/.sky ] && rm -rf ~/.sky

# Deploy single-node cluster
sky launch --num-nodes 1 --yes --no-setup \
  --cluster genecad examples/configs/cluster.sky.yaml

# Run example inference pipeline and evaluation
sky exec --cluster genecad examples/configs/task.sky.yaml

# SSH to the node
ssh genecad

# Terminate the node
sky down genecad
```

Better GPUs are typically more cost-efficient, but wasteful in development, so we recommend using SkyPilot to be able to switch between cheap sources for both.  See the [Throughput](#throughput) section below for context on how much time and cost is required to run GeneCAD for different GPUs.  SkyPilot makes shopping around for these much more convenient, e.g.:

<details><summary>SkyPilot GPU rates</summary>

```bash
> sky show-gpus L4:1 # low-end development GPU
GPU  QTY  CLOUD   INSTANCE_TYPE  DEVICE_MEM  vCPUs  HOST_MEM  HOURLY_PRICE  HOURLY_SPOT_PRICE  REGION
L4   1.0  RunPod  1x_L4_SECURE   -           4      24GB      $ 0.440       -                  CA
L4   1.0  GCP     g2-standard-4  24GB        4      16GB      $ 0.705       $ 0.282            us-east4
L4   1.0  AWS     g6.xlarge      22GB        4      16GB      $ 0.805       $ 0.232            us-east-2
L4   1.0  AWS     g6.2xlarge     22GB        8      32GB      $ 0.978       $ 0.323            us-east-2
L4   1.0  AWS     g6.4xlarge     22GB        16     64GB      $ 1.323       $ 0.475            us-east-2
L4   1.0  AWS     gr6.4xlarge    22GB        16     128GB     $ 1.539       $ 0.423            us-east-2
L4   1.0  AWS     g6.8xlarge     22GB        32     128GB     $ 2.014       $ 0.673            us-east-2
L4   1.0  AWS     gr6.8xlarge    22GB        32     256GB     $ 2.446       $ 0.558            us-east-2
L4   1.0  AWS     g6.16xlarge    22GB        64     256GB     $ 3.397       $ 1.200            us-east-1

> sky show-gpus H100:1 # high-end production GPU
GPU   QTY  CLOUD       INSTANCE_TYPE                       DEVICE_MEM  vCPUs  HOST_MEM  HOURLY_PRICE  HOURLY_SPOT_PRICE  REGION
H100  1.0  Hyperbolic  1x-H100-75-722                      80GB        75     722GB     $ 1.290       -                  default
H100  1.0  Vast        1x-H100_SXM-32-65536                80GB        32     64GB      $ 1.870       $ 0.000            , US, NA
H100  1.0  Vast        1x-H100_NVL-32-65536                94GB        32     64GB      $ 2.250       $ 0.000            Bulgaria, BG, EU
H100  1.0  RunPod      1x_H100_SECURE                      -           16     80GB      $ 2.390       -                  CA
H100  1.0  Lambda      gpu_1x_h100_pcie                    80GB        26     200GB     $ 2.490       -                  europe-central-1
H100  1.0  Lambda      gpu_1x_h100_sxm5                    80GB        26     225GB     $ 3.290       -                  europe-central-1
H100  1.0  Cudo        epyc-genoa-h100-nvl-pcie_1x2v4gb    94GB        2      4GB       $ 2.499       -                  au-melbourne-1
H100  1.0  Cudo        epyc-genoa-h100-nvl-pcie_1x4v8gb    94GB        4      8GB       $ 2.529       -                  au-melbourne-1
H100  1.0  Cudo        epyc-genoa-h100-nvl-pcie_1x12v24gb  94GB        12     24GB      $ 2.646       -                  au-melbourne-1
H100  1.0  Cudo        epyc-genoa-h100-nvl-pcie_1x24v48gb  94GB        24     48GB      $ 2.823       -                  au-melbourne-1
H100  1.0  nebius      gpu-h100-sxm_1gpu-16vcpu-200gb      80GB        16     200GB     $ 2.950       $ 1.250            eu-north1
H100  1.0  GCP         a3-highgpu-1g                       80GB        26     234GB     $ 5.383       $ 2.525            us-central1
H100  1.0  Paperspace  H100                                -           15     80GB      $ 5.950       -                  East Coast (NY2)
H100  1.0  DO          gpu-h100x1-80gb                     80GB        20     240GB     $ 6.740       -                  tor1
H100  1.0  AWS         p5.4xlarge                          80GB        16     256GB     $ 6.880       -                  us-east-1
H100  1.0  Azure       Standard_NC40ads_H100_v5            -           40     320GB     $ 6.980       $ 6.980            eastus
```

</details>


If you wish to use multiple nodes with SkyPilot, then you can construct a [Task](https://docs.skypilot.co/en/v0.5.0/reference/yaml-spec.html) like this based on [environment variables for distributed execution](https://docs.skypilot.co/en/latest/running-jobs/distributed-jobs.html#environment-variables)

```yaml
workdir: .
run: |
  export WORLD_SIZE=$SKYPILOT_NUM_NODES
  export RANK=$SKYPILOT_NODE_RANK
  ... # Execute GeneCAD pipeline as before
```

The `RANK` and `WORLD_SIZE` environment variables are not used by any part of the pipeline other than the actual prediction step, which writes Xarray datasets in the format `predictions.{rank}.zarr` (see [Outputs](#outputs) for details).  GeneCAD does not currently support reading these files in a SkyPilot cluster.  GeneCAD was built in an HPC/SLURM cluster with shared storage (see [Using SLURM](#using-slurm) for details on that), so you would need to copy all of these datasets to a networked filesystem (depending on your cloud provider) or gather them all on node before continuing the pipeline.  Feel free to open an issue if you want to use a distributed cluster with no shared storage, and we can likely help make that work.

### Using SLURM

To use SLURM, we recommend following the `uv` instructions above to initialize a virtual environment.  You will need to source the appropriate TCL/Lmod modules for your cluster, e.g. CUDA 12.8 and Python 3.12.  Other combinations will likely work as well, as long as they are newer than CUDA 12.4, Python 3.11 and PyTorch 2.5.1 (our earliest tested versions).

See the [Docker](#docker) section above for context on the basics of running an example pipeline.  The rest of these instructions focus on how to distribute work in a more realistic setting.  E.g. it is often simplest to begin by running the GeneCAD pipeline with an interactive SLURM allocation like this:

```bash
# Allocate interactive session:
salloc -partition=gpu-queue --nodes=4 --ntasks=4

# Within interactive session:
cd /path/to/GeneCAD
INPUT_FILE=/path/to/input.fasta \
OUTPUT_DIR=/path/to/output \
SPECIES_ID=species_id \
CHR_ID=chromosome_id \
LAUNCHER="srun python" \
make -f pipelines/prediction all
```

This will distribute the computation of model predictions across nodes (4 nodes in the example).  For offline batch scheduling with `sbatch`, we suggest following this pattern:

```bash
# Create SLURM wrapper script
cat << 'EOF' > genecad.slurm
#!/bin/bash

export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NNODES
export LAUNCHER="srun python"

exec "$@"
EOF

# Schedule processing for a single FASTA file across 4 nodes
sbatch \
  --partition=gpu-queue \
  --nodes=4 \
  --ntasks=4 \
genecad.slurm \
  INPUT_FILE=/path/to/input.fasta \
  OUTPUT_DIR=/path/to/output \
  SPECIES_ID=species_id \
  CHR_ID=chromosome_id \
  make -f pipelines/prediction all
```

The GeneCAD pipeline relies only on settings for `RANK` and `WORLD_SIZE` to distribute the computation of model predictions.  All pre and post processing steps around that require only CPUs and are executed on a single host.  Depending on your cluster topology, you may wish to break up the pipeline like this instead:

```bash
export INPUT_FILE=/path/to/input.fasta
export OUTPUT_DIR=/path/to/output
export SPECIES_ID=species_id
export CHR_ID=chromosome_id

# CPU-only preprocessing (FASTA -> Xarray)
sbatch -p cpu-queue -N 1 -n 1 genecad.slurm make -f pipelines/prediction sequences

# Distributed prediction generation on GPU nodes
sbatch -p gpu-queue -N 10 -n 10 genecad.slurm make -f pipelines/prediction predictions

# CPU-only post-processing (Xarray -> GFF)
sbatch -p cpu-queue -N 1 -n 1 genecad.slurm make -f pipelines/prediction annotations
```

Note that currently only one GPU per-node is supported.  This means that you cannot invoke `make predictions` with `--n-tasks-per-node` greater than 1.


## Inference

The inference pipeline requires these inputs at a minimum:

```bash
INPUT_FILE=/path/to/input.fasta \
OUTPUT_DIR=/path/to/output \
SPECIES_ID=species_id \
CHR_ID=chromosome_id \
make -f pipelines/prediction all
```

Note that the `SPECIES_ID` variable is used for more informative logging and it is added to attributes in resulting Xarray datasets (along with `CHR_ID`).  It can be any descriptive string, e.g. `athaliana` or `arabidopsis_thaliana`.  The `CHR_ID` variable must be a string matching the name of the target chromosome in the input FASTA file.  It too will be included in logging and attributes.

See the [pipelines/prediction](pipelines/prediction) Makefile for more details on other configurable parameters.

### Steps

The pipeline consists of 3 main steps:

1. Extract sequences from the input FASTA file
2. Generate token and feature logits from the GeneCAD classifier
3. Generate gene/transcript/feature annotations from the token and feature logits (as GFF)

Each of these has an associated `make` target, i.e. `make sequences`, `make predictions`, `make annotations` (respectively).

### Dry run

The shell commands run by `make` can be seen with either the `--dry-run` or `-n` flags, each of which does the same thing.  This tells `make` to print all the commands that *would* be executed, which can be helpful for debugging or clarity on what the pipeline actually does.  It can also be useful for executing individual steps manually.

Additionally, this can be very useful for determining what targets already exist.  I.e. if you run `make predictions` instead of `make all`, the pipeline will only run up to the target for generating token logits from the GeneCAD classifier.  Running `make -n annotations` (or `make -n all`, which is an alias), would then show that only the GFF export needs to be run rather than the whole pipeline.

### Outputs

The inference pipeline will generate results (in `$OUTPUT_DIR`) with the following structure:

```bash
tree -L 5 -I '[0-9]*|[.]' $OUTPUT_DIR
├── pipeline
│   ├── intervals.zarr
│   │   ├── intervals
│   │   │   ├── decoding
│   │   │   ├── entity_index
│   │   │   ├── entity_name
│   │   │   ├── interval
│   │   │   ├── start
│   │   │   ├── stop
│   │   │   └── strand
│   │   └── sequences
│   │       ├── feature
│   │       ├── feature_logits
│   │       ├── feature_predictions
│   │       ├── sequence
│   │       ├── strand
│   │       └── token
│   ├── predictions__raw__feat_len_2__gene_len_30.gff
│   ├── predictions__raw__feat_len_2__gene_len_30__has_req_feats.gff
│   ├── predictions__raw__feat_len_2.gff
│   ├── predictions__raw.gff
│   ├── predictions.zarr
│   │   └── predictions.0.zarr
│   │       ├── negative
│   │       │   ├── feature
│   │       │   ├── feature_logits
│   │       │   ├── feature_predictions
│   │       │   ├── sequence
│   │       │   ├── token
│   │       │   ├── token_logits
│   │       │   └── token_predictions
│   │       └── positive
│   │           ├── feature
│   │           ├── feature_logits
│   │           ├── feature_predictions
│   │           ├── sequence
│   │           ├── token
│   │           ├── token_logits
│   │           └── token_predictions
│   └── sequences.zarr
│       └── athaliana
└── predictions.gff
```

The `predictions.gff` file contains the final annotations while intermediate results in `pipeline` are returned for debugging or further analysis.  If needed, tokenized sequences are available as `sequences.zarr` and logits from the underlying GeneCAD model are present in `predictions.zarr`.

### Throughput

This table shows observed GeneCAD inference throughput on different devices and clouds. Of note:

- Datacenter-class GPUs (`A100`s/`H100`s) are 3-6x more cost efficient than older, highly available development GPUs like NVIDIA `L4`'s on Google Cloud
- `GeneCAD-Large` is ~3x more expensive to run than `GeneCAD-Small`

| Provider | Instance              | Model           | Throughput (bp/s) | Cost ($/hr) | Cost per Mbp ($) |
|----------|-----------------------|-----------------|-------------------|-------------|------------------|
| GCE      | g2-standard-32 / nvidia-l4 | GeneCAD-Small | 4,063             | 1.7344      | 0.1186           |
| Lambda | gpu_1x_a100_sxm4 | GeneCAD-Small | 17,687 | 1.2900 | 0.0202 |
| Lambda   | gpu_1x_h100_pcie      | GeneCAD-Small | 22,020            | 2.4900      | 0.0314           |
| Lambda   | gpu_2x_h100_sxm5      | GeneCAD-Small | 34,290            | 6.3800      | 0.0517           |
| Lambda   | gpu_2x_h100_sxm5      | GeneCAD-Large | 11,223            | 6.3800      |   0.1579         |

These estimates can be used to extrapolate costs and GPU hours necessary for a few reference genomes.  The estimated costs below all assume the lowest `Cost per Mbp ($)` observed above on Lambda `A100 40G SXM4` GPUs (17,687 bp/s, $0.0202 / Mbp) with the `GeneCAD-Small` model.  Any cost can be multiplied by ~3 to approximate the cost of using the `GeneCAD-Large` model instead.

| Genome | Length (Mbp) | GPU Hours | Cost ($) |
|----------|-------------|-----------|----------|
| Arabidopsis thaliana (thale cress) | 120 | 2.12 | 2.42 |
| Zea mays (corn)| 2,182 | 34.27 | 44.04 |
| Hordeum vulgare (barley) | 4,224 | 66.34 | 85.32 |
| Triticum aestivum (wheat) | 14,577 | 227.00 | 294.46 |

For example, this table is saying that it costs about $44 to run GeneCAD-Small on the Zea mays genome to produce a predicted annotations (`gff`) file.

## Refinement

After generating initial predictions, you can refine them using the [ReelProtein](https://onlinelibrary.wiley.com/doi/10.1111/tpj.70483?af=R) pipeline. This post-processing step merges fragmented gene predictions, generates protein embeddings using ProtT5, scores sequences using XGBoost models, and produces a filtered GFF file.

The pipeline handles model downloading automatically via (Hugging Face)[https://huggingface.co/plantcad/reelprotein].

```bash
uv run python scripts/refine.py \
  --gff /path/to/predictions.gff \
  --genome /path/to/genome.fa \
  --out /path/to/refined_output.gff
```
## Evaluation

A simple way to evaluate predicted annotations is with the gffcompare tool.  It can either be [built from source](https://github.com/gpertea/gffcompare?tab=readme-ov-file#building-from-source) or [installed via conda](https://anaconda.org/bioconda/gffcompare).  See [examples/scripts/run_evaluation.sh](examples/scripts/run_evaluation.sh) for an example of how to use it.  See [Dockerfile](Dockerfile) for an example installation from source.

While `gffcompare` offers a useful starting point for evaluating in silico annotations, it is limited by its inability to accurately assess regions independent of predicted UTRs.  UTRs are known to be very difficult to predict accurately from DNA alone and these predictions are of limited utility in many applications.  As a result, we offer a separate evaluation tool for assessing performance of whole-transcript annotations that excludes UTRs.  This tool produces metrics aggregated to several levels (much like `gffcompare`) as well as a `transcript_cds` score that scores perfect, predicted annotations of introns and exons over translated regions only.  Here is some example usage:

```bash
python scripts/gff.py evaluate \
  --reference /path/to/reference.gff \
  --input /path/to/predictions.gff \
  --output /path/to/results
cat /path/to/results/gffeval.stats.tsv
# level                                precision   recall   f1
# transcript_cds                          77.34    47.39  58.77
# transcript_intron                       68.70    42.09  52.20
# transcript                               0.00     0.00   0.00
# exon_cds_longest_transcript_only        83.06    60.16  69.78
# exon_longest_transcript_only            60.05    40.73  48.54
# intron_cds_longest_transcript_only      87.58    66.24  75.43
# intron_longest_transcript_only          89.21    62.00  73.16
# exon_cds                                94.04    47.10  62.77
# exon                                    65.14    28.72  39.86
# intron_cds                              97.29    51.15  67.05
# intron                                  96.32    45.36  61.68
```

For each metric level, three values are reported: precision, recall, and F1 score. The GFF specified by `--reference` contains all true features, while the gff specified by `--input` provides the predicted features. Below is a definition of a match, or true positive, for each level.

| Level                              | Description                                                                                                                                                                                                                                                                 |
|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| transcript_cds                     | Instances where a predicted transcript's CDS exactly matches the true transcript's CDS. Discrepancies in the UTRs are allowed.                                                                                                                                              |
| transcript_intron                  | Instances where a predicted transcript's intron boundaries exactly match the true transcript's intron boundaries. Discrepancies in the transcription start and stop are allowed, as long as they do not alter splice sites.                                                 |
| transcript                         | Instances where a predicted transcript exactly matches the true transcript, including CDS and UTRs.                                                                                                                                                                         |
| exon_cds_longest_transcript_only   | Instances where a predicted exon's boundaries exactly match the true exon's boundaries, taking into account only the exons from the longest true transcript and excluding UTR exons. If an exon is split between UTR and CDS sequence, only the CDS portion is condsidered. |
| exon_longest_transcript_only       | Instances where a predicted exon's boundaries exactly match a true exon's boundaries, taking into account only the exons from the longest true transcript.                                                                                                                  |
| intron_cds_longest_transcript_only | Instances where a predicted intron's boundaries exactly match the true intron's boundaries, taking into account only the introns from the longest true transcript and excluding introns that border UTR sequence.                                                           |
| intron_longest_transcript_only     | Instances where a predicted intron's boundaries exactly match a true intron's boundaries, taking into account only the introns from the longest true transcript.                                                                                                            |
| exon_cds                           | Instances where a predicted exon's boundaries exactly match the true exon's boundaries, excluding UTR exons. If an exon is split between UTR and CDS sequence, only the CDS portion is condsidered. Exons from alternative transcripts are considered.                      |
| exon                               | Instances where a predicted exon's boundaries exactly match a true exon's boundaries. Exons from alternative transcripts are considered.                                                                                                                                    |
| intron_cds                         | Instances where a predicted intron's boundaries exactly match the true intron's boundaries, excluding introns that border UTR sequence. Introns from alternative transcripts are considered.                                                                                |
| intron                             | Instances where a predicted intron's boundaries exactly match a true intron's boundaries. Exons from alternative transcripts are considered.                                                                                                                                |

Note: for the first three transcript-level metrics, a true positive result is counted whenever the predicted transcript matches at least one of the true transcripts for a gene with alternative splicing.

## Development

### Environment

Use this to initialize a local development environment, with or without a GPU:

```bash
uv sync --dev
pre-commit install

# Check types
pyrefly check --summarize-errors

# Run tests that don't require GPUs
pytest -vrs
```

### Hugging Face

HF checkpoints should be uploaded to https://huggingface.co/plantcad.  Currently, the Lightning checkpoints used as-is so there is little to this other than a single upload:

```bash
# cd /path/to/checkpoints/small-base
hf upload plantcad/GeneCAD-l8-d768-PC2-Small model.ckpt --repo-type model
# cd /path/to/checkpoints/large-base
hf upload plantcad/GeneCAD-l8-d768-PC2-Large model.ckpt --repo-type model
```

In the future, this may include a conversion to safetensors/gguf format first.

### Docker

To build the Docker image, launch a Lambda VM via SkyPilot (which installs Docker), and then build the image from there.

```bash
# On Lambda, you need to add default user to docker group
sudo usermod -aG docker ubuntu && newgrp docker

# Build the image
docker build --progress=plain --no-cache -t genecad:v1.0.1 .

# Test the build
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace genecad:v1.0.1 \
  bash examples/scripts/run_inference.sh && \
  bash examples/scripts/run_evaluation.sh
```

Publish to GitHub Container Registry:

```bash
# Tag and push to GitHub Container Registry
IMAGE=ghcr.io/plantcad/genecad
docker tag genecad:v1.0.1 $IMAGE:v1.0.1
docker tag genecad:v1.0.1 $IMAGE:latest
# Requires a GitHub personal access token with "write:packages"
# stored in GHCR_TOKEN
echo $GHCR_TOKEN | docker login ghcr.io -u <github-username> --password-stdin
docker push $IMAGE:v1.0.1
docker push $IMAGE:latest
```

### Reproduction

This code demonstrates how to reproduce published GeneCAD results for a particular species, i.e. Juglans regia (Walnut) chromosome 1:

```bash
# Build environment via uv and source it
source .venv/bin/activate

export INPUT_DIR=data
export OUTPUT_DIR=results
mkdir -p $INPUT_DIR $OUTPUT_DIR

# Download FASTA and GFF files
hf download plantcad/genecad-dev data/fasta/Juglans_regia.Walnut_2.0.dna.toplevel_chr1.fa --repo-type dataset --local-dir .
hf download plantcad/genecad-dev data/gff/Juglans_regia.Walnut_2.0.60_chr1.gff3 --repo-type dataset --local-dir .

# Run inference pipeline in container
docker run --rm --gpus all \
  -v $(pwd):/workspace -w /workspace \
  -e OUTPUT_DIR \
  -e INPUT_FILE=data/fasta/Juglans_regia.Walnut_2.0.dna.toplevel_chr1.fa \
  -e SPECIES_ID=jregia -e CHR_ID=1 \
  ghcr.io/plantcad/genecad:latest \
  make -f pipelines/prediction all # add -n for dry run first to make sure paths are correct

# Run gffcompare (inside or outside container)
docker run --rm \
  -v $(pwd):/workspace -w /workspace \
  ghcr.io/plantcad/genecad:latest gffcompare \
  -r data/gff/Juglans_regia.Walnut_2.0.60_chr1.gff3 \
  -C -o $OUTPUT_DIR/gffcompare $OUTPUT_DIR/predictions.gff
cat $OUTPUT_DIR/gffcompare.stats
# #= Summary for dataset: results/predictions.gff
# #     Query mRNAs :    2253 in    2253 loci  (1786 multi-exon transcripts)
# #            (0 multi-transcript loci, ~1.0 transcripts per locus)
# # Reference mRNAs :    3472 in    2589 loci  (2896 multi-exon)
# # Super-loci w/ reference transcripts:     1991
# #-----------------| Sensitivity | Precision  |
#         Base level:    79.7     |    76.7    |
#         Exon level:    60.8     |    75.0    |
#       Intron level:    76.1     |    92.1    |
# Intron chain level:    44.6     |    72.4    |
#   Transcript level:    43.9     |    67.7    |
#        Locus level:    58.5     |    67.7    |

#      Matching intron chains:    1293
#        Matching transcripts:    1525
#               Matching loci:    1514

#           Missed exons:    2844/15331	( 18.6%)
#            Novel exons:     827/12229	(  6.8%)
#         Missed introns:    2139/12078	( 17.7%)
#          Novel introns:     537/9976	(  5.4%)
#            Missed loci:     594/2589	( 22.9%)
#             Novel loci:     241/2253	( 10.7%)

#  Total union super-loci across all input datasets: 2232
# 2253 out of 2253 consensus transcripts written in results/gffcompare.combined.gtf (0 discarded as redundant)
```
