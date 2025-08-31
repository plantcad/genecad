# GeneCAD

## Setup

WIP

## Inference


### SLURM

In practice, it is often simplest to run the GeneCAD pipeline with an interactive SLURM allocation like this:

```bash
# Allocate interactive session:
salloc -partition=gpu-queue --nodes=4 --ntasks=4

# Within interactive session:
cd /path/to/GeneCAD
INPUT_FILE=/path/to/input.gff \
OUTPUT_DIR=/path/to/output \
LAUNCHER="srun python" \
make pipelines/prediction all
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

# Schedule processing for a single GFF file across 4 nodes
sbatch \
  --partition=gpu-queue \
  --nodes=4 \
  --ntasks=4 \
genecad.slurm \
  INPUT_FILE=/path/to/input.gff \
  OUTPUT_DIR=/path/to/output \
  make pipelines/prediction all
```

The GeneCAD pipeline relies only on settings for `RANK` and `WORLD_SIZE` to distribute the computation of model predictions.  All pre and post processing steps around that require only CPUs and are executed on a single host.  Depending on your cluster topology, you may wish to break up the pipeline like this instead:

```bash
export INPUT_FILE=/path/to/input.gff
export OUTPUT_DIR=/path/to/output

# CPU-only preprocessing (FASTA -> Xarray)
sbatch -p cpu-queue -N 1 -n 1 genecad.slurm make sequences

# Distributed prediction generation on GPU nodes
sbatch -p gpu-queue -N 10 -n 10 genecad.slurm make predictions

# CPU-only post-processing (Xarray -> GFF)
sbatch -p cpu-queue -N 1 -n 1 genecad.slurm make annotations
```

Note that currently only one GPU per-node is supported.  This means that you cannot invoke `make predictions` with `--n-tasks-per-node` greater than 1.
