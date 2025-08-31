# GeneCAD

## Setup

TODO:

- Provide `uv` instructions
- Build Docker image

## Inference

This example demonstrates how to run the GeneCAD inference pipeline for a single chromosome:

```bash
# Choose a destination directory for all results
export OUTPUT_DIR=/tmp/results

# Download Arabidopsis chr 4 FASTA (smallest chromosome; ~18Mb)
curl -O https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-61/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.chromosome.4.fa.gz

# Run prediction pipeline
INPUT_FILE=Arabidopsis_thaliana.TAIR10.dna.chromosome.4.fa.gz \
SPECIES_ID=athaliana \
CHR_ID=4 \
make -f pipelines/prediction all
```

For convenience, see [genecad_inference_pipeline.log](docs/logs/genecad_inference_pipeline.log) for logs from a complete run of this example.

### Details

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

## Evaluation

A simple way to evaluate predicted annotations is with the gffcompare tool.  It can be [built from source](https://github.com/gpertea/gffcompare?tab=readme-ov-file#building-from-source) or [installed via conda](https://anaconda.org/bioconda/gffcompare).  Continuing on the example above:

```bash
# Download associated Arabidopsis chr 4 reference annotations
curl -O https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-61/gff3/arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.61.chromosome.4.gff3.gz
gzip -d Arabidopsis_thaliana.TAIR10.61.chromosome.4.gff3.gz

# Install gffcompare, e.g.:
# conda install bioconda::gffcompare

# Run comparison
gffcompare \
  -r Arabidopsis_thaliana.TAIR10.61.chromosome.4.gff3 -C \
  -o $OUTPUT_DIR/gffcompare \
  $OUTPUT_DIR/predictions.gff
cat $OUTPUT_DIR/gffcompare.stats
# ...
#= Summary for dataset: /tmp/results/predictions.gff
#     Query mRNAs :    3605 in    3605 loci  (2915 multi-exon transcripts)
#            (0 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :    8185 in    4803 loci  (6624 multi-exon)
# Super-loci w/ reference transcripts:     3470
#  -----------------| Sensitivity | Precision  |
#         Base level:    81.3     |    95.5    |
#         Exon level:    67.7     |    90.2    |
#       Intron level:    81.5     |    97.4    |
# Intron chain level:    37.7     |    85.8    |
#   Transcript level:    30.5     |    69.3    |
#        Locus level:    51.4     |    69.3    |
# ...
```

## SLURM

In practice, it is often simplest to run the GeneCAD pipeline with an interactive SLURM allocation like this:

```bash
# Allocate interactive session:
salloc -partition=gpu-queue --nodes=4 --ntasks=4

# Within interactive session:
cd /path/to/GeneCAD
INPUT_FILE=/path/to/input.fasta \
OUTPUT_DIR=/path/to/output \
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
  make -f pipelines/prediction all
```

The GeneCAD pipeline relies only on settings for `RANK` and `WORLD_SIZE` to distribute the computation of model predictions.  All pre and post processing steps around that require only CPUs and are executed on a single host.  Depending on your cluster topology, you may wish to break up the pipeline like this instead:

```bash
export INPUT_FILE=/path/to/input.fasta
export OUTPUT_DIR=/path/to/output

# CPU-only preprocessing (FASTA -> Xarray)
sbatch -p cpu-queue -N 1 -n 1 genecad.slurm make -f pipelines/prediction sequences

# Distributed prediction generation on GPU nodes
sbatch -p gpu-queue -N 10 -n 10 genecad.slurm make -f pipelines/prediction predictions

# CPU-only post-processing (Xarray -> GFF)
sbatch -p cpu-queue -N 1 -n 1 genecad.slurm make -f pipelines/prediction annotations
```

Note that currently only one GPU per-node is supported.  This means that you cannot invoke `make predictions` with `--n-tasks-per-node` greater than 1.
