#!/bin/bash
set -euo pipefail

# Set input/output locations
export INPUT_DIR=data
export OUTPUT_DIR=results
mkdir -p $INPUT_DIR $OUTPUT_DIR

# Define inputs for Arabidopsis chr 4 (smallest chromosome; ~18Mb)
FASTA_URL=https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-61/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.chromosome.4.fa.gz
FASTA_FILE=$(basename $FASTA_URL)
FASTA_SPECIES=athaliana
FASTA_CHR_ID=4

# Download FASTA file
echo "Downloading FASTA file from $FASTA_URL to $INPUT_DIR/$FASTA_FILE"
curl --output-dir $INPUT_DIR -O $FASTA_URL

# Run prediction pipeline
echo "Running prediction pipeline for INPUT_FILE=$INPUT_DIR/$FASTA_FILE"
INPUT_FILE=$INPUT_DIR/$FASTA_FILE SPECIES_ID=$FASTA_SPECIES CHR_ID=$FASTA_CHR_ID \
make -f pipelines/prediction all
