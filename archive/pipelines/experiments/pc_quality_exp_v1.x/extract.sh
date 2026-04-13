#!/bin/bash

#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2:00:00

# PC Quality Filter Experiment - FASTA Sequence Extraction Script
# Usage: ./extract.sh [SPECIES_LIST]
#
# Arguments:
#   SPECIES_LIST - Optional space-separated list of species (default: jregia pvulgaris carabica zmays ntabacum nsylvestris)

set -euo pipefail

# Parse arguments - use default species list if none provided
DEFAULT_SPECIES_LIST="jregia pvulgaris carabica zmays ntabacum nsylvestris"
SPECIES_LIST="${1:-$DEFAULT_SPECIES_LIST}"

echo "Starting PC Quality Filter Experiment - FASTA Sequence Extraction"
echo "Species: $(echo "$SPECIES_LIST" | tr ' ' ',')"
echo "$(date): Beginning FASTA sequence extraction"

# Define paths
RAW_DIR="$DATA_DIR/testing_data/fasta"
PREDICT_DIR="$PIPE_DIR/predict"

# Extract sequences for each species
for SPECIES_LOWER in $SPECIES_LIST; do
    SPECIES_ID=$(echo "$SPECIES_LOWER" | sed 's/./\u&/')

    if [ ! -d "$PREDICT_DIR/$SPECIES_LOWER/sequences.zarr" ]; then
        echo "Extracting FASTA sequences for $SPECIES_ID..."
        mkdir -p "$PREDICT_DIR/$SPECIES_LOWER"
        python scripts/extract.py extract_fasta_sequences \
          --input-dir "$RAW_DIR" \
          --species-id "$SPECIES_ID" \
          --tokenizer-path "$TOKENIZER_MODEL_PATH" \
          --output "$PREDICT_DIR/$SPECIES_LOWER/sequences.zarr"
    else
        echo "Sequences already exist for $SPECIES_LOWER, skipping"
    fi
done

echo "$(date): FASTA sequence extraction completed successfully!"
