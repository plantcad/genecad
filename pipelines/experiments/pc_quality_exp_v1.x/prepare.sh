#!/bin/bash

#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2:00:00

# PC Quality Filter Experiment - Unified Data Preparation Script
# Usage: ./prepare.sh {1.0|1.1|1.2}

set -euo pipefail

# Parse arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 {1.0|1.1|1.2}"
    echo "  1.0 - Athaliana only"
    echo "  1.1 - Athaliana + Osativa (high quality, multiple epochs equivalent)"
    echo "  1.2 - Athaliana + Osativa + Gmax + Hvulgare + Ptrichocarpa (mixed quality, single epoch equivalent)"
    exit 1
fi

VERSION="$1"

# Validate version
if [ "$VERSION" != "1.0" ] && [ "$VERSION" != "1.1" ] && [ "$VERSION" != "1.2" ]; then
    echo "ERROR: Invalid version '$VERSION'"
    echo "Must be '1.0', '1.1', or '1.2'"
    exit 1
fi

# Set version-specific configurations
case "$VERSION" in
    "1.0")
        SPECIES_LIST="Athaliana"
        DESCRIPTION="Athaliana only"
        ;;
    "1.1")
        SPECIES_LIST="Athaliana Osativa"
        DESCRIPTION="Athaliana + Osativa (high quality, multiple epochs equivalent)"
        ;;
    "1.2")
        SPECIES_LIST="Athaliana Osativa Gmax Hvulgare Ptrichocarpa"
        DESCRIPTION="Athaliana + Osativa + Gmax + Hvulgare + Ptrichocarpa (mixed quality, single epoch equivalent)"
        ;;
esac

echo "Starting PC Quality Filter Experiment - Data Preparation v$VERSION"
echo "Species: $DESCRIPTION"
echo "$(date): Beginning data preparation pipeline"

# Define paths
RAW_DIR="$DATA_DIR/training_data"
EXTRACT_DIR="$PIPE_DIR/extract/v$VERSION"
TRANSFORM_DIR="$PIPE_DIR/transform/v$VERSION"
PREP_DIR="$PIPE_DIR/prep/v$VERSION"

# Create output directories
mkdir -p "$EXTRACT_DIR" "$TRANSFORM_DIR" "$PREP_DIR/splits"

# Step 1: Apply PC quality filter to GFF files
echo "$(date): Step 1 - Applying PC quality filter to GFF files"
mkdir -p "$EXTRACT_DIR/gff_filtered"

echo "Filtering GFF files with PC quality filter..."
python scripts/gff.py filter_to_pc_quality_score_pass \
  --input-dir "$RAW_DIR/gff_tagged" \
  --output-dir "$EXTRACT_DIR/gff_filtered" \
  --species-ids $SPECIES_LIST

# Step 2: Extract GFF features (using filtered GFF files)
echo "$(date): Step 2 - Extracting GFF features"
python scripts/extract.py extract_gff_features \
  --input-dir "$EXTRACT_DIR/gff_filtered" \
  --species-id $SPECIES_LIST \
  --output "$EXTRACT_DIR/raw_features.parquet"

# Step 3: Filter features
echo "$(date): Step 3 - Filtering features"
python scripts/transform.py filter_features \
  --input "$EXTRACT_DIR/raw_features.parquet" \
  --output-features "$TRANSFORM_DIR/features.parquet" \
  --output-filters "$TRANSFORM_DIR/filters.parquet" \
  --remove-incomplete-features yes

# Step 4: Stack features
echo "$(date): Step 4 - Stacking features"
python scripts/transform.py stack_features \
  --input "$TRANSFORM_DIR/features.parquet" \
  --output "$TRANSFORM_DIR/intervals.parquet"

# Step 5: Create labels
echo "$(date): Step 5 - Creating labels"
python scripts/transform.py create_labels \
  --input-features "$TRANSFORM_DIR/intervals.parquet" \
  --input-filters "$TRANSFORM_DIR/filters.parquet" \
  --output "$TRANSFORM_DIR/labels.zarr" \
  --remove-incomplete-features yes

# Step 6: Extract FASTA sequences
echo "$(date): Step 6 - Extracting FASTA sequences"
python scripts/extract.py extract_fasta_sequences \
  --input-dir "$RAW_DIR/fasta" \
  --species-id $SPECIES_LIST \
  --tokenizer-path "$MODEL_PATH" \
  --output "$EXTRACT_DIR/tokens.zarr"

# Step 7: Create sequence dataset
echo "$(date): Step 7 - Creating sequence dataset"
python scripts/transform.py create_sequence_dataset \
  --input-labels "$TRANSFORM_DIR/labels.zarr" \
  --input-tokens "$EXTRACT_DIR/tokens.zarr" \
  --output-path "$TRANSFORM_DIR/sequences.zarr" \
  --num-workers 16

# Step 8: Generate training windows
echo "$(date): Step 8 - Generating training windows"
python scripts/sample.py generate_training_windows \
  --input "$TRANSFORM_DIR/sequences.zarr" \
  --output "$TRANSFORM_DIR/windows.zarr" \
  --num-workers 16

# Step 9: Generate training splits
echo "$(date): Step 9 - Generating training splits"
python scripts/sample.py generate_training_splits \
  --input "$TRANSFORM_DIR/windows.zarr" \
  --train-output "$PREP_DIR/splits/train.zarr" \
  --valid-output "$PREP_DIR/splits/valid.zarr" \
  --valid-proportion .025

echo "$(date): Data preparation v$VERSION completed successfully!"
echo "Training data ready at: $PREP_DIR/splits/train.zarr"
echo "Validation data ready at: $PREP_DIR/splits/valid.zarr"
