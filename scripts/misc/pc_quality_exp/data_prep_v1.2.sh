#!/bin/bash

# PC Quality Filter Experiment - Data Preparation v1.2
# Species: Athaliana + Osativa + Gmax + Hvulgare + Ptrichocarpa (mixed quality, single epoch equivalent)
# Generated from: make -f pipelines/training -n all SPECIES_IDS="Athaliana Osativa Gmax Hvulgare Ptrichocarpa" RUN_VERSION=v1.2

set -euo pipefail

echo "Starting PC Quality Filter Experiment - Data Preparation v1.2"
echo "Species: Athaliana + Osativa + Gmax + Hvulgare + Ptrichocarpa"
echo "$(date): Beginning data preparation pipeline"

# Define paths
RAW_DIR="/work/10459/eczech/vista/data/dna/plant_caduceus_genome_annotation_task/data_share_20250326/training_data"
PIPE_DIR="/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline"
EXTRACT_DIR="$PIPE_DIR/extract/v1.2"
TRANSFORM_DIR="$PIPE_DIR/transform/v1.2"
PREP_DIR="$PIPE_DIR/prep/v1.2"
MODEL_PATH="kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000"

# Create output directories
mkdir -p "$EXTRACT_DIR" "$TRANSFORM_DIR" "$PREP_DIR/splits"

# Step 1: Apply PC quality filter to GFF files
echo "$(date): Step 1 - Applying PC quality filter to GFF files"
mkdir -p "$EXTRACT_DIR/gff_filtered"

echo "Filtering GFF files with PC quality filter..."
python scripts/gff.py filter_to_pc_quality_score_pass \
  --input-dir "$RAW_DIR/gff_tagged" \
  --output-dir "$EXTRACT_DIR/gff_filtered" \
  --species-ids Athaliana Osativa Gmax Hvulgare Ptrichocarpa

# Step 2: Extract GFF features (using filtered GFF files)
echo "$(date): Step 2 - Extracting GFF features"
python scripts/extract.py extract_gff_features \
  --input-dir "$EXTRACT_DIR/gff_filtered" \
  --species-id Athaliana Osativa Gmax Hvulgare Ptrichocarpa \
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
  --species-id Athaliana Osativa Gmax Hvulgare Ptrichocarpa \
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

echo "$(date): Data preparation v1.2 completed successfully!"
echo "Training data ready at: $PREP_DIR/splits/train.zarr"
echo "Validation data ready at: $PREP_DIR/splits/valid.zarr" 