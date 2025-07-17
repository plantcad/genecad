#!/bin/bash

#SBATCH -p gh
#SBATCH -N 20
#SBATCH -n 20
#SBATCH -t 2:00:00

# PC Quality Filter Experiment - Prediction Generation Script
# Usage: ./predict.sh {1.0|1.1|1.2}

set -euo pipefail

# Parse arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 {1.0|1.1|1.2}"
    echo "  1.0 - Use v1.0 model (Athaliana only)"
    echo "  1.1 - Use v1.1 model (Athaliana + Osativa)"
    echo "  1.2 - Use v1.2 model (All 5 species from v1.1 checkpoint)"
    exit 1
fi

MODEL_VERSION="$1"

# Validate model version
if [ "$MODEL_VERSION" != "1.0" ] && [ "$MODEL_VERSION" != "1.1" ] && [ "$MODEL_VERSION" != "1.2" ]; then
    echo "ERROR: Invalid model version '$MODEL_VERSION'"
    echo "Must be '1.0', '1.1', or '1.2'"
    exit 1
fi

# Set model-specific configurations
case "$MODEL_VERSION" in
    "1.0")
        MODEL_DESCRIPTION="v1.0 (Athaliana only)"
        SWEEP_DIR="sweep-v1.0__cfg_013__arch_all__frzn_yes__lr_1e-04"
        ;;
    "1.1")
        MODEL_DESCRIPTION="v1.1 (Athaliana + Osativa)"
        SWEEP_DIR="sweep-v1.1__cfg_013__arch_all__frzn_yes__lr_1e-04"
        ;;
    "1.2")
        MODEL_DESCRIPTION="v1.2 (All 5 species fresh start)"
        SWEEP_DIR="sweep-v1.2__cfg_013__arch_all__frzn_yes__lr_1e-04"
        ;;
esac

RUN_VERSION="v$MODEL_VERSION"

echo "Starting PC Quality Filter Experiment - Prediction Generation $RUN_VERSION"
echo "Model: $MODEL_DESCRIPTION"
echo "Species: jregia, pvulgaris, carabica"
echo "$(date): Beginning prediction generation"

# Define paths
RAW_DIR="$DATA_DIR/testing_data"
PREDICT_DIR="$PIPE_DIR/predict"
MODEL_CHECKPOINT="$PIPE_DIR/sweep/$SWEEP_DIR/checkpoints/last.ckpt"

# Define species to evaluate
SPECIES_LIST="jregia pvulgaris carabica"
CHR_ID="chr1"

echo "Model checkpoint: $MODEL_CHECKPOINT"
echo "Output directory: $PREDICT_DIR"

# Verify checkpoint exists
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "ERROR: Model checkpoint not found at $MODEL_CHECKPOINT"
    echo "Please ensure v$MODEL_VERSION training has completed successfully"
    exit 1
fi

# Process each species
for SPECIES in $SPECIES_LIST; do
    echo "$(date): ========================================="
    echo "$(date): Processing species: $SPECIES"
    echo "$(date): ========================================="

    # Convert to proper species ID (capitalize first letter)
    SPECIES_ID=$(echo "$SPECIES" | sed 's/./\u&/')
    SPECIES_DIR="$PREDICT_DIR/$SPECIES/runs/$RUN_VERSION/$CHR_ID"

    echo "Species ID: $SPECIES_ID"
    echo "Species directory: $SPECIES_DIR"

    # Create output directories
    mkdir -p "$SPECIES_DIR/predictions"

    # Verify sequences exist
    if [ ! -d "$PREDICT_DIR/$SPECIES/sequences.zarr" ]; then
        echo "ERROR: Sequences not found for $SPECIES at $PREDICT_DIR/$SPECIES/sequences.zarr"
        echo "Please run extract.sh first to generate sequences"
        exit 1
    fi
    echo "$(date): Using existing sequences for $SPECIES"

    # Step 1: Generate predictions
    echo "$(date): Step 1 - Generating predictions for $SPECIES"
    if [ ! -d "$SPECIES_DIR/predictions" ] || [ -z "$(ls -A "$SPECIES_DIR/predictions" 2>/dev/null)" ]; then
        echo "Generating predictions for $SPECIES..."
        srun bin/tacc \
          python scripts/predict.py create_predictions \
          --input "$PREDICT_DIR/$SPECIES/sequences.zarr" \
          --output-dir "$SPECIES_DIR/predictions" \
          --model-path "$MODEL_PATH" \
          --model-checkpoint "$MODEL_CHECKPOINT" \
          --species-id "$SPECIES_ID" \
          --chromosome-id "$CHR_ID" \
          --batch-size 32
    else
        echo "Predictions already exist for $SPECIES, skipping generation"
    fi

    echo "$(date): Completed prediction generation for $SPECIES"
done

echo "$(date): Prediction generation $RUN_VERSION completed successfully!"
echo "Predictions available in: $PREDICT_DIR/{jregia,pvulgaris,carabica}/runs/$RUN_VERSION/$CHR_ID/predictions/"
