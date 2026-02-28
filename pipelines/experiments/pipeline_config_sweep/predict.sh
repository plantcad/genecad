#!/bin/bash

#SBATCH -p gh
#SBATCH -N 20
#SBATCH -n 20
#SBATCH -t 2:00:00

# GeneCAD Pipeline Configuration Sweep - Prediction Generation Script
# Usage: ./predict.sh {1.0|1.1|1.2|1.3|1.4|1.5}

set -euo pipefail

# Parse arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 {1.0|1.1|1.2|1.3|1.4|1.5}"
    echo "  1.0 - Use v1.0 model (Athaliana only)"
    echo "  1.1 - Use v1.1 model (Athaliana + Osativa)"
    echo "  1.2 - Use v1.2 model (All 5 species from v1.1 checkpoint)"
    echo "  1.3 - Use v1.3 model (Athaliana + Osativa with randomized base encoder)"
    echo "  1.4 - Use v1.4 model (Athaliana + Osativa with large PlantCAD base model)"
    echo "  1.5 - Use v1.5 model (All 5 species from v1.4 checkpoint, large base model)"
    exit 1
fi

MODEL_VERSION="$1"

# Validate model version
if [ "$MODEL_VERSION" != "1.0" ] && [ "$MODEL_VERSION" != "1.1" ] && [ "$MODEL_VERSION" != "1.2" ] && [ "$MODEL_VERSION" != "1.3" ] && [ "$MODEL_VERSION" != "1.4" ] && [ "$MODEL_VERSION" != "1.5" ]; then
    echo "ERROR: Invalid model version '$MODEL_VERSION'"
    echo "Must be '1.0', '1.1', '1.2', '1.3', '1.4', or '1.5'"
    exit 1
fi

# Set model-specific configurations
case "$MODEL_VERSION" in
    "1.0")
        MODEL_DESCRIPTION="v1.0 (Athaliana only)"
        SWEEP_DIR="sweep-v1.0__cfg_013__rand_no__arch_all__frzn_yes__lr_1e-04"
        BASE_MODEL_PATH="kuleshov-group/PlantCAD2-Small-l24-d0768"  # pragma: allowlist secret
        DTYPE="float32"
        ;;
    "1.1")
        MODEL_DESCRIPTION="v1.1 (Athaliana + Osativa)"
        SWEEP_DIR="sweep-v1.1__cfg_013__rand_no__arch_all__frzn_yes__lr_1e-04"
        BASE_MODEL_PATH="kuleshov-group/PlantCAD2-Small-l24-d0768"  # pragma: allowlist secret
        DTYPE="float32"
        ;;
    "1.2")
        MODEL_DESCRIPTION="v1.2 (All 5 species from v1.1 checkpoint)"
        SWEEP_DIR="sweep-v1.2__cfg_013__rand_no__arch_all__frzn_yes__lr_1e-04"
        BASE_MODEL_PATH="kuleshov-group/PlantCAD2-Small-l24-d0768"  # pragma: allowlist secret
        DTYPE="float32"
        ;;
    "1.3")
        MODEL_DESCRIPTION="v1.3 (Athaliana + Osativa with randomized base encoder)"
        SWEEP_DIR="sweep-v1.3__cfg_016__rand_yes__arch_all__frzn_yes__lr_1e-04"
        BASE_MODEL_PATH="kuleshov-group/PlantCAD2-Small-l24-d0768"  # pragma: allowlist secret
        DTYPE="float32"
        ;;
    "1.4")
        MODEL_DESCRIPTION="v1.4 (Athaliana + Osativa with large PlantCAD base model)"
        SWEEP_DIR="sweep-v1.4__cfg_013__rand_no__arch_all__frzn_yes__lr_1e-04"
        BASE_MODEL_PATH="kuleshov-group/PlantCAD2-Large-l48-d1536"  # pragma: allowlist secret
        DTYPE="bfloat16"
        ;;
    "1.5")
        MODEL_DESCRIPTION="v1.5 (All 5 species from v1.4 checkpoint, large base model)"
        SWEEP_DIR="sweep-v1.5__cfg_013__rand_no__arch_all__frzn_yes__lr_1e-04"
        BASE_MODEL_PATH="kuleshov-group/PlantCAD2-Large-l48-d1536"  # pragma: allowlist secret
        DTYPE="bfloat16"
        ;;
esac

RUN_VERSION="v$MODEL_VERSION"

echo "$(date): Starting GeneCAD Pipeline Configuration Sweep - Prediction Generation $RUN_VERSION"
echo "$(date): Model: $MODEL_DESCRIPTION"
echo "$(date): Beginning prediction generation"

# Define paths
RAW_DIR="$DATA_DIR/testing_data"
PREDICT_DIR="$PIPE_DIR/predict"
MODEL_CHECKPOINT="$PIPE_DIR/sweep/$SWEEP_DIR/checkpoints/last.ckpt"

# Define species to evaluate
SPECIES_LIST="jregia pvulgaris carabica zmays ntabacum nsylvestris"
CHR_ID="chr1"

echo "$(date): Model checkpoint: $MODEL_CHECKPOINT"
echo "$(date): Output directory: $PREDICT_DIR"
echo "$(date): Species to evaluate: $SPECIES_LIST"

# Verify checkpoint exists
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "$(date): ERROR: Model checkpoint not found at $MODEL_CHECKPOINT"
    echo "$(date): Please ensure v$MODEL_VERSION training has completed successfully"
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

    echo "$(date): Species ID: $SPECIES_ID"
    echo "$(date): Species directory: $SPECIES_DIR"

    # Create output directories
    echo "$(date): Creating output directories"
    mkdir -p "$SPECIES_DIR/predictions"

    # Verify sequences exist
    if [ ! -d "$PREDICT_DIR/$SPECIES/sequences.zarr" ]; then
        echo "$(date): ERROR: Sequences not found for $SPECIES at $PREDICT_DIR/$SPECIES/sequences.zarr"
        echo "$(date): Please run extract.sh first to generate sequences"
        exit 1
    fi
    echo "$(date): Using existing sequences for $SPECIES"

    # Step 1: Generate predictions
    echo "$(date): Step 1 - Generating predictions for $SPECIES"
    if [ ! -d "$SPECIES_DIR/predictions" ] || [ -z "$(ls -A "$SPECIES_DIR/predictions" 2>/dev/null)" ]; then
        echo "$(date): Generating predictions for $SPECIES in $SPECIES_DIR/predictions ..."
        srun bin/tacc \
          python scripts/predict.py create_predictions \
          --input $PREDICT_DIR/$SPECIES/sequences.zarr \
          --output-dir $SPECIES_DIR/predictions \
          --model-path $BASE_MODEL_PATH \
          --model-checkpoint $MODEL_CHECKPOINT \
          --dtype $DTYPE \
          --species-id $SPECIES_ID \
          --chromosome-id $CHR_ID \
          --batch-size 32
    else
        echo "$(date): Predictions already exist for $SPECIES, skipping generation"
    fi

    echo "$(date): Completed prediction generation for $SPECIES"
done

echo "$(date): Prediction generation $RUN_VERSION completed successfully!"
echo "$(date): Predictions available in: $PREDICT_DIR/{$(echo $SPECIES_LIST | sed 's/ /,/g')}/runs/$RUN_VERSION/$CHR_ID/predictions/"
