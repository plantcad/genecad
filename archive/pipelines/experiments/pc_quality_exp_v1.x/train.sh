#!/bin/bash

# PC Quality Filter Experiment - Unified Training Script
# Usage: ./train.sh {1.0|1.1|1.2|1.3|1.4|1.5}

set -euo pipefail

# Parse arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 {1.0|1.1|1.2|1.3|1.4|1.5}"
    echo "  1.0 - Athaliana only (fresh start)"
    echo "  1.1 - Athaliana + Osativa (fresh start)"
    echo "  1.2 - Athaliana + Osativa + Gmax + Hvulgare + Ptrichocarpa (initialize from v1.1 checkpoint)"
    echo "  1.3 - Athaliana + Osativa with randomized base encoder (fresh start)"
    echo "  1.4 - Athaliana + Osativa with large PlantCAD base model (fresh start)"
    echo "  1.5 - All 5 species (initialize from v1.4 checkpoint, large base model)"
    exit 1
fi

VERSION="$1"

# Validate version
if [ "$VERSION" != "1.0" ] && [ "$VERSION" != "1.1" ] && [ "$VERSION" != "1.2" ] && [ "$VERSION" != "1.3" ] && [ "$VERSION" != "1.4" ] && [ "$VERSION" != "1.5" ]; then
    echo "ERROR: Invalid version '$VERSION'"
    echo "Must be '1.0', '1.1', '1.2', '1.3', '1.4', or '1.5'"
    exit 1
fi

# Set version-specific configurations
case "$VERSION" in
    "1.0")
        SPECIES_DESCRIPTION="Athaliana only (fresh start)"
        N_NODES=16
        TIME_LIMIT="2:00:00"
        EPOCHS=3
        CONFIG_INDEX=13
        BASE_MODEL_PATH="kuleshov-group/PlantCAD2-Small-l24-d0768"
        ;;
    "1.1")
        SPECIES_DESCRIPTION="Athaliana + Osativa (fresh start)"
        N_NODES=16
        TIME_LIMIT="2:00:00"
        EPOCHS=3
        CONFIG_INDEX=13
        BASE_MODEL_PATH="kuleshov-group/PlantCAD2-Small-l24-d0768"
        ;;
    "1.2")
        SPECIES_DESCRIPTION="Athaliana + Osativa + Gmax + Hvulgare + Ptrichocarpa (initialize from v1.1 checkpoint)"
        N_NODES=16
        TIME_LIMIT="8:00:00"
        EPOCHS=1
        CONFIG_INDEX=13
        BASE_MODEL_PATH="kuleshov-group/PlantCAD2-Small-l24-d0768"
        ;;
    "1.3")
        SPECIES_DESCRIPTION="Athaliana + Osativa with randomized base encoder (fresh start)"
        N_NODES=16
        TIME_LIMIT="2:00:00"
        EPOCHS=3
        CONFIG_INDEX=16
        BASE_MODEL_PATH="kuleshov-group/PlantCAD2-Small-l24-d0768"
        ;;
    "1.4")
        SPECIES_DESCRIPTION="Athaliana + Osativa with large PlantCAD base model (fresh start)"
        N_NODES=16
        TIME_LIMIT="8:00:00"
        EPOCHS=3
        CONFIG_INDEX=13
        BASE_MODEL_PATH="kuleshov-group/PlantCAD2-Large-l48-d1536"
        ;;
    "1.5")
        SPECIES_DESCRIPTION="All 5 species (initialize from v1.4 checkpoint, large base model)"
        N_NODES=16
        TIME_LIMIT="8:00:00"
        EPOCHS=1
        CONFIG_INDEX=13
        BASE_MODEL_PATH="kuleshov-group/PlantCAD2-Large-l48-d1536"
        ;;
esac

echo "Starting PC Quality Filter Experiment - Training v$VERSION"
echo "Species: $SPECIES_DESCRIPTION"
echo "Run ID: v$VERSION"
echo "$(date): Beginning training"

# Set checkpoint paths if needed
if [ "$VERSION" = "1.2" ]; then
    V1_1_CHECKPOINT="$PIPE_DIR/sweep/sweep-v1.1__cfg_013__rand_no__arch_all__frzn_yes__lr_1e-04/checkpoints/last.ckpt"
fi

if [ "$VERSION" = "1.5" ]; then
    V1_4_CHECKPOINT="$PIPE_DIR/sweep/sweep-v1.4__cfg_013__rand_no__arch_all__frzn_yes__lr_1e-04/checkpoints/last.ckpt"
fi

# Copy training data from v1.1 for v1.3 and v1.4 if it doesn't exist
if [ "$VERSION" = "1.3" ] || [ "$VERSION" = "1.4" ]; then
    if [ ! -d "$PIPE_DIR/prep/v$VERSION" ]; then
        echo "$(date): Creating v$VERSION training data directory (symlinked from v1.1)"
        mkdir -p "$PIPE_DIR/prep/v$VERSION"
        ln -sf "$PIPE_DIR/prep/v1.1/splits" "$PIPE_DIR/prep/v$VERSION/splits"
        echo "$(date): Symlinked v1.1 training data to v$VERSION"
    else
        echo "$(date): v$VERSION training data already exists"
    fi
fi

# Copy training data from v1.2 for v1.5 if it doesn't exist
if [ "$VERSION" = "1.5" ]; then
    if [ ! -d "$PIPE_DIR/prep/v$VERSION" ]; then
        echo "$(date): Creating v$VERSION training data directory (symlinked from v1.2)"
        mkdir -p "$PIPE_DIR/prep/v$VERSION"
        ln -sf "$PIPE_DIR/prep/v1.2/splits" "$PIPE_DIR/prep/v$VERSION/splits"
        echo "$(date): Symlinked v1.2 training data to v$VERSION"
    else
        echo "$(date): v$VERSION training data already exists"
    fi
fi

# Ensure log directory exists
mkdir -p local/logs/exec

echo "$(date): Starting training for v$VERSION"
echo "Training data: $PIPE_DIR/prep/v$VERSION/splits/train.zarr"
echo "Validation data: $PIPE_DIR/prep/v$VERSION/splits/valid.zarr"
echo "Output directory: $PIPE_DIR/sweep"

# Check for v1.1 checkpoint if running v1.2
if [ "$VERSION" = "1.2" ]; then
    echo "Loading checkpoint from: $V1_1_CHECKPOINT"
    if [ ! -f "$V1_1_CHECKPOINT" ]; then
        echo "ERROR: v1.1 checkpoint not found at $V1_1_CHECKPOINT"
        echo "Please ensure v1.1 training has completed successfully before running v1.2"
        exit 1
    fi
    echo "$(date): Checkpoint verified"
fi

# Check for v1.4 checkpoint if running v1.5
if [ "$VERSION" = "1.5" ]; then
    echo "Loading checkpoint from: $V1_4_CHECKPOINT"
    if [ ! -f "$V1_4_CHECKPOINT" ]; then
        echo "ERROR: v1.4 checkpoint not found at $V1_4_CHECKPOINT"
        echo "Please ensure v1.4 training has completed successfully before running v1.5"
        exit 1
    fi
    echo "$(date): Checkpoint verified"
fi

# Define paths for run ID tracking and cleanup
RUN_OUTPUT_DIR="$PIPE_DIR/sweep/sweep-v${VERSION}__cfg_$(printf "%03d" $CONFIG_INDEX)__rand_no__arch_all__frzn_yes__lr_1e-04"
if [ "$VERSION" = "1.3" ]; then
    RUN_OUTPUT_DIR="$PIPE_DIR/sweep/sweep-v${VERSION}__cfg_$(printf "%03d" $CONFIG_INDEX)__rand_yes__arch_all__frzn_yes__lr_1e-04"
fi
CHECKPOINT_DIR="$RUN_OUTPUT_DIR/checkpoints"

# Clean up existing checkpoint directory if it exists
echo "$(date): Using checkpoint directory: $CHECKPOINT_DIR"
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "$(date): Removing existing checkpoint directory: $CHECKPOINT_DIR"
    rm -rf "$CHECKPOINT_DIR"
fi

# Run training with proper resource allocation and run-id
srun -p gh -N $N_NODES -n $N_NODES --tasks-per-node 1 -t $TIME_LIMIT \
  --output local/logs/pc_quality_exp/train_v$VERSION.log \
  --error local/logs/pc_quality_exp/train_v$VERSION.log \
  bin/tacc \
python scripts/sweep.py run \
  --train-dataset "$PIPE_DIR/prep/v$VERSION/splits/train.zarr" \
  --val-dataset "$PIPE_DIR/prep/v$VERSION/splits/valid.zarr" \
  --output-dir "$PIPE_DIR/sweep" \
  --configuration-index $CONFIG_INDEX \
  --num-workers 16 --prefetch-factor 3 \
  --batch-size 8 --accumulate-grad-batches 1 \
  --train-eval-frequency 200 --val-check-interval 200 --limit-val-batches 1.0 \
  --checkpoint-frequency 200 --log-frequency 1 --enable-visualization yes \
  --epochs $EPOCHS --learning-rate-decay none \
  --num-nodes $N_NODES --strategy ddp \
  --base-encoder-path "$BASE_MODEL_PATH" --torch-compile no \
  --project-name pc-genome-annot --run-name sweep-v$VERSION \
  $([ "$VERSION" = "1.2" ] && echo "--checkpoint $V1_1_CHECKPOINT --checkpoint-type model") \
  $([ "$VERSION" = "1.5" ] && echo "--checkpoint $V1_4_CHECKPOINT --checkpoint-type model")

echo "$(date): Training v$VERSION completed!"

# Verify checkpoint was created
FINAL_CHECKPOINT="$CHECKPOINT_DIR/last.ckpt"
if [ -f "$FINAL_CHECKPOINT" ]; then
    echo "$(date): ✓ Checkpoint successfully created at: $FINAL_CHECKPOINT"
else
    echo "$(date): ✗ ERROR: Expected checkpoint not found at: $FINAL_CHECKPOINT"
    echo "$(date): Available files in checkpoint directory:"
    ls -la "$CHECKPOINT_DIR/" 2>/dev/null || echo "Checkpoint directory does not exist"
    exit 1
fi
