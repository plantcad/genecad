#!/bin/bash

# PC Quality Filter Experiment - Unified Training Script
# Usage: ./train.sh {1.0|1.1|1.2}

set -euo pipefail

# Parse arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 {1.0|1.1|1.2|1.3}"
    echo "  1.0 - Athaliana only (fresh start)"
    echo "  1.1 - Athaliana + Osativa (fresh start)"
    echo "  1.2 - Athaliana + Osativa + Gmax + Hvulgare + Ptrichocarpa (initialize from v1.1 checkpoint)"
    echo "  1.3 - Athaliana + Osativa with randomized base encoder (fresh start)"
    exit 1
fi

VERSION="$1"

# Validate version
if [ "$VERSION" != "1.0" ] && [ "$VERSION" != "1.1" ] && [ "$VERSION" != "1.2" ] && [ "$VERSION" != "1.3" ]; then
    echo "ERROR: Invalid version '$VERSION'"
    echo "Must be '1.0', '1.1', '1.2', or '1.3'"
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
        ;;
    "1.1")
        SPECIES_DESCRIPTION="Athaliana + Osativa (fresh start)"
        N_NODES=16
        TIME_LIMIT="2:00:00"
        EPOCHS=3
        CONFIG_INDEX=13
        ;;
    "1.2")
        SPECIES_DESCRIPTION="Athaliana + Osativa + Gmax + Hvulgare + Ptrichocarpa (initialize from v1.1 checkpoint)"
        N_NODES=16
        TIME_LIMIT="8:00:00"
        EPOCHS=1
        CONFIG_INDEX=13
        ;;
    "1.3")
        SPECIES_DESCRIPTION="Athaliana + Osativa with randomized base encoder (fresh start)"
        N_NODES=16
        TIME_LIMIT="2:00:00"
        EPOCHS=3
        CONFIG_INDEX=16
        ;;
esac

echo "Starting PC Quality Filter Experiment - Training v$VERSION"
echo "Species: $SPECIES_DESCRIPTION"
echo "Run ID: v$VERSION"
echo "$(date): Beginning training"

# Define paths

# Set checkpoint path for v1.2 if needed
if [ "$VERSION" = "1.2" ]; then
    V1_1_CHECKPOINT="$PIPE_DIR/sweep/sweep-v1.1__cfg_013__rand_no__arch_all__frzn_yes__lr_1e-04/checkpoints/last.ckpt"
fi

# Copy training data from v1.1 for v1.3 if it doesn't exist
if [ "$VERSION" = "1.3" ]; then
    if [ ! -d "$PIPE_DIR/prep/v1.3" ]; then
        echo "$(date): Creating v1.3 training data directory (symlinked from v1.1)"
        mkdir -p "$PIPE_DIR/prep/v1.3"
        ln -sf "$PIPE_DIR/prep/v1.1/splits" "$PIPE_DIR/prep/v1.3/splits"
        echo "$(date): Symlinked v1.1 training data to v1.3"
    else
        echo "$(date): v1.3 training data already exists"
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
  --base-encoder-path "$MODEL_PATH" --torch-compile no \
  --project-name pc-genome-annot --run-name sweep-v$VERSION \
  $([ "$VERSION" = "1.2" ] && echo "--checkpoint $V1_1_CHECKPOINT --checkpoint-type model")

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
