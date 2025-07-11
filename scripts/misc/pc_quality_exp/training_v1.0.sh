#!/bin/bash

# PC Quality Filter Experiment - Training v1.0
# Species: Athaliana only (fresh start)
# Run ID: v1.0

set -euo pipefail

echo "Starting PC Quality Filter Experiment - Training v1.0"
echo "Species: Athaliana only (fresh start)"
echo "Run ID: v1.0"
echo "$(date): Beginning training"

# Define paths
PIPE_DIR="/scratch/10459/eczech/data/dna/plant_caduceus_genome_annotation_task/pipeline"
MODEL_PATH="kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000"

# Ensure log directory exists
mkdir -p local/logs/exec

echo "$(date): Starting training for v1.0"
echo "Training data: $PIPE_DIR/prep/v1.0/splits/train.zarr"
echo "Validation data: $PIPE_DIR/prep/v1.0/splits/valid.zarr"
echo "Output directory: $PIPE_DIR/sweep"

# Run training with proper resource allocation and run-id
srun -p gh -N 16 -n 16 --tasks-per-node 1 -t 2:00:00 bin/tacc \
  python scripts/sweep.py run \
  --train-dataset "$PIPE_DIR/prep/v1.0/splits/train.zarr" \
  --val-dataset "$PIPE_DIR/prep/v1.0/splits/valid.zarr" \
  --output-dir "$PIPE_DIR/sweep" \
  --configuration-index 13 \
  --num-workers 16 --prefetch-factor 3 \
  --batch-size 8 --accumulate-grad-batches 1 \
  --train-eval-frequency 200 --val-check-interval 200 --limit-val-batches 1.0 \
  --checkpoint-frequency 200 --log-frequency 1 --enable-visualization yes \
  --epochs 3 --learning-rate-decay none \
  --num-nodes \$SLURM_NNODES --strategy ddp \
  --base-encoder-path "$MODEL_PATH" --torch-compile no \
  --project-name pc-genome-annot --run-name sweep-v1.0 \
  --run-id v1.0 \
  2>&1 | tee local/logs/exec/training_v1.0.log

echo "$(date): Training v1.0 completed!"
echo "Checkpoint should be available at: $PIPE_DIR/sweep/sweep-v1.0__cfg_013__arch_all__frzn_yes__lr_1e-04/pc-genome-annot/v1.0/checkpoints/last.ckpt" 