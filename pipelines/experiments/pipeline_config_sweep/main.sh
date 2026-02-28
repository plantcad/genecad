#!/bin/bash

# GeneCAD Pipeline Configuration Sweep - Main Orchestration Script
# Usage: Run commands manually and incrementally

set -euo pipefail

# Initialization
# -------------------------------------------------
# Define experiment directory path
EXPERIMENT_DIR="pipelines/experiments/pipeline_config_sweep"

# Source environment initialization
source $EXPERIMENT_DIR/init.sh

# Define log directory path
LOG_DIR="local/logs/pipeline_config_sweep"

# Define model versions to process
MODEL_VERSIONS="1.0 1.1 1.2 1.3 1.4 1.5"
# -------------------------------------------------

echo "Starting GeneCAD Pipeline Configuration Sweep"
echo "$(date): Beginning main orchestration"

# Ensure log directory exists
mkdir -p $LOG_DIR

echo "$(date): =================================="
echo "$(date): PHASE 1: Data Preparation"
echo "$(date): =================================="

for version in $MODEL_VERSIONS; do
# Skip v1.3 and v1.4 because they use the same training data as v1.1
if [ "$version" != "1.3" ] && [ "$version" != "1.4" ]; then
    sbatch -p gg -N 1 -n 1 -t 2:00:00 \
        --output $LOG_DIR/prepare_v${version}.log --error $LOG_DIR/prepare_v${version}.log \
        $EXPERIMENT_DIR/prepare.sh $version
fi
done

echo "$(date): Data preparation phase completed successfully!"

echo "$(date): =================================="
echo "$(date): PHASE 2: Training"
echo "$(date): =================================="

# Training v1.0 (Athaliana only - 3 epochs, 1 hr 45 min)
$EXPERIMENT_DIR/train.sh 1.0

# Training v1.1 (Athaliana + Osativa - 3 epochs, 1 hr 45 min)
$EXPERIMENT_DIR/train.sh 1.1

# Training v1.2 (All 5 species from v1.1 checkpoint - 1 epoch, 3 hours)
$EXPERIMENT_DIR/train.sh 1.2

# Training v1.3 (Athaliana + Osativa with randomized base encoder - 3 epochs, 1 hr 45 min)
$EXPERIMENT_DIR/train.sh 1.3

# Training v1.4 (Athaliana + Osativa with large PlantCAD base model - 3 epochs, 3 hr 20 min)
$EXPERIMENT_DIR/train.sh 1.4

# Training v1.5 (All 5 species from v1.4 checkpoint with large base model - 1 epoch, 5 hr 10 min)
$EXPERIMENT_DIR/train.sh 1.5

echo "$(date): Training phase completed successfully!"

echo "$(date): =================================="
echo "$(date): PHASE 3: Sequence Extraction"
echo "$(date): =================================="

# Allocate compute node for sequence extraction
idev -p gg -N 1 -n 1 -t 2:00:00

# Extract FASTA sequences for testing species
$EXPERIMENT_DIR/extract.sh 2>&1 | tee -a $LOG_DIR/extract.log

echo "$(date): Sequence extraction phase completed successfully!"

echo "$(date): =================================="
echo "$(date): PHASE 4: Prediction Generation"
echo "$(date): =================================="

for version in $MODEL_VERSIONS; do
sbatch -p gh -N 20 -n 20 -t 2:00:00 \
    --output $LOG_DIR/predict_v${version}.log --error $LOG_DIR/predict_v${version}.log \
    $EXPERIMENT_DIR/predict.sh $version
done

echo "$(date): Prediction generation phase completed successfully!"

echo "$(date): =================================="
echo "$(date): PHASE 5: Evaluation"
echo "$(date): =================================="

for version in $MODEL_VERSIONS; do
for ground_truth in original pc-filtered; do
for decoding_method in viterbi; do
sbatch -p gg -N 1 -n 1 -t 2:00:00 \
    --output $LOG_DIR/evaluate_v${version}_${ground_truth}_${decoding_method}.log --error $LOG_DIR/evaluate_v${version}_${ground_truth}_${decoding_method}.log \
    $EXPERIMENT_DIR/evaluate.sh $version $ground_truth $decoding_method
done
done
done

# Interactive execution:
# idev -p gg -N 1 -n 1 -t 2:00:00
# $EXPERIMENT_DIR/evaluate.sh 1.0 original 2>&1 | tee -a $LOG_DIR/evaluate_v1.0_original.log

echo "$(date): Evaluation phase completed successfully!"
echo "$(date): GeneCAD Pipeline Configuration Sweep completed!"
