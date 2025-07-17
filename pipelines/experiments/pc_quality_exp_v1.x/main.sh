#!/bin/bash

# PC Quality Filter Experiment - Main Orchestration Script
# Usage: Run commands manually and incrementally

set -euo pipefail

# Define experiment directory path
EXPERIMENT_DIR="pipelines/experiments/pc_quality_exp_v1.x"

# Source environment initialization
source $EXPERIMENT_DIR/init.sh

# Define log directory path
LOG_DIR="local/logs/pc_quality_exp"

echo "Starting PC Quality Filter Experiment Pipeline"
echo "$(date): Beginning main orchestration"

# Ensure log directory exists
mkdir -p $LOG_DIR

echo "$(date): =================================="
echo "$(date): PHASE 1: Data Preparation"
echo "$(date): =================================="

# Allocate compute node for data preparation
idev -p gg -N 1 -n 1 -t 2:00:00

# Data preparation v1.0
$EXPERIMENT_DIR/prepare.sh 1.0 2>&1 | tee -a $LOG_DIR/prepare_v1.0.log

# Data preparation v1.1
$EXPERIMENT_DIR/prepare.sh 1.1 2>&1 | tee -a $LOG_DIR/prepare_v1.1.log

# Data preparation v1.2
$EXPERIMENT_DIR/prepare.sh 1.2 2>&1 | tee -a $LOG_DIR/prepare_v1.2.log

echo "$(date): Data preparation phase completed successfully!"

echo "$(date): =================================="
echo "$(date): PHASE 2: Training"
echo "$(date): =================================="

# Training v1.0 (Athaliana only - 16 nodes, 2 hours)
$EXPERIMENT_DIR/train.sh 1.0

# Training v1.1 (Athaliana + Osativa - 16 nodes, 2 hours)
$EXPERIMENT_DIR/train.sh 1.1

# Training v1.2 (All 5 species from v1.1 checkpoint - 16 nodes, 8 hours)
$EXPERIMENT_DIR/train.sh 1.2

echo "$(date): Training phase completed successfully!"

echo "$(date): =================================="
echo "$(date): PHASE 3: Sequence Extraction"
echo "$(date): =================================="

# Allocate compute node for sequence extraction
idev -p gg -N 1 -n 1 -t 2:00:00

# Extract FASTA sequences for testing species (jregia, pvulgaris)
$EXPERIMENT_DIR/extract.sh 2>&1 | tee -a $LOG_DIR/extract.log

echo "$(date): Sequence extraction phase completed successfully!"

echo "$(date): =================================="
echo "$(date): PHASE 4: Prediction Generation"
echo "$(date): =================================="

# Allocate GPU nodes for prediction generation
idev -p gh -N 20 -n 20 --tasks-per-node 1 -t 2:00:00

# Prediction generation v1.0
$EXPERIMENT_DIR/predict.sh 1.0 2>&1 | tee -a $LOG_DIR/predict_v1.0.log

# Prediction generation v1.1
$EXPERIMENT_DIR/predict.sh 1.1 2>&1 | tee -a $LOG_DIR/predict_v1.1.log

# Prediction generation v1.2
$EXPERIMENT_DIR/predict.sh 1.2 2>&1 | tee -a $LOG_DIR/predict_v1.2.log

echo "$(date): Prediction generation phase completed successfully!"

echo "$(date): =================================="
echo "$(date): PHASE 5: Evaluation"
echo "$(date): =================================="

# Allocate compute node for evaluation
idev -p gg -N 1 -n 1 -t 2:00:00

# Evaluation v1.0 with original ground truth
$EXPERIMENT_DIR/evaluate.sh 1.0 original 2>&1 | tee -a $LOG_DIR/evaluate_v1.0_original.log

# Evaluation v1.0 with PC-filtered ground truth
$EXPERIMENT_DIR/evaluate.sh 1.0 pc-filtered 2>&1 | tee -a $LOG_DIR/evaluate_v1.0_pc-filtered.log

# Evaluation v1.1 with original ground truth
$EXPERIMENT_DIR/evaluate.sh 1.1 original 2>&1 | tee -a $LOG_DIR/evaluate_v1.1_original.log

# Evaluation v1.1 with PC-filtered ground truth
$EXPERIMENT_DIR/evaluate.sh 1.1 pc-filtered 2>&1 | tee -a $LOG_DIR/evaluate_v1.1_pc-filtered.log

# Evaluation v1.2 with original ground truth
$EXPERIMENT_DIR/evaluate.sh 1.2 original 2>&1 | tee -a $LOG_DIR/evaluate_v1.2_original.log

# Evaluation v1.2 with PC-filtered ground truth
$EXPERIMENT_DIR/evaluate.sh 1.2 pc-filtered 2>&1 | tee -a $LOG_DIR/evaluate_v1.2_pc-filtered.log

echo "$(date): Evaluation phase completed successfully!"
echo "$(date): PC Quality Filter Experiment pipeline completed!"
