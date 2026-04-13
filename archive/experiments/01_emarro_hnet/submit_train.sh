#!/bin/bash
#SBATCH -J genecad_train_emarro
#SBATCH -o /work/10373/zongyan/vista/genecad_project/logs/%x-%A_%a.log
#SBATCH -p gh
#SBATCH -N 4                 # 4 nodes per task
#SBATCH -n 4                 # 4 tasks total
#SBATCH -t 06:00:00          # 6 hours
#SBATCH -A MCB24097

set -euo pipefail

# Load required modules on TACC (adjust if TACC uses different module names)
# module load python3
# module load cuda/12.0

# Set dynamic paths for the job
export GENECAD_DIR="/work/10373/zongyan/vista/genecad_project/genecad"
export WORK_DIR="/work/10373/zongyan/vista/genecad_project/genecad_result"
export RAW_GFF_DIR="/work/10373/zongyan/vista/genecad_project/plantcad_architecture_tests/zero_shot_filter/gff_filtered_top_transcripts"
export RAW_FASTA_DIR="/work/10373/zongyan/vista/genecad_project/plantcad_architecture_tests/training_data/fasta/raw_fastas"

# Python executable (assuming venv is in genecad/.venv)
export PYTHON="${GENECAD_DIR}/.venv/bin/python"

# Determine GPUs per node (TACC gh partition usually has multiple GPUs per node)
export NUM_GPUS=4 # usually 4 for gh partition, adjust if needed

echo "============================================"
echo " Starting GeneCAD Training Job on TACC"
echo " Time: $(date)"
echo " Job ID: $SLURM_JOB_ID"
echo " Nodes: $SLURM_JOB_NODELIST"
echo "============================================"

# Using srun to execute the training script across the allocated nodes
cd "$GENECAD_DIR"
srun "$GENECAD_DIR/experiments/01_emarro_hnet/train.sh"

echo "============================================"
echo " Job Complete"
echo " Time: $(date)"
echo "============================================"
