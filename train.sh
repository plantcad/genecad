#!/bin/bash
# =============================================================================
# GeneCAD Fine-Tuning Pipeline
# Fine-tunes the GeneCAD classifier head on a multi-species plant dataset.
#
# Usage:
#   bash train.sh [OPTIONS]
#
# Options:
#   -o, --output DIR        Output directory for checkpoints and logs
#                           (default: genecad_result/training)
#   -r, --run-name NAME     WandB run name  (default: genecad-plant-multispecies)
#   -p, --project NAME      WandB project   (default: genecad)
#   -g, --gpus N            Number of GPUs  (default: 2)
#   -b, --batch-size N      Per-GPU batch size (default: 4)
#   -l, --lr RATE           Learning rate   (default: 2e-4)
#   -h, --help              Show this message
#
# Requirements:
#   - Linux, CUDA 12, uv (https://docs.astral.sh/uv/)
#   - huggingface-cli login  (to download training data)
#   - WandB account          (optional; disable with WANDB_MODE=disabled)
# =============================================================================
set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    sed -n '3,21p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

OUTPUT_DIR="genecad_result/training"
RUN_NAME="genecad-plant-multispecies"
PROJECT_NAME="genecad"
NUM_GPUS=1
BATCH_SIZE=4
LEARNING_RATE=2e-4

while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--output)     OUTPUT_DIR="$2";    shift 2 ;;
    -r|--run-name)   RUN_NAME="$2";      shift 2 ;;
    -p|--project)    PROJECT_NAME="$2";  shift 2 ;;
    -g|--gpus)       NUM_GPUS="$2";      shift 2 ;;
    -b|--batch-size) BATCH_SIZE="$2";    shift 2 ;;
    -l|--lr)         LEARNING_RATE="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

# ── Fixed hyperparameters ─────────────────────────────────────────────────────
BASE_MODEL="emarro/pcad2-200M-cnet-baseline"   # Downloaded from HuggingFace
SPECIES_IDS="Athaliana Osativa Gmax Hvulgare Ptrichocarpa"
ACCUM_GRAD=384        # Effective batch = BATCH_SIZE * ACCUM_GRAD * NUM_GPUS
EPOCHS=1
ARCHITECTURE="all"
HEAD_LAYERS=8
TOKEN_EMBED_DIM=256
BASE_FROZEN="no"
NUM_WORKERS=8
VALID_PROPORTION=0.05

# ── Derived directories ───────────────────────────────────────────────────────
PIPELINE_DIR="$OUTPUT_DIR/pipeline"
EXTRACT_DIR="$PIPELINE_DIR/extract"
TRANSFORM_DIR="$PIPELINE_DIR/transform"
PREP_DIR="$PIPELINE_DIR/prep"
DATA_DIR="$PIPELINE_DIR/data"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"

PYTHON="uv run python"
export PYTHONPATH=.

# Download a single file from a HuggingFace dataset repo using the Python API
# Usage: hf_download <repo_id> <remote_path> <local_dir>
hf_download() {
    local repo_id="$1"
    local remote_path="$2"
    local local_dir="$3"
    uv run python -c "
import sys
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='${repo_id}',
    filename='${remote_path}',
    repo_type='dataset',
    local_dir='${local_dir}',
    local_dir_use_symlinks=False,
)
"
}

# ── Banner ────────────────────────────────────────────────────────────────────
echo "============================================"
echo " GeneCAD Fine-Tuning"
echo " Base model:  $BASE_MODEL"
echo " Species:     $SPECIES_IDS"
echo " Output:      $OUTPUT_DIR"
echo " GPUs:        $NUM_GPUS"
echo " Batch size:  $BATCH_SIZE (accum: $ACCUM_GRAD)"
echo " LR:          $LEARNING_RATE"
echo "============================================"

mkdir -p "$EXTRACT_DIR" "$TRANSFORM_DIR" "$PREP_DIR/splits" \
         "$DATA_DIR/gff" "$DATA_DIR/fasta" "$CHECKPOINT_DIR"

# =============================================================================
# Step 0: Download training data from HuggingFace
# =============================================================================
echo ""
echo "[0/8] Downloading training data from HuggingFace..."

HF_REPO="plantcad/genecad-dev"

for species in Athaliana Osativa Gmax Hvulgare Ptrichocarpa; do
  dst="$DATA_DIR/gff/${species}_top_transcript.gff3"
  if [ ! -f "$dst" ]; then
    echo "  Downloading GFF: $species"
    hf_download "$HF_REPO" \
      "data/gff/training/${species}_top_transcript.gff3" \
      "$DATA_DIR/gff"
    # hf_hub_download preserves the remote path structure; flatten to dst
    mv "$DATA_DIR/gff/data/gff/training/${species}_top_transcript.gff3" "$dst" 2>/dev/null || true
  else
    echo "  Skipped (exists): $species GFF"
  fi
done

declare -A FASTA_FILES
FASTA_FILES["Athaliana_447_TAIR10.fa.gz"]="data/fasta/training/Athaliana_447_TAIR10.fa.gz"
FASTA_FILES["Osativa_323_v7.0.fa.gz"]="data/fasta/training/Osativa_323_v7.0.fa.gz"
FASTA_FILES["Gmax_880_v6.0.fa.gz"]="data/fasta/training/Gmax_880_v6.0.fa.gz"
FASTA_FILES["Hvulgare_462_r1.fa.gz"]="data/fasta/training/Hvulgare_462_r1.fa.gz"
FASTA_FILES["Ptrichocarpa_533_v4.0.fa.gz"]="data/fasta/training/Ptrichocarpa_533_v4.0.fa.gz"

for fname in "${!FASTA_FILES[@]}"; do
  dst="$DATA_DIR/fasta/$fname"
  hf_path="${FASTA_FILES[$fname]}"
  if [ ! -f "$dst" ]; then
    echo "  Downloading FASTA: $fname"
    hf_download "$HF_REPO" "$hf_path" "$DATA_DIR/fasta"
    mv "$DATA_DIR/fasta/$hf_path" "$dst" 2>/dev/null || true
  else
    echo "  Skipped (exists): $fname"
  fi
done

# =============================================================================
# Step 1: Create symlinks with naming convention expected by src/config.py
# =============================================================================
echo ""
echo "[1/8] Setting up species data links..."

ln -sf "$DATA_DIR/gff/Athaliana_top_transcript.gff3"   "$DATA_DIR/gff/Athaliana_447_Araport11.gene.gff3"   2>/dev/null || true
ln -sf "$DATA_DIR/fasta/Athaliana_447_TAIR10.fa.gz"    "$DATA_DIR/fasta/Athaliana_447.fasta"               2>/dev/null || true

ln -sf "$DATA_DIR/gff/Osativa_top_transcript.gff3"     "$DATA_DIR/gff/Osativa_323_v7.0.gene.gff3"          2>/dev/null || true
ln -sf "$DATA_DIR/fasta/Osativa_323_v7.0.fa.gz"        "$DATA_DIR/fasta/Osativa_323.fasta"                  2>/dev/null || true

ln -sf "$DATA_DIR/gff/Gmax_top_transcript.gff3"        "$DATA_DIR/gff/Gmax_880_Wm82.a6.v1.gene.gff3"       2>/dev/null || true
ln -sf "$DATA_DIR/fasta/Gmax_880_v6.0.fa.gz"           "$DATA_DIR/fasta/Gmax_880_v6.0.fa.gz"               2>/dev/null || true

ln -sf "$DATA_DIR/gff/Hvulgare_top_transcript.gff3"    "$DATA_DIR/gff/HvulgareMorex_702_V3.gene.gff3"       2>/dev/null || true
ln -sf "$DATA_DIR/fasta/Hvulgare_462_r1.fa.gz"         "$DATA_DIR/fasta/HvulgareMorex_702_V3.fa.gz"         2>/dev/null || true

ln -sf "$DATA_DIR/gff/Ptrichocarpa_top_transcript.gff3" "$DATA_DIR/gff/Ptrichocarpa_533_v4.1.gene.gff3"    2>/dev/null || true
ln -sf "$DATA_DIR/fasta/Ptrichocarpa_533_v4.0.fa.gz"   "$DATA_DIR/fasta/Ptrichocarpa_533_v4.0.fa.gz"       2>/dev/null || true

echo "  Done."

# =============================================================================
# Step 2: Extract GFF features
# =============================================================================
echo ""
echo "[2/8] Extracting GFF features..."
if [ ! -f "$EXTRACT_DIR/raw_features.parquet" ]; then
  $PYTHON scripts/extract.py extract_gff_features \
    --input-dir "$DATA_DIR/gff" \
    --species-id $SPECIES_IDS \
    --output "$EXTRACT_DIR/raw_features.parquet"
  echo "  Created: raw_features.parquet"
else
  echo "  Skipped (already exists)"
fi

# =============================================================================
# Step 3: Extract and tokenize FASTA sequences
# =============================================================================
echo ""
echo "[3/8] Extracting and tokenizing FASTA sequences..."
if [ ! -d "$EXTRACT_DIR/tokens.zarr" ]; then
  $PYTHON scripts/extract.py extract_fasta_sequences \
    --input-dir "$DATA_DIR/fasta" \
    --species-id $SPECIES_IDS \
    --tokenizer-path "$BASE_MODEL" \
    --output "$EXTRACT_DIR/tokens.zarr"
  echo "  Created: tokens.zarr"
else
  echo "  Skipped (already exists)"
fi

# =============================================================================
# Step 4: Filter features
# =============================================================================
echo ""
echo "[4/8] Filtering features..."
if [ ! -f "$TRANSFORM_DIR/features.parquet" ]; then
  $PYTHON scripts/transform.py filter_features \
    --input "$EXTRACT_DIR/raw_features.parquet" \
    --output-features "$TRANSFORM_DIR/features.parquet" \
    --output-filters "$TRANSFORM_DIR/filters.parquet" \
    --remove-incomplete-features yes
  echo "  Created: features.parquet"
else
  echo "  Skipped (already exists)"
fi

# =============================================================================
# Step 5: Stack features
# =============================================================================
echo ""
echo "[5/8] Stacking features..."
if [ ! -f "$TRANSFORM_DIR/intervals.parquet" ]; then
  $PYTHON scripts/transform.py stack_features \
    --input "$TRANSFORM_DIR/features.parquet" \
    --output "$TRANSFORM_DIR/intervals.parquet"
  echo "  Created: intervals.parquet"
else
  echo "  Skipped (already exists)"
fi

# =============================================================================
# Step 6: Create labels
# =============================================================================
echo ""
echo "[6/8] Creating labels..."
if [ ! -d "$TRANSFORM_DIR/labels.zarr" ]; then
  $PYTHON scripts/transform.py create_labels \
    --input-features "$TRANSFORM_DIR/intervals.parquet" \
    --input-filters "$TRANSFORM_DIR/filters.parquet" \
    --output "$TRANSFORM_DIR/labels.zarr" \
    --remove-incomplete-features yes
  echo "  Created: labels.zarr"
else
  echo "  Skipped (already exists)"
fi

# =============================================================================
# Step 7: Create sequence dataset and training splits
# =============================================================================
echo ""
echo "[7/8] Creating sequence dataset and training splits..."
if [ ! -d "$TRANSFORM_DIR/sequences.zarr" ]; then
  $PYTHON scripts/transform.py create_sequence_dataset \
    --input-labels "$TRANSFORM_DIR/labels.zarr" \
    --input-tokens "$EXTRACT_DIR/tokens.zarr" \
    --output-path "$TRANSFORM_DIR/sequences.zarr" \
    --num-workers $NUM_WORKERS
  echo "  Created: sequences.zarr"
else
  echo "  Skipped (already exists)"
fi

if [ ! -d "$PREP_DIR/splits/train.zarr" ]; then
  $PYTHON scripts/sample.py generate_training_windows \
    --input "$TRANSFORM_DIR/sequences.zarr" \
    --output "$TRANSFORM_DIR/windows.zarr" \
    --intergenic-proportion 0.15 \
    --num-workers $NUM_WORKERS

  $PYTHON scripts/sample.py generate_training_splits \
    --input "$TRANSFORM_DIR/windows.zarr" \
    --train-output "$PREP_DIR/splits/train.zarr" \
    --valid-output "$PREP_DIR/splits/valid.zarr" \
    --valid-proportion $VALID_PROPORTION
  echo "  Created: train.zarr + valid.zarr"
else
  echo "  Skipped (already exists)"
fi

# =============================================================================
# Step 8: Train
# =============================================================================
echo ""
echo "[8/8] Starting training..."
echo "  Base model:     $BASE_MODEL"
echo "  Architecture:   $ARCHITECTURE (frozen=$BASE_FROZEN)"
echo "  Effective batch: $((BATCH_SIZE * ACCUM_GRAD * NUM_GPUS))"
echo "  Output:         $OUTPUT_DIR"
echo ""

$PYTHON scripts/train.py \
  --train-dataset "$PREP_DIR/splits/train.zarr" \
  --val-dataset   "$PREP_DIR/splits/valid.zarr" \
  --output-dir    "$CHECKPOINT_DIR" \
  --base-encoder-path    "$BASE_MODEL" \
  --base-encoder-frozen  "$BASE_FROZEN" \
  --architecture         "$ARCHITECTURE" \
  --head-encoder-layers  $HEAD_LAYERS \
  --token-embedding-dim  $TOKEN_EMBED_DIM \
  --batch-size           $BATCH_SIZE \
  --accumulate-grad-batches $ACCUM_GRAD \
  --epochs               $EPOCHS \
  --learning-rate        $LEARNING_RATE \
  --learning-rate-decay  cosine \
  --num-workers          $NUM_WORKERS \
  --prefetch-factor      2 \
  --gpu                  $NUM_GPUS \
  --strategy             ddp \
  --checkpoint-frequency 1000 \
  --val-check-interval   1000 \
  --train-eval-frequency 1000 \
  --limit-val-batches    1.0 \
  --log-frequency        1 \
  --enable-visualization no \
  --torch-compile        no \
  --auto-class-weights   yes \
  --project-name         "$PROJECT_NAME" \
  --run-name             "$RUN_NAME" \
  2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "============================================"
echo " Training complete!"
echo " Checkpoints: $CHECKPOINT_DIR"
echo " Log:         $OUTPUT_DIR/training.log"
echo "============================================"
