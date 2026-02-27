#!/bin/bash
# Fine-tune GeneCAD classifier on Emarro HNet base encoder
# Usage: ./run_genecad_training.sh
set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Emarro HNet base model (from HuggingFace)
BASE_MODEL="emarro/pcad2-200M-cnet-baseline"

# Training data paths (Set these to TACC locations before running)
DATA_ROOT="${DATA_ROOT:-/scratch/10373/zongyan/genecad_data}"
SPECIES_IDS="Zmays Athaliana Gmax Hvulgare Osativa Ptrichocarpa"

# Pipeline directories
WORK_DIR="${WORK_DIR:-/workdir/zl843/GeneCAD/genecad_result}"
PIPELINE_DIR="$WORK_DIR/training_emarro"
EXTRACT_DIR="$PIPELINE_DIR/extract"
TRANSFORM_DIR="$PIPELINE_DIR/transform"
PREP_DIR="$PIPELINE_DIR/prep"

# Training hyperparameters
BATCH_SIZE=4          # Per-GPU batch size (reduce if OOM)
ACCUM_GRAD=4          # Effective batch size = BATCH_SIZE * ACCUM_GRAD * NUM_GPUS = 4*4*2 = 32
EPOCHS=1
LEARNING_RATE=1e-4
ARCHITECTURE="all"    # encoder-only | sequence-only | classifier-only | all
HEAD_LAYERS=8
TOKEN_EMBED_DIM=128   # Reduced when using "all" architecture
BASE_FROZEN="yes"     # Freeze Emarro base encoder
NUM_WORKERS=4
VALID_PROPORTION=0.05

# Output
OUTPUT_DIR="$PIPELINE_DIR/training_output"
RUN_NAME="emarro-hnet-6-species-v1"
PROJECT_NAME="genecad-emarro"

# Python executable
PYTHON="${PYTHON:-/workdir/zl843/GeneCAD/genecad/.venv/bin/python}"

echo "============================================"
echo " GeneCAD Fine-Tuning on Emarro HNet"
echo " Base model: $BASE_MODEL"
echo " Species: $SPECIES_IDS"
echo "============================================"

cd "${GENECAD_DIR:-/workdir/zl843/GeneCAD/genecad}"
export PYTHONPATH=.

# =============================================================================
# Step 0: Prepare data directories
# =============================================================================
echo ""
echo "[0/8] Preparing data directories..."
mkdir -p "$EXTRACT_DIR" "$TRANSFORM_DIR" "$PREP_DIR/splits" "$OUTPUT_DIR"

# Create symlinked data directories that match expected structure
# GFF files must be in a 'gff' subdirectory
# FASTA files must be in a 'fasta' subdirectory
DATA_DIR="$PIPELINE_DIR/data"
mkdir -p "$DATA_DIR/gff" "$DATA_DIR/fasta"

# Check and link files
declare -A GFF_EXPECTED
declare -A FASTA_EXPECTED

GFF_EXPECTED[Zmays]="Zea_mays-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3"
FASTA_EXPECTED[Zmays]="Zea_mays-B73-REFERENCE-NAM-5.0.fa"

GFF_EXPECTED[Athaliana]="Athaliana_447_Araport11.gene.gff3"
FASTA_EXPECTED[Athaliana]="Athaliana_447.fasta"

GFF_EXPECTED[Gmax]="Gmax_880_Wm82.a6.v1.gene.gff3"
FASTA_EXPECTED[Gmax]="Gmax_880_v6.0.fa"

GFF_EXPECTED[Hvulgare]="HvulgareMorex_702_V3.gene.gff3"
FASTA_EXPECTED[Hvulgare]="HvulgareMorex_702_V3.fa"

GFF_EXPECTED[Osativa]="Osativa_323_v7.0.gene.gff3"
FASTA_EXPECTED[Osativa]="Osativa_323.fasta"

GFF_EXPECTED[Ptrichocarpa]="Ptrichocarpa_533_v4.1.gene.gff3"
FASTA_EXPECTED[Ptrichocarpa]="Ptrichocarpa_533_v4.0.fa"

for species in $SPECIES_IDS; do
    SRC_GFF="$DATA_ROOT/$species/${species}_top_transcript.gff3"
    SRC_FASTA="$DATA_ROOT/$species/${species}.fa"
    
    if [ ! -f "$SRC_FASTA" ]; then
        SRC_FASTA="$DATA_ROOT/$species/${species}.fasta"
    fi

    if [ ! -f "$SRC_GFF" ]; then
        echo "ERROR: GFF not found for $species at $SRC_GFF"
        exit 1
    fi
    if [ ! -f "$SRC_FASTA" ]; then
        echo "ERROR: FASTA not found for $species at $SRC_FASTA"
        exit 1
    fi

    DST_GFF="$DATA_DIR/gff/${GFF_EXPECTED[$species]}"
    DST_FASTA="$DATA_DIR/fasta/${FASTA_EXPECTED[$species]}"

    ln -sf "$SRC_GFF" "$DST_GFF"
    ln -sf "$SRC_FASTA" "$DST_FASTA"

    echo "  GFF:   $SRC_GFF -> $DST_GFF"
    echo "  FASTA: $SRC_FASTA -> $DST_FASTA"
done

# =============================================================================
# Step 1: Extract GFF features
# =============================================================================
echo ""
echo "[1/8] Extracting GFF features..."
if [ ! -f "$EXTRACT_DIR/raw_features.parquet" ]; then
    $PYTHON scripts/extract.py extract_gff_features \
        --input-dir "$DATA_DIR/gff" \
        --species-id $SPECIES_IDS \
        --output "$EXTRACT_DIR/raw_features.parquet"
    echo "  Created: $EXTRACT_DIR/raw_features.parquet"
else
    echo "  Skipped (already exists)"
fi

# =============================================================================
# Step 2: Extract and tokenize FASTA sequences
# =============================================================================
echo ""
echo "[2/8] Extracting and tokenizing FASTA sequences..."
if [ ! -d "$EXTRACT_DIR/tokens.zarr" ]; then
    $PYTHON scripts/extract.py extract_fasta_sequences \
        --input-dir "$DATA_DIR/fasta" \
        --species-id $SPECIES_IDS \
        --tokenizer-path "$BASE_MODEL" \
        --output "$EXTRACT_DIR/tokens.zarr"
    echo "  Created: $EXTRACT_DIR/tokens.zarr"
else
    echo "  Skipped (already exists)"
fi

# =============================================================================
# Step 3: Filter features
# =============================================================================
echo ""
echo "[3/8] Filtering features..."
if [ ! -f "$TRANSFORM_DIR/features.parquet" ]; then
    $PYTHON scripts/transform.py filter_features \
        --input "$EXTRACT_DIR/raw_features.parquet" \
        --output-features "$TRANSFORM_DIR/features.parquet" \
        --output-filters "$TRANSFORM_DIR/filters.parquet" \
        --remove-incomplete-features yes
    echo "  Created: $TRANSFORM_DIR/features.parquet"
else
    echo "  Skipped (already exists)"
fi

# =============================================================================
# Step 4: Stack features
# =============================================================================
echo ""
echo "[4/8] Stacking features..."
if [ ! -f "$TRANSFORM_DIR/intervals.parquet" ]; then
    $PYTHON scripts/transform.py stack_features \
        --input "$TRANSFORM_DIR/features.parquet" \
        --output "$TRANSFORM_DIR/intervals.parquet"
    echo "  Created: $TRANSFORM_DIR/intervals.parquet"
else
    echo "  Skipped (already exists)"
fi

# =============================================================================
# Step 5: Create labels
# =============================================================================
echo ""
echo "[5/8] Creating labels..."
if [ ! -d "$TRANSFORM_DIR/labels.zarr" ]; then
    $PYTHON scripts/transform.py create_labels \
        --input-features "$TRANSFORM_DIR/intervals.parquet" \
        --input-filters "$TRANSFORM_DIR/filters.parquet" \
        --output "$TRANSFORM_DIR/labels.zarr" \
        --remove-incomplete-features yes
    echo "  Created: $TRANSFORM_DIR/labels.zarr"
else
    echo "  Skipped (already exists)"
fi

# =============================================================================
# Step 6: Create sequence dataset (join labels + tokens)
# =============================================================================
echo ""
echo "[6/8] Creating sequence dataset..."
if [ ! -d "$TRANSFORM_DIR/sequences.zarr" ]; then
    $PYTHON scripts/transform.py create_sequence_dataset \
        --input-labels "$TRANSFORM_DIR/labels.zarr" \
        --input-tokens "$EXTRACT_DIR/tokens.zarr" \
        --output-path "$TRANSFORM_DIR/sequences.zarr" \
        --num-workers $NUM_WORKERS
    echo "  Created: $TRANSFORM_DIR/sequences.zarr"
else
    echo "  Skipped (already exists)"
fi

# =============================================================================
# Step 7: Generate training windows and splits
# =============================================================================
echo ""
echo "[7/8] Generating training windows and splits..."
if [ ! -d "$TRANSFORM_DIR/windows.zarr" ]; then
    $PYTHON scripts/sample.py generate_training_windows \
        --input "$TRANSFORM_DIR/sequences.zarr" \
        --output "$TRANSFORM_DIR/windows.zarr" \
        --intergenic-proportion 0.3 \
        --num-workers $NUM_WORKERS
    echo "  Created: $TRANSFORM_DIR/windows.zarr"
fi

if [ ! -d "$PREP_DIR/splits/train.zarr" ]; then
    $PYTHON scripts/sample.py generate_training_splits \
        --input "$TRANSFORM_DIR/windows.zarr" \
        --train-output "$PREP_DIR/splits/train.zarr" \
        --valid-output "$PREP_DIR/splits/valid.zarr" \
        --valid-proportion $VALID_PROPORTION
    echo "  Created: $PREP_DIR/splits/train.zarr + valid.zarr"
else
    echo "  Skipped (already exists)"
fi

# =============================================================================
# Step 8: Run training
# =============================================================================
echo ""
echo "[8/8] Starting training..."

# Checkpoint resumption logic
CHECKPOINT_ARGS=""
LATEST_CKPT=$(ls -t "$OUTPUT_DIR/checkpoints"/*.ckpt 2>/dev/null | head -n 1 || true)
if [ -n "$LATEST_CKPT" ]; then
    echo "  Found checkpoint to resume: $LATEST_CKPT"
    CHECKPOINT_ARGS="--checkpoint $LATEST_CKPT --checkpoint-type trainer"
else
    echo "  No checkpoint found. Starting fresh."
fi

echo "  Architecture:    $ARCHITECTURE"
echo "  Base encoder:    $BASE_MODEL (frozen=$BASE_FROZEN)"
echo "  Batch size:      $BATCH_SIZE (grad accum: $ACCUM_GRAD)"
echo "  Learning rate:   $LEARNING_RATE"
echo "  Epochs:          $EPOCHS"
echo "  Output:          $OUTPUT_DIR"
echo ""

$PYTHON scripts/train.py \
    --train-dataset "$PREP_DIR/splits/train.zarr" \
    --val-dataset "$PREP_DIR/splits/valid.zarr" \
    --output-dir "$OUTPUT_DIR" \
    --base-encoder-path "$BASE_MODEL" \
    --base-encoder-frozen "$BASE_FROZEN" \
    --architecture "$ARCHITECTURE" \
    --head-encoder-layers $HEAD_LAYERS \
    --token-embedding-dim $TOKEN_EMBED_DIM \
    --batch-size $BATCH_SIZE \
    --accumulate-grad-batches $ACCUM_GRAD \
    --epochs $EPOCHS \
    --learning-rate $LEARNING_RATE \
    --learning-rate-decay none \
    --num-workers $NUM_WORKERS \
    --prefetch-factor 2 \
    --gpu "${NUM_GPUS:-2}" \
    --strategy ddp \
    --checkpoint-frequency 200 \
    --val-check-interval 200 \
    --train-eval-frequency 200 \
    --limit-val-batches 1.0 \
    --log-frequency 1 \
    --enable-visualization yes \
    --torch-compile no \
    --project-name "$PROJECT_NAME" \
    --run-name "$RUN_NAME" \
    $CHECKPOINT_ARGS \
    2>&1 | tee -a "$OUTPUT_DIR/training.log"

echo ""
echo "============================================"
echo " Training complete!"
echo " Output: $OUTPUT_DIR"
echo " Checkpoints: $OUTPUT_DIR/checkpoints/"
echo "============================================"
