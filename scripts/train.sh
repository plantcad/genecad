#!/bin/bash
set -euo pipefail

# =============================================================================
# GeneCAD Multispecies Fine-Tuning Wrapper
# =============================================================================

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Required Arguments:"
    echo "  -d, --data-dir <dir>      Directory containing 'gff/' and 'fasta/' prefixed datasets"
    echo "  -s, --species <list>      Quoted list of species IDs supported in src/config.py (e.g., \"Athaliana Osativa\")"
    echo "  -o, --output <dir>        Directory to save training logs, data, and model checkpoints"
    echo ""
    echo "Optional Arguments:"
    echo "  -m, --model <path>        Checkpoint to continue training from (default: Zong-Yan/genecad_5-species)"
    echo "                            (Set to 'none' to train from scratch on the base encoder)"
    echo "  -b, --base <name>         Base PlantCAD model (default: emarro/pcad2-200M-cnet-baseline)"
    echo "  -e, --epochs <num>        Number of epochs to train (default: 1)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -d data/ -s \"Athaliana Osativa\" -o results/ -m Zong-Yan/genecad_5-species"
    exit 1
}

# Defaults
DATA_DIR=""
SPECIES_IDS=""
OUTPUT_DIR=""
RESUME_CHECKPOINT="Zong-Yan/genecad_5-species"
BASE_MODEL="emarro/pcad2-200M-cnet-baseline"
EPOCHS=1

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--data-dir) DATA_DIR="$2"; shift ;;
        -s|--species) SPECIES_IDS="$2"; shift ;;
        -o|--output) OUTPUT_DIR="$2"; shift ;;
        -m|--model) RESUME_CHECKPOINT="$2"; shift ;;
        -b|--base) BASE_MODEL="$2"; shift ;;
        -e|--epochs) EPOCHS="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [ -z "$DATA_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$SPECIES_IDS" ]; then
    echo "Error: Missing required arguments."
    usage
fi

# Resolve absolute paths safely
DATA_DIR=$(realpath "$DATA_DIR")
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

if [[ "$RESUME_CHECKPOINT" == *"/"* ]] && [ -f "$RESUME_CHECKPOINT" ]; then
    RESUME_CHECKPOINT=$(realpath "$RESUME_CHECKPOINT")
fi

# Get absolute path to the genecad repository root
SCRIPT_DIR=$(dirname "$(realpath "$0")")
GENECAD_DIR=$(dirname "$SCRIPT_DIR")
cd "$GENECAD_DIR" || exit 1
export PYTHONPATH="$GENECAD_DIR"

# Ensure output directory structure exists
EXTRACT_DIR="$OUTPUT_DIR/extract"
TRANSFORM_DIR="$OUTPUT_DIR/transform"
PREP_DIR="$OUTPUT_DIR/prep"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
mkdir -p "$EXTRACT_DIR" "$TRANSFORM_DIR" "$PREP_DIR/splits" "$CHECKPOINT_DIR"

# Launcher detection
if command -v uv >/dev/null 2>&1; then
    PYTHON="uv run --extra torch python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python"
fi

# Hyperparameters (locked for optimal multispecies focal loss convergence)
BATCH_SIZE=2
ACCUM_GRAD=16
LEARNING_RATE=1e-4
ARCHITECTURE="all"
HEAD_LAYERS=8
TOKEN_EMBED_DIM=128
BASE_FROZEN="no"
NUM_WORKERS=6
VALID_PROPORTION=0.05
PROJECT_NAME="genecad-emarro"
RUN_NAME="emarro-hnet-multispecies-focal-sqrt"

echo "============================================"
echo " GeneCAD Fine-Tuning Pipeline"
echo " Base model: $BASE_MODEL"
echo " Resume:     $RESUME_CHECKPOINT"
echo " Species:    $SPECIES_IDS"
echo " Output:     $OUTPUT_DIR"
echo "============================================"

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
        --intergenic-proportion 0.1 \
        --num-workers $NUM_WORKERS
fi

if [ ! -d "$PREP_DIR/splits/train.zarr" ]; then
    $PYTHON scripts/sample.py generate_training_splits \
        --input "$TRANSFORM_DIR/windows.zarr" \
        --train-output "$PREP_DIR/splits/train.zarr" \
        --valid-output "$PREP_DIR/splits/valid.zarr" \
        --valid-proportion $VALID_PROPORTION
else
    echo "  Skipped (already exists)"
fi

# =============================================================================
# Step 8: Run training
# =============================================================================
echo ""
echo "[8/8] Starting training..."

CHECKPOINT_ARGS=""
if [ -n "$RESUME_CHECKPOINT" ] && [ "$RESUME_CHECKPOINT" != "none" ]; then
    echo "  Restoring weights from: $RESUME_CHECKPOINT"
    CHECKPOINT_ARGS="--checkpoint $RESUME_CHECKPOINT --checkpoint-type model"
fi

# Detect GPUs
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)
if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=1
    STRATEGY="auto"
else
    STRATEGY="ddp"
fi

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
    --learning-rate-decay cosine \
    --num-workers $NUM_WORKERS \
    --prefetch-factor 2 \
    --gpu $NUM_GPUS \
    --strategy $STRATEGY \
    --checkpoint-frequency 1000 \
    --val-check-interval 1000 \
    --train-eval-frequency 1000 \
    --limit-val-batches 1.0 \
    --log-frequency 1 \
    --enable-visualization yes \
    --randomize-base no \
    --loss-weighting inverse-frequency \
    --compute-class-frequencies yes \
    --project-name "$PROJECT_NAME" \
    --run-name "$RUN_NAME" \
    $CHECKPOINT_ARGS \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "============================================"
echo " Training complete!"
echo " Checkpoints available in: $CHECKPOINT_DIR"
echo "============================================"
