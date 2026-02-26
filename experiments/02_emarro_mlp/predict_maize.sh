#!/bin/bash
set -e
cd /workdir/zl843/GeneCAD/genecad || exit 1

# =================================================================
# Usage: ./predict_maize.sh
# =================================================================

# 1. Configuration
# ----------------
INPUT_FILE="/workdir/zl843/GeneCAD/fine-tuning/input_file/Zmays/Zmays_833_Zm-B73-REFERENCE-NAM-5.0_chr1.fa"
OUTPUT_DIR="/workdir/zl843/GeneCAD/genecad_result/prediction_emarro_mlp_maize_chr1"
SPECIES_ID="Zmays"
CHR_ID="chr1"       # Target Chromosome ID in output
INPUT_CHR="chr1"    # Source Chromosome ID in FASTA header

# Model Paths
# Using the new baseline model as per recent integration
BASE_MODEL="/workdir/zl843/GeneCAD/pcad2-200M-cnet-mlp"
HEAD_MODEL="/workdir/zl843/GeneCAD/genecad_result/training_emarro_mlp/training_output/checkpoints/last.ckpt"

# Pipeline Config
TOKENIZER_PATH="$BASE_MODEL" # Use the same model for tokenization
BATCH_SIZE=32
DTYPE="bfloat16"
REQUIRE_UTRS="no"  # Set to "yes" for precision-leaning, "no" for recall-leaning

# Python executable (use uv run)
PYTHON="uv run python"
export PYTHONPATH=.

# Create output directories
PIPELINE_DIR="${OUTPUT_DIR}/pipeline"
mkdir -p "$PIPELINE_DIR"

echo "================================================================="
echo "Running GeneCAD Prediction Pipeline (Maize)"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_DIR"
echo "Base Model: $BASE_MODEL"
echo "================================================================="

# 2. Extract Sequences
# --------------------
echo "[1/6] Extracting sequences..."
$PYTHON scripts/extract.py extract_fasta_file \
    --species-id "$SPECIES_ID" \
    --fasta-file "$INPUT_FILE" \
    --chrom-map "$INPUT_CHR:$CHR_ID" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --output "$PIPELINE_DIR/sequences.zarr"

# 3. Predict Tokens
# -----------------
echo "[2/6] Generating token predictions..."
rm -rf "$PIPELINE_DIR/predictions.zarr"
$PYTHON scripts/predict.py create_predictions \
    --input "$PIPELINE_DIR/sequences.zarr" \
    --output-dir "$PIPELINE_DIR/predictions.zarr" \
    --model-path "$BASE_MODEL" \
    --model-checkpoint "$HEAD_MODEL" \
    --species-id "$SPECIES_ID" \
    --chromosome-id "$CHR_ID" \
    --batch-size "$BATCH_SIZE" \
    --dtype "$DTYPE"

# 4. Detect Intervals
# -------------------
echo "[3/6] Detecting intervals (Viterbi decoding)..."
$PYTHON scripts/predict.py detect_intervals \
    --input-dir "$PIPELINE_DIR/predictions.zarr" \
    --output "$PIPELINE_DIR/intervals.zarr" \
    --decoding-methods "direct,viterbi" \
    --remove-incomplete-features yes

# 5. Export Raw GFF
# -----------------
echo "[4/6] Exporting raw GFF..."
$PYTHON scripts/predict.py export_gff \
    --input "$PIPELINE_DIR/intervals.zarr" \
    --output "$PIPELINE_DIR/predictions__raw.gff" \
    --decoding-method direct \
    --min-transcript-length 3 \
    --strip-introns yes

# 6. Post-processing Filters
# --------------------------
echo "[5/6] Filtering features..."

# Filter small features
$PYTHON scripts/gff.py filter_to_min_feature_length \
    --input "$PIPELINE_DIR/predictions__raw.gff" \
    --output "$PIPELINE_DIR/predictions__raw__feat_len_2.gff" \
    --feature-types "five_prime_UTR,three_prime_UTR,CDS" \
    --min-length 2

# Filter short genes
$PYTHON scripts/gff.py filter_to_min_gene_length \
    --input "$PIPELINE_DIR/predictions__raw__feat_len_2.gff" \
    --output "$PIPELINE_DIR/predictions__raw__feat_len_2__gene_len_30.gff" \
    --min-length 30

# Filter invalid genes (require UTRs? Makefile says yes)
$PYTHON scripts/gff.py filter_to_valid_genes \
    --input "$PIPELINE_DIR/predictions__raw__feat_len_2__gene_len_30.gff" \
    --output "$PIPELINE_DIR/predictions__raw__feat_len_2__gene_len_30__has_req_feats.gff" \
    --require-utrs "$REQUIRE_UTRS"

# 7. Final Output
# ---------------
cp "$PIPELINE_DIR/predictions__raw__feat_len_2__gene_len_30__has_req_feats.gff" "$OUTPUT_DIR/predictions.gff"

echo "================================================================="
echo "Done! Final predictions saved to:"
echo "$OUTPUT_DIR/predictions.gff"
echo "================================================================="
