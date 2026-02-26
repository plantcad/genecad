#!/bin/bash
set -e
cd /workdir/zl843/GeneCAD/genecad || exit 1

# =================================================================
# Multi-Chromosome GeneCAD Prediction Pipeline
# Usage: ./predict_all_chroms.sh
#
# Runs the full GeneCAD prediction pipeline on ALL chromosomes in a
# multi-chromosome FASTA file, then merges the per-chromosome GFF
# outputs into a single GFF with unique, chromosome-prefixed IDs.
# =================================================================

# 1. Configuration
# ----------------
INPUT_FILE="/workdir/zl843/GeneCAD/fine-tuning/input_file/Zmays/Zmays_833_Zm-B73-REFERENCE-NAM-5.0.fa"
OUTPUT_DIR="/workdir/zl843/GeneCAD/genecad_result/prediction_emarro_hnet_zmays_all"
SPECIES_ID="Zmays"

# Model Paths
BASE_MODEL="/workdir/zl843/GeneCAD/pcad2-200M-cnet-baseline"
HEAD_MODEL="/workdir/zl843/GeneCAD/genecad_result/training_emarro/training_output/checkpoints/last-v1.ckpt"

# Pipeline Config
TOKENIZER_PATH="$BASE_MODEL"
BATCH_SIZE=32
DTYPE="bfloat16"
PYTHON="uv run python"
export PYTHONPATH=.

# Merge script location (same directory as this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERGE_SCRIPT="${SCRIPT_DIR}/merge_gff.py"

# Create top-level output directory
mkdir -p "$OUTPUT_DIR"

# 2. Discover chromosomes from FASTA headers
# -------------------------------------------
echo "================================================================="
echo "Discovering chromosomes from FASTA file..."
echo "================================================================="

# Extract chromosome IDs from FASTA headers (first field after ">")
CHROM_IDS=$(grep "^>" "$INPUT_FILE" | sed 's/^>//' | awk '{print $1}')
CHROM_COUNT=$(echo "$CHROM_IDS" | wc -l)

echo "Found $CHROM_COUNT chromosomes/sequences:"
echo "$CHROM_IDS"
echo ""

# 3. Loop over each chromosome
# -----------------------------
PRECISION_GFFS=()
RECALL_GFFS=()

for CHR_ID in $CHROM_IDS; do
    echo ""
    echo "================================================================="
    echo "Processing chromosome: $CHR_ID"
    echo "================================================================="

    # Per-chromosome output directory
    CHR_OUTPUT_DIR="${OUTPUT_DIR}/${CHR_ID}"
    PIPELINE_DIR="${CHR_OUTPUT_DIR}/pipeline"
    mkdir -p "$PIPELINE_DIR"

    # --- Step 1: Extract Sequences ---
    echo "[${CHR_ID}] [1/6] Extracting sequences..."
    $PYTHON scripts/extract.py extract_fasta_file \
        --species-id "$SPECIES_ID" \
        --fasta-file "$INPUT_FILE" \
        --chrom-map "${CHR_ID}:${CHR_ID}" \
        --tokenizer-path "$TOKENIZER_PATH" \
        --output "$PIPELINE_DIR/sequences.zarr"

    # --- Step 2: Predict Tokens ---
    echo "[${CHR_ID}] [2/6] Generating token predictions..."
    rm -rf "$PIPELINE_DIR/predictions.zarr"
    $PYTHON scripts/predict.py create_predictions \
        --input "$PIPELINE_DIR/sequences.zarr" \
        --output-dir "$PIPELINE_DIR/predictions.zarr" \
        --model-path "$BASE_MODEL" \
        --model-checkpoint "$HEAD_MODEL" \
        --species-id "$SPECIES_ID" \
        --chromosome-id "$CHR_ID" \
        --batch-size "$BATCH_SIZE" \
        --dtype "$DTYPE" \
        --window-size 8192 \
        --stride 4096

    # --- Step 3: Detect Intervals ---
    echo "[${CHR_ID}] [3/6] Detecting intervals (Viterbi decoding)..."
    $PYTHON scripts/predict.py detect_intervals \
        --input-dir "$PIPELINE_DIR/predictions.zarr" \
        --output "$PIPELINE_DIR/intervals.zarr" \
        --decoding-methods "direct,viterbi" \
        --remove-incomplete-features yes

    # --- Step 4: Export Raw GFF ---
    echo "[${CHR_ID}] [4/6] Exporting raw GFF..."
    $PYTHON scripts/predict.py export_gff \
        --input "$PIPELINE_DIR/intervals.zarr" \
        --output "$PIPELINE_DIR/predictions__raw.gff" \
        --decoding-method direct \
        --min-transcript-length 3 \
        --strip-introns yes

    # --- Step 5: Post-processing Filters ---
    echo "[${CHR_ID}] [5/6] Filtering features..."

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

    # Filter invalid genes (precision: require UTRs)
    $PYTHON scripts/gff.py filter_to_valid_genes \
        --input "$PIPELINE_DIR/predictions__raw__feat_len_2__gene_len_30.gff" \
        --output "$PIPELINE_DIR/predictions_precision__raw.gff" \
        --require-utrs "yes"

    # Filter invalid genes (recall: no UTRs required)
    $PYTHON scripts/gff.py filter_to_valid_genes \
        --input "$PIPELINE_DIR/predictions__raw__feat_len_2__gene_len_30.gff" \
        --output "$PIPELINE_DIR/predictions_recall__raw.gff" \
        --require-utrs "no"

    # --- Step 6: Copy final per-chromosome outputs ---
    echo "[${CHR_ID}] [6/6] Saving per-chromosome results..."
    cp "$PIPELINE_DIR/predictions_precision__raw.gff" "$CHR_OUTPUT_DIR/predictions_precision.gff"
    cp "$PIPELINE_DIR/predictions_recall__raw.gff" "$CHR_OUTPUT_DIR/predictions_recall.gff"

    PRECISION_GFFS+=("$CHR_OUTPUT_DIR/predictions_precision.gff")
    RECALL_GFFS+=("$CHR_OUTPUT_DIR/predictions_recall.gff")

    echo "[${CHR_ID}] Done!"
done

# 4. Merge all per-chromosome GFFs into single files
# ---------------------------------------------------
echo ""
echo "================================================================="
echo "Merging per-chromosome GFFs into single files..."
echo "================================================================="

# Merge precision GFFs
python3 "$MERGE_SCRIPT" \
    --output "$OUTPUT_DIR/predictions_precision_merged.gff" \
    --inputs "${PRECISION_GFFS[@]}"

# Merge recall GFFs
python3 "$MERGE_SCRIPT" \
    --output "$OUTPUT_DIR/predictions_recall_merged.gff" \
    --inputs "${RECALL_GFFS[@]}"

echo ""
echo "================================================================="
echo "All done! Final merged predictions saved to:"
echo "Precision: $OUTPUT_DIR/predictions_precision_merged.gff"
echo "Recall:    $OUTPUT_DIR/predictions_recall_merged.gff"
echo "================================================================="
