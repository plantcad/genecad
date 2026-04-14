#!/bin/bash
set -e
cd "$(dirname "$0")" || exit 1

# =================================================================
# Multi-Chromosome GeneCAD Prediction Pipeline
# =================================================================

usage() {
    cat << 'USAGE'
Usage: predict.sh [OPTIONS]

Run the complete GeneCAD annotation pipeline on a genomic FASTA file.
Models are downloaded automatically from Hugging Face on first run.

Options:
  -i, --input PATH      Input genome FASTA file
                        (default: downloads Arabidopsis thaliana TAIR12 example)
  -o, --output DIR      Output directory (default: genecad_result/Athaliana_predictions)
  -s, --species NAME    Species label prefixed on output filenames (default: Athaliana)
  -m, --mode MODE       Model to use: plant | animal  (default: plant)
  -b, --batch-size N    Inference batch size (default: auto — scaled to GPU VRAM)
  -h, --help            Show this help message

Batch size is chosen automatically based on available GPU VRAM:
  >= 70 GB  ->  32   (A100 80G / H100)
  >= 35 GB  ->  16   (A100 40G)
  >= 20 GB  ->   8   (RTX 3090/4090, L40S)
  >= 14 GB  ->   4   (RTX 3080 Ti, V100 16G, T4)
  <  14 GB  ->   2   (older / smaller GPUs)

Examples:
  # Annotate a plant genome (default, batch size auto-detected)
  bash predict.sh -i data/my_plant.fa -o output/ -s Zmays -m plant

  # Override batch size manually
  bash predict.sh -i data/my_plant.fa -o output/ -s Zmays -b 8
USAGE
    exit 0
}

INPUT_FILE="data/example/GCA_978657495.1_TAIR12_genomic_5.fa.gz"
OUTPUT_DIR="genecad_result/Athaliana_predictions"
SPECIES_ID="Athaliana"
MODE="plant"
BATCH_SIZE_ARG="auto"

while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--input)      INPUT_FILE="$2";      shift 2 ;;
    -o|--output)     OUTPUT_DIR="$2";      shift 2 ;;
    -s|--species)    SPECIES_ID="$2";      shift 2 ;;
    -m|--mode)       MODE="$2";            shift 2 ;;
    -b|--batch-size) BATCH_SIZE_ARG="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

echo "================================================================="
echo "GeneCAD Prediction Pipeline"
echo "================================================================="

# Download default Arabidopsis sequence if missing
if [[ "$INPUT_FILE" == "data/example/GCA_978657495.1_TAIR12_genomic_5.fa.gz" && ! -f "$INPUT_FILE" ]]; then
    echo "Downloading default Arabidopsis thaliana sequence..."
    mkdir -p "$(dirname "$INPUT_FILE")"
    wget -qO "$INPUT_FILE" "https://huggingface.co/datasets/plantcad/genecad-dev/resolve/main/data/fasta/example/GCA_978657495.1_TAIR12_genomic_5.fa.gz"
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input FASTA file '$INPUT_FILE' not found."
    exit 1
fi

# Model selection based on --mode
case "$MODE" in
  plant)
    BASE_MODEL="emarro/pcad2-200M-cnet-baseline"
    HEAD_MODEL="zongyanliu/genecad_5-species"
    ;;
  animal)
    BASE_MODEL="emarro/pcad2_vert_small"
    HEAD_MODEL="zongyanliu/genecad_vert"
    ;;
  *)
    echo "Error: Unknown mode '$MODE'. Valid options are: plant, animal"
    exit 1
    ;;
esac

echo "Input FASTA: $INPUT_FILE"
echo "Output Dir:  $OUTPUT_DIR"
echo "Species ID:  $SPECIES_ID"
echo "Mode:        $MODE  ($BASE_MODEL + $HEAD_MODEL)"
echo "================================================================="

# Pipeline Config
TOKENIZER_PATH="$BASE_MODEL"
DTYPE="bfloat16"
PYTHON="uv run python"
export PYTHONPATH=.

# Auto-detect batch size from GPU VRAM if not set manually
if [[ "$BATCH_SIZE_ARG" == "auto" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        GPU_MEM_GB=$(( GPU_MEM_MB / 1024 ))
        if   [[ $GPU_MEM_GB -ge 70 ]]; then BATCH_SIZE=32
        elif [[ $GPU_MEM_GB -ge 35 ]]; then BATCH_SIZE=16
        elif [[ $GPU_MEM_GB -ge 20 ]]; then BATCH_SIZE=8
        elif [[ $GPU_MEM_GB -ge 14 ]]; then BATCH_SIZE=4
        else                                BATCH_SIZE=2
        fi
        echo "Auto batch size: ${BATCH_SIZE} (detected ${GPU_MEM_GB} GB GPU VRAM)"
    else
        BATCH_SIZE=8
        echo "Warning: nvidia-smi not found. Defaulting to batch size ${BATCH_SIZE}."
    fi
else
    BATCH_SIZE="$BATCH_SIZE_ARG"
    echo "Using manual batch size: ${BATCH_SIZE}"
fi

# Merge script location (same directory as this script)
MERGE_SCRIPT="scripts/merge_gff.py"

# Create top-level output directory
mkdir -p "$OUTPUT_DIR"

# 2. Discover chromosomes from FASTA headers
# -------------------------------------------
echo "================================================================="
echo "Discovering chromosomes from FASTA file..."
echo "================================================================="

# Extract chromosome IDs from FASTA headers (handles both .fa and .fa.gz)
if [[ "$INPUT_FILE" == *.gz ]]; then
    CHROM_IDS=$(zcat "$INPUT_FILE" | grep "^>" | sed 's/^>//' | awk '{print $1}')
else
    CHROM_IDS=$(grep "^>" "$INPUT_FILE" | sed 's/^>//' | awk '{print $1}')
fi
CHROM_COUNT=$(echo "$CHROM_IDS" | wc -l)

echo "Found $CHROM_COUNT chromosomes/sequences:"
echo "$CHROM_IDS"
echo ""

# 3. Loop over each chromosome
# -----------------------------
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

    # If the final per-chromosome GFF already exists, skip all steps for this chromosome
    if [[ -f "$CHR_OUTPUT_DIR/predictions_recall.gff" ]]; then
        echo "[${CHR_ID}] Already complete — skipping all steps (delete $CHR_OUTPUT_DIR to rerun)"
        RECALL_GFFS+=("$CHR_OUTPUT_DIR/predictions_recall.gff")
        continue
    fi

    # --- Step 1: Extract Sequences ---
    if [[ -e "$PIPELINE_DIR/sequences.zarr" ]]; then
        echo "[${CHR_ID}] [1/6] Skipping — sequences.zarr already exists"
    else
        echo "[${CHR_ID}] [1/6] Extracting sequences..."
        $PYTHON scripts/extract.py extract_fasta_file \
            --species-id "$SPECIES_ID" \
            --fasta-file "$INPUT_FILE" \
            --chrom-map "${CHR_ID}:${CHR_ID}" \
            --tokenizer-path "$TOKENIZER_PATH" \
            --output "$PIPELINE_DIR/sequences.zarr"
    fi

    # --- Step 2: Predict Tokens ---
    if [[ -e "$PIPELINE_DIR/predictions.zarr" ]]; then
        echo "[${CHR_ID}] [2/6] Skipping — predictions.zarr already exists"
    else
        echo "[${CHR_ID}] [2/6] Generating token predictions..."
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
    fi

    # --- Step 3: Detect Intervals ---
    if [[ -e "$PIPELINE_DIR/intervals.zarr" ]]; then
        echo "[${CHR_ID}] [3/6] Skipping — intervals.zarr already exists"
    else
        echo "[${CHR_ID}] [3/6] Detecting intervals (Viterbi decoding)..."
        $PYTHON scripts/predict.py detect_intervals \
            --input-dir "$PIPELINE_DIR/predictions.zarr" \
            --output "$PIPELINE_DIR/intervals.zarr" \
            --decoding-methods "direct,viterbi" \
            --remove-incomplete-features yes
    fi

    # --- Step 4: Export Raw GFF ---
    if [[ -f "$PIPELINE_DIR/predictions__raw.gff" ]]; then
        echo "[${CHR_ID}] [4/6] Skipping — predictions__raw.gff already exists"
    else
        echo "[${CHR_ID}] [4/6] Exporting raw GFF..."
        $PYTHON scripts/predict.py export_gff \
            --input "$PIPELINE_DIR/intervals.zarr" \
            --output "$PIPELINE_DIR/predictions__raw.gff" \
            --decoding-method direct \
            --min-transcript-length 3 \
            --strip-introns yes
    fi

    # --- Step 5: Post-processing Filters ---
    echo "[${CHR_ID}] [5/6] Filtering features..."

    if [[ -f "$PIPELINE_DIR/predictions__raw__feat_len_2.gff" ]]; then
        echo "[${CHR_ID}]   Skipping feature-length filter — output already exists"
    else
        $PYTHON scripts/gff.py filter_to_min_feature_length \
            --input "$PIPELINE_DIR/predictions__raw.gff" \
            --output "$PIPELINE_DIR/predictions__raw__feat_len_2.gff" \
            --feature-types "five_prime_UTR,three_prime_UTR,CDS" \
            --min-length 2
    fi

    if [[ -f "$PIPELINE_DIR/predictions__raw__feat_len_2__gene_len_30.gff" ]]; then
        echo "[${CHR_ID}]   Skipping gene-length filter — output already exists"
    else
        $PYTHON scripts/gff.py filter_to_min_gene_length \
            --input "$PIPELINE_DIR/predictions__raw__feat_len_2.gff" \
            --output "$PIPELINE_DIR/predictions__raw__feat_len_2__gene_len_30.gff" \
            --min-length 30
    fi

    if [[ -f "$PIPELINE_DIR/predictions_recall__raw.gff" ]]; then
        echo "[${CHR_ID}]   Skipping valid-gene filter — output already exists"
    else
        $PYTHON scripts/gff.py filter_to_valid_genes \
            --input "$PIPELINE_DIR/predictions__raw__feat_len_2__gene_len_30.gff" \
            --output "$PIPELINE_DIR/predictions_recall__raw.gff" \
            --require-utrs "no"
    fi

    # --- Step 6: Copy final per-chromosome outputs ---
    echo "[${CHR_ID}] [6/6] Saving per-chromosome results..."
    cp "$PIPELINE_DIR/predictions_recall__raw.gff" "$CHR_OUTPUT_DIR/predictions_recall.gff"

    RECALL_GFFS+=("$CHR_OUTPUT_DIR/predictions_recall.gff")

    echo "[${CHR_ID}] Done!"
done

# 4. Merge all per-chromosome GFFs into single files
# ---------------------------------------------------
echo ""
echo "================================================================="
echo "Merging per-chromosome GFFs into single files..."
echo "================================================================="

RAW_GFF="$OUTPUT_DIR/${SPECIES_ID}_GeneCAD_raw.gff"
FINAL_GFF="$OUTPUT_DIR/${SPECIES_ID}_GeneCAD_final.gff"

if [[ -f "$RAW_GFF" ]]; then
    echo "Skipping merge — ${SPECIES_ID}_GeneCAD_raw.gff already exists"
else
    python3 "$MERGE_SCRIPT" \
        --output "$RAW_GFF" \
        --inputs "${RECALL_GFFS[@]}"
fi

echo ""
echo "================================================================="
echo "5. Running protein refinement on merged predictions..."
echo "================================================================="

if [[ -f "$FINAL_GFF" ]]; then
    echo "Skipping refinement — ${SPECIES_ID}_GeneCAD_final.gff already exists"
else
    $PYTHON scripts/refine.py \
        --gff "$RAW_GFF" \
        --genome "$INPUT_FILE" \
        --out "$FINAL_GFF" \
        --filter-unmerged
fi

echo ""
echo "================================================================="
echo "All done! Final predictions saved to:"
echo "Raw:   $RAW_GFF"
echo "Final: $FINAL_GFF"
echo "================================================================="


