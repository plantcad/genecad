#!/bin/bash
set -e
cd "$(dirname "$0")/../.." || exit 1

# =================================================================
# Multi-Chromosome GeneCAD Prediction Pipeline
# =================================================================

usage() {
    cat << 'USAGE'
Usage: predict_all_chroms.sh [OPTIONS]

Run the complete GeneCAD annotation pipeline on a genomic FASTA file.
Models are downloaded automatically from Hugging Face on first run.

Options:
  -i, --input PATH      Input genome FASTA file
                        (default: downloads Arabidopsis thaliana TAIR12 example)
  -o, --output DIR      Output directory (default: genecad_result/Athaliana_predictions)
  -s, --species NAME    Species label prefixed on output filenames (default: Athaliana)
  -m, --mode MODE       Model to use: plant | animal  (default: plant)
  -h, --help            Show this help message

Examples:
  # Annotate a plant genome (default)
  bash predict_all_chroms.sh -i data/my_plant.fa -o output/ -s Zmays -m plant

  # Annotate an animal genome
  bash predict_all_chroms.sh -i data/my_animal.fa -o output/ -s Hsapiens -m animal
USAGE
    exit 0
}

INPUT_FILE="data/example/GCA_978657495.1_TAIR12_genomic_5.fna"
OUTPUT_DIR="genecad_result/Athaliana_predictions"
SPECIES_ID="Athaliana"
MODE="plant"

while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--input)   INPUT_FILE="$2"; shift 2 ;;
    -o|--output)  OUTPUT_DIR="$2"; shift 2 ;;
    -s|--species) SPECIES_ID="$2"; shift 2 ;;
    -m|--mode)    MODE="$2";       shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

echo "================================================================="
echo "GeneCAD Prediction Pipeline"
echo "================================================================="

# Download default Arabidopsis sequence if missing
if [[ "$INPUT_FILE" == "data/example/GCA_978657495.1_TAIR12_genomic_5.fna" && ! -f "$INPUT_FILE" ]]; then
    echo "Downloading default Arabidopsis thaliana sequence..."
    mkdir -p "$(dirname "$INPUT_FILE")"
    wget -qO "$INPUT_FILE" "https://huggingface.co/datasets/plantcad/genecad-dev/resolve/main/data/fasta/GCA_978657495.1_TAIR12_genomic_5.fna"
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input FASTA file '$INPUT_FILE' not found."
    exit 1
fi

# Model selection based on --mode
case "$MODE" in
  plant)
    BASE_MODEL="plantcad/pcad2-200M-cnet-baseline"
    HEAD_MODEL="Zong-Yan/genecad_plant"
    ;;
  animal)
    BASE_MODEL="emarro/pcad2_vert_small"
    HEAD_MODEL="Zong-Yan/genecad_vert"
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

    # Filter invalid genes (recall: no UTRs required)
    $PYTHON scripts/gff.py filter_to_valid_genes \
        --input "$PIPELINE_DIR/predictions__raw__feat_len_2__gene_len_30.gff" \
        --output "$PIPELINE_DIR/predictions_recall__raw.gff" \
        --require-utrs "no"

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

# Merge recall GFFs
python3 "$MERGE_SCRIPT" \
    --output "$OUTPUT_DIR/${SPECIES_ID}_GeneCAD_raw.gff" \
    --inputs "${RECALL_GFFS[@]}"

echo ""
echo "================================================================="
echo "5. Running protein refinement on merged predictions..."
echo "================================================================="

$PYTHON scripts/refine.py \
    --gff "$OUTPUT_DIR/${SPECIES_ID}_GeneCAD_raw.gff" \
    --genome "$INPUT_FILE" \
    --out "$OUTPUT_DIR/${SPECIES_ID}_GeneCAD_final.gff" \
    --filter-unmerged

echo ""
echo "================================================================="
echo "All done! Final predictions saved to:"
echo "Raw:   $OUTPUT_DIR/${SPECIES_ID}_GeneCAD_raw.gff"
echo "Final: $OUTPUT_DIR/${SPECIES_ID}_GeneCAD_final.gff"
echo "================================================================="
