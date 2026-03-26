#!/bin/bash
set -e

# ==========================================
# GeneCAD Inference Wrapper
# ==========================================

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Required Arguments:"
    echo "  -i, --input <path>     Path to the input genome FASTA file"
    echo "  -o, --output <dir>     Directory to save prediction results"
    echo ""
    echo "Optional Arguments:"
    echo "  -m, --model <path>     Trained GeneCAD head model checkpoint (default: Zong-Yan/genecad_5-species)"
    echo "  -b, --base <name>      Base PlantCAD model (default: emarro/pcad2-200M-cnet-baseline)"
    echo "  -s, --species <id>     Identifier for the species (default: unknown_species)"
    echo "  -c, --chroms <list>    Comma-separated list of specific chromosomes to predict (default: auto-detect all)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --input genome.fna --output results/ --model checkpoints/last.ckpt --species arabidopsis"
    exit 1
}

# Defaults
INPUT_FILE=""
OUTPUT_DIR=""
HEAD_MODEL="Zong-Yan/genecad_5-species"
BASE_MODEL="emarro/pcad2-200M-cnet-baseline"
SPECIES_ID="unknown_species"
SPECIFIC_CHROMS=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--input) INPUT_FILE="$2"; shift ;;
        -o|--output) OUTPUT_DIR="$2"; shift ;;
        -m|--model) HEAD_MODEL="$2"; shift ;;
        -b|--base) BASE_MODEL="$2"; shift ;;
        -s|--species) SPECIES_ID="$2"; shift ;;
        -c|--chroms) SPECIFIC_CHROMS="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Missing required arguments."
    usage
fi

# Resolve absolute paths so we can safely change directories
INPUT_FILE=$(realpath "$INPUT_FILE")
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

# Only resolve realpath for local model files, not huggingface paths
if [[ "$HEAD_MODEL" == *"/"* ]] && [ -f "$HEAD_MODEL" ]; then
    HEAD_MODEL=$(realpath "$HEAD_MODEL")
fi


# Get absolute path to the genecad repository root (where the Makefile lives)
SCRIPT_DIR=$(dirname "$(realpath "$0")")
GENECAD_DIR=$(dirname "$SCRIPT_DIR")
cd "$GENECAD_DIR" || exit 1

# Extract chromosome names if not explicitly provided
if [ -z "$SPECIFIC_CHROMS" ]; then
    echo "Analyzing FASTA file to detect chromosomes..."
    CHROM_IDS=$(grep "^>" "$INPUT_FILE" | sed 's/^>//' | awk '{print $1}')
else
    # Replace commas with spaces
    CHROM_IDS=$(echo "$SPECIFIC_CHROMS" | tr ',' ' ')
fi

TOTAL=$(echo "$CHROM_IDS" | wc -w)
if [ "$TOTAL" -gt 50 ]; then
    echo "WARNING: Detected $TOTAL sequences in the FASTA file."
    echo "GeneCAD is designed to predict on chromosome-level assemblies. Processing thousands of small unplaced contigs will be very slow."
fi

# Detect GPUs
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi
echo "Detected $NUM_GPUS GPU(s) available for inference."

# Launchers
if [ -f ".venv/bin/torchrun" ]; then
    GPU_LAUNCHER=".venv/bin/torchrun --nproc_per_node=$NUM_GPUS"
else
    GPU_LAUNCHER="torchrun --nproc_per_node=$NUM_GPUS"
fi

# uv is highly recommended, fallback to standard python if missing
if command -v uv >/dev/null 2>&1; then
    CPU_LAUNCHER="uv run --extra torch python"
else
    CPU_LAUNCHER="python"
fi

mkdir -p "$OUTPUT_DIR"
MERGED_GFF="${OUTPUT_DIR}/merged_predictions.gff"
> "$MERGED_GFF" # Clear/create the merged file

for CHR_ID in $CHROM_IDS; do
    echo "========================================"
    echo "Running GeneCAD Inference: Chr $CHR_ID"
    echo "========================================"
    
    CHR_OUT="${OUTPUT_DIR}/${CHR_ID}"
    
    # Run GeneCAD Makefile pipeline
    INPUT_FILE="$INPUT_FILE" \
    OUTPUT_DIR="$CHR_OUT" \
    SPECIES_ID="$SPECIES_ID" \
    CHR_ID="$CHR_ID" \
    BASE_MODEL_PATH="$BASE_MODEL" \
    HEAD_MODEL_PATH="$HEAD_MODEL" \
    PRED_BATCH_SIZE=32 \
    REQUIRE_UTRS="no" \
    LAUNCHER="$CPU_LAUNCHER" \
    GPU_LAUNCHER="$GPU_LAUNCHER" \
    make -f pipelines/prediction all
    
    # Check if the GFF was successfully generated
    if [ -f "${CHR_OUT}/predictions.gff" ]; then
        # Append to merged file, injecting the chromosome ID to ensure unique feature IDs
        awk -v chr="$CHR_ID" '
            /^#/ { next }
            {
                gsub(/ID=/, "ID=" chr "_");
                gsub(/Parent=/, "Parent=" chr "_");
                print $0
            }
        ' "${CHR_OUT}/predictions.gff" >> "$MERGED_GFF"
    else
        echo "WARNING: No predictions.gff found for chromosome $CHR_ID. Skipping merge."
    fi
done

echo "Adding standard GFF header to the final merged file..."
sed -i '1s/^/##gff-version 3\n/' "$MERGED_GFF"

echo "========================================"
echo "SUCCESS! GeneCAD inference complete."
echo "Merged GFF annotations saved to: $MERGED_GFF"
echo "========================================"
