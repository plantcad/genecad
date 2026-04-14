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
  -b, --batch-size N    Inference batch size per GPU (default: auto — scaled to GPU VRAM)
  -g, --gpus LIST       Comma-separated GPU IDs to use, or 'all' for all available GPUs.
                        Chromosomes are distributed across GPUs in parallel.
                        (default: 0 — single GPU, sequential)
  -h, --help            Show this help message

Batch size is chosen automatically based on available GPU VRAM (per GPU):
  >= 70 GB  ->  80   (A100 80G / H100)
  >= 35 GB  ->  40   (A100 40G)
  >= 20 GB  ->  24   (RTX 3090/4090, L40S)
  >= 14 GB  ->  16   (RTX 3080 Ti, V100 16G, T4)
  <  14 GB  ->   8   (older / smaller GPUs)

Note: This auto-scaling pushes GPU utility to its limit. If you encounter CUDA Out-Of-Memory
      (OOM) errors, manually lower the batch size with the '-b' flag (e.g., '-b 32').

Multi-GPU usage:
  Chromosomes are distributed round-robin across the specified GPUs and processed
  in parallel. Each GPU runs its own chromosome job independently.
  Example: --gpus 0,1,2,3   (use GPUs 0–3, one chromosome per GPU at a time)
           --gpus all        (auto-detect and use all available GPUs)

Examples:
  # Annotate a plant genome (default, batch size auto-detected, single GPU)
  bash predict.sh -i data/my_plant.fa -o output/ -s Zmays -m plant

  # Use all available GPUs
  bash predict.sh -i data/my_plant.fa -o output/ -s Zmays --gpus all

  # Use specific GPUs with custom batch size
  bash predict.sh -i data/my_plant.fa -o output/ -s Zmays --gpus 0,1 -b 32
USAGE
    exit 0
}

INPUT_FILE="data/example/GCA_978657495.1_TAIR12_genomic_5.fa.gz"
OUTPUT_DIR="genecad_result/Athaliana_predictions"
SPECIES_ID="Athaliana"
MODE="plant"
BATCH_SIZE_ARG="auto"
GPUS_ARG="0"

while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--input)      INPUT_FILE="$2";      shift 2 ;;
    -o|--output)     OUTPUT_DIR="$2";      shift 2 ;;
    -s|--species)    SPECIES_ID="$2";      shift 2 ;;
    -m|--mode)       MODE="$2";            shift 2 ;;
    -b|--batch-size) BATCH_SIZE_ARG="$2"; shift 2 ;;
    -g|--gpus)       GPUS_ARG="$2";       shift 2 ;;
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

# =================================================================
# GPU Resolution
# =================================================================

# Resolve GPU list: 'all' -> all detected GPU IDs, else split CSV
if [[ "$GPUS_ARG" == "all" ]]; then
    GPU_IDS_ALL=""
    if command -v nvidia-smi &>/dev/null; then
        # Only use output if nvidia-smi exits cleanly (exit code 0)
        if nvidia-smi &>/dev/null; then
            GPU_IDS_ALL=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr '\n' ',' | sed 's/,$//')
        fi
    fi
    if [[ -z "$GPU_IDS_ALL" ]]; then
        echo "Warning: '--gpus all' requested but could not detect GPUs via nvidia-smi. Falling back to GPU 0."
        GPU_LIST_STR="0"
    else
        GPU_LIST_STR="$GPU_IDS_ALL"
    fi
else
    GPU_LIST_STR="$GPUS_ARG"
fi

# Parse GPU list into a bash array
IFS=',' read -ra GPU_ARRAY <<< "$GPU_LIST_STR"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "Using GPU(s): ${GPU_ARRAY[*]}  (${NUM_GPUS} total)"
echo "Input FASTA: $INPUT_FILE"
echo "Output Dir:  $OUTPUT_DIR"
echo "Species ID:  $SPECIES_ID"
echo "Mode:        $MODE  ($BASE_MODEL + $HEAD_MODEL)"
echo "================================================================="

# =================================================================
# Per-GPU batch size detection
# =================================================================
# Returns the batch size appropriate for a given GPU index.
resolve_batch_size_for_gpu() {
    local gpu_id="$1"
    if [[ "$BATCH_SIZE_ARG" != "auto" ]]; then
        echo "$BATCH_SIZE_ARG"
        return
    fi
    # Check that nvidia-smi is present AND working before querying
    if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null; then
        echo "8"  # safe fallback when GPU info is unavailable
        return
    fi
    local GPU_MEM_MB
    GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
        --id="$gpu_id" 2>/dev/null | head -1 | tr -d ' ')
    if [[ -z "$GPU_MEM_MB" ]] || ! [[ "$GPU_MEM_MB" =~ ^[0-9]+$ ]]; then
        echo "8"  # guard against non-numeric output
        return
    fi
    local GPU_MEM_GB=$(( GPU_MEM_MB / 1024 ))
    if   [[ $GPU_MEM_GB -ge 70 ]]; then echo 80
    elif [[ $GPU_MEM_GB -ge 35 ]]; then echo 40
    elif [[ $GPU_MEM_GB -ge 20 ]]; then echo 24
    elif [[ $GPU_MEM_GB -ge 14 ]]; then echo 16
    else                                 echo 8
    fi
}

# Pre-compute batch sizes per GPU once
declare -A GPU_BATCH_SIZES
for gpu_id in "${GPU_ARRAY[@]}"; do
    bs=$(resolve_batch_size_for_gpu "$gpu_id")
    GPU_BATCH_SIZES[$gpu_id]=$bs
    if [[ "$BATCH_SIZE_ARG" == "auto" ]]; then
        # Attempt to show detected VRAM for informational output
        if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
            GPU_MEM_MB_FOR_LOG=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits --id="$gpu_id" 2>/dev/null | head -1 | tr -d ' ')
            GPU_MEM_GB_FOR_LOG=$(( GPU_MEM_MB_FOR_LOG / 1024 ))
            echo "  GPU ${gpu_id}: batch size ${bs} (detected ${GPU_MEM_GB_FOR_LOG} GB VRAM)"
        else
            echo "  GPU ${gpu_id}: batch size ${bs} (VRAM auto-detect unavailable, using safe default)"
        fi
    else
        echo "  GPU ${gpu_id}: manual batch size ${bs}"
    fi
done
if [[ "$BATCH_SIZE_ARG" == "auto" ]]; then
    echo "  > TIP: If you run into CUDA Out-Of-Memory (OOM) errors, lower this using '-b <size>'"
fi

# Merge script location (same directory as this script)
MERGE_SCRIPT="scripts/merge_gff.py"

# Pipeline config
TOKENIZER_PATH="$BASE_MODEL"
DTYPE="bfloat16"
# Use the pre-built venv python if available (Docker container sets VIRTUAL_ENV=/build/.venv),
# or activate the local .venv from uv sync, otherwise fall back to uv run (local dev mode).
if [[ -n "$VIRTUAL_ENV" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
    VIRTUAL_ENV="$(pwd)/.venv"
    PYTHON=".venv/bin/python"
else
    PYTHON="uv run python"
fi
export PYTHONPATH=.

# Create top-level output directory
mkdir -p "$OUTPUT_DIR"

# =================================================================
# Step 1: Discover chromosomes from FASTA headers
# =================================================================
echo "================================================================="
echo "Discovering chromosomes from FASTA file..."
echo "================================================================="

if [[ "$INPUT_FILE" == *.gz ]]; then
    CHROM_IDS=$(zcat "$INPUT_FILE" | grep "^>" | sed 's/^>//' | awk '{print $1}')
else
    CHROM_IDS=$(grep "^>" "$INPUT_FILE" | sed 's/^>//' | awk '{print $1}')
fi
CHROM_COUNT=$(echo "$CHROM_IDS" | wc -l)

echo "Found $CHROM_COUNT chromosomes/sequences:"
echo "$CHROM_IDS"
echo ""

# =================================================================
# Step 2: Per-chromosome processing (parallel if multi-GPU)
# =================================================================

# Runs the full 6-step pipeline for one chromosome on a specific GPU.
# Called as a background job when multi-GPU mode is active.
process_chromosome() {
    local CHR_ID="$1"
    local GPU_ID="$2"
    local BATCH_SIZE="$3"
    local LOG_PREFIX="[${CHR_ID}|GPU${GPU_ID}]"

    local CHR_OUTPUT_DIR="${OUTPUT_DIR}/${CHR_ID}"
    local PIPELINE_DIR="${CHR_OUTPUT_DIR}/pipeline"
    mkdir -p "$PIPELINE_DIR"

    # Skip if already complete
    if [[ -f "$CHR_OUTPUT_DIR/predictions_recall.gff" ]]; then
        echo "${LOG_PREFIX} Already complete — skipping all steps (delete $CHR_OUTPUT_DIR to rerun)"
        return 0
    fi

    # Each job gets its own GPU via CUDA_VISIBLE_DEVICES.
    # Inside the process, the GPU appears as device index 0.
    export CUDA_VISIBLE_DEVICES="$GPU_ID"

    # --- Step 1: Extract Sequences ---
    if [[ -e "$PIPELINE_DIR/sequences.zarr" ]]; then
        echo "${LOG_PREFIX} [1/6] Skipping — sequences.zarr already exists"
    else
        echo "${LOG_PREFIX} [1/6] Extracting sequences..."
        $PYTHON scripts/extract.py extract_fasta_file \
            --species-id "$SPECIES_ID" \
            --fasta-file "$INPUT_FILE" \
            --chrom-map "${CHR_ID}:${CHR_ID}" \
            --tokenizer-path "$TOKENIZER_PATH" \
            --output "$PIPELINE_DIR/sequences.zarr"
    fi

    # --- Step 2: Predict Tokens ---
    if [[ -e "$PIPELINE_DIR/predictions.zarr" ]]; then
        echo "${LOG_PREFIX} [2/6] Skipping — predictions.zarr already exists"
    else
        echo "${LOG_PREFIX} [2/6] Generating token predictions (GPU ${GPU_ID}, batch=${BATCH_SIZE})..."
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
        echo "${LOG_PREFIX} [3/6] Skipping — intervals.zarr already exists"
    else
        echo "${LOG_PREFIX} [3/6] Detecting intervals (Viterbi decoding)..."
        $PYTHON scripts/predict.py detect_intervals \
            --input-dir "$PIPELINE_DIR/predictions.zarr" \
            --output "$PIPELINE_DIR/intervals.zarr" \
            --decoding-methods "direct,viterbi" \
            --remove-incomplete-features yes
    fi

    # --- Step 4: Export Raw GFF ---
    if [[ -f "$PIPELINE_DIR/predictions__raw.gff" ]]; then
        echo "${LOG_PREFIX} [4/6] Skipping — predictions__raw.gff already exists"
    else
        echo "${LOG_PREFIX} [4/6] Exporting raw GFF..."
        $PYTHON scripts/predict.py export_gff \
            --input "$PIPELINE_DIR/intervals.zarr" \
            --output "$PIPELINE_DIR/predictions__raw.gff" \
            --decoding-method direct \
            --min-transcript-length 3 \
            --strip-introns yes
    fi

    # --- Step 5: Post-processing Filters ---
    echo "${LOG_PREFIX} [5/6] Filtering features..."

    if [[ -f "$PIPELINE_DIR/predictions__raw__feat_len_2.gff" ]]; then
        echo "${LOG_PREFIX}   Skipping feature-length filter — output already exists"
    else
        $PYTHON scripts/gff.py filter_to_min_feature_length \
            --input "$PIPELINE_DIR/predictions__raw.gff" \
            --output "$PIPELINE_DIR/predictions__raw__feat_len_2.gff" \
            --feature-types "five_prime_UTR,three_prime_UTR,CDS" \
            --min-length 2
    fi

    if [[ -f "$PIPELINE_DIR/predictions__raw__feat_len_2__gene_len_30.gff" ]]; then
        echo "${LOG_PREFIX}   Skipping gene-length filter — output already exists"
    else
        $PYTHON scripts/gff.py filter_to_min_gene_length \
            --input "$PIPELINE_DIR/predictions__raw__feat_len_2.gff" \
            --output "$PIPELINE_DIR/predictions__raw__feat_len_2__gene_len_30.gff" \
            --min-length 30
    fi

    if [[ -f "$PIPELINE_DIR/predictions_recall__raw.gff" ]]; then
        echo "${LOG_PREFIX}   Skipping valid-gene filter — output already exists"
    else
        $PYTHON scripts/gff.py filter_to_valid_genes \
            --input "$PIPELINE_DIR/predictions__raw__feat_len_2__gene_len_30.gff" \
            --output "$PIPELINE_DIR/predictions_recall__raw.gff" \
            --require-utrs "no"
    fi

    # --- Step 6: Copy final per-chromosome outputs ---
    echo "${LOG_PREFIX} [6/6] Saving per-chromosome results..."
    cp "$PIPELINE_DIR/predictions_recall__raw.gff" "$CHR_OUTPUT_DIR/predictions_recall.gff"

    echo "${LOG_PREFIX} Done!"
}

# Export function and required variables so subshells can access them
export -f process_chromosome
export OUTPUT_DIR SPECIES_ID BASE_MODEL HEAD_MODEL TOKENIZER_PATH DTYPE PYTHON PYTHONPATH

# =================================================================
# Dispatch: sequential (1 GPU) or parallel (multi-GPU)
# =================================================================

RECALL_GFFS=()
declare -A JOB_PIDS   # pid -> chr_id
declare -A JOB_STATUS # chr_id -> exit code
CHR_ARRAY=()
while IFS= read -r chr; do
    CHR_ARRAY+=("$chr")
done <<< "$CHROM_IDS"

if [[ $NUM_GPUS -eq 1 ]]; then
    # -------------------------------------------------------
    # Single-GPU: run sequentially, streaming output directly
    # -------------------------------------------------------
    SINGLE_GPU="${GPU_ARRAY[0]}"
    SINGLE_BATCH="${GPU_BATCH_SIZES[$SINGLE_GPU]}"
    echo "Running ${CHROM_COUNT} chromosomes sequentially on GPU ${SINGLE_GPU} (batch=${SINGLE_BATCH})"
    echo ""

    for CHR_ID in "${CHR_ARRAY[@]}"; do
        echo ""
        echo "================================================================="
        echo "Processing chromosome: $CHR_ID"
        echo "================================================================="
        process_chromosome "$CHR_ID" "$SINGLE_GPU" "$SINGLE_BATCH"
        RECALL_GFFS=("${RECALL_GFFS[@]}" "${OUTPUT_DIR}/${CHR_ID}/predictions_recall.gff")
    done

else
    # -------------------------------------------------------
    # Multi-GPU: distribute chromosomes round-robin, parallel
    # -------------------------------------------------------
    echo "Running ${CHROM_COUNT} chromosome(s) in parallel across ${NUM_GPUS} GPU(s)"
    echo ""

    # Create per-job log directory
    LOG_DIR="${OUTPUT_DIR}/.logs"
    mkdir -p "$LOG_DIR"

    # Group chromosomes by target GPU (round-robin)
    declare -A GPU_QUEUES
    for i in "${!CHR_ARRAY[@]}"; do
        CHR_ID="${CHR_ARRAY[$i]}"
        GPU_IDX=$(( i % NUM_GPUS ))
        GPU_ID="${GPU_ARRAY[$GPU_IDX]}"
        if [[ -z "${GPU_QUEUES[$GPU_ID]}" ]]; then
            GPU_QUEUES[$GPU_ID]="$CHR_ID"
        else
            GPU_QUEUES[$GPU_ID]="${GPU_QUEUES[$GPU_ID]} $CHR_ID"
        fi
    done

    # Launch one worker per GPU
    for GPU_ID in "${!GPU_QUEUES[@]}"; do
        BATCH_SIZE="${GPU_BATCH_SIZES[$GPU_ID]}"
        QUEUE_STR="${GPU_QUEUES[$GPU_ID]}"
        
        # Start a subshell background worker for this GPU
        (
            for CHR_ID in $QUEUE_STR; do
                JOB_LOG="${LOG_DIR}/${CHR_ID}.log"
                echo "  Dispatching ${CHR_ID} -> GPU ${GPU_ID} (batch=${BATCH_SIZE})  [log: ${JOB_LOG}]"
                
                if ! process_chromosome "$CHR_ID" "$GPU_ID" "$BATCH_SIZE" >"$JOB_LOG" 2>&1; then
                    echo "ERROR_IN_WORKER" > "$JOB_LOG.status"
                    exit 1
                fi
            done
        ) &
        JOB_PIDS[$!]="$GPU_ID"
    done

    echo ""
    echo "Started ${NUM_GPUS} GPU worker(s). Waiting for completion..."
    echo "(Tail individual logs in ${LOG_DIR}/ to monitor progress)"
    echo ""

    # Wait for all workers and collect exit codes
    FAILED=0
    for pid in "${!JOB_PIDS[@]}"; do
        GPU_ID="${JOB_PIDS[$pid]}"
        if wait "$pid"; then
            echo "  [OK]   Worker for GPU ${GPU_ID}"
        else
            echo "  [FAIL] Worker for GPU ${GPU_ID} — check logs in ${LOG_DIR}/ for details"
            (( FAILED++ ))
        fi
    done

    if [[ $FAILED -gt 0 ]]; then
        echo ""
        echo "ERROR: ${FAILED} GPU worker(s) failed. Aborting merge step."
        echo "Fix the errors above and re-run (completed chromosomes will be skipped)."
        exit 1
    fi

    echo ""
    echo "All chromosome jobs completed successfully."

    # Collect GFF paths in original chromosome order
    for CHR_ID in "${CHR_ARRAY[@]}"; do
        RECALL_GFFS=("${RECALL_GFFS[@]}" "${OUTPUT_DIR}/${CHR_ID}/predictions_recall.gff")
    done
fi

# =================================================================
# Step 3: Merge all per-chromosome GFFs into single files
# =================================================================
echo ""
echo "================================================================="
echo "Merging per-chromosome GFFs into single files..."
echo "================================================================="

RAW_GFF="$OUTPUT_DIR/${SPECIES_ID}_GeneCAD_raw.gff"
FINAL_GFF="$OUTPUT_DIR/${SPECIES_ID}_GeneCAD_final.gff"

if [[ -f "$RAW_GFF" ]]; then
    echo "Skipping merge — ${SPECIES_ID}_GeneCAD_raw.gff already exists"
else
    $PYTHON "$MERGE_SCRIPT" \
        --output "$RAW_GFF" \
        --inputs "${RECALL_GFFS[@]}"
fi

echo ""
echo "================================================================="
echo "Step 4: Running protein refinement on merged predictions..."
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
