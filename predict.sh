#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"

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
  -n, --top-n-contigs N Predict only the N longest FASTA sequences (default: all)
    -l, --min-transcript-length N
                                                Minimum transcript length (bp) used during GFF export.
                                                Higher values can reduce tiny fragments and speed up export.
                                                (default: 3)
    -c, --cpu-workers N   CPU worker processes used in GFF export transcript grouping.
                                                Uses an order-preserving map so outputs remain deterministic.
                                                (default: 1)
  -b, --batch-size N    Inference batch size per GPU (default: auto — scaled to GPU VRAM)
  -g, --gpus LIST       Comma-separated GPU IDs to use, or 'all' for all available GPUs.
                        Chromosomes are distributed across GPUs in parallel.
                        (default: 0 — single GPU, sequential)
  --launcher CMD        Custom entrypoint command to launch predict.py (e.g. 'srun python').
                        If set, overrides automatic DDP/SLURM detection.
                        Can also be set via LAUNCHER environment variable.
  -h, --help            Show this help message

Batch size auto-detection:
  Starting guess = max(8, floor(free_gb × 0.80))
  nvidia-smi reports free memory *before* Python/model load, so the guess may overshoot.
  On failure the batch size is reduced by 20% and retried (up to 20 times). This finds
  a near-optimal size rather than jumping to half. The final working value is printed
  so you can pin it with '-b N' on future runs and skip probing entirely.

Multi-GPU dispatch (chosen automatically):
  chromosomes < GPUs  →  DDP (torchrun): all GPUs collaborate on each chromosome.
                         Ensures all GPUs are used even for small genomes.
  chromosomes ≥ GPUs  →  Per-GPU parallel: each GPU handles its own chromosomes
                         independently; up to N chromosomes run simultaneously.
                         Avoids repeated torchrun process spawns for large genomes.
  Example: --gpus 0,1,2,3   or   --gpus all

Examples:
  # Annotate a plant genome (default, batch size auto-detected, single GPU)
  bash predict.sh -i data/my_plant.fa -o output/ -s Zmays -m plant

  # Use all available GPUs
  bash predict.sh -i data/my_plant.fa -o output/ -s Zmays --gpus all

  # Use specific GPUs with custom batch size
  bash predict.sh -i data/my_plant.fa -o output/ -s Zmays --gpus 0,1 -b 32

    # Run only the 50 longest contigs/scaffolds
    bash predict.sh -i data/my_plant.fa -o output/ -s Zmays --top-n-contigs 50
USAGE
    exit 0
}

INPUT_FILE="data/example/GCA_978657495.1_TAIR12_genomic_5.fa.gz"
OUTPUT_DIR="genecad_result/Athaliana_predictions"
SPECIES_ID="Athaliana"
MODE="plant"
BATCH_SIZE_ARG="auto"
GPUS_ARG="0"
TOP_N_CONTIGS="all"
MIN_TRANSCRIPT_LENGTH="3"
CPU_WORKERS="1"
LAUNCHER_ARG="${LAUNCHER:-}"

while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--input)      INPUT_FILE="$2";      shift 2 ;;
    -o|--output)     OUTPUT_DIR="$2";      shift 2 ;;
    -s|--species)    SPECIES_ID="$2";      shift 2 ;;
    -m|--mode)       MODE="$2";            shift 2 ;;
    -n|--top-n-contigs) TOP_N_CONTIGS="$2"; shift 2 ;;
    -l|--min-transcript-length) MIN_TRANSCRIPT_LENGTH="$2"; shift 2 ;;
    -c|--cpu-workers) CPU_WORKERS="$2"; shift 2 ;;
    -b|--batch-size) BATCH_SIZE_ARG="$2"; shift 2 ;;
    -g|--gpus)       GPUS_ARG="$2";       shift 2 ;;
    --launcher)      LAUNCHER_ARG="$2";   shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

if [[ "$TOP_N_CONTIGS" != "all" ]]; then
    if ! [[ "$TOP_N_CONTIGS" =~ ^[0-9]+$ ]] || [[ "$TOP_N_CONTIGS" -lt 1 ]]; then
        echo "Error: --top-n-contigs must be a positive integer or 'all'."
        exit 1
    fi
fi

if ! [[ "$MIN_TRANSCRIPT_LENGTH" =~ ^[0-9]+$ ]]; then
    echo "Error: --min-transcript-length must be a non-negative integer."
    exit 1
fi

if ! [[ "$CPU_WORKERS" =~ ^[0-9]+$ ]] || [[ "$CPU_WORKERS" -lt 1 ]]; then
    echo "Error: --cpu-workers must be a positive integer."
    exit 1
fi

echo "================================================================="
echo "GeneCAD Prediction Pipeline"
echo "================================================================="

# Download default Arabidopsis sequence if missing
if [[ "$INPUT_FILE" == "data/example/GCA_978657495.1_TAIR12_genomic_5.fa.gz" && ! -f "$INPUT_FILE" ]]; then
    echo "Downloading default Arabidopsis thaliana sequence..."
    mkdir -p "$(dirname "$INPUT_FILE")"
    wget -qO "$INPUT_FILE" "https://huggingface.co/datasets/plantcad/genecad-dev/resolve/main/data/plant/fasta/example/GCA_978657495.1_TAIR12_genomic_5.fa.gz"
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input FASTA file '$INPUT_FILE' not found."
    exit 1
fi

# Model selection based on --mode
case "$MODE" in
  plant)
    BASE_MODEL="emarro/pcad2-200M-cnet-baseline"
    HEAD_MODEL="plantcad/genecad_plant"
    ;;
  animal)
    BASE_MODEL="emarro/pcad2_vert_small"
    HEAD_MODEL="plantcad/genecad_vert"
    ;;
  *)
    echo "Error: Unknown mode '$MODE'. Valid options are: plant, animal"
    exit 1
    ;;
esac

# =================================================================
# GPU Resolution
# =================================================================

if [[ "$GPUS_ARG" == "all" ]]; then
    GPU_IDS_ALL=""
    if command -v nvidia-smi &>/dev/null; then
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

IFS=',' read -ra GPU_ARRAY <<< "$GPU_LIST_STR"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "Using GPU(s): ${GPU_ARRAY[*]}  (${NUM_GPUS} total)"
echo "Input FASTA: $INPUT_FILE"
echo "Output Dir:  $OUTPUT_DIR"
echo "Species ID:  $SPECIES_ID"
echo "Mode:        $MODE  ($BASE_MODEL + $HEAD_MODEL)"
echo "Top contigs: $TOP_N_CONTIGS"
echo "Min tx len:  $MIN_TRANSCRIPT_LENGTH"
echo "CPU workers: $CPU_WORKERS"
echo "================================================================="

# =================================================================
# Per-GPU batch size detection
# =================================================================
resolve_batch_size_for_gpu() {
    local gpu_id="$1"
    if [[ "$BATCH_SIZE_ARG" != "auto" ]]; then
        echo "$BATCH_SIZE_ARG"
        return
    fi
    if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null; then
        echo "8"
        return
    fi
    local GPU_MEM_MB
    GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        --id="$gpu_id" 2>/dev/null | head -1 | tr -d ' ')
    if [[ -z "$GPU_MEM_MB" ]] || ! [[ "$GPU_MEM_MB" =~ ^[0-9]+$ ]]; then
        echo "8"
        return
    fi
    # Start from a GPU-memory-based estimate. nvidia-smi memory.free is sampled
    # before Python, PyTorch, and the model load, so this is a starting point;
    # the retry loop below still shrinks it if the real run needs less.
    local FREE_GB=$(( GPU_MEM_MB / 1024 ))
    # Use a more aggressive starting point so inference fills more of the GPU
    # before the retry loop has to back off.
    local BS=$(( FREE_GB * 9 / 10 ))   # × 0.90
    [[ $BS -lt 8 ]] && BS=8
    echo $BS
}

declare -A GPU_BATCH_SIZES
for gpu_id in "${GPU_ARRAY[@]}"; do
    bs=$(resolve_batch_size_for_gpu "$gpu_id")
    GPU_BATCH_SIZES[$gpu_id]=$bs
    if [[ "$BATCH_SIZE_ARG" == "auto" ]]; then
        if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
            GPU_MEM_MB_FOR_LOG=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$gpu_id" 2>/dev/null | head -1 | tr -d ' ')
            echo "  GPU ${gpu_id}: batch size ${bs} (detected ${GPU_MEM_MB_FOR_LOG} MiB free VRAM)"
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

MERGE_SCRIPT="$SCRIPT_DIR/scripts/merge_gff.py"
TOKENIZER_PATH="$BASE_MODEL"
DTYPE="bfloat16"

if [[ -n "${GENECAD_PYTHON:-}" && -x "$GENECAD_PYTHON" ]]; then
    PYTHON="$GENECAD_PYTHON"
elif [[ -n "$VIRTUAL_ENV" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
    VIRTUAL_ENV="$(pwd)/.venv"
    PYTHON=".venv/bin/python"
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    VIRTUAL_ENV="$SCRIPT_DIR/.venv"
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON="uv run python"
fi
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_DIR"

# =================================================================
# Step 1: Discover chromosomes from FASTA headers
# =================================================================
echo "================================================================="
echo "Discovering chromosomes from FASTA file..."
echo "================================================================="

if [[ "$TOP_N_CONTIGS" == "all" ]]; then
    if [[ "$INPUT_FILE" == *.gz ]]; then
        CHROM_IDS=$(zcat "$INPUT_FILE" | grep "^>" | sed 's/^>//' | awk '{print $1}')
    else
        CHROM_IDS=$(grep "^>" "$INPUT_FILE" | sed 's/^>//' | awk '{print $1}')
    fi
else
    TOP_IDS=""
    if [[ "$INPUT_FILE" == *.gz ]]; then
        TOP_IDS=$(
            zcat "$INPUT_FILE" | awk '
                /^>/ {
                    if (id != "") print len "\t" id;
                    id = $0;
                    sub(/^>/, "", id);
                    split(id, parts, /[ \t]/);
                    id = parts[1];
                    len = 0;
                    next;
                }
                {
                    gsub(/[ \t\r\n]/, "", $0);
                    len += length($0);
                }
                END {
                    if (id != "") print len "\t" id;
                }
            ' | sort -nr -k1,1 | head -n "$TOP_N_CONTIGS" | awk '{print $2}'
        )
        CHROM_IDS=$(awk '
            NR==FNR {
                keep[$1] = 1;
                next;
            }
            /^>/ {
                id = $0;
                sub(/^>/, "", id);
                split(id, parts, /[ \t]/);
                id = parts[1];
                if (id in keep) print id;
            }
        ' <(printf '%s\n' "$TOP_IDS") <(zcat "$INPUT_FILE"))
    else
        TOP_IDS=$(
            awk '
                /^>/ {
                    if (id != "") print len "\t" id;
                    id = $0;
                    sub(/^>/, "", id);
                    split(id, parts, /[ \t]/);
                    id = parts[1];
                    len = 0;
                    next;
                }
                {
                    gsub(/[ \t\r\n]/, "", $0);
                    len += length($0);
                }
                END {
                    if (id != "") print len "\t" id;
                }
            ' "$INPUT_FILE" | sort -nr -k1,1 | head -n "$TOP_N_CONTIGS" | awk '{print $2}'
        )
        CHROM_IDS=$(awk '
            NR==FNR {
                keep[$1] = 1;
                next;
            }
            /^>/ {
                id = $0;
                sub(/^>/, "", id);
                split(id, parts, /[ \t]/);
                id = parts[1];
                if (id in keep) print id;
            }
        ' <(printf '%s\n' "$TOP_IDS") "$INPUT_FILE")
    fi
fi

CHROM_COUNT=$(echo "$CHROM_IDS" | sed '/^$/d' | wc -l)
if [[ "$CHROM_COUNT" -eq 0 ]]; then
    echo "Error: No sequences found in FASTA after applying filters."
    exit 1
fi

if [[ "$TOP_N_CONTIGS" == "all" ]]; then
    echo "Found $CHROM_COUNT chromosomes/sequences:"
else
    echo "Selected top $CHROM_COUNT longest chromosomes/sequences:"
fi
echo "$CHROM_IDS"
echo ""

# =================================================================
# Step 2: Per-chromosome pipeline (all 6 steps)
# =================================================================

process_chromosome() {
    local CHR_ID="$1"
    local BATCH_SIZE="$2"
    local GPU_ID="$3"   # which GPU this chromosome runs on
    local LOG_PREFIX="[${CHR_ID}@GPU${GPU_ID}]"

    local CHR_OUTPUT_DIR="${OUTPUT_DIR}/${CHR_ID}"
    local PIPELINE_DIR="${CHR_OUTPUT_DIR}/pipeline"
    mkdir -p "$PIPELINE_DIR"

    if [[ -f "$CHR_OUTPUT_DIR/predictions_recall.gff" ]]; then
        echo "${LOG_PREFIX} Already complete — skipping (delete $CHR_OUTPUT_DIR to rerun)"
        return 0
    fi

    # --- Step 1: Extract Sequences ---
    if [[ -e "$PIPELINE_DIR/sequences.zarr" ]]; then
        echo "${LOG_PREFIX} [1/6] Skipping — sequences.zarr already exists"
    else
        echo "${LOG_PREFIX} [1/6] Extracting sequences..."
        $PYTHON "$SCRIPT_DIR/scripts/extract.py" extract_fasta_file \
            --species-id "$SPECIES_ID" \
            --fasta-file "$INPUT_FILE" \
            --chrom-map "${CHR_ID}:${CHR_ID}" \
            --tokenizer-path "$TOKENIZER_PATH" \
            --output "$PIPELINE_DIR/sequences.zarr"
    fi

    # --- Step 2: Predict ---
    if [[ -e "$PIPELINE_DIR/predictions.zarr" ]]; then
        echo "${LOG_PREFIX} [2/6] Skipping — predictions.zarr already exists"
    else
        local gpu_id="$GPU_ID"
        local bs="$BATCH_SIZE"
        local attempt=0
        local max_attempts=20  # 20% reduction per step: ~20 steps to go from 256→1
        local success=0
        while [[ $attempt -lt $max_attempts ]]; do
            local exit_code=0
            if [[ "$PREDICT_MODE" == "ddp" ]]; then
                echo "${LOG_PREFIX} [2/6] Prediction via DDP (batch=${bs}/GPU, attempt $((attempt+1))/${max_attempts})..."
                CUDA_VISIBLE_DEVICES="$GPU_LIST_STR" \
                $PY_LAUNCHER \
                    "$SCRIPT_DIR/scripts/predict.py" create_predictions \
                    --chromosome-id "$CHR_ID" \
                    --input "$PIPELINE_DIR/sequences.zarr" \
                    --output-dir "$PIPELINE_DIR/predictions.zarr" \
                    --model-path "$BASE_MODEL" \
                    --model-checkpoint "$HEAD_MODEL" \
                    --species-id "$SPECIES_ID" \
                    --batch-size "$bs" \
                    --dtype "$DTYPE" \
                    --window-size 8192 \
                    --stride 4096 || exit_code=$?
            elif [[ "$PREDICT_MODE" == "ddp_slurm" ]]; then
                echo "${LOG_PREFIX} [2/6] Prediction via SLURM distributed (batch=${bs}/GPU, attempt $((attempt+1))/${max_attempts})..."
                $PY_LAUNCHER "$SCRIPT_DIR/scripts/predict.py" create_predictions \
                    --chromosome-id "$CHR_ID" \
                    --input "$PIPELINE_DIR/sequences.zarr" \
                    --output-dir "$PIPELINE_DIR/predictions.zarr" \
                    --model-path "$BASE_MODEL" \
                    --model-checkpoint "$HEAD_MODEL" \
                    --species-id "$SPECIES_ID" \
                    --batch-size "$bs" \
                    --dtype "$DTYPE" \
                    --window-size 8192 \
                    --stride 4096 || exit_code=$?
            else
                echo "${LOG_PREFIX} [2/6] Prediction on GPU ${gpu_id} (batch=${bs}, attempt $((attempt+1))/${max_attempts})..."
                CUDA_VISIBLE_DEVICES="$gpu_id" \
                $PY_LAUNCHER "$SCRIPT_DIR/scripts/predict.py" create_predictions \
                    --chromosome-id "$CHR_ID" \
                    --input "$PIPELINE_DIR/sequences.zarr" \
                    --output-dir "$PIPELINE_DIR/predictions.zarr" \
                    --model-path "$BASE_MODEL" \
                    --model-checkpoint "$HEAD_MODEL" \
                    --species-id "$SPECIES_ID" \
                    --batch-size "$bs" \
                    --dtype "$DTYPE" \
                    --tqdm-position "$gpu_id" \
                    --window-size 8192 \
                    --stride 4096 || exit_code=$?
            fi
            if [[ $exit_code -eq 0 ]]; then
                success=1
                if [[ $attempt -gt 0 ]]; then
                    echo "${LOG_PREFIX} [2/6] Succeeded with batch size ${bs} after ${attempt} retry(s)."
                    echo "${LOG_PREFIX}   TIP: Add '-b ${bs}' to future runs to skip probing."
                fi
                break
            fi
            local next_bs=$(( bs * 4 / 5 ))   # reduce by 20%
            [[ $next_bs -ge $bs ]] && next_bs=$(( bs - 1 ))  # guard against bs<5 rounding to same value
            [[ $next_bs -lt 1 ]] && next_bs=1
            echo "${LOG_PREFIX} [2/6] Failed (exit ${exit_code}). Reducing batch size by 20%%: ${bs} → ${next_bs}..."
            bs=$next_bs
            rm -rf "$PIPELINE_DIR/predictions.zarr"
            attempt=$(( attempt + 1 ))
        done
        if [[ $success -ne 1 ]]; then
            echo "${LOG_PREFIX} ERROR: Prediction failed after ${max_attempts} attempts (last batch size: ${bs})."
            echo "${LOG_PREFIX}   This is likely not an OOM issue. Check stderr above."
            return 1
        fi
    fi

    # --- Step 3: Detect Intervals ---
    if [[ -e "$PIPELINE_DIR/intervals.zarr" ]]; then
        echo "${LOG_PREFIX} [3/6] Skipping — intervals.zarr already exists"
    else
        echo "${LOG_PREFIX} [3/6] Detecting intervals (Viterbi decoding)..."
        $PYTHON "$SCRIPT_DIR/scripts/predict.py" detect_intervals \
            --input-dir "$PIPELINE_DIR/predictions.zarr" \
            --output "$PIPELINE_DIR/intervals.zarr" \
            --decoding-methods "viterbi" \
            --remove-incomplete-features yes \
            --domain "$MODE"
    fi

    # --- Step 4: Export Raw GFF ---
    if [[ -f "$PIPELINE_DIR/predictions__raw.gff" ]]; then
        echo "${LOG_PREFIX} [4/6] Skipping — predictions__raw.gff already exists"
    else
        echo "${LOG_PREFIX} [4/6] Exporting raw GFF..."
        local export_tqdm_args=()
        if [[ -n "$gpu_id" ]]; then
            export_tqdm_args=(--tqdm-position "$gpu_id")
        fi
        $PYTHON "$SCRIPT_DIR/scripts/predict.py" export_gff \
            --input "$PIPELINE_DIR/intervals.zarr" \
            --output "$PIPELINE_DIR/predictions__raw.gff" \
            --decoding-method viterbi \
            --min-transcript-length "$MIN_TRANSCRIPT_LENGTH" \
            --cpu-workers "$CPU_WORKERS" \
            --strip-introns yes \
            "${export_tqdm_args[@]}"
    fi

    # --- Step 5: Post-processing Filters ---
    echo "${LOG_PREFIX} [5/6] Filtering features..."

    if [[ -f "$PIPELINE_DIR/predictions__raw__feat_len_2.gff" ]]; then
        echo "${LOG_PREFIX}   Skipping feature-length filter — output already exists"
    else
        $PYTHON "$SCRIPT_DIR/scripts/gff.py" filter_to_min_feature_length \
            --input "$PIPELINE_DIR/predictions__raw.gff" \
            --output "$PIPELINE_DIR/predictions__raw__feat_len_2.gff" \
            --feature-types "five_prime_UTR,three_prime_UTR,CDS" \
            --min-length 2
    fi

    if [[ -f "$PIPELINE_DIR/predictions__raw__feat_len_2__gene_len_30.gff" ]]; then
        echo "${LOG_PREFIX}   Skipping gene-length filter — output already exists"
    else
        $PYTHON "$SCRIPT_DIR/scripts/gff.py" filter_to_min_gene_length \
            --input "$PIPELINE_DIR/predictions__raw__feat_len_2.gff" \
            --output "$PIPELINE_DIR/predictions__raw__feat_len_2__gene_len_30.gff" \
            --min-length 30
    fi

    if [[ -f "$PIPELINE_DIR/predictions_recall__raw.gff" ]]; then
        echo "${LOG_PREFIX}   Skipping valid-gene filter — output already exists"
    else
        $PYTHON "$SCRIPT_DIR/scripts/gff.py" filter_to_valid_genes \
            --input "$PIPELINE_DIR/predictions__raw__feat_len_2__gene_len_30.gff" \
            --output "$PIPELINE_DIR/predictions_recall__raw.gff" \
            --require-utrs "no"
    fi

    # --- Step 6: Copy final output ---
    echo "${LOG_PREFIX} [6/6] Saving results..."
    cp "$PIPELINE_DIR/predictions_recall__raw.gff" "$CHR_OUTPUT_DIR/predictions_recall.gff"
    echo "${LOG_PREFIX} Done!"
}

export -f process_chromosome
export OUTPUT_DIR SPECIES_ID BASE_MODEL HEAD_MODEL TOKENIZER_PATH DTYPE PYTHON PYTHONPATH
export GPU_LIST_STR NUM_GPUS

CHR_ARRAY=()
while IFS= read -r chr; do
    CHR_ARRAY+=("$chr")
done <<< "$CHROM_IDS"

# =================================================================
# Choose dispatch strategy
#
#   DDP  (torchrun)  — when chromosomes < GPUs:
#     All GPUs collaborate on each chromosome; processed sequentially.
#     Ensures every GPU is busy even for tiny genomes.
#
#   Per-GPU parallel — when chromosomes >= GPUs:
#     Each GPU owns its chromosomes independently; up to NUM_GPUS
#     chromosomes run at the same time. Avoids 1000× torchrun spawns.
# =================================================================

echo "================================================================="
# Only switch to SLURM distributed mode when the shell is actually running in
# a multi-rank context. A bare SLURM allocation (SLURM_JOB_ID only) should keep
# normal per-GPU scheduling, otherwise we serialize chromosomes and underutilize GPUs.
if [[ "${WORLD_SIZE:-1}" -gt 1 || "${SLURM_NTASKS:-1}" -gt 1 || -n "${SLURM_PROCID:-}" || -n "${SLURM_LOCALID:-}" ]]; then
    PREDICT_MODE="ddp_slurm"
    DDP_BATCH="${GPU_BATCH_SIZES[${GPU_ARRAY[0]}]}"
    for gid in "${GPU_ARRAY[@]}"; do
        [[ "${GPU_BATCH_SIZES[$gid]}" -lt "$DDP_BATCH" ]] && DDP_BATCH="${GPU_BATCH_SIZES[$gid]}"
    done
    echo "Processing ${CHROM_COUNT} chromosome(s) in SLURM distributed mode."
    echo "  (SLURM/WORLD_SIZE environment detected → DDP handled by SLURM integration)"
    echo "  Batch size per GPU: ${DDP_BATCH}"
elif [[ $NUM_GPUS -gt 1 && $CHROM_COUNT -lt $NUM_GPUS ]]; then
    PREDICT_MODE="ddp"
    # Use the minimum batch size across GPUs so no single GPU OOMs.
    DDP_BATCH="${GPU_BATCH_SIZES[${GPU_ARRAY[0]}]}"
    for gid in "${GPU_ARRAY[@]}"; do
        [[ "${GPU_BATCH_SIZES[$gid]}" -lt "$DDP_BATCH" ]] && DDP_BATCH="${GPU_BATCH_SIZES[$gid]}"
    done
    echo "Processing ${CHROM_COUNT} chromosome(s) with DDP across all ${NUM_GPUS} GPUs."
    echo "  (${CHROM_COUNT} chromosomes < ${NUM_GPUS} GPUs → DDP uses all GPUs per chromosome)"
    echo "  Batch size per GPU: ${DDP_BATCH}"
else
    PREDICT_MODE="single"
    echo "Processing ${CHROM_COUNT} chromosome(s) in parallel — one GPU per chromosome."
    [[ $NUM_GPUS -gt 1 ]] && echo "  Up to ${NUM_GPUS} chromosomes run simultaneously."
fi

# Set the launcher dynamically
if [[ -n "$LAUNCHER_ARG" ]]; then
    PY_LAUNCHER="$LAUNCHER_ARG"
    echo "  Custom launcher overridden: $PY_LAUNCHER"
elif [[ "$PREDICT_MODE" == "ddp" ]]; then
    PY_LAUNCHER="$PYTHON -m torch.distributed.run --standalone --nproc_per_node=${NUM_GPUS}"
else
    PY_LAUNCHER="$PYTHON"
fi

export PREDICT_MODE PY_LAUNCHER
echo "================================================================="

FAILED=0

if [[ "$PREDICT_MODE" == "ddp" || "$PREDICT_MODE" == "ddp_slurm" ]]; then
    # Sequential DDP — all GPUs on each chromosome one at a time
    for CHR_ID in "${CHR_ARRAY[@]}"; do
        process_chromosome "$CHR_ID" "$DDP_BATCH" "" || FAILED=$(( FAILED + 1 ))
    done
else
    # Per-GPU parallel — round-robin, NUM_GPUS concurrent jobs
    declare -a PIDS=()
    chr_idx=0
    for CHR_ID in "${CHR_ARRAY[@]}"; do
        gpu_id="${GPU_ARRAY[$(( chr_idx % NUM_GPUS ))]}"
        bs="${GPU_BATCH_SIZES[$gpu_id]}"

        # Wait for the oldest slot before launching, keeping exactly NUM_GPUS live jobs
        if [[ ${#PIDS[@]} -ge $NUM_GPUS ]]; then
            wait "${PIDS[0]}"
            [[ $? -ne 0 ]] && FAILED=$(( FAILED + 1 ))
            PIDS=("${PIDS[@]:1}")
        fi

        process_chromosome "$CHR_ID" "$bs" "$gpu_id" &
        PIDS+=($!)
        chr_idx=$(( chr_idx + 1 ))
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid"
        [[ $? -ne 0 ]] && FAILED=$(( FAILED + 1 ))
    done
fi

if [[ $FAILED -gt 0 ]]; then
    echo "ERROR: $FAILED chromosome(s) failed. See output above for details."
    exit 1
fi

RECALL_GFFS=()
for CHR_ID in "${CHR_ARRAY[@]}"; do
    RECALL_GFFS=("${RECALL_GFFS[@]}" "${OUTPUT_DIR}/${CHR_ID}/predictions_recall.gff")
done

# =================================================================
# Merge all per-chromosome GFFs into single files
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
echo "Running protein refinement on merged predictions..."
echo "================================================================="

if [[ -f "$FINAL_GFF" ]]; then
    echo "Skipping refinement — ${SPECIES_ID}_GeneCAD_final.gff already exists"
else
    $PYTHON "$SCRIPT_DIR/scripts/refine.py" \
        --gff "$RAW_GFF" \
        --genome "$INPUT_FILE" \
        --out "$FINAL_GFF" \
        --gpus "$GPU_LIST_STR"
fi

echo ""
echo "================================================================="
echo "All done! Final predictions saved to:"
echo "Raw:   $RAW_GFF"
echo "Final: $FINAL_GFF"
echo "================================================================="
