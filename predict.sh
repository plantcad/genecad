#!/bin/bash
set -e
if (( BASH_VERSINFO[0] < 4 )); then
    echo "Error: predict.sh needs bash 4 or newer (this is bash $BASH_VERSION)." >&2
    exit 1
fi
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
  -i, --input-fasta PATH      Input genome FASTA file
                        (default: downloads Arabidopsis thaliana TAIR12 example)
  -o, --output-dir DIR      Output directory (default: genecad_result/Athaliana_predictions)
  -s, --species-id NAME    Species label prefixed on output filenames (default: Athaliana)
  -m, --mode MODE       Model to use: plant | animal  (default: plant)
  -n, --top-n-contigs N Predict only the N longest FASTA sequences (default: all)
    -l, --min-transcript-length N
                                                Minimum transcript length (bp) used during GFF export.
                                                Higher values can reduce tiny fragments and speed up export.
                                                (default: 3)
      --orf-max-shift N       Maximum distance (nt, in spliced transcript coordinates)
                                                that the start and stop codon may be moved when
                                                repairing CDS boundaries against the genome
                                                sequence. Repairs never alter exon structure and
                                                never leave the predicted exonic sequence; models
                                                that cannot be resolved are flagged partial=true
                                                rather than forced. Use 0 to disable repair.
                                                (default: 300)
    --decoder MODE          How per-base predictions become gene models (default: hybrid).
                                                hybrid: plain 5-state Viterbi genome-wide, then frame-aware
                                                decoding only around genes that need it: transcripts ORF
                                                repair cannot fix are re-decoded locally, and consecutive
                                                same-strand genes that one frame-aware transcript spans are
                                                merged. Frame-aware decoding constrains the CDS to begin on
                                                ATG, end on a stop codon, stay in frame across introns and
                                                contain no in-frame stop.
                                                frame-aware: frame-aware decoding of whole chromosomes
                                                (v0.5.0 default; creates more tiny and split genes).
                                                plain: 5-state Viterbi only.
    --no-frame-aware        Deprecated, use --decoder plain instead.
    --merge-max-gap N       Largest gap (bp) between consecutive same-strand genes for which
                                                hybrid decoding tries a merge (default: 20000).
    --keep-partial          Keep transcripts that cannot be made a valid ORF (flagged
                                                partial=true) instead of dropping them.
    --clean-intermediates   Save disk space. Delete the sequence and interval files of a
                                                chromosome once its predictions are decoded, and the
                                                prediction files of all chromosomes once the final GFF
                                                is written. The GFF files are kept, so a run that is
                                                repeated afterwards is not redone. Prediction files
                                                are needed again only if the hybrid GFF in
                                                intermediate/ is deleted; they are then predicted again.
    --allow-missing-predictions
                                                Hybrid decoding needs the prediction files of every
                                                sequence. If they were deleted (for example to save disk
                                                space) the sequence is predicted again, which needs a
                                                GPU. With this option it is not: those sequences keep
                                                their partial genes and are not merged.
    --min-intron-length N   Shortest intron frame-aware decoding may emit (default: 20).
                                                Guards against short introns being invented to step over
                                                an in-frame stop codon. Lower it for compact genomes
                                                with genuinely short introns.
    --min-coding-run-length N     Runs of coding sequence adjacent to an intron shorter than this
                                                are penalized, not forbidden (default: 9; see
                                                --exon-length-strictness). Guards against an intron
                                                being invented immediately after the start codon or
                                                immediately after another intron, without also
                                                destroying genuine short exons. Use 0 to disable.
    --exon-length-strictness N
                                                How strongly to penalize a run of coding sequence below
                                                --min-coding-run-length (default: 16). 0 removes the
                                                penalty; larger values converge on treating
                                                --min-coding-run-length as a hard floor.
    --allow-u12-introns     Also allow U12-type AT-AC introns during frame-aware
                                                decoding (default: only GT-AG/GC-AG). These are real but
                                                rare; enabling this roughly doubles the intron state count
                                                and slows decoding accordingly.
  -c, --cpu-workers N   CPU worker processes used in GFF export transcript grouping.
                                                Uses an order-preserving map so outputs remain deterministic.
                                                (default: 1)
      --max-parallel-chromosomes N|auto
                                                How many chromosomes are decoded and exported to GFF at
                                                the same time. This is not a number of CPUs. Frame-aware
                                                decoding needs about 0.35 GB of RAM per Mb of chromosome
                                                and plain or hybrid decoding about 0.05 GB, so several
                                                large chromosomes at once can run out of memory. auto: as
                                                many as fit in the available RAM, at most one per CPU core
                                                (up to 16).
                                                (default: auto)
  -b, --batch-size N    Inference batch size per GPU (default: auto — scaled to GPU VRAM)
  -g, --gpus LIST       Comma-separated GPU IDs to use, or 'all' for all available GPUs.
                        Chromosomes are distributed across GPUs in parallel.
                        (default: 0 — single GPU, sequential)
  --launcher CMD        Custom entrypoint command to launch predict.py (e.g. 'srun python').
                        If set, overrides automatic DDP/SLURM detection.
                        Can also be set via LAUNCHER environment variable.
  --model-checkpoint PATH_OR_REPO
                        Override the GeneCAD head checkpoint selected by --mode.
                        Accepts a local .ckpt path or a Hugging Face model repo ID.
  -h, --help            Show this help message

Batch size auto-detection:
  Starting guess = min(35, max(8, floor(free_gb × 0.90)))
  The cap of 35 avoids a CUDA illegal-memory-access crash seen at larger batches on 80 GB H100s.
  nvidia-smi reports free memory *before* Python/model load, so the guess may overshoot.
  On CUDA OOM the worker reduces the batch by 20% and retries the missing windows
  without reloading the model. It stops if even one window cannot fit. Other errors
  stop the worker immediately; completed segments remain available for resume.
  Whatever value a chromosome succeeds with is also cached per GPU under
  <output-dir>/.state/ and reused as the starting point for the next chromosome
  on that GPU, so probing/retrying only happens once per GPU as long as the
  hardware and free VRAM stay the same.

Multi-GPU dispatch (chosen automatically):
  chromosomes < GPUs  →  DDP (torchrun): all GPUs collaborate on each chromosome.
                         Ensures all GPUs are used even for small genomes.
  chromosomes ≥ GPUs  →  Per-GPU parallel: each GPU handles its own chromosomes
                         independently; up to N chromosomes run simultaneously.
                         Each GPU loads its model once for its scaffold manifest.
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
ORF_MAX_SHIFT="300"
DECODER="hybrid"
KEEP_PARTIAL="0"
ALLOW_MISSING_PREDICTIONS="0"
CLEAN_INTERMEDIATES="0"
MERGE_MAX_GAP="20000"
MIN_INTRON_LENGTH="20"
MIN_CODING_RUN_LENGTH="9"
EXON_LENGTH_STRICTNESS="16"
ALLOW_U12_INTRONS="0"
CPU_WORKERS="1"
MAX_PARALLEL_CHROMOSOMES="auto"
LAUNCHER_ARG="${LAUNCHER:-}"
MODEL_CHECKPOINT_ARG=""
BASE_MODEL_ARG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--input-fasta)      INPUT_FILE="$2";      shift 2 ;;
    -o|--output-dir)     OUTPUT_DIR="$2";      shift 2 ;;
    -s|--species-id)    SPECIES_ID="$2";      shift 2 ;;
    -m|--mode)       MODE="$2";            shift 2 ;;
    -n|--top-n-contigs) TOP_N_CONTIGS="$2"; shift 2 ;;
    -l|--min-transcript-length) MIN_TRANSCRIPT_LENGTH="$2"; shift 2 ;;
    --orf-max-shift) ORF_MAX_SHIFT="$2"; shift 2 ;;
    --decoder) DECODER="$2"; shift 2 ;;
    --no-frame-aware)
        echo "Warning: --no-frame-aware is deprecated, use --decoder plain instead." >&2
        DECODER="plain"; shift ;;
    --merge-max-gap) MERGE_MAX_GAP="$2"; shift 2 ;;
    --keep-partial) KEEP_PARTIAL="1"; shift ;;
    --allow-missing-predictions) ALLOW_MISSING_PREDICTIONS="1"; shift ;;
    --clean-intermediates) CLEAN_INTERMEDIATES="1"; shift ;;
    --min-intron-length) MIN_INTRON_LENGTH="$2"; shift 2 ;;
    --min-coding-run-length) MIN_CODING_RUN_LENGTH="$2"; shift 2 ;;
    --exon-length-strictness) EXON_LENGTH_STRICTNESS="$2"; shift 2 ;;
    --allow-u12-introns) ALLOW_U12_INTRONS="1"; shift ;;
    -c|--cpu-workers) CPU_WORKERS="$2"; shift 2 ;;
    --max-parallel-chromosomes) MAX_PARALLEL_CHROMOSOMES="$2"; shift 2 ;;
    -b|--batch-size) BATCH_SIZE_ARG="$2"; shift 2 ;;
    -g|--gpus)       GPUS_ARG="$2";       shift 2 ;;
    --launcher)      LAUNCHER_ARG="$2";   shift 2 ;;
    --model-checkpoint) MODEL_CHECKPOINT_ARG="$2"; shift 2 ;;
    --base-model)    BASE_MODEL_ARG="$2"; shift 2 ;;
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

if ! [[ "$ORF_MAX_SHIFT" =~ ^[0-9]+$ ]]; then
    echo "Error: --orf-max-shift must be a non-negative integer."
    exit 1
fi

if ! [[ "$MIN_INTRON_LENGTH" =~ ^[0-9]+$ ]] || [[ "$MIN_INTRON_LENGTH" -lt 5 ]]; then
    echo "Error: --min-intron-length must be an integer of at least 5."
    exit 1
fi

if ! [[ "$MIN_CODING_RUN_LENGTH" =~ ^[0-9]+$ ]]; then
    echo "Error: --min-coding-run-length must be a non-negative integer."
    exit 1
fi

if ! [[ "$EXON_LENGTH_STRICTNESS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Error: --exon-length-strictness must be a non-negative number."
    exit 1
fi

if ! [[ "$CPU_WORKERS" =~ ^[0-9]+$ ]] || [[ "$CPU_WORKERS" -lt 1 ]]; then
    echo "Error: --cpu-workers must be a positive integer."
    exit 1
fi

if [[ "$MAX_PARALLEL_CHROMOSOMES" != "auto" ]] && { ! [[ "$MAX_PARALLEL_CHROMOSOMES" =~ ^[0-9]+$ ]] || [[ "$MAX_PARALLEL_CHROMOSOMES" -lt 1 ]]; }; then
    echo "Error: --max-parallel-chromosomes must be a positive integer or 'auto'."
    exit 1
fi

case "$DECODER" in
    hybrid|plain) FRAME_AWARE="0" ;;
    frame-aware)  FRAME_AWARE="1" ;;
    *) echo "Error: --decoder must be hybrid, frame-aware or plain."; exit 1 ;;
esac

if ! [[ "$MERGE_MAX_GAP" =~ ^[0-9]+$ ]]; then
    echo "Error: --merge-max-gap must be a non-negative integer."
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
    HEAD_MODEL="plantcad/genecad_animal"
    ;;
  *)
    echo "Error: Unknown mode '$MODE'. Valid options are: plant, animal"
    exit 1
    ;;
esac

if [[ -n "$MODEL_CHECKPOINT_ARG" ]]; then
    HEAD_MODEL="$MODEL_CHECKPOINT_ARG"
fi
if [[ -n "$BASE_MODEL_ARG" ]]; then
    BASE_MODEL="$BASE_MODEL_ARG"
    TOKENIZER_PATH="$BASE_MODEL"
fi

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
echo "ORF shift:   $ORF_MAX_SHIFT"
echo "Decoder:     $DECODER (keep partial: $KEEP_PARTIAL, merge max gap: $MERGE_MAX_GAP)"
echo "Frame-aware: $FRAME_AWARE (min intron $MIN_INTRON_LENGTH, min exon $MIN_CODING_RUN_LENGTH @ strictness $EXON_LENGTH_STRICTNESS, allow U12 introns: $ALLOW_U12_INTRONS)"
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
    # the prediction worker shrinks it if the real run needs less.
    local FREE_GB=$(( GPU_MEM_MB / 1024 ))
    # Use a more aggressive starting point so inference fills more of the GPU
    # before the prediction worker has to back off.
    local BS=$(( FREE_GB * 9 / 10 ))   # × 0.90
    [[ $BS -lt 8 ]] && BS=8
    # Cap at 35 (what a 40 GB A100 gets). On 80 GB H100s the estimate is 71, and
    # runs at 71 died on the first batch with "CUDA error: an illegal memory
    # access", which the worker cannot recover from the way it does from OOM.
    [[ $BS -gt 35 ]] && BS=35
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

# Persistent workers save an OOM-reduced batch cap here for the next run.
# Each GPU has its own file; rank zero writes the shared DDP starting cap.
BATCH_SIZE_STATE_DIR="$OUTPUT_DIR/.state"
mkdir -p "$BATCH_SIZE_STATE_DIR"

# An output directory holds the results of one FASTA file: chromosomes of another genome
# with the same names would silently reuse its results. The file is recorded by path,
# size and modification time, and by the SHA-256 of its content (decompressed), which is
# computed only when one of those changed: the same sequences moved, touched or
# compressed are still accepted.
INPUT_FILE="$INPUT_FILE" OUTPUT_DIR="$OUTPUT_DIR" \
    $PYTHON - "$BATCH_SIZE_STATE_DIR/input_fasta.json" <<'PYEOF' || exit 1
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

record = Path(sys.argv[1])
fasta = Path(os.environ["INPUT_FILE"])
stat = fasta.stat()
current = {
    "path": str(fasta.resolve()),
    "size": stat.st_size,
    "mtime_ns": stat.st_mtime_ns,
}
previous = json.loads(record.read_text()) if record.is_file() else None
if previous and all(previous.get(key) == value for key, value in current.items()):
    sys.exit(0)
digest = hashlib.sha256()
with (gzip.open if fasta.name.endswith(".gz") else open)(fasta, "rb") as handle:
    for block in iter(lambda: handle.read(1 << 22), b""):
        digest.update(block)
current["sha256"] = digest.hexdigest()
output = Path(os.environ["OUTPUT_DIR"])
has_results = any(path.name != ".state" for path in output.iterdir())
if previous and previous.get("sha256") != current["sha256"] and has_results:
    print(
        f"ERROR: {output} has the results of another FASTA file ({previous['path']}, "
        f"{previous['size']} bytes). Use a new output directory for {fasta}, or delete "
        f"{output} to start over.",
        file=sys.stderr,
    )
    sys.exit(1)
temporary = record.with_name(record.name + ".tmp")
temporary.write_text(json.dumps(current) + "\n")
temporary.replace(record)
PYEOF

# =================================================================
# Step 1: Discover chromosomes from FASTA headers
# =================================================================
echo "================================================================="
echo "Discovering chromosomes from FASTA file..."
echo "================================================================="

# Sequences without bases have no genes and cannot be predicted, so they are left
# out. Each FASTA header is printed with 1 if its sequence has bases, else 0.
list_sequences() {
    awk '
        /^>/ {
            if (started) print seen "\t" id;
            sub(/^>/, "");
            id = $1;
            seen = 0;
            started = 1;
            next;
        }
        !seen && NF { seen = 1 }
        END { if (started) print seen "\t" id }
    '
}

EMPTY_IDS=""
if [[ "$TOP_N_CONTIGS" == "all" ]]; then
    if [[ "$INPUT_FILE" == *.gz ]]; then
        SEQUENCE_LIST=$(zcat "$INPUT_FILE" | list_sequences)
    else
        SEQUENCE_LIST=$(list_sequences < "$INPUT_FILE")
    fi
    CHROM_IDS=$(awk -F '\t' '$1 == 1 { print $2 }' <<< "$SEQUENCE_LIST")
    EMPTY_IDS=$(awk -F '\t' '$1 == 0 { print $2 }' <<< "$SEQUENCE_LIST")
else
    TOP_IDS=""
    if [[ "$INPUT_FILE" == *.gz ]]; then
        TOP_IDS=$(
            zcat "$INPUT_FILE" | awk '
                /^>/ {
                    if (id != "" && len > 0) print len "\t" id;
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
                    if (id != "" && len > 0) print len "\t" id;
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
                    if (id != "" && len > 0) print len "\t" id;
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
                    if (id != "" && len > 0) print len "\t" id;
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

EMPTY_COUNT=$(echo "$EMPTY_IDS" | sed '/^$/d' | wc -l)
if [[ "$EMPTY_COUNT" -gt 0 ]]; then
    echo "WARNING: Skipping $EMPTY_COUNT sequence(s) without bases: $(echo "$EMPTY_IDS" | head -n 5 | paste -sd ' ')$([[ "$EMPTY_COUNT" -gt 5 ]] && echo " ...")"
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
# Step 1.5: Batch-extract sequences for all chromosomes in one pass
# =================================================================
# extract_fasta.py's per-chromosome mode (used below as a fallback) re-parses
# the whole FASTA file from the start for every chromosome it's asked for —
# O(chromosome count x file size) overall on assemblies with many
# contigs/scaffolds. Do it once instead: collect every chromosome that still
# needs its sequences.zarr and hand them all to extract_fasta.py in a single
# pass over the file via --manifest. Each chromosome is still written
# atomically to its own independent path (see atomic_output_path in
# src/atomic_io.py), so this composes with per-chromosome resume exactly as
# before.
echo "================================================================="
echo "Extracting sequences for all chromosomes (single pass over FASTA)..."
echo "================================================================="
EXTRACT_MANIFEST="$BATCH_SIZE_STATE_DIR/extract_manifest.json"
# Large scaffold lists can exceed the OS environment/argument size limit.
export -n CHROM_IDS
CHROM_IDS_FILE="$BATCH_SIZE_STATE_DIR/chromosome_ids.txt"
printf '%s\n' "$CHROM_IDS" > "$CHROM_IDS_FILE"
# Finished sequences are not predicted again in the first pass. Hybrid decoding, which reads
# the prediction files of every sequence, predicts again the ones that were deleted when it is
# about to run (see before the hybrid step).
NEEDS_LOGITS=0
export NEEDS_LOGITS

extract_needed_sequences() {
EXTRACT_MANIFEST_COUNT=$(OUTPUT_DIR="$OUTPUT_DIR" $PYTHON - "$EXTRACT_MANIFEST" "$CHROM_IDS_FILE" <<'PYEOF'
import json
import os
import sys

manifest_path = sys.argv[1]
output_dir = os.environ["OUTPUT_DIR"]
with open(sys.argv[2]) as fh:
    chrom_ids = [c.rstrip("\n") for c in fh if c.strip()]

entries = []
for chrom_id in chrom_ids:
    chrom_dir = os.path.join(output_dir, chrom_id)
    sequences_zarr = os.path.join(chrom_dir, f"sequences_{chrom_id}.zarr")
    filtered_gff = os.path.join(chrom_dir, f"predictions_filtered_{chrom_id}.gff")
    # Skip chromosomes that already have sequences.zarr, or are already
    # fully done (no need to re-extract sequences for those on resume).
    predictions_done = os.path.isfile(
        os.path.join(chrom_dir, f"predictions_{chrom_id}", "_SUCCESS.json")
    )
    finished = os.path.isfile(filtered_gff) and (
        predictions_done or os.environ.get("NEEDS_LOGITS") != "1"
    )
    if os.path.exists(sequences_zarr) or finished:
        continue
    entries.append({"chromosome_id": chrom_id, "output_zarr": sequences_zarr})

with open(manifest_path, "w") as fh:
    json.dump(entries, fh)

print(len(entries))
PYEOF
)

if [[ "$EXTRACT_MANIFEST_COUNT" -gt 0 ]]; then
    echo "Extracting $EXTRACT_MANIFEST_COUNT chromosome(s) needing sequences.zarr..."
    $PYTHON "$SCRIPT_DIR/scripts/extract_fasta.py" \
        --species-id "$SPECIES_ID" \
        --input-fasta "$INPUT_FILE" \
        --model-path "$TOKENIZER_PATH" \
        --manifest "$EXTRACT_MANIFEST"
else
    echo "All chromosomes already have sequences.zarr — nothing to extract."
fi
}

# The options each sequence's GFF depends on. A sequence decoded with other options is
# decoded again: its interval, raw and filtered files are removed here, so that the
# steps below rebuild them (and predict it again if its prediction files are gone).
DECODE_SETTINGS="mode=$MODE frame_aware=$FRAME_AWARE min_transcript_length=$MIN_TRANSCRIPT_LENGTH"
if [[ "$FRAME_AWARE" == "1" ]]; then
    DECODE_SETTINGS+=" min_intron_length=$MIN_INTRON_LENGTH min_coding_run_length=$MIN_CODING_RUN_LENGTH"
    DECODE_SETTINGS+=" exon_length_strictness=$EXON_LENGTH_STRICTNESS allow_u12_introns=$ALLOW_U12_INTRONS"
fi
export DECODE_SETTINGS
OUTPUT_DIR="$OUTPUT_DIR" $PYTHON - "$CHROM_IDS_FILE" <<'PYEOF'
import os
import shutil
import sys
from pathlib import Path

current = os.environ["DECODE_SETTINGS"]
output = Path(os.environ["OUTPUT_DIR"])
redone, adopted = [], []
for chrom in Path(sys.argv[1]).read_text().split():
    directory = output / chrom
    filtered = directory / f"predictions_filtered_{chrom}.gff"
    record = directory / "decoding_settings.txt"
    if not filtered.is_file():
        continue
    if not record.is_file():
        # Decoded by an earlier version, which kept no record: assume these options.
        temporary = record.with_name(record.name + ".tmp")
        temporary.write_text(current + "\n")
        temporary.replace(record)
        adopted.append(chrom)
        continue
    previous = record.read_text().strip()
    if previous == current:
        continue
    old = dict(item.split("=", 1) for item in previous.split())
    new = dict(item.split("=", 1) for item in current.split())
    keys = list(new) + [k for k in old if k not in new]
    key = next(k for k in keys if old.get(k) != new.get(k))
    redone.append(f"{chrom} ({key} {old.get(key, '-')} -> {new.get(key, '-')})")
    for path in (
        filtered,
        directory / f"predictions_raw_{chrom}.gff",
        directory / f"intervals_{chrom}.zarr",
        record,
    ):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
if redone:
    print(
        f"Decoding options changed for {len(redone)} sequence(s), e.g. {redone[0]}: "
        "they are decoded again."
    )
if adopted:
    print(
        f"{len(adopted)} sequence(s), e.g. {adopted[0]}, were decoded by an earlier "
        "version, which did not record its options; they are kept and taken to match "
        "the current options. To decode them again, delete their "
        "predictions_filtered_<ID>.gff."
    )
PYEOF

extract_needed_sequences
echo ""

# =================================================================
# Step 2: Per-chromosome pipeline (all 6 steps)
# =================================================================

# Removes only files with these exact names inside the output directory.
# clean_decoded_chromosome ID: the sequence and interval files, no longer used once
# the chromosome has its filtered GFF.
clean_decoded_chromosome() {
    local dir="$OUTPUT_DIR/$1"
    [[ -n "$OUTPUT_DIR" && -s "$dir/predictions_filtered_$1.gff" ]] || return 0
    rm -rf "$dir/sequences_$1.zarr" "$dir/intervals_$1.zarr"
}

# clean_predictions ID...: the prediction files, which hybrid decoding reads last.
# Nothing is removed unless the final GFF exists.
clean_predictions() {
    [[ -n "$OUTPUT_DIR" && -s "$FINAL_GFF" ]] || return 0
    local id
    for id in "$@"; do
        rm -rf "$OUTPUT_DIR/$id/predictions_$id" "$OUTPUT_DIR/$id/predictions_$id.lock"
        rm -rf "$OUTPUT_DIR/$id/sequences_$id.zarr" "$OUTPUT_DIR/$id/intervals_$id.zarr"
    done
}

process_chromosome() {
    local CHR_ID="$1"
    local GPU_ID="$3"   # which GPU this chromosome runs on
    local LOG_PREFIX="[${CHR_ID}@GPU${GPU_ID}]"

    local CHR_OUTPUT_DIR="${OUTPUT_DIR}/${CHR_ID}"

    # Files to be produced
    local SEQUENCES_ZARR="$CHR_OUTPUT_DIR/sequences_$CHR_ID.zarr"
    local PREDICTIONS_DIR="$CHR_OUTPUT_DIR/predictions_$CHR_ID"
    local INTERVALS_ZARR="$CHR_OUTPUT_DIR/intervals_$CHR_ID.zarr"
    local RAW_GENECAD_GFF="$CHR_OUTPUT_DIR/predictions_raw_$CHR_ID.gff"
    local FILTERED_GENECAD_GFF="$CHR_OUTPUT_DIR/predictions_filtered_$CHR_ID.gff"


    if [[ -f $FILTERED_GENECAD_GFF ]]; then
        echo "${LOG_PREFIX} Already complete — skipping (delete $CHR_OUTPUT_DIR to rerun)"
        return 0
    fi

    # --- Step 1: Extract Sequences ---
    if [[ -e $SEQUENCES_ZARR ]]; then
        echo "${LOG_PREFIX} [1/8] Skipping — sequences.zarr already exists"
    else
        echo "${LOG_PREFIX} [1/8] Extracting sequences..."
        $PYTHON "$SCRIPT_DIR/scripts/extract_fasta.py" \
            --species-id "$SPECIES_ID" \
            --input-fasta "$INPUT_FILE" \
            --chrom-map "${CHR_ID}:${CHR_ID}" \
            --model-path "$TOKENIZER_PATH" \
            --output-zarr "$SEQUENCES_ZARR" || return $?
    fi

    # Prediction workers finish before these CPU stages run.
    # Check the completion marker, not just the output directory.
    local gpu_id="$GPU_ID"
    if [[ ! -f "$PREDICTIONS_DIR/_SUCCESS.json" ]]; then
        echo "${LOG_PREFIX} ERROR: Prediction has not completed; refusing downstream processing."
        return 1
    fi

    # --- Step 3: Detect Intervals ---
    if [[ -e $INTERVALS_ZARR ]]; then
        echo "${LOG_PREFIX} [3/8] Skipping — intervals.zarr already exists"
    else
        echo "${LOG_PREFIX} [3/8] Detecting intervals (Viterbi decoding)..."
        $PYTHON "$SCRIPT_DIR/scripts/detect_intervals.py" \
            --input-dir "$PREDICTIONS_DIR" \
            --output-zarr "$INTERVALS_ZARR" \
            --domain "$MODE" \
            "${FRAME_AWARE_ARGS[@]}" || return $?
    fi

    # --- Step 4: Export Raw GFF ---
    if [[ -f $RAW_GENECAD_GFF ]]; then
        echo "${LOG_PREFIX} [4/8] Skipping — predictions_raw.gff already exists"
    else
        echo "${LOG_PREFIX} [4/8] Exporting raw GFF..."
        local export_tqdm_args=()
        if [[ -n "$gpu_id" ]]; then
            export_tqdm_args=(--tqdm-position "$gpu_id")
        fi
        $PYTHON "$SCRIPT_DIR/scripts/export_gff.py" \
            --input-zarr "$INTERVALS_ZARR" \
            --output-gff "$RAW_GENECAD_GFF" \
            --min-transcript-length "$MIN_TRANSCRIPT_LENGTH" \
            --cpu-workers "$CPU_WORKERS" \
            "${export_tqdm_args[@]}" || return $?
    fi

    # --- Step 5: Post-processing Filters ---
    echo "${LOG_PREFIX} [5/8] Filtering features..."

    if [[ -f $FILTERED_GENECAD_GFF ]]; then
        echo "${LOG_PREFIX}   Skipping feature-length filter — output already exists"
    else
        $PYTHON "$SCRIPT_DIR/scripts/filter_raw_gff.py" \
            --input-gff "$RAW_GENECAD_GFF" \
            --output-gff "$FILTERED_GENECAD_GFF" || return $?
    fi

    printf '%s\n' "$DECODE_SETTINGS" > "$CHR_OUTPUT_DIR/decoding_settings.txt.tmp"
    mv -f "$CHR_OUTPUT_DIR/decoding_settings.txt.tmp" "$CHR_OUTPUT_DIR/decoding_settings.txt"

    if [[ "$CLEAN_INTERMEDIATES" == "1" ]]; then
        clean_decoded_chromosome "$CHR_ID"
    fi
    echo "${LOG_PREFIX} Done!"
}

# process_chromosome_group GPU_ID ID...: steps 3-5 of process_chromosome for
# short sequences whose predictions are complete. Each step runs once for all the
# sequences that need it, since starting the three steps takes longer than decoding
# a short scaffold. If a step fails, the sequences it did not finish are processed one
# at a time, so that the error is reported for its own sequence and the others finish.
process_chromosome_group() {
    local GPU_ID="$1"
    shift
    local LOG_PREFIX="[$1 and $(( $# - 1 )) more@GPU${GPU_ID}]"
    local manifests="" status=0 failed=0 id
    echo "${LOG_PREFIX} Decoding $# short sequences together"
    manifests=$(mktemp -d "$BATCH_SIZE_STATE_DIR/decode_group.XXXXXX") || status=$?
    [[ $status -eq 0 ]] && OUTPUT_DIR="$OUTPUT_DIR" $PYTHON - "$manifests" "$@" <<'PYEOF' || status=$?
import json
import os
import sys
from pathlib import Path

output = Path(os.environ["OUTPUT_DIR"])
steps = {"intervals": [], "raw": [], "filtered": []}
for chrom in sys.argv[2:]:
    directory = output / chrom
    entry = {
        "chromosome_id": chrom,
        "predictions_dir": str(directory / f"predictions_{chrom}"),
        "intervals_zarr": str(directory / f"intervals_{chrom}.zarr"),
        "raw_gff": str(directory / f"predictions_raw_{chrom}.gff"),
        "filtered_gff": str(directory / f"predictions_filtered_{chrom}.gff"),
    }
    # The same checks as process_chromosome makes before each step.
    if not os.path.exists(entry["intervals_zarr"]):
        steps["intervals"].append(entry)
    if not os.path.isfile(entry["raw_gff"]):
        steps["raw"].append(entry)
    if not os.path.isfile(entry["filtered_gff"]):
        steps["filtered"].append(entry)
for step, entries in steps.items():
    if entries:
        (Path(sys.argv[1]) / f"{step}.json").write_text(json.dumps(entries))
PYEOF
    if [[ $status -eq 0 && -f "$manifests/intervals.json" ]]; then
        echo "${LOG_PREFIX} [3/8] Detecting intervals (Viterbi decoding)..."
        $PYTHON "$SCRIPT_DIR/scripts/detect_intervals.py" \
            --manifest "$manifests/intervals.json" \
            --domain "$MODE" \
            "${FRAME_AWARE_ARGS[@]}" || status=$?
    fi
    if [[ $status -eq 0 && -f "$manifests/raw.json" ]]; then
        echo "${LOG_PREFIX} [4/8] Exporting raw GFF..."
        local export_tqdm_args=()
        if [[ -n "$GPU_ID" ]]; then
            export_tqdm_args=(--tqdm-position "$GPU_ID")
        fi
        $PYTHON "$SCRIPT_DIR/scripts/export_gff.py" \
            --manifest "$manifests/raw.json" \
            --min-transcript-length "$MIN_TRANSCRIPT_LENGTH" \
            --cpu-workers "$CPU_WORKERS" \
            "${export_tqdm_args[@]}" || status=$?
    fi
    if [[ $status -eq 0 && -f "$manifests/filtered.json" ]]; then
        echo "${LOG_PREFIX} [5/8] Filtering features..."
        $PYTHON "$SCRIPT_DIR/scripts/filter_raw_gff.py" \
            --manifest "$manifests/filtered.json" || status=$?
    fi
    [[ -n "$manifests" ]] && rm -rf "$manifests"

    # A sequence with a filtered GFF now was decoded here, with these options.
    for id in "$@"; do
        [[ -f "$OUTPUT_DIR/$id/predictions_filtered_$id.gff" ]] || continue
        printf '%s\n' "$DECODE_SETTINGS" > "$OUTPUT_DIR/$id/decoding_settings.txt.tmp"
        mv -f "$OUTPUT_DIR/$id/decoding_settings.txt.tmp" "$OUTPUT_DIR/$id/decoding_settings.txt"
        if [[ "$CLEAN_INTERMEDIATES" == "1" ]]; then
            clean_decoded_chromosome "$id"
        fi
    done
    if [[ $status -ne 0 ]]; then
        echo "${LOG_PREFIX} A step failed (exit status $status); decoding the unfinished sequences one at a time."
        for id in "$@"; do
            [[ -f "$OUTPUT_DIR/$id/predictions_filtered_$id.gff" ]] && continue
            if ! process_chromosome "$id" "" "$GPU_ID"; then
                echo "$id" >> "$DECODE_FAILURES"
                failed=1
            fi
        done
        return "$failed"
    fi
    echo "${LOG_PREFIX} Done!"
}

# Run one persistent model process per GPU (one distributed process group in DDP).
run_prediction_manifest() {
    local key="$1" bs="$2" manifest="$3"
    local args=(
        "$SCRIPT_DIR/scripts/predict.py"
        --manifest "$manifest"
        --model-path "$BASE_MODEL"
        --model-checkpoint "$HEAD_MODEL"
        --species-id "$SPECIES_ID"
        --batch-size "$bs"
        --batch-size-cache "$BATCH_SIZE_STATE_DIR/batch_size_${key}.txt"
        --dtype "$DTYPE" --window-size 8192 --stride 4096
    )
    echo "[worker $key] Loading model once for scaffold manifest $manifest"
    if [[ "$PREDICT_MODE" == "single" ]]; then
        CUDA_VISIBLE_DEVICES="$key" $PY_LAUNCHER "${args[@]}" --tqdm-position "$key"
    elif [[ "$PREDICT_MODE" == "ddp" ]]; then
        CUDA_VISIBLE_DEVICES="$GPU_LIST_STR" $PY_LAUNCHER "${args[@]}"
    else
        $PY_LAUNCHER "${args[@]}"
    fi
}

run_prediction_workers() {
    local worker_keys="$GPU_LIST_STR"
    [[ "$PREDICT_MODE" != "single" ]] && worker_keys="ddp"
    local active_workers
    local chromosome_file="$BATCH_SIZE_STATE_DIR/chromosome_ids.txt"
    export -n CHROM_IDS
    printf '%s\n' "$CHROM_IDS" > "$chromosome_file" || return $?
    active_workers=$(OUTPUT_DIR="$OUTPUT_DIR" \
        WORKER_KEYS="$worker_keys" STATE_DIR="$BATCH_SIZE_STATE_DIR" \
        $PYTHON - "$chromosome_file" <<'PYEOF'
import json
import os
import sys
from pathlib import Path

keys = os.environ["WORKER_KEYS"].split(",")
work = {key: [] for key in keys}
output = Path(os.environ["OUTPUT_DIR"])
index = 0
for chrom in Path(sys.argv[1]).read_text().splitlines():
    if not chrom.strip():
        continue
    directory = output / chrom
    predictions_done = (directory / f"predictions_{chrom}" / "_SUCCESS.json").is_file()
    if (directory / f"predictions_filtered_{chrom}.gff").is_file() and (
        predictions_done or os.environ.get("NEEDS_LOGITS") != "1"
    ):
        continue
    work[keys[index % len(keys)]].append({
        "chromosome_id": chrom,
        "sequence_zarr": str(directory / f"sequences_{chrom}.zarr"),
        "predictions_dir": str(directory / f"predictions_{chrom}"),
    })
    index += 1
for key, entries in work.items():
    path = Path(os.environ["STATE_DIR"]) / f"predict_manifest_{key}.json"
    path.write_text(json.dumps(entries))
    if entries:
        print(key)
PYEOF
    ) || return $?
    if [[ -z "$active_workers" ]]; then
        echo "All scaffold outputs are complete; no model workers needed."
        return 0
    fi
    local key bs
    local failed=0
    local pids=()
    while IFS= read -r key; do
        if [[ "$key" == "ddp" ]]; then
            bs="$DDP_BATCH"
            run_prediction_manifest "$key" "$bs" "$BATCH_SIZE_STATE_DIR/predict_manifest_${key}.json" || failed=1
        else
            bs="${GPU_BATCH_SIZES[$key]}"
            run_prediction_manifest "$key" "$bs" "$BATCH_SIZE_STATE_DIR/predict_manifest_${key}.json" &
            pids+=($!)
        fi
    done <<< "$active_workers"
    local pid
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    return "$failed"
}

if [[ "$FRAME_AWARE" == "1" ]]; then
    FRAME_AWARE_ARGS=(--input-fasta "$INPUT_FILE" --min-intron-length "$MIN_INTRON_LENGTH" --min-coding-run-length "$MIN_CODING_RUN_LENGTH" --exon-length-strictness "$EXON_LENGTH_STRICTNESS")
    if [[ "$ALLOW_U12_INTRONS" == "1" ]]; then
        FRAME_AWARE_ARGS+=(--allow-u12-introns)
    fi
else
    FRAME_AWARE_ARGS=()
fi

export -f process_chromosome clean_decoded_chromosome
export CLEAN_INTERMEDIATES OUTPUT_DIR SPECIES_ID BASE_MODEL HEAD_MODEL TOKENIZER_PATH DTYPE PYTHON PYTHONPATH
export GPU_LIST_STR NUM_GPUS BATCH_SIZE_STATE_DIR

CHR_ARRAY=()
while IFS= read -r chr; do
    CHR_ARRAY+=("$chr")
done <<< "$CHROM_IDS"

# =================================================================
# How many chromosomes may be decoded and exported at once
#
# Frame-aware decoding holds a chromosome's whole prediction in memory: about
# 0.35 GB of RAM per Mb (measured on maize NAM: 0.31 GB/Mb). Plain and hybrid
# decoding read the prediction segment by segment and need about 0.035 GB/Mb
# (48 GB for the 1.37 Gb chromosome 5 of Vicia faba); 0.05 is assumed. One
# chromosome per GPU at once can overflow a node for genomes with several large
# chromosomes, so by default only as many run as fit.
# =================================================================

# resolve_parallel_chromosomes REQUESTED MAX_AUTO LARGEST_BP AVAILABLE_KB [GB_PER_MB]
# Prints how many chromosomes to run at once: REQUESTED if it is a number. If it is
# auto, as many as fit in AVAILABLE_KB of RAM, but no more than MAX_AUTO.
resolve_parallel_chromosomes() {
    if [[ "$1" != "auto" ]]; then
        echo "$1"
        return
    fi
    awk -v max_auto="$2" -v bp="$3" -v kb="$4" -v per_mb="${5:-0.35}" 'BEGIN {
        need = bp / 1e6 * per_mb * 1048576
        n = (need > 0) ? int(kb * 0.9 / need) : max_auto
        if (n > max_auto) n = max_auto
        if (n < 1) n = 1
        print n
    }'
}

# Length of the longest sequence in the input FASTA, in bp.
largest_sequence_bp() {
    if [[ -f "$INPUT_FILE.fai" ]]; then
        cut -f2 "$INPUT_FILE.fai" | sort -n | tail -1
        return
    fi
    local cat_cmd=cat
    [[ "$INPUT_FILE" == *.gz ]] && cat_cmd=zcat
    $cat_cmd "$INPUT_FILE" | awk '/^>/ { if (n > max) max = n; n = 0; next }
        { n += length($0) } END { if (n > max) max = n; print max + 0 }'
}

# The smallest value of a cgroup limit found in this process's cgroup and its parents.
# Works with cgroup v2 (memory.max, cpu.max) and v1 (memory.limit_in_bytes, cpu.cfs_*).
# Prints nothing when no limit is set. CGROUP_ROOT and CGROUP_FILE exist for the tests.
cgroup_limit() {
    local what="$1"   # memory (kB) or cpu (cores, rounded up)
    local root="${CGROUP_ROOT:-/sys/fs/cgroup}" file="${CGROUP_FILE:-/proc/self/cgroup}"
    local dir rel value period best=""
    rel=$(awk -F: '$1 == "0" { print $3; exit }' "$file" 2>/dev/null || true)
    if [[ -f "$root/cgroup.controllers" || -n "$rel" ]]; then   # cgroup v2
        dir="$root$rel"
        while [[ "$dir" == "$root"* ]]; do
            if [[ "$what" == "memory" ]]; then
                value=$(cat "$dir/memory.max" 2>/dev/null || true)
                [[ "$value" =~ ^[0-9]+$ ]] && value=$(( value / 1024 )) || value=""
            else
                value="" period=""
                if [[ -r "$dir/cpu.max" ]]; then read -r value period < "$dir/cpu.max" || true; fi
                [[ "$value" =~ ^[0-9]+$ && "$period" =~ ^[0-9]+$ && "$period" -gt 0 ]] \
                    && value=$(( (value + period - 1) / period )) || value=""
            fi
            [[ -n "$value" ]] && { [[ -z "$best" || "$value" -lt "$best" ]] && best="$value"; }
            [[ "$dir" == "$root" ]] && break
            dir="${dir%/*}"
        done
    else   # cgroup v1: each controller has its own tree
        local controller=memory
        [[ "$what" == "cpu" ]] && controller=cpu
        rel=$(awk -F: -v c="$controller" '$2 ~ "(^|,)" c "(,|$)" { print $3; exit }' "$file" 2>/dev/null || true)
        dir="$root/$controller$rel"
        while [[ "$dir" == "$root/$controller"* ]]; do
            if [[ "$what" == "memory" ]]; then
                value=$(cat "$dir/memory.limit_in_bytes" 2>/dev/null || true)
                [[ "$value" =~ ^[0-9]+$ ]] && value=$(( value / 1024 )) || value=""
            else
                value=$(cat "$dir/cpu.cfs_quota_us" 2>/dev/null || true)
                period=$(cat "$dir/cpu.cfs_period_us" 2>/dev/null || true)
                [[ "$value" =~ ^[0-9]+$ && "$period" =~ ^[0-9]+$ && "$period" -gt 0 ]] \
                    && value=$(( (value + period - 1) / period )) || value=""
            fi
            [[ -n "$value" ]] && { [[ -z "$best" || "$value" -lt "$best" ]] && best="$value"; }
            [[ "$dir" == "$root/$controller" ]] && break
            dir="${dir%/*}"
        done
    fi
    echo "$best"
}

# CPU cores this job may use. nproc reads OMP_NUM_THREADS, which batch wrappers set to 1,
# so it is run with that unset; it still respects taskset and Slurm binding. A cgroup CPU
# quota (docker --cpus) is not visible to nproc, so it is applied here. env(1) is avoided
# on purpose: a container image may put a different env first in PATH.
available_cores() {
    local cores quota
    cores=$( (unset OMP_NUM_THREADS OMP_THREAD_LIMIT; nproc 2>/dev/null) \
        || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
    quota=$(cgroup_limit cpu)
    [[ "$quota" =~ ^[0-9]+$ ]] && (( quota >= 1 && quota < cores )) && cores=$quota
    echo "$cores"
}

# Memory this job may use, in kB: MemAvailable, capped by the cgroup and Slurm limits.
# Prints 0 when it cannot be read, which makes auto parallelism fall back to one at a time.
available_memory_kb() {
    local kb limit
    kb=$(awk '/^MemAvailable:/ { print $2 }' "${MEMINFO_FILE:-/proc/meminfo}" 2>/dev/null || true)
    [[ "$kb" =~ ^[0-9]+$ ]] || kb=0
    limit=$(cgroup_limit memory)
    if [[ "$limit" =~ ^[0-9]+$ ]] && (( limit < kb )); then
        kb=$limit
    fi
    if [[ "${SLURM_MEM_PER_NODE:-}" =~ ^[0-9]+$ ]] && (( SLURM_MEM_PER_NODE * 1024 < kb )); then
        kb=$(( SLURM_MEM_PER_NODE * 1024 ))
    elif [[ "${SLURM_MEM_PER_CPU:-}" =~ ^[0-9]+$ && "${SLURM_CPUS_ON_NODE:-}" =~ ^[0-9]+$ ]] \
        && (( SLURM_MEM_PER_CPU * SLURM_CPUS_ON_NODE * 1024 < kb )); then
        kb=$(( SLURM_MEM_PER_CPU * SLURM_CPUS_ON_NODE * 1024 ))
    fi
    echo "$kb"
}

if [[ "$DECODER" == "frame-aware" ]]; then
    DECODE_GB_PER_MB="0.35"
else
    DECODE_GB_PER_MB="0.05"
fi

if [[ "$MAX_PARALLEL_CHROMOSOMES" == "auto" ]]; then
    LARGEST_BP=$(largest_sequence_bp)
    AVAILABLE_KB=$(available_memory_kb)
    # Decoding is single-threaded, so allow one chromosome per free core (each export
    # also starts CPU_WORKERS processes), but never fewer than one per GPU or more than 16.
    MAX_AUTO=$(( $(available_cores) / CPU_WORKERS ))
    (( MAX_AUTO > 16 )) && MAX_AUTO=16
    (( MAX_AUTO < NUM_GPUS )) && MAX_AUTO=$NUM_GPUS
    PARALLEL_CHROMOSOMES=$(resolve_parallel_chromosomes auto "$MAX_AUTO" "$LARGEST_BP" "$AVAILABLE_KB" "$DECODE_GB_PER_MB")
    echo "Decoding up to $PARALLEL_CHROMOSOMES chromosome(s) at once" \
        "(longest sequence $(( LARGEST_BP / 1000000 )) Mb, $(( AVAILABLE_KB / 1048576 )) GB RAM available, CPU allows $MAX_AUTO)"
    if [[ "$AVAILABLE_KB" -eq 0 ]]; then
        echo "WARNING: could not read the available memory; decoding one chromosome at a time."
    elif awk -v bp="$LARGEST_BP" -v kb="$AVAILABLE_KB" -v per_mb="$DECODE_GB_PER_MB" \
        'BEGIN { exit !(bp / 1e6 * per_mb * 1048576 > kb) }'; then
        echo "WARNING: decoding the longest sequence may need more RAM than is available."
    fi
else
    PARALLEL_CHROMOSOMES=$(resolve_parallel_chromosomes "$MAX_PARALLEL_CHROMOSOMES" 1 0 0)
fi

# =================================================================
# Choose dispatch strategy
#
#   DDP  (torchrun)  — when chromosomes < GPUs:
#     All GPUs collaborate on each chromosome; processed sequentially.
#     Ensures every GPU is busy even for tiny genomes.
#
#   Per-GPU parallel — when chromosomes >= GPUs:
#     Each GPU owns its chromosomes independently. The CPU stages run up to
#     PARALLEL_CHROMOSOMES chromosomes at the same time. Avoids 1000× torchrun spawns.
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

if ! run_prediction_workers; then
    echo "ERROR: A prediction worker failed. Completed segments are retained; rerun to resume."
    exit 1
fi

# Short sequences are decoded in groups (process_chromosome_group): one line of
# decode_jobs.txt is one chromosome, or a group's IDs separated by spaces. Short
# sequences are spread over at least as many groups as run at once, and a group
# holds about GROUP_SEQUENCE_BP bases, longest first into the group with the fewest.
SHORT_SEQUENCE_BP=1000000
GROUP_SEQUENCE_BP=20000000
DECODE_JOBS_FILE="$BATCH_SIZE_STATE_DIR/decode_jobs.txt"
OUTPUT_DIR="$OUTPUT_DIR" SHORT_SEQUENCE_BP="$SHORT_SEQUENCE_BP" \
    GROUP_SEQUENCE_BP="$GROUP_SEQUENCE_BP" PARALLEL_CHROMOSOMES="$PARALLEL_CHROMOSOMES" \
    $PYTHON - "$CHROM_IDS_FILE" > "$DECODE_JOBS_FILE" <<'PYEOF' || exit $?
import heapq
import json
import math
import os
import sys
from pathlib import Path

output = Path(os.environ["OUTPUT_DIR"])
with open(sys.argv[1], newline="") as handle:
    ids = handle.read().split("\n")
if ids[-1] == "":
    ids.pop()
jobs, short = [], []
for chrom in ids:
    directory = output / chrom
    predictions = directory / f"predictions_{chrom}"
    length = None
    if (
        chrom
        and not (directory / f"predictions_filtered_{chrom}.gff").is_file()
        and (predictions / "_SUCCESS.json").is_file()
        and (directory / f"sequences_{chrom}.zarr").exists()
    ):
        try:
            length = int(json.loads((predictions / "run.json").read_text())["length"])
        except (OSError, ValueError, KeyError, TypeError):
            length = None
    if length is not None and length < int(os.environ["SHORT_SEQUENCE_BP"]):
        short.append((chrom, length))
    else:
        jobs.append([chrom])
count = min(
    len(short),
    max(
        int(os.environ["PARALLEL_CHROMOSOMES"]),
        math.ceil(sum(n for _, n in short) / int(os.environ["GROUP_SEQUENCE_BP"])),
    ),
)
groups = [[] for _ in range(count)]
heap = [(0, i) for i in range(count)]
for position, (chrom, n) in sorted(enumerate(short), key=lambda item: -item[1][1]):
    load, i = heapq.heappop(heap)
    groups[i].append((position, chrom))
    heapq.heappush(heap, (load + n, i))
# Groups start in the order of their first sequence in the FASTA file.
jobs += [[chrom for _, chrom in group] for group in sorted(sorted(g) for g in groups)]
for job in jobs:
    print(" ".join(job))
PYEOF
DECODE_JOBS=()
while IFS= read -r job; do
    DECODE_JOBS+=("$job")
done < "$DECODE_JOBS_FILE"

# Sequences that failed, one per line; a group adds each of its own.
DECODE_FAILURES="$BATCH_SIZE_STATE_DIR/decode_failures.txt"
: > "$DECODE_FAILURES"

# start_decode_job JOB BATCH GPU_ID: one line of decode_jobs.txt.
start_decode_job() {
    local members
    read -ra members <<< "$1"
    if [[ ${#members[@]} -gt 1 ]]; then
        process_chromosome_group "$3" "${members[@]}"
    else
        process_chromosome "$1" "$2" "$3"
    fi
}

# record_decode_failure JOB: a group has already listed its failed sequences.
record_decode_failure() {
    [[ "$1" == *" "* ]] || echo "$1" >> "$DECODE_FAILURES"
}

FAILED=0

if [[ "$PREDICT_MODE" == "ddp" || "$PREDICT_MODE" == "ddp_slurm" ]]; then
    # Predictions are finished; run the CPU stages for each chromosome.
    for JOB in "${DECODE_JOBS[@]}"; do
        if ! start_decode_job "$JOB" "$DDP_BATCH" ""; then
            FAILED=$(( FAILED + 1 ))
            record_decode_failure "$JOB"
        fi
    done
else
    # Per-GPU parallel — round-robin, at most PARALLEL_CHROMOSOMES concurrent jobs
    declare -a PIDS=() PID_JOBS=()
    chr_idx=0
    for JOB in "${DECODE_JOBS[@]}"; do
        gpu_id="${GPU_ARRAY[$(( chr_idx % NUM_GPUS ))]}"
        bs="${GPU_BATCH_SIZES[$gpu_id]}"

        # Wait for the oldest slot before launching, keeping at most PARALLEL_CHROMOSOMES live jobs
        if [[ ${#PIDS[@]} -ge $PARALLEL_CHROMOSOMES ]]; then
            if ! wait "${PIDS[0]}"; then
                FAILED=$(( FAILED + 1 ))
                record_decode_failure "${PID_JOBS[0]}"
            fi
            PIDS=("${PIDS[@]:1}")
            PID_JOBS=("${PID_JOBS[@]:1}")
        fi

        start_decode_job "$JOB" "$bs" "$gpu_id" &
        PIDS+=($!)
        PID_JOBS+=("$JOB")
        chr_idx=$(( chr_idx + 1 ))
    done
    for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
            FAILED=$(( FAILED + 1 ))
            record_decode_failure "${PID_JOBS[$i]}"
        fi
    done
fi

if [[ $FAILED -gt 0 ]]; then
    # A failed group counts once in FAILED, but each of its failed sequences is listed.
    FAILED_SEQUENCES=$(sort -u "$DECODE_FAILURES" | wc -l)
    (( FAILED_SEQUENCES > FAILED )) && FAILED=$FAILED_SEQUENCES
    echo "ERROR: $FAILED chromosome(s) failed. See output above for details."
    exit 1
fi

RECALL_GFFS=()
for CHR_ID in "${CHR_ARRAY[@]}"; do
    RECALL_GFFS=("${RECALL_GFFS[@]}" "${OUTPUT_DIR}/${CHR_ID}/predictions_filtered_$CHR_ID.gff")
done

# A finished stage is skipped, but only while the files it was built from are unchanged.
# refresh_stage OUTPUT INPUT...: when OUTPUT was built from other inputs it is moved aside
# (OUTPUT.stale, removed once the run succeeds) so that it is built again.
refresh_stage() {
    local output="$1" reason status=0
    shift
    [[ -f "$output" ]] || return 0
    reason=$($PYTHON "$SCRIPT_DIR/scripts/stage_inputs.py" check "$output" "$@") || status=$?
    if [[ $status -eq 10 ]]; then
        echo "Rebuilding $(basename "$output"): $reason"
        mv -f "$output" "$output.stale"
        rm -f "$output.inputs.json"
    elif [[ $status -ne 0 ]]; then
        return "$status"
    fi
}

# record_stage OUTPUT INPUT...: remember which inputs OUTPUT was built from.
record_stage() {
    $PYTHON "$SCRIPT_DIR/scripts/stage_inputs.py" record "$@"
}

# =================================================================
# Merge all per-chromosome GFFs into single files
# =================================================================
echo ""
echo "================================================================="
echo "[6/8] Merging per-chromosome GFFs into single files..."
echo "================================================================="

# The final annotation is the only GFF at the top of OUTPUT_DIR; the files it is
# built from are kept in intermediate/ for troubleshooting.
INTERMEDIATE_DIR="$OUTPUT_DIR/intermediate"
mkdir -p "$INTERMEDIATE_DIR"
RAW_GFF="$INTERMEDIATE_DIR/${SPECIES_ID}_GeneCAD_raw.gff"
ORF_GFF="$INTERMEDIATE_DIR/${SPECIES_ID}_GeneCAD_orf.gff"
FINAL_GFF="$OUTPUT_DIR/${SPECIES_ID}_GeneCAD_final.gff"

# Merge again every time and keep the existing file only when it is identical, so that
# a sequence that was processed again is never missing from the merged annotation, and
# the steps after it are redone only when something really changed.
$PYTHON "$MERGE_SCRIPT" \
    --output-gff "$RAW_GFF.new" \
    --input-gffs "${RECALL_GFFS[@]}"
if [[ -f "$RAW_GFF" ]] && cmp -s "$RAW_GFF.new" "$RAW_GFF"; then
    rm -f "$RAW_GFF.new"
    echo "${SPECIES_ID}_GeneCAD_raw.gff is up to date"
else
    [[ -f "$RAW_GFF" ]] && echo "Per-chromosome results changed: ${SPECIES_ID}_GeneCAD_raw.gff was rebuilt"
    mv -f "$RAW_GFF.new" "$RAW_GFF"
fi

echo ""
echo "================================================================="
echo "[7/8] Repairing CDS boundaries against the genome sequence..."
echo "================================================================="

# Hybrid decoding needs the partial transcripts to rescue them, and drops
# the unrescued ones itself.
KEEP_PARTIAL_ARGS=()
if [[ "$KEEP_PARTIAL" == "1" || "$DECODER" == "hybrid" ]]; then
    KEEP_PARTIAL_ARGS=(--keep-partial)
fi
# A stage is redone when one of the options it depends on changed.
ORF_SETTINGS=(--setting "max_shift=$ORF_MAX_SHIFT" --setting "keep_partial=${#KEEP_PARTIAL_ARGS[@]}")
[[ "$ORF_MAX_SHIFT" -eq 0 ]] || refresh_stage "$ORF_GFF" "$RAW_GFF" "${ORF_SETTINGS[@]}"
if [[ "$ORF_MAX_SHIFT" -eq 0 ]]; then
    echo "Skipping ORF repair — disabled via --orf-max-shift 0"
    ORF_GFF="$RAW_GFF"
elif [[ -f "$ORF_GFF" ]]; then
    echo "Skipping ORF repair — ${SPECIES_ID}_GeneCAD_orf.gff already exists"
    record_stage "$ORF_GFF" "$RAW_GFF" "${ORF_SETTINGS[@]}"
else
    $PYTHON "$SCRIPT_DIR/scripts/fix_orf.py" \
        --input-gff "$RAW_GFF" \
        --input-fasta "$INPUT_FILE" \
        --output-gff "$ORF_GFF" \
        --max-shift "$ORF_MAX_SHIFT" \
        --report "$INTERMEDIATE_DIR/${SPECIES_ID}_GeneCAD_orf_report.tsv" \
        "${KEEP_PARTIAL_ARGS[@]}"
    record_stage "$ORF_GFF" "$RAW_GFF" "${ORF_SETTINGS[@]}"
fi

REFINE_INPUT_GFF="$ORF_GFF"
if [[ "$DECODER" == "hybrid" ]]; then
    HYBRID_GFF="$INTERMEDIATE_DIR/${SPECIES_ID}_GeneCAD_hybrid.gff"
    echo ""
    echo "[7/8] Hybrid decoding: rescuing partial and merging split genes..."
    HYBRID_SETTINGS=(
        --setting "mode=$MODE" --setting "max_gap=$MERGE_MAX_GAP"
        --setting "keep_partial=$KEEP_PARTIAL"
        --setting "min_intron_length=$MIN_INTRON_LENGTH"
        --setting "min_coding_run_length=$MIN_CODING_RUN_LENGTH"
        --setting "exon_length_strictness=$EXON_LENGTH_STRICTNESS"
        --setting "allow_u12_introns=$ALLOW_U12_INTRONS"
        --setting "allow_missing_predictions=$ALLOW_MISSING_PREDICTIONS"
    )
    refresh_stage "$HYBRID_GFF" "$ORF_GFF" "${HYBRID_SETTINGS[@]}"
    if [[ -f "$HYBRID_GFF" ]]; then
        echo "Skipping hybrid decoding — ${SPECIES_ID}_GeneCAD_hybrid.gff already exists"
        record_stage "$HYBRID_GFF" "$ORF_GFF" "${HYBRID_SETTINGS[@]}"
    else
        # Hybrid decoding reads the prediction files of every sequence. Predict again
        # the ones that were deleted (this does nothing when all of them exist).
        predict_for_hybrid() {
            NEEDS_LOGITS=1
            export NEEDS_LOGITS
            extract_needed_sequences
            run_prediction_workers || {
                echo "ERROR: Could not predict the sequences whose prediction files are missing."
                exit 1
            }
        }
        if [[ "$ALLOW_MISSING_PREDICTIONS" != "1" ]]; then
            predict_for_hybrid
        fi
        HYBRID_ARGS=()
        if [[ "$KEEP_PARTIAL" == "1" ]]; then
            HYBRID_ARGS+=(--keep-partial)
        fi
        if [[ "$ALLOW_MISSING_PREDICTIONS" == "1" ]]; then
            HYBRID_ARGS+=(--allow-missing-predictions)
        fi
        if [[ "$ALLOW_U12_INTRONS" == "1" ]]; then
            HYBRID_ARGS+=(--allow-u12-introns)
        fi
        run_hybrid() {
            $PYTHON "$SCRIPT_DIR/scripts/hybrid_decode.py" \
                --input-gff "$ORF_GFF" \
                --input-fasta "$INPUT_FILE" \
                --predictions-root "$OUTPUT_DIR" \
                --output-gff "$HYBRID_GFF" \
                --domain "$MODE" \
                --max-gap "$MERGE_MAX_GAP" \
                --workers "$CPU_WORKERS" \
                --min-intron-length "$MIN_INTRON_LENGTH" \
                --min-coding-run-length "$MIN_CODING_RUN_LENGTH" \
                --exon-length-strictness "$EXON_LENGTH_STRICTNESS" \
                "${HYBRID_ARGS[@]}"
        }
        hybrid_status=0
        run_hybrid || hybrid_status=$?
        # Status 3: some prediction files failed verification. hybrid_decode.py removed
        # their completion markers, so the damaged windows are predicted again.
        if [[ "$hybrid_status" -eq 3 && "$ALLOW_MISSING_PREDICTIONS" != "1" ]]; then
            echo "Predicting the damaged prediction files again..."
            predict_for_hybrid
            hybrid_status=0
            run_hybrid || hybrid_status=$?
        fi
        if [[ "$hybrid_status" -ne 0 ]]; then
            echo "ERROR: Hybrid decoding failed (exit status $hybrid_status)."
            exit "$hybrid_status"
        fi
        record_stage "$HYBRID_GFF" "$ORF_GFF" "${HYBRID_SETTINGS[@]}"
    fi
    REFINE_INPUT_GFF="$HYBRID_GFF"
fi

echo ""
echo "================================================================="
echo "[8/8] Running protein refinement on merged predictions..."
echo "================================================================="

refresh_stage "$FINAL_GFF" "$REFINE_INPUT_GFF"
if [[ -f "$FINAL_GFF" ]]; then
    echo "Skipping refinement — ${SPECIES_ID}_GeneCAD_final.gff already exists"
else
    $PYTHON "$SCRIPT_DIR/scripts/refine.py" \
        --input-gff "$REFINE_INPUT_GFF" \
        --input-fasta "$INPUT_FILE" \
        --output-gff "$FINAL_GFF" \
        --gpus "$GPU_LIST_STR"
    record_stage "$FINAL_GFF" "$REFINE_INPUT_GFF"
fi
rm -f "$FINAL_GFF.stale" "$RAW_GFF.stale" "$ORF_GFF.stale" "${HYBRID_GFF:-$RAW_GFF}.stale"

if [[ "$CLEAN_INTERMEDIATES" == "1" ]]; then
    clean_predictions "${CHR_ARRAY[@]}"
    echo "Removed the sequence, interval and prediction files (--clean-intermediates)."
fi

echo ""
echo "================================================================="
echo "All done!"
echo ""
echo "Final annotation (use this file):"
echo "  $FINAL_GFF"
echo ""
echo "Intermediate files, for troubleshooting (a rerun reuses them, so keep them while you may rerun):"
echo "  $INTERMEDIATE_DIR/"
echo "================================================================="
