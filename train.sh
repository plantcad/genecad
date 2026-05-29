#!/bin/bash
# =============================================================================
# GeneCAD Fine-Tuning Pipeline
# Fine-tunes the GeneCAD classifier head on a multi-species plant dataset.
#
# Usage:
#   bash train.sh [OPTIONS]
#
# Options:
#   -m, --domain MODE       Training domain: plant or animal (default: plant)
#   -o, --output DIR        Output directory for checkpoints and logs
#                           (default: genecad_result/training/<domain>)
#   -r, --run-name NAME     WandB run name  (default: genecad-plant-multispecies)
#   -p, --project NAME      WandB project   (default: genecad)
#   -g, --gpus SPEC         GPU selection: number of GPUs, comma list, or 'all'
#                           Examples: 2, 0,1,3, all  (default: 1)
#   -b, --batch-size N      Per-GPU batch size (default: 4)
#   -e, --effective N       Effective batch size (default: 384)
#   -l, --lr RATE           Learning rate   (default: 2e-4)
#   --gff-parallel N        Parallel workers for Step 2 GFF extraction by species (default: 1)
#   --window-size N         Training window size for sample generation (default: 8192)
#   --intergenic-proportion P
#                           Intergenic window ratio in [0,1] for sample generation (default: 0.15)
#   --base-frozen yes|no    Freeze base encoder during training (default: domain preset)
#   --auto-class-weights yes|no
#                           Compute class weights from train set before training (default: yes)
#   --label-qc yes|no       Run split/label QC before training (default: yes)
#   --min-non-intergenic-ratio P
#                           Fail if train non-intergenic token ratio < P (default: 0.01)
#   --min-core-class-tokens N
#                           Fail if core classes have too few tokens (default: 100)
#   --base-model ID         Override base encoder model ID
#   --animal-input-dir DIR  Animal FASTA source dir (default: <output>/pipeline/data/animal/fasta/training)
#   --animal-gff-dir DIR    Animal GFF source dir (default: <output>/pipeline/data/animal/gff/training)
#   --hf-upload-repo ID     Upload artifacts to Hugging Face repo (optional)
#   --hf-upload-type TYPE   model or dataset repo type (default: model)
#   --hf-upload-path PATH   Path inside HF repo (default: run name)
#   -h, --help              Show this message
#
# Requirements:
#   - Linux, CUDA 12, uv (https://docs.astral.sh/uv/)
#   - huggingface-cli login  (to download training data)
#   - WandB account          (optional; disable with WANDB_MODE=disabled)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
cd "$SCRIPT_DIR"

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
  sed -n '3,36p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

DOMAIN="plant"
OUTPUT_BASE_DIR="genecad_result/training"
OUTPUT_DIR="$OUTPUT_BASE_DIR"
OUTPUT_DIR_USER_SET=0
RUN_NAME=""
PROJECT_NAME="genecad-emarro"
GPUS_ARG="1"
NUM_GPUS=1
CUDA_VISIBLE_DEVICES_TRAIN=""
BATCH_SIZE=4
TARGET_EFFECTIVE_BATCH=384
LEARNING_RATE=2e-4
EPOCHS_OVERRIDE=""
GFF_PARALLEL=1
WINDOW_SIZE=8192
INTERGENIC_PROPORTION=0.15
BASE_FROZEN_OVERRIDE=""
AUTO_CLASS_WEIGHTS="yes"
BASE_MODEL=""
LABEL_QC="yes"
MIN_NON_INTERGENIC_RATIO="0.01"
MIN_CORE_CLASS_TOKENS=100

ANIMAL_INPUT_DIR=""
ANIMAL_GFF_DIR=""

HF_UPLOAD_REPO=""
HF_UPLOAD_TYPE="model"
HF_UPLOAD_PATH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--domain)     DOMAIN="$2"; shift 2 ;;
    -o|--output)     OUTPUT_DIR="$2"; OUTPUT_DIR_USER_SET=1; shift 2 ;;
    -r|--run-name)   RUN_NAME="$2";      shift 2 ;;
    -p|--project)    PROJECT_NAME="$2";  shift 2 ;;
    -g|--gpus)       GPUS_ARG="$2";      shift 2 ;;
    -b|--batch-size) BATCH_SIZE="$2";    shift 2 ;;
    -e|--effective)  TARGET_EFFECTIVE_BATCH="$2"; shift 2 ;;
    -l|--lr)         LEARNING_RATE="$2"; shift 2 ;;
    --epochs)        EPOCHS_OVERRIDE="$2"; shift 2 ;;
    --gff-parallel)  GFF_PARALLEL="$2"; shift 2 ;;
    --window-size)   WINDOW_SIZE="$2"; shift 2 ;;
    --intergenic-proportion) INTERGENIC_PROPORTION="$2"; shift 2 ;;
    --base-frozen)   BASE_FROZEN_OVERRIDE="$2"; shift 2 ;;
    --auto-class-weights) AUTO_CLASS_WEIGHTS="$2"; shift 2 ;;
    --base-model)    BASE_MODEL="$2"; shift 2 ;;
    --label-qc)      LABEL_QC="$2"; shift 2 ;;
    --min-non-intergenic-ratio) MIN_NON_INTERGENIC_RATIO="$2"; shift 2 ;;
    --min-core-class-tokens) MIN_CORE_CLASS_TOKENS="$2"; shift 2 ;;
    --animal-input-dir) ANIMAL_INPUT_DIR="$2"; shift 2 ;;
    --animal-gff-dir) ANIMAL_GFF_DIR="$2"; shift 2 ;;
    --hf-upload-repo) HF_UPLOAD_REPO="$2"; shift 2 ;;
    --hf-upload-type) HF_UPLOAD_TYPE="$2"; shift 2 ;;
    --hf-upload-path) HF_UPLOAD_PATH="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

case "$DOMAIN" in
  plant|animal) ;;
  *)
    echo "ERROR: --domain must be 'plant' or 'animal'"
    exit 1
    ;;
esac

if [[ "$OUTPUT_DIR_USER_SET" -eq 0 ]]; then
  OUTPUT_DIR="$OUTPUT_BASE_DIR/$DOMAIN"
fi

if [[ "$DOMAIN" == "plant" ]]; then
  BASE_MODEL="${BASE_MODEL:-emarro/pcad2-200M-cnet-baseline}"
  SPECIES_IDS="Athaliana Osativa Gmax Hvulgare Ptrichocarpa"
  EPOCHS=1
  ARCHITECTURE="all"
  HEAD_LAYERS=8
  TOKEN_EMBED_DIM=256
  BASE_FROZEN="no"
  NUM_WORKERS=8
  VALID_PROPORTION=0.05
  RUN_NAME="${RUN_NAME:-genecad-plant-multispecies}"
else
  BASE_MODEL="${BASE_MODEL:-emarro/pcad2_vert_small}"
  SPECIES_IDS="Drerio Ggallus Hsapiens Xtropicalis Mmusculus"
  EPOCHS=1
  ARCHITECTURE="all"
  HEAD_LAYERS=8
  TOKEN_EMBED_DIM=256  # Increased from 128 to match plant capacity (animal gene structures are more complex)
  BASE_FROZEN="no"
  NUM_WORKERS=4
  VALID_PROPORTION=0.02
  RUN_NAME="${RUN_NAME:-genecad-animal-multispecies}"
fi

if [[ -z "$HF_UPLOAD_PATH" ]]; then
  HF_UPLOAD_PATH="$RUN_NAME"
fi

if [[ "$HF_UPLOAD_TYPE" != "model" && "$HF_UPLOAD_TYPE" != "dataset" ]]; then
  echo "ERROR: --hf-upload-type must be 'model' or 'dataset'"
  exit 1
fi

if ! [[ "$GFF_PARALLEL" =~ ^[0-9]+$ ]] || [[ "$GFF_PARALLEL" -lt 1 ]]; then
  echo "ERROR: --gff-parallel must be a positive integer"
  exit 1
fi

if ! [[ "$WINDOW_SIZE" =~ ^[0-9]+$ ]] || [[ "$WINDOW_SIZE" -lt 1 ]]; then
  echo "ERROR: --window-size must be a positive integer"
  exit 1
fi

if ! [[ "$INTERGENIC_PROPORTION" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]]; then
  echo "ERROR: --intergenic-proportion must be in [0,1]"
  exit 1
fi

if [[ -n "$BASE_FROZEN_OVERRIDE" && "$BASE_FROZEN_OVERRIDE" != "yes" && "$BASE_FROZEN_OVERRIDE" != "no" ]]; then
  echo "ERROR: --base-frozen must be 'yes' or 'no'"
  exit 1
fi

if [[ "$AUTO_CLASS_WEIGHTS" != "yes" && "$AUTO_CLASS_WEIGHTS" != "no" ]]; then
  echo "ERROR: --auto-class-weights must be 'yes' or 'no'"
  exit 1
fi

if [[ "$LABEL_QC" != "yes" && "$LABEL_QC" != "no" ]]; then
  echo "ERROR: --label-qc must be 'yes' or 'no'"
  exit 1
fi

if ! [[ "$MIN_NON_INTERGENIC_RATIO" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]]; then
  echo "ERROR: --min-non-intergenic-ratio must be in [0,1]"
  exit 1
fi

if ! [[ "$MIN_CORE_CLASS_TOKENS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --min-core-class-tokens must be a non-negative integer"
  exit 1
fi

if [[ -n "$EPOCHS_OVERRIDE" ]] && ( ! [[ "$EPOCHS_OVERRIDE" =~ ^[0-9]+$ ]] || [[ "$EPOCHS_OVERRIDE" -lt 1 ]] ); then
  echo "ERROR: --epochs must be a positive integer"
  exit 1
fi

if [[ -n "$BASE_FROZEN_OVERRIDE" ]]; then
  BASE_FROZEN="$BASE_FROZEN_OVERRIDE"
fi

if [[ -n "$EPOCHS_OVERRIDE" ]]; then
  EPOCHS="$EPOCHS_OVERRIDE"
fi

# Resolve GPU selection.
# Accepted forms:
#   - integer N: use first N visible GPUs (backward-compatible behavior)
#   - comma list: use exact GPU IDs via CUDA_VISIBLE_DEVICES
#   - all: detect all available GPU IDs via nvidia-smi
GPUS_ARG="${GPUS_ARG//[[:space:]]/}"
if [[ "$GPUS_ARG" == "all" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: --gpus all requires nvidia-smi to detect GPU IDs."
    echo "       Use an explicit list (e.g. --gpus 0,1) or count (e.g. --gpus 2)."
    exit 1
  fi
  CUDA_VISIBLE_DEVICES_TRAIN=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr '\n' ',' | sed 's/,$//')
  if [[ -z "$CUDA_VISIBLE_DEVICES_TRAIN" ]]; then
    echo "ERROR: --gpus all requested, but no GPUs were detected."
    exit 1
  fi
elif [[ "$GPUS_ARG" =~ ^[0-9]+$ ]]; then
  NUM_GPUS="$GPUS_ARG"
elif [[ "$GPUS_ARG" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  CUDA_VISIBLE_DEVICES_TRAIN="$GPUS_ARG"
else
  echo "ERROR: --gpus must be an integer count, a comma list of GPU IDs, or 'all'"
  echo "       Examples: --gpus 2, --gpus 0,1,3, --gpus all"
  exit 1
fi

if [[ -n "$CUDA_VISIBLE_DEVICES_TRAIN" ]]; then
  IFS=',' read -ra _GPU_IDS <<< "$CUDA_VISIBLE_DEVICES_TRAIN"
  NUM_GPUS=${#_GPU_IDS[@]}
  export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_TRAIN"
fi

if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "ERROR: --gpus must resolve to a positive number of GPUs"
  exit 1
fi

# Prevent accidental reuse of stale artifacts from a previous run in another domain.
PIPELINE_DATA_DIR="$OUTPUT_DIR/pipeline/data"
OPPOSITE_DOMAIN="plant"
if [[ "$DOMAIN" == "plant" ]]; then
  OPPOSITE_DOMAIN="animal"
fi

if [[ -d "$PIPELINE_DATA_DIR/$OPPOSITE_DOMAIN" && "${GENECAD_ALLOW_MIXED_PIPELINE:-no}" != "yes" ]]; then
  echo "ERROR: Found existing $OPPOSITE_DOMAIN artifacts under $PIPELINE_DATA_DIR"
  echo "       while running --domain $DOMAIN. This can silently reuse stale data."
  echo ""
  echo "Choose one of:"
  echo "  1) Use a separate output dir, e.g. --output genecad_result/training/$DOMAIN"
  echo "  2) Remove stale artifacts: rm -rf $OUTPUT_DIR/pipeline"
  echo "  3) Bypass this guard: GENECAD_ALLOW_MIXED_PIPELINE=yes bash train.sh ..."
  exit 1
fi

# ── Dynamic Memory Settings ───────────────────────────────────────────────────
# Minimize GPU fragmentation out-of-memory errors
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Maintain effective batch size regardless of physical GPU constraints
ACCUM_GRAD=$(( TARGET_EFFECTIVE_BATCH / (BATCH_SIZE * NUM_GPUS) ))
if [ $ACCUM_GRAD -lt 1 ]; then
    ACCUM_GRAD=1
fi

print_batch_recommendations() {
  echo " Batch recommendations (for ${NUM_GPUS} GPU(s), effective=${TARGET_EFFECTIVE_BATCH}):"
  for bs in 4 6 8 10 12 16; do
    local acc=$(( TARGET_EFFECTIVE_BATCH / (bs * NUM_GPUS) ))
    if [[ "$acc" -lt 1 ]]; then
      acc=1
    fi
    local eff=$(( bs * acc * NUM_GPUS ))
    echo "   - --batch-size ${bs} => accumulate ${acc} (effective ${eff})"
  done
  echo "   Tip: prefer larger --batch-size first; increase effective batch only after that."
}

# ── Derived directories ───────────────────────────────────────────────────────
PIPELINE_DIR="$OUTPUT_DIR/pipeline"
EXTRACT_DIR="$PIPELINE_DIR/extract"
TRANSFORM_DIR="$PIPELINE_DIR/transform"
PREP_DIR="$PIPELINE_DIR/prep"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
QC_DIR="$OUTPUT_DIR/qc"
DATA_DIR="$PIPELINE_DIR/data/$DOMAIN"
GFF_DIR="$DATA_DIR/gff/training"
FASTA_DIR="$DATA_DIR/fasta/training"

if [[ -z "$ANIMAL_INPUT_DIR" ]]; then
  ANIMAL_INPUT_DIR="$FASTA_DIR"
fi

if [[ -z "$ANIMAL_GFF_DIR" ]]; then
  ANIMAL_GFF_DIR="$GFF_DIR"
fi

if [[ -n "${GENECAD_PYTHON:-}" && -x "$GENECAD_PYTHON" ]]; then
  PYTHON_CMD=("$GENECAD_PYTHON")
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_CMD=("${VIRTUAL_ENV}/bin/python")
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  PYTHON_CMD=("$SCRIPT_DIR/.venv/bin/python")
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_CMD=(".venv/bin/python")
else
  PYTHON_CMD=(uv run python)
fi

run_python() {
  "${PYTHON_CMD[@]}" "$@"
}
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Download a single file from a HuggingFace dataset repo using the Python API
# Usage: hf_download <repo_id> <remote_path> <local_dir>
hf_download() {
    local repo_id="$1"
    local remote_path="$2"
    local local_dir="$3"
    "${PYTHON_CMD[@]}" -c "
import sys
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='${repo_id}',
    filename='${remote_path}',
    repo_type='dataset',
    local_dir='${local_dir}',
    local_dir_use_symlinks=False,
)
"
}

# Normalize animal GFFs to the feature types expected by the training extractor.
# This keeps first-clone runs robust even when raw Ensembl GFFs are provided.
normalize_animal_gff() {
  local src="$1"
  local dst="$2"

  ANIMAL_GFF_SRC="$src" ANIMAL_GFF_DST="$dst" "${PYTHON_CMD[@]}" - <<'PY'
import gzip
import os

src = os.environ["ANIMAL_GFF_SRC"]
dst = os.environ["ANIMAL_GFF_DST"]
tmp = dst + ".tmp"

parent_feature_types = {"gene", "mRNA", "transcript"}
child_feature_types = {
  "exon",
  "CDS",
  "five_prime_UTR",
  "three_prime_UTR",
}


def parse_attrs(attr_field: str) -> dict[str, str]:
  attrs = {}
  for part in attr_field.rstrip(";").split(";"):
    if "=" not in part:
      continue
    k, v = part.split("=", 1)
    attrs[k.strip()] = v.strip()
  return attrs


def split_parents(parent_value: str) -> list[str]:
  if not parent_value:
    return []
  return [p.strip() for p in parent_value.split(",") if p.strip()]


def attrs_to_str(attrs: dict[str, str]) -> str:
  return ";".join(f"{k}={v}" for k, v in attrs.items())


gene_ids: set[str] = set()
transcripts: dict[str, dict] = {}
gene_to_tx: dict[str, list[str]] = {}

# Pass 1: collect genes and transcript metadata.
with gzip.open(src, "rt") as fin:
  for line in fin:
    if line.startswith("#"):
      continue
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 9:
      continue
    ftype = parts[2]
    if ftype not in parent_feature_types:
      continue
    attrs = parse_attrs(parts[8])
    fid = attrs.get("ID", "")
    if not fid:
      continue
    if ftype == "gene":
      gene_ids.add(fid)
    else:
      parent = split_parents(attrs.get("Parent", ""))
      parent_gene = parent[0] if parent else ""
      if not parent_gene:
        continue
      start = int(parts[3])
      end = int(parts[4])
      tags = attrs.get("tag", "")
      is_canonical = "Ensembl_canonical" in {t.strip() for t in tags.split(",") if t.strip()}
      transcripts[fid] = {
        "gene": parent_gene,
        "start": start,
        "end": end,
        "is_canonical": is_canonical,
      }
      gene_to_tx.setdefault(parent_gene, []).append(fid)

# Select a single transcript per gene:
# 1) Prefer Ensembl_canonical, 2) break ties by longest genomic span.
selected_tx_by_gene: dict[str, str] = {}
selected_tx_ids: set[str] = set()
for gene_id, tx_ids in gene_to_tx.items():
  if gene_id not in gene_ids or not tx_ids:
    continue
  candidates = [tid for tid in tx_ids if transcripts.get(tid, {}).get("is_canonical")]
  if not candidates:
    candidates = tx_ids
  best_tid = max(
    candidates,
    key=lambda tid: transcripts[tid]["end"] - transcripts[tid]["start"],
  )
  selected_tx_by_gene[gene_id] = best_tid
  selected_tx_ids.add(best_tid)

valid_gene_ids = {g for g, t in selected_tx_by_gene.items() if t in selected_tx_ids}

os.makedirs(os.path.dirname(dst), exist_ok=True)
gene_written = 0
mrna_written = 0
transcript_without_longest = 0

with gzip.open(src, "rt") as fin, gzip.open(tmp, "wt") as fout:
  for line in fin:
    if line.startswith("#"):
      fout.write(line)
      continue
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 9:
      continue
    ftype = parts[2]
    attrs = parse_attrs(parts[8])

    if ftype == "gene":
      if attrs.get("ID", "") in valid_gene_ids:
        fout.write(line)
        gene_written += 1
      continue

    if ftype in {"mRNA", "transcript"}:
      tid = attrs.get("ID", "")
      parents = split_parents(attrs.get("Parent", ""))
      parent_gene = parents[0] if parents else ""
      if parent_gene in selected_tx_by_gene and selected_tx_by_gene[parent_gene] == tid:
        # The extractor expects transcript type mRNA and uses longest=1 for canonical transcript.
        parts[2] = "mRNA"
        attrs["longest"] = "1"
        if attrs.get("longest") != "1":
          transcript_without_longest += 1
        parts[8] = attrs_to_str(attrs)
        fout.write("\t".join(parts) + "\n")
        mrna_written += 1
      continue

    if ftype in child_feature_types:
      parents = split_parents(attrs.get("Parent", ""))
      kept_parents = [p for p in parents if p in selected_tx_ids]
      if kept_parents:
        # Ensembl can reuse IDs like CDS:ENSDARP... across multiple segments.
        # Rewrite child IDs to a segment-unique key that is stable and parser-safe.
        child_parent = kept_parents[0]
        attrs["Parent"] = child_parent
        attrs["ID"] = (
          f"{ftype}:{child_parent}:{parts[3]}:{parts[4]}:{parts[6]}"
        )
        parts[8] = attrs_to_str(attrs)
        fout.write("\t".join(parts) + "\n")
      continue

if transcript_without_longest > 0:
  raise RuntimeError(
      f"Normalization invariant failed for {src}: found mRNA without longest=1"
  )

if gene_written != mrna_written:
  raise RuntimeError(
      f"Normalization invariant failed for {src}: genes={gene_written}, mRNA={mrna_written}. "
      "Expected exactly one transcript per retained gene."
  )

os.replace(tmp, dst)
print(
    f"Normalized GFF: {src} -> {dst} "
    f"(genes={gene_written}, mRNA={mrna_written}, one-longest-per-gene=yes)"
)
PY
}

hf_upload_folder() {
  local repo_id="$1"
  local folder_path="$2"
  local repo_type="$3"
  local path_in_repo="$4"
  local commit_message="$5"

  HF_UPLOAD_REPO_ID="$repo_id" \
  HF_UPLOAD_FOLDER="$folder_path" \
  HF_UPLOAD_REPO_TYPE="$repo_type" \
  HF_UPLOAD_PATH_IN_REPO="$path_in_repo" \
  HF_UPLOAD_COMMIT_MSG="$commit_message" \
  "${PYTHON_CMD[@]}" -c "
import os
from huggingface_hub import HfApi

api = HfApi()
repo_id = os.environ['HF_UPLOAD_REPO_ID']
folder_path = os.environ['HF_UPLOAD_FOLDER']
repo_type = os.environ['HF_UPLOAD_REPO_TYPE']
path_in_repo = os.environ['HF_UPLOAD_PATH_IN_REPO']
commit_message = os.environ['HF_UPLOAD_COMMIT_MSG']

api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
api.upload_folder(
  repo_id=repo_id,
  repo_type=repo_type,
  folder_path=folder_path,
  path_in_repo=path_in_repo,
  commit_message=commit_message,
)
print(f'Uploaded {folder_path} to {repo_id}/{path_in_repo}')
"
}

# ── Banner ────────────────────────────────────────────────────────────────────
echo "============================================"
echo " GeneCAD Fine-Tuning"
echo " Domain:      $DOMAIN"
echo " Base model:  $BASE_MODEL"
echo " Species:     $SPECIES_IDS"
echo " Output:      $OUTPUT_DIR"
echo " GPUs:        $NUM_GPUS"
if [[ -n "$CUDA_VISIBLE_DEVICES_TRAIN" ]]; then
  echo " CUDA devices: $CUDA_VISIBLE_DEVICES_TRAIN"
fi
echo " Batch size:  $BATCH_SIZE (accum: $ACCUM_GRAD)"
echo " LR:          $LEARNING_RATE"
echo " Epochs:      $EPOCHS"
echo " GFF parallel: $GFF_PARALLEL"
print_batch_recommendations
echo "============================================"

mkdir -p "$EXTRACT_DIR" "$TRANSFORM_DIR" "$PREP_DIR/splits" \
         "$GFF_DIR" "$FASTA_DIR" "$CHECKPOINT_DIR" "$QC_DIR"

# =============================================================================
# Step 0: Prepare training data
# =============================================================================
echo ""
if [[ "$DOMAIN" == "plant" ]]; then
  echo "[0/8] Downloading training data from HuggingFace..."

  HF_REPO="plantcad/genecad-dev"

  for species in Athaliana Osativa Gmax Hvulgare Ptrichocarpa; do
    dst="$GFF_DIR/${species}_top_transcript.gff3"
    if [ ! -f "$dst" ]; then
      echo "  Downloading GFF: $species"
      hf_download "$HF_REPO" \
        "data/plant/gff/training/${species}_top_transcript.gff3" \
        "$GFF_DIR"
      # hf_hub_download preserves the remote path structure; flatten to dst
      mv "$GFF_DIR/data/plant/gff/training/${species}_top_transcript.gff3" "$dst" 2>/dev/null || true
    else
      echo "  Skipped (exists): $species GFF"
    fi
  done

  declare -A FASTA_FILES
  FASTA_FILES["Athaliana_447_TAIR10.fa.gz"]="data/plant/fasta/training/Athaliana_447_TAIR10.fa.gz"
  FASTA_FILES["Osativa_323_v7.0.fa.gz"]="data/plant/fasta/training/Osativa_323_v7.0.fa.gz"
  FASTA_FILES["Gmax_880_v6.0.fa.gz"]="data/plant/fasta/training/Gmax_880_v6.0.fa.gz"
  FASTA_FILES["Hvulgare_462_r1.fa.gz"]="data/plant/fasta/training/Hvulgare_462_r1.fa.gz"
  FASTA_FILES["Ptrichocarpa_533_v4.0.fa.gz"]="data/plant/fasta/training/Ptrichocarpa_533_v4.0.fa.gz"

  for fname in "${!FASTA_FILES[@]}"; do
    dst="$FASTA_DIR/$fname"
    hf_path="${FASTA_FILES[$fname]}"
    if [ ! -f "$dst" ]; then
      echo "  Downloading FASTA: $fname"
      hf_download "$HF_REPO" "$hf_path" "$FASTA_DIR"
      mv "$FASTA_DIR/$hf_path" "$dst" 2>/dev/null || true
    else
      echo "  Skipped (exists): $fname"
    fi
  done
else
  echo "[0/8] Preparing animal training inputs..."

  HF_REPO="plantcad/genecad-dev"
  mkdir -p "$ANIMAL_GFF_DIR" "$ANIMAL_INPUT_DIR"

  # Download raw animal files from HF if they're missing locally.
  declare -A ANIMAL_GFF_REMOTE
  ANIMAL_GFF_REMOTE["Danio_rerio.GRCz11.115.chr.gff3.gz"]="data/animal/gff/training/Danio_rerio.GRCz11.115.chr.gff3.gz"
  ANIMAL_GFF_REMOTE["Gallus_gallus.bGalGal1.mat.broiler.GRCg7b.115.chr.gff3.gz"]="data/animal/gff/training/Gallus_gallus.bGalGal1.mat.broiler.GRCg7b.115.chr.gff3.gz"
  ANIMAL_GFF_REMOTE["Homo_sapiens.GRCh38.115.chr.gff3.gz"]="data/animal/gff/training/Homo_sapiens.GRCh38.115.chr.gff3.gz"
  ANIMAL_GFF_REMOTE["Xenopus_tropicalis.UCB_Xtro_10.0.115.chr.gff3.gz"]="data/animal/gff/training/Xenopus_tropicalis.UCB_Xtro_10.0.115.chr.gff3.gz"
  ANIMAL_GFF_REMOTE["Mus_musculus.GRCm39.115.chr.gff3.gz"]="data/animal/gff/training/Mus_musculus.GRCm39.115.chr.gff3.gz"

  declare -A ANIMAL_FASTA_REMOTE
  ANIMAL_FASTA_REMOTE["Danio_rerio.GRCz11.dna.toplevel.fa.gz"]="data/animal/fasta/training/Danio_rerio.GRCz11.dna.toplevel.fa.gz"
  ANIMAL_FASTA_REMOTE["Gallus_gallus.bGalGal1.mat.broiler.GRCg7b.dna.toplevel.fa.gz"]="data/animal/fasta/training/Gallus_gallus.bGalGal1.mat.broiler.GRCg7b.dna.toplevel.fa.gz"
  ANIMAL_FASTA_REMOTE["Homo_sapiens.GRCh38.dna.toplevel.fa.gz"]="data/animal/fasta/training/Homo_sapiens.GRCh38.dna.toplevel.fa.gz"
  ANIMAL_FASTA_REMOTE["Xenopus_tropicalis.UCB_Xtro_10.0.dna.toplevel.fa.gz"]="data/animal/fasta/training/Xenopus_tropicalis.UCB_Xtro_10.0.dna.toplevel.fa.gz"
  ANIMAL_FASTA_REMOTE["Mus_musculus.GRCm39.dna.toplevel.fa.gz"]="data/animal/fasta/training/Mus_musculus.GRCm39.dna.toplevel.fa.gz"

  for fname in "${!ANIMAL_GFF_REMOTE[@]}"; do
    if [ ! -f "$ANIMAL_GFF_DIR/$fname" ]; then
      echo "  Downloading animal GFF: $fname"
      hf_download "$HF_REPO" "${ANIMAL_GFF_REMOTE[$fname]}" "$ANIMAL_GFF_DIR"
      mv "$ANIMAL_GFF_DIR/${ANIMAL_GFF_REMOTE[$fname]}" "$ANIMAL_GFF_DIR/$fname" 2>/dev/null || true
    fi
  done

  for fname in "${!ANIMAL_FASTA_REMOTE[@]}"; do
    if [ ! -f "$ANIMAL_INPUT_DIR/$fname" ]; then
      echo "  Downloading animal FASTA: $fname"
      hf_download "$HF_REPO" "${ANIMAL_FASTA_REMOTE[$fname]}" "$ANIMAL_INPUT_DIR"
      mv "$ANIMAL_INPUT_DIR/${ANIMAL_FASTA_REMOTE[$fname]}" "$ANIMAL_INPUT_DIR/$fname" 2>/dev/null || true
    fi
  done

  # Prefer pre-processed names when present; otherwise use raw Ensembl names.
  DRERIO_GFF="$ANIMAL_GFF_DIR/Drerio_GRCz11.gene.gff3.gz"
  [ -f "$DRERIO_GFF" ] || DRERIO_GFF="$ANIMAL_GFF_DIR/Danio_rerio.GRCz11.115.chr.gff3.gz"
  GGALLUS_GFF="$ANIMAL_GFF_DIR/Ggallus_GRCg7b.gene.gff3.gz"
  [ -f "$GGALLUS_GFF" ] || GGALLUS_GFF="$ANIMAL_GFF_DIR/Gallus_gallus.bGalGal1.mat.broiler.GRCg7b.115.chr.gff3.gz"
  HSAPIENS_GFF="$ANIMAL_GFF_DIR/Hsapiens_GRCh38.gene.gff3.gz"
  [ -f "$HSAPIENS_GFF" ] || HSAPIENS_GFF="$ANIMAL_GFF_DIR/Homo_sapiens.GRCh38.115.chr.gff3.gz"
  XTROP_GFF="$ANIMAL_GFF_DIR/Xtropicalis_UCBXtro10.gene.gff3.gz"
  [ -f "$XTROP_GFF" ] || XTROP_GFF="$ANIMAL_GFF_DIR/Xenopus_tropicalis.UCB_Xtro_10.0.115.chr.gff3.gz"
  MMUSCULUS_GFF="$ANIMAL_GFF_DIR/Mmusculus_GRCm39.gene.gff3.gz"
  [ -f "$MMUSCULUS_GFF" ] || MMUSCULUS_GFF="$ANIMAL_GFF_DIR/Mus_musculus.GRCm39.115.chr.gff3.gz"

  DRERIO_FASTA="$ANIMAL_INPUT_DIR/Drerio_GRCz11.fa.gz"
  [ -f "$DRERIO_FASTA" ] || DRERIO_FASTA="$ANIMAL_INPUT_DIR/Danio_rerio.GRCz11.dna.toplevel.fa.gz"
  GGALLUS_FASTA="$ANIMAL_INPUT_DIR/Ggallus_GRCg7b.fa.gz"
  [ -f "$GGALLUS_FASTA" ] || GGALLUS_FASTA="$ANIMAL_INPUT_DIR/Gallus_gallus.bGalGal1.mat.broiler.GRCg7b.dna.toplevel.fa.gz"
  HSAPIENS_FASTA="$ANIMAL_INPUT_DIR/Hsapiens_GRCh38.fa.gz"
  [ -f "$HSAPIENS_FASTA" ] || HSAPIENS_FASTA="$ANIMAL_INPUT_DIR/Homo_sapiens.GRCh38.dna.toplevel.fa.gz"
  XTROP_FASTA="$ANIMAL_INPUT_DIR/Xtropicalis_UCBXtro10.fa.gz"
  [ -f "$XTROP_FASTA" ] || XTROP_FASTA="$ANIMAL_INPUT_DIR/Xenopus_tropicalis.UCB_Xtro_10.0.dna.toplevel.fa.gz"
  MMUSCULUS_FASTA="$ANIMAL_INPUT_DIR/Mmusculus_GRCm39.fa.gz"
  [ -f "$MMUSCULUS_FASTA" ] || MMUSCULUS_FASTA="$ANIMAL_INPUT_DIR/Mus_musculus.GRCm39.dna.toplevel.fa.gz"

  for f in \
      "$DRERIO_GFF" "$DRERIO_FASTA" \
      "$GGALLUS_GFF" "$GGALLUS_FASTA" \
      "$HSAPIENS_GFF" "$HSAPIENS_FASTA" \
      "$XTROP_GFF" "$XTROP_FASTA" \
      "$MMUSCULUS_GFF" "$MMUSCULUS_FASTA"; do
      if [ ! -f "$f" ]; then
          echo "ERROR: File not found: $f"
          exit 1
      fi
  done

  # Build canonical training GFFs expected by src/config.py and extractor.
  normalize_animal_gff "$DRERIO_GFF" "$GFF_DIR/Drerio_GRCz11.gene.gff3.gz"
  normalize_animal_gff "$GGALLUS_GFF" "$GFF_DIR/Ggallus_GRCg7b.gene.gff3.gz"
  normalize_animal_gff "$HSAPIENS_GFF" "$GFF_DIR/Hsapiens_GRCh38.gene.gff3.gz"
  normalize_animal_gff "$XTROP_GFF" "$GFF_DIR/Xtropicalis_UCBXtro10.gene.gff3.gz"
  normalize_animal_gff "$MMUSCULUS_GFF" "$GFF_DIR/Mmusculus_GRCm39.gene.gff3.gz"

  DRERIO_GFF="$GFF_DIR/Drerio_GRCz11.gene.gff3.gz"
  GGALLUS_GFF="$GFF_DIR/Ggallus_GRCg7b.gene.gff3.gz"
  HSAPIENS_GFF="$GFF_DIR/Hsapiens_GRCh38.gene.gff3.gz"
  XTROP_GFF="$GFF_DIR/Xtropicalis_UCBXtro10.gene.gff3.gz"
  MMUSCULUS_GFF="$GFF_DIR/Mmusculus_GRCm39.gene.gff3.gz"

  # Resolve FASTA paths to absolute to avoid relative-link surprises.
  DRERIO_FASTA="$(readlink -f "$DRERIO_FASTA")"
  GGALLUS_FASTA="$(readlink -f "$GGALLUS_FASTA")"
  HSAPIENS_FASTA="$(readlink -f "$HSAPIENS_FASTA")"
  XTROP_FASTA="$(readlink -f "$XTROP_FASTA")"
  MMUSCULUS_FASTA="$(readlink -f "$MMUSCULUS_FASTA")"
fi

# =============================================================================
# Step 1: Create symlinks with naming convention expected by src/config.py
# =============================================================================
echo ""
echo "[1/8] Setting up species data links..."
if [[ "$DOMAIN" == "plant" ]]; then
  ln -sf "Athaliana_top_transcript.gff3"   "$GFF_DIR/Athaliana_447_Araport11.gene.gff3"   2>/dev/null || true
  ln -sf "Athaliana_447_TAIR10.fa.gz"      "$FASTA_DIR/Athaliana_447.fasta"               2>/dev/null || true

  ln -sf "Osativa_top_transcript.gff3"     "$GFF_DIR/Osativa_323_v7.0.gene.gff3"          2>/dev/null || true
  ln -sf "Osativa_323_v7.0.fa.gz"          "$FASTA_DIR/Osativa_323.fasta"                  2>/dev/null || true

  ln -sf "Gmax_top_transcript.gff3"        "$GFF_DIR/Gmax_880_Wm82.a6.v1.gene.gff3"       2>/dev/null || true

  ln -sf "Hvulgare_top_transcript.gff3"    "$GFF_DIR/HvulgareMorex_702_V3.gene.gff3"       2>/dev/null || true
  ln -sf "Hvulgare_462_r1.fa.gz"           "$FASTA_DIR/HvulgareMorex_702_V3.fa.gz"         2>/dev/null || true

  ln -sf "Ptrichocarpa_top_transcript.gff3" "$GFF_DIR/Ptrichocarpa_533_v4.1.gene.gff3"    2>/dev/null || true
else
  # GFF files are already materialized as canonical files in Step 0.

  ln -sf "$DRERIO_FASTA"    "$FASTA_DIR/Drerio_GRCz11.fa.gz"         2>/dev/null || true
  ln -sf "$GGALLUS_FASTA"   "$FASTA_DIR/Ggallus_GRCg7b.fa.gz"        2>/dev/null || true
  ln -sf "$HSAPIENS_FASTA"  "$FASTA_DIR/Hsapiens_GRCh38.fa.gz"       2>/dev/null || true
  ln -sf "$XTROP_FASTA"     "$FASTA_DIR/Xtropicalis_UCBXtro10.fa.gz" 2>/dev/null || true
  ln -sf "$MMUSCULUS_FASTA" "$FASTA_DIR/Mmusculus_GRCm39.fa.gz"      2>/dev/null || true
fi

echo "  Done."

# =============================================================================
# Step 2: Extract GFF features
# =============================================================================
echo ""
echo "[2/8] Extracting GFF features..."
if [ ! -f "$EXTRACT_DIR/raw_features.parquet" ]; then
  if [[ "$GFF_PARALLEL" -gt 1 ]]; then
    echo "  Running species-parallel GFF extraction with $GFF_PARALLEL workers"
    SPECIES_EXTRACT_DIR="$EXTRACT_DIR/by_species"
    mkdir -p "$SPECIES_EXTRACT_DIR"
    # Clear stale shards from failed prior runs to avoid reusing invalid parquet outputs.
    rm -f "$SPECIES_EXTRACT_DIR"/*.parquet

    active_jobs=0
    for sid in $SPECIES_IDS; do
      out_file="$SPECIES_EXTRACT_DIR/${sid}.parquet"
      (
        run_python scripts/extract_train.py extract_gff_features \
          --input-dir "$GFF_DIR" \
          --species-id "$sid" \
          --output "$out_file"
      ) &

      active_jobs=$((active_jobs + 1))
      if [ "$active_jobs" -ge "$GFF_PARALLEL" ]; then
        wait -n
        active_jobs=$((active_jobs - 1))
      fi
    done
    wait

    SPECIES_EXTRACT_DIR="$SPECIES_EXTRACT_DIR" \
    SPECIES_LIST="$SPECIES_IDS" \
    MERGED_OUT="$EXTRACT_DIR/raw_features.parquet" \
    run_python - <<'PY'
import os
import pandas as pd

species = os.environ["SPECIES_LIST"].split()
base = os.environ["SPECIES_EXTRACT_DIR"]
out = os.environ["MERGED_OUT"]

paths = [os.path.join(base, f"{sid}.parquet") for sid in species]
missing = [p for p in paths if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(f"Missing species parquet files: {missing}")

dfs = [pd.read_parquet(p) for p in paths]
pd.concat(dfs, ignore_index=True).to_parquet(out)
print(f"Merged {len(paths)} species parquet files into {out}")
PY
  else
    run_python scripts/extract_train.py extract_gff_features \
      --input-dir "$GFF_DIR" \
      --species-id $SPECIES_IDS \
      --output "$EXTRACT_DIR/raw_features.parquet"
  fi
  echo "  Created: raw_features.parquet"
else
  echo "  Skipped (already exists)"
fi

# =============================================================================
# Step 3: Extract and tokenize FASTA sequences
# =============================================================================
echo ""
echo "[3/8] Extracting and tokenizing FASTA sequences..."
if [ ! -d "$EXTRACT_DIR/tokens.zarr" ]; then
  run_python scripts/extract_train.py extract_fasta_sequences \
    --input-dir "$FASTA_DIR" \
    --species-id $SPECIES_IDS \
    --tokenizer-path "$BASE_MODEL" \
    --output "$EXTRACT_DIR/tokens.zarr"
  echo "  Created: tokens.zarr"
else
  echo "  Skipped (already exists)"
fi

# =============================================================================
# Step 4: Filter features
# =============================================================================
echo ""
echo "[4/8] Filtering features..."
if [ ! -f "$TRANSFORM_DIR/features.parquet" ]; then
  run_python scripts/transform.py filter_features \
    --input "$EXTRACT_DIR/raw_features.parquet" \
    --output-features "$TRANSFORM_DIR/features.parquet" \
    --output-filters "$TRANSFORM_DIR/filters.parquet" \
    --remove-incomplete-features yes
  echo "  Created: features.parquet"
else
  echo "  Skipped (already exists)"
fi

# =============================================================================
# Step 5: Stack features
# =============================================================================
echo ""
echo "[5/8] Stacking features..."
if [ ! -f "$TRANSFORM_DIR/intervals.parquet" ]; then
  run_python scripts/transform.py stack_features \
    --input "$TRANSFORM_DIR/features.parquet" \
    --output "$TRANSFORM_DIR/intervals.parquet"
  echo "  Created: intervals.parquet"
else
  echo "  Skipped (already exists)"
fi

echo "  Running feature-table QC..."
run_python scripts/qc.py feature_qc \
  --features-parquet "$TRANSFORM_DIR/features.parquet" \
  --summary-json "$QC_DIR/feature_qc_summary.json"
echo "  Feature QC passed"

# =============================================================================
# Step 6: Create labels
# =============================================================================
echo ""
echo "[6/8] Creating labels..."
if [ ! -d "$TRANSFORM_DIR/labels.zarr" ]; then
  run_python scripts/transform.py create_labels \
    --input-features "$TRANSFORM_DIR/intervals.parquet" \
    --input-filters "$TRANSFORM_DIR/filters.parquet" \
    --output "$TRANSFORM_DIR/labels.zarr" \
    --remove-incomplete-features yes
  echo "  Created: labels.zarr"
else
  echo "  Skipped (already exists)"
fi

# =============================================================================
# Step 7: Create sequence dataset and training splits
# =============================================================================
echo ""
echo "[7/8] Creating sequence dataset and training splits..."
if [ ! -d "$TRANSFORM_DIR/sequences.zarr" ]; then
  run_python scripts/transform.py create_sequence_dataset \
    --input-labels "$TRANSFORM_DIR/labels.zarr" \
    --input-tokens "$EXTRACT_DIR/tokens.zarr" \
    --output-path "$TRANSFORM_DIR/sequences.zarr" \
    --num-workers $NUM_WORKERS
  echo "  Created: sequences.zarr"
else
  echo "  Skipped (already exists)"
fi

if [ ! -d "$PREP_DIR/splits/train.zarr" ]; then
  run_python scripts/sample.py generate_training_windows \
    --input "$TRANSFORM_DIR/sequences.zarr" \
    --output "$TRANSFORM_DIR/windows.zarr" \
    --window-size $WINDOW_SIZE \
    --intergenic-proportion $INTERGENIC_PROPORTION \
    --num-workers $NUM_WORKERS

  run_python scripts/sample.py generate_training_splits \
    --input "$TRANSFORM_DIR/windows.zarr" \
    --train-output "$PREP_DIR/splits/train.zarr" \
    --valid-output "$PREP_DIR/splits/valid.zarr" \
    --valid-proportion $VALID_PROPORTION
  echo "  Created: train.zarr + valid.zarr"
else
  echo "  Skipped (already exists)"
fi

if [[ "$LABEL_QC" == "yes" ]]; then
  echo "  Running label/split QC..."
  run_python scripts/qc.py label_qc \
    --train-dataset "$PREP_DIR/splits/train.zarr" \
    --valid-dataset "$PREP_DIR/splits/valid.zarr" \
    --summary-json "$QC_DIR/label_qc_summary.json" \
    --min-non-intergenic-ratio "$MIN_NON_INTERGENIC_RATIO" \
    --min-core-class-tokens "$MIN_CORE_CLASS_TOKENS" \
    --min-species-in-train 2 \
    --allow-missing-species-in-valid yes
  echo "  Label QC passed"
else
  echo "  Label QC disabled (--label-qc no)"
fi

# =============================================================================
# Step 8: Train
# =============================================================================
echo ""
echo "[8/8] Starting training..."
echo "  Base model:     $BASE_MODEL"
echo "  Architecture:   $ARCHITECTURE (frozen=$BASE_FROZEN)"
echo "  Effective batch: $((BATCH_SIZE * ACCUM_GRAD * NUM_GPUS))"
echo "  Output:         $OUTPUT_DIR"
echo ""

run_python scripts/train.py \
  --train-dataset "$PREP_DIR/splits/train.zarr" \
  --val-dataset   "$PREP_DIR/splits/valid.zarr" \
  --output-dir    "$OUTPUT_DIR" \
  --base-encoder-path    "$BASE_MODEL" \
  --base-encoder-frozen  "$BASE_FROZEN" \
  --architecture         "$ARCHITECTURE" \
  --head-encoder-layers  $HEAD_LAYERS \
  --token-embedding-dim  $TOKEN_EMBED_DIM \
  --batch-size           $BATCH_SIZE \
  --accumulate-grad-batches $ACCUM_GRAD \
  --epochs               $EPOCHS \
  --learning-rate        $LEARNING_RATE \
  --learning-rate-decay  cosine \
  --num-workers          $NUM_WORKERS \
  --prefetch-factor      2 \
  --gpu                  $NUM_GPUS \
  --strategy             ddp \
  --checkpoint-frequency 2000 \
  --val-check-interval   10000 \
  --train-eval-frequency 10000 \
  --limit-val-batches    1.0 \
  --log-frequency        1 \
  --enable-visualization no \
  --torch-compile        no \
  --auto-class-weights   "$AUTO_CLASS_WEIGHTS" \
  --project-name         "$PROJECT_NAME" \
  --run-name             "$RUN_NAME" \
  2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "============================================"
echo " Training complete!"
echo " Checkpoints: $CHECKPOINT_DIR"
echo " Log:         $OUTPUT_DIR/training.log"

echo " Selecting best checkpoint by validation metric..."
run_python scripts/qc.py select_best_checkpoint \
  --output-dir "$OUTPUT_DIR" \
  --metric "valid__entity__overall/f1" \
  --mode max \
  --summary-json "$QC_DIR/best_checkpoint_summary.json"
echo " Best checkpoint link: $CHECKPOINT_DIR/best.ckpt"

if [[ -n "$HF_UPLOAD_REPO" ]]; then
  echo ""
  echo " Uploading artifacts to Hugging Face..."
  hf_upload_folder \
    "$HF_UPLOAD_REPO" \
    "$CHECKPOINT_DIR" \
    "$HF_UPLOAD_TYPE" \
    "$HF_UPLOAD_PATH/checkpoints" \
    "Upload GeneCAD checkpoints: $RUN_NAME"

  hf_upload_folder \
    "$HF_UPLOAD_REPO" \
    "$OUTPUT_DIR" \
    "$HF_UPLOAD_TYPE" \
    "$HF_UPLOAD_PATH/logs" \
    "Upload GeneCAD logs and metadata: $RUN_NAME"

  echo " Uploaded to https://huggingface.co/$HF_UPLOAD_REPO"
fi

echo "============================================"
