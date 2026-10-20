#!/usr/bin/env bash
# Native Linux ARM64 / GH200 installation without uv sync.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
GENECAD_VENV="${VIRTUAL_ENV:-$REPO_DIR/.venv}"
cache_base="${XDG_CACHE_HOME:-$HOME/.cache}"
if [[ -n "${WORK:-}" ]]; then cache_base="$WORK/.cache"; fi
GENECAD_WHEEL_DIR="${GENECAD_WHEEL_DIR:-$cache_base/genecad/arm64-cu128-cp312}"
SKIP_GPU_CHECK=0

usage() {
  cat <<'EOF'
Usage: bash scripts/install_arm64.sh [--venv PATH] [--wheel-dir PATH] [--jobs N] [--skip-gpu-check]

Install GeneCAD with native ARM64 PyTorch and CUDA extensions on Linux/GH200.
Requires uv, git, a C/C++ compiler, CUDA 12.8 (nvcc), and Python 3.12.
uv can download Python 3.12 if it is not installed. Uses official ARM64
PyTorch/Triton wheels and builds FlashAttention, causal-conv1d and Mamba.
Completed wheels are cached and reused by subsequent installations.

--venv PATH       Target environment (default: active venv, otherwise .venv).
--wheel-dir PATH  Persistent native wheel cache (GENECAD_WHEEL_DIR).
--jobs N          Maximum parallel compiler jobs (MAX_JOBS; default 8).
--skip-gpu-check  Build on a CPU compute node; GPU validation is still required.
EOF
}

while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --venv|--wheel-dir|--jobs)
      [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --venv) GENECAD_VENV="$2" ;;
        --wheel-dir) GENECAD_WHEEL_DIR="$2" ;;
        --jobs) export MAX_JOBS="$2" ;;
      esac
      shift 2 ;;
    --skip-gpu-check) SKIP_GPU_CHECK=1; shift ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] || {
  echo "This installer requires Linux aarch64; use install_release.sh on x86-64." >&2; exit 1;
}
for tool in uv git nvcc cc c++ flock; do
  command -v "$tool" >/dev/null || { echo "Required command not found: $tool" >&2; exit 1; }
done
nvcc --version | grep -q 'release 12.8' || {
  echo "Load a CUDA 12.8 toolkit before installing (Vista: module load gcc/14.2.0 cuda/12.8)." >&2; exit 1;
}
export MAX_JOBS="${MAX_JOBS:-8}" NVCC_THREADS="${NVCC_THREADS:-2}"
[[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs must be positive" >&2; exit 2; }
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}" TOKIO_WORKER_THREADS="${TOKIO_WORKER_THREADS:-2}"
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-2}"
export TORCH_CUDA_ARCH_LIST=9.0 FLASH_ATTN_CUDA_ARCHS=90
export FLASH_ATTENTION_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE

mkdir -p "$GENECAD_WHEEL_DIR"
GENECAD_WHEEL_DIR="$(cd "$GENECAD_WHEEL_DIR" && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$GENECAD_WHEEL_DIR/uv-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$GENECAD_WHEEL_DIR/pip-cache}"
# Keep two installations from writing into the same wheel cache.
exec 9>"$GENECAD_WHEEL_DIR/.install.lock"
flock 9
if [[ ! -x "$GENECAD_VENV/bin/python" ]]; then
  uv venv --python 3.12 --seed "$GENECAD_VENV"
fi
GENECAD_VENV="$(cd "$GENECAD_VENV" && pwd)"
PYTHON="$GENECAD_VENV/bin/python"
export PATH="$GENECAD_VENV/bin:$PATH"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "GeneCAD requires Python 3.12"'
uv pip install --python "$PYTHON" 'pip>=25,<26' 'setuptools>=70.1,<80' \
  'wheel>=0.45,<0.49' packaging ninja psutil

build_dir="$(mktemp -d "$GENECAD_WHEEL_DIR/build.XXXXXXXX")"
trap 'status=$?; if ((status)); then echo "Installation failed; build files: $build_dir" >&2; fi' EXIT

build_wheel() {
  local pattern="$1" source="$2"
  local wheels=()
  shopt -s nullglob
  wheels=("$GENECAD_WHEEL_DIR"/$pattern)
  shopt -u nullglob
  if ((${#wheels[@]} == 0)); then
    "$PYTHON" -m pip wheel --no-deps --no-build-isolation --wheel-dir "$build_dir/wheels" "$source"
    shopt -s nullglob
    wheels=("$build_dir/wheels"/$pattern)
    shopt -u nullglob
    ((${#wheels[@]} == 1)) || { echo "Expected one native wheel for $pattern" >&2; exit 1; }
    cp "${wheels[0]}" "$GENECAD_WHEEL_DIR/$(basename "${wheels[0]}").tmp"
    mv "$GENECAD_WHEEL_DIR/$(basename "${wheels[0]}").tmp" "$GENECAD_WHEEL_DIR/$(basename "${wheels[0]}")"
    wheels=("$GENECAD_WHEEL_DIR/$(basename "${wheels[0]}")")
  fi
  ((${#wheels[@]} == 1)) || { echo "Ambiguous wheel cache for $pattern" >&2; exit 1; }
  uv pip install --python "$PYTHON" "${wheels[0]}"
}

# PyPI has only x86-64 Triton 3.3.1 wheels. PyTorch's CUDA index also
# publishes the matching official ARM64 wheel; keep its compiler/tool versions.
triton_name=triton-3.3.1-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
triton_wheel="$GENECAD_WHEEL_DIR/$triton_name"
triton_sha256=9b1993c3ba470117dcf2888ff116d82809ed7a4d3cd05074fc40ff39190b2ce8
if [[ ! -f "$triton_wheel" ]]; then
  "$PYTHON" -m pip download --only-binary=:all: --no-deps \
    --index-url https://download.pytorch.org/whl/cu128 \
    --dest "$build_dir/downloads" 'triton==3.3.1'
  mv "$build_dir/downloads/$triton_name" "$triton_wheel"
fi
printf '%s  %s\n' "$triton_sha256" "$triton_wheel" | sha256sum --check --status || {
  echo "Cached Triton wheel checksum mismatch; use a new --wheel-dir." >&2; exit 1;
}
uv pip install --python "$PYTHON" "$triton_wheel"
uv pip install --python "$PYTHON" 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PYTHON" --editable "$REPO_DIR" \
  'lightning~=2.5.1' 'torchmetrics~=1.7.0' einops 'transformers==4.57.6'
build_wheel 'causal_conv1d-1.5.0.post8-cp312-cp312-*aarch64.whl' \
  'git+https://github.com/Dao-AILab/causal-conv1d@82867a9d2e6907cc0f637ac6aff318f696838548'
build_wheel 'mamba_ssm-2.2.4-cp312-cp312-*aarch64.whl' \
  'git+https://github.com/state-spaces/mamba@95d8aba8a8c75aedcaa6143713b11e745e7cd0d9'
build_wheel 'flash_attn-2.8.3-cp312-cp312-*aarch64.whl' 'flash-attn==2.8.3'
uv pip check --python "$PYTHON"
"$PYTHON" -m pip freeze > "$GENECAD_VENV/genecad-requirements.txt"
sha256sum "$GENECAD_WHEEL_DIR"/*.whl > "$GENECAD_VENV/genecad-wheels.sha256"
if ((SKIP_GPU_CHECK)); then
  "$PYTHON" "$SCRIPT_DIR/check_install.py" --imports-only --output "$GENECAD_VENV/genecad-import-check.json"
  echo "CPU build complete; run $PYTHON $SCRIPT_DIR/check_install.py on a GPU node."
else
  "$PYTHON" "$SCRIPT_DIR/check_install.py" --output "$GENECAD_VENV/genecad-install-check.json"
fi
rm -rf -- "$build_dir"
echo "GeneCAD installed. Activate with: source $GENECAD_VENV/bin/activate"
