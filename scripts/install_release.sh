#!/usr/bin/env bash
set -euo pipefail

# Grace Hopper uses an ARM CPU. Keep the existing x86-64 release installation
# below, and use native dependencies rather than x86 wheels on ARM64.
if [[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]]; then
  exec bash "$(dirname "${BASH_SOURCE[0]}")/install_arm64.sh" "$@"
fi

# Install GeneCAD from a tracked GitHub release wheel.
# Usage:
#   bash scripts/install_release.sh
# Optional env vars:
#   GENECAD_VERSION=0.7.0
#   PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu128

GENECAD_VERSION="${GENECAD_VERSION:-0.7.0}"
PYTORCH_CUDA_INDEX="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
WHEEL_URL="https://github.com/plantcad/genecad/releases/download/v${GENECAD_VERSION}/genecad-${GENECAD_VERSION}-py3-none-any.whl"

# --- Preflight checks -------------------------------------------------------
# The prebuilt FlashAttention wheel needs glibc >= 2.32, and the PlantCAD2
# backbone imports flash_attn, so older systems (e.g. RHEL/Rocky 8) cannot
# run predictions from a native install. Use the container image there.
GLIBC_VERSION="$(ldd --version 2>/dev/null | head -n1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
if [[ -n "${GLIBC_VERSION}" ]] && \
   [[ "$(printf '%s\n' 2.32 "${GLIBC_VERSION}" | sort -V | head -n1)" != "2.32" ]]; then
  echo "ERROR: glibc ${GLIBC_VERSION} found; GeneCAD's native install needs glibc >= 2.32." >&2
  echo "Use the container image instead (works on any glibc):" >&2
  echo "  apptainer pull genecad.sif docker://ghcr.io/plantcad/genecad_v1:latest" >&2
  echo "See docs/alternative_installation_methods.md. Set GENECAD_SKIP_GLIBC_CHECK=1 to override." >&2
  [[ "${GENECAD_SKIP_GLIBC_CHECK:-0}" == 1 ]] || exit 1
fi

# mamba-ssm and causal-conv1d are compiled from source and need a CUDA 12.x
# toolkit (nvcc). Prefer CUDA_HOME, then nvcc on PATH, then /usr/local/cuda-12*.
nvcc_major() { "$1/bin/nvcc" --version 2>/dev/null | grep -oE 'release [0-9]+' | grep -oE '[0-9]+'; }
if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
if [[ -z "${CUDA_HOME:-}" || "$(nvcc_major "${CUDA_HOME}")" != 12 ]]; then
  for cand in $(ls -d /usr/local/cuda-12* 2>/dev/null | sort -rV); do
    if [[ "$(nvcc_major "$cand")" == 12 ]]; then CUDA_HOME="$cand"; break; fi
  done
fi
if [[ -z "${CUDA_HOME:-}" || "$(nvcc_major "${CUDA_HOME}")" != 12 ]]; then
  echo "ERROR: a CUDA 12.x toolkit (nvcc) is required to build mamba-ssm/causal-conv1d." >&2
  echo "On HPC try 'module load cuda/12.8', or set CUDA_HOME to a CUDA 12.x install." >&2
  echo "Alternatively use the container image (no compilation needed)." >&2
  exit 1
fi
export CUDA_HOME PATH="${CUDA_HOME}/bin:${PATH}"
echo "Using CUDA toolkit: ${CUDA_HOME}"

echo "Installing GeneCAD v${GENECAD_VERSION} from release wheel..."
uv pip install \
  --extra-index-url "${PYTORCH_CUDA_INDEX}" \
  --index-strategy unsafe-best-match \
  "genecad[torch] @ ${WHEEL_URL}" \
  "transformers<5"

echo "Installing mamba dependencies..."
uv pip install \
  --extra-index-url "${PYTORCH_CUDA_INDEX}" \
  --index-strategy unsafe-best-match \
  --no-build-isolation \
  "mamba-ssm @ git+https://github.com/state-spaces/mamba@v2.2.4" \
  "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d@v1.5.0.post8"

if [ -f "pyproject.toml" ]; then
  echo "Installing local package in editable mode to synchronize CLI with source code..."
  echo "(The release wheel download has already been counted.)"
  uv pip install -e .
fi

if [ -x ".venv/bin/genecad" ] && ! command -v genecad >/dev/null 2>&1; then
  mkdir -p "$HOME/.local/bin"
  ln -sf "$(pwd)/.venv/bin/genecad" "$HOME/.local/bin/genecad"
  echo "==================================================================="
  echo "✔ Symlinked 'genecad' to ~/.local/bin/genecad"
  echo "You can now run 'genecad' from anywhere without activating the virtual environment!"
  if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo ""
    echo "Note: ~/.local/bin is not in your PATH. Please add it to your ~/.bashrc."
  fi
  echo "==================================================================="
fi

echo "Done."
