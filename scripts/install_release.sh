#!/usr/bin/env bash
set -euo pipefail

# Install GeneCAD from a tracked GitHub release wheel.
# Usage:
#   bash scripts/install_release.sh
# Optional env vars:
#   GENECAD_VERSION=0.1.0
#   PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu128

GENECAD_VERSION="${GENECAD_VERSION:-0.1.0}"
PYTORCH_CUDA_INDEX="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
WHEEL_URL="https://github.com/plantcad/genecad/releases/download/v${GENECAD_VERSION}/genecad-${GENECAD_VERSION}-py3-none-any.whl"

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
