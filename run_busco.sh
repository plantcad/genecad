#!/bin/bash
# Flexible BUSCO runner script
# Usage: ./run_busco.sh [--env ENV_NAME_OR_PATH] [BUSCO_ARGS...]
# Example: ./run_busco.sh --env busco-5.5.0 -i input.fa -l embryophyta_odb10 -o output --mode genome

set -e

# Default environment name
ENV_NAME="busco-5.5.0"

# Parse optional --env argument
if [[ "$1" == "--env" ]]; then
  ENV_NAME="$2"
  shift 2
fi

# Try to activate the environment
if [[ -f /programs/miniconda3/bin/activate ]]; then
  source /programs/miniconda3/bin/activate "$ENV_NAME"
else
  echo "[ERROR] Could not find conda activate script at /programs/miniconda3/bin/activate" >&2
  exit 1
fi

# Check if busco is available
if ! command -v busco &> /dev/null; then
  echo "[ERROR] BUSCO is not installed or not in PATH in environment '$ENV_NAME'" >&2
  exit 2
fi

# Run BUSCO with all remaining arguments
busco "$@"
