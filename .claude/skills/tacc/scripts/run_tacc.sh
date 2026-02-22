#!/bin/bash
set -euo pipefail

# Run a command on a TACC compute node with the standard environment.
# Finds the first running "gg" or "gh" node from squeue automatically,
# or accepts an explicit node name.
#
# Usage:
#   bash .claude/skills/tacc/scripts/run_tacc.sh "python scripts/predict.py ..."
#   bash .claude/skills/tacc/scripts/run_tacc.sh --node c641-102 "python scripts/predict.py ..."

NODE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --node)
            NODE="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

CMD="$*"
if [ -z "$CMD" ]; then
    echo "Usage: bash .claude/skills/tacc/scripts/run_tacc.sh [--node NODE] COMMAND"
    exit 1
fi

if [ -z "$NODE" ]; then
    NODE=$(ssh tacc "squeue -u \$USER -h -t R -p gg,gh-dev,gh -o '%N' | head -1" 2>/dev/null | tail -1)
    if [ -z "$NODE" ]; then
        echo "ERROR: No running compute node found. Allocate one first with idev."
        exit 1
    fi
    echo "Auto-detected node: $NODE"
fi

echo "Running on $NODE: $CMD"
ssh tacc "ssh $NODE '\$HOME/local/bin/tacc_env.sh $CMD'"
