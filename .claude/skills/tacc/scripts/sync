#!/bin/bash
set -euo pipefail

# Sync local code to TACC via rsync (default) or git.
# Usage:
#   bash .claude/skills/tacc/scripts/sync_tacc.sh          # rsync (fast, no commit needed)
#   bash .claude/skills/tacc/scripts/sync_tacc.sh --git    # git push + pull (requires clean commit)

REMOTE_REPO="\$WORK/repos/genecad"
MODE="rsync"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --git)
            MODE="git"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ "$MODE" = "rsync" ]; then
    echo "Syncing via rsync..."
    rsync -Pz --exclude '.*' --exclude '__pycache__' \
      -a ./ tacc:$REMOTE_REPO/
    echo "Rsync complete."
else
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    echo "Pushing local branch '$BRANCH' to origin..."
    git push origin "$BRANCH"
    echo "Pulling on TACC ($REMOTE_REPO)..."
    ssh tacc "bash -l -c 'source ~/.bashrc && cd $REMOTE_REPO && git fetch origin $BRANCH && git checkout $BRANCH && git pull origin $BRANCH'"
    echo "Git sync complete."
fi
