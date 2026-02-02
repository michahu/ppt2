#!/bin/bash
# run_in_container.sh - Execute commands inside the Singularity container
# Usage: ./container/run_in_container.sh <command> [args...]
#
# Examples:
#   ./container/run_in_container.sh bash              # Interactive shell
#   ./container/run_in_container.sh python --version  # Run python
#   ./container/run_in_container.sh nvidia-smi        # Check GPU access

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SIF_PATH="${SCRIPT_DIR}/ppt2.sif"
SINGULARITY="/share/apps/apptainer/bin/singularity"

# Verify container exists
if [ ! -f "$SIF_PATH" ]; then
    echo "Error: Container not found at $SIF_PATH"
    echo "Run ./container/pull_container.sh first"
    exit 1
fi

# Execute with appropriate bind mounts
exec $SINGULARITY exec \
    --nv \
    --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
    --bind "/scratch:/scratch" \
    --bind "$HOME:$HOME" \
    --pwd "$PROJECT_ROOT" \
    "$SIF_PATH" \
    "$@"
