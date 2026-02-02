#!/bin/bash
# setup_env.sh - Install uv and sync dependencies inside the container
# Usage: ./container/setup_env.sh
#
# This script uses pre-built flash-attn wheels from:
# https://github.com/mjun0812/flash-attention-prebuild-wheels
#
# No GPU node required for installation (pre-built wheel is used).
# However, a GPU is still required to run the code.

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

echo "=== Setting up environment inside container ==="
echo "Project root: $PROJECT_ROOT"
echo ""

$SINGULARITY exec \
    --nv \
    --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
    --bind "/scratch:/scratch" \
    --bind "$HOME:$HOME" \
    --pwd "$PROJECT_ROOT" \
    "$SIF_PATH" \
    bash -c '
        set -e

        echo "Installing uv..."
        export PATH="$HOME/.local/bin:$PATH"
        if ! command -v uv &> /dev/null; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.local/bin:$PATH"
        fi
        echo "uv version: $(uv --version)"
        echo ""

        cd '"$PROJECT_ROOT"'

        echo "Syncing dependencies..."
        export UV_LINK_MODE=copy
        uv sync --python 3.11

        echo ""
        echo "=== Setup complete ==="
        echo "Virtual environment created at: .venv/"
        echo ""
        echo "Verifying flash-attn installation..."
        .venv/bin/python -c "import flash_attn; print(f\"flash-attn version: {flash_attn.__version__}\")"
    '

echo ""
echo "Environment setup complete. You can now submit jobs with:"
echo "  sbatch slurm_launch_phase1.sh <run_name> <model_size>"
