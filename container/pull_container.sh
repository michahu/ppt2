#!/bin/bash
# pull_container.sh - Pull NVIDIA NGC PyTorch container
# Usage: ./container/pull_container.sh
#
# This can be run on a login node (no GPU required).
# The container includes CUDA 12.6 development headers needed for flash-attn.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Singularity/Apptainer path
SINGULARITY="/share/apps/apptainer/bin/singularity"

# Use /dev/shm for temp (supports xattrs, unlike Lustre /scratch)
# Use /scratch for cache (large storage for downloaded blobs)
export APPTAINER_TMPDIR="/dev/shm/${USER}_apptainer_tmp"
export APPTAINER_CACHEDIR="${SCRATCH:-/scratch/${USER}}/apptainer_cache"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# Check /dev/shm has enough space (need ~50GB for NGC PyTorch)
SHM_AVAIL=$(df -BG /dev/shm | awk 'NR==2 {print $4}' | tr -d 'G')
if [ "$SHM_AVAIL" -lt 50 ]; then
    echo "WARNING: /dev/shm has only ${SHM_AVAIL}GB available, may need ~50GB"
    echo "If build fails, try running on a compute node with more memory"
fi

# Container configuration
NGC_IMAGE="nvcr.io/nvidia/pytorch:24.12-py3"
SIF_NAME="ppt2.sif"
SIF_PATH="${SCRIPT_DIR}/${SIF_NAME}"

echo "=== PPT2 Container Setup ==="
echo "NGC Image: $NGC_IMAGE"
echo "Output: $SIF_PATH"
echo ""

# Check if container already exists
if [ -f "$SIF_PATH" ]; then
    echo "Container already exists at $SIF_PATH"
    echo "Delete it first if you want to re-pull:"
    echo "  rm $SIF_PATH"
    exit 0
fi

# Pull the container
echo "Pulling container from NGC..."
echo "This may take 10-30 minutes depending on network speed."
echo ""

$SINGULARITY pull "$SIF_PATH" "docker://$NGC_IMAGE"

echo ""
echo "=== Container pulled successfully ==="
echo "Container saved to: $SIF_PATH"
echo "Size: $(du -h "$SIF_PATH" | cut -f1)"
echo ""
echo "Next steps:"
echo "  1. Get on a GPU node: srun --gres=gpu:1 -C 'a100|h100' --mem=64GB --time=1:00:00 --pty bash"
echo "  2. Run setup: ./container/setup_env.sh"
