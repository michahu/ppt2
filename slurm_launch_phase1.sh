#!/bin/bash
#SBATCH --job-name=ppt2-phase1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=48:00:00

# Parse command line arguments
RUN_NAME=${1:-"run01"}
MODEL_SIZE=${2:-"190M"}
SEED=${3:-"12536"}
CHECKPOINT=${4:-""}
LOAD_EMBEDDINGS=${5:-"true"}  # "true" or "false"
ALPHA=${6:-"1.0"}

# Slurm sets the node name automatically
NODE_NAME=$(hostname)

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "Starting job on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node name: $NODE_NAME"
echo "Run name: $RUN_NAME"
echo "Model size: $MODEL_SIZE"
echo "Seed: $SEED"
if [ -n "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
else
    echo "Checkpoint: (none - training from scratch)"
fi
echo "Load embeddings: $LOAD_EMBEDDINGS"
echo "Alpha: $ALPHA"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"

# ===========================================
# SINGULARITY CONFIGURATION
# ===========================================
SINGULARITY="/share/apps/apptainer/bin/singularity"
PROJECT_ROOT="/scratch/myh2014/ppt2"
SIF_PATH="${PROJECT_ROOT}/container/ppt2.sif"

# Verify container exists
if [ ! -f "$SIF_PATH" ]; then
    echo "Error: Container not found at $SIF_PATH"
    echo "Run ./container/pull_container.sh first"
    exit 1
fi

# Define singularity exec command with bind mounts
SING_EXEC="$SINGULARITY exec --nv \
    --bind /scratch:/scratch \
    --bind $HOME:$HOME \
    --pwd ${PROJECT_ROOT} \
    ${SIF_PATH}"

# Auto-setup: ensure uv and dependencies are installed
if [ ! -d "${PROJECT_ROOT}/.venv" ]; then
    echo "Virtual environment not found. Running first-time setup..."
    $SING_EXEC bash -c "
        export PATH=\$HOME/.local/bin:\$PATH
        if ! command -v uv &> /dev/null; then
            echo 'Installing uv...'
            curl -LsSf https://astral.sh/uv/install.sh | sh
        fi
        cd ${PROJECT_ROOT}
        export MAX_JOBS=4
        uv sync --python 3.11
    "
fi

# Source WANDB configuration (outside container)
if [ -f "${PROJECT_ROOT}/.config.sh" ]; then
    source "${PROJECT_ROOT}/.config.sh"
fi

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print environment info
echo "Singularity container: $SIF_PATH"
$SING_EXEC python --version
$SING_EXEC nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "Working directory: ${PROJECT_ROOT}"

# Build command with optional arguments
CMD="python ./scripts/train_olmo2.py train_single '$RUN_NAME' '$NODE_NAME' '$MODEL_SIZE'"

# Add checkpoint if provided
if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD '$CHECKPOINT'"
fi

# Add seed
CMD="$CMD --seed=$SEED"

# Add --no-load-embeddings if LOAD_EMBEDDINGS is false
if [ "$LOAD_EMBEDDINGS" = "false" ]; then
    CMD="$CMD --no-load-embeddings"
fi

# Add alpha if not default
if [ "$ALPHA" != "1.0" ]; then
    CMD="$CMD --alpha=$ALPHA"
fi

echo "Running: $CMD"
$SING_EXEC bash -c "
    source ${PROJECT_ROOT}/.venv/bin/activate
    export WANDB_API_KEY='${WANDB_API_KEY}'
    export WANDB_PROJECT='${WANDB_PROJECT:-ppt2}'
    cd ${PROJECT_ROOT}
    $CMD
"

# Print completion info
echo "Job completed at $(date)"
echo "Exit code: $?"
