#!/bin/bash
#SBATCH --job-name=ppt2-phase1-reinit
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=48:00:00

# Parse command line arguments
RUN_NAME=${1:-"run01"}
MODEL_SIZE=${2:-"190M"}
CHECKPOINT=${3:-""}
FIRST_K=${4:-2}
METHOD=${5:-"orthogonal"}
INIT_SCALE=${6:-1.0}
shift 6
LAYERS="$@"  # All remaining arguments are layer indices

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
if [ -n "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
else
    echo "Checkpoint: (none - training from scratch)"
fi
echo "First k heads: $FIRST_K"
echo "Reinit method: $METHOD"
echo "Init scale: $INIT_SCALE"
echo "Layers to reinitialize: $LAYERS"
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
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
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

# Run the training script with reinitialization
if [ -n "$CHECKPOINT" ]; then
    echo "Running: python ./scripts/phase1_reinit.py train_single $RUN_NAME $NODE_NAME $MODEL_SIZE $CHECKPOINT --first_k $FIRST_K --method $METHOD --init_scale $INIT_SCALE --layers $LAYERS"
    $SING_EXEC bash -c "
        source ${PROJECT_ROOT}/.venv/bin/activate
        export WANDB_API_KEY='${WANDB_API_KEY}'
        export WANDB_PROJECT='${WANDB_PROJECT:-ppt2}'
        cd ${PROJECT_ROOT}
        python ./scripts/phase1_reinit.py train_single '$RUN_NAME' '$NODE_NAME' '$MODEL_SIZE' '$CHECKPOINT' --first_k $FIRST_K --method $METHOD --init_scale $INIT_SCALE --layers $LAYERS
    "
else
    echo "Running: python ./scripts/phase1_reinit.py train_single $RUN_NAME $NODE_NAME $MODEL_SIZE --first_k $FIRST_K --method $METHOD --init_scale $INIT_SCALE --layers $LAYERS"
    $SING_EXEC bash -c "
        source ${PROJECT_ROOT}/.venv/bin/activate
        export WANDB_API_KEY='${WANDB_API_KEY}'
        export WANDB_PROJECT='${WANDB_PROJECT:-ppt2}'
        cd ${PROJECT_ROOT}
        python ./scripts/phase1_reinit.py train_single '$RUN_NAME' '$NODE_NAME' '$MODEL_SIZE' --first_k $FIRST_K --method $METHOD --init_scale $INIT_SCALE --layers $LAYERS
    "
fi

# Print completion info
echo "Job completed at $(date)"
echo "Exit code: $?"
