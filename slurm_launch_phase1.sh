#!/bin/bash
#SBATCH --job-name=ppt2-phase1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 -C "h200"
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_375_general

# Default values
RUN_NAME="run01"
MODEL_SIZE="190M"
SEED="12536"
CHECKPOINT=""
LOAD_EMBEDDINGS="true"
ALPHA="1.0"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --load-embeddings)
            LOAD_EMBEDDINGS="true"
            shift
            ;;
        --no-load-embeddings)
            LOAD_EMBEDDINGS="false"
            shift
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --run-name NAME       Run name (default: run01)"
            echo "  --model-size SIZE     Model size (default: 190M)"
            echo "  --seed SEED           Random seed (default: 12536)"
            echo "  --checkpoint PATH     Checkpoint path (default: none)"
            echo "  --load-embeddings     Load embeddings from checkpoint (default)"
            echo "  --no-load-embeddings  Don't load embeddings"
            echo "  --alpha VALUE         Alpha value (default: 1.0)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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
    --bind $HOME:$HOME \
    --bind /etc/ssl/certs:/etc/ssl/certs:ro \
    --bind /etc/pki:/etc/pki:ro \
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
    export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda-12.6/compat/lib.real
    export TRITON_LIBCUDA_PATH=/usr/local/cuda-12.6/compat/lib.real
    source ${PROJECT_ROOT}/.venv/bin/activate
    export WANDB_API_KEY='${WANDB_API_KEY}'
    export WANDB_PROJECT='${WANDB_PROJECT:-ppt2}'
    export PYTHONPATH='${PROJECT_ROOT}:\$PYTHONPATH'
    cd ${PROJECT_ROOT}
    $CMD
"

# export TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache
# export TORCH_INDUCTOR_PAD_MM_BENCHMARK=0
# export TORCHINDUCTOR_FX_GRAPH_CACHE=0
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print completion info
echo "Job completed at $(date)"
echo "Exit code: $?"
