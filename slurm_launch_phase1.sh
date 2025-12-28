#!/bin/bash
#SBATCH --job-name=ppt2-phase1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 -C "a100|h100"
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

# Load modules (adjust based on your cluster setup)
# module load python/3.12
module avail cuda
module load cuda/11.6.2

# Activate virtual environment
source .venv/bin/activate

# Ensure we're in the right directory
cd /home/myh2014/code/ppt2

# Source WANDB configuration
source .config.sh

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Print environment info
echo "Python version: $(python --version)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"

# Run the training script with the new model_size argument
# Arguments order: train_single RUN_NAME NODE_NAME [MODEL_SIZE] [CHECKPOINT]
# if [ -n "$CHECKPOINT" ]; then
#     echo "Running: python ./scripts/phase1_nyu_wd_ablation.py train_single $RUN_NAME $NODE_NAME $MODEL_SIZE $CHECKPOINT"
#     python ./scripts/phase1_nyu_wd_ablation.py train_single "$RUN_NAME" "$NODE_NAME" "$MODEL_SIZE" "$CHECKPOINT"
# else
#     echo "Running: python ./scripts/phase1_nyu_wd_ablation.py train_single $RUN_NAME $NODE_NAME $MODEL_SIZE"
#     python ./scripts/phase1_nyu_wd_ablation.py train_single "$RUN_NAME" "$NODE_NAME" "$MODEL_SIZE"
# fi

# if [ -n "$CHECKPOINT" ]; then
#     echo "Running: python ./scripts/phase1_nyu_data_constrained.py train_single $RUN_NAME $NODE_NAME $MODEL_SIZE $CHECKPOINT"
#     python ./scripts/phase1_nyu_data_constrained.py train_single "$RUN_NAME" "$NODE_NAME" "$MODEL_SIZE" "$CHECKPOINT"
# else
#     echo "Running: python ./scripts/phase1_nyu_data_constrained.py train_single $RUN_NAME $NODE_NAME $MODEL_SIZE"
#     python ./scripts/phase1_nyu_data_constrained.py train_single "$RUN_NAME" "$NODE_NAME" "$MODEL_SIZE"
# fi

# Build command with optional arguments
CMD="python ./scripts/train_olmo2.py train_single \"$RUN_NAME\" \"$NODE_NAME\" \"$MODEL_SIZE\""

# Add checkpoint if provided
if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD \"$CHECKPOINT\""
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
eval $CMD

# Print completion info
echo "Job completed at $(date)"
echo "Exit code: $?"