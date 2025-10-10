#!/bin/bash
#SBATCH --job-name=ppt2-phase0
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Parse command line arguments
RUN_NAME=${1:-"run01"}

# Slurm sets the node name automatically
NODE_NAME=$(hostname)

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "Starting job on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node name: $NODE_NAME"
echo "Run name: $RUN_NAME"
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

# Run the training script with the same arguments as in README
echo "Running: python ./scripts/phase0_nyu.py train_single $RUN_NAME $NODE_NAME"
python ./scripts/phase0_nyu.py train_single "$RUN_NAME" "$NODE_NAME"

# Print completion info
echo "Job completed at $(date)"
echo "Exit code: $?"