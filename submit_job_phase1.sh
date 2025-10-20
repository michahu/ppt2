#!/bin/bash

# Convenience script for submitting Slurm jobs
# Usage: ./submit_job_phase1.sh [run_name] [model_size] [checkpoint]
# Examples:
#   ./submit_job_phase1.sh run01 190M
#   ./submit_job_phase1.sh run01 1B ./runs/190M_v0/step500
#   ./submit_job_phase1.sh run01 190M ./runs/190M_v0/step500

RUN_NAME=${1:-"run01"}
MODEL_SIZE=${2:-"190M"}
CHECKPOINT=${3:-""}

echo "Submitting Slurm job with:"
echo "  Run name: $RUN_NAME"
echo "  Model size: $MODEL_SIZE"
if [ -n "$CHECKPOINT" ]; then
    echo "  Checkpoint: $CHECKPOINT"
else
    echo "  Checkpoint: (none - training from scratch)"
fi
echo "  Node will be assigned by Slurm"

# Submit the job
sbatch slurm_launch_phase1.sh "$RUN_NAME" "$MODEL_SIZE" "$CHECKPOINT"

# Check job status
echo ""
echo "Job submitted. Check status with: squeue -u $USER"