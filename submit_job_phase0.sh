#!/bin/bash

# Convenience script for submitting Slurm jobs for Phase 0
# Usage: ./submit_job.sh [run_name] [model_size]
# Examples:
#   ./submit_job.sh run01 190M
#   ./submit_job.sh run01 1B
#   ./submit_job.sh run01      # defaults to 1B

RUN_NAME=${1:-"run01"}
MODEL_SIZE=${2:-"1B"}

echo "Submitting Slurm job with:"
echo "  Run name: $RUN_NAME"
echo "  Model size: $MODEL_SIZE"
echo "  Node will be assigned by Slurm"

# Submit the job
sbatch slurm_launch_phase0.sh "$RUN_NAME" "$MODEL_SIZE"

# Check job status
echo ""
echo "Job submitted. Check status with: squeue -u $USER"
