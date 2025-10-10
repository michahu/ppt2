#!/bin/bash

# Convenience script for submitting Slurm jobs
# Usage: ./submit_job.sh [run_name]

RUN_NAME=${1:-"run01"}

echo "Submitting Slurm job with:"
echo "  Run name: $RUN_NAME"
echo "  Node will be assigned by Slurm"

# Submit the job
sbatch slurm_launch.sh "$RUN_NAME"

# Check job status
echo ""
echo "Job submitted. Check status with: squeue -u $USER"