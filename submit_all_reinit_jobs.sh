#!/bin/bash

# Script to submit reinitialization jobs - single job or batch of 12 jobs
# 
# Usage for batch (all 12 jobs):
#   ./submit_all_reinit_jobs.sh all [base_run_name] [model_size] [checkpoint] [first_k] [method]
# 
# Usage for single job:
#   ./submit_all_reinit_jobs.sh [run_name] [model_size] [checkpoint] [first_k] [method] [layers...]
# 
# Examples:
#   # Submit all 12 jobs
#   ./submit_all_reinit_jobs.sh all reinit_sweep 190M "" 2 orthogonal
#   ./submit_all_reinit_jobs.sh all reinit_sweep 190M "" 2 zeros
#   ./submit_all_reinit_jobs.sh all reinit_sweep 190M ./runs/baseline/step500 2 orthogonal
#   
#   # Submit single job
#   ./submit_all_reinit_jobs.sh reinit_test 190M "" 2 orthogonal 0 1
#   ./submit_all_reinit_jobs.sh reinit_test 190M "" 2 zeros 0 1 2 3
#   ./submit_all_reinit_jobs.sh reinit_all 190M "" 4 zeros all

MODE=${1:-"all"}

# Check if first argument is "all" to determine batch mode
if [ "$MODE" == "all" ]; then
    # Batch mode: submit all 12 jobs
    BASE_RUN_NAME=${2:-"reinit_sweep"}
    MODEL_SIZE=${3:-"190M"}
    CHECKPOINT=${4:-""}
    FIRST_K=${5:-2}
    METHOD=${6:-"orthogonal"}
    
    echo "=========================================="
    echo "Submitting 12 Reinitialization Jobs"
    echo "=========================================="
    echo "Base run name: $BASE_RUN_NAME"
    echo "Model size: $MODEL_SIZE"
    if [ -n "$CHECKPOINT" ]; then
        echo "Checkpoint: $CHECKPOINT"
    else
        echo "Checkpoint: (none - training from scratch)"
    fi
    echo "First k heads: $FIRST_K"
    echo "Reinit method: $METHOD"
    echo "=========================================="
    echo ""
    
    # Array to store job IDs
    declare -a job_ids
    
    # Submit 12 jobs with increasing numbers of layers
    for num_layers in {1..12}; do
        # Generate layer sequence: 0 1 2 ... (num_layers-1)
        layers=""
        for ((i=0; i<num_layers; i++)); do
            layers="$layers $i"
        done
        
        # Create a descriptive run name
        RUN_NAME="${BASE_RUN_NAME}_${METHOD}_layers0to$((num_layers-1))_k${FIRST_K}"
        
        echo "[$num_layers/12] Submitting job: $RUN_NAME"
        echo "  Layers: $layers"
        
        # Submit the job and capture the job ID
        if [ -n "$CHECKPOINT" ]; then
            job_output=$(sbatch slurm_launch_phase1_reinit.sh "$RUN_NAME" "$MODEL_SIZE" "$CHECKPOINT" "$FIRST_K" "$METHOD" $layers)
        else
            job_output=$(sbatch slurm_launch_phase1_reinit.sh "$RUN_NAME" "$MODEL_SIZE" "" "$FIRST_K" "$METHOD" $layers)
        fi
        
        # Extract job ID from sbatch output (typically "Submitted batch job 12345")
        job_id=$(echo $job_output | awk '{print $NF}')
        job_ids+=($job_id)
        
        echo "  Job ID: $job_id"
        echo ""
    done
    
    echo "=========================================="
    echo "All 12 jobs submitted!"
    echo "=========================================="
    echo ""
    echo "Job IDs:"
    for i in "${!job_ids[@]}"; do
        echo "  Job $((i+1)): ${job_ids[$i]}"
    done
    echo ""
    echo "Check job status with: squeue -u $USER"
    echo "Cancel all jobs with: scancel ${job_ids[@]}"
    echo ""
    echo "Monitor logs in: logs/slurm-<job_id>.out"
else
    # Single job mode
    RUN_NAME=$1
    MODEL_SIZE=${2:-"190M"}
    CHECKPOINT=${3:-""}
    FIRST_K=${4:-2}
    METHOD=${5:-"orthogonal"}
    shift 5
    LAYERS="$@"
    
    # Default to layers 0 1 if not specified
    if [ -z "$LAYERS" ]; then
        LAYERS="0 1"
    fi
    
    echo "Submitting single reinitialization job:"
    echo "  Run name: $RUN_NAME"
    echo "  Model size: $MODEL_SIZE"
    if [ -n "$CHECKPOINT" ]; then
        echo "  Checkpoint: $CHECKPOINT"
    else
        echo "  Checkpoint: (none - training from scratch)"
    fi
    echo "  First k heads: $FIRST_K"
    echo "  Reinit method: $METHOD"
    echo "  Layers to reinit: $LAYERS"
    
    # Submit the job
    sbatch slurm_launch_phase1_reinit.sh "$RUN_NAME" "$MODEL_SIZE" "$CHECKPOINT" "$FIRST_K" "$METHOD" $LAYERS
    
    echo ""
    echo "Job submitted. Check status with: squeue -u $USER"
fi

