# ppt2

## NYU compute

### Interactive mode
```bash
source .venv/bin/activate
python ./scripts/phase0_nyu.py train_single run01 [node-number].hpc.nyu.edu
```

### Slurm batch mode
```bash
# Submit job with default run name (run01)
./submit_job.sh

# Submit job with custom run name
./submit_job.sh my_experiment_name

# Or submit directly with sbatch
sbatch slurm_launch.sh run01
```

## Ai2 compute

```bash
# Make sure config builds successfully
python scripts/phase0_ai2.py dry_run phase0 ai2/jupiter

# Launch config on Beaker
python scripts/phase0_ai2_launch.py phase0 ai2/jupiter
```