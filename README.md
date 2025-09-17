# ppt2

## NYU compute
```bash
source .venv/bin/activate
python ./scripts/phase0_nyu.py train_single run01 [node-number].hpc.nyu.edu
```

## Ai2 compute

```bash
# Make sure config builds successfully
python scripts/phase0_ai2.py dry_run phase0 ai2/jupiter

# Launch config on Beaker
python scripts/phase0_ai2_launch.py phase0 ai2/jupiter
```