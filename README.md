## Setup

1. Create a conda environment:
   ```
   micromamba create -n scaling-v3 python=3.9
   ```
2. Install pip dependencies:
   ```
   conda activate scaling-v3
   pip install -r requirements.txt
   ```

## Typechecking
Run the command
```
mypy
```

## Formatting
Run the command
```
black --check src
```

## Best practices
- Any job that is submitted to slurm should probably be tracked with wandb.
  This makes is much easier to keep track of jobs and debug jobs that crash
  (since wandb logs things like memory usage).
