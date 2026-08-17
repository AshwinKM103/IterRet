# IterRet Configuration

## Quick Start

```bash
# vLLM server (default, requires server running)
python scripts/train.py

# Mock LLM (offline testing)
python scripts/train.py llm.type=mock

# Enable WandB tracking
python scripts/train.py wandb.enabled=true run_name=my_run

# Custom LoCoMo dataset
python scripts/train.py experiment.locomo_path=/path/to/dataset.json

# Evaluate from checkpoint
python scripts/evaluate.py checkpoint.load_dir=/mnt/ssd/users/durgesh/iterret/checkpoints/my_run

# Change multiple settings
python scripts/train.py llm.type=mock experiment.max_iterations=10 wandb.enabled=true
```

## Configuration Structure

```
config/
  config.yaml   # Consolidated: llm, experiment, logging, checkpoint, wandb, run_name, seed
  sweep/
    locomo_grid.yaml  # Grid search parameters
```

Single `config.yaml` file — no composition, all settings in one place.

## Key Settings

**LLM** (`llm.*`)

- `llm.type`: `mock` (deterministic, offline) | `openai_compatible` (vLLM server, default)
- `llm.base_url`: vLLM server URL
- `llm.model`: Model name
- `llm.max_tokens`: Max output length (default: 1024)

**Experiment** (`experiment.*`)

- `experiment.locomo_path`: Dataset file (default: auto-search)
- `experiment.bootstrap_fraction`: Memory building data fraction (default: 0.1)
- `experiment.max_iterations`: Max reasoning steps (default: 5)
- `experiment.categories`: Eval categories [1,2,3,4] or add 5 for adversarial

**Checkpoint** (`checkpoint.*`)

- `checkpoint.load_dir`: Load from saved run
- `checkpoint.save_every_n_conversations`: Save cadence (0 = only at bootstrap end)

**Logging** (`logging.*`)

- `logging.console_level`: Terminal output (default: INFO)
- `logging.file_level`: File output (default: DEBUG)
- `logging.json_log_file`: JSON log file name (default: run.jsonl)

**WandB** (`wandb.*`)

- `wandb.enabled`: Enable tracking (default: true)
- `wandb.mode`: online | offline | disabled (default: online)

**Run** (`run_name`, `seed`)

- `run_name`: WandB run name & checkpoint dir (default: Hydra timestamp)
- `seed`: Random seed (default: 0)

## Checkpoints

Saved at `<checkpoint.root>/<run_name>/`:

```
experience_bank.json     # Learned memory bank
metadata.json            # Run info + full config snapshot
<id>_ctc_graph.json     # Optional CTC graphs per conversation
```

Restore with `checkpoint.load_dir=/path/to/run`.

## Sweeps

Edit `config/sweep/locomo_grid.yaml` to define grid search:

```yaml
grid:
  experiment.bootstrap_fraction: [0.05, 0.1, 0.2]
fixed_overrides:
  wandb.enabled: true
```

Run with `python scripts/sweep.py --sweep-config config/sweep/locomo_grid.yaml`.

## Output Directories

- Run logs: `outputs/<YYYY-MM-DD>/<HH-MM-SS>/`
- Multirun: `outputs/multirun/<YYYY-MM-DD>/<HH-MM-SS>/<job_num>/`
