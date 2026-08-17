# Experiment Module

Experiment tracking, checkpointing, and structured logging for IterRet runs.

## Key Classes

**`CheckpointManager`**

Save/load experiment state:

- `save_checkpoint(experience_bank, metadata, ctc_graph_snapshot)` → `<root>/<run_name>/`
  - `experience_bank.json`: Serialized bank (embeddings stripped, re-derived on load)
  - `metadata.json`: Run name, timestamp, full Hydra config
  - Optional: `<sample_id>_ctc_graph.json` (per-conversation snapshots)

- `load_checkpoint(checkpoint_dir, embedding_backend)` → ExperienceBank

**`ExperimentLogger`**

Unified logging: console + file + JSON-lines + optional wandb

- `setup_logging(config, hydra_run_dir)`: Configure iterret logger with three handlers
- `ExperimentLogger.create(wandb_config, ...)`: Optional wandb integration (degrades to no-op if disabled or init fails)
- Structured fields: `timestamp`, `level`, `service`, `message`, plus any `structured=` kwargs

## Usage

```bash
# Save checkpoint
python scripts/train.py run_name=my_run

# Load checkpoint for evaluation
python scripts/evaluate.py checkpoint.load_dir=/mnt/ssd/users/durgesh/iterret/checkpoints/my_run

# Override checkpoint root
python scripts/train.py checkpoint.root=/path/to/checkpoints
```

Set `ITERRET_CHECKPOINT_ROOT` in `.env` for persistent overrides.

## Dependencies

- `iterret.memory.experience_bank`: Bank serialization
- `wandb` (optional): Cloud tracking
- `hydra-core`, `python-dotenv`: Config + env loading
