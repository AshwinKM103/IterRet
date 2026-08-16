# Experiment Module

Experiment tracking, checkpointing, and structured logging for IterRet runs.

## Overview

The experiment module provides infrastructure for reproducible experiment execution:

1. **Checkpointing** (`checkpoint.py`): Save/load experiment state (experience banks, metadata, graph snapshots)
2. **Logging** (`exp_logging.py`): Structured logging (console, file, JSON-lines) and wandb integration

This is an optional layer on top of the core IterRet functions; `run_locomo_eval.py` (argparse-based) remains the fastest path for one-off manual runs. The experiment infrastructure adds Hydra config composition, checkpointing, and structured metrics.

## Key Classes

### `CheckpointManager` (`checkpoint.py`)

Saves and loads experiment checkpoints:

- **`save_checkpoint(experience_bank, metadata, ctc_graph_snapshot)`**: Write to `<root>/<run_name>/`:
  - `experience_bank.json`: Serialized bank (embeddings stripped, re-derived on load)
  - `metadata.json`: Run name, created timestamp, full Hydra config
  - Optional: `<sample_id>_ctc_graph.json` (per-conversation graph snapshots)

- **`load_checkpoint(checkpoint_dir, embedding_backend)`**: Reconstruct ExperienceBank from disk
  - Embeddings are NOT serialized (environment choice, not run state)
  - Caller must provide `EmbeddingBackend` (recovered via `build_default_embedding_backend()`)

### `ExperimentLogger` (`exp_logging.py`)

Unified logging: console + file + JSON-lines + optional wandb:

- **`setup_logging(config, hydra_run_dir)`**: Configure `iterret` logger with three handlers:
  - Console: human-readable (level from config)
  - Plain-text file: level from config
  - JSON-lines: always DEBUG, one structured JSON object per line

- **`ExperimentLogger.create(wandb_config, ...)`**: Optional wandb integration:
  - Degrades to no-op if `wandb.enabled=false`, package missing, or init fails
  - Callers never need `if wandb_enabled:` branches; all methods safe to call

- **Structured fields** (per `.claude/rules/monitoring.md`): `timestamp`, `level`, `service`, `message`, plus any `structured=` kwargs

## Data Storage

Checkpoint root defaults to `/mnt/ssd/users/durgesh/iterret/checkpoints` (never `/home`, per `CLAUDE.md` storage invariants). Configure via `config.yaml`'s `checkpoint` section (see `docs/config-guide.md`):

```bash
python scripts/train.py checkpoint.root=/path/to/checkpoints
python scripts/evaluate.py checkpoint.load_dir=/path/to/saved/checkpoint
```

or set `ITERRET_CHECKPOINT_ROOT` in `.env` for the root, and use CLI overrides for specific runs.

## Dependencies

- `iterret.memory.experience_bank`: ExperienceBank serialization
- `wandb` (optional): For cloud experiment tracking
- `hydra-core`, `python-dotenv` (already required by experiment infrastructure)
