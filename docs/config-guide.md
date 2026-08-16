# IterRet Hydra config guide

IterRet's experiment infrastructure (`config/`, `scripts/train.py`,
`scripts/evaluate.py`, `scripts/sweep.py`) is additive: `run_locomo_eval.py`
and its argparse CLI are untouched and remain the fastest path for a
one-off manual run. The Hydra path adds config composition, checkpointing,
and wandb/structured logging on top of the same underlying functions
(`build_offline_memory`, `evaluate_conversation`, `resolve_locomo_path`,
etc. -- `scripts/train.py` and `scripts/evaluate.py` import them directly
from `run_locomo_eval.py` rather than duplicating them).

## Layout

```
config/
  config.yaml              # root: composes the groups below, plus run_name/seed/wandb/hydra.run.dir
  llm/
    mock.yaml              # @package llm -- MockLLMClient, no deps
    openai_compatible.yaml # @package llm -- OpenAICompatibleLLMClient
  experiment/
    locomo_default.yaml    # @package experiment -- mirrors run_locomo_eval.py's argparse surface
  logging/
    default.yaml           # @package logging -- console/file levels, JSON-lines filename
  checkpoint/
    default.yaml           # @package checkpoint -- root dir, load_dir, save cadence
  sweep/
    locomo_grid.yaml        # NOT composed into defaults; read directly by scripts/sweep.py
```

Each `<group>/<name>.yaml` file's first line is `# @package <group>`, so its
keys land at `cfg.<group>.*` in the composed config rather than nested
under an extra `<group>.<name>` level.

## Composing and overriding

`config/config.yaml`'s `defaults` list picks `llm: mock`, `experiment:
locomo_default`, `logging: default`, `checkpoint: default` unless
overridden. Any leaf can be overridden from the CLI with a dotted path:

```bash
# swap in a real LLM server
python scripts/train.py llm=openai_compatible llm.base_url=http://localhost:8000/v1

# change one experiment leaf
python scripts/evaluate.py experiment.max_eval_conversations=5

# point at a specific checkpoint to evaluate
python scripts/evaluate.py checkpoint.load_dir=/mnt/ssd/users/durgesh/iterret/checkpoints/locomo_bootstrap_v1

# enable wandb (needs WANDB_API_KEY in .env or the environment)
python scripts/train.py wandb.enabled=true run_name=locomo_bootstrap_v1
```

To add a new named variant, drop a new file in a group directory (e.g.
`config/experiment/locomo_adversarial.yaml` with `categories: [1,2,3,4,5]`)
and select it with `experiment=locomo_adversarial`.

## Run identity and output layout

- `run_name` (root config): used as the wandb run name and the checkpoint
  subdirectory name (`checkpoint.root/<run_name>/`). Leave it `null` to
  fall back to Hydra's per-run timestamp directory name
  (`scripts/train.py`'s `resolve_run_name`).
- `hydra.run.dir` (single run) / `hydra.sweep.dir` + `hydra.sweep.subdir`
  (multirun): where Hydra writes its own `.hydra/` config snapshot, plus
  where `iterret.exp_logging.setup_logging` writes `run.log` / `run.jsonl`,
  and where `scripts/evaluate.py` writes `eval_results.json`.
- `hydra.job.chdir: false` is deliberate: IterRet's data loading
  (`resolve_locomo_path`, `_KNOWN_LOCOMO_PATHS`) uses paths relative to the
  invocation directory. Hydra's default of `chdir: true` would silently
  break that resolution.

## Environment variables (`.env`)

Copy `.env.example` to `.env` (gitignored) and fill in real values.
`scripts/train.py` / `scripts/evaluate.py` call `load_dotenv()` before
Hydra resolves the config, so `${oc.env:VAR,default}` interpolations in
`config/checkpoint/default.yaml` (`ITERRET_CHECKPOINT_ROOT`) and
`config/config.yaml` (`WANDB_ENTITY`) pick up `.env` values automatically.

| Variable                  | Used by                                   | Default when unset                           |
| ------------------------- | ----------------------------------------- | -------------------------------------------- |
| `ITERRET_LLM_BASE_URL`    | `iterret.llm_client.resolve_llm_base_url` | `http://localhost:8000/v1`                   |
| `ITERRET_LLM_MODEL`       | `iterret.llm_client.resolve_llm_model`    | `Qwen/Qwen3-4B-Instruct-2507`                |
| `WANDB_API_KEY`           | `wandb.init` (via the `wandb` package)    | none -- required if `wandb.enabled=true`     |
| `WANDB_ENTITY`            | `config/config.yaml`'s `wandb.entity`     | wandb's configured default entity            |
| `ITERRET_CHECKPOINT_ROOT` | `config/checkpoint/default.yaml`'s `root` | `/mnt/ssd/users/durgesh/iterret/checkpoints` |

## Checkpoints

`iterret.experiment.CheckpointManager(checkpoint.root, run_name)` writes:

```
<checkpoint.root>/<run_name>/
  experience_bank.json   # ExperienceBank.to_dict() -- embeddings stripped, re-derived on load
  metadata.json           # run_name, created_at, full resolved Hydra config
  <sample_id>_ctc_graph.json  # optional, one per bootstrap conversation (scripts/train.py doesn't pass any by default)
```

`ExperienceBank.load` requires an `EmbeddingBackend` instance from the
caller (embeddings aren't serialized -- which backend to use is an
environment choice, not run state); `scripts/evaluate.py` reconstructs one
via `build_default_embedding_backend()` (sentence-transformers if
available, keyword-overlap fallback otherwise -- same fallback chain as
`ExperienceBank`'s existing default).

Per `CLAUDE.md`'s storage invariants, `checkpoint.root` defaults to
`/mnt/ssd/users/durgesh/iterret/checkpoints`, never a path under `/home`.
Never commit anything written there.

## Sweeps

`config/sweep/locomo_grid.yaml` is a plain YAML file, not part of Hydra's
config groups -- `scripts/sweep.py` reads it directly and turns:

```yaml
grid:
  experiment.bootstrap_fraction: [0.05, 0.1, 0.2]
fixed_overrides:
  wandb.enabled: true
```

into a single Hydra multirun invocation:

```bash
python scripts/evaluate.py -m experiment.bootstrap_fraction=0.05,0.1,0.2 wandb.enabled=True
```

using Hydra's built-in basic sweeper (no custom grid-iteration code to
maintain). `python scripts/sweep.py --sweep-config <path> --dry-run` prints
the command without running it.

## Logging

`iterret.exp_logging.setup_logging` configures the `iterret` logger with
three handlers per run: console (`logging.console_level`), a plain-text
file (`logging.file_level`, `<json_log_file stem>.log`), and
`JsonLinesHandler` (always `DEBUG`, one JSON object per line at
`<hydra_run_dir>/<logging.json_log_file>`), matching the structured-logging
field set from `.claude/rules/monitoring.md`
(`timestamp`/`level`/`service`/`message` plus any `structured=` extras).

`iterret.exp_logging.ExperimentLogger.create(cfg.wandb, ...)` degrades to a
no-op whenever `wandb.enabled=false`, the `wandb` package isn't installed,
or `wandb.init` fails (bad/missing API key, no network) -- callers never
need an `if wandb_enabled:` branch; `log_config` / `log_metrics` /
`log_artifact` / `finish` are always safe to call.

## Known limitations / integration checklist

- `hydra-core`, `omegaconf`, `wandb`, `python-dotenv` are declared in the
  `iterret` optional-dependency group in the repo-root `pyproject.toml`
  (alongside the existing `langgraph`/`openai` pins) -- run
  `pip install -e '.[iterret]'` (or add them to your active env directly)
  before using `scripts/`.
- `scripts/train.py` / `scripts/evaluate.py` add the repo root to
  `sys.path` at import time so they can import `run_locomo_eval` as a
  plain module; this only works when invoked as `python scripts/train.py`
  from (or with cwd at) the `IterRet/` directory, matching how
  `run_locomo_eval.py` itself expects to be run.
- Multi-GPU / distributed training is out of scope -- there is no gradient
  training in this pipeline (see `scripts/train.py`'s module docstring for
  why "train" means "build the experience bank" here).
- `scripts/sweep.py` shells out to `scripts/evaluate.py -m`; it does not
  itself parallelize sweep points -- add a Hydra launcher plugin
  (`hydra-joblib-launcher`, `hydra-submitit-launcher`) via
  `hydra/launcher: <name>` in `config/config.yaml`'s defaults if sweep
  points should run concurrently.
