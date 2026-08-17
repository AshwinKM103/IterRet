# IterRet

Iterative Retrieval for Episodic Task Solving. A closed-loop memory reconstruction system for LLM agents.

## Quick Start

```bash
# Install
pip install -e '.[iterret]'

# Run with mock LLM (offline)
python scripts/train.py llm.type=mock

# Run with vLLM server
python scripts/train.py  # requires vLLM running at http://localhost:8000/v1
```

See `docs/config-guide.md` for full configuration.

## Architecture

IterRet implements a closed-loop controller that reasons over a Cue-Tag-Content (CTC) memory graph:

1. **Planning**: Extract query cues, activate tags, select traversal actions
2. **Retrieval**: Follow active actions through the graph (cue→tag→content)
3. **Routing/Reflection**: Merge new content, judge gap resolution, refine or answer
4. **Synthesis**: Generate final answer from accumulated evidence

## Modules

- **`iterret/experiment/`**: Checkpointing and structured logging
- **`iterret/models/`**: LLM client abstraction (OpenAI-compatible, mock)
- **`iterret/memory/`**: Experience banks, graph retrieval, state management
- **`iterret/data/`**: CTC graph construction, dataset loading (LoCoMo)
- **`iterret/utils/`**: Metrics, JSON parsing, LLM judging

## Entry Points

- **`scripts/train.py`**: Build experience bank from bootstrap data (Hydra config)
- **`scripts/evaluate.py`**: Evaluate on full dataset from checkpoint
- **`scripts/sweep.py`**: Grid search over hyperparameters

## Configuration

All settings in `config/config.yaml` (consolidated). Override via CLI:

```bash
python scripts/train.py \
  llm.type=openai_compatible \
  experiment.max_iterations=10 \
  wandb.enabled=true \
  run_name=my_run
```

Key sections: `llm`, `experiment`, `checkpoint`, `logging`, `wandb`, `run_name`, `seed`.

## Dependencies

- `torch >=2.7,<2.9`
- `sentence-transformers` (optional, for neural embeddings)
- `openai` (optional, for OpenAI-compatible LLM servers)
- `hydra-core`, `wandb`, `python-dotenv` (for experiment infrastructure)

Install with: `pip install -e '.[iterret]'` or `pip install -e '.[iterret,hypermem,memocr]'` for full C-AIMMS.
