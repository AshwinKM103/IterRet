"""Experiment lifecycle: checkpointing (`checkpoint.py`) and run logging
(`exp_logging.py`).

Re-exports both submodules' public names here so `from iterret.experiment
import CheckpointManager` keeps working exactly as it did when this was a
single flat `iterret/experiment.py` module (scripts/train.py,
scripts/evaluate.py, and tests/test_experiment.py all import this way).
"""

from __future__ import annotations

from .checkpoint import CheckpointManager, CheckpointMetadata, CheckpointNotFoundError
from .exp_logging import ExperimentLogger, JsonLinesHandler, log_structured, setup_logging

__all__ = [
    "CheckpointManager",
    "CheckpointMetadata",
    "CheckpointNotFoundError",
    "ExperimentLogger",
    "JsonLinesHandler",
    "log_structured",
    "setup_logging",
]
