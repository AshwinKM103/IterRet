"""Experiment logging: console + file + structured JSON + optional wandb.

Named `exp_logging` rather than `logging` deliberately -- every other module
in this package (llm_client.py, offline_pipeline.py, ...) does `import
logging` and expects the stdlib module; a same-directory `logging.py` would
not shadow it under Python 3's absolute-import default, but the name
collision is exactly the kind of thing that turns into a confusing bug the
next time someone adds a relative import, so it's avoided here.

Two independent concerns live in this module:
  - `setup_logging`: stdlib `logging.Logger` with a console handler and a
    file handler, plus a `JsonLinesHandler` that appends one JSON object per
    record to a `.jsonl` file (structured logs per .claude/rules/monitoring.md).
  - `ExperimentLogger`: a thin, optional wandb wrapper. When
    `wandb.enabled=false` (the default) or the `wandb` package/API key is
    unavailable, it degrades to a no-op that still logs metrics through the
    stdlib logger above, so scripts never need an `if wandb:` branch.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class JsonLinesHandler(logging.Handler):
    """Appends one JSON object per log record to a file (JSON Lines format).

    Formats each log record as a JSON object with timestamp, level, service,
    logger name, message, and any extra fields from record.structured. One
    JSON object per line, suitable for streaming processing.

    Never raises into the caller: broken log destinations degrade to silently
    dropping structured records rather than crashing the experiment run
    (console/file handlers still carry the message).

    Attributes:
        path: Path to the JSON Lines log file.

    Example:
        >>> handler = JsonLinesHandler("logs/run.jsonl")
        >>> logger.addHandler(handler)
        >>> logger.info("training started", extra={"structured": {"epoch": 1}})
        # Appends: {"timestamp": "...", "level": "INFO", ..., "epoch": 1}
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        """Handle a log record by writing it as JSON to the file.

        Args:
            record: The log record to handle.

        Note:
            Never raises; catches and silently logs errors via handleError().
        """
        try:
            payload = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
                "level": record.levelname,
                "service": "iterret",
                "logger": record.name,
                "message": record.getMessage(),
            }
            extra = getattr(record, "structured", None)
            if isinstance(extra, dict):
                payload.update(extra)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
        except Exception:
            self.handleError(record)


def setup_logging(
    *,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    json_log_file: str | Path,
    logger_name: str = "iterret",
) -> logging.Logger:
    """Configure and return a logger with console + file + JSON-lines handlers.

    Sets up three handlers:
    - Console (human-readable, level from console_level)
    - Plain-text file (level from file_level, same name as json_log_file.log)
    - JSON-lines file (always DEBUG, one JSON object per line)

    Idempotent: calling it twice for the same logger_name replaces handlers
    rather than stacking duplicates (needed for Hydra multirun).

    Args:
        console_level: Logging level for console output (default "INFO").
        file_level: Logging level for file output (default "DEBUG").
        json_log_file: Path to JSON-lines log file (plain-text version uses .log suffix).
        logger_name: Logger name to configure (default "iterret").

    Returns:
        The configured logging.Logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    json_path = Path(json_log_file)
    text_log_path = json_path.with_suffix(".log")
    file_handler = logging.FileHandler(text_log_path, encoding="utf-8")
    file_handler.setLevel(getattr(logging, file_level.upper()))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    json_handler = JsonLinesHandler(json_path)
    json_handler.setLevel(logging.DEBUG)
    logger.addHandler(json_handler)

    return logger


def log_structured(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Log a message with extra structured fields (for JSON-lines handlers).

    Args:
        logger: The logger instance.
        level: Log level (e.g., logging.INFO, logging.ERROR).
        message: The log message.
        **fields: Extra fields to include in structured record (merged into JSON).

    Example:
        >>> logger = logging.getLogger("iterret")
        >>> log_structured(logger, logging.INFO, "training step", epoch=1, loss=0.5)
        # JSON-lines handler appends: {"...", "epoch": 1, "loss": 0.5}
    """
    logger.log(level, message, extra={"structured": fields})


@dataclass
class ExperimentLogger:
    """Wraps wandb for experiment logging; degrades to no-op when unavailable.

    Provides a unified interface for logging configurations, metrics, and artifacts
    to both local structured logs and wandb (if enabled and available).

    Always construct via ExperimentLogger.create(), not directly. The factory
    method applies graceful degradation: if wandb is disabled, unavailable, or
    fails to initialize, all methods become no-ops but the local logger still
    records everything.

    Attributes:
        enabled: True if wandb is enabled and successfully initialized.
        logger: The local logging.Logger used for all structured logs.
        _run: The wandb.Run instance (None if disabled or unavailable).

    Example:
        >>> exp_logger = ExperimentLogger.create(cfg.wandb, logger=logger, run_name="run1")
        >>> exp_logger.log_config(hydra_config)
        >>> exp_logger.log_metrics({"loss": 0.5}, step=1)
        >>> exp_logger.finish()
    """

    enabled: bool
    logger: logging.Logger
    _run: Any = field(default=None, repr=False)

    @classmethod
    def create(
        cls, wandb_cfg: Any, *, logger: logging.Logger, run_name: str | None
    ) -> ExperimentLogger:
        enabled = bool(getattr(wandb_cfg, "enabled", False))
        if not enabled:
            return cls(enabled=False, logger=logger)

        try:
            import wandb
        except ImportError:
            logger.warning(
                "wandb.enabled=true but the `wandb` package is not installed "
                "(pip install -e '.[iterret]' after adding wandb -- see pyproject.toml). "
                "Falling back to file/console logging only."
            )
            return cls(enabled=False, logger=logger)

        try:
            run = wandb.init(
                project=getattr(wandb_cfg, "project", "iterret"),
                entity=getattr(wandb_cfg, "entity", None),
                name=run_name,
                mode=getattr(wandb_cfg, "mode", "online"),
                tags=list(getattr(wandb_cfg, "tags", []) or []),
                group=getattr(wandb_cfg, "group", None),
            )
        except Exception as exc:  # network/auth failures shouldn't abort the experiment
            logger.warning("wandb.init failed (%s); continuing without wandb.", exc)
            return cls(enabled=False, logger=logger)

        instance = cls(enabled=True, logger=logger)
        instance._run = run
        return instance

    def log_config(self, config_dict: dict[str, Any]) -> None:
        """Log the run configuration.

        Always logs to the local structured logger; also syncs to wandb if enabled.

        Args:
            config_dict: The full resolved Hydra config.
        """
        log_structured(self.logger, logging.INFO, "run config", **{"config": config_dict})
        if self.enabled and self._run is not None:
            self._run.config.update(config_dict, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        """Log metrics (e.g., accuracy, loss, F1 score).

        Always logs to the local structured logger; also syncs to wandb if enabled.

        Args:
            metrics: Dict of metric name → value.
            step: Optional step number (iteration or epoch).
        """
        log_structured(self.logger, logging.INFO, "metrics", metrics=metrics, step=step)
        if self.enabled and self._run is not None:
            self._run.log(metrics, step=step)

    def log_artifact(
        self, path: str | Path, *, name: str, artifact_type: str = "checkpoint"
    ) -> None:
        """Log an artifact (file or directory).

        Always logs to the local structured logger; also syncs to wandb if enabled
        and initialized. Handles both files and directories.

        Args:
            path: Path to the file or directory to log.
            name: Artifact name (displayed in wandb).
            artifact_type: Artifact category (default "checkpoint").

        Example:
            >>> exp_logger.log_artifact("/path/to/model.pkl", name="checkpoint_001")
        """
        log_structured(self.logger, logging.INFO, "artifact saved", path=str(path), name=name)
        if not self.enabled or self._run is None:
            return
        try:
            import wandb

            artifact = wandb.Artifact(name=name, type=artifact_type)
            if Path(path).is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
            self._run.log_artifact(artifact)
        except Exception as exc:
            self.logger.warning("wandb artifact logging failed (%s)", exc)

    def finish(self) -> None:
        """Finalize the run (sync to wandb if enabled).

        Safe to call even if wandb is disabled or not initialized.
        """
        if self.enabled and self._run is not None:
            self._run.finish()
