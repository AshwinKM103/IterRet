"""Logging utilities for HyperMem pipeline.

Provides structured logging with configurable verbosity and format. Designed to
work with both development (console) and production (JSON, wandb) environments.

This module provides get_logger() for per-module logger creation and configure_logging()
for global logging setup. Supports:
- Multiple output destinations (console, file, wandb)
- Structured JSON logging for machine parsing
- Experiment tracking integration

Typical usage:

    from hypermem.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Starting pipeline", extra={"stage": 1})
"""

import json
import logging
import os
import re
import sys
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# Patterns for secret-like key=value pairs in free-text log messages, e.g.
# "api_key=sk-abc123", "token: eyJhbGciOi...", "password='hunter2'".
_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|token|password|secret)(\s*[:=]\s*)([^\s,;]+)"
)
_SECRET_REDACTION = r"\1\2[REDACTED]"

_ROOT_CONFIGURED = False


def sanitize(message: str) -> str:
    """Redact secret-like values (api_key, token, password, secret) from a log message.

    Matches ``<key><sep><value>`` where key is one of api_key/token/password/secret
    (case-insensitive, `-`/`_` variants) and sep is `:` or `=`. The value is replaced
    with ``[REDACTED]``; the key and separator are left intact for readability.

    Args:
        message: Raw log message that may contain secret-like key=value pairs.

    Returns:
        The message with any matched secret values redacted.
    """
    if not message:
        return message
    return _SECRET_PATTERN.sub(_SECRET_REDACTION, message)


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON.

    Emits: timestamp, level, component (logger name), message (sanitized),
    experiment_name, run_id. The latter two are read from the LogRecord if
    present (e.g. via `logging.LoggerAdapter` or `extra={...}`), defaulting
    to None so the shape is stable across records.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "component": record.name,
            "message": sanitize(record.getMessage()),
            "experiment_name": getattr(record, "experiment_name", None),
            "run_id": getattr(record, "run_id", None),
        }
        # Structured context attached via StructuredLogger (stage/task_id/metadata).
        # Only included when present so the JSON shape stays stable for callers
        # that use the plain get_logger() path without this context.
        stage = getattr(record, "stage", None)
        if stage is not None:
            payload["stage"] = stage
        task_id = getattr(record, "task_id", None)
        if task_id is not None:
            payload["task_id"] = task_id
        metadata = getattr(record, "metadata", None)
        if metadata is not None:
            payload["metadata"] = metadata
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(cfg: Any) -> None:
    """One-time root logger setup driven by a HyperMemConfig-shaped config.

    Adds a file handler (writing to ``cfg.logging.log_dir``) and, when
    ``cfg.logging.json_format`` is true, a JSONFormatter, to the root logger.
    Idempotent: calling this more than once does not add duplicate handlers.

    Args:
        cfg: A config object exposing ``cfg.logging.level``, ``cfg.logging.log_dir``,
            and ``cfg.logging.json_format`` (e.g. the composed HyperMemConfig, or
            any object/DictConfig with the same attribute shape).

    Note:
        Typed as ``Any`` rather than a Protocol because callers pass either the
        dataclass-based HyperMemConfig or an OmegaConf ``DictConfig`` built from
        Hydra composition; both satisfy the same duck-typed attribute shape but
        share no common base class.
    """
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED:
        return

    log_cfg = cfg.logging
    level = getattr(logging, str(getattr(log_cfg, "level", "INFO")).upper(), logging.INFO)
    log_dir = Path(getattr(log_cfg, "log_dir", "logs"))
    json_format = bool(getattr(log_cfg, "json_format", False))

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if json_format:
        formatter: logging.Formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"hypermem-{uuid.uuid4().hex[:8]}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _ROOT_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance with INFO-level console output.

    Creates or retrieves a logger and configures it with a console handler if
    not already configured. Safe to call multiple times; subsequent calls return
    the cached logger without re-adding handlers.

    Args:
        name: Logger identifier (typically __name__ of calling module). Used to
            organize logs by module hierarchy (e.g., "hypermem.extractors.episode").

    Returns:
        logging.Logger: Configured logger instance at INFO level with console
        output (stdout).

    Example:
        >>> from hypermem.utils.logger import get_logger
        >>> logger = get_logger(__name__)  # name = "hypermem.main.eval"
        >>> logger.info("Pipeline started")
        2026-01-15 10:30:45,123 - hypermem.main.eval - INFO - Pipeline started
        >>>
        >>> # Each module gets its own logger; logs are hierarchical
        >>> logger2 = get_logger("hypermem.extractors")
        >>> logger2.debug("Debug message")  # Suppressed (level=INFO by default)

    Note:
        Logs are output to stdout (not stderr) to avoid conflicts with stderr-based
        progress bars and error streams. Use configure_logging() for more control
        over formatting and output destinations.
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)

    return logger


class LogLevel(Enum):
    """Supported severities for :class:`StructuredLogger`, mapped onto stdlib levels."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARN = logging.WARNING
    ERROR = logging.ERROR


class _ConsoleContextFormatter(logging.Formatter):
    """Console formatter that appends structured context (stage/task_id/metadata).

    Produces human-readable lines for local development while still surfacing the
    same context fields the JSON formatter emits, e.g.:
    ``2026-08-15 10:30:45 - hypermem.main.stage1 - INFO - Starting extraction [stage=1 task_id=abc metadata={'n': 3}]``
    """

    def format(self, record: logging.LogRecord) -> str:
        record.msg = sanitize(record.getMessage())
        record.args = ()
        base = super().format(record)

        context_parts: list[str] = []
        stage = getattr(record, "stage", None)
        if stage is not None:
            context_parts.append(f"stage={stage}")
        task_id = getattr(record, "task_id", None)
        if task_id is not None:
            context_parts.append(f"task_id={task_id}")
        metadata = getattr(record, "metadata", None)
        if metadata:
            context_parts.append(f"metadata={metadata}")

        if context_parts:
            return f"{base} [{' '.join(context_parts)}]"
        return base


class StructuredLogger:
    """Structured logger with console (dev) and JSON (prod) output modes.

    Wraps the stdlib ``logging`` module so every call site can attach pipeline
    context (``stage``, ``task_id``, ``metadata``) without hand-formatting
    messages. Output format and level are configurable per-instance or via
    environment variables:

    - ``HYPERMEM_LOG_FORMAT``: ``"json"`` or ``"console"`` (default: ``"console"``).
    - ``HYPERMEM_LOG_LEVEL``: ``"DEBUG" | "INFO" | "WARN" | "ERROR"`` (default: ``"INFO"``).

    Example:
        >>> logger = StructuredLogger(__name__)
        >>> logger.info("Starting memory extraction", stage="1", task_id=task_id)
        >>> logger.debug("Processing conversation", metadata={"conv_id": conv.id})
        >>> logger.error("Failed to extract memory", exc_info=True)
    """

    def __init__(
        self,
        name: str,
        level: Optional["LogLevel"] = None,
        json_output: Optional[bool] = None,
    ) -> None:
        resolved_level = level or LogLevel[os.environ.get("HYPERMEM_LOG_LEVEL", "INFO").upper()]
        if json_output is None:
            json_output = os.environ.get("HYPERMEM_LOG_FORMAT", "console").lower() == "json"

        self._logger = logging.getLogger(name)
        self._logger.setLevel(resolved_level.value)

        # Reconfigure handlers if the requested format differs from what's attached,
        # so callers can switch modes (e.g. in tests) without leaking handlers.
        desired_formatter_cls = JSONFormatter if json_output else _ConsoleContextFormatter
        has_desired_handler = any(
            isinstance(getattr(h, "formatter", None), desired_formatter_cls)
            for h in self._logger.handlers
        )
        if not has_desired_handler:
            for h in list(self._logger.handlers):
                self._logger.removeHandler(h)
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(resolved_level.value)
            if json_output:
                handler.setFormatter(JSONFormatter())
            else:
                handler.setFormatter(
                    _ConsoleContextFormatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                )
            self._logger.addHandler(handler)

    def log(
        self,
        level: LogLevel,
        message: str,
        stage: Optional[str] = None,
        task_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        exc_info: bool = False,
        **kwargs: Any,
    ) -> None:
        """Emit a structured log entry with pipeline context.

        Extra keyword arguments are folded into ``metadata`` so call sites can
        pass ad-hoc fields (e.g. ``logger.info("done", conv_id=conv_id)``)
        without constructing a dict by hand.
        """
        if kwargs:
            metadata = {**(metadata or {}), **kwargs}
        self._logger.log(
            level.value,
            message,
            exc_info=exc_info,
            extra={"stage": stage, "task_id": task_id, "metadata": metadata},
        )

    def debug(self, msg: str, **kwargs: Any) -> None:
        self.log(LogLevel.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self.log(LogLevel.INFO, msg, **kwargs)

    def warn(self, msg: str, **kwargs: Any) -> None:
        self.log(LogLevel.WARN, msg, **kwargs)

    def error(self, msg: str, exc_info: bool = False, **kwargs: Any) -> None:
        self.log(LogLevel.ERROR, msg, exc_info=exc_info, **kwargs)

