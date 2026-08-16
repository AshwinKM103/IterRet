"""Checkpoint management for IterRet experiment runs.

A checkpoint is a directory containing:
  - `experience_bank.json`      -- ExperienceBank.to_dict() (experience_bank.py)
  - `metadata.json`             -- run identity, Hydra config snapshot, timing
  - `<sample_id>_ctc_graph.json` (optional, one per bootstrap conversation)

This mirrors the ad hoc `--save-graphs-dir` behavior already in
run_locomo_eval.py (experience_bank.json + per-conversation CTC graphs) but
adds the run metadata Hydra-managed experiments need to be reproducible:
which config produced this bank, when, and with what embedding backend (the
backend itself is never serialized -- ExperienceBank.load requires the
caller to pass one back in, since KeywordOverlapEmbeddingBackend vs
SentenceTransformerEmbeddingBackend is an environment choice, not run state).

Per .claude/rules/storage-invariants.md / CLAUDE.md: checkpoints are large
artifacts and belong under /mnt/ssd/users/durgesh/, never /home. Configure
via config.yaml's `checkpoint.root` key (see docs/config-guide.md). CheckpointManager
does not special-case or validate that path; it trusts the caller's config.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iterret.data.ctc_graph import CueTagContentGraph
from iterret.memory.experience_bank import EmbeddingBackend, ExperienceBank


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when a checkpoint is not found at the target directory.

    Raised by CheckpointManager.load(), load_graph(), or exists() when the
    expected checkpoint files (experience_bank.json or graph files) are missing.

    Example:
        >>> try:
        ...     manager = CheckpointManager("/path", "run1")
        ...     manager.load(backend)
        ... except CheckpointNotFoundError as e:
        ...     print(f"Checkpoint not found: {e}")
    """


@dataclass
class CheckpointMetadata:
    """Metadata associated with a checkpoint for reproducibility.

    Attributes:
        run_name: The run name (used as the checkpoint directory name).
        created_at: ISO 8601 timestamp of checkpoint creation.
        config: The full resolved Hydra config used for the run.
    """

    run_name: str
    created_at: str
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a JSON-compatible dict.

        Returns:
            Dict with keys "run_name", "created_at", "config".

        Example:
            >>> meta = CheckpointMetadata("run1", "2025-08-17T10:00:00", {"lr": 0.001})
            >>> d = meta.to_dict()
            >>> d["run_name"]  # "run1"
        """
        return {"run_name": self.run_name, "created_at": self.created_at, "config": self.config}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        """Deserialize metadata from a dict (e.g., from JSON).

        Args:
            data: Dict with keys "run_name", "created_at", "config". Missing
                keys are filled with defaults (empty/default values).

        Returns:
            CheckpointMetadata instance.

        Example:
            >>> d = {"run_name": "run1", "created_at": "2025-08-17T10:00:00", "config": {}}
            >>> meta = CheckpointMetadata.from_dict(d)
            >>> meta.run_name  # "run1"
        """
        return cls(
            run_name=data.get("run_name", "unknown"),
            created_at=data.get("created_at", ""),
            config=data.get("config", {}),
        )


class CheckpointManager:
    """Saves/loads an ExperienceBank (+ optional CTC graphs) plus run metadata.

    Checkpoint files are written under `root/run_name/`:
      - experience_bank.json: Serialized ExperienceBank (embeddings stripped)
      - metadata.json: Run metadata and resolved Hydra config
      - <sample_id>_ctc_graph.json: Optional per-conversation CTC graph snapshots
    """

    def __init__(self, root: str | Path, run_name: str) -> None:
        """Initialize a checkpoint manager for a given run.

        Args:
            root: Root directory for all checkpoints (typically /mnt/ssd/...).
            run_name: Name of this run (becomes the subdirectory).
        """
        self.root = Path(root)
        self.run_name = run_name
        self.run_dir = self.root / run_name

    def save(
        self,
        bank: ExperienceBank,
        *,
        config: dict[str, Any],
        graphs: dict[str, CueTagContentGraph] | None = None,
    ) -> Path:
        """Write a checkpoint (bank + metadata + optional graphs).

        Creates self.run_dir if needed. Writes:
          - experience_bank.json: Serialized ExperienceBank
          - metadata.json: Checkpoint metadata and resolved config
          - {sample_id}_ctc_graph.json: For each graph in the graphs dict

        Args:
            bank: The ExperienceBank to save (embeddings stripped).
            config: The full resolved Hydra config (for reproducibility and auditing).
            graphs: Optional dict of sample_id → CueTagContentGraph snapshots.

        Returns:
            The path to the created/updated run directory (self.run_dir).

        Example:
            >>> manager = CheckpointManager("/mnt/ssd", "run1")
            >>> path = manager.save(bank, config=hydra_config, graphs={"sample1": graph})
            >>> (path / "experience_bank.json").exists()  # True
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)

        bank.save(str(self.run_dir / "experience_bank.json"))

        metadata = CheckpointMetadata(
            run_name=self.run_name,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            config=config,
        )
        with (self.run_dir / "metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(metadata.to_dict(), fh, indent=2)

        for sample_id, graph in (graphs or {}).items():
            graph.save(str(self.run_dir / f"{sample_id}_ctc_graph.json"))

        return self.run_dir

    def load(
        self, backend: EmbeddingBackend, *, load_dir: str | Path | None = None
    ) -> tuple[ExperienceBank, CheckpointMetadata]:
        """Load a previously saved checkpoint (bank + metadata).

        Re-derives embeddings using the provided backend (embeddings are
        not serialized, so the caller must provide the backend).

        Args:
            backend: EmbeddingBackend to use for re-encoding experience entries.
            load_dir: Optional override for the directory to load from
                (default: self.run_dir; used when resuming under a different name).

        Returns:
            Tuple of (ExperienceBank, CheckpointMetadata).

        Raises:
            CheckpointNotFoundError: If no checkpoint exists at the target directory.
        """
        run_dir = Path(load_dir) if load_dir is not None else self.run_dir
        bank_path = run_dir / "experience_bank.json"
        if not bank_path.exists():
            raise CheckpointNotFoundError(
                f"No checkpoint at {run_dir} (missing {bank_path.name}). "
                "Check checkpoint.load_dir in your config."
            )

        bank = ExperienceBank.load(str(bank_path), backend)

        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as fh:
                metadata = CheckpointMetadata.from_dict(json.load(fh))
        else:
            metadata = CheckpointMetadata(run_name=run_dir.name, created_at="", config={})

        return bank, metadata

    def load_graph(
        self, sample_id: str, *, load_dir: str | Path | None = None
    ) -> CueTagContentGraph:
        """Load a saved CTC graph for a specific conversation.

        Args:
            sample_id: The sample ID (conversation identifier, e.g., "s1").
            load_dir: Optional override for the directory to load from
                (default: self.run_dir).

        Returns:
            The loaded CueTagContentGraph.

        Raises:
            CheckpointNotFoundError: If {sample_id}_ctc_graph.json doesn't exist.

        Example:
            >>> manager = CheckpointManager("/mnt/ssd", "run1")
            >>> graph = manager.load_graph("sample_1")
            >>> len(graph.cues) > 0  # True
        """
        run_dir = Path(load_dir) if load_dir is not None else self.run_dir
        graph_path = run_dir / f"{sample_id}_ctc_graph.json"
        if not graph_path.exists():
            raise CheckpointNotFoundError(f"No saved CTC graph for '{sample_id}' at {graph_path}.")
        return CueTagContentGraph.load(str(graph_path))

    def exists(self, *, load_dir: str | Path | None = None) -> bool:
        """Check if a checkpoint exists at the target directory.

        Checks for the presence of experience_bank.json (the checkpoint marker).

        Args:
            load_dir: Optional override for the directory to check
                (default: self.run_dir).

        Returns:
            True if experience_bank.json exists, False otherwise.

        Example:
            >>> manager = CheckpointManager("/mnt/ssd", "run1")
            >>> if manager.exists():
            ...     bank, meta = manager.load(backend)
        """
        run_dir = Path(load_dir) if load_dir is not None else self.run_dir
        return (run_dir / "experience_bank.json").exists()
