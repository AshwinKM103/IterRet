from __future__ import annotations

import json
import logging

import pytest
from iterret.data.ctc_graph import CueTagContentGraph
from iterret.experiment import CheckpointManager, CheckpointNotFoundError
from iterret.experiment.exp_logging import (
    ExperimentLogger,
    JsonLinesHandler,
    log_structured,
    setup_logging,
)
from iterret.memory.experience_bank import ExperienceBank, KeywordOverlapEmbeddingBackend


def _bank_with_entries() -> ExperienceBank:
    bank = ExperienceBank(KeywordOverlapEmbeddingBackend())
    bank.add_experience("condition A", "situation A", "experience A", "Planning")
    bank.add_experience("condition B", "situation B", "experience B", "Reflection")
    return bank


class TestCheckpointManager:
    def test_save_creates_bank_and_metadata_files(self, tmp_path):
        manager = CheckpointManager(tmp_path, "run_1")
        run_dir = manager.save(_bank_with_entries(), config={"seed": 0})

        assert run_dir == tmp_path / "run_1"
        assert (run_dir / "experience_bank.json").exists()
        assert (run_dir / "metadata.json").exists()

        with (run_dir / "metadata.json").open() as fh:
            metadata = json.load(fh)
        assert metadata["run_name"] == "run_1"
        assert metadata["config"] == {"seed": 0}
        assert metadata["created_at"]  # non-empty timestamp

    def test_load_round_trips_bank_contents(self, tmp_path):
        manager = CheckpointManager(tmp_path, "run_1")
        original = _bank_with_entries()
        manager.save(original, config={})

        backend = KeywordOverlapEmbeddingBackend()
        loaded, metadata = manager.load(backend)

        assert len(loaded.planning_bank) == 1
        assert len(loaded.reflection_bank) == 1
        assert loaded.planning_bank[0]["condition"] == "condition A"
        assert loaded.reflection_bank[0]["experience"] == "experience B"
        assert metadata.run_name == "run_1"

    def test_load_missing_checkpoint_raises(self, tmp_path):
        manager = CheckpointManager(tmp_path, "does_not_exist")
        with pytest.raises(CheckpointNotFoundError):
            manager.load(KeywordOverlapEmbeddingBackend())

    def test_exists_reflects_save_state(self, tmp_path):
        manager = CheckpointManager(tmp_path, "run_1")
        assert manager.exists() is False
        manager.save(_bank_with_entries(), config={})
        assert manager.exists() is True

    def test_save_writes_named_ctc_graphs_and_load_graph_reads_them_back(self, tmp_path):
        manager = CheckpointManager(tmp_path, "run_1")
        graph = CueTagContentGraph()
        graph.add_content("c1", "some episode text")

        manager.save(_bank_with_entries(), config={}, graphs={"conv_42": graph})

        assert (tmp_path / "run_1" / "conv_42_ctc_graph.json").exists()
        loaded_graph = manager.load_graph("conv_42")
        assert "c1" in loaded_graph.contents

    def test_load_missing_graph_raises(self, tmp_path):
        manager = CheckpointManager(tmp_path, "run_1")
        manager.save(_bank_with_entries(), config={})
        with pytest.raises(CheckpointNotFoundError):
            manager.load_graph("nonexistent_conv")

    def test_load_dir_override_reads_from_a_different_run(self, tmp_path):
        source = CheckpointManager(tmp_path, "source_run")
        source.save(_bank_with_entries(), config={})

        resuming = CheckpointManager(tmp_path, "resuming_run")
        loaded, metadata = resuming.load(
            KeywordOverlapEmbeddingBackend(), load_dir=tmp_path / "source_run"
        )
        assert metadata.run_name == "source_run"
        assert len(loaded.planning_bank) == 1


class TestLogging:
    def test_setup_logging_writes_console_file_and_json_handlers(self, tmp_path):
        json_log_file = tmp_path / "run.jsonl"
        logger = setup_logging(
            console_level="INFO",
            file_level="DEBUG",
            json_log_file=json_log_file,
            logger_name="iterret_test_setup",
        )
        assert len(logger.handlers) == 3
        assert (tmp_path / "run.log").exists()

    def test_json_lines_handler_emits_structured_record(self, tmp_path):
        json_path = tmp_path / "structured.jsonl"
        handler = JsonLinesHandler(json_path)
        logger = logging.getLogger("iterret_test_jsonlines")
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        log_structured(logger, logging.INFO, "metrics logged", accuracy=0.9, step=3)

        lines = json_path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["message"] == "metrics logged"
        assert record["accuracy"] == 0.9
        assert record["step"] == 3
        assert record["service"] == "iterret"

    def test_json_lines_handler_never_raises_on_unserializable_extra(self, tmp_path):
        json_path = tmp_path / "structured.jsonl"
        handler = JsonLinesHandler(json_path)
        logger = logging.getLogger("iterret_test_jsonlines_unserializable")
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # object() isn't JSON-serializable -- emit() must swallow the error.
        log_structured(logger, logging.INFO, "bad payload", obj=object())  # should not raise


class TestExperimentLogger:
    def test_disabled_config_produces_noop_logger(self, tmp_path):
        logger = logging.getLogger("iterret_test_explogger")

        class _Cfg:
            enabled = False

        exp_logger = ExperimentLogger.create(_Cfg(), logger=logger, run_name="test_run")
        assert exp_logger.enabled is False

        # These must not raise even though wandb was never initialized.
        exp_logger.log_config({"a": 1})
        exp_logger.log_metrics({"loss": 0.1})
        exp_logger.finish()

    def test_enabled_but_wandb_missing_falls_back_to_noop(self, tmp_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "wandb":
                raise ImportError("no wandb installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        logger = logging.getLogger("iterret_test_explogger_missing_wandb")

        class _Cfg:
            enabled = True
            project = "iterret"
            entity = None
            mode = "online"
            tags: list = []
            group = None

        exp_logger = ExperimentLogger.create(_Cfg(), logger=logger, run_name="test_run")
        assert exp_logger.enabled is False
