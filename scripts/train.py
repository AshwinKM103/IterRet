"""Hydra-managed offline stage: bootstrap an ExperienceBank from a LoCoMo
slice and checkpoint it.

This is IterRet's analog of "training" -- there's no gradient descent here
(R2-Mem's experience distillation, iterret/offline_pipeline.py, is what
`build_offline_memory` drives), but the shape is the same: consume data,
produce an artifact (the bank), checkpoint it, log metrics. Evaluation
against the held-out slice is `scripts/evaluate.py`, kept separate so a
bank can be evaluated repeatedly (different categories, different judge
settings) without rebuilding it.

Equivalent to calling `build_offline_memory` directly, but driven by Hydra
config instead of argparse, and checkpointed instead of only kept in memory
for the immediate eval that follows it.

Usage:
    python scripts/train.py
    python scripts/train.py llm=openai_compatible experiment.bootstrap_fraction=0.2
    python scripts/train.py wandb.enabled=true run_name=locomo_bootstrap_v1
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(
    Path(__file__).resolve().parents[1] / ".env"
)  # populates WANDB_*, ITERRET_* before Hydra resolves oc.env

from config.constants import DEFAULT_LLM_MAX_TOKENS  # noqa: E402
from iterret.data.locomo_data import (  # noqa: E402
    load_raw_locomo,
    resolve_locomo_path,
    split_bootstrap_eval,
)
from iterret.experiment import CheckpointManager  # noqa: E402
from iterret.experiment.exp_logging import ExperimentLogger, setup_logging  # noqa: E402
from iterret.memory.offline_pipeline import build_offline_memory  # noqa: E402
from iterret.models.llm_client import (  # noqa: E402
    LLMClient,
    MockLLMClient,
    OpenAICompatibleLLMClient,
)


def build_llm_client(llm_cfg: DictConfig) -> LLMClient:
    if llm_cfg.type == "mock":
        return MockLLMClient()
    if llm_cfg.type == "openai_compatible":
        return OpenAICompatibleLLMClient(
            base_url=llm_cfg.get("base_url"),
            model=llm_cfg.get("model"),
            api_key=llm_cfg.get("api_key", "not-needed"),
            max_tokens=llm_cfg.get("max_tokens", DEFAULT_LLM_MAX_TOKENS),
        )
    raise ValueError(f"Unknown llm.type '{llm_cfg.type}' (expected 'mock' or 'openai_compatible')")


def resolve_run_name(cfg: DictConfig) -> str:
    if cfg.run_name:
        return str(cfg.run_name)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    return Path(hydra_cfg.runtime.output_dir).name


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    run_name = resolve_run_name(cfg)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)

    logger = setup_logging(
        console_level=cfg.logging.console_level,
        file_level=cfg.logging.file_level,
        json_log_file=output_dir / cfg.logging.json_log_file,
    )
    exp_logger = ExperimentLogger.create(cfg.wandb, logger=logger, run_name=run_name)
    config_dict: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    exp_logger.log_config(config_dict)

    logger.info("starting train run '%s' (llm=%s)", run_name, cfg.llm.type)

    llm = build_llm_client(cfg.llm)
    locomo_path = resolve_locomo_path(cfg.experiment.locomo_path)
    logger.info("loading LoCoMo data from %s", locomo_path)
    raw_conversations = load_raw_locomo(locomo_path)
    bootstrap_raw, eval_raw = split_bootstrap_eval(
        raw_conversations, bootstrap_fraction=cfg.experiment.bootstrap_fraction
    )
    logger.info(
        "%d conversation(s) total: %d for bootstrap, %d held out",
        len(raw_conversations),
        len(bootstrap_raw),
        len(eval_raw),
    )

    categories = tuple(int(c) for c in cfg.experiment.categories)
    bank = build_offline_memory(
        bootstrap_raw,
        llm,
        max_turns=cfg.experiment.max_turns_per_conversation,
        categories=categories,
        bootstrap_questions_per_conv=cfg.experiment.bootstrap_questions_per_conv,
        max_iterations=cfg.experiment.max_iterations,
        max_chars_per_call=cfg.experiment.max_chars_per_call,
    )

    exp_logger.log_metrics(
        {
            "bank/planning_entries": len(bank.planning_bank),
            "bank/reflection_entries": len(bank.reflection_bank),
            "data/n_bootstrap_conversations": len(bootstrap_raw),
            "data/n_eval_conversations": len(eval_raw),
        }
    )

    manager = CheckpointManager(cfg.checkpoint.root, run_name)
    run_dir = manager.save(bank, config=config_dict)
    logger.info("checkpointed bank to %s", run_dir)
    exp_logger.log_artifact(run_dir, name=f"{run_name}-experience-bank")

    exp_logger.finish()


if __name__ == "__main__":
    main()
