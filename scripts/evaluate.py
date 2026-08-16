"""Hydra-managed evaluation: load (or build) an ExperienceBank, evaluate it
against the held-out LoCoMo slice, log per-category metrics.

Two modes, chosen by whether `checkpoint.load_dir` is set:
  - set:   load a bank checkpointed by scripts/train.py, evaluate only.
  - unset: build the bank in-process first (same as run_locomo_eval.py's
           single-shot bootstrap+eval), then evaluate -- convenient for
           quick iteration without a separate train step.

Usage:
    python scripts/evaluate.py \
        checkpoint.load_dir=/mnt/ssd/users/durgesh/iterret/checkpoints/locomo_bootstrap_v1
    python scripts/evaluate.py wandb.enabled=true experiment.max_eval_conversations=5
"""

from __future__ import annotations

import json
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

from iterret.data.locomo_data import (  # noqa: E402
    CATEGORY_NAMES,
    load_raw_locomo,
    parse_conversation,
    split_bootstrap_eval,
)
from iterret.data.memory_builder import build_ctc_graph_from_dialogue  # noqa: E402
from iterret.experiment import CheckpointManager  # noqa: E402
from iterret.experiment.exp_logging import ExperimentLogger, setup_logging  # noqa: E402
from iterret.memory.experience_bank import (  # noqa: E402
    ExperienceBank,
    build_default_embedding_backend,
)
from run_locomo_eval import (  # noqa: E402
    build_offline_memory,
    evaluate_conversation,
    print_results_table,
    resolve_locomo_path,
)

from scripts.train import build_llm_client, resolve_run_name  # noqa: E402


def per_category_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    by_category: dict[int, list[dict[str, Any]]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    metrics: dict[str, float] = {}
    for category, items in by_category.items():
        name = CATEGORY_NAMES.get(category, str(category)).lower().replace("-", "_")
        metrics[f"eval/{name}/f1"] = sum(i["f1"] for i in items) / len(items)
        judged = [i for i in items if i["correct"] is not None]
        if judged:
            metrics[f"eval/{name}/judge_accuracy"] = sum(1 for i in judged if i["correct"]) / len(
                judged
            )

    if results:
        metrics["eval/overall/f1"] = sum(r["f1"] for r in results) / len(results)
        judged_all = [r for r in results if r["correct"] is not None]
        if judged_all:
            metrics["eval/overall/judge_accuracy"] = sum(
                1 for r in judged_all if r["correct"]
            ) / len(judged_all)
        metrics["eval/overall/avg_iterations"] = sum(r["iterations"] for r in results) / len(
            results
        )
    return metrics


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

    llm = build_llm_client(cfg.llm)
    locomo_path = resolve_locomo_path(cfg.experiment.locomo_path)
    logger.info("loading LoCoMo data from %s", locomo_path)
    raw_conversations = load_raw_locomo(locomo_path)
    bootstrap_raw, eval_raw = split_bootstrap_eval(
        raw_conversations, bootstrap_fraction=cfg.experiment.bootstrap_fraction
    )
    if cfg.experiment.max_eval_conversations is not None:
        eval_raw = eval_raw[: cfg.experiment.max_eval_conversations]

    manager = CheckpointManager(cfg.checkpoint.root, run_name)
    backend = build_default_embedding_backend()
    bank: ExperienceBank
    if cfg.checkpoint.load_dir:
        logger.info("loading checkpoint from %s", cfg.checkpoint.load_dir)
        bank, metadata = manager.load(backend, load_dir=cfg.checkpoint.load_dir)
        logger.info(
            "loaded bank from run '%s' (created %s)", metadata.run_name, metadata.created_at
        )
    else:
        logger.info("no checkpoint.load_dir set; building bank in-process from bootstrap slice")
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

    all_results: list[dict[str, Any]] = []
    categories = tuple(int(c) for c in cfg.experiment.categories)
    for raw_conv in eval_raw:
        conv = parse_conversation(
            raw_conv,
            max_turns=cfg.experiment.max_turns_per_conversation,
            categories=categories,
            max_questions=cfg.experiment.max_questions_per_conversation,
        )
        logger.info(
            "[eval] %s: distilling %d turn(s), answering %d question(s)",
            conv["sample_id"],
            len(conv["turns"]),
            len(conv["questions"]),
        )
        graph = build_ctc_graph_from_dialogue(
            conv["turns"], llm, max_chars_per_call=cfg.experiment.max_chars_per_call
        )
        conv_results = evaluate_conversation(
            conv,
            graph,
            llm,
            bank,
            max_iterations=cfg.experiment.max_iterations,
            run_judge=not cfg.experiment.skip_llm_judge,
        )
        all_results.extend(conv_results)

    print_results_table(all_results)
    metrics = per_category_metrics(all_results)
    exp_logger.log_metrics(metrics)
    logger.info("eval metrics: %s", metrics)

    results_path = output_dir / "eval_results.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": config_dict,
                "n_eval_conversations": len(eval_raw),
                "metrics": metrics,
                "results": all_results,
            },
            fh,
            indent=2,
        )
    logger.info("wrote full eval results to %s", results_path)
    exp_logger.log_artifact(results_path, name=f"{run_name}-eval-results", artifact_type="results")

    exp_logger.finish()


if __name__ == "__main__":
    main()
