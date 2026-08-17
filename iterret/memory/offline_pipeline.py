from __future__ import annotations

import time
from typing import Any

from config.constants import CATEGORY_NAMES

from ..data.ctc_graph import CueTagContentGraph
from ..data.locomo_data import LoCoMoConversation, parse_conversation
from ..data.memory_builder import build_ctc_graph_from_dialogue
from ..models.llm_client import LLMClient
from ..state import DEFAULT_MAX_ITERATIONS, new_state
from ..utils.llm_judge import judge_answer
from ..utils.logger import get_logger
from ..utils.metrics import token_f1
from . import evaluator, learner
from .experience_bank import ExperienceBank, build_default_embedding_backend, empty_experience_bank
from .graph import build_graph

logger = get_logger(__name__)


def collect_trajectories(
    seed_questions: list[str],
    graph: CueTagContentGraph,
    llm: LLMClient,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> list[dict[str, Any]]:
    unguided_bank = empty_experience_bank()
    compiled = build_graph(llm, graph, unguided_bank)

    records: list[dict[str, Any]] = []
    for question in seed_questions:
        initial_state = new_state(question, max_iterations=max_iterations)
        try:
            final_state = compiled.invoke(initial_state, config={"recursion_limit": 25})
        except Exception as exc:  # noqa: BLE001 -- one bad bootstrap question shouldn't lose the rest
            logger.warning(f"Bootstrap trajectory for {question!r} failed, skipping: {exc}")
            continue
        records.append(
            {
                "original_query": question,
                "final_evidence": list(final_state.get("accumulated_evidence", [])),
                "steps": list(final_state.get("search_trajectory", [])),
            }
        )
    return records


def construct_experience_banks(
    trajectory_records: list[dict[str, Any]],
    llm: LLMClient,
    *,
    high_threshold: int = evaluator.HIGH_THRESHOLD,
    low_threshold: int = evaluator.LOW_THRESHOLD,
) -> ExperienceBank:
    bank = ExperienceBank(build_default_embedding_backend())

    for record in trajectory_records:
        original_query = record["original_query"]
        final_evidence = record["final_evidence"]

        for step in record["steps"]:
            if step["decision"] == "answer":
                continue  # rubrics target Planning/Reflection steps only (R2-Mem Sec. 4.2)

            score, reason_and_advice = evaluator.score_step(step, llm)
            quality = evaluator.classify(
                score, high_threshold=high_threshold, low_threshold=low_threshold
            )
            if quality == "discard":
                continue

            entry = learner.distill_experience(
                step,
                reason_and_advice,
                quality,
                step["module"],
                llm,
                original_query=original_query,
                accumulated_evidence=final_evidence,
            )
            bank.add_experience(
                entry["condition"], entry["situation"], entry["experience"], step["module"]
            )

    return bank


def build_offline_memory(
    bootstrap_raw: list,
    llm: LLMClient,
    *,
    max_turns: int | None,
    categories: tuple,
    bootstrap_questions_per_conv: int,
    max_iterations: int,
    max_chars_per_call: int,
) -> ExperienceBank:
    """Bootstrap an ExperienceBank from a slice of raw LoCoMo conversations.

    For each conversation: distills it into a CTC graph, collects unguided
    Planning/Reflection trajectories over its seed questions, then scores and
    distills those trajectories into experience entries (R2-Mem Sec. 4.2).
    """
    all_records: list[dict[str, Any]] = []
    for raw_conv in bootstrap_raw:
        conv = parse_conversation(
            raw_conv,
            max_turns=max_turns,
            categories=categories,
            max_questions=bootstrap_questions_per_conv,
        )
        logger.info(
            "[bootstrap] %s: distilling %d turn(s) into a CTC graph...",
            conv["sample_id"],
            len(conv["turns"]),
        )
        graph = build_ctc_graph_from_dialogue(
            conv["turns"], llm, max_chars_per_call=max_chars_per_call
        )

        seed_questions = [q["question"] for q in conv["questions"]]
        logger.info(
            "[bootstrap] %s: collecting %d unguided trajectory(ies)...",
            conv["sample_id"],
            len(seed_questions),
        )
        records = collect_trajectories(seed_questions, graph, llm, max_iterations=max_iterations)
        all_records.extend(records)

    logger.info(
        "[bootstrap] scoring %d trajectory(ies) and distilling experience...", len(all_records)
    )
    bank = construct_experience_banks(
        all_records,
        llm,
        high_threshold=evaluator.HIGH_THRESHOLD,
        low_threshold=evaluator.LOW_THRESHOLD,
    )
    logger.info(
        "[bootstrap] built experience bank: %d Planning + %d Reflection entries",
        len(bank.planning_bank),
        len(bank.reflection_bank),
    )
    return bank


def evaluate_conversation(
    conv: LoCoMoConversation,
    graph: CueTagContentGraph,
    llm: LLMClient,
    bank: ExperienceBank,
    *,
    max_iterations: int,
    run_judge: bool,
) -> list[dict[str, Any]]:
    """Run the closed-loop episode for every question in a conversation and score it.

    Returns one result dict per question (predicted answer, token-F1 against
    gold, optional LLM-judge correctness, iteration count, and wall time).
    """
    compiled = build_graph(llm, graph, bank)

    results = []
    for qa in conv["questions"]:
        start = time.time()
        initial_state = new_state(qa["question"], max_iterations=max_iterations)
        try:
            final_state = compiled.invoke(
                initial_state, config={"recursion_limit": 8 * max_iterations + 5}
            )
            predicted = final_state.get("final_answer") or ""
            iterations = final_state.get("iteration_count", 0)
        except Exception as exc:  # noqa: BLE001 -- one bad question shouldn't abort the whole eval run
            predicted, iterations = f"[error: {exc}]", 0
        elapsed = time.time() - start

        f1 = token_f1(predicted, qa["answer"])
        correct: bool | None = (
            judge_answer(qa["question"], qa["answer"], predicted, llm) if run_judge else None
        )

        results.append(
            {
                "sample_id": conv["sample_id"],
                "category": qa["category"],
                "question": qa["question"],
                "gold_answer": qa["answer"],
                "predicted_answer": predicted,
                "f1": f1,
                "correct": correct,
                "iterations": iterations,
                "elapsed_sec": elapsed,
            }
        )
    return results


def print_results_table(results: list[dict[str, Any]]) -> None:
    """Print a per-category and overall summary table of evaluation results to stdout."""
    by_category: dict[int, list[dict[str, Any]]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    def _avg(items: list[dict[str, Any]], key: str) -> float:
        return sum(item[key] for item in items) / len(items) if items else 0.0

    have_judge = any(r["correct"] is not None for r in results)
    header = f"{'Category':<14}{'N':>6}{'F1':>9}" + (f"{'Judge(%)':>11}" if have_judge else "")
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for category in sorted(by_category):
        items = by_category[category]
        name = CATEGORY_NAMES.get(category, str(category))
        line = f"{name:<14}{len(items):>6}{_avg(items, 'f1') * 100:>8.2f}%"
        if have_judge:
            judged = [i for i in items if i["correct"] is not None]
            judge_pct = (
                100 * sum(1 for i in judged if i["correct"]) / len(judged) if judged else 0.0
            )
            line += f"{judge_pct:>10.2f}%"
        print(line)
    print("-" * len(header))
    overall_judge = ""
    if have_judge:
        judged_all = [r for r in results if r["correct"] is not None]
        overall_pct = (
            100 * sum(1 for r in judged_all if r["correct"]) / len(judged_all)
            if judged_all
            else 0.0
        )
        overall_judge = f"{overall_pct:>10.2f}%"
    print(f"{'Overall':<14}{len(results):>6}{_avg(results, 'f1') * 100:>8.2f}%{overall_judge}")
    print("=" * len(header))

    avg_iter = _avg(results, "iterations")
    total_time = sum(r["elapsed_sec"] for r in results)
    print(
        f"\nAvg iterations/question: {avg_iter:.2f}  |  Total eval wall time: {total_time:.1f}s "
        f"over {len(results)} question(s)"
    )
