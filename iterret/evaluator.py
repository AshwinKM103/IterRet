from __future__ import annotations

import json
from typing import Literal

from .json_utils import parse_json_object
from .llm_client import LLMClient
from .state import SearchStep

PLANNING_RUBRICS = [
    "Info Needs Coverage",
    "Info Needs Non-Redundancy",
    "Tool-Info Alignment",
    "Planning Efficiency",
]

REFLECTION_RUBRICS = [
    "Sufficiency Judgment Accuracy",
    "Minimal Sufficiency Recognition",
    "Follow-up Query Quality",
    "Answer Completeness Awareness",
]

HIGH_THRESHOLD = 10  # total rubric score (0-12) at/above which a step is distilled as high-quality
LOW_THRESHOLD = (
    5  # total rubric score at/below which a step is distilled as low-quality (corrective)
)

Quality = Literal["good", "bad", "discard"]

_EVALUATOR_SYSTEM_PROMPT = """rubric_evaluation
You are an expert evaluator for an AI memory deep search system (R2-Mem style).
Score the given step against its module's rubric dimensions, 0-3 points
each, and give a short reason plus actionable advice for future steps.
Reply as JSON: {"module": str, "rubrics": {<dimension>: int, ...}, "reason_and_advice": str}.
"""


def _rubrics_for(module: str) -> list[str]:
    return PLANNING_RUBRICS if module == "Planning" else REFLECTION_RUBRICS


def _sum_rubric_scores(rubrics: dict[str, object], dimensions: list[str]) -> int:
    total = 0
    for dimension in dimensions:
        try:
            total += int(rubrics.get(dimension, 0))
        except (TypeError, ValueError):
            continue
    return total


def score_step(step: SearchStep, llm: LLMClient) -> tuple[int, str]:
    """Scores one trajectory step against its module's rubric.

    Returns ``(total_score, reason_and_advice)``. The LLM reports per-dimension
    scores; the total is summed here in code rather than trusted verbatim
    from the model, matching R2-Mem's Evaluator contract (judge_model.py:
    per-step "rubrics" dict summed into a 0-12 total_score).
    """
    dimensions = _rubrics_for(step["module"])
    user_prompt = json.dumps(
        {
            "module": step["module"],
            "rubric_dimensions": dimensions,
            "query_used": step["query_used"],
            "action_taken": step["action_taken"],
            "found_summary": step["found_summary"],
            "decision": step["decision"],
        }
    )
    raw = llm.chat(_EVALUATOR_SYSTEM_PROMPT, user_prompt)
    parsed = parse_json_object(raw)
    rubrics = parsed.get("rubrics", {})
    if not isinstance(rubrics, dict):
        rubrics = {}
    total_score = _sum_rubric_scores(rubrics, dimensions)
    reason_and_advice = str(
        parsed.get("reason_and_advice", "") or ("unparseable evaluator reply" if not parsed else "")
    )
    return total_score, reason_and_advice


def classify(
    score: int, *, high_threshold: int = HIGH_THRESHOLD, low_threshold: int = LOW_THRESHOLD
) -> Quality:
    if score >= high_threshold:
        return "good"
    if score <= low_threshold:
        return "bad"
    return "discard"
