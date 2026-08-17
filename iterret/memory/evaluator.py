"""Rubric-based evaluation of closed-loop steps.

Defines Planning and Reflection rubrics (retrieval quality rubric) and evaluates
individual steps via LLM judgment.
"""

from __future__ import annotations

import json
from typing import Literal

from ..models.llm_client import LLMClient
from ..state import SearchStep
from ..utils.json_utils import parse_json_object
from ..utils.metrics import sum_rubric_scores
from ..utils.prompts import RUBRIC_EVALUATOR_SYSTEM_PROMPT

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


def _rubrics_for(module: str) -> list[str]:
    return PLANNING_RUBRICS if module == "Planning" else REFLECTION_RUBRICS


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
    raw = llm.chat(RUBRIC_EVALUATOR_SYSTEM_PROMPT, user_prompt)
    parsed = parse_json_object(raw)
    rubrics = parsed.get("rubrics", {})
    if not isinstance(rubrics, dict):
        rubrics = {}
    total_score = sum_rubric_scores(rubrics, dimensions)
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
