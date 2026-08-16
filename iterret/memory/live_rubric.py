"""Real-time rubric-based scoring for closed-loop steps.

Evaluates Planning and Reflection decisions during episodes using COLM rubric
dimensions (COLM §1.4.2).
"""

from __future__ import annotations

import json
from typing import Any, cast

from ..models.llm_client import LLMClient
from ..utils.json_utils import parse_json_object

COLM_RUBRIC_DIMENSIONS = [
    "query specificity",
    "evidence coverage",
    "gap-gap redundancy",
]

_LIVE_RUBRIC_SYSTEM_PROMPT = """live_rubric_scoring
You are a rubric evaluator for a real-time memory deep search system (COLM).
Score the current Reflect step decision along rubric dimensions,
0-3 points each. Focus on: how targeted the refined query is (query
specificity), how much of the question the evidence answers (evidence
coverage), and whether the gaps list contains redundant/overlapping items
(gap-gap redundancy).
Reply as JSON: {"rubrics": {<dimension>: int, ...}, "reason": str}.
"""


def _sum_rubric_scores(rubrics: dict[str, object], dimensions: list[str]) -> int:
    """Sum per-dimension integer scores, ignoring missing or non-integer values."""
    total = 0
    for dimension in dimensions:
        try:
            value = rubrics.get(dimension, 0)
            total += int(cast(Any, value))
        except (TypeError, ValueError):
            continue
    return total


def score_reflect_step(
    original_query: str,
    evidence: list[str],
    gaps: list[str],
    next_query: str,
    llm: LLMClient,
) -> tuple[dict[str, int], str]:
    """Scores the current Reflect step along COLM rubric dimensions.

    Returns (per_dimension_scores, reason). COLM §1.4.2: rubric evaluator
    scores each Reflect step along query specificity, evidence coverage,
    and gap-gap redundancy.

    The LLM reports per-dimension scores; the total is summed here in code
    rather than trusted verbatim from the model, matching the pattern of
    iterret/evaluator.py.
    """
    user_prompt = json.dumps(
        {
            "original_query": original_query,
            "evidence": evidence,
            "gaps": gaps,
            "next_query": next_query,
            "rubric_dimensions": COLM_RUBRIC_DIMENSIONS,
        }
    )
    raw = llm.chat(_LIVE_RUBRIC_SYSTEM_PROMPT, user_prompt)
    parsed = parse_json_object(raw)
    rubrics = parsed.get("rubrics", {})
    if not isinstance(rubrics, dict):
        rubrics = {}
    per_dimension: dict[str, int] = {}
    for dimension in COLM_RUBRIC_DIMENSIONS:
        score = rubrics.get(dimension, 0)
        try:
            per_dimension[dimension] = int(score)
        except (TypeError, ValueError):
            per_dimension[dimension] = 0
    reason = str(
        parsed.get("reason", "") or ("unparseable live rubric reply" if not parsed else "")
    )
    return per_dimension, reason
