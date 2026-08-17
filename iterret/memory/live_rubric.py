"""Real-time rubric-based scoring for closed-loop steps.

Evaluates Planning and Reflection decisions during episodes using COLM rubric
dimensions (retrieval quality rubric).
"""

from __future__ import annotations

import json

from ..models.llm_client import LLMClient
from ..utils.json_utils import parse_json_object
from ..utils.prompts import LIVE_RUBRIC_SYSTEM_PROMPT

COLM_RUBRIC_DIMENSIONS = [
    "query specificity",
    "evidence coverage",
    "gap-gap redundancy",
]


def score_reflect_step(
    original_query: str,
    evidence: list[str],
    gaps: list[str],
    next_query: str,
    llm: LLMClient,
) -> tuple[dict[str, int], str]:
    """Scores the current Reflect step along COLM rubric dimensions.

    Returns (per_dimension_scores, reason). retrieval quality rubric: rubric evaluator
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
    raw = llm.chat(LIVE_RUBRIC_SYSTEM_PROMPT, user_prompt)
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
