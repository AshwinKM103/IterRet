"""Learning experiences from trajectories for the experience bank.

Distills past closed-loop steps into training examples for the Planning and
Reflection experience banks (R2-Mem learning phase).
"""

from __future__ import annotations

import json

from ..models.llm_client import LLMClient
from ..state import SearchStep
from ..utils.json_utils import parse_json_object
from ..utils.prompts import EXPERIENCE_DISTILLATION_SYSTEM_PROMPT
from .experience_bank import ExperienceEntry, Module, planning_condition, reflection_condition

_HIGH_QUALITY_FRAMING = (
    "This step was judged HIGH-quality. Distill a best-practice pattern from it "
    "so future steps can replicate the behavior that earned the high score."
)
_LOW_QUALITY_FRAMING = (
    "This step was judged LOW-quality. Extract a corrective experience so future "
    "steps avoid repeating this mistake."
)


def distill_experience(
    step: SearchStep,
    reason_and_advice: str,
    quality: str,
    module: Module,
    llm: LLMClient,
    *,
    original_query: str,
    accumulated_evidence: list[str],
) -> ExperienceEntry:
    """Second-stage Learner call (R2-Mem's Exp_Planning/Exp_Reflection).

    ``condition`` and ``module`` are already known from the caller (the step
    being distilled), so the LLM is only asked for the two fields it can't
    already be told: ``situation`` and ``experience``.
    """
    framing = _HIGH_QUALITY_FRAMING if quality == "good" else _LOW_QUALITY_FRAMING
    system_prompt = EXPERIENCE_DISTILLATION_SYSTEM_PROMPT.format(framing=framing)
    user_prompt = json.dumps(
        {
            "module": module,
            "quality": quality,
            "query_used": step["query_used"],
            "action_taken": step["action_taken"],
            "found_summary": step["found_summary"],
            "decision": step["decision"],
            "diagnosed_reason_and_advice": reason_and_advice,
            "live_rubric_scores": step.get("rubric_scores", {}),
        }
    )
    raw = llm.chat(system_prompt, user_prompt)
    parsed = parse_json_object(raw)
    situation = str(parsed.get("situation") or "an ambiguous retrieval situation")
    experience = str(parsed.get("experience") or f"IF {situation} THEN proceed cautiously")

    condition = (
        planning_condition(step["query_used"])
        if module == "Planning"
        else reflection_condition(original_query, accumulated_evidence)
    )
    return ExperienceEntry(
        condition=condition,
        situation=situation,
        experience=experience,
        module=module,
        embedding=None,  # filled in by ExperienceBank.add_experience
    )
