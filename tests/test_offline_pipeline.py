from __future__ import annotations

import json
from unittest.mock import patch

from iterret.memory import evaluator, offline_pipeline
from iterret.memory.experience_bank import ExperienceBank, KeywordOverlapEmbeddingBackend
from iterret.models.llm_client import LLMClient


class ScriptedLLMClient(LLMClient):
    """Returns replies in order; records every prompt it was asked to answer."""

    def __init__(self, replies: list) -> None:
        self._replies = list(replies)
        self.calls: list = []

    def chat(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self._replies.pop(0)


def _step(module: str, iteration: int, rubric_scores: dict | None = None) -> dict:
    return {
        "iteration": iteration,
        "module": module,
        "query_used": f"query {iteration}",
        "action_taken": "cue_to_tag",
        "found_summary": "found stuff",
        "decision": "retrieve" if module == "Planning" else "reflect",
        "rubric_scores": rubric_scores or {},
    }


def _rubric_reply(module: str, per_dimension_score: int, reason: str) -> str:
    dimensions = (
        evaluator.PLANNING_RUBRICS if module == "Planning" else evaluator.REFLECTION_RUBRICS
    )
    rubrics = {dimension: per_dimension_score for dimension in dimensions}
    return json.dumps({"rubrics": rubrics, "reason_and_advice": reason})


def _distill_reply(situation: str, experience: str) -> str:
    return json.dumps({"situation": situation, "experience": experience})


def _sample_records() -> list:
    # step 0: Planning, 4x3 = 12 -> "good"  -> distilled
    # step 1: Reflection, 4x1 = 4 -> "bad"  -> distilled
    # step 2: Planning, 4x2 = 8  -> "discard" -> NOT distilled
    return [
        {
            "original_query": "q",
            "final_evidence": ["ev1"],
            "steps": [_step("Planning", 0), _step("Reflection", 1), _step("Planning", 2)],
        }
    ]


def _scripted_replies_for_sample_records() -> list:
    return [
        _rubric_reply("Planning", 3, "great planning"),
        _distill_reply("s1", "e1"),
        _rubric_reply("Reflection", 1, "poor reflection"),
        _distill_reply("s2", "e2"),
        _rubric_reply("Planning", 2, "average planning"),
        # no distill reply queued for step 2: it must never be requested
    ]


def test_construct_experience_banks_only_distills_flagged_steps() -> None:
    llm = ScriptedLLMClient(_scripted_replies_for_sample_records())

    with patch(
        "iterret.memory.offline_pipeline.build_default_embedding_backend",
        return_value=KeywordOverlapEmbeddingBackend(),
    ):
        bank = offline_pipeline.construct_experience_banks(_sample_records(), llm)

    assert len(bank.planning_bank) == 1
    assert len(bank.reflection_bank) == 1
    assert bank.planning_bank[0]["situation"] == "s1"
    assert bank.reflection_bank[0]["situation"] == "s2"
    assert (
        len(llm.calls) == 5
    )  # 3 evaluator calls + 2 learner calls (discard skips the learner call)


def test_construct_experience_banks_calls_add_experience_only_for_flagged_steps() -> None:
    llm = ScriptedLLMClient(_scripted_replies_for_sample_records())
    real_add_experience = ExperienceBank.add_experience

    with (
        patch(
            "iterret.memory.offline_pipeline.build_default_embedding_backend",
            return_value=KeywordOverlapEmbeddingBackend(),
        ),
        patch.object(
            ExperienceBank, "add_experience", autospec=True, side_effect=real_add_experience
        ) as mocked,
    ):
        offline_pipeline.construct_experience_banks(_sample_records(), llm)

    assert mocked.call_count == 2  # step 0 (good) and step 1 (bad); step 2 (discard) never calls it


def test_construct_experience_banks_forwards_live_rubric_scores_to_learner() -> None:
    """Fix 2: live COLM-dimension rubric scores computed during Reflect
    (already stored on the trajectory step, previously discarded) must
    reach the learner's distillation call, not just the offline 8-dimension
    evaluator."""
    live_scores = {"query specificity": 3, "evidence coverage": 2, "gap-gap redundancy": 0}
    records = [
        {
            "original_query": "q",
            "final_evidence": ["ev1"],
            "steps": [_step("Reflection", 0, rubric_scores=live_scores)],
        }
    ]
    llm = ScriptedLLMClient(
        [
            _rubric_reply("Reflection", 3, "great reflection"),  # evaluator: 4x3=12 -> "good"
            _distill_reply("s1", "e1"),  # learner
        ]
    )

    with patch(
        "iterret.memory.offline_pipeline.build_default_embedding_backend",
        return_value=KeywordOverlapEmbeddingBackend(),
    ):
        offline_pipeline.construct_experience_banks(records, llm)

    distill_call = llm.calls[-1]
    sent = json.loads(distill_call[1])
    assert sent["live_rubric_scores"] == live_scores
