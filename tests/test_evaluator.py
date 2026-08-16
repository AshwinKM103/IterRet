from __future__ import annotations

import json

from iterret.memory.evaluator import (
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
    PLANNING_RUBRICS,
    REFLECTION_RUBRICS,
    classify,
    score_step,
)
from iterret.models.llm_client import LLMClient


class StubLLMClient(LLMClient):
    """Deterministic LLM double so evaluator tests never make a real call."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_call = None

    def chat(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        self.last_call = (system_prompt, user_prompt)
        return self.reply


def _step(module: str = "Planning") -> dict:
    return {
        "iteration": 0,
        "module": module,
        "query_used": "who is x",
        "action_taken": "cue_to_tag",
        "found_summary": "found nothing",
        "decision": "retrieve" if module == "Planning" else "reflect",
    }


def test_score_step_sums_rubric_dimensions() -> None:
    rubrics = {dimension: 3 for dimension in PLANNING_RUBRICS}
    llm = StubLLMClient(json.dumps({"rubrics": rubrics, "reason_and_advice": "solid plan"}))

    score, reason_and_advice = score_step(_step("Planning"), llm)

    assert score == 12
    assert reason_and_advice == "solid plan"


def test_score_step_ignores_missing_and_non_integer_dimensions() -> None:
    rubrics = {PLANNING_RUBRICS[0]: 2, PLANNING_RUBRICS[1]: "not-a-number"}
    llm = StubLLMClient(json.dumps({"rubrics": rubrics, "reason_and_advice": ""}))

    score, _ = score_step(_step("Planning"), llm)

    assert score == 2  # remaining two dimensions are absent -> treated as 0


def test_score_step_sends_reflection_rubric_for_reflection_steps() -> None:
    rubrics = {dimension: 1 for dimension in REFLECTION_RUBRICS}
    llm = StubLLMClient(json.dumps({"rubrics": rubrics, "reason_and_advice": "ok"}))

    score_step(_step("Reflection"), llm)

    sent_dimensions = json.loads(llm.last_call[1])["rubric_dimensions"]
    assert sent_dimensions == REFLECTION_RUBRICS


def test_score_step_defaults_to_zero_on_unparseable_reply() -> None:
    llm = StubLLMClient("not json at all")

    score, reason_and_advice = score_step(_step("Planning"), llm)

    assert score == 0
    assert reason_and_advice == "unparseable evaluator reply"


def test_classify_boundaries() -> None:
    assert HIGH_THRESHOLD == 10
    assert LOW_THRESHOLD == 5

    assert classify(HIGH_THRESHOLD) == "good"
    assert classify(12) == "good"
    assert classify(LOW_THRESHOLD) == "bad"
    assert classify(0) == "bad"
    for discarded_score in range(LOW_THRESHOLD + 1, HIGH_THRESHOLD):
        assert classify(discarded_score) == "discard"
