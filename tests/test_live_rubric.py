from __future__ import annotations

import json

from iterret.memory.live_rubric import COLM_RUBRIC_DIMENSIONS, score_reflect_step
from iterret.models.llm_client import LLMClient


class StubLLMClient(LLMClient):
    """Deterministic LLM double for testing."""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def chat(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        return self.reply


def test_score_reflect_step_sums_dimension_scores() -> None:
    rubrics = {dimension: 2 for dimension in COLM_RUBRIC_DIMENSIONS}
    llm = StubLLMClient(json.dumps({"rubrics": rubrics, "reason": "test reason"}))

    scores, reason = score_reflect_step("q", [], [], "q'", llm)

    assert scores == {
        "query specificity": 2,
        "evidence coverage": 2,
        "gap-gap redundancy": 2,
    }
    assert reason == "test reason"


def test_score_reflect_step_handles_missing_dimensions() -> None:
    rubrics = {"query specificity": 3}
    llm = StubLLMClient(json.dumps({"rubrics": rubrics, "reason": "partial"}))

    scores, _ = score_reflect_step("q", [], [], "q'", llm)

    assert scores["query specificity"] == 3
    assert scores["evidence coverage"] == 0
    assert scores["gap-gap redundancy"] == 0


def test_score_reflect_step_handles_non_integer_dimensions() -> None:
    rubrics = {
        "query specificity": "not-a-number",
        "evidence coverage": 2,
        "gap-gap redundancy": 1,
    }
    llm = StubLLMClient(json.dumps({"rubrics": rubrics, "reason": "mixed"}))

    scores, _ = score_reflect_step("q", [], [], "q'", llm)

    assert scores["query specificity"] == 0
    assert scores["evidence coverage"] == 2
    assert scores["gap-gap redundancy"] == 1


def test_score_reflect_step_handles_unparseable_reply() -> None:
    llm = StubLLMClient("not json at all")

    scores, reason = score_reflect_step("q", [], [], "q'", llm)

    assert all(score == 0 for score in scores.values())
    assert reason == "unparseable live rubric reply"


def test_score_reflect_step_handles_empty_reply() -> None:
    llm = StubLLMClient("")

    scores, reason = score_reflect_step("q", [], [], "q'", llm)

    assert all(score == 0 for score in scores.values())
    assert reason == "unparseable live rubric reply"
