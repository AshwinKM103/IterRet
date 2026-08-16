from __future__ import annotations

import json

from iterret.learner import distill_experience
from iterret.llm_client import LLMClient


class StubLLMClient(LLMClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_call = None

    def chat(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        self.last_call = (system_prompt, user_prompt)
        return self.reply


def _step(module: str, query: str = "who is x") -> dict:
    return {
        "iteration": 0,
        "module": module,
        "query_used": query,
        "action_taken": "cue_to_tag",
        "found_summary": "found nothing",
        "decision": "retrieve" if module == "Planning" else "reflect",
    }


def test_distill_experience_uses_planning_condition_for_planning_module() -> None:
    llm = StubLLMClient(json.dumps({"situation": "s", "experience": "e"}))

    entry = distill_experience(
        _step("Planning", query="who wrote hamlet"),
        "reason",
        "good",
        "Planning",
        llm,
        original_query="original",
        accumulated_evidence=["ev"],
    )

    assert entry["condition"] == "who wrote hamlet"  # planning_condition == the refined query
    assert entry["situation"] == "s"
    assert entry["experience"] == "e"
    assert entry["module"] == "Planning"


def test_distill_experience_uses_reflection_condition_for_reflection_module() -> None:
    llm = StubLLMClient(json.dumps({"situation": "s", "experience": "e"}))

    entry = distill_experience(
        _step("Reflection"),
        "reason",
        "bad",
        "Reflection",
        llm,
        original_query="original query",
        accumulated_evidence=["ev1", "ev2"],
    )

    assert entry["condition"] == "original query | evidence so far: ev1; ev2"
    assert entry["module"] == "Reflection"


def test_distill_experience_falls_back_when_llm_reply_is_unparseable() -> None:
    llm = StubLLMClient("not json")

    entry = distill_experience(
        _step("Planning"),
        "reason",
        "bad",
        "Planning",
        llm,
        original_query="q",
        accumulated_evidence=[],
    )

    assert entry["situation"] == "an ambiguous retrieval situation"
    assert "an ambiguous retrieval situation" in entry["experience"]


def test_distill_experience_framing_differs_by_quality() -> None:
    good_llm = StubLLMClient(json.dumps({"situation": "s", "experience": "e"}))
    distill_experience(
        _step("Planning"),
        "r",
        "good",
        "Planning",
        good_llm,
        original_query="q",
        accumulated_evidence=[],
    )

    bad_llm = StubLLMClient(json.dumps({"situation": "s", "experience": "e"}))
    distill_experience(
        _step("Planning"),
        "r",
        "bad",
        "Planning",
        bad_llm,
        original_query="q",
        accumulated_evidence=[],
    )

    assert "HIGH-quality" in good_llm.last_call[0]
    assert "LOW-quality" in bad_llm.last_call[0]
