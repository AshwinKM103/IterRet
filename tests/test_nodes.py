from __future__ import annotations

import json

from iterret.data.ctc_graph import CueTagContentGraph
from iterret.memory.experience_bank import ExperienceBank, KeywordOverlapEmbeddingBackend
from iterret.memory.nodes import extract_cue_terms, retrieve_node, router_node
from iterret.models.llm_client import LLMClient
from iterret.state import IterRetState


class MockLLMClientForRouter(LLMClient):
    """Mock LLM client that returns a configurable routing decision."""

    def __init__(self, action: str = "retrieve", reasoning: str = "mock reasoning") -> None:
        self.action = action
        self.reasoning = reasoning
        self.calls: list = []

    def chat(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        self.calls.append((system_prompt, user_prompt))
        return json.dumps({"action": self.action, "reasoning": self.reasoning})


class FailingLLMClient(LLMClient):
    """LLM client that raises an exception."""

    def __init__(self) -> None:
        self.calls: list = []

    def chat(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        self.calls.append((system_prompt, user_prompt))
        raise RuntimeError("LLM service unavailable")


def test_router_node_hard_stop_max_iterations() -> None:
    """Hard-stop: max iterations reached forces 'answer' without LLM call."""
    llm = MockLLMClientForRouter()
    state: IterRetState = {
        "original_query": "test query",
        "iteration_count": 5,
        "max_iterations": 5,
        "information_gaps": ["gap 1", "gap 2"],
        "accumulated_evidence": ["ev 1"],
        "consecutive_stuck_reflects": 0,
    }

    result = router_node(state, llm)

    assert result["_router_decision"] == "answer"
    assert len(llm.calls) == 0  # No LLM call made


def test_router_node_hard_stop_no_gaps() -> None:
    """Hard-stop: empty gaps forces 'answer' without LLM call."""
    llm = MockLLMClientForRouter()
    state: IterRetState = {
        "original_query": "test query",
        "iteration_count": 2,
        "max_iterations": 5,
        "information_gaps": [],
        "accumulated_evidence": ["ev 1", "ev 2"],
        "consecutive_stuck_reflects": 0,
    }

    result = router_node(state, llm)

    assert result["_router_decision"] == "answer"
    assert len(llm.calls) == 0  # No LLM call made


def test_router_node_hard_stop_stuck_reflects() -> None:
    """Hard-stop: max consecutive stuck reflects forces 'answer' without LLM call."""
    llm = MockLLMClientForRouter()
    state: IterRetState = {
        "original_query": "test query",
        "iteration_count": 2,
        "max_iterations": 5,
        "information_gaps": ["gap 1"],
        "accumulated_evidence": ["ev 1"],
        "consecutive_stuck_reflects": 2,  # DEFAULT_MAX_STUCK_REFLECTS = 2
    }

    result = router_node(state, llm)

    assert result["_router_decision"] == "answer"
    assert len(llm.calls) == 0  # No LLM call made


def test_router_node_llm_decides_retrieve() -> None:
    """When hard-stops are false, LLM's 'retrieve' decision is used."""
    llm = MockLLMClientForRouter(action="retrieve", reasoning="more gaps to explore")
    state: IterRetState = {
        "original_query": "test query",
        "iteration_count": 1,
        "max_iterations": 5,
        "information_gaps": ["gap 1", "gap 2"],
        "accumulated_evidence": ["ev 1"],
        "consecutive_stuck_reflects": 0,
    }

    result = router_node(state, llm)

    assert result["_router_decision"] == "retrieve"
    assert len(llm.calls) == 1  # LLM was called once


def test_router_node_llm_decides_answer() -> None:
    """When hard-stops are false, LLM's 'answer' decision is used."""
    llm = MockLLMClientForRouter(action="answer", reasoning="sufficient evidence gathered")
    state: IterRetState = {
        "original_query": "test query",
        "iteration_count": 2,
        "max_iterations": 5,
        "information_gaps": ["gap 1"],
        "accumulated_evidence": ["ev 1", "ev 2"],
        "consecutive_stuck_reflects": 0,
    }

    result = router_node(state, llm)

    assert result["_router_decision"] == "answer"
    assert len(llm.calls) == 1  # LLM was called once


def test_router_node_llm_invalid_action_defaults_to_retrieve() -> None:
    """LLM returns invalid action; falls back to 'retrieve'."""
    llm = MockLLMClientForRouter(action="invalid_action")
    state: IterRetState = {
        "original_query": "test query",
        "iteration_count": 1,
        "max_iterations": 5,
        "information_gaps": ["gap 1"],
        "accumulated_evidence": ["ev 1"],
        "consecutive_stuck_reflects": 0,
    }

    result = router_node(state, llm)

    assert result["_router_decision"] == "retrieve"
    assert len(llm.calls) == 1  # LLM was called


def test_router_node_llm_failure_falls_back_to_retrieve() -> None:
    """LLM call fails; falls back to 'retrieve'."""
    llm = FailingLLMClient()
    state: IterRetState = {
        "original_query": "test query",
        "iteration_count": 1,
        "max_iterations": 5,
        "information_gaps": ["gap 1"],
        "accumulated_evidence": ["ev 1"],
        "consecutive_stuck_reflects": 0,
    }

    result = router_node(state, llm)

    assert result["_router_decision"] == "retrieve"
    assert len(llm.calls) == 1  # LLM was called (and failed)


def test_router_node_preserves_state() -> None:
    """Router node preserves all state fields and only adds _router_decision."""
    llm = MockLLMClientForRouter(action="answer")
    state: IterRetState = {
        "original_query": "test query",
        "iteration_count": 1,
        "max_iterations": 5,
        "information_gaps": ["gap 1"],
        "accumulated_evidence": ["ev 1"],
        "consecutive_stuck_reflects": 0,
    }

    result = router_node(state, llm)

    # All original fields should be preserved
    assert result["original_query"] == "test query"
    assert result["iteration_count"] == 1
    assert result["max_iterations"] == 5
    assert result["information_gaps"] == ["gap 1"]
    assert result["accumulated_evidence"] == ["ev 1"]
    assert result["consecutive_stuck_reflects"] == 0
    # New decision field added
    assert "_router_decision" in result


# ----------------------------------------------------------------------
# Fix 4: experience-bank advice reaching router_node's LLM decision
# ----------------------------------------------------------------------


class RaisingBank:
    """Fake ExperienceBank whose .retrieve() always raises, to exercise the
    try/except fallback path when advice retrieval itself fails."""

    def retrieve(self, condition: str, situation: str, module: str, k: int = 3) -> list:
        raise RuntimeError("bank backend unavailable")


def _mock_router_llm() -> MockLLMClientForRouter:
    return MockLLMClientForRouter(action="answer", reasoning="advice says stop")


def test_router_node_injects_bank_advice_into_llm_prompt() -> None:
    """When a bank is supplied and there's no hard-stop, its retrieved
    experience advice must reach the router's LLM decision prompt."""
    bank = ExperienceBank(KeywordOverlapEmbeddingBackend())
    bank.add_experience(
        condition="test query | evidence so far: ev 1",
        situation="a query situation",
        experience="IF evidence is thin THEN keep retrieving",
        module="Reflection",
    )
    llm = _mock_router_llm()
    state: IterRetState = {
        "original_query": "test query",
        "iteration_count": 1,
        "max_iterations": 5,
        "information_gaps": ["gap 1"],
        "accumulated_evidence": ["ev 1"],
        "consecutive_stuck_reflects": 0,
    }

    result = router_node(state, llm, bank=bank)

    assert result["_router_decision"] == "answer"
    # the final (decision) call's user_prompt must carry the retrieved advice text
    final_user_prompt = json.loads(llm.calls[-1][1])
    assert "experience_advice" in final_user_prompt
    assert "keep retrieving" in final_user_prompt["experience_advice"]


def test_router_node_hard_stop_never_touches_bank() -> None:
    """Hard-stop conditions must still bypass the LLM entirely, so a bank
    (even one that would raise) must never be consulted."""
    bank = RaisingBank()
    llm = _mock_router_llm()
    state: IterRetState = {
        "original_query": "test query",
        "iteration_count": 5,
        "max_iterations": 5,
        "information_gaps": ["gap 1"],
        "accumulated_evidence": ["ev 1"],
        "consecutive_stuck_reflects": 0,
    }

    result = router_node(state, llm, bank=bank)

    assert result["_router_decision"] == "answer"
    assert len(llm.calls) == 0


def test_router_node_bank_failure_falls_back_to_retrieve() -> None:
    """If advice retrieval itself raises, the whole decision falls back to
    'retrieve', matching the existing LLM-failure fallback behavior."""
    bank = RaisingBank()
    llm = _mock_router_llm()
    state: IterRetState = {
        "original_query": "test query",
        "iteration_count": 1,
        "max_iterations": 5,
        "information_gaps": ["gap 1"],
        "accumulated_evidence": ["ev 1"],
        "consecutive_stuck_reflects": 0,
    }

    result = router_node(state, llm, bank=bank)

    assert result["_router_decision"] == "retrieve"


def test_router_node_without_bank_still_works_unchanged() -> None:
    """bank defaults to None: existing callers that don't pass one keep
    working exactly as before (no advice retrieval attempted)."""
    llm = MockLLMClientForRouter(action="retrieve")
    state: IterRetState = {
        "original_query": "test query",
        "iteration_count": 1,
        "max_iterations": 5,
        "information_gaps": ["gap 1"],
        "accumulated_evidence": ["ev 1"],
        "consecutive_stuck_reflects": 0,
    }

    result = router_node(state, llm)

    assert result["_router_decision"] == "retrieve"
    assert len(llm.calls) == 1


# ----------------------------------------------------------------------
# Fix 1: cue extraction re-derives from (E, G) every retrieval round
# ----------------------------------------------------------------------


class ScriptedRetrieveLLM(LLMClient):
    """Returns canned replies keyed off the prompt marker each node in
    nodes.py already tags its system prompts with, mirroring MockLLMClient's
    own dispatch style but with test-controlled content."""

    def __init__(
        self,
        cue_terms: list[str] | None = None,
        actions: list[str] | None = None,
        exclude_tags: list[str] | None = None,
    ) -> None:
        self.cue_terms = cue_terms or []
        self.actions = actions or ["cue_to_tag", "tag_to_content"]
        self.exclude_tags = exclude_tags or []
        self.calls: list[tuple[str, str]] = []

    def chat(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        self.calls.append((system_prompt, user_prompt))
        if "cue_extraction" in system_prompt:
            return json.dumps({"cues": self.cue_terms})
        if "situation_abstraction" in system_prompt:
            return json.dumps({"situation": "an abstracted situation"})
        if "action_selection" in system_prompt:
            return json.dumps({"actions": self.actions, "exclude_tags": self.exclude_tags})
        return json.dumps({})


def _two_person_graph() -> CueTagContentGraph:
    graph = CueTagContentGraph()
    graph.add_content("v1", "Alice went to the market")
    graph.add_content("v2", "Bob stayed home")
    graph.link("alice", "tagA", "v1")
    graph.link("bob", "tagB", "v2")
    return graph


def _base_state(query: str, gaps: list[str] | None = None) -> IterRetState:
    return {
        "original_query": query,
        "current_refined_query": query,
        "active_set": {"cues": [], "tags": [], "contents": []},
        "visited_content_ids": [],
        "accumulated_evidence": [],
        "information_gaps": gaps or [],
        "search_trajectory": [],
        "iteration_count": 0,
    }


def test_extract_cue_terms_parses_llm_cue_list() -> None:
    llm = ScriptedRetrieveLLM(cue_terms=["alice", "market"])

    terms = extract_cue_terms("who did alice see", [], [], llm)

    assert terms == ["alice", "market"]


def test_extract_cue_terms_returns_empty_on_non_list_reply() -> None:
    class BadReplyLLM(LLMClient):
        def chat(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
            return json.dumps({"cues": "not-a-list"})

    assert extract_cue_terms("q", [], [], BadReplyLLM()) == []


def test_retrieve_node_recomputes_cues_every_round_not_once() -> None:
    """The one-shot `if not active_set["cues"]` guard must be gone: cues
    should be re-derived from the current (E, G) state on every call, and
    previously-found cues must be unioned in, not lost."""
    graph = _two_person_graph()
    bank = ExperienceBank(KeywordOverlapEmbeddingBackend())

    llm_round1 = ScriptedRetrieveLLM(cue_terms=["alice"])
    state = _base_state("alice")
    state = retrieve_node(state, graph, bank, llm_round1)
    assert state["active_set"]["cues"] == ["alice"]

    # Round 2: the query text alone never mentions "bob" -- only the
    # evidence/gap-derived cue extraction call surfaces it. If cues were
    # only matched once against the original query, this would never
    # appear.
    llm_round2 = ScriptedRetrieveLLM(cue_terms=["bob"])
    state["information_gaps"] = ["need info about bob"]
    state = retrieve_node(state, graph, bank, llm_round2)

    assert "bob" in state["active_set"]["cues"]
    assert "alice" in state["active_set"]["cues"]  # union, not reset


# ----------------------------------------------------------------------
# Fix 5: gap-aware pruning before content load + inner CTC loop
# ----------------------------------------------------------------------


def test_retrieve_node_pruning_is_gap_aware_and_precedes_content_load() -> None:
    graph = _two_person_graph()
    bank = ExperienceBank(KeywordOverlapEmbeddingBackend())
    llm = ScriptedRetrieveLLM(cue_terms=["alice", "bob"], exclude_tags=["tagB"])
    state = _base_state("alice bob", gaps=["only care about alice"])

    state = retrieve_node(state, graph, bank, llm)

    assert "v1" in state["active_set"]["contents"]
    assert "v2" not in state["active_set"]["contents"]  # pruned via excluded tagB

    action_calls = [c for c in llm.calls if "action_selection" in c[0]]
    assert len(action_calls) == 1
    sent = json.loads(action_calls[0][1])
    assert sent["information_gaps"] == ["only care about alice"]


def _chain_graph() -> CueTagContentGraph:
    """v1 -(c0,g0)-> reachable; v1 activates (c1,g1) -> reaches v2;
    v2 activates (c2,g2) -> reaches v3. A 3-hop chain for exercising the
    inner CTC loop's content_to_cue_tag-driven expansion."""
    graph = CueTagContentGraph()
    graph.add_content("v1", "start node")
    graph.add_content("v2", "mid node")
    graph.add_content("v3", "end node")
    graph.link("c0", "g0", "v1")
    graph.link("c1", "g1", "v1")
    graph.link("c1", "g1", "v2")
    graph.link("c2", "g2", "v2")
    graph.link("c2", "g2", "v3")
    return graph


def test_retrieve_node_inner_loop_hops_until_no_new_content() -> None:
    from iterret.config import TraversalLimits

    graph = _chain_graph()
    bank = ExperienceBank(KeywordOverlapEmbeddingBackend())
    llm = ScriptedRetrieveLLM(
        cue_terms=["c0"], actions=["cue_to_tag", "tag_to_content", "content_to_cue_tag"]
    )
    state = _base_state("c0")
    state["active_set"]["cues"] = ["c0"]

    state = retrieve_node(state, graph, bank, llm, limits=TraversalLimits())

    # default max_inner_ctc_iterations=3 is exactly enough to walk the full chain
    assert set(state["active_set"]["contents"]) == {"v1", "v2", "v3"}


def test_retrieve_node_inner_loop_respects_its_own_budget() -> None:
    from iterret.config import TraversalLimits

    graph = _chain_graph()
    bank = ExperienceBank(KeywordOverlapEmbeddingBackend())
    llm = ScriptedRetrieveLLM(
        cue_terms=["c0"], actions=["cue_to_tag", "tag_to_content", "content_to_cue_tag"]
    )
    state = _base_state("c0")
    state["active_set"]["cues"] = ["c0"]

    state = retrieve_node(
        state, graph, bank, llm, limits=TraversalLimits(max_inner_ctc_iterations=1)
    )

    # capped at 1 inner hop: only the directly-reachable v1 is found this round
    assert state["active_set"]["contents"] == ["v1"]
