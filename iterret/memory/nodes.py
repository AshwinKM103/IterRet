"""Closed-loop nodes: planning, retrieval, routing, and answer synthesis.

Implements the four steps of closed-loop episode state management: planning_node, retrieve_node,
routing_node (reflection), and answer_node, orchestrated by LanggraphGraphBuilder.
"""

from __future__ import annotations

import json
from typing import Literal

from ..config import TraversalLimits
from ..data.ctc_graph import CueTagContentGraph
from ..models.llm_client import LLMClient
from ..state import DEFAULT_MAX_ITERATIONS, DEFAULT_MAX_STUCK_REFLECTS, IterRetState, SearchStep
from ..utils.json_utils import parse_json_object
from ..utils.prompts import (
    ACTION_SELECTION_SYSTEM_PROMPT,
    CUE_EXTRACTION_SYSTEM_PROMPT,
    FINAL_ANSWER_SYSTEM_PROMPT,
    ROUTER_DECISION_SYSTEM_PROMPT,
    ROUTING_AND_REFLECTION_SYSTEM_PROMPT,
    SITUATION_SYSTEM_PROMPT,
)
from .evidence_tracker import EvidenceTracker
from .experience_bank import ExperienceBank, planning_condition, reflection_condition
from .live_rubric import score_reflect_step


def abstract_situation(condition: str, llm: LLMClient) -> str:
    """LLM_sigma(c_i): abstracts a condition into a matchable situation (R2-Mem Eq. 11)."""
    raw = llm.chat(SITUATION_SYSTEM_PROMPT, json.dumps({"condition": condition}))
    return parse_json_object(raw).get("situation") or condition


def extract_cue_terms(
    query: str, evidence: list[str], gaps: list[str], llm: LLMClient
) -> list[str]:
    """LLM-based cue extraction (closed-loop episode state management step 1), re-run every retrieval
    round against the current (E, G) state rather than the query alone.
    Mirrors MRAgent's `extract_question_keys` (agent/agent.py:566-572). The
    returned terms are matched against the graph's actual cue ids by the
    caller (see `_refresh_active_cues`) via lexical subset matching -- an
    LLM can't be trusted to emit ids that exist verbatim in the graph, so
    its raw output text is fed into that matcher rather than applied
    directly as cue ids.
    """
    raw = llm.chat(
        CUE_EXTRACTION_SYSTEM_PROMPT,
        json.dumps({"query": query, "evidence": evidence, "gaps": gaps}),
    )
    parsed = parse_json_object(raw)
    terms = parsed.get("cues", [])
    if not isinstance(terms, list):
        return []
    return [str(term) for term in terms if isinstance(term, (str, int, float))]


def _refresh_active_cues(
    query: str,
    evidence: list[str],
    gaps: list[str],
    graph: CueTagContentGraph,
    llm: LLMClient,
    limits: TraversalLimits,
    active_set: dict[str, list[str]],
) -> None:
    """CTC Step 1 (closed-loop episode state management step 1): re-derive retrieval cues from the
    current evidence-gap state (E, G) every round -- not a one-shot,
    query-only match. Newly-matched cues are UNIONed with cues already in
    the active set (not reset), so cues discovered in earlier rounds are
    never lost.
    """
    cue_terms = extract_cue_terms(query, evidence, gaps, llm)
    combined_text = " ".join([query, *cue_terms, *evidence, *gaps])
    matched = graph.match_query_to_cues(combined_text, max_matches=limits.max_active_cues)
    merged = sorted(set(active_set["cues"]) | matched)
    active_set["cues"] = merged[: limits.max_active_cues]


def _select_traversal_actions(
    query: str,
    active_set: dict[str, list[str]],
    advice_text: str,
    gaps: list[str],
    llm: LLMClient,
) -> tuple[list[str], set[str]]:
    """f_select (Eq. 10): Planning-experience-guided AND gap (G)-aware --
    `exclude_tags` is the pruning decision applied in `_prune_tags_by_gap`.
    """
    selection_prompt = json.dumps(
        {
            "query": query,
            "active_set": active_set,
            "planning_advice": advice_text,
            "information_gaps": gaps,
        }
    )
    raw = llm.chat(ACTION_SELECTION_SYSTEM_PROMPT, selection_prompt)
    parsed = parse_json_object(raw)
    actions = parsed.get("actions") or ["cue_to_tag", "tag_to_content"]
    exclude_tags = set(parsed.get("exclude_tags", []))
    return actions, exclude_tags


def _expand_tags(
    graph: CueTagContentGraph,
    active_set: dict[str, list[str]],
    actions: list[str],
    query: str,
    limits: TraversalLimits,
) -> set[str]:
    """CTC Step 2 (closed-loop episode state management step 2): tag-guided expansion -- read the
    tags of neighboring cues and select the promising next hops."""
    tags = set(active_set["tags"])
    if "cue_to_tag" in actions:
        tags |= graph.forward_cue_to_tag(active_set["cues"])
        if len(tags) > limits.max_active_tags:
            tags = set(graph.rank_tags_by_relevance(tags, query)[: limits.max_active_tags])
    return tags


def _prune_tags_by_gap(tags: set[str], exclude_tags: set[str]) -> list[str]:
    """CTC Step 4 (closed-loop episode state management step 4): prune hops whose tags are
    inconsistent with the current information gap G, BEFORE their content
    is loaded. `exclude_tags` was decided gap-aware by
    `_select_traversal_actions` above; this is the deterministic
    application of that judgment, and it runs strictly before
    `_load_content`.
    """
    return [tag for tag in tags if tag not in exclude_tags]


def _load_content(
    graph: CueTagContentGraph,
    active_set: dict[str, list[str]],
    tags: list[str],
    actions: list[str],
    visited: set[str],
    already_found: set[str],
    query: str,
    limits: TraversalLimits,
) -> set[str]:
    """CTC Step 3 (closed-loop episode state management step 3): load the full content of the
    pruned, surviving hops and append it to this round's candidate pool."""
    if "tag_to_content" not in actions:
        return set()
    new_contents = graph.forward_tag_to_content(
        active_set["cues"], tags, exclude_content_ids=visited | already_found
    )
    if len(new_contents) > limits.max_new_content_per_round:
        new_contents = set(
            graph.rank_contents_by_relevance(new_contents, query)[
                : limits.max_new_content_per_round
            ]
        )
    return new_contents


def _expand_from_new_content(
    graph: CueTagContentGraph,
    active_set: dict[str, list[str]],
    actions: list[str],
    limits: TraversalLimits,
) -> list[str]:
    """Continuation of CTC Step 2: content already loaded (in a PRIOR inner
    hop) can itself activate new (cue, tag) pairs (Pi_v->(c,g)); this feeds
    those newly-discovered tags into the NEXT hop's tag-guided expansion
    (`_expand_tags`), and mutates `active_set["cues"]` in place with any
    newly-discovered cues, before that same hop's pruning/loading runs.
    """
    if "content_to_cue_tag" not in actions or not active_set["contents"]:
        return []
    tags: list[str] = []
    for cue_id, tag in graph.reverse_content_to_cue_tag(active_set["contents"]):
        if cue_id not in active_set["cues"] and len(active_set["cues"]) < limits.max_active_cues:
            active_set["cues"].append(cue_id)
        if tag not in tags and len(tags) < limits.max_active_tags:
            tags.append(tag)
    return tags


def retrieve_node(
    state: IterRetState,
    graph: CueTagContentGraph,
    bank: ExperienceBank,
    llm: LLMClient,
    limits: TraversalLimits | None = None,
) -> IterRetState:
    """The CTC traversal (closed-loop episode state management): 5 distinguishable steps --
    (1) cue extraction, (2) tag-guided expansion, (3) content loading,
    (4) gap-conditioned pruning (before loading), (5) evidence-triggered
    stopping -- run as their own inner loop, returning control to the
    outer closed-loop controller (graph.py) only once that inner loop's
    own stopping condition is met.
    """
    limits = limits or TraversalLimits()
    query = state.get("current_refined_query") or state["original_query"]
    active_set = state.setdefault("active_set", {"cues": [], "tags": [], "contents": []})
    visited = set(state.get("visited_content_ids", []))
    evidence = state.get("accumulated_evidence", [])
    gaps = state.get("information_gaps", [])

    # Step 1: cue extraction, re-derived from (E, G) every round.
    _refresh_active_cues(query, evidence, gaps, graph, llm, limits, active_set)

    # Planning experience retrieval (R2-Mem Eq. 12, c_i = q_i)
    condition = planning_condition(query)
    situation = abstract_situation(condition, llm)
    advice_entries = bank.retrieve(condition, situation, module="Planning")
    advice_text = "; ".join(entry["experience"] for entry in advice_entries)

    actions, exclude_tags = _select_traversal_actions(query, active_set, advice_text, gaps, llm)

    all_new_contents: set[str] = set()
    inner_rounds = 0
    for inner_rounds in range(1, limits.max_inner_ctc_iterations + 1):
        # Continuation of Step 2: content loaded in the PRIOR inner hop can itself
        # activate new (cue, tag) pairs -- fold those into this hop's cues/candidate
        # tags before expanding, so a discovered hop is actually used the next round
        # instead of being found one round too late to matter.
        extra_tags = _expand_from_new_content(graph, active_set, actions, limits)
        # Step 2: tag-guided expansion -- select promising next hops.
        candidate_tags = _expand_tags(graph, active_set, actions, query, limits) | set(extra_tags)
        # Step 4: prune hops inconsistent with G, BEFORE content is loaded.
        pruned_tags = _prune_tags_by_gap(candidate_tags, exclude_tags)
        # Step 3: load full content of the surviving hops.
        new_contents = _load_content(
            graph, active_set, pruned_tags, actions, visited, all_new_contents, query, limits
        )

        # Order by relevance to the current query, not alphabetically: the whole point of
        # capping via rank_*_by_relevance above is to put the genuinely relevant items
        # first, and an LLM reading a long list pays the most attention to what's earliest
        # in it -- re-sorting alphabetically here would undo that by burying e.g. "LGBTQ
        # Support Group" behind 50+ alphabetically-earlier but irrelevant tags before the
        # model ever reads that far.
        active_set["tags"] = graph.rank_tags_by_relevance(pruned_tags, query)
        active_set["contents"] = sorted(set(active_set["contents"]) | new_contents)
        all_new_contents |= new_contents

        # Step 5: evidence-triggered stopping -- halt this inner CTC loop once a hop adds
        # nothing new (or the inner budget above is exhausted), then return control to the
        # outer closed-loop controller.
        if not new_contents:
            break

    state["_scratch_new_retrieval"] = graph.rank_contents_by_relevance(all_new_contents, query)
    state["active_set"] = active_set
    state["iteration_count"] = state.get("iteration_count", 0) + 1

    trajectory = state.setdefault("search_trajectory", [])
    trajectory.append(
        SearchStep(
            iteration=state["iteration_count"],
            module="Planning",
            query_used=query,
            action_taken="+".join(actions)
            + (f" (excluded tags: {sorted(exclude_tags)})" if exclude_tags else "")
            + f" [{inner_rounds} inner CTC step(s)]",
            found_summary=f"{len(all_new_contents)} new content node(s)",
            decision="retrieve",
            rubric_scores={},
        )
    )
    return state


def reflect_node(
    state: IterRetState, graph: CueTagContentGraph, bank: ExperienceBank, llm: LLMClient
) -> IterRetState:
    new_content_ids = state.get("_scratch_new_retrieval", [])
    new_content_ids = [cid for cid in new_content_ids if cid in graph.contents]
    # The LLM can only return ids for content it was actually shown an id
    # for, so pair each id with its text rather than handing over bare text
    # (which it would otherwise just echo back, never matching a real id).
    new_content_payload = [
        {"id": cid, "text": graph.contents[cid].display_text()} for cid in new_content_ids
    ]

    original_query = state["original_query"]
    evidence = state.setdefault("accumulated_evidence", [])
    gaps = state.setdefault("information_gaps", [])
    tracker = EvidenceTracker(evidence, gaps)

    # Reflection experience retrieval (R2-Mem Eq. 12, c_i = [Q + m_i])
    condition = reflection_condition(original_query, evidence)
    situation = abstract_situation(condition, llm)
    advice_entries = bank.retrieve(condition, situation, module="Reflection")
    advice_text = "; ".join(entry["experience"] for entry in advice_entries)

    routing_prompt = json.dumps(
        {
            "original_query": original_query,
            "evidence": tracker.evidence,
            "gaps": tracker.gaps,
            "new_content": new_content_payload,
            "reflection_advice": advice_text,
        }
    )
    raw = llm.chat(ROUTING_AND_REFLECTION_SYSTEM_PROMPT, routing_prompt)
    parsed = parse_json_object(raw)

    kept = parsed.get("kept_content_ids", "ALL")
    if kept == "ALL":
        kept_ids = new_content_ids
    else:
        kept_ids = [cid for cid in kept if cid in graph.contents]
        if not kept_ids and new_content_ids:
            # The model didn't return ids that match anything it was shown
            # (e.g. it echoed text instead of ids) -- fail open instead of
            # silently discarding everything that was just retrieved.
            kept_ids = new_content_ids
    kept_texts = [graph.contents[cid].display_text() for cid in kept_ids]
    made_progress = tracker.add_evidence(kept_texts)
    tracker.update_gaps(parsed.get("resolved_gaps", []), parsed.get("new_gaps", []))

    state["information_gaps"] = tracker.gaps
    state["accumulated_evidence"] = tracker.evidence
    state["visited_content_ids"] = sorted(
        set(state.get("visited_content_ids", [])) | set(new_content_ids)
    )
    state["_scratch_new_retrieval"] = []
    state["consecutive_stuck_reflects"] = (
        0 if made_progress else state.get("consecutive_stuck_reflects", 0) + 1
    )

    next_query = parsed.get("next_query") or ""
    if tracker.gaps and next_query:
        state["current_refined_query"] = f"{original_query} | {next_query}"
    elif not tracker.gaps:
        state["current_refined_query"] = original_query

    rubric_scores, _ = score_reflect_step(
        original_query, tracker.evidence, tracker.gaps, next_query, llm
    )

    trajectory = state.setdefault("search_trajectory", [])
    trajectory.append(
        SearchStep(
            iteration=state.get("iteration_count", 0),
            module="Reflection",
            query_used=state.get("current_refined_query", original_query),
            action_taken="f_route+gap_update",
            found_summary=f"kept {len(kept_texts)} item(s); {len(tracker.gaps)} gap(s) remain",
            decision="reflect",
            rubric_scores=rubric_scores,
        )
    )
    return state


def router_node(
    state: IterRetState, llm: LLMClient, bank: ExperienceBank | None = None
) -> IterRetState:
    """LLM-based routing decision: when hard-stop conditions are false,
    ask the LLM whether to retrieve again or answer now. Hard-stop conditions
    (iteration budget, empty gaps, consecutive stuck reflects) always override.

    When ``bank`` is provided, Reflection experience advice (retrieval quality rubric:
    known high-quality behaviors and known failure modes should bias "the
    router's context") is retrieved and injected into the LLM decision
    prompt -- the same `bank.retrieve(...)` call pattern used by
    `retrieve_node`/`reflect_node`. ``bank`` is optional (defaults to
    None, meaning "no advice") purely so callers that don't have a bank
    handy can still invoke this node; production wiring (graph.py) always
    passes one.
    """
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    gaps = state.get("information_gaps", [])
    stuck = state.get("consecutive_stuck_reflects", 0)

    # Hard-stop conditions override: deterministic, no LLM call
    if iteration >= max_iterations:
        state["_router_decision"] = "answer"
        return state
    if not gaps:
        state["_router_decision"] = "answer"
        return state
    if stuck >= DEFAULT_MAX_STUCK_REFLECTS:
        state["_router_decision"] = "answer"
        return state

    # No hard-stop: ask the LLM to decide, informed by bank advice if available
    original_query = state["original_query"]
    evidence = state.get("accumulated_evidence", [])
    try:
        advice_text = ""
        if bank is not None:
            condition = reflection_condition(original_query, evidence)
            situation = abstract_situation(condition, llm)
            advice_entries = bank.retrieve(condition, situation, module="Reflection")
            advice_text = "; ".join(entry["experience"] for entry in advice_entries)

        router_prompt = json.dumps(
            {
                "original_query": original_query,
                "accumulated_evidence": evidence,
                "information_gaps": gaps,
                "iteration": iteration,
                "max_iterations": max_iterations,
                "experience_advice": advice_text,
            }
        )
        raw = llm.chat(ROUTER_DECISION_SYSTEM_PROMPT, router_prompt)
        parsed = parse_json_object(raw)
        decision = parsed.get("action", "retrieve")
        if decision not in ("retrieve", "answer"):
            decision = "retrieve"
    except Exception:
        # LLM call (advice retrieval or the decision itself) failed; fall back to
        # "retrieve" to keep the pipeline moving
        decision = "retrieve"

    state["_router_decision"] = decision
    return state


def route_after_reflect(state: IterRetState) -> Literal["retrieve", "answer"]:
    """Read the router's decision (already computed by router_node) and return it."""
    decision = state.get("_router_decision", "retrieve")
    if isinstance(decision, str) and decision in ("retrieve", "answer"):
        return decision
    return "retrieve"


def answer_node(state: IterRetState, llm: LLMClient) -> IterRetState:
    evidence_block = "\n".join(f"- {item}" for item in state.get("accumulated_evidence", []))
    user_prompt = f"Question: {state['original_query']}\nEvidence:\n{evidence_block}"
    state["final_answer"] = llm.chat(FINAL_ANSWER_SYSTEM_PROMPT, user_prompt)

    trajectory = state.setdefault("search_trajectory", [])
    trajectory.append(
        SearchStep(
            iteration=state.get("iteration_count", 0),
            module="Reflection",
            query_used=state["original_query"],
            action_taken="answer",
            found_summary="final answer synthesized",
            decision="answer",
            rubric_scores={},
        )
    )
    return state
