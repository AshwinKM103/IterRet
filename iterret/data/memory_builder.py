"""Memory graph construction from raw dialogue.

LLM-guided extraction of episodes, semantics, and topics to build a
Cue-Tag-Content graph from dialogue turns.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, TypedDict

from config.constants import DEFAULT_MAX_CHARS_PER_CALL

from ..models.llm_client import LLMClient
from ..utils.json_utils import parse_json_object
from ..utils.logger import get_logger
from ..utils.prompts import (
    EPISODE_EXTRACTION_SYSTEM_PROMPT,
    SEMANTIC_EXTRACTION_SYSTEM_PROMPT,
    TOPIC_ABSTRACTION_SYSTEM_PROMPT,
)
from .ctc_graph import CueTagContentGraph

logger = get_logger(__name__)


class DialogueTurn(TypedDict, total=False):
    speaker: str
    text: str
    time: str | None


def _warn(message: str) -> None:
    """Warning logged via unified logger.

    Args:
        message: The warning message.
    """
    logger.warning(message)


def _safe_chat(
    system_prompt: str, user_prompt: str, llm: LLMClient, *, on_error: str
) -> dict[str, Any]:
    """Chat with LLM and parse JSON, with graceful error handling.

    Calls llm.chat(), attempts JSON parsing, and returns a dict. If any error
    occurs (network failure, over-budget request, parse failure), logs a warning
    and returns an empty dict instead of crashing the graph build.

    Args:
        system_prompt: The system role prompt.
        user_prompt: The user message.
        llm: The LLMClient to use.
        on_error: Message to prepend to any warning (e.g., "episode extraction failed").

    Returns:
        Parsed JSON dict, or {} if any error occurs.
    """
    try:
        raw = llm.chat(system_prompt, user_prompt)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any backend, any failure mode
        _warn(f"{on_error}: {exc}")
        return {}
    return parse_json_object(raw)


def _iter_chunks(items: list[dict], *, text_key: str, max_chars: int) -> Iterator[list[dict]]:
    """Batch items by character count to stay within token budget.

    Groups items into batches such that the joined text length (by item[text_key])
    stays under max_chars (roughly 4 chars per token). If a single item exceeds
    max_chars, it gets its own batch rather than being dropped.

    Args:
        items: List of dict items to batch.
        text_key: Key in each item dict containing the text to measure.
        max_chars: Target maximum total characters per batch.

    Yields:
        Lists of items, each batch under the character budget.

    Example:
        >>> items = [{"text": "a"*100}, {"text": "b"*100}]
        >>> list(_iter_chunks(items, text_key="text", max_chars=150))
        # [[{"text": "a"*100}], [{"text": "b"*100}]]
    """
    batch: list[dict] = []
    batch_len = 0
    for item in items:
        item_len = len(item[text_key])
        if batch and batch_len + item_len > max_chars:
            yield batch
            batch, batch_len = [], 0
        batch.append(item)
        batch_len += item_len
    if batch:
        yield batch


def _extract_episode(turn: DialogueTurn, llm: LLMClient) -> dict[str, Any]:
    """Extract episodic tag and cues from a dialogue turn.

    LLM-based extraction of the relational pattern (tag) and retrieval terms (cues)
    for one dialogue turn. Returns fallback values if extraction fails.

    Args:
        turn: A DialogueTurn with speaker and text.
        llm: The LLMClient to use.

    Returns:
        Dict with keys "tag" (str) and "cues" (list[str]). Fallback tag is
        "Mention" and fallback cues list includes the speaker name if available.

    Example:
        >>> result = _extract_episode(DialogueTurn(speaker="Alice", text="I got a new job!"), llm)
        >>> result["tag"]  # e.g., "Job Change"
        >>> "Alice" in result["cues"]  # True
    """
    parsed = _safe_chat(
        EPISODE_EXTRACTION_SYSTEM_PROMPT,
        json.dumps(dict(turn)),
        llm,
        on_error="episode extraction failed, using fallback tag/cues",
    )
    tag = str(parsed.get("tag") or "Mention")
    cues = [str(c) for c in parsed.get("cues") or [] if str(c).strip()]
    if turn.get("speaker") and turn["speaker"] not in cues:
        cues.append(turn["speaker"])
    if not cues:
        cues = ["Unknown"]
    return {"tag": tag, "cues": cues}


def _extract_semantics(
    episode_summaries: list[dict[str, Any]], llm: LLMClient, *, max_chars: int
) -> list[dict[str, Any]]:
    """Extract stable semantic facts from episode summaries.

    Batches episodes by character count and sends each batch to the LLM for
    extraction of entity-level cues, aspects, and facts. Failed batches degrade
    to empty semantics rather than crashing the build.

    Args:
        episode_summaries: List of dicts with "text", "tag", "content_id" keys.
        llm: The LLMClient to use.
        max_chars: Max characters per LLM call (token budget limit).

    Returns:
        List of dicts with keys "cue" (str), "tag" (str), "content" (str).

    Example:
        >>> episodes = [{"text": "Alice is a doctor", "tag": "Job", "content_id": "e1"}]
        >>> semantics = _extract_semantics(episodes, llm, max_chars=4000)
        >>> len(semantics) > 0  # Typically true if extraction succeeds
    """
    cleaned: list[dict] = []
    for chunk in _iter_chunks(episode_summaries, text_key="text", max_chars=max_chars):
        full_text = "\n".join(s["text"] for s in chunk)
        parsed = _safe_chat(
            SEMANTIC_EXTRACTION_SYSTEM_PROMPT,
            full_text,
            llm,
            on_error=f"semantic extraction skipped a {len(chunk)}-episode chunk",
        )
        for item in parsed.get("semantics") or []:
            if not isinstance(item, dict):
                continue
            cue, tag, content = item.get("cue"), item.get("tag"), item.get("content")
            if cue and tag and content:
                cleaned.append({"cue": str(cue), "tag": str(tag), "content": str(content)})
    return cleaned


def _abstract_topics(
    episode_summaries: list[dict[str, Any]], llm: LLMClient, *, max_chars: int
) -> list[dict[str, Any]]:
    """Identify recurring themes across episodes (topic abstraction layer).

    Groups episodes with shared themes into topic nodes. Single-episode topics
    are skipped. Failed batches degrade to empty topics rather than crashing.

    Args:
        episode_summaries: List of dicts with "text", "tag", "content_id" keys.
        llm: The LLMClient to use.
        max_chars: Max characters per LLM call (token budget limit).

    Returns:
        List of dicts with keys "topic" (str) and "episode_ids" (list[str]).
        Guaranteed that each topic contains at least 2 episodes.

    Example:
        >>> episodes = [
        ...     {"text": "Alice went shopping", "content_id": "e1"},
        ...     {"text": "Bob went shopping", "content_id": "e2"},
        ... ]
        >>> topics = _abstract_topics(episodes, llm, max_chars=4000)
        >>> topics[0]["episode_ids"]  # e.g., ['e1', 'e2']
    """
    if len(episode_summaries) < 2:
        return []
    valid_ids = {e["content_id"] for e in episode_summaries}
    topics: list[dict] = []
    for chunk in _iter_chunks(episode_summaries, text_key="text", max_chars=max_chars):
        if len(chunk) < 2:
            continue  # can't form a >=2-episode topic from a single-episode chunk
        parsed = _safe_chat(
            TOPIC_ABSTRACTION_SYSTEM_PROMPT,
            json.dumps({"episodes": chunk}),
            llm,
            on_error=f"topic abstraction skipped a {len(chunk)}-episode chunk",
        )
        for item in parsed.get("topics") or []:
            if not isinstance(item, dict):
                continue
            topic_text = item.get("topic")
            episode_ids = [eid for eid in item.get("episode_ids") or [] if eid in valid_ids]
            if topic_text and len(episode_ids) >= 2:
                topics.append({"topic": str(topic_text), "episode_ids": episode_ids})
    return topics


def build_ctc_graph_from_dialogue(
    turns: list[DialogueTurn],
    llm: LLMClient,
    *,
    max_chars_per_call: int = DEFAULT_MAX_CHARS_PER_CALL,
) -> CueTagContentGraph:
    """Build a complete CTC graph from dialogue, running all three extraction stages.

    Orchestrates the full memory graph construction pipeline:
      1. **Episodic layer**: Extract tag and cues from each turn, create links
      2. **Semantic layer**: Extract stable entity-level facts from episodes
      3. **Topic layer**: Identify recurring themes across episodes

    Each stage can degrade gracefully if LLM calls fail (e.g., over-budget requests).
    Exceptions are logged as warnings and skipped rather than crashing the build.

    Args:
        turns: List of DialogueTurn dicts (speaker, text, optional time).
        llm: The LLMClient to use for all three extraction stages.
        max_chars_per_call: Max characters per LLM call for batching (semantic and topic
            stages). Default 4000 (conservative for ~8k-token context). Lower for smaller
            servers, raise for larger context windows.

    Returns:
        A fully constructed CueTagContentGraph with episodic, semantic, and topic layers.

    Example:
        >>> turns = [
        ...     DialogueTurn(speaker="Alice", text="I got a new job!", time="2025-08-17"),
        ...     DialogueTurn(speaker="Bob", text="Congrats!", time="2025-08-17"),
        ... ]
        >>> llm = OpenAICompatibleLLMClient()
        >>> graph = build_ctc_graph_from_dialogue(turns, llm)
        >>> len(graph.contents)  # Episodic + semantic + topic nodes
        >>> "Alice" in graph.cues  # True (extracted from turn)
    """
    graph = CueTagContentGraph()
    episode_summaries: list[dict] = []

    # 1. Episodic layer: one (cue, tag, episode) set of links per turn.
    for i, turn in enumerate(turns):
        content_id = f"e{i + 1}"
        text = f"{turn.get('speaker', 'Unknown')}: {turn.get('text', '')}"
        graph.add_content(content_id, text, layer="episodic", time=turn.get("time"))

        extracted = _extract_episode(turn, llm)
        for cue in extracted["cues"]:
            graph.link(cue, extracted["tag"], content_id)

        episode_summaries.append({"content_id": content_id, "tag": extracted["tag"], "text": text})

    # 2. Semantic layer: stable facts anchored to entity-level cues.
    for j, semantic in enumerate(
        _extract_semantics(episode_summaries, llm, max_chars=max_chars_per_call)
    ):
        content_id = f"s{j + 1}"
        graph.add_content(content_id, semantic["content"], layer="semantic")
        graph.link(semantic["cue"], semantic["tag"], content_id)

    # 3. Abstraction layer: topic nodes wired to their constituent episodes.
    for k, topic in enumerate(
        _abstract_topics(episode_summaries, llm, max_chars=max_chars_per_call)
    ):
        content_id = f"t{k + 1}"
        graph.add_content(
            content_id, f"Topic: {topic['topic']}", layer="topic", topic_links=topic["episode_ids"]
        )

    return graph
