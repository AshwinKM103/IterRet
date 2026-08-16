"""LoCoMo dataset loading and path resolution.

Handles loading LoCoMo conversations from disk and resolving dataset paths
by subset (train/test/val) and format (json/jsonl/csv).
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from .memory_builder import DialogueTurn

CATEGORY_NAMES: dict[int, str] = {
    1: "Multi-hop",
    2: "Temporal",
    3: "Open-domain",
    4: "Single-hop",
    5: "Adversarial",
}
DEFAULT_EVAL_CATEGORIES: tuple[int, ...] = (
    1,
    2,
    3,
    4,
)  # exclude adversarial, matching MRAgent/MemR3


class LoCoMoQuestion(TypedDict):
    """A single question-answer pair from LoCoMo dataset.

    Attributes:
        question: The question text.
        answer: The ground-truth answer text.
        category: Question category (1-5, see CATEGORY_NAMES).
        evidence: List of evidence passage ids that support the answer.
    """

    question: str
    answer: str
    category: int
    evidence: list[str]


class LoCoMoConversation(TypedDict):
    """A full LoCoMo conversation sample with questions.

    Attributes:
        sample_id: Unique conversation identifier.
        turns: List of dialogue turns (speaker, text, time).
        questions: List of question-answer pairs to evaluate against.
    """

    sample_id: str
    turns: list[DialogueTurn]
    questions: list[LoCoMoQuestion]


def load_raw_locomo(path: str) -> list[Any]:
    """Load raw LoCoMo dataset from JSON file.

    Args:
        path: Path to the LoCoMo JSON file.

    Returns:
        List of raw conversation dicts (before parsing/processing).

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.

    Example:
        >>> conversations = load_raw_locomo("locomo.json")
        >>> len(conversations)  # Number of conversations
    """
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _session_keys(conversation: dict) -> list[str]:
    """Extract session keys from a conversation dict, sorted numerically.

    Finds all keys matching pattern "session_N" (excluding "_date_time" keys)
    and sorts them by numeric session index.

    Args:
        conversation: Raw conversation dict.

    Returns:
        List of session keys sorted numerically (session_1, session_2, ...).

    Example:
        >>> conv = {"session_1": [...], "session_2": [...], "sample_id": "..."}
        >>> _session_keys(conv)  # ['session_1', 'session_2']
    """
    keys = [k for k in conversation if k.startswith("session_") and not k.endswith("_date_time")]
    # sort numerically by the session index rather than lexicographically
    # (lexicographic would put session_10 before session_2).
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def _turn_text(turn: dict) -> str:
    """Extract and annotate text from a turn, including images if present.

    Args:
        turn: A turn dict with optional "text" and "blip_caption" keys.

    Returns:
        The text, with image captions appended in [shared an image: ...] format.

    Example:
        >>> turn = {"text": "Look at this!", "blip_caption": "A cat"}
        >>> _turn_text(turn)  # "Look at this! [shared an image: A cat]"
    """
    text = turn.get("text", "")
    caption = turn.get("blip_caption")
    if caption:
        text = f"{text} [shared an image: {caption}]"
    return text


def conversation_to_turns(
    conversation: dict, *, max_turns: int | None = None
) -> list[DialogueTurn]:
    """Flatten all sessions into a single ordered DialogueTurn list.

    Processes all sessions in numeric order, extracting speaker, text, and
    time (from the session's date_time). Stops after max_turns if provided.

    Args:
        conversation: Raw conversation dict with session_N keys.
        max_turns: Optional maximum turns to extract (default: all).

    Returns:
        List of DialogueTurn dicts in chronological order.

    Example:
        >>> conversation = {
        ...     "session_1": [{"speaker": "A", "text": "hi", "blip_caption": None}],
        ...     "session_1_date_time": "2025-08-17",
        ... }
        >>> turns = conversation_to_turns(conversation)
        >>> turns[0]["time"]  # '2025-08-17'
    """
    turns: list[DialogueTurn] = []
    for session_key in _session_keys(conversation):
        time = conversation.get(f"{session_key}_date_time")
        for turn in conversation[session_key]:
            turns.append(
                DialogueTurn(
                    speaker=turn.get("speaker", "Unknown"), text=_turn_text(turn), time=time
                )
            )
            if max_turns is not None and len(turns) >= max_turns:
                return turns
    return turns


def parse_conversation(
    raw_conv: dict,
    *,
    max_turns: int | None = None,
    categories: tuple[int, ...] = DEFAULT_EVAL_CATEGORIES,
    max_questions: int | None = None,
) -> LoCoMoConversation:
    """Parse a raw LoCoMo conversation into a typed LoCoMoConversation.

    Filters questions by category (default excludes adversarial) and extracts
    dialogue turns up to max_turns if provided.

    Args:
        raw_conv: Raw conversation dict from LoCoMo.
        max_turns: Optional max dialogue turns to extract.
        categories: Question categories to include (default: 1-4, excludes 5=adversarial).
        max_questions: Optional max questions to extract.

    Returns:
        LoCoMoConversation with sample_id, turns, and filtered questions.

    Example:
        >>> raw = {"sample_id": "1", "conversation": {...}, "qa": [...]}
        >>> parsed = parse_conversation(raw, categories=(1, 2, 3, 4))
        >>> parsed["sample_id"]  # "1"
        >>> len(parsed["questions"]) > 0  # True (usually)
    """
    questions: list[LoCoMoQuestion] = []
    for qa in raw_conv.get("qa", []):
        category = qa.get("category")
        if category not in categories:
            continue
        answer = qa.get("answer", qa.get("adversarial_answer"))
        if answer is None:
            continue
        questions.append(
            LoCoMoQuestion(
                question=qa["question"],
                answer=str(answer),
                category=category,
                evidence=list(qa.get("evidence", [])),
            )
        )
        if max_questions is not None and len(questions) >= max_questions:
            break

    return LoCoMoConversation(
        sample_id=raw_conv.get("sample_id", "unknown"),
        turns=conversation_to_turns(raw_conv["conversation"], max_turns=max_turns),
        questions=questions,
    )


def split_bootstrap_eval(
    raw_conversations: list,
    *,
    bootstrap_fraction: float = 0.1,
) -> tuple[list, list]:
    """Split conversations into bootstrap (seed) and evaluation sets.

    Deterministically splits the first ceil(N * bootstrap_fraction) conversations
    as bootstrap seed (for experience bank accumulation), and the rest as held-out
    evaluation. Order is preserved (not shuffled) for reproducibility.

    Args:
        raw_conversations: List of raw conversation dicts.
        bootstrap_fraction: Fraction for bootstrap set (default 0.1 = 10%).

    Returns:
        Tuple of (bootstrap_conversations, eval_conversations).

    Example:
        >>> convs = [{"id": i} for i in range(10)]
        >>> bootstrap, eval_set = split_bootstrap_eval(convs, bootstrap_fraction=0.1)
        >>> len(bootstrap)  # ceil(10 * 0.1) = 1
        >>> len(eval_set)  # 9
    """
    n = len(raw_conversations)
    n_bootstrap = max(1, round(n * bootstrap_fraction)) if bootstrap_fraction > 0 else 0
    n_bootstrap = min(n_bootstrap, n - 1) if n > 1 else n
    return raw_conversations[:n_bootstrap], raw_conversations[n_bootstrap:]
