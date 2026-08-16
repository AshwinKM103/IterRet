"""Robust JSON extraction from LLM outputs.

LLM responses often contain noise: markdown code fences, preamble text, trailing
commentary, or slightly malformed JSON. This module provides utilities to robustly
extract valid JSON objects from such imperfect outputs without requiring perfect
formatting.

The core function parse_json_object() implements a multi-strategy extraction
approach that tries direct parsing, markdown fence extraction, and brace-substring
extraction in sequence, returning the first valid JSON object found or an empty
dict on failure.

This utility is essential for IterRet's closed-loop reasoning: LLM responses
(judgments, rankings, refined queries, evidence synthesis) are structured as JSON,
but LLM outputs are unpredictable. Robust extraction decouples LLM quality from
downstream system stability.

Functions:
    parse_json_object: Extract a JSON object from noisy LLM output (multi-strategy).

Example:
    >>> from iterret.utils.json_utils import parse_json_object
    >>> raw_response = '''```json
    ... {"correct": true, "reason": "The answer matches"}
    ... ```'''
    >>> obj = parse_json_object(raw_response)
    >>> obj
    {'correct': True, 'reason': 'The answer matches'}
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def parse_json_object(raw: str) -> dict[str, Any]:
    """Extract a JSON object from LLM output (which may contain noise).

    LLMs often produce responses with preamble, markdown code fences, trailing
    commentary, or malformed JSON. This function robustly extracts a valid JSON
    object using a multi-strategy approach that attempts increasingly lenient
    parsing strategies:

    1. **Direct parse**: Parse the raw input as-is (fastest, handles valid JSON).
    2. **Markdown fence extraction**: Extract content from ```json {...}``` blocks
       (common for LLMs instructed to format JSON in fences).
    3. **Widest brace substring**: Extract the substring from the first '{' to
       the last '}' (tolerates preamble before and trailing text after the object).

    Returns the first valid JSON object encountered. If no strategy succeeds,
    returns an empty dict (safe fallback for downstream code).

    This function is used throughout IterRet to extract structured judgments,
    decisions, and rankings from LLM outputs without requiring perfect formatting.

    Args:
        raw: The raw LLM response text (may be empty, None, or non-string).
            Treated as raw string; no pre-processing applied before extraction.

    Returns:
        Parsed JSON object as dict[str, Any]. If parsing fails or input is
        invalid, returns {} (empty dict). Callers should check for empty dict
        or use .get() with defaults to handle extraction failure gracefully.

    Example:
        >>> # Standard LLM fence format
        >>> parse_json_object('```json\\n{"correct": true, "reason": "Match"}\\n```')
        {'correct': True, 'reason': 'Match'}

        >>> # LLM with preamble and trailing text
        >>> raw = '''Here's my judgment:\\n{"correct": false}\\nDone.'''
        >>> parse_json_object(raw)
        {'correct': False}

        >>> # Malformed input: returns empty dict
        >>> parse_json_object("no json here")
        {}

        >>> # Invalid input types
        >>> parse_json_object(None)
        {}
        >>> parse_json_object("")
        {}

    Raises:
        None: This function never raises exceptions. Invalid input or parsing
            failure results in returning {} (empty dict).

    See also:
        judge_answer: Uses parse_json_object to extract LLM judgment verdicts.
    """
    if not raw or not isinstance(raw, str):
        return {}

    candidates = [raw.strip()]

    fence_match = _FENCE_RE.search(raw)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    # Fall back to the widest brace-delimited substring (handles preamble /
    # trailing commentary around an otherwise-valid object).
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}
