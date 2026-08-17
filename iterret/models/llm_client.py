"""LLM client abstraction for IterRet.

Provides pluggable LLM backends: OpenAI-compatible endpoints (vLLM, LM Studio,
Ollama), and a deterministic mock for testing and offline trajectory collection.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

from config.constants import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MAX_TOKENS, DEFAULT_LLM_MODEL

# Defaults match `vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000 ...`.
# The "model" field in chat-completions requests MUST equal the name vLLM
# reports at /v1/models (the full HF repo id, unless --served-model-name
# was passed), so these are overridable via CLI flags or env vars rather
# than hardcoded -- see resolve_llm_base_url / resolve_llm_model below.
ENV_LLM_BASE_URL = "ITERRET_LLM_BASE_URL"
ENV_LLM_MODEL = "ITERRET_LLM_MODEL"


def resolve_llm_base_url(cli_value: str | None = None) -> str:
    """Resolve the LLM endpoint URL.

    Priority: CLI value > environment variable > default.

    Args:
        cli_value: Value passed on CLI (e.g., via Hydra config override).

    Returns:
        The resolved base URL.
    """
    return cli_value or os.environ.get(ENV_LLM_BASE_URL) or DEFAULT_LLM_BASE_URL


def resolve_llm_model(cli_value: str | None = None) -> str:
    """Resolve the LLM model name.

    Priority: CLI value > environment variable > default.

    Args:
        cli_value: Value passed on CLI (e.g., via Hydra config override).

    Returns:
        The resolved model name.
    """
    return cli_value or os.environ.get(ENV_LLM_MODEL) or DEFAULT_LLM_MODEL


class LLMClient(ABC):
    """Abstract base class for LLM implementations.

    All implementations must support the chat interface for condition abstraction,
    cue extraction, routing, and other decision steps in the closed loop.
    """

    @abstractmethod
    def chat(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        """Send a chat message and return the model's response.

        Args:
            system_prompt: The system role prompt.
            user_prompt: The user message.
            temperature: Sampling temperature (default 0.0).

        Returns:
            The raw text content of the model's reply.

        Raises:
            RuntimeError or similar if the backend cannot process the request.
        """


class OpenAICompatibleLLMClient(LLMClient):
    """Wraps an OpenAI-compatible chat-completions endpoint.

    Construction never raises even if the ``openai`` package is missing --
    that only surfaces as a clear RuntimeError the first time ``chat()`` is
    actually called, so importing this module stays side-effect free.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str = "not-needed",
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    ) -> None:
        """Initialize an OpenAI-compatible LLM client.

        Never raises on construction (lazy loading): only the first call to
        chat() will surface missing dependencies or connection failures.

        Args:
            base_url: Base URL of the OpenAI-compatible endpoint
                (resolves from ITERRET_LLM_BASE_URL or default).
            model: Model name (resolves from ITERRET_LLM_MODEL or default).
            api_key: API authentication key (default "not-needed" for local vLLM).
            max_tokens: Maximum tokens in response (default 1024).
        """
        self.base_url = resolve_llm_base_url(base_url)
        self.model = resolve_llm_model(model)
        self.api_key = api_key
        self.max_tokens = max_tokens
        self._client = None
        try:
            import openai  # noqa: F401
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=api_key)
        except Exception:
            self._client = None

    def chat(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        """Send a chat message and return the model's response.

        Args:
            system_prompt: The system prompt (role: system).
            user_prompt: The user message (role: user).
            temperature: Sampling temperature (default 0.0 for deterministic).

        Returns:
            The raw text content of the model's response.

        Raises:
            RuntimeError: If the openai package is missing or endpoint unreachable.
        """
        if self._client is None:
            raise RuntimeError(
                "OpenAICompatibleLLMClient has no usable client (the `openai` package is "
                "missing or the endpoint could not be reached). Install `openai` and point "
                f"--llm-base-url at a running server (tried {self.base_url})."
            )
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""


class MockLLMClient(LLMClient):
    """Deterministic, stateful mock so the whole pipeline (offline + online)
    runs without a real model. It inspects the prompt for a handful of
    keywords this package's own modules use and returns plausible JSON.
    """

    def __init__(self) -> None:
        self._call_count = 0

    def chat(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str:
        self._call_count += 1
        text = system_prompt + "\n" + user_prompt

        if "situation_abstraction" in text:
            return self._situation_reply(user_prompt)
        if "cue_extraction" in text:
            return self._cue_extraction_reply(user_prompt)
        if "action_selection" in text:
            return self._action_selection_reply(user_prompt)
        if "routing_and_reflection" in text:
            return self._routing_reply(user_prompt)
        if "live_rubric_scoring" in text:
            return self._live_rubric_reply(user_prompt)
        if "rubric_evaluation" in text:
            return self._evaluation_reply(user_prompt)
        if "experience_distillation" in text:
            return self._distillation_reply(user_prompt)
        if "final_answer" in text:
            return self._answer_reply(user_prompt)
        if "episode_extraction" in text:
            return self._episode_extraction_reply(user_prompt)
        if "semantic_extraction" in text:
            return self._semantic_extraction_reply(user_prompt)
        if "topic_abstraction" in text:
            return self._topic_abstraction_reply(user_prompt)
        if "answer_judge" in text:
            return self._judge_reply(user_prompt)
        return json.dumps({})

    # -- canned replies, shaped to match what each caller parses --------
    def _situation_reply(self, user_prompt: str) -> str:
        try:
            condition = json.loads(user_prompt).get("condition", "")
        except (json.JSONDecodeError, AttributeError):
            condition = user_prompt
        return json.dumps({"situation": f"a query situation abstracted from: {condition[:60]}"})

    def _cue_extraction_reply(self, user_prompt: str) -> str:
        try:
            payload = json.loads(user_prompt)
        except (json.JSONDecodeError, TypeError):
            payload = {}
        query = str(payload.get("query", ""))
        gaps = payload.get("gaps", [])
        terms = [word for word in query.split() if len(word) > 3]
        for gap in gaps:
            terms.extend(word for word in str(gap).split() if len(word) > 3)
        seen, deduped = set(), []
        for term in terms:
            if term not in seen:
                seen.add(term)
                deduped.append(term)
        return json.dumps({"cues": deduped[:10]})

    def _action_selection_reply(self, user_prompt: str) -> str:
        actions = ["cue_to_tag", "tag_to_content"]
        if self._call_count > 4:
            actions = ["content_to_cue_tag"]
        return json.dumps({"actions": actions, "exclude_tags": []})

    def _routing_reply(self, user_prompt: str) -> str:
        # progressively resolve gaps so the mock demo converges in a few iterations
        resolved_all = self._call_count >= 3
        return json.dumps(
            {
                "kept_content_ids": "ALL",
                "resolved_gaps": ["initial: no evidence gathered yet"]
                if self._call_count == 1
                else [],
                "new_gaps": [] if resolved_all else ["need more supporting detail"],
                "next_query": "" if resolved_all else "additional supporting detail",
            }
        )

    def _live_rubric_reply(self, user_prompt: str) -> str:
        try:
            dimensions = json.loads(user_prompt).get("rubric_dimensions", [])
        except (json.JSONDecodeError, AttributeError):
            dimensions = []
        per_dimension_score = 2
        rubrics = {dimension: per_dimension_score for dimension in dimensions}
        return json.dumps(
            {
                "rubrics": rubrics,
                "reason": f"mock live rubric reason for step {self._call_count}",
            }
        )

    def _evaluation_reply(self, user_prompt: str) -> str:
        # alternate good/bad dimension scores so both quality branches get exercised offline
        try:
            dimensions = json.loads(user_prompt).get("rubric_dimensions", [])
        except (json.JSONDecodeError, AttributeError):
            dimensions = []
        per_dimension_score = 3 if self._call_count % 2 == 0 else 1
        rubrics = {dimension: per_dimension_score for dimension in dimensions}
        return json.dumps(
            {
                "rubrics": rubrics,
                "reason_and_advice": f"mock rubric reason and advice for step {self._call_count}",
            }
        )

    def _distillation_reply(self, user_prompt: str) -> str:
        return json.dumps(
            {
                "situation": "a query requiring iterative graph traversal",
                "experience": "IF the situation is ambiguous THEN prefer cue_to_tag before "
                "tag_to_content and avoid re-expanding already-visited tags",
            }
        )

    def _answer_reply(self, user_prompt: str) -> str:
        return "Based on the gathered evidence, here is the answer (mock LLM)."

    def _episode_extraction_reply(self, user_prompt: str) -> str:
        try:
            turn = json.loads(user_prompt)
        except (json.JSONDecodeError, TypeError):
            turn = {}
        text = str(turn.get("text", ""))
        speaker = str(turn.get("speaker", ""))
        # crude proper-noun heuristic: capitalized words that aren't the first word
        words = text.replace(",", " ").replace(".", " ").split()
        cues = [w for i, w in enumerate(words) if i > 0 and w[:1].isupper()]
        if speaker:
            cues.append(speaker)
        seen, deduped = set(), []
        for cue in cues:
            if cue not in seen:
                seen.add(cue)
                deduped.append(cue)
        return json.dumps({"tag": "Mention", "cues": deduped[:6] or [speaker or "Unknown"]})

    def _semantic_extraction_reply(self, user_prompt: str) -> str:
        first_line = user_prompt.splitlines()[0] if user_prompt else ""
        speaker = first_line.split(":")[0].strip() if ":" in first_line else "Unknown"
        return json.dumps(
            {
                "semantics": [
                    {
                        "cue": speaker,
                        "tag": "General",
                        "content": f"{speaker} appears across this conversation.",
                    }
                ]
            }
        )

    def _judge_reply(self, user_prompt: str) -> str:
        try:
            parsed = json.loads(user_prompt)
        except (json.JSONDecodeError, TypeError):
            parsed = {}
        predicted = str(parsed.get("predicted_answer", "")).lower()
        correct = (
            bool(predicted)
            and "cannot be determined" not in predicted
            and "mock llm" not in predicted
        )
        return json.dumps({"correct": correct, "reason": "mock heuristic judge"})

    def _topic_abstraction_reply(self, user_prompt: str) -> str:
        try:
            episodes = json.loads(user_prompt).get("episodes", [])
        except (json.JSONDecodeError, AttributeError):
            episodes = []
        ids = [e["content_id"] for e in episodes if isinstance(e, dict) and "content_id" in e]
        if len(ids) < 2:
            return json.dumps({"topics": []})
        return json.dumps({"topics": [{"topic": "General conversation", "episode_ids": ids}]})
