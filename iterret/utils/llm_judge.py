"""LLM-based answer evaluation and semantic judgment.

Provides utilities to judge whether a predicted answer correctly addresses a
question, using an LLM judge rather than token-level metrics or exact match.
This approach is significantly more lenient and semantically aware than
string-based metrics (F1, BLEU), permitting paraphrasing, reformatting, unit
conversions, and partial answers as long as the core factual content aligns.

In IterRet's closed-loop reasoning, LLM judgment is used in:
1. **Quality scoring**: Rubric-based judgment of evidence quality and relevance
   during reflection phases (COLM §1.4.2).
2. **Answer verification**: Semantic judgment of final answers against reference
   answers during episode evaluation.

The judge is implemented as a prompt-based LLM call with a system prompt that
instructs the model to be lenient about paraphrasing, formatting, and units,
and to output structured JSON (correct: bool, reason: str).

Functions:
    judge_answer: Judge whether a predicted answer is correct via LLM.

Module dependencies:
    - LLMClient: Provides chat interface to the LLM.
    - parse_json_object: Extracts JSON from the judge's response.

Example:
    >>> from iterret.utils.llm_judge import judge_answer
    >>> from iterret.models.llm_client import LLMClient
    >>> llm = LLMClient()  # initialized with model and API
    >>> is_correct = judge_answer(
    ...     question="What is the capital of France?",
    ...     gold_answer="Paris",
    ...     predicted_answer="The city of Paris",
    ...     llm=llm
    ... )
    >>> print(is_correct)  # True (semantic match)
"""

from __future__ import annotations

import json

from ..models.llm_client import LLMClient
from .json_utils import parse_json_object

_JUDGE_SYSTEM_PROMPT = """answer_judge
You are grading whether a predicted answer correctly answers a question,
given a reference (gold) answer. Be lenient about paraphrasing, formatting,
units, and partial vs. full names/dates -- mark it correct if it conveys
the same factual content as the reference answer. Mark it incorrect if it
is missing, contradicts the reference, or says the information could not
be found while the reference shows it was answerable.
Reply as JSON: {"correct": true or false, "reason": str}.
"""


def judge_answer(question: str, gold_answer: str, predicted_answer: str, llm: LLMClient) -> bool:
    """Judge whether a predicted answer is correct via semantic LLM evaluation.

    Sends a structured prompt to an LLM judge that scores whether a predicted
    answer correctly addresses a question, given a reference answer. The judge
    is instructed to be lenient about paraphrasing, formatting, units, and
    partial vs. full names/dates—marking an answer correct if it conveys the
    same factual content, and incorrect only if it is missing, contradicts the
    reference, or claims the information could not be found (while the reference
    shows it was answerable).

    This approach is used in IterRet's evaluation loop to assess final answers
    and intermediate evidence quality during reflection phases. It is more
    semantically aware than token-level F1 but more costly (requires an LLM call).

    Implementation:
    1. Structures question, gold_answer, and predicted_answer as JSON input.
    2. Calls the LLM with a lenient system prompt (temperature=0.0 for stability).
    3. Extracts JSON from the LLM response using parse_json_object.
    4. Returns the "correct" boolean field, with fallback string parsing.

    Args:
        question: The original user question (e.g., "What is the capital of
            France?"). Included in the judge prompt for full context.
        gold_answer: The reference (gold) correct answer (e.g., "Paris").
            Used as the normative standard for judgment.
        predicted_answer: The answer produced by the retrieval system
            (e.g., "The city of Paris, France"). Judged against gold_answer.
        llm: An LLMClient instance configured with the model to use for
            judgment (e.g., Claude, GPT-4). Must support the chat() method
            with system/user prompts and temperature parameter.

    Returns:
        True if the LLM deems the predicted answer correct (semantically
        matches the gold answer), False otherwise. Always returns a boolean.

    Example:
        >>> from iterret.utils.llm_judge import judge_answer
        >>> from iterret.models.llm_client import LLMClient
        >>> llm = LLMClient(model="gpt-4")
        >>> # Example 1: Paraphrased but correct
        >>> judge_answer(
        ...     question="What is the capital of France?",
        ...     gold_answer="Paris",
        ...     predicted_answer="The city of Paris",
        ...     llm=llm
        ... )
        True

        >>> # Example 2: Incorrect or missing information
        >>> judge_answer(
        ...     question="What is the capital of France?",
        ...     gold_answer="Paris",
        ...     predicted_answer="I could not find this information",
        ...     llm=llm
        ... )
        False

    Raises:
        None: This function does not raise exceptions. If the LLM response
            cannot be parsed or the "correct" field is malformed, returns a
            conservative value based on string parsing (true/yes/1/correct).

    See also:
        parse_json_object: Extracts JSON from noisy LLM response.
        token_f1: Token-level alternative (faster, but less lenient).
    """
    user_prompt = json.dumps(
        {
            "question": question,
            "reference_answer": gold_answer,
            "predicted_answer": predicted_answer,
        }
    )
    raw = llm.chat(_JUDGE_SYSTEM_PROMPT, user_prompt, temperature=0.0)
    parsed = parse_json_object(raw)
    verdict = parsed.get("correct")
    if isinstance(verdict, bool):
        return verdict
    return str(verdict).strip().lower() in ("true", "yes", "1", "correct")
