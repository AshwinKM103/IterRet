"""Answer evaluation metrics for QA tasks.

Implements token-level F1 score and precision/recall computation via
token overlap, used to score final answers in closed-loop episodes.
"""

from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_answer(text: str) -> list[str]:
    """Normalize answer text to lowercase tokens for token-level metrics.

    Converts text to lowercase and removes all punctuation, leaving only
    alphanumeric tokens separated by whitespace. This normalization is a
    prerequisite for token-level F1 and precision/recall computation, ensuring
    that "The answer." and "the answer" are treated as identical.

    Args:
        text: The answer text to normalize. Converted to string if non-string
            input is provided.

    Returns:
        List of lowercase alphanumeric tokens (split by whitespace). May be
        empty if input is blank or contains only punctuation.

    Example:
        >>> normalize_answer("What is AI?")
        ['what', 'is', 'ai']
        >>> normalize_answer("42")
        ['42']
        >>> normalize_answer("!!!???")
        []

    See also:
        token_f1: Uses normalized tokens to compute F1 score.
        precision_recall_f1: Uses normalized tokens for all three metrics.
    """
    text = _PUNCT_RE.sub(" ", str(text).lower())
    return text.split()


def token_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 score between prediction and gold answer.

    Computes the harmonic mean of precision and recall at the token level,
    using bag-of-tokens comparison after normalization. This metric rewards
    systems that retrieve the correct factual content, regardless of word order
    or paraphrasing. Unlike exact-match or BLEU, F1 is more lenient and aligns
    better with QA evaluation (used in SQuAD, MS-MARCO, etc.).

    The computation:
    1. Normalize both prediction and gold to lowercase tokens.
    2. Count token overlaps (recall is overlap / gold_tokens; precision is
       overlap / pred_tokens).
    3. Compute F1 = 2 * P * R / (P + R).

    Args:
        prediction: The predicted answer text (any length). Non-string inputs
            are converted to string.
        gold: The reference (gold) correct answer text. Used as the normative
            set for recall computation.

    Returns:
        F1 score as float in [0, 1]:
            - 1.0: Perfect match (identical tokens).
            - 0.5: 50% precision and recall.
            - 0.0: No overlapping tokens.

    Example:
        >>> token_f1("artificial intelligence", "AI")
        0.0  # No overlapping tokens after normalization
        >>> token_f1("the capital of France is Paris", "Paris")
        0.5  # 1 overlap, pred has 6 tokens, gold has 1: F1 = 2*(1/6)*(1/1) / ((1/6) + 1)
        >>> token_f1("Paris", "Paris")
        1.0  # Perfect match

    Raises:
        None: This function does not raise exceptions. Returns 0.0 for edge cases
            (empty predictions or gold answers).

    See also:
        precision_recall_f1: Returns all three metrics (P, R, F1) in one call.
        normalize_answer: Tokenization and normalization logic.
    """
    return precision_recall_f1(prediction, gold)[2]


def precision_recall_f1(prediction: str, gold: str) -> tuple[float, float, float]:
    """Compute token-level precision, recall, and F1 score.

    Implements bag-of-tokens comparison: normalizes both prediction and gold
    to lowercase tokens, counts overlapping tokens (up to multiplicity), and
    computes precision (overlap / pred_tokens), recall (overlap / gold_tokens),
    and their harmonic mean (F1).

    This metric is standard in QA evaluation and favors systems that retrieve
    correct factual content. It is more lenient than exact-match but stricter
    than BLEU, making it appropriate for factual QA where token order is less
    important than content accuracy.

    The implementation handles edge cases gracefully:
    - Both empty: returns (1.0, 1.0, 1.0) — both are "correct" vacuously.
    - Only prediction empty: returns (0.0, 0.0, 0.0) — missing prediction.
    - Only gold empty: returns (0.0, 0.0, 0.0) — spurious prediction.
    - Tokens repeat: counts each token occurrence up to multiplicity in the gold.

    Args:
        prediction: The predicted answer text. Normalized to lowercase tokens
            and used as the set from which precision is computed.
        gold: The reference (gold) correct answer text. Normalized to lowercase
            tokens and used as the set from which recall is computed.

    Returns:
        Tuple of (precision, recall, f1), each in [0, 1]:
            - precision: fraction of predicted tokens that appear in gold.
            - recall: fraction of gold tokens that appear in prediction.
            - f1: harmonic mean (2 * P * R / (P + R), or 0.0 if both are 0).

    Example:
        >>> precision_recall_f1("Paris France", "Paris")
        (0.5, 1.0, 0.6666...)  # Pred has 2 tokens (1 overlap); gold has 1
        >>> precision_recall_f1("The answer is 42", "42")
        (0.25, 1.0, 0.4)  # 1 overlap / 4 predicted vs. 1 overlap / 1 gold
        >>> precision_recall_f1("", "")
        (1.0, 1.0, 1.0)  # Both empty: vacuous success

    Raises:
        None: This function does not raise exceptions. Non-string inputs are
            converted to strings; empty results are handled as edge cases above.

    See also:
        token_f1: Returns only the F1 score (the third return value).
        normalize_answer: Tokenization logic underlying this computation.
    """
    pred_tokens = normalize_answer(prediction)
    gold_tokens = normalize_answer(gold)
    if not pred_tokens or not gold_tokens:
        f1 = 1.0 if not pred_tokens and not gold_tokens else 0.0
        return f1, f1, f1

    pred_counts: dict = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    overlap = 0
    for tok in gold_tokens:
        if pred_counts.get(tok, 0) > 0:
            overlap += 1
            pred_counts[tok] -= 1

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1
