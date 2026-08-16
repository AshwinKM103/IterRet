# Utils Module

Utility functions for metrics, LLM evaluation, and JSON parsing.

## Overview

The utils module provides cross-cutting helpers used throughout IterRet:

- **JSON parsing**: Robust extraction of JSON objects from LLM outputs
- **Metrics**: Rubric-based evaluation of retrieval and reasoning quality
- **LLM judging**: Judge-based scoring for dialogue quality and relevance

## Key Functions

### JSON Utilities (`json_utils.py`)

**`parse_json_object(text: str) -> dict[str, Any]`**

Extracts a JSON object from LLM response text (which may contain markdown, surrounding text, or malformed JSON):

- Looks for the first `{...}` substring matching balanced braces
- Parses as JSON; returns empty dict `{}` on parse failure (never raises)
- Used throughout memory building and routing to handle LLM JSON output robustly

### Metrics (`metrics.py`)

**`compute_retrieval_metrics(predictions, references, k_values) -> dict[str, float]`**

Compute standard retrieval metrics:

- **Recall@k**: Fraction of relevant items in top-k
- **NDCG@k**: Normalized Discounted Cumulative Gain (rank-aware relevance)
- **MRR**: Mean Reciprocal Rank (position of first relevant item)

Returns averaged scores over all queries; supports customizable `k` values (e.g., `[1, 3, 5]`).

### LLM Judge (`llm_judge.py`)

**`judge_response_quality(response: str, llm: LLMClient, *, rubric: dict[str, str]) -> dict[str, int]`**

Score a response against a multi-dimensional rubric using an LLM judge:

- Accepts custom rubric as dict of dimension → description
- Returns dimension → score mapping
- Used by `live_rubric.score_reflect_step()` to evaluate routing decisions in real-time

## Type Hints

- **`SearchStep`** (from `state.py`): Trajectory entry with iteration, module, query, action, rubric scores
- **`TraversalLimits`** (from `config.py`): Active-set size constraints per round

## Dependencies

- `iterret.models.llm_client`: LLMClient for judge-based scoring
- External: None (no external metrics libraries; custom implementations)
