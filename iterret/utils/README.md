# Utils Module

Utility functions for metrics, LLM evaluation, and JSON parsing.

## Key Functions

**JSON Parsing** (`json_utils.py`)

`parse_json_object(text: str) -> dict[str, Any]`

Extracts a JSON object from LLM response text (may contain markdown, surrounding text, or malformed JSON):

- Finds first `{...}` substring with balanced braces
- Parses as JSON; returns empty dict `{}` on failure (never raises)
- Used throughout memory building and routing

**Metrics** (`metrics.py`)

`compute_retrieval_metrics(predictions, references, k_values) -> dict[str, float]`

Compute standard retrieval metrics:

- **Recall@k**: Fraction of relevant items in top-k
- **NDCG@k**: Normalized Discounted Cumulative Gain (rank-aware)
- **MRR**: Mean Reciprocal Rank (position of first relevant item)

Returns averaged scores; supports customizable k values (e.g., `[1, 3, 5]`).

**LLM Judge** (`llm_judge.py`)

`judge_response_quality(response: str, llm: LLMClient, *, rubric: dict[str, str]) -> dict[str, int]`

Score a response against a multi-dimensional rubric using an LLM judge:

- Accepts custom rubric as dict of dimension → description
- Returns dimension → score mapping
- Used by `live_rubric.score_reflect_step()` to evaluate routing decisions

## Dependencies

- `iterret.models.llm_client`: LLM for judge-based scoring
