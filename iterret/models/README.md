# Models Module

LLM client abstraction for IterRet's closed-loop memory reconstruction.

## Overview

The models module provides a pluggable LLM interface for all text-generation tasks in IterRet:

- Condition abstraction (`abstract_situation`)
- Cue extraction (`extract_cue_terms`)
- Action selection (`select_actions_for_traversal`)
- Content routing and reflection (`route_and_reflect`)
- Final answer synthesis (`synthesize_answer`)

## Key Classes

### `LLMClient` (Abstract)

Base interface for all LLM implementations:

```python
def chat(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str
```

Implementations **must**:

- Accept system prompt + user prompt
- Support optional `temperature` parameter (default 0.0 for deterministic outputs)
- Return raw text content of the model's response
- Raise `RuntimeError` if dependencies are missing or the endpoint fails

### `OpenAICompatibleLLMClient`

Production client for OpenAI-compatible endpoints (vLLM, LM Studio, Ollama with compatible API):

- **Constructor**:
  - `base_url`: Endpoint (resolves from `ITERRET_LLM_BASE_URL` env var or CLI override)
  - `model`: Model name (resolves from `ITERRET_LLM_MODEL` env var or CLI override)
  - `api_key`: Auth token (default: `"not-needed"` for local vLLM)
  - `max_tokens`: Response truncation limit (default: 1024)

- **Error Handling**: Never raises on construction; only surfaces missing `openai` package or connection failure when `chat()` is first called, allowing graceful fallback.

### `MockLLMClient`

Deterministic test double: stateful, returns scripted responses. Used in offline trajectory collection and testing.

## Configuration

LLM selection and parameters are configured via `config.yaml`'s `llm` section (see `docs/config-guide.md`):

```bash
# Use mock LLM (default)
python scripts/train.py llm=mock

# Use OpenAI-compatible endpoint
python scripts/train.py llm=openai_compatible llm.base_url=http://localhost:8000/v1 llm.model=Qwen/Qwen3-4B-Instruct-2507
```

Environment variables (`ITERRET_LLM_BASE_URL`, `ITERRET_LLM_MODEL`) provide defaults if not overridden in config.

## Dependencies

- `openai` package (optional, required only for `OpenAICompatibleLLMClient`)
- Environment: `ITERRET_LLM_BASE_URL`, `ITERRET_LLM_MODEL` (optional, fall back to defaults)
