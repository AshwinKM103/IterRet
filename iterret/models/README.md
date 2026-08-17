# Models Module

LLM client abstraction for IterRet's closed-loop memory reconstruction.

## Key Classes

**`LLMClient` (Abstract)**

Base interface for all LLM implementations:

```python
def chat(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0) -> str
```

- Must accept system + user prompt
- Must support optional `temperature` (default 0.0 for deterministic output)
- Must return raw text response
- Raises `RuntimeError` if dependencies missing or endpoint fails

**`OpenAICompatibleLLMClient`**

Production client for OpenAI-compatible endpoints (vLLM, LM Studio, Ollama):

- `base_url`: Endpoint (env: `ITERRET_LLM_BASE_URL`)
- `model`: Model name (env: `ITERRET_LLM_MODEL`)
- `api_key`: Auth token (default: `not-needed` for local vLLM)
- `max_tokens`: Response limit (default: 1024)

Error handling deferred to first `chat()` call for graceful fallback.

**`MockLLMClient`**

Deterministic test double with scripted responses. Used for offline testing.

## Configuration

```bash
# Mock LLM (offline)
python scripts/train.py llm.type=mock

# OpenAI-compatible endpoint
python scripts/train.py llm.type=openai_compatible llm.base_url=http://localhost:8000/v1
```

Environment variables (`ITERRET_LLM_BASE_URL`, `ITERRET_LLM_MODEL`) provide defaults.

## Dependencies

- `openai` (optional, required only for `OpenAICompatibleLLMClient`)
