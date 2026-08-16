# Data Module

Graph construction and dataset loading for IterRet's memory system.

## Overview

The data module builds and loads Cue-Tag-Content (CTC) memory graphs from dialogue or conversational data. It handles both structured graph construction and dataset-specific loading (currently: LoCoMo conversations).

## Key Components

### CTC Graph (`ctc_graph.py`)

Implements the memory graph structure and traversal operators:

- **`CueTagContentGraph`**: In-memory graph with:
  - Cues (named entities, predicates)
  - Content nodes (episodic/semantic/topic layers)
  - Edges: cue→tag→content bidirectional mappings
  - Traversal operators: `cue_to_tag()`, `tag_to_content()`, `content_to_cue_tag()`, etc.

- **Content Layers**:
  - `episodic`: Single dialogue turn with timestamp
  - `semantic`: Stable entity-anchored facts (preferences, attributes)
  - `topic`: Recurring themes grouping 2+ episodes

### Memory Builder (`memory_builder.py`)

Extracts graph structure from raw dialogue via LLM-guided extraction:

- **`build_cue_tag_content_graph(turns, llm)`**: End-to-end graph construction:
  1. Extract episodes (tag + cues per turn)
  2. Extract semantics (entity-anchored facts)
  3. Extract topics (thematic grouping)
  4. Build graph edges

- **`_extract_episode(turn, llm)`**: Parse one dialogue turn → tag (relational pattern) + cues (entities)
- **`_extract_semantics(turns, llm)`**: Extract stable facts anchored to entities
- **`_extract_topics(episodes, llm)`**: Group episodes by recurring themes
- **`_iter_chunks(items, text_key, max_chars)`**: Batch items for token-budget-aware LLM calls

### Dataset Loading (`locomo_data.py`)

LoCoMo-specific loading and path resolution:

- **`_KNOWN_LOCOMO_PATHS`**: Hardcoded dataset paths (relative to invocation dir)
- **`resolve_locomo_path(subset, format)`**: Resolve dataset path by subset (train/test/val) and format (json/jsonl/csv)
- **`load_locomo_conversations(subset, format)`**: Load conversations from disk
- **`iter_locomo_eval_conversations(subset)`**: Iterator for evaluation (handles caching)

## Data Flow

```
Raw dialogue turns → Memory builder (LLM extraction) → CTC graph
→ Traversal operators → Retrieval during online loop
```

## Type Hints

- **`DialogueTurn`**: `TypedDict(speaker, text, time)`
- **`ContentLayer`**: `Literal["episodic", "semantic", "topic"]`
- **`CueNode`**: Cue ID + set of tags
- **`ContentNode`**: Content ID, text, layer, optional timestamp, tags, topic links

## Dependencies

- `iterret.models.llm_client`: LLMClient for extraction prompts
- `iterret.utils.json_utils`: JSON parsing utilities
- External: None (graph construction is pure Python + LLM calls)
