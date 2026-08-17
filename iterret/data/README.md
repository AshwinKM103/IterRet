# Data Module

Graph construction and dataset loading for IterRet's memory system.

## Key Components

**CTC Graph** (`ctc_graph.py`)

In-memory graph with:

- Cues (named entities, predicates)
- Content nodes (episodic/semantic/topic layers)
- Edges: cue→tag→content bidirectional mappings
- Traversal operators: `cue_to_tag()`, `tag_to_content()`, `content_to_cue_tag()`, etc.

**Memory Builder** (`memory_builder.py`)

Extracts graph structure from raw dialogue via LLM-guided extraction:

- `build_cue_tag_content_graph(turns, llm)`: End-to-end construction
  1. Extract episodes (tag + cues per turn)
  2. Extract semantics (entity-anchored facts)
  3. Extract topics (thematic grouping)
  4. Build graph edges

- `_extract_episode(turn, llm)`: Parse dialogue turn → tag + cues
- `_extract_semantics(turns, llm)`: Extract stable facts
- `_extract_topics(episodes, llm)`: Group episodes by themes
- `_iter_chunks(items, text_key, max_chars)`: Batch items for token budget

**Dataset Loading** (`locomo_data.py`)

LoCoMo-specific loading:

- `resolve_locomo_path(subset, format)`: Resolve dataset path
- `load_locomo_conversations(subset, format)`: Load from disk
- `iter_locomo_eval_conversations(subset)`: Iterator with caching

## Content Layers

- `episodic`: Single dialogue turn with timestamp
- `semantic`: Stable entity-anchored facts (preferences, attributes)
- `topic`: Recurring themes grouping 2+ episodes

## Data Flow

```
Raw dialogue turns
  → Memory builder (LLM extraction)
  → CTC graph
  → Traversal operators
  → Retrieval during online loop
```

## Dependencies

- `iterret.models.llm_client`: LLM for extraction
- `iterret.utils.json_utils`: JSON parsing
