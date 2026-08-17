# Memory Module

Core memory system: experience banks, graph retrieval, and state management for IterRet's closed-loop memory reconstruction.

## Architecture

**Experience Banks** (`experience_bank.py`)

Episodic stores of past retrieval decisions (Planning and Reflection experiences) indexed by condition and situation. Used to guide traversal via few-shot prompting.

**Graph Retrieval** (`nodes.py`)

Implements the closed-loop controller:

- `planning_node`: Extract query cues, refresh active tags, select traversal actions
- `routing_node` (alias `reflection_node`): Prune/merge content, judge gap resolution, propose refined query
- `answer_node`: Synthesize final answer from accumulated evidence

**Graph Structure** (`ctc_graph.py`)

Cue-Tag-Content memory graph where content retrieval is mediated by:

- Cues (entities, predicates)
- Tags (relational patterns)
- Content layers (episodic/semantic/topic)

## Key Classes

- **`ExperienceBank`**: Holds Planning/Reflection entries. Supports retrieval by semantic similarity (pluggable embedding backend).
- **`EmbeddingBackend`**: Abstract interface for embedding strategies.
- **`IterRetState`**: Full closed-loop state (evidence, gaps, active set, trajectory, iteration count).
- **`CueTagContentGraph`**: Graph structure and traversal operators.

## Main Functions

- `planning_node(state, graph, llm, bank, limits)`: Step 1 — extract cues, activate tags, select actions
- `routing_node(state, graph, llm, bank, limits)`: Step 2 — prune/merge content, update gaps, refine query
- `answer_node(state, llm)`: Synthesize final answer
- `evaluate_conversation(state, llm, ...)`: Run one full episode until answer or max iterations

## Data Flow

```
Offline (scripts/train.py):
  Bootstrap trajectories → Build experience bank → Checkpoint to JSON

Online (scripts/evaluate.py):
  Load bank + graph → Run closed-loop episodes

Per iteration:
  Planning → Retrieve → Route/Reflect → Answer or refine
```

## Dependencies

- `iterret.data.ctc_graph`: Graph traversal
- `iterret.models.llm_client`: LLM for extraction and routing
- `iterret.state`: State and trajectory types
- `sentence-transformers` (optional): Neural embedding backend
