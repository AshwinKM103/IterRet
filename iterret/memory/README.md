# Memory Module

Core memory system: experience banks, graph retrieval, and state management for IterRet's closed-loop memory reconstruction.

## Overview

The memory module implements a two-tier memory architecture:

1. **Experience Banks** (`experience_bank.py`): Episodic stores of past retrieval decisions (Planning and Reflection experiences) indexed by condition and situation. Used to guide traversal via few-shot prompting during graph traversal.

2. **Graph Retrieval** (`nodes.py`): Implements the closed-loop controller (COLM §1.4.3):
   - **`planning_node`**: Extracts query cues, refreshes active tags, selects traversal actions
   - **`routing_node`** (alias `reflection_node`): Prunes/merges new content, judges gap resolution, proposes refined query
   - **`answer_node`**: Synthesizes final answer from accumulated evidence

3. **Graph Structure** (`ctc_graph.py`, embedded): A Cue-Tag-Content memory graph where content retrieval is mediated by named cues (entities, predicates) → tags (relational patterns) → episodic/semantic/topic layers.

## Key Classes

- **`ExperienceBank`**: Holds Planning and Reflection experience entries. Supports retrieval by semantic similarity (pluggable embedding backend: SentenceTransformer or keyword-overlap fallback).
- **`EmbeddingBackend`**: Abstract interface for embedding strategies (used by ExperienceBank for experience retrieval).
- **`IterRetState` (in `state.py`)**: TypedDict holding the full closed-loop state: accumulated evidence, gaps, active set, search trajectory, iteration count.
- **`CueTagContentGraph`** (in `data/`): Graph structure and traversal operators (cue→tag, tag→content, content→cue+tag).

## Main Functions

- **`planning_node(state, graph, llm, bank, limits)`**: Step 1 (COLM §1.4.3): extract cues, activate tags, select traversal actions.
- **`routing_node(state, graph, llm, bank, limits)`**: Step 2 (COLM §1.4.3): prune/merge content into evidence, update gaps, refine query.
- **`answer_node(state, llm)`**: Synthesize final answer from accumulated evidence.
- **`evaluate_conversation(state, llm, ...)`**: Run one full closed-loop episode: initialize state, loop plan→retrieve→reflect until answer or max iterations reached.

## Data Flow

1. **Offline phase** (`scripts/train.py`): Build experience bank from bootstrap trajectories, checkpoint to JSON.
2. **Online phase** (`scripts/evaluate.py`, `run_locomo_eval.py`): Load bank + graph, run closed-loop episodes.
3. **Within a loop iteration**:
   - Planning: extract cues, activate tags from graph, retrieve Planning experiences
   - Retrieve: follow active traversal actions (cue→tag, tag→content, etc.)
   - Route/Reflect: merge new content, retrieve Reflection experiences, judge gaps, refine query or answer

## Dependencies

- `iterret.data.ctc_graph`: CueTagContentGraph for traversal
- `iterret.models.llm_client`: LLMClient for condition abstraction, action selection, routing
- `iterret.state`: IterRetState, SearchStep types
- `sentence-transformers` (optional): For neural embedding backend in ExperienceBank
