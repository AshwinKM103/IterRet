"""Data loading and memory graph construction for IterRet.

This package handles two critical tasks:
  1. Loading external datasets (LoCoMo conversations) and formatting as dialogue turns
  2. Building Cue-Tag-Content (CTC) memory graphs from raw dialogue using LLM-guided extraction

Key modules:
  - ctc_graph: Graph data structures and traversal operators (cues, tags, content nodes)
  - memory_builder: LLM-based episodic, semantic, and topic layer extraction
  - locomo_data: LoCoMo dataset loading and conversation parsing

Typical usage:

    from iterret.data import memory_builder, locomo_data
    from iterret.models.llm_client import OpenAICompatibleLLMClient

    raw_conversations = locomo_data.load_raw_locomo("data.json")
    for raw_conv in raw_conversations:
        parsed = locomo_data.parse_conversation(raw_conv)
        llm = OpenAICompatibleLLMClient()
        graph = memory_builder.build_ctc_graph_from_dialogue(
            parsed["turns"], llm
        )
"""
