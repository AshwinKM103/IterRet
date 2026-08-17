"""Centralized LLM prompts for closed-loop retrieval and reflection.

All system and user prompt templates used throughout IterRet for memory building,
planning, routing, and answer synthesis are defined here for consistency and reusability.
"""

# Episode extraction prompts (memory_builder.py)
EPISODE_EXTRACTION_SYSTEM_PROMPT = """episode_extraction
You build a Cue-Tag-Episode memory graph (MRAgent style) from one dialogue
turn. Read the turn and produce:
- "tag": a short phrase (<=4 words) summarizing the relational pattern of
  this episode (e.g. "Pet Adoption", "Job Change").
- "cues": a list of fine-grained cues -- entities, names, attributes, or
  salient descriptors explicitly mentioned in the turn (e.g. speaker name,
  named entities, key nouns). 2-6 cues.
Reply as JSON: {"tag": str, "cues": [str, ...]}.
"""

# Semantic extraction prompts (memory_builder.py)
SEMANTIC_EXTRACTION_SYSTEM_PROMPT = """semantic_extraction
You extract stable, entity-anchored semantic facts (MRAgent Cue-Tag-Semantic
layer) from a dialogue transcript -- personal attributes, preferences, or
general facts that persist across episodes rather than describing a single
event. For each fact, identify the entity-level cue it is anchored to (e.g.
a person's name), an aspect-level tag (e.g. "Preference", "Occupation",
"Personality"), and the fact content as a short sentence.
Reply as JSON: {"semantics": [{"cue": str, "tag": str, "content": str}, ...]}.
If there are no stable facts beyond what episodes already capture, reply
with {"semantics": []}.
"""

# Topic abstraction prompts (memory_builder.py)
TOPIC_ABSTRACTION_SYSTEM_PROMPT = """topic_abstraction
You group episodic memory units (MRAgent Abstraction layer) into topic
nodes. Given a list of episodes (id, tag, text), identify recurring themes
shared across two or more episodes and summarize each as a topic.
Reply as JSON: {"topics": [{"topic": str, "episode_ids": [str, ...]}, ...]}.
Every episode_ids entry must be one of the given episode ids. Skip topics
that would only contain a single episode.
"""

# Situation abstraction prompts (nodes.py)
SITUATION_SYSTEM_PROMPT = """situation_abstraction
Abstract the given condition into a short, general situation description
(no concrete entity names), so it can be matched against past experience.
Reply as JSON: {"situation": str}.
"""

# Traversal action-selection prompts (nodes.py)
ACTION_SELECTION_SYSTEM_PROMPT = """action_selection
You control traversal over a Cue-Tag-Content memory graph (MRAgent style).
Given the query, the current active set, any retrieved Planning experience
advice, and the current information_gaps (G), choose which traversal
action(s) to take this round: "cue_to_tag", "tag_to_content", and/or
"content_to_cue_tag". Also list any tags that should be excluded --
"exclude_tags" is how paths get pruned BEFORE their content is loaded, so
exclude any tag whose likely content is inconsistent with (i.e. unlikely to
help resolve) information_gaps, not just tags the experience advice warns
against.
In most rounds you should select BOTH "cue_to_tag" AND "tag_to_content"
together: activating tags without then fetching the content behind them
makes no progress and wastes a round. The active_set's tags/cues are
already ordered with the most relevant ones first, so it is normal (and
expected) for later entries to look unrelated -- do not let that stop you
from selecting "tag_to_content" once any clearly relevant tag is present.
Only select "cue_to_tag" alone if there are currently no tags at all yet.
Reply as JSON: {"actions": [str, ...], "exclude_tags": [str, ...]}.
"""

# Cue extraction prompts (nodes.py)
CUE_EXTRACTION_SYSTEM_PROMPT = """cue_extraction
Given the current query, the evidence gathered so far, and the remaining
information gaps, extract retrieval cue terms -- named entities, time
expressions, and causal predicates -- worth indexing into a Cue-Tag-Content
memory graph for the NEXT retrieval round. Prioritize terms tied to the
unresolved gaps, not just the original query text.
Reply as JSON: {"cues": [str, ...]}.
"""

# Route/reflect prompts (nodes.py)
ROUTING_AND_REFLECTION_SYSTEM_PROMPT = """routing_and_reflection
You perform f_route (prune/merge newly retrieved content into evidence)
and f_reflect (judge whether gaps remain and propose a refined query) for
an active memory reconstruction loop (MRAgent + MemR3 style), informed by
retrieved Reflection experience advice.
"new_content" is a list of {"id": str, "text": str} objects -- each "id"
is that item's unique content id. "kept_content_ids" MUST be either the
literal string "ALL", or a list containing only those "id" values for the
items worth keeping as evidence (never the "text" itself, and never an id
that isn't in "new_content").
Reply as JSON: {"kept_content_ids": "ALL" or [str, ...], "resolved_gaps": [str, ...],
"new_gaps": [str, ...], "next_query": str}.
"""

# Final answer synthesis prompts (nodes.py)
FINAL_ANSWER_SYSTEM_PROMPT = """final_answer
Synthesize a final answer to the question using ONLY the provided evidence
bullets. Do not reference gaps, search trajectory, or graph internals.
"""

# Router decision prompts (nodes.py)
ROUTER_DECISION_SYSTEM_PROMPT = """router_decision
You are the routing decision maker in an active memory reconstruction loop.
Given the current query, accumulated evidence, information gaps, iteration budget,
and any retrieved experience_advice (known high-quality behaviors and known
failure modes from past trajectories), decide whether to retrieve more evidence
or answer now with what's available. Let experience_advice bias your decision
when it directly speaks to a situation like this one.

Choose "retrieve" if the gaps are worth chasing further given the iteration budget.
Choose "answer" if you judge the current evidence is sufficient or the gaps are less important.

Reply as JSON: {"action": "retrieve" | "answer", "reasoning": "..."}.
"""

# Live rubric scoring prompts (live_rubric.py)
LIVE_RUBRIC_SYSTEM_PROMPT = """live_rubric_scoring
You are a rubric evaluator for a real-time memory deep search system (COLM).
Score the current Reflect step decision along rubric dimensions,
0-3 points each. Focus on: how targeted the refined query is (query
specificity), how much of the question the evidence answers (evidence
coverage), and whether the gaps list contains redundant/overlapping items
(gap-gap redundancy).
Reply as JSON: {"rubrics": {<dimension>: int, ...}, "reason": str}.
"""

# Rubric evaluator prompts (evaluator.py)
RUBRIC_EVALUATOR_SYSTEM_PROMPT = """rubric_evaluation
You are an expert evaluator for an AI memory deep search system (R2-Mem style).
Score the given step against its module's rubric dimensions, 0-3 points
each, and give a short reason plus actionable advice for future steps.
Reply as JSON: {"module": str, "rubrics": {<dimension>: int, ...}, "reason_and_advice": str}.
"""

# Answer-judge prompts (llm_judge.py)
ANSWER_JUDGE_SYSTEM_PROMPT = """answer_judge
You are grading whether a predicted answer correctly answers a question,
given a reference (gold) answer. Be lenient about paraphrasing, formatting,
units, and partial vs. full names/dates -- mark it correct if it conveys
the same factual content as the reference answer. Mark it incorrect if it
is missing, contradicts the reference, or says the information could not
be found while the reference shows it was answerable.
Reply as JSON: {"correct": true or false, "reason": str}.
"""

# Experience distillation prompts (learner.py); {framing} is filled in by the
# caller with the high- or low-quality framing text before the LLM call.
EXPERIENCE_DISTILLATION_SYSTEM_PROMPT = """experience_distillation
You are an AI TRACE Strategist/Auditor (R2-Mem style). {framing}
Derive a GENERALIZABLE experience from this judged trajectory step. When
`live_rubric_scores` is present (COLM per-dimension scores -- query
specificity, evidence coverage, gap-gap redundancy -- captured live while
the step actually ran), use it as additional signal for which aspect of the
step most needs reinforcing (if scores are high) or correcting (if low);
it is not a replacement for the 8-dimension rubric_evaluation judgment
already given, only an extra, real-time-captured perspective on it. The
output experience must follow the form: IF <abstract situation> THEN
<strategy>. Do not copy concrete surface facts from the trace; treat the
diagnosed evaluation reason as authoritative supervision.
Reply as JSON: {{"situation": str, "experience": str}}.
"""
