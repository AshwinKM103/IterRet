"""IterRet state and trajectory types.

Defines the full state of a closed-loop memory reconstruction episode and
trajectory tracking for visualization and evaluation.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from config.constants import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_STUCK_REFLECTS,  # noqa: F401 -- re-exported for callers
)


class SearchStep(TypedDict):
    """One step in the closed-loop search trajectory.

    Records a single iteration of the memory retrieval process: the module
    that executed (planning or reflection), the query used, the retrieval action,
    what was discovered, the router's decision, and quality judgments. Each
    SearchStep is appended to search_trajectory in IterRetState, forming a
    complete trace of the episode for visualization, debugging, and evaluation.

    This type is used to build a detailed audit log of the reasoning loop,
    enabling analysis of decision quality and retrieval effectiveness.

    Attributes:
        iteration: Iteration number (0-indexed), 0 on first step, incremented
            with each Planning or Reflection module execution.
        module: Which module produced this step: "Planning" (retrieval planning)
            or "Reflection" (evidence synthesis and gap analysis). Typed to
            enforce this binary choice (R2-Mem Eq. 6).
        query_used: The refined query passed to this module. On iteration 0,
            this is the original query; on later iterations, it may be refined
            by prior Reflection steps.
        action_taken: The retrieval action executed by this step. Examples:
            "cue_to_tag" (expand cues to tags via CTC), "tag_to_content"
            (retrieve content for selected tags), "reflect_on_gaps" (summarize
            and identify gaps).
        found_summary: Brief summary of what was found or processed in this step
            (1-2 sentences). Examples: "Retrieved 5 content nodes relevant to
            ML"; "No new gaps identified; evidence is coherent."
        decision: Router decision outcome after this step:
            - "retrieve": Proceed to another retrieval iteration.
            - "reflect": Perform reflection and gap analysis.
            - "answer": Proceed to answer synthesis.
        rubric_scores: Per-dimension quality scores for the retrieval step.
            Dict maps dimension names (e.g., "relevance", "clarity", "coherence")
            to integer scores. Used to track quality of retrieved evidence.
            Example: {"relevance": 3, "clarity": 2, "coherence": 3}.

    Example:
        >>> step = SearchStep(
        ...     iteration=0,
        ...     module="Planning",
        ...     query_used="How does transformer attention work?",
        ...     action_taken="cue_to_tag, tag_to_content",
        ...     found_summary="Retrieved 3 content nodes on attention mechanisms",
        ...     decision="retrieve",
        ...     rubric_scores={"relevance": 4, "clarity": 3}
        ... )

    See also:
        IterRetState: Contains a list of SearchStep (search_trajectory field).
    """

    iteration: int
    module: Literal["Planning", "Reflection"]  # R2-Mem step typing (Eq. 6)
    query_used: str
    action_taken: str
    found_summary: str
    decision: str  # "retrieve" | "reflect" | "answer"
    rubric_scores: dict[
        str, int
    ]  # per-dimension closed-loop memory retrieval rubric scores (retrieval quality rubric)


class IterRetState(TypedDict, total=False):
    r"""Full state of a closed-loop memory reconstruction episode (closed-loop episode state management).

    Tracks the entire state of one memory retrieval episode: the original query,
    accumulated evidence, information gaps, the active retrieval frontier, visited
    nodes, search trajectory, and routing decisions. The state machine flows through
    this state object as the closed loop executes: Planning → Retrieval → Reflection →
    Routing → (Retrieve again or Answer).

    IterRetState is a TypedDict with total=False, allowing optional fields for
    intermediate states where not all fields have been populated yet. All fields
    should be present by episode completion.

    State invariants (see closed-loop episode state management for formal definitions):
    - original_query (q): Never changes.
    - current_refined_query (q^ret_k): May be refined by Reflection steps per MemR3 Eq. 5.
    - accumulated_evidence (E_k / H^(t)): Monotonically grows; never shrinks.
    - information_gaps (G_k): Updated by Reflection; may shrink as evidence resolves gaps.
    - active_set (Z^(t)): Frontier of retrieval nodes (cues → tags → contents).
    - visited_content_ids: Masking set for retrieval; prevents revisiting nodes.
    - iteration_count: Monotonically increments from 0 to max_iterations.
    - search_trajectory: Appends one SearchStep per Planning/Reflection execution.

    Attributes:
        original_query: The original user query (q). Immutable throughout the episode.
            Example: "How does transformer attention work?"
        current_refined_query: The query as refined by Reflection steps
            (q^ret_k = q ⊕ δq_k; MemR3 Eq. 5). On iteration 0, equals
            original_query. May be updated by Reflection to clarify or expand scope.
        accumulated_evidence: Evidence gathered so far (E_k / H^(t)). A list of
            text summaries of retrieved content. Monotonically grows; used to
            synthesize final answers and identify remaining gaps.
        information_gaps: Unresolved information gaps (G_k). A list of gap
            descriptions (e.g., "Details on positional encodings", "Efficiency
            comparisons"). Updated by Reflection based on evidence; guides next
            retrieval iteration.
        active_set: The active retrieval frontier (Z^(t)): a dict with keys
            "cues", "tags", "contents", each a list of IDs/strings. Represents
            the node frontier in the CTC graph at this iteration.
            Example: {"cues": ["transformer"], "tags": ["attention", "position"],
            "contents": ["doc_123", "doc_456"]}.
        visited_content_ids: Content IDs already retrieved; used for masked
            retrieval (MemR3 Eq. 5: M \ M_ret_{k-1}). Prevents infinite loops
            by ensuring each content node is retrieved at most once.
        search_trajectory: List of SearchStep records, one appended per Planning
            or Reflection module execution. Forms a complete audit log of the
            episode for visualization and analysis.
        iteration_count: Current iteration number (0-indexed). Incremented after
            each Planning/Reflection pair. Used to check termination conditions.
        max_iterations: Maximum iterations allowed (n_max; closed-loop episode state management, typically 5).
            Episode terminates when iteration_count >= max_iterations.
        consecutive_stuck_reflects: Count of consecutive Reflection steps with no
            gap reduction (toward n_cap; typically 2). Triggers early exit if
            evidence is not progressing. Resets to 0 when gaps are resolved.
        _scratch_new_retrieval: Transient list of content IDs retrieved in the
            current Planning round, awaiting Reflection's pruning and evaluation.
            Cleared after each Reflection step. Used for internal routing logic.
        final_answer: The final synthesized answer (str), or None until
            answer_synthesis produces it. Set once during episode, never changed.
        _router_decision: Internal routing decision state ("retrieve" or "answer"),
            set by the router_node to guide the next step in the closed loop.

    Example:
        >>> from iterret.state import new_state
        >>> state = new_state("What is photosynthesis?", max_iterations=5)
        >>> state["original_query"]
        'What is photosynthesis?'
        >>> state["iteration_count"]
        0
        >>> state["accumulated_evidence"]
        []
        >>> state["information_gaps"]
        ['initial: no evidence gathered yet']

        >>> # After Planning and Retrieval:
        >>> state["iteration_count"] = 1
        >>> state["accumulated_evidence"] = ["Photosynthesis converts light to chemical energy"]
        >>> state["active_set"]["contents"] = ["article_001"]
        >>> state["_scratch_new_retrieval"] = ["article_001"]

    See also:
        SearchStep: Individual steps in the search_trajectory.
        new_state: Factory function to initialize a fresh IterRetState.
        DEFAULT_MAX_ITERATIONS: Default max_iterations value (5).
        DEFAULT_MAX_STUCK_REFLECTS: Default threshold for consecutive_stuck_reflects (2).
    """

    original_query: str  # q
    current_refined_query: str  # q^ret_k = q (+) delta_q_k  (MemR3 Eq. 5)
    accumulated_evidence: list[str]  # E_k / H^(t)
    information_gaps: list[str]  # G_k
    active_set: dict[str, list[str]]  # Z^(t): {"cues": [...], "tags": [...], "contents": [...]}
    visited_content_ids: list[str]  # masked retrieval, MemR3 Eq. 5: M \ M_ret_{k-1}
    search_trajectory: list[SearchStep]
    iteration_count: int
    max_iterations: int  # n_max
    consecutive_stuck_reflects: int  # toward n_cap
    _scratch_new_retrieval: list[
        str
    ]  # content ids retrieved this round, awaiting Reflect's f_route
    final_answer: str | None
    _router_decision: Literal["retrieve", "answer"]  # routing decision made by router_node


def new_state(original_query: str, *, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> IterRetState:
    """Initialize a fresh episode state for a new query.

    Factory function that creates a new IterRetState with all fields properly
    initialized for the start of a closed-loop memory reconstruction episode.
    This is the entry point for beginning retrieval over a memory graph: the
    user provides a query, and this function sets up the empty state ready for
    the first Planning/Retrieval iteration.

    Initial state invariants:
    - accumulated_evidence and visited_content_ids are empty (nothing retrieved yet).
    - information_gaps contains one placeholder ("initial: no evidence gathered yet").
    - active_set has empty cues, tags, and contents (frontier not yet computed).
    - search_trajectory is empty (no steps recorded yet).
    - iteration_count and consecutive_stuck_reflects are both 0.
    - _router_decision is not yet set (will be set by first router_node call).
    - final_answer is None (will be set by answer_synthesis or remain None).

    Args:
        original_query: The user's question (q). This string is immutable and
            serves as the baseline query throughout the episode. Examples:
            "What is transformer attention?", "How does photosynthesis work?".
        max_iterations: Maximum iterations for the closed-loop reasoning (n_max;
            closed-loop episode state management, Table 1). Default 5. Controls early termination if
            iteration_count reaches this limit before convergence.

    Returns:
        A new IterRetState (TypedDict) with all fields initialized:
        - original_query and current_refined_query set to the input query.
        - accumulated_evidence, visited_content_ids, active_set (cues/tags/contents),
          and search_trajectory all empty.
        - information_gaps initialized with placeholder gap (allows downstream code
          to assume this field is never empty).
        - iteration_count and consecutive_stuck_reflects zeroed.
        - _scratch_new_retrieval empty.
        - final_answer None.

    Example:
        >>> from iterret.state import new_state, DEFAULT_MAX_ITERATIONS
        >>> query = "What are the applications of machine learning?"
        >>> state = new_state(query)
        >>> state["original_query"]
        'What are the applications of machine learning?'
        >>> state["iteration_count"]
        0
        >>> state["accumulated_evidence"]
        []
        >>> state["max_iterations"]
        5

        >>> # Custom max_iterations
        >>> state = new_state(query, max_iterations=10)
        >>> state["max_iterations"]
        10

    Raises:
        None: This function does not raise exceptions. It unconditionally
            returns a valid IterRetState.

    See also:
        IterRetState: The type being initialized.
        DEFAULT_MAX_ITERATIONS: Default value for max_iterations (5).
        DEFAULT_MAX_STUCK_REFLECTS: Default threshold for stuck reflection loops (2).
    """
    return IterRetState(
        original_query=original_query,
        current_refined_query=original_query,
        accumulated_evidence=[],
        information_gaps=["initial: no evidence gathered yet"],
        active_set={"cues": [], "tags": [], "contents": []},
        visited_content_ids=[],
        search_trajectory=[],
        iteration_count=0,
        max_iterations=max_iterations,
        consecutive_stuck_reflects=0,
        _scratch_new_retrieval=[],
        final_answer=None,
    )
