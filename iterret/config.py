"""Traversal configuration and limits for CTC graph retrieval.

Defines constraints on active-set sizes at each step of the closed-loop
memory reconstruction (COLM §1.4.3, Table 1) to keep retrieval deterministic,
reproducible, and computationally tractable.

The IterRet closed loop involves multiple nested loops:
1. **Outer loop**: Planning → Retrieval → Reflection → Routing (iterates up to n_max times).
2. **Inner loop (CTC traversal)**: Expand cues to tags, re-rank tags, load content
   (bounded by MAX_INNER_CTC_ITERATIONS to prevent explosion).

TraversalLimits provides four independent bounds:
- max_active_cues: Cap on cues in the active frontier per round.
- max_active_tags: Cap on tags after re-ranking per round.
- max_new_content_per_round: Cap on newly retrieved content per round.
- max_inner_ctc_iterations: Cap on CTC traversal's internal expand/prune/load cycles.

Module constants:
    MAX_ACTIVE_CUES: Default 40, cap on cues in active set per round.
    MAX_ACTIVE_TAGS: Default 15, cap on tags after re-ranking.
    MAX_NEW_CONTENT_PER_ROUND: Default 25, cap on newly retrieved content nodes.
    MAX_INNER_CTC_ITERATIONS: Default 3, cap on CTC traversal inner hops.

Classes:
    TraversalLimits: Frozen dataclass encapsulating all four limits.

Example:
    >>> from iterret.config import TraversalLimits
    >>> limits = TraversalLimits(
    ...     max_active_cues=40,
    ...     max_active_tags=15,
    ...     max_new_content_per_round=25,
    ...     max_inner_ctc_iterations=3
    ... )
    >>> limits.max_active_cues
    40

See also COLM §1.4.3 for formal definitions and theoretical justification.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Cap on cues kept in the active frontier after Planning phase.
#: During retrieval planning, the system identifies relevant cues (e.g., entities,
#: key terms) from the query. Only the top MAX_ACTIVE_CUES are retained for
#: expansion into tags. Prevents excessive branching into tag expansion.
#: Used in retrieve_node and TraversalLimits.
MAX_ACTIVE_CUES = 40

#: Cap on tags kept in the active frontier after semantic re-ranking.
#: After expanding cues to tags via CTC and re-ranking by relevance, only the
#: top MAX_ACTIVE_TAGS are retained for content retrieval. Controls the branching
#: factor for loading content nodes. Typical range: 10-20.
#: Used in retrieve_node and TraversalLimits.
MAX_ACTIVE_TAGS = 15

#: Cap on newly-retrieved content nodes per closed-loop iteration.
#: After loading content for selected tags, only this many new nodes are added
#: to accumulated evidence per iteration. Prevents memory explosion and ensures
#: reasonable evidence bases across many iterations. Typical range: 15-30.
#: Used in retrieve_node and TraversalLimits.
MAX_NEW_CONTENT_PER_ROUND = 25

#: Cap on the CTC traversal's internal expand-prune-load iteration loop.
#: The CTC graph traversal involves expand (cues→tags), semantic re-ranking (prune),
#: and content loading (load) as internal hops. This limit bounds how many times the
#: loop cycles before returning control to the outer closed-loop controller
#: (COLM §1.4.3 step 5). Independent of the outer retrieve/reflect budget
#: (n_max / n_cap). Typical range: 2-5. Prevents infinite loops in CTC traversal.
#: Used in retrieve_node and TraversalLimits.
MAX_INNER_CTC_ITERATIONS = 3


@dataclass(frozen=True)
class TraversalLimits:
    """Active-set pruning configuration (COLM §1.4.3, Table 1).

    Frozen dataclass encapsulating all pruning limits for the IterRet closed
    loop. These limits ensure that retrieval remains deterministic, reproducible,
    and computationally tractable by bounding the size of intermediate sets at
    each decision point.

    The limits are applied independently and sequentially:
    1. **max_active_cues**: After Planning phase identifies cues, cap at this size.
    2. **max_active_tags**: After expanding cues to tags and re-ranking, cap at this size.
    3. **max_new_content_per_round**: After loading content for tags, cap at this size.
    4. **max_inner_ctc_iterations**: Within CTC traversal (expand→prune→load cycle),
       cap the number of iterations to prevent infinite loops or excessive computation.

    Frozen dataclass means instances are immutable; this ensures configuration
    consistency across an episode and makes state caching safe.

    Attributes:
        max_active_cues: Cap on cues in the active set per Planning round
            (default 40 from MAX_ACTIVE_CUES). After planning identifies cues
            (e.g., entities or query terms), only the top N are retained. Limits
            the fan-out into tag expansion. Must be > 0.
        max_active_tags: Cap on tags after re-ranking per round (default 15 from
            MAX_ACTIVE_TAGS). After cue-to-tag expansion and semantic re-ranking,
            only the top N tags are retained. Controls the branching factor for
            content retrieval. Must be > 0.
        max_new_content_per_round: Cap on newly-retrieved content nodes per round
            (default 25 from MAX_NEW_CONTENT_PER_ROUND). After loading content for
            selected tags, only this many new content nodes are added to the
            evidence base per iteration. Prevents memory explosion. Must be > 0.
        max_inner_ctc_iterations: Cap on CTC traversal's internal hop loop (default 3
            from MAX_INNER_CTC_ITERATIONS). The CTC graph traversal involves
            expand (cues→tags), prune (semantic re-ranking), and load (content
            retrieval) steps. This limit caps how many times the loop cycles
            before returning control to the outer closed-loop controller
            (COLM §1.4.3 step 5). Must be > 0.

    Example:
        >>> from iterret.config import TraversalLimits, MAX_ACTIVE_CUES
        >>> # Default configuration
        >>> limits = TraversalLimits()
        >>> limits.max_active_cues
        40
        >>> limits.max_active_tags
        15
        >>> limits.max_new_content_per_round
        25
        >>> limits.max_inner_ctc_iterations
        3

        >>> # Custom configuration for memory-constrained environments
        >>> tight_limits = TraversalLimits(
        ...     max_active_cues=20,
        ...     max_active_tags=8,
        ...     max_new_content_per_round=10,
        ...     max_inner_ctc_iterations=2
        ... )
        >>> tight_limits.max_active_cues
        20

    Raises:
        None: This is a dataclass with no validation in __post_init__.
            Callers should ensure all fields are positive integers.

    See also:
        MAX_ACTIVE_CUES, MAX_ACTIVE_TAGS, MAX_NEW_CONTENT_PER_ROUND,
        MAX_INNER_CTC_ITERATIONS: Module-level constants used as defaults.

    References:
        COLM §1.4.3, Table 1: Formal definitions and theoretical justification
            for these pruning bounds in the context of MemR3 and IterRet.
    """

    max_active_cues: int = MAX_ACTIVE_CUES
    max_active_tags: int = MAX_ACTIVE_TAGS
    max_new_content_per_round: int = MAX_NEW_CONTENT_PER_ROUND
    max_inner_ctc_iterations: int = MAX_INNER_CTC_ITERATIONS
