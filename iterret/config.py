from __future__ import annotations

from dataclasses import dataclass

MAX_ACTIVE_CUES = 40  # retrieve_node: cap on cues kept in the active set per round
MAX_ACTIVE_TAGS = 15  # retrieve_node: cap on tags kept in the active set after re-ranking
MAX_NEW_CONTENT_PER_ROUND = 25  # retrieve_node: cap on newly-retrieved content nodes per round


@dataclass(frozen=True)
class TraversalLimits:
    """COLM §1.4.3: active-set pruning with configurable size per round."""

    max_active_cues: int = MAX_ACTIVE_CUES
    max_active_tags: int = MAX_ACTIVE_TAGS
    max_new_content_per_round: int = MAX_NEW_CONTENT_PER_ROUND
