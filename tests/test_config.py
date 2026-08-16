from __future__ import annotations

from iterret.config import (
    MAX_ACTIVE_CUES,
    MAX_ACTIVE_TAGS,
    MAX_NEW_CONTENT_PER_ROUND,
    TraversalLimits,
)


def test_traversal_limits_defaults() -> None:
    limits = TraversalLimits()

    assert limits.max_active_cues == MAX_ACTIVE_CUES
    assert limits.max_active_tags == MAX_ACTIVE_TAGS
    assert limits.max_new_content_per_round == MAX_NEW_CONTENT_PER_ROUND


def test_traversal_limits_custom_values() -> None:
    limits = TraversalLimits(max_active_cues=5, max_active_tags=10, max_new_content_per_round=15)

    assert limits.max_active_cues == 5
    assert limits.max_active_tags == 10
    assert limits.max_new_content_per_round == 15


def test_traversal_limits_partial_override() -> None:
    limits = TraversalLimits(max_active_cues=20)

    assert limits.max_active_cues == 20
    assert limits.max_active_tags == MAX_ACTIVE_TAGS
    assert limits.max_new_content_per_round == MAX_NEW_CONTENT_PER_ROUND


def test_traversal_limits_is_frozen() -> None:
    limits = TraversalLimits()

    try:
        limits.max_active_cues = 999  # type: ignore
        assert False, "Should not be able to assign to frozen dataclass"
    except (AttributeError, TypeError):
        pass
