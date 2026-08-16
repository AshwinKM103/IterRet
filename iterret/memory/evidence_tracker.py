"""Tracking evidence and information gaps during closed-loop retrieval."""

from __future__ import annotations


class EvidenceTracker:
    """Tracks accumulated evidence and information gaps during closed-loop retrieval.

    COLM §1: Maintains E_k (accumulated_evidence) and G_k (information_gaps)
    across iteration rounds. Evidence is appended incrementally; gaps are
    resolved (removed) and new gaps discovered as retrieval progresses.

    Attributes:
        evidence: List of evidence strings accumulated so far.
        gaps: List of information gaps yet to be resolved.

    Example:
        >>> tracker = EvidenceTracker([], ["need to find Alice's job"])
        >>> tracker.add_evidence(["Alice works at Company X"])  # True (made progress)
        >>> tracker.update_gaps(
        ...     resolved=["need to find Alice's job"],
        ...     new_gaps=["need start date"]
        ... )
    """

    def __init__(self, evidence: list[str], gaps: list[str]) -> None:
        self.evidence = evidence
        self.gaps = gaps

    def add_evidence(self, texts: list[str]) -> bool:
        """Add new evidence texts, skipping duplicates.

        Args:
            texts: List of evidence strings to append.

        Returns:
            True if at least one new evidence was added, False if all were duplicates.

        Example:
            >>> tracker = EvidenceTracker([], [])
            >>> tracker.add_evidence(["Fact 1"])  # True
            >>> tracker.add_evidence(["Fact 1"])  # False (duplicate)
        """
        made_progress = False
        for text in texts:
            if text not in self.evidence:
                self.evidence.append(text)
                made_progress = True
        return made_progress

    def update_gaps(self, resolved: list[str], new_gaps: list[str]) -> None:
        """Update gap list: remove resolved gaps, append new gaps (no duplicates).

        Args:
            resolved: Gaps that were resolved in this round (will be removed).
            new_gaps: New gaps discovered in this round (will be appended if not duplicate).

        Example:
            >>> tracker = EvidenceTracker([], ["gap 1", "gap 2"])
            >>> tracker.update_gaps(resolved=["gap 1"], new_gaps=["gap 3"])
            >>> tracker.gaps  # ["gap 2", "gap 3"]
        """
        resolved_set = set(resolved)
        self.gaps = [g for g in self.gaps if g not in resolved_set]
        for gap in new_gaps:
            if gap not in self.gaps:
                self.gaps.append(gap)
