from __future__ import annotations


class EvidenceTracker:
    """COLM §1: Evidence Tracker component — owns E_k (accumulated_evidence)
    and G_k (information_gaps) across loop rounds."""

    def __init__(self, evidence: list[str], gaps: list[str]) -> None:
        self.evidence = evidence
        self.gaps = gaps

    def add_evidence(self, texts: list[str]) -> bool:
        """Appends new evidence texts not already present. Returns True if
        anything was actually added (used by reflect_node's stuck-reflect
        counter)."""
        made_progress = False
        for text in texts:
            if text not in self.evidence:
                self.evidence.append(text)
                made_progress = True
        return made_progress

    def update_gaps(self, resolved: list[str], new_gaps: list[str]) -> None:
        """Remove resolved gaps, then append genuinely new gaps (no duplicates)."""
        resolved_set = set(resolved)
        self.gaps = [g for g in self.gaps if g not in resolved_set]
        for gap in new_gaps:
            if gap not in self.gaps:
                self.gaps.append(gap)
