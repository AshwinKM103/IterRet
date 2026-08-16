from __future__ import annotations

from iterret.memory.evidence_tracker import EvidenceTracker


def test_add_evidence_deduplicates() -> None:
    tracker = EvidenceTracker(["fact1"], [])

    made_progress = tracker.add_evidence(["fact1", "fact2"])

    assert made_progress is True
    assert tracker.evidence == ["fact1", "fact2"]


def test_add_evidence_no_progress_on_duplicates() -> None:
    tracker = EvidenceTracker(["fact1"], [])

    made_progress = tracker.add_evidence(["fact1"])

    assert made_progress is False
    assert tracker.evidence == ["fact1"]


def test_add_evidence_empty_list_no_progress() -> None:
    tracker = EvidenceTracker(["fact1"], [])

    made_progress = tracker.add_evidence([])

    assert made_progress is False
    assert tracker.evidence == ["fact1"]


def test_update_gaps_removes_resolved() -> None:
    tracker = EvidenceTracker([], ["gap1", "gap2", "gap3"])

    tracker.update_gaps(["gap1", "gap3"], [])

    assert tracker.gaps == ["gap2"]


def test_update_gaps_adds_new() -> None:
    tracker = EvidenceTracker([], ["gap1"])

    tracker.update_gaps([], ["gap2", "gap3"])

    assert tracker.gaps == ["gap1", "gap2", "gap3"]


def test_update_gaps_removes_and_adds() -> None:
    tracker = EvidenceTracker([], ["gap1", "gap2"])

    tracker.update_gaps(["gap1"], ["gap3"])

    assert tracker.gaps == ["gap2", "gap3"]


def test_update_gaps_deduplicates_new_gaps() -> None:
    tracker = EvidenceTracker([], ["gap1"])

    tracker.update_gaps([], ["gap2", "gap2", "gap3"])

    assert tracker.gaps == ["gap1", "gap2", "gap3"]


def test_update_gaps_does_not_re_add_existing() -> None:
    tracker = EvidenceTracker([], ["gap1", "gap2"])

    tracker.update_gaps([], ["gap2", "gap3"])

    assert tracker.gaps == ["gap1", "gap2", "gap3"]
