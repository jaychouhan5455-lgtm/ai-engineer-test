"""Unit tests for the clause validation logic in parser/clause_parser.py.

Tests cover all four validation invariants enforced by validate_clauses():
  1. Non-zero clause count.
  2. No empty id / title / text fields.
  3. No duplicate ids.
  4. Sequential clause numbering within each section (no gaps, starts at 1).

Each test exercises exactly one failure mode in isolation so that the
error messages remain easy to attribute to a specific rule.
"""

from __future__ import annotations

import pytest

from parser.clause_parser import validate_clauses


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_clause(section: str, number: int, title: str = "Title", text: str = "Body") -> dict:
    """Build a minimal valid clause dictionary.

    Args:
        section: Section name used in the id prefix.
        number: Clause number used in the id suffix.
        title: Optional clause title string.
        text: Optional clause body text string.

    Returns:
        A dict with 'id', 'title', and 'text' keys.
    """
    return {
        "id": f"{section} - {number}",
        "title": title,
        "text": text,
    }


def _valid_clauses() -> list[dict]:
    """Return a minimal valid clause list spanning two sections.

    Returns:
        List of five clause dicts: three in 'Section A', two in 'Section B'.
    """
    return [
        _make_clause("Section A", 1),
        _make_clause("Section A", 2),
        _make_clause("Section A", 3),
        _make_clause("Section B", 1),
        _make_clause("Section B", 2),
    ]


# ─── Happy path ───────────────────────────────────────────────────────────────


class TestValidClausesPasses:
    """validate_clauses() must return None (no exception) for a valid list."""

    def test_single_clause_passes(self) -> None:
        clauses = [_make_clause("SHELLVOY 5 Part II", 1)]
        assert validate_clauses(clauses) is None

    def test_multiple_sections_pass(self) -> None:
        assert validate_clauses(_valid_clauses()) is None

    def test_real_id_format_passes(self) -> None:
        clauses = [
            {"id": "SHELLVOY 5 Part II - 1", "title": "Condition of Vessel", "text": "Body."},
            {"id": "Shell Additional Clauses - 1", "title": "Indemnity Clause", "text": "Body."},
            {"id": "Essar Rider Clauses - 1", "title": "International Regs", "text": "Body."},
        ]
        assert validate_clauses(clauses) is None


# ─── Rule 1: non-zero count ───────────────────────────────────────────────────


class TestEmptyListFails:
    """An empty clause list must raise ValueError."""

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_clauses([])


# ─── Rule 2: no empty fields ──────────────────────────────────────────────────


class TestEmptyFieldsFail:
    """Clauses with blank or missing required fields must raise ValueError."""

    def test_empty_id_raises(self) -> None:
        clauses = [{"id": "", "title": "T", "text": "Body"}]
        with pytest.raises(ValueError, match="id"):
            validate_clauses(clauses)

    def test_whitespace_only_id_raises(self) -> None:
        clauses = [{"id": "   ", "title": "T", "text": "Body"}]
        with pytest.raises(ValueError, match="id"):
            validate_clauses(clauses)

    def test_empty_title_raises(self) -> None:
        clauses = [{"id": "Sec - 1", "title": "", "text": "Body"}]
        with pytest.raises(ValueError, match="title"):
            validate_clauses(clauses)

    def test_empty_text_raises(self) -> None:
        clauses = [{"id": "Sec - 1", "title": "T", "text": ""}]
        with pytest.raises(ValueError, match="text"):
            validate_clauses(clauses)

    def test_missing_title_key_raises(self) -> None:
        clauses = [{"id": "Sec - 1", "text": "Body"}]
        with pytest.raises(ValueError, match="title"):
            validate_clauses(clauses)

    def test_missing_text_key_raises(self) -> None:
        clauses = [{"id": "Sec - 1", "title": "T"}]
        with pytest.raises(ValueError, match="text"):
            validate_clauses(clauses)

    def test_error_message_contains_index(self) -> None:
        """The ValueError message must identify which clause failed."""
        clauses = [
            _make_clause("S", 1),
            {"id": "S - 2", "title": "", "text": "Body"},
        ]
        with pytest.raises(ValueError, match="index 1"):
            validate_clauses(clauses)


# ─── Rule 3: no duplicate ids ─────────────────────────────────────────────────


class TestDuplicateIdsFail:
    """Repeated id values must raise ValueError."""

    def test_exact_duplicate_raises(self) -> None:
        clauses = [
            _make_clause("Sec", 1),
            _make_clause("Sec", 1),  # duplicate
        ]
        with pytest.raises(ValueError, match="duplicate"):
            validate_clauses(clauses)

    def test_error_message_contains_duplicate_id(self) -> None:
        clauses = [
            {"id": "SHELLVOY 5 Part II - 5", "title": "T", "text": "B"},
            {"id": "SHELLVOY 5 Part II - 5", "title": "T2", "text": "B2"},
        ]
        with pytest.raises(ValueError, match="SHELLVOY 5 Part II - 5"):
            validate_clauses(clauses)

    def test_non_duplicate_ids_pass(self) -> None:
        clauses = [
            _make_clause("Sec", 1),
            _make_clause("Sec", 2),
        ]
        validate_clauses(clauses)  # must not raise


# ─── Rule 4: sequential numbering ────────────────────────────────────────────


class TestSequentialNumberingFails:
    """Non-contiguous clause numbers within a section must raise ValueError."""

    def test_gap_in_sequence_raises(self) -> None:
        clauses = [
            _make_clause("Sec", 1),
            _make_clause("Sec", 3),  # gap: 2 is missing
        ]
        with pytest.raises(ValueError, match="non-sequential"):
            validate_clauses(clauses)

    def test_sequence_starting_at_two_raises(self) -> None:
        clauses = [
            _make_clause("Sec", 2),
            _make_clause("Sec", 3),
        ]
        with pytest.raises(ValueError, match="non-sequential"):
            validate_clauses(clauses)

    def test_each_section_evaluated_independently(self) -> None:
        """Numbering resets per section; each must be independently sequential."""
        clauses = [
            _make_clause("A", 1),
            _make_clause("A", 2),
            _make_clause("B", 1),  # B starts fresh from 1 — valid
            _make_clause("B", 2),
        ]
        validate_clauses(clauses)  # must not raise

    def test_out_of_order_ids_still_pass_if_contiguous(self) -> None:
        """Numbers need not arrive in sorted order; sorted check is applied."""
        clauses = [
            _make_clause("Sec", 2),
            _make_clause("Sec", 1),
            _make_clause("Sec", 3),
        ]
        validate_clauses(clauses)  # sorted → [1,2,3] — valid

    def test_malformed_id_format_raises(self) -> None:
        """An id without ' - <number>' suffix must raise ValueError."""
        clauses = [{"id": "BadFormat1", "title": "T", "text": "B"}]
        with pytest.raises(ValueError, match="format"):
            validate_clauses(clauses)

    def test_non_integer_number_raises(self) -> None:
        """A non-numeric clause number must raise ValueError."""
        clauses = [{"id": "Sec - abc", "title": "T", "text": "B"}]
        with pytest.raises(ValueError, match="not an integer"):
            validate_clauses(clauses)

    def test_section_name_with_hyphens_parsed_correctly(self) -> None:
        """Section names that contain ' - ' must still parse correctly."""
        clauses = [
            {"id": "SHELLVOY 5 Part II - 1", "title": "T", "text": "B"},
            {"id": "SHELLVOY 5 Part II - 2", "title": "T", "text": "B"},
        ]
        validate_clauses(clauses)  # must not raise
