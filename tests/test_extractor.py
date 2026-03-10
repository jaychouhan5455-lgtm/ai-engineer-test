"""Unit tests for the PDF extraction module (extractor/pdf_extractor.py).

Tests cover:
  - Strikethrough detection via font flags (_is_strikethrough).
  - Page text assembly excluding strikethrough spans (_extract_page_text
    behaviour validated through the public interface).
  - download_pdf() network error propagation.
  - extract_text_from_pdf() error conditions (missing file, bad start_page).
  - sample_text vs full_text length invariant.

These tests use pytest and do NOT require an Anthropic API key or a real
PDF — they mock pymupdf internals where necessary.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from extractor.pdf_extractor import (
    STRIKETHROUGH_FLAG,
    _is_strikethrough,
    download_pdf,
    extract_text_from_pdf,
)


# ─── _is_strikethrough ────────────────────────────────────────────────────────


class TestIsStrikethrough:
    """Tests for the strikethrough-detection helper."""

    def test_returns_true_when_bit_3_set(self) -> None:
        """A span with flags & 8 == 8 is strikethrough."""
        span = {"flags": STRIKETHROUGH_FLAG}
        assert _is_strikethrough(span) is True

    def test_returns_false_when_bit_3_clear(self) -> None:
        """A span whose flags do not include bit 3 is not strikethrough."""
        span = {"flags": 0}
        assert _is_strikethrough(span) is False

    def test_returns_false_for_bold_only(self) -> None:
        """Bold flag (bit 4 = 16) must not trigger strikethrough detection."""
        span = {"flags": 16}
        assert _is_strikethrough(span) is False

    def test_returns_true_when_multiple_flags_include_bit_3(self) -> None:
        """Strikethrough is still detected when combined with other flags."""
        span = {"flags": STRIKETHROUGH_FLAG | 16 | 2}  # strikethrough + bold + italic
        assert _is_strikethrough(span) is True

    def test_missing_flags_key_treated_as_zero(self) -> None:
        """A span with no 'flags' key defaults to 0 (not strikethrough)."""
        span: dict = {}
        assert _is_strikethrough(span) is False

    def test_flags_zero_is_not_strikethrough(self) -> None:
        """Explicit flags=0 is never strikethrough."""
        assert _is_strikethrough({"flags": 0}) is False

    def test_flags_value_7_is_not_strikethrough(self) -> None:
        """flags=7 (bits 0-2 set, bit 3 clear) is not strikethrough."""
        assert _is_strikethrough({"flags": 7}) is False

    def test_flags_value_8_is_strikethrough(self) -> None:
        """Exact value 8 triggers strikethrough."""
        assert _is_strikethrough({"flags": 8}) is True

    def test_flags_value_255_is_strikethrough(self) -> None:
        """All bits set — bit 3 is included, so strikethrough is True."""
        assert _is_strikethrough({"flags": 255}) is True


# ─── download_pdf ─────────────────────────────────────────────────────────────


class TestDownloadPdf:
    """Tests for the PDF download helper."""

    def test_raises_http_error_on_bad_status(self, tmp_path: Path) -> None:
        """HTTP 404 from the server must propagate as requests.HTTPError."""
        import requests

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404")

        with patch("extractor.pdf_extractor.requests.get", return_value=mock_response):
            with pytest.raises(requests.HTTPError):
                download_pdf("http://example.com/missing.pdf", str(tmp_path / "out.pdf"))

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """download_pdf must create nested parent directories if needed."""
        nested = tmp_path / "a" / "b" / "c" / "test.pdf"

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b"%PDF-1.4 fake"]

        with patch("extractor.pdf_extractor.requests.get", return_value=mock_response):
            result = download_pdf("http://example.com/test.pdf", str(nested))

        assert nested.exists()
        assert result == str(nested.resolve())

    def test_returns_resolved_path(self, tmp_path: Path) -> None:
        """Return value must be the resolved absolute path string."""
        save_path = tmp_path / "charter.pdf"

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b"%PDF-1.4 fake"]

        with patch("extractor.pdf_extractor.requests.get", return_value=mock_response):
            result = download_pdf("http://example.com/charter.pdf", str(save_path))

        assert result == str(save_path.resolve())


# ─── extract_text_from_pdf ────────────────────────────────────────────────────


class TestExtractTextFromPdf:
    """Tests for the main PDF text extraction function."""

    def test_raises_file_not_found_for_missing_pdf(self, tmp_path: Path) -> None:
        """A non-existent path must raise FileNotFoundError with the path in message."""
        missing = str(tmp_path / "no_such_file.pdf")
        with pytest.raises(FileNotFoundError, match="no_such_file.pdf"):
            extract_text_from_pdf(missing)

    def test_raises_value_error_when_start_page_out_of_range(
        self, tmp_path: Path
    ) -> None:
        """start_page >= page_count must raise ValueError."""
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"placeholder")

        mock_doc = MagicMock()
        mock_doc.page_count = 3

        with patch("extractor.pdf_extractor.fitz.open", return_value=mock_doc):
            with pytest.raises(ValueError, match="out of range"):
                extract_text_from_pdf(str(fake_pdf), start_page=10)

    def test_sample_text_is_subset_of_full_text(self, tmp_path: Path) -> None:
        """sample_text must be derived from full_text and never longer than it.

        The composite sample covers the whole document (evenly-spaced windows
        plus section-boundary snippets), so the old invariant that pages beyond
        SAMPLE_PAGE_COUNT are absent from the sample no longer holds.  The
        guaranteed invariants are:
          1. len(full) >= len(sample)
          2. The opening page content always appears in the sample.
          3. Every character of sample is a contiguous slice of full_text
             (the sample is built by joining slices of full_text).
        """
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"placeholder")

        # Build 10 fake pages of text.
        page_texts = [f"Page {i} content" for i in range(10)]

        def make_mock_page(text: str) -> MagicMock:
            span = {"type": 0, "flags": 0, "text": text}
            line = {"spans": [span]}
            block = {"type": 0, "lines": [line]}
            page_dict = {"blocks": [block]}
            mock_page = MagicMock()
            mock_page.get_text.return_value = page_dict
            mock_page.get_drawings.return_value = []
            mock_page.first_annot = None
            return mock_page

        mock_doc = MagicMock()
        mock_doc.page_count = 10
        mock_doc.__getitem__ = lambda self, i: make_mock_page(page_texts[i])

        with patch("extractor.pdf_extractor.fitz.open", return_value=mock_doc):
            sample, full = extract_text_from_pdf(str(fake_pdf), start_page=0)

        # full_text must be at least as long as the sample.
        assert len(full) >= len(sample)

        # The first page always appears in the sample (composite sample always
        # includes the document opening).
        assert page_texts[0] in sample

        # Every snippet in the sample must be a substring of full_text.
        for snippet in sample.split("\n\n[...]\n\n"):
            stripped = snippet.strip()
            if stripped:
                assert stripped in full

    def test_strikethrough_spans_excluded_from_output(self, tmp_path: Path) -> None:
        """Spans with flags & 8 must not appear in either returned string."""
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"placeholder")

        keep_span = {"type": 0, "flags": 0, "text": "KEEP THIS"}
        skip_span = {"type": 0, "flags": STRIKETHROUGH_FLAG, "text": "SKIP THIS"}
        line = {"spans": [keep_span, skip_span]}
        block = {"type": 0, "lines": [line]}
        page_dict = {"blocks": [block]}

        mock_page = MagicMock()
        mock_page.get_text.return_value = page_dict
        mock_page.get_drawings.return_value = []
        mock_page.first_annot = None

        mock_doc = MagicMock()
        mock_doc.page_count = 2
        mock_doc.__getitem__ = lambda self, i: mock_page

        with patch("extractor.pdf_extractor.fitz.open", return_value=mock_doc):
            sample, full = extract_text_from_pdf(str(fake_pdf), start_page=0)

        assert "KEEP THIS" in full
        assert "SKIP THIS" not in full
        assert "SKIP THIS" not in sample

    def test_non_text_blocks_are_ignored(self, tmp_path: Path) -> None:
        """Blocks with type != 0 (e.g. image blocks) must not cause errors."""
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"placeholder")

        image_block = {"type": 1}  # image block — no 'lines' key
        text_block = {
            "type": 0,
            "lines": [{"spans": [{"flags": 0, "text": "Hello"}]}],
        }
        page_dict = {"blocks": [image_block, text_block]}

        mock_page = MagicMock()
        mock_page.get_text.return_value = page_dict
        mock_page.get_drawings.return_value = []
        mock_page.first_annot = None

        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.__getitem__ = lambda self, i: mock_page

        with patch("extractor.pdf_extractor.fitz.open", return_value=mock_doc):
            sample, full = extract_text_from_pdf(str(fake_pdf), start_page=0)

        assert "Hello" in full
