"""PDF text extraction module for charter party documents.

Uses pymupdf (fitz) to open PDF files and iterate over every text span
on each page.  Spans whose font-flags field has bit 3 set (flags & 8)
are identified as strikethrough text and are silently discarded, leaving
only the clean, agreed-upon charter party wording.

The module also provides a helper to download a PDF from a remote URL
so that the rest of the pipeline never has to deal with HTTP concerns.

Typical usage:
    pdf_path = download_pdf(url, "./charter_party.pdf")
    sample_text, full_text = extract_text_from_pdf(pdf_path, start_page=5)
"""

import re

import fitz  # pymupdf
import requests
from pathlib import Path


# ─── constants ────────────────────────────────────────────────────────────────

# Font-flag bit that signals a strikethrough span in this PDF renderer.
STRIKETHROUGH_FLAG: int = 8

# Drawn strike lines in this PDF are often encoded as very thin rectangles.
MAX_STRIKETHROUGH_THICKNESS: float = 2.0

# Require a meaningful portion of the span to be crossed so short incidental
# overlaps do not suppress valid text.
MIN_STRIKETHROUGH_OVERLAP_RATIO: float = 0.6

# Number of Part-II pages included in the schema-discovery sample.
SAMPLE_PAGE_COUNT: int = 4

# HTTP request timeout (seconds) for PDF downloads.
DOWNLOAD_TIMEOUT: int = 60

# Keywords used to auto-detect where Part II begins in an unknown PDF.
PART_II_MARKERS: list[str] = ["PART II", "Part II", "PART 2", "Part 2"]


# ─── public API ───────────────────────────────────────────────────────────────


def find_part_ii_start(pdf_path: str) -> int:
    """Scan a PDF and return the 0-based page index where Part II begins.

    Iterates every page from the beginning and returns the index of the
    first page whose text contains any of the ``PART_II_MARKERS`` strings.
    Falls back to page 0 if no marker is found, so extraction still runs.

    Args:
        pdf_path: Local path to the PDF file.

    Returns:
        Zero-based index of the first Part II page, or 0 if not detected.
    """
    # Regex: "PART II" or "Part II" on its own line, not followed by "(" which
    # would indicate a Part I sub-reference like "Part I(F)" or "PART II(D)".
    _PART_II_HEADER = re.compile(r"^\s*PART\s+II\s*$", re.MULTILINE | re.IGNORECASE)

    doc: fitz.Document = fitz.open(pdf_path)
    for page_index in range(doc.page_count):
        page_text = doc[page_index].get_text()
        if _PART_II_HEADER.search(page_text):
            doc.close()
            return page_index
    doc.close()
    return 0


def download_pdf(url: str, save_path: str) -> str:
    """Download a PDF from *url* and write it to *save_path*.

    Args:
        url: The fully-qualified HTTP/HTTPS URL of the PDF.
        save_path: Local filesystem path where the file will be saved.

    Returns:
        The resolved absolute path of the saved file as a string.

    Raises:
        requests.HTTPError: If the server returns a non-2xx status code.
        OSError: If the file cannot be written to *save_path*.
    """
    response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
    response.raise_for_status()

    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                fh.write(chunk)

    return str(output.resolve())


def _build_strikethrough_lines(page: fitz.Page) -> list[tuple[float, float, float, float]]:
    """Extract all near-horizontal drawn lines from a page.

    Many charter party PDFs encode strikethrough as a vector line drawn
    over the text rather than via font flags. Some generators emit these
    as straight line segments while others emit very thin filled rectangles.
    This function normalises both forms into horizontal line tuples so that
    :func:`_is_drawn_strikethrough` can check each span's bounding box
    against them.

    Args:
        page: A pymupdf ``Page`` object.

    Returns:
        A list of ``(x0, y, x1, y)`` tuples describing each horizontal
        strike candidate found in the page's vector drawing layer.
    """
    h_lines: list[tuple[float, float, float, float]] = []
    for drawing in page.get_drawings():
        for item in drawing.get("items", []):
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) > MAX_STRIKETHROUGH_THICKNESS:
                    continue
                x0 = min(p1.x, p2.x)
                x1 = max(p1.x, p2.x)
                y = (p1.y + p2.y) / 2
            elif item[0] == "re":
                rect = item[1]
                if rect.width < rect.height:
                    continue
                if rect.height > MAX_STRIKETHROUGH_THICKNESS:
                    continue
                x0 = rect.x0
                x1 = rect.x1
                y = (rect.y0 + rect.y1) / 2
            else:
                continue

            if x1 - x0 > 3:            # ignore tiny stubs
                h_lines.append((x0, y, x1, y))
    return h_lines


def _is_drawn_strikethrough(
    span: dict,
    h_lines: list[tuple[float, float, float, float]],
) -> bool:
    """Return True if a drawn horizontal line passes through *span*.

    Checks whether any line from *h_lines* (pre-computed for the page)
    crosses the MIDDLE ZONE of the span's bounding box and overlaps enough
    of it horizontally to be a real strike line.

    To avoid false positives from page borders, underlines and table
    dividers, the line must pass through the inner 60 % of the character
    height (between 20 % and 80 % from the top of the bbox).  Lines that
    fall outside this band — i.e. very close to the top or bottom edges of
    the characters — are decorative rather than strikethrough marks.

    Args:
        span: A pymupdf span dictionary.
        h_lines: Horizontal lines from :func:`_build_strikethrough_lines`.

    Returns:
        True if a drawn strikethrough line covers this span.
    """
    bbox = span.get("bbox")
    if not bbox:
        return False
    x0s, y0s, x1s, y1s = bbox
    height = max(y1s - y0s, 1e-6)

    # Only lines that fall within the inner 60 % of the character height
    # are candidates for strikethrough.  This excludes underlines (near the
    # baseline) and top-border / header-rule lines (near the ascender).
    inner_top = y0s + height * 0.20
    inner_bot = y0s + height * 0.80

    struck_width = 0.0
    for lx0, ly, lx1, _ in h_lines:
        if ly < inner_top or ly > inner_bot:
            continue
        if lx0 >= x1s or lx1 <= x0s:
            continue
        struck_width += min(x1s, lx1) - max(x0s, lx0)

    return (struck_width / max(x1s - x0s, 1e-6)) >= MIN_STRIKETHROUGH_OVERLAP_RATIO


def _is_strikethrough(span: dict, h_lines: list[tuple[float, float, float, float]] | None = None) -> bool:
    """Return True if *span* represents struck-through (deleted) text.

    Detects strikethrough via two mechanisms:

    1. **Font flag** — bit 3 of ``span["flags"]`` is set by the PDF renderer.
    2. **Drawn line** — a horizontal vector line is painted over the span
       (common in charter party documents that use markup layers).

    Args:
        span: A pymupdf span dictionary as returned by
              ``page.get_text("dict")``.
        h_lines: Optional list of horizontal lines from
                 :func:`_build_strikethrough_lines` for drawn-line detection.

    Returns:
        True when the span should be excluded from extraction.
    """
    if bool(span.get("flags", 0) & STRIKETHROUGH_FLAG):
        return True
    if h_lines and _is_drawn_strikethrough(span, h_lines):
        return True
    return False


def _is_word_struck(
    wx0: float,
    wy0: float,
    wx1: float,
    wy1: float,
    h_lines: list[tuple[float, float, float, float]],
    strike_rects: list,
    font_strike_ranges: list[tuple[float, float, float, float]],
) -> bool:
    """Return True if the word bounding box is struck through by any method.

    Checks drawn lines, StrikeOut annotations, and font-flag strike ranges
    (pre-built from span-level data).  Any positive horizontal overlap
    between the word box and a strike mark counts as a hit.
    """
    height = max(wy1 - wy0, 1e-6)
    inner_top = wy0 + height * 0.20
    inner_bot = wy0 + height * 0.80
    wy_mid = (wy0 + wy1) / 2

    # Font-flag strike: span containing this word had the strikethrough flag set
    for fx0, fy_mid, fx1, fheight in font_strike_ranges:
        if abs(fy_mid - wy_mid) > fheight:
            continue
        if fx0 < wx1 and fx1 > wx0:
            return True

    # Drawn-line strike: any overlap with the word is enough
    for lx0, ly, lx1, _ in h_lines:
        if ly < inner_top or ly > inner_bot:
            continue
        if lx0 >= wx1 or lx1 <= wx0:
            continue
        return True

    # Annotation strike
    for rect in strike_rects:
        if wx0 < rect.x1 and wx1 > rect.x0 and wy0 < rect.y1 and wy1 > rect.y0:
            return True

    return False


def _extract_page_text(page: fitz.Page) -> str:
    """Extract clean (non-strikethrough) text from a single PDF page.

    Works at word level: each word's bounding box is checked independently
    against drawn lines, StrikeOut annotations, and font-flag spans.  This
    correctly handles partial-span strikes where only one or two words inside
    a long span are crossed out (e.g. 'noon' replaced by '2359 HRS').

    Args:
        page: A pymupdf ``Page`` object to process.

    Returns:
        A single string containing all non-strikethrough text on the page,
        with lines separated by newline characters.
    """
    h_lines = _build_strikethrough_lines(page)
    strike_rects = _build_annotation_strikes(page)

    # Build font-strike ranges from span-level data (flags & 8)
    font_strike_ranges: list[tuple[float, float, float, float]] = []
    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if bool(span.get("flags", 0) & STRIKETHROUGH_FLAG):
                    b = span["bbox"]
                    height = max(b[3] - b[1], 1e-6)
                    font_strike_ranges.append((b[0], (b[1] + b[3]) / 2, b[2], height))

    # Process words — (x0, y0, x1, y1, word, block_no, line_no, word_no)
    from collections import defaultdict
    line_words: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
    for word_data in page.get_text("words"):  # type: ignore[attr-defined]
        wx0, wy0, wx1, wy1, word_text, block_no, line_no, word_no = word_data
        if _is_word_struck(wx0, wy0, wx1, wy1, h_lines, strike_rects, font_strike_ranges):
            continue
        if word_text.strip():
            line_words[(block_no, line_no)].append((word_no, word_text))

    lines: list[str] = []
    for key in sorted(line_words):
        words_in_line = [w for _, w in sorted(line_words[key])]
        text = " ".join(words_in_line)
        if text.strip():
            lines.append(text)

    return "\n".join(lines)


def _build_annotation_strikes(page: fitz.Page) -> list[fitz.Rect]:
    """Return rectangles corresponding to strikeout annotations on *page*.

    Many authoring tools encode strikethrough using dedicated **StrikeOut**
    annotations instead of plain vector lines or font flags. PyMuPDF exposes
    these via the annotation API with a type name of ``"StrikeOut"``.

    This helper collects the union rectangles for all such annotations so that
    :func:`_is_strikethrough_v2` can compare text spans against them.
    """
    strike_rects: list[fitz.Rect] = []

    annot = page.first_annot
    while annot is not None:
        annot_type = annot.type
        # annot.type is a (code, name) tuple, e.g. (9, "StrikeOut").
        if isinstance(annot_type, tuple) and annot_type[1] == "StrikeOut":
            # Prefer quadpoints (exact text quads) when available, otherwise
            # fall back to the annotation's bounding rect.
            quads = getattr(annot, "vertices", None) or getattr(annot, "quad_points", None)
            if quads:
                # vertices/quad_points is a flat list of coordinates
                pts = [fitz.Point(quads[i], quads[i + 1]) for i in range(0, len(quads), 2)]
                for i in range(0, len(pts), 4):
                    quad = pts[i : i + 4]
                    if len(quad) == 4:
                        xs = [p.x for p in quad]
                        ys = [p.y for p in quad]
                        strike_rects.append(fitz.Rect(min(xs), min(ys), max(xs), max(ys)))
            else:
                strike_rects.append(annot.rect)
        annot = annot.next

    return strike_rects


def _is_annot_strikethrough(
    span: dict,
    strike_rects: list[fitz.Rect],
) -> bool:
    """Return True if *span* lies inside any strikeout-annotation rectangle."""
    bbox = span.get("bbox")
    if not bbox or not strike_rects:
        return False

    span_rect = fitz.Rect(*bbox)
    span_area = span_rect.get_area() or 1e-6

    covered_area = 0.0
    for rect in strike_rects:
        if not span_rect.intersects(rect):
            continue
        intersection = span_rect & rect
        covered_area += intersection.get_area()

    return (covered_area / span_area) >= MIN_STRIKETHROUGH_OVERLAP_RATIO


def _is_strikethrough_v2(
    span: dict,
    h_lines: list[tuple[float, float, float, float]] | None,
    strike_rects: list[fitz.Rect] | None,
) -> bool:
    """Enhanced strikethrough detection used by the main extractor.

    In addition to the original font-flag and drawn-line logic, this also
    treats any span that falls under a StrikeOut annotation as deleted.
    """
    if bool(span.get("flags", 0) & STRIKETHROUGH_FLAG):
        return True
    if h_lines and _is_drawn_strikethrough(span, h_lines):
        return True
    if strike_rects and _is_annot_strikethrough(span, strike_rects):
        return True
    return False


def _consolidate_margin_headings(text: str) -> str:
    """Merge multi-line left-margin headings into a single line.

    In SHELLVOY-style PDFs the clause margin heading often wraps across
    several short lines in the left column, e.g.::

        General
        average/
        New Jason
        Clause
             36. General average shall be payable …

    pymupdf returns each visual line separately.  When the LLM receives
    four short fragments it may only use the first one or two, producing
    an incomplete title ("General average clause" instead of
    "General average/New Jason Clause"), or it may reverse the order
    ("Paramount clause" instead of "Clause paramount").

    This function detects the pattern — a run of short (≤ 35-char),
    non-indented lines immediately before an indented clause-number line
    — and joins them into one line so the LLM always sees the full
    heading intact and in the correct order.
    """
    lines = text.splitlines()
    result: list[str] = []

    for line in lines:
        # A clause-number line: optional leading spaces + digit(s) + period + space
        # Matches both "     36. text" and "36. text" (no leading indent)
        if re.match(r"^\s*\d{1,3}\.\s", line):
            # Walk back through result to collect short heading fragments.
            # A valid heading fragment must:
            #   - be short (≤ 35 chars)
            #   - contain at least one letter (excludes bare page numbers like "477")
            #   - not be indented (≥ 3 leading spaces = body text)
            heading_parts: list[str] = []
            while result:
                prev = result[-1]
                stripped = prev.strip()
                is_short = stripped and len(stripped) <= 35
                has_letter = bool(re.search(r"[a-zA-Z]", stripped))
                not_indented = not re.match(r"^\s{3,}", prev)
                if is_short and has_letter and not_indented:
                    heading_parts.insert(0, stripped)
                    result.pop()
                else:
                    break
            # Inline the heading directly into the clause-number line so it
            # always travels with the clause body into the same chunk and the
            # LLM cannot mistake it for trailing text of the previous clause.
            # Single-fragment headings are kept as-is on their own line.
            if len(heading_parts) > 1:
                merged = " ".join(heading_parts)
                # Embed as [TITLE: ...] prefix on the clause line itself
                line = f"[TITLE: {merged}] {line.lstrip()}"
            elif heading_parts:
                # Single-line heading — keep as separate line (no ambiguity)
                result.append(heading_parts[0])
        result.append(line)

    return "\n".join(result)


def extract_text_from_pdf(
    pdf_path: str,
    start_page: int = 5,
) -> tuple[str, str]:
    """Extract clean text from a PDF, skipping strikethrough spans.

    Opens the PDF at *pdf_path*, processes every page from *start_page*
    (0-indexed) onwards, and filters out any span whose font flags indicate
    strikethrough text.

    Returns two strings:
    - ``sample_text`` — the concatenated text of the first
      ``SAMPLE_PAGE_COUNT`` pages of the extracted range, used by LLM
      Pass 1 (schema discovery) to keep the prompt short.
    - ``full_text`` — the concatenated text of *all* extracted pages,
      used by LLM Pass 2 (clause extraction).

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        start_page: Zero-based page index from which extraction begins.
                    Defaults to 5 (page 6 in human-readable numbering),
                    which is the first page of Part II in SHELLVOY 5.

    Returns:
        A tuple ``(sample_text, full_text)`` where both values are plain
        strings with newline-separated lines.

    Raises:
        FileNotFoundError: If *pdf_path* does not point to an existing file.
        fitz.FileDataError: If the file is not a valid PDF.
        ValueError: If *start_page* is out of range for the document.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc: fitz.Document = fitz.open(str(path))
    total_pages = doc.page_count

    if start_page >= total_pages:
        raise ValueError(
            f"start_page {start_page} is out of range for a "
            f"{total_pages}-page document."
        )

    page_texts: list[str] = []
    for page_index in range(start_page, total_pages):
        page = doc[page_index]
        page_text = _extract_page_text(page)
        if page_text.strip():
            page_texts.append(page_text)

    doc.close()

    full_text = _consolidate_margin_headings("\n\n".join(page_texts))

    # Build a composite sample that spans the whole document so that schema
    # discovery (Pass 1) can see every section header, not just the first few
    # pages.  The composite sample includes the document opening plus a snippet
    # around each point where clause numbering restarts at "1." — the universal
    # signal that a new section has begun.
    sample_text = _build_composite_sample(full_text)

    return sample_text, full_text


# Chars of context to include before and after a detected section boundary.
_BOUNDARY_PRE_CHARS: int = 100
_BOUNDARY_POST_CHARS: int = 2000

# Number of evenly-spaced positions sampled across the document to guarantee
# all section headers are visible even when clause numbering never restarts.
_EVENLY_SPACED_SAMPLES: int = 8

# Chars taken at each evenly-spaced position.
_EVENLY_SPACED_WINDOW: int = 1500

# Regex: a non-numbered header line immediately followed by a restarted "1."
# clause sequence — catches sections whose numbering restarts from 1.
_SECTION_RESTART_RE = re.compile(
    r"([^\n]{5,120})\n+(?:[ \t]*\n)*(?:1\.|1\s+[A-Z])",
    re.MULTILINE,
)


def _build_composite_sample(full_text: str) -> str:
    """Build a compact sample that always includes every section header.

    Uses two complementary strategies so that section headers are captured
    regardless of whether clause numbering restarts:

    1. **Restart detection** — scans for lines immediately followed by a
       clause "1." to catch sections whose numbering restarts from 1.
    2. **Evenly-spaced windows** — takes fixed-size snippets at
       ``_EVENLY_SPACED_SAMPLES`` positions spread evenly across the document
       so that headers in continuation-numbered sections are also visible.

    Overlapping ranges are merged; the final snippets are joined with
    ``[...]`` separators so the LLM knows text is omitted between them.

    Args:
        full_text: The complete joined plain text of all Part II pages.

    Returns:
        A single string containing representative snippets joined by
        ``[...]`` separators.
    """
    if not full_text:
        return ""

    # ── Strategy 1: clause-number restarts ────────────────────────────────────
    boundary_positions: list[int] = []
    for m in _SECTION_RESTART_RE.finditer(full_text):
        header_line = m.group(1).strip()
        if re.match(r"^\d+[\.\s]", header_line):
            continue
        boundary_positions.append(m.start())

    # ── Strategy 2: evenly-spaced windows ─────────────────────────────────────
    doc_len = len(full_text)
    for i in range(_EVENLY_SPACED_SAMPLES):
        pos = int(doc_len * i / _EVENLY_SPACED_SAMPLES)
        # Snap to the start of the nearest line.
        pos = full_text.rfind("\n", 0, pos) + 1 if pos > 0 else 0
        boundary_positions.append(pos)

    # ── Merge all ranges ───────────────────────────────────────────────────────
    # Always start with the document opening (first 2 000 chars).
    ranges: list[tuple[int, int]] = [(0, min(2000, doc_len))]

    for pos in sorted(set(boundary_positions)):
        start = max(0, pos - _BOUNDARY_PRE_CHARS)
        end = min(doc_len, pos + max(_BOUNDARY_POST_CHARS, _EVENLY_SPACED_WINDOW))
        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))

    snippets = [full_text[s:e] for s, e in ranges]
    return "\n\n[...]\n\n".join(snippets)
