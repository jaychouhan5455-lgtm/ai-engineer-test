"""Clause parser for charter party documents using a two-pass LLM pipeline.

This module orchestrates the full extraction workflow:

  Pass 1 — Schema Discovery
      A sample of the document text (first 4 Part-II pages) is sent to Claude.
      The LLM analyses the document's structural conventions and returns a JSON
      schema that describes how clauses, sub-clauses and sections are delimited
      in *this specific document*.  No patterns are hardcoded here.

  Pass 2 — Clause Extraction
      The full document text together with the discovered schema is sent to
      Claude.  The LLM uses the schema to locate every top-level clause in
      every section and returns a JSON array of clause objects, each with an
      ``id``, ``title`` and ``text`` field.

  Validation
      The extracted list is checked for completeness and internal consistency
      (no empty fields, no duplicate IDs, sequential numbering, non-zero count)
      before the caller writes it to disk.

Typical usage (orchestrated by main.py):
    client = anthropic.Anthropic()
    schema  = discover_schema(sample_text, client)
    clauses = extract_clauses(full_text, schema, client)
    validate_clauses(clauses)
"""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI


# ─── Model ────────────────────────────────────────────────────────────────────

LLM_MODEL: str = "gpt-4o-mini"
MAX_TOKENS_SCHEMA: int = 4096
MAX_TOKENS_EXTRACT: int = 16000
SECTION_CHUNK_SIZE: int = 12_000   # chars; sections larger than this are chunked
MAX_EXTRACT_RETRIES: int = 2       # retries on JSON parse failure


# ─── Prompt constants ─────────────────────────────────────────────────────────

SCHEMA_DISCOVERY_SYSTEM: str = (
    "You are an expert maritime legal document analyst. "
    "Your sole task is to examine the structural conventions of charter party "
    "documents and return precise JSON descriptions of those conventions. "
    "You NEVER invent information; everything you return must be directly "
    "evidenced by the text you are given. "
    "You return ONLY valid JSON — no markdown fences, no prose."
)

SCHEMA_DISCOVERY_PROMPT: str = """Analyse the structural conventions of the following charter party document (strikethrough text already removed).

CRITICAL DISTINCTION — sections vs clause titles:
- A SECTION is a MAJOR DIVISION of the document, introduced by a prominent standalone header on its own line such as "PART II", "SHELL ADDITIONAL CLAUSES", "ESSAR RIDER CLAUSES (1st Dec 2006)".  There are typically only 2-5 sections in the whole document.
- Clause titles like "Condition of Vessel", "Voyage Orders", "Laytime" are the headings of individual numbered clauses INSIDE a section.  Do NOT list clause titles as sections.

Return a single JSON object with these keys:

{{
  "sections": [
    {{
      "name": "<full official section name exactly as it appears in the document header, e.g. 'SHELLVOY 5 Part II', 'SHELL ADDITIONAL CLAUSES', 'ESSAR RIDER CLAUSES (1st Dec 2006)'>",
      "top_level_clause_pattern": "<regex-style description of how top-level clauses start, e.g. '^\\d+\\.' meaning an integer followed by a period at the start of a line>",
      "sub_clause_patterns": "<patterns used for sub-clauses that must NOT be treated as top-level, e.g. '(a)', '(b)', '(i)', '(1)', '(2)', '(1)(a)'>",
      "has_inline_headings": "<true if clause titles appear as a short bold/caps phrase on the same line as '1.' before the body text, false if no separate heading>",
      "first_clause_signal": "<literal text that starts clause 1 of this section, e.g. '1.' or '1. CONDITION'>"
    }}
  ],
  "section_transition_signals": [
    "<standalone header text that marks the start of each new section>"
  ],
  "numbering_restarts_per_section": true,
  "notes": "<any other structural observations, e.g. two-column layout, uppercase headings, continuation pages>"
}}

DOCUMENT TEXT:
{sample_text}
"""

# ─── Schema discovery (LLM Pass 1) ────────────────────────────────────────────


def discover_schema(sample_text: str, client: OpenAI) -> dict[str, Any]:
    """Analyse a document sample and return a structural schema via OpenAI.

    Sends the full Part II text to the LLM and asks it to describe how the
    document is organised: section names, clause numbering styles,
    sub-clause patterns, and section-transition signals.  The returned schema
    is document-specific and requires no hardcoded patterns in this codebase.

    Args:
        sample_text: Full plain text of Part II, with strikethrough spans
                     already removed.
        client: An initialised ``openai.OpenAI`` API client.

    Returns:
        A Python dictionary representing the discovered schema.  Keys include
        ``sections``, ``section_transition_signals``, ``sub_clause_patterns``,
        ``numbering_restarts_per_section``, and ``notes``.

    Raises:
        ValueError: If the LLM response cannot be parsed as valid JSON.
        openai.APIError: If the API call fails.
    """
    prompt = SCHEMA_DISCOVERY_PROMPT.format(sample_text=sample_text)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS_SCHEMA,
        messages=[
            {"role": "system", "content": SCHEMA_DISCOVERY_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip any accidental markdown fences.
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        schema: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        # LLMs sometimes write regex strings with bare \d, \s, etc. which are
        # invalid JSON escape sequences.  Fix them by doubling the backslashes
        # inside string values only, then retry.
        fixed = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', raw)
        try:
            schema = json.loads(fixed)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Schema discovery returned invalid JSON.\n"
                f"Raw LLM response:\n{raw}\n"
                f"Parse error: {exc}"
            ) from exc

    # Log discovered section names so problems are visible immediately.
    for sec in schema.get("sections", []):
        print(f"          section: {sec.get('name', '?')!r}")

    return schema


# ─── Prompt constants (Pass 2) ─────────────────────────────────────────────────

EXTRACTION_SYSTEM: str = (
    "You are an expert maritime legal document parser. "
    "Your sole task is to extract structured clause data from charter party text "
    "and return it as a JSON array. "
    "You follow the structural schema provided precisely and make no assumptions "
    "beyond what the schema and document text tell you. "
    "You return ONLY a valid JSON array — no markdown fences, no prose, no commentary."
)

SECTION_EXTRACTION_PROMPT: str = """Extract ALL clauses from the following single section of a charter party document.

SECTION NAME: {section_name}

STRUCTURAL SCHEMA (describes the whole document for context):
{schema_json}

EXTRACTION RULES:

TOP-LEVEL CLAUSE IDENTIFICATION (most important rule):
A top-level clause starts ONLY when a line begins with a plain arabic integer followed by a period — e.g. "1.", "2.", "13.", "44." — at the very start of a line.  That integer is the clause number.
Sub-parts such as (a), (b), (c), (i), (ii), (iii), (1), (2), (3), (1)(a), (2)(b) etc. that appear INSIDE a clause body are NEVER top-level clauses.  They must be included verbatim inside the parent clause "text" field, not extracted as separate entries.

RULES:
1. Extract ONLY top-level clauses (identified by the rule above).  Everything between two consecutive top-level clause markers is the body of the first clause.
2. Each clause id must use the format:  "{section_name} - <Clause Number>"
   Example: "{section_name} - 1", "{section_name} - 12"
3. Number clauses sequentially starting from 1 in the ORDER they appear — do NOT skip or leave gaps.
4. TITLE — HOW TO IDENTIFY IT (PRIORITY ORDER):
   a) If the clause line starts with a [TITLE: ...] annotation — e.g.
         "[TITLE: General average/ New Jason Clause] 36. General average shall..."
      use the text inside [TITLE: ...] VERBATIM as the title.  Strip the annotation before storing the body in "text".
   b) Otherwise look for a SHORT label (≤ 60 chars, 2–6 words) on its own line IMMEDIATELY BEFORE the clause number.  Use that label verbatim.
   c) If neither (a) nor (b) applies, derive a ≤ 6 word subject summary from the clause content.
   NEVER copy a full sentence from the clause body as the title.
5. TEXT: Must contain the COMPLETE clause body verbatim, including all sub-parts (a), (b), (i), (ii), (1), (2) etc.  Do NOT paraphrase or alter the wording.  If the whole clause is one sentence with no sub-parts, copy it verbatim here (and also use up to 60 chars as the title per rule 4).
6. Preserve the EXACT ORDER clauses appear in the text.  Do NOT reorder or group.
7. Do NOT include section header lines in the output.
8. Strikethrough text is already removed from the input.  If a clause number appears with no visible title or body, skip it entirely.
9. Every clause object must have exactly three keys: "id", "title", "text".  No extra keys.

Return a JSON array of clause objects for this section only.  Do not include clauses from other sections.

SECTION TEXT:
{section_text}
"""


# ─── Clause extraction (LLM Pass 2) ───────────────────────────────────────────


def _locate_sections(full_text: str, sections: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Split *full_text* into per-section slices using schema section names.

    For each section entry in *sections*, searches for its name (and common
    variants) in *full_text* to find the start position.  Returns a list of
    ``(section_name, section_text)`` pairs in document order.

    If a section header cannot be located, its slice runs from the previous
    section end to the next found header (or end-of-text).

    Args:
        full_text: Complete Part II plain text.
        sections: The ``sections`` list from the schema returned by
                  :func:`discover_schema`.

    Returns:
        A list of ``(name, text)`` tuples, one per section, in order.
    """
    positions: list[tuple[int, int]] = []  # (char_pos, section_index)

    for idx, section in enumerate(sections):
        name = section.get("name", "").strip()
        # Build several candidate search strings from the section name.
        candidates = [name]
        # Also try the name with normalised whitespace as a regex.
        normalised = re.sub(r"\s+", r"\\s+", re.escape(name))
        m = re.search(normalised, full_text, re.IGNORECASE)
        if m:
            positions.append((m.start(), idx))
            continue
        # Try progressively shorter prefixes (first 40/20 chars).
        for length in (40, 20):
            prefix = name[:length].strip()
            if prefix:
                pos = full_text.lower().find(prefix.lower())
                if pos >= 0:
                    positions.append((pos, idx))
                    break
        else:
            # Could not locate — record as -1; will be filled in below.
            positions.append((-1, idx))

    # Sort by char position; fill unlocated sections with a fallback.
    positions.sort(key=lambda t: (t[0] if t[0] >= 0 else float("inf")))
    located = [(p, i) for p, i in positions if p >= 0]

    if not located:
        # No section headers found — treat entire text as one section.
        return [(sections[0].get("name", "Section 1") if sections else "Section 1", full_text)]

    # If there is substantial content before the first located section header,
    # attribute it to the first *unlocated* section (typically the opening section
    # whose header the LLM named differently from how it appears in the raw text).
    if located[0][0] > 500:
        located_indices = {i for _, i in located}
        unlocated = [i for i in range(len(sections)) if i not in located_indices]
        if unlocated:
            # Prepend the first unlocated section starting at position 0.
            located.insert(0, (0, unlocated[0]))

    result: list[tuple[str, str]] = []
    for rank, (start_pos, sec_idx) in enumerate(located):
        end_pos = located[rank + 1][0] if rank + 1 < len(located) else len(full_text)
        section_name = sections[sec_idx].get("name", f"Section {sec_idx + 1}")
        result.append((section_name, full_text[start_pos:end_pos]))

    return result


def _heading_adjusted_boundary(text: str, clause_pos: int) -> int:
    """Return a boundary position that includes any margin heading lines
    appearing immediately before the clause-number line at *clause_pos*.

    In SHELLVOY-style PDFs a margin heading (e.g. ``General average/ New
    Jason Clause``) appears on 1–4 short lines BEFORE the "N." clause line.
    If the chunk is split exactly at the "N." position those heading lines
    end up in the previous chunk, so the LLM never sees the heading together
    with the clause body it belongs to.  This function moves the split point
    backwards past any such heading lines so they travel with the clause.
    """
    # Strip the trailing newline so split() doesn't produce a trailing ""
    pre = text[:clause_pos].rstrip("\n")
    lines = pre.split("\n")
    heading_count = 0
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue  # skip blank separator lines without stopping
        # A margin heading line is: short, contains a letter, not indented
        if (len(stripped) <= 35
                and re.search(r"[a-zA-Z]", stripped)
                and not re.match(r"^\s{3,}", line)):
            heading_count += 1
        else:
            break
    if heading_count == 0:
        return clause_pos
    # Move the boundary back past those heading lines (each line + its \n)
    # Count from the END of lines (last N non-empty heading lines)
    back_by = 0
    found = 0
    for line in reversed(lines):
        back_by += len(line) + 1  # +1 for the \n
        stripped = line.strip()
        if stripped:
            found += 1
            if found == heading_count:
                break
    # Also account for the \n that was rstripped from pre
    back_by += len(text[clause_pos - 1:clause_pos]) if clause_pos > 0 else 0
    return max(0, clause_pos - back_by)


def _split_into_chunks(text: str, max_chars: int) -> list[str]:
    """Split *text* into chunks of at most *max_chars* characters.

    Splits ONLY at top-level clause boundaries (lines that begin with an
    integer followed by a period, e.g. "13.").  This ensures every chunk
    starts at the beginning of a top-level clause so the LLM never sees a
    chunk starting mid-clause with a sub-clause marker like "(2)" or "(a)".

    Each boundary is adjusted backwards to include any short margin heading
    lines that appear immediately before the clause number, so the heading
    always travels with its clause body into the same chunk.
    """
    if len(text) <= max_chars:
        return [text]

    # Collect start positions of every top-level clause (digit + period at
    # start of line), adjusted to include any preceding margin heading.
    raw_positions = [m.start() for m in re.finditer(r"(?m)^\d+\.", text)]
    boundaries = [0] + [_heading_adjusted_boundary(text, p) for p in raw_positions]

    # Deduplicate and sort.
    boundaries = sorted(set(boundaries))

    chunks: list[str] = []
    chunk_start = 0

    for i, bp in enumerate(boundaries[1:], start=1):
        # Would adding up to this boundary exceed max_chars?
        if bp - chunk_start > max_chars:
            # Flush everything up to the previous boundary.
            prev_bp = boundaries[i - 1]
            if prev_bp > chunk_start:
                chunks.append(text[chunk_start:prev_bp])
                chunk_start = prev_bp
            # If a single clause itself is too long, just let it through as one chunk.

    # Final chunk.
    if chunk_start < len(text):
        chunks.append(text[chunk_start:])

    return chunks if chunks else [text]


def _call_llm_extract(
    section_name: str,
    section_text: str,
    schema: dict[str, Any],
    client: OpenAI,
) -> list[dict[str, str]]:
    """Single LLM call for one section chunk; retries on JSON parse failure."""
    schema_json = json.dumps(schema, indent=2)
    prompt = SECTION_EXTRACTION_PROMPT.format(
        section_name=section_name,
        schema_json=schema_json,
        section_text=section_text,
    )

    last_exc: Exception | None = None
    for attempt in range(1, MAX_EXTRACT_RETRIES + 1):
        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS_EXTRACT,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            clauses: list[dict[str, str]] = json.loads(raw)
            if not isinstance(clauses, list):
                raise ValueError(
                    f"Expected a JSON array for section '{section_name}', "
                    f"got {type(clauses).__name__}."
                )
            return clauses
        except json.JSONDecodeError:
            # LLMs sometimes emit bare \n, \o etc. inside string values.
            # Fix invalid escape sequences and retry once.
            fixed = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', raw)
            try:
                clauses = json.loads(fixed)
                if isinstance(clauses, list):
                    return clauses
            except json.JSONDecodeError as exc:
                last_exc = ValueError(
                    f"Clause extraction for section '{section_name}' returned invalid JSON "
                    f"(attempt {attempt}/{MAX_EXTRACT_RETRIES}).\n"
                    f"Raw LLM response (first 500 chars):\n{raw[:500]}\n"
                    f"Parse error: {exc}"
                )

    raise last_exc  # type: ignore[misc]


def _extract_one_section(
    section_name: str,
    section_text: str,
    schema: dict[str, Any],
    client: OpenAI,
) -> list[dict[str, str]]:
    """Extract clauses from a single section, chunking if the text is large.

    Sections larger than ``SECTION_CHUNK_SIZE`` characters are split at
    paragraph boundaries.  Each chunk is extracted independently, then all
    results are merged and the clause ids are renumbered sequentially so the
    final list is contiguous (``section_name - 1``, ``section_name - 2``, …).

    Args:
        section_name: Official name of the section (used in clause ids).
        section_text: Plain text for this section only.
        schema: The full document schema for structural context.
        client: An initialised OpenAI API client.

    Returns:
        List of clause dicts with ``id``, ``title``, ``text`` keys.

    Raises:
        ValueError: If any LLM response is not valid JSON or not a list.
    """
    chunks = _split_into_chunks(section_text, SECTION_CHUNK_SIZE)
    all_clauses: list[dict[str, str]] = []

    for i, chunk in enumerate(chunks, start=1):
        if len(chunks) > 1:
            print(f"            chunk {i}/{len(chunks)} ({len(chunk):,} chars)…")
        all_clauses.extend(_call_llm_extract(section_name, chunk, schema, client))

    # Renumber ids sequentially across chunks (LLM may restart at 1 per chunk).
    for seq, clause in enumerate(all_clauses, start=1):
        clause["id"] = f"{section_name} - {seq}"

    return all_clauses


def extract_clauses(
    full_text: str,
    schema: dict[str, Any],
    client: OpenAI,
) -> list[dict[str, str]]:
    """Extract all charter party clauses section-by-section using the schema.

    Splits *full_text* into per-section slices (using section names from the
    schema as boundary markers), then calls the LLM once per section.  This
    avoids hitting output-token limits that arise when trying to extract all
    109 clauses in a single call.

    Args:
        full_text: Complete plain text of Part II with no strikethrough spans.
        schema: The structural schema dictionary returned by
                :func:`discover_schema`.
        client: An initialised ``anthropic.Anthropic`` API client.

    Returns:
        A merged list of clause dictionaries (``id``, ``title``, ``text``)
        covering all sections in document order.

    Raises:
        ValueError: If any section's LLM response is invalid or empty.
        anthropic.APIError: If any API call fails.
    """
    sections = schema.get("sections", [])

    if not sections:
        # No section info — fall back to single-pass extraction of everything.
        sections = [{"name": "Charter Party"}]

    section_slices = _locate_sections(full_text, sections)

    all_clauses: list[dict[str, str]] = []
    for section_name, section_text in section_slices:
        print(f"        Extracting section: {section_name!r} "
              f"({len(section_text):,} chars)…")
        clauses = _extract_one_section(section_name, section_text, schema, client)
        print(f"          → {len(clauses)} clause(s) found.")
        all_clauses.extend(clauses)

    return all_clauses


# ─── Validation ────────────────────────────────────────────────────────────────


def validate_clauses(clauses: list[dict[str, str]]) -> None:
    """Validate extracted clauses for completeness and internal consistency.

    Runs four checks in order, raising a descriptive :class:`ValueError` on
    the first failure found:

    1. **Non-zero count** — the list must contain at least one clause.
    2. **No empty fields** — every clause must have a non-blank ``id``,
       ``title``, and ``text``.
    3. **No duplicate ids** — every ``id`` value must be unique across the list.
    4. **Sequential numbering** — within each section (identified by the prefix
       before the last hyphen in the id) clause numbers must form a contiguous
       sequence starting at 1 with no gaps.

    Args:
        clauses: The list of clause dictionaries returned by
                 :func:`extract_clauses`.

    Returns:
        None.  The function is a pure guard — it either passes silently or
        raises.

    Raises:
        ValueError: With a descriptive message identifying the specific
                    validation failure and which clause(s) are involved.
    """
    # ── 1. Non-zero count ──────────────────────────────────────────────────────
    if not clauses:
        raise ValueError(
            "Validation failed: clause list is empty. "
            "The extractor returned no clauses."
        )

    # ── 2. No empty fields ────────────────────────────────────────────────────
    # Auto-fix: if text is empty but title is not, the LLM put a one-liner
    # clause entirely in the title field — copy it to text so both are filled.
    for clause in clauses:
        if clause.get("title", "").strip() and not clause.get("text", "").strip():
            clause["text"] = clause["title"]

    required_keys = ("id", "title", "text")
    for index, clause in enumerate(clauses):
        for key in required_keys:
            value = clause.get(key, "")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Validation failed: clause at index {index} has an empty "
                    f"or missing '{key}' field. Clause: {clause!r}"
                )

    # ── 3. No duplicate ids ───────────────────────────────────────────────────
    seen_ids: dict[str, int] = {}
    for index, clause in enumerate(clauses):
        clause_id = clause["id"]
        if clause_id in seen_ids:
            raise ValueError(
                f"Validation failed: duplicate id '{clause_id}' found at "
                f"indices {seen_ids[clause_id]} and {index}."
            )
        seen_ids[clause_id] = index

    # ── 4. Sequential numbering within each section ───────────────────────────
    section_numbers: dict[str, list[int]] = {}
    for clause in clauses:
        clause_id = clause["id"]
        # id format: "<Section Name> - <Number>"
        parts = clause_id.rsplit(" - ", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(
                f"Validation failed: id '{clause_id}' does not match the "
                f"expected format '<Section Name> - <Number>'."
            )
        section_name, number_str = parts[0].strip(), parts[1].strip()
        try:
            number = int(number_str)
        except ValueError:
            raise ValueError(
                f"Validation failed: clause number '{number_str}' in id "
                f"'{clause_id}' is not an integer."
            )
        section_numbers.setdefault(section_name, []).append(number)

    for section, numbers in section_numbers.items():
        sorted_numbers = sorted(numbers)
        expected = list(range(1, len(sorted_numbers) + 1))
        if sorted_numbers != expected:
            raise ValueError(
                f"Validation failed: non-sequential clause numbers in section "
                f"'{section}'. Found {sorted_numbers}, expected {expected}."
            )
