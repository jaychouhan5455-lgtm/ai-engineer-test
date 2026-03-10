# Charter Party Document Parser

A Python application that extracts structured clause data from maritime charter party PDFs using a two-pass LLM pipeline built on the OpenAI API.

Built as a solution to the Marcura AI engineering challenge.

---

## Overview

Charter party documents (e.g., SHELLVOY 5, BPVOY4, ASBATANKVOY) are complex legal agreements with:

- **Strikethrough text** — original printed clauses crossed out by the parties; detected and removed by **pymupdf at extraction time** (Python, not the LLM) so the LLM only ever receives the clean, agreed wording
- **Multi-column layouts** — clause headings in a left column, body text in a right column
- **Multiple re-numbered sections** — each section restarts clause numbering from 1
- **Nested sub-clauses** — `(a)(i)`, `(1)(b)` etc. that must not be split from their parent

This parser handles all of these correctly without hardcoding any document-specific patterns.

---

## Architecture

```
PDF (remote or local)
        │
        ▼
┌───────────────────────────────────────────┐
│  Stage 1 — PDF Extraction  (pymupdf)      │
│  • Open PDF with fitz                     │
│  • Iterate pages from Part II start page  │
│  • Check font flags, drawn lines,         │  ← Strikethrough removed here in
│    and StrikeOut annotations → skip      │     Python (pymupdf), NOT by the LLM
│  • Collect clean text per page            │
│  • Return composite sample_text           │
│    (evenly-spaced windows across doc) +   │
│           full_text (all pages)           │
└───────────────┬───────────────────────────┘
                │
        sample_text
                │
                ▼
┌───────────────────────────────────────────┐
│  Stage 2 — Schema Discovery  (LLM Pass 1) │
│  • Send composite sample to GPT-4o-mini   │
│  • LLM analyses structure dynamically:    │
│      - Section names                      │
│      - Clause numbering format            │
│      - Sub-clause patterns                │
│      - Section transition signals         │
│  • Returns document-specific JSON schema  │
└───────────────┬───────────────────────────┘
                │
        schema + full_text
                │
                ▼
┌───────────────────────────────────────────┐
│  Stage 3 — Clause Extraction (LLM Pass 2) │
│  • Send full text + schema to GPT-4o-mini │
│  • LLM locates every top-level clause     │
│  • Returns JSON array:                    │
│      id: "<Section Name> - <Number>"      │
│      title: clause heading                │
│      text: complete body incl sub-clauses │
└───────────────┬───────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────┐
│  Stage 4 — Validation  (Python)           │
│  • Non-zero clause count                  │
│  • No empty id / title / text fields      │
│  • No duplicate ids                       │
│  • Sequential numbering per section       │
└───────────────┬───────────────────────────┘
                │
                ▼
        output/clauses.json
```

The two-pass LLM design means the parser **generalises to any charter party form** — the schema is learned from the document itself, not from hardcoded rules.

---

## Project Structure

```
.
├── extractor/
│   ├── __init__.py
│   └── pdf_extractor.py       # pymupdf extraction + strikethrough filtering
├── parser/
│   ├── __init__.py
│   └── clause_parser.py       # Schema discovery + clause extraction + validation
├── tests/
│   ├── __init__.py
│   ├── test_extractor.py      # Unit tests for PDF extraction logic
│   └── test_validator.py      # Unit tests for validation logic
├── output/
│   └── clauses.json           # Extracted clauses (submission artefact)
├── main.py                    # CLI entry point
├── requirements.txt
├── .env.example               # Environment variable template
└── .gitignore
```

---

## Setup

### Prerequisites

- Python 3.10 or higher
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd charter-party-parser

# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

---

## Usage

### Default (downloads the SHELLVOY 5 example PDF)

```bash
python3 main.py
```

### Specify a custom PDF URL

```bash
python3 main.py --url https://example.com/my-charter-party.pdf
```

### Use a locally available PDF

```bash
python3 main.py --pdf ./my-charter-party.pdf
```

### Change the output path

```bash
python3 main.py --output ./results/extracted.json
```

### Adjust the Part II start page (0-based)

```bash
python3 main.py --start-page 5
```

### All options

```
python3 main.py --help

usage: charter-party-parser [-h] [--url URL | --pdf PATH] [--output PATH] [--start-page N]

options:
  --url URL        URL of the charter party PDF to download
  --pdf PATH       Path to a locally available charter party PDF
  --output PATH    Path for the output JSON file (default: ./output/clauses.json)
  --start-page N   Zero-based page index where Part II begins (default: 5)
```

---

## Output Format

The output is a JSON array written to `output/clauses.json`. Each clause has exactly three keys:

```json
[
  {
    "id": "SHELLVOY 5 Part II - 1",
    "title": "Condition of Vessel",
    "text": "Owners shall exercise due diligence to ensure that from the time when the obligation to proceed to the loading port(s) attaches..."
  },
  {
    "id": "Shell Additional Clauses - 1",
    "title": "Indemnity Clause",
    "text": "If Charterers by telex, facsimile or other form of written communication that specifically refers to this Clause/Addendum..."
  },
  {
    "id": "Essar Rider Clauses - 1",
    "title": "International Regulations Clause",
    "text": "Vessel to comply with all national and international regulations in force at the beginning of this Charter Party..."
  }
]
```

Section identity is encoded in the `id` field only — there is no top-level `section` key.

---

## Running Tests

```bash
python3 -m pytest tests/ -v
```

All 38 tests run without an API key or real PDF (pymupdf internals are mocked).

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | **Yes** | OpenAI API key for GPT-4o-mini access |
| `PDF_URL` | No | Override the default PDF source URL |
| `PDF_SAVE_PATH` | No | Override the local save path for the downloaded PDF |

Copy `.env.example` to `.env` and fill in your values. **Never commit `.env`.**

---

## Document Support

The parser is designed to generalise beyond SHELLVOY 5. Because the schema is discovered dynamically by the LLM, it should work with:

- SHELLVOY 6
- BPVOY 4
- ASBATANKVOY
- WORLDSCALE
- Any charter party with readable Part II text

The only assumption is that Part II begins at a known page offset (configurable via `--start-page`).

---

## Model

Uses `gpt-4o-mini` via the OpenAI Chat Completions API.

- Fast and cost-efficient with a 128K token context window
- Strong instruction following for structured JSON output
- Reliable legal document reasoning for clause boundary detection

---

## Design Decisions — Pros and Cons

### Two-pass LLM pipeline (schema discovery → clause extraction)

| | |
|---|---|
| **Pro** | The schema is learned from the document itself, so the parser generalises to any charter party form without hardcoded patterns. |
| **Pro** | Pass 1 is cheap (small sample) and Pass 2 benefits from the structural context, which reduces hallucination and mis-splits. |
| **Pro** | Section boundaries, clause numbering styles, and sub-clause patterns all adapt automatically per document. |
| **Con** | Two API round-trips per run increase latency. A single-pass approach would be faster for simple, uniform documents. |
| **Con** | If schema discovery mis-identifies a section name or numbering style, the error propagates into every clause extracted in Pass 2. |

---

### GPT-4o-mini as the LLM

| | |
|---|---|
| **Pro** | Significantly cheaper than GPT-4o or Claude Sonnet for the same task, making bulk processing viable. |
| **Pro** | 128K token context window fits a full charter party in a single extraction call per section. |
| **Pro** | Fast inference keeps end-to-end pipeline time under 10 minutes for a typical document. |
| **Con** | Weaker reasoning than larger models (GPT-4o, Claude Sonnet) — occasional clause boundary mis-detection on complex nested sub-clauses. |
| **Con** | JSON output reliability is lower; a retry loop is required to handle malformed responses. |

---

### pymupdf (fitz) for PDF text extraction

| | |
|---|---|
| **Pro** | Span-level access exposes font flags, bounding boxes, and drawing layers — essential for accurate strikethrough detection. |
| **Pro** | Fast native library; extraction of an 80-page PDF takes under a second. |
| **Pro** | Handles multi-column layouts by preserving per-span coordinates rather than naively concatenating lines. |
| **Con** | PDF rendering varies between generators; some producers encode strikethrough as annotations, some as drawn vector lines, and some via font flags — requiring three separate detection strategies. |
| **Con** | Scanned PDFs (image-only) produce no text spans and would require an OCR pre-processing step. |

---

### Three-strategy strikethrough detection (font flags + drawn lines + StrikeOut annotations)

| | |
|---|---|
| **Pro** | Handles all three encoding conventions found in real charter party PDFs, maximising recall of deleted text. |
| **Pro** | Drawn-line detection uses a geometric inner-zone heuristic (20 %–80 % of character height) that correctly ignores page borders and underlines. |
| **Con** | The inner-zone threshold (20 %–80 %) is a heuristic; unusually positioned strike lines in non-standard PDFs could be missed or produce false negatives. |
| **Con** | Additional detection passes add complexity and surface area for false positives compared to relying on font flags alone. |

---

### Composite sample for schema discovery (evenly-spaced windows across the full document)

| | |
|---|---|
| **Pro** | Guarantees all section headers are visible to the LLM regardless of where they appear and regardless of whether clause numbering restarts. |
| **Pro** | Keeps the schema discovery prompt well within token limits by sampling snippets rather than sending the full document. |
| **Con** | Window boundaries are fixed-size and may split a section header across two windows if it falls near the edge of a window. |
| **Con** | For very long documents, the composite sample may still miss low-frequency structural patterns that appear only once. |

---

## Known Issues

### `PART II - 21` title extracted as "Ice" instead of "Overage Insurance"

In `output/clauses.json`, clause `PART II - 21` is assigned the title **"Ice"** by the LLM. The correct title as shown in the PDF margin heading is **"Overage Insurance"**. This is a title extraction error caused by the LLM picking up the wrong margin heading from the PDF's multi-column layout. The clause body text itself is extracted correctly. This issue has not been fixed in the current version.

---

### Section-by-section extraction with chunking

| | |
|---|---|
| **Pro** | Avoids output-token limits that arise when trying to extract all clauses in a single call. |
| **Pro** | Errors are isolated per section — a failure in one section does not discard clauses from others. |
| **Pro** | Each chunk always starts at a top-level clause boundary, preventing the LLM from seeing a mid-clause fragment without context. |
| **Con** | Chunking increases the total number of API calls linearly with document length, raising cost and latency for very large documents. |
| **Con** | The renumbering step after merging chunks assumes the LLM extracts clauses in document order, which is not guaranteed. |
