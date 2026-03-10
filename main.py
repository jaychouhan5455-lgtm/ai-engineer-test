"""Entry point for the Charter Party Document Parser.

Provides a CLI interface to download a charter party PDF, extract
structured clause data via the two-pass LLM pipeline, and write the
result to output/clauses.json.

Pipeline overview:
  1. Download (or accept a local) PDF.
  2. Extract clean text with pymupdf, filtering out strikethrough spans.
  3. LLM Pass 1 — discover the document's structural schema.
  4. LLM Pass 2 — extract all clauses using the schema.
  5. Validate the extracted clauses.
  6. Write the result to the output JSON file.

Usage:
    python main.py
    python main.py --url <pdf_url>
    python main.py --pdf <local_pdf_path>
    python main.py --output <output_json_path>
    python main.py --start-page <0-based page index>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI, APIError as OpenAIAPIError
from dotenv import load_dotenv

from extractor.pdf_extractor import download_pdf, extract_text_from_pdf, find_part_ii_start
from parser.clause_parser import discover_schema, extract_clauses, validate_clauses


# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_PDF_URL: str = (
    "https://shippingforum.wordpress.com/wp-content/uploads/2012/09/"
    "voyage-charter-example.pdf"
)
DEFAULT_PDF_SAVE_PATH: str = "./charter_party.pdf"
DEFAULT_OUTPUT_PATH: str = "./output/clauses.json"
DEFAULT_START_PAGE: int = 5  # 0-based; page 6 = first page of Part II


# ─── CLI ──────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns:
        A configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        prog="charter-party-parser",
        description=(
            "Parse a charter party PDF and extract structured clause data "
            "using a two-pass LLM pipeline."
        ),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--url",
        metavar="URL",
        default=None,
        help=(
            "URL of the charter party PDF to download. "
            f"Defaults to the SHELLVOY 5 example if neither --url nor --pdf is given."
        ),
    )
    source.add_argument(
        "--pdf",
        metavar="PATH",
        default=None,
        help="Path to a locally available charter party PDF.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path for the output JSON file. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--start-page",
        metavar="N",
        type=int,
        default=None,
        help=(
            "Zero-based page index from which text extraction begins "
            f"(i.e. the first page of Part II). "
            "If omitted, the page is auto-detected by scanning for 'PART II'."
        ),
    )
    return parser


# ─── Pipeline orchestration ────────────────────────────────────────────────────


def run_pipeline(
    pdf_path: str,
    output_path: str,
    start_page: int,
    client: OpenAI,
) -> None:
    """Run the full clause-extraction pipeline for a given PDF.

    Executes the four pipeline stages in order and writes the final JSON
    array to *output_path*.  Progress is reported to stdout at each stage.

    Args:
        pdf_path: Local filesystem path to the charter party PDF.
        output_path: Destination path for the output ``clauses.json`` file.
        start_page: Zero-based index of the first page to extract (Part II).
        client: An initialised ``anthropic.Anthropic`` API client.

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
        ValueError: If extraction or validation fails.
        OpenAIAPIError: If any LLM API call fails.
    """
    # Stage 1 — PDF text extraction.
    print(f"[1/4] Extracting text from PDF (start page {start_page + 1})…")
    sample_text, full_text = extract_text_from_pdf(pdf_path, start_page=start_page)
    print(
        f"      Extracted {len(full_text):,} characters "
        f"({len(sample_text):,} in sample)."
    )

    # Stage 2 — Schema discovery.
    # Use the composite sample (not full_text) to stay well within TPM limits.
    # The sample takes chunks at 8 positions across the document so all section
    # headers are represented regardless of where they appear.
    print("[2/4] Discovering document structure via LLM (Pass 1)…")
    schema = discover_schema(sample_text, client)
    sections_found = len(schema.get("sections", []))
    print(f"      Schema discovered: {sections_found} section(s) identified.")

    # Stage 3 — Clause extraction.
    print("[3/4] Extracting clauses via LLM (Pass 2)…")
    clauses = extract_clauses(full_text, schema, client)
    print(f"      Extracted {len(clauses)} clause(s).")

    # Stage 4 — Validation.
    print("[4/4] Validating extracted clauses…")
    validate_clauses(clauses)
    print("      Validation passed.")

    # Write output.
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(clauses, fh, indent=2, ensure_ascii=False)
    print(f"\nDone. {len(clauses)} clauses written to {out.resolve()}")


# ─── Entry point ──────────────────────────────────────────────────────────────


def main() -> int:
    """Parse CLI arguments and run the charter party pipeline.

    Returns:
        Exit code: 0 on success, 1 on any error.
    """
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY is not set. "
            "Add your OpenAI API key to .env.",
            file=sys.stderr,
        )
        return 1

    parser = build_arg_parser()
    args = parser.parse_args()

    client = OpenAI(api_key=api_key)

    # Resolve PDF path.
    if args.pdf:
        pdf_path = args.pdf
        if not Path(pdf_path).exists():
            print(f"Error: PDF not found at '{pdf_path}'.", file=sys.stderr)
            return 1
    else:
        url = args.url or os.getenv("PDF_URL") or DEFAULT_PDF_URL
        save_path = os.getenv("PDF_SAVE_PATH") or DEFAULT_PDF_SAVE_PATH
        print(f"Downloading PDF from:\n  {url}")
        try:
            pdf_path = download_pdf(url, save_path)
            print(f"  Saved to {pdf_path}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"Error downloading PDF: {exc}", file=sys.stderr)
            return 1

    # Resolve start page — auto-detect if not explicitly supplied.
    if args.start_page is not None:
        start_page = args.start_page
    else:
        start_page = find_part_ii_start(pdf_path)
        print(f"Auto-detected Part II start: page {start_page + 1} (0-based index {start_page})\n")

    try:
        run_pipeline(
            pdf_path=pdf_path,
            output_path=args.output,
            start_page=start_page,
            client=client,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"\nPipeline error: {exc}", file=sys.stderr)
        return 1
    except OpenAIAPIError as exc:
        print(f"\nOpenAI API error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
