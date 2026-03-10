"""Parser package for charter party clause extraction.

Implements a two-pass LLM pipeline using the Anthropic Claude API:
  Pass 1 — Schema discovery: analyse document structure dynamically.
  Pass 2 — Clause extraction: extract all clauses using the discovered schema.

Also provides a validation layer that verifies completeness and integrity
of the extracted clause data before it is written to disk.
"""
