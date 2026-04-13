"""
pipeline/preprocessing/normalizer.py
─────────────────────────────────────
Normalize raw document text to clean Markdown before chunking.

Problem:
    Source documents may use non-standard formatting conventions
    that the HeadingChunker cannot recognize as section boundaries.

Strategy:
    Convert known formatting patterns → standard Markdown H1 headings.
    HeadingChunker stays generic — all document-specific logic lives here.

Supported conversions:
    **Bold text alone on a line**  →  # Bold text
    === Title ===                  →  # Title

Usage:
    from pipeline.preprocessing.normalizer import normalize

    doc.content = normalize(doc.content)
"""

from __future__ import annotations

import re

# **Bold text** alone on a line — used in employee profiles
_BOLD_HEADING_PATTERN = re.compile(r"^\*\*(?!.*:\*\*$)(.+?)\*\*$", re.MULTILINE)

def normalize(text: str) -> str:
    """
    Convert non-standard heading formats to Markdown H1.
    Safe to run on any text — patterns are specific enough to avoid
    false positives on inline bold or other markdown.
    """
    text = _BOLD_HEADING_PATTERN.sub(lambda m: f"# {m.group(1)}", text)
    return text