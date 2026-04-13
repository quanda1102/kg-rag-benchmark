"""
pipeline/preprocessing/chunker.py
───────────────────────────────────
Split documents into chunks.

Strategy: heading (H1)
  Split on lines starting with "# " (level-1 heading).
  Each chunk = one section with its heading as context.
  Chunks that are too short get merged with the next one.
  Chunks that are too long get split by paragraph (\n\n).

Dataclass:
  Chunk
    id          — "{doc_id}__chunk_{n}"
    doc_id      — source document id
    heading     — H1 heading text (empty string if before first heading)
    content     — full text of the section including heading line
    char_count  — len(content)

Usage:
    from pipeline.preprocessing.loader import load_documents
    from pipeline.preprocessing.chunker import HeadingChunker

    docs   = load_documents("data/plaintext")
    chunks = HeadingChunker(min_chars=100, max_chars=2000).chunk_all(docs)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    id: str
    doc_id: str
    heading: str          # H1 heading text, empty if preamble
    content: str          # full section text including heading line
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.content)

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return f"Chunk(id={self.id!r}, heading={self.heading!r}, chars={self.char_count}, preview={preview!r})"


# ── Heading chunker ────────────────────────────────────────────────────────────

_H1_PATTERN = re.compile(r"^#\s+(.+)$", re.MULTILINE)


class HeadingChunker:
    """
    Split documents on Markdown H1 headings (lines starting with '# ').

    Args:
        min_chars:  Sections shorter than this get merged into the next section.
                    Prevents tiny orphan chunks from lone headings.
                    Default: 100 chars.
        max_chars:  Sections longer than this get split by paragraph (\\n\\n).
                    Prevents oversized chunks that dilute embeddings.
                    Default: 2000 chars.
    """

    def __init__(self, min_chars: int = 100, max_chars: int = 2000) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars

    # ── Public API ─────────────────────────────────────────────────────────────

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split a single Document into Chunks."""
        raw_sections = self._split_on_h1(doc.content)
        merged       = self._merge_short(raw_sections)
        split        = self._split_large(merged)
        return self._to_chunks(split, doc.id)

    def chunk_all(self, docs: list[Document]) -> list[Chunk]:
        """Split all documents. Returns flat list of Chunks."""
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk(doc))

        print(
            f"✓ HeadingChunker: {len(docs)} docs → {len(all_chunks)} chunks "
            f"(min_chars={self.min_chars}, max_chars={self.max_chars})"
        )
        return all_chunks

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _split_on_h1(text: str) -> list[tuple[str, str]]:
        """
        Split text on H1 headings.
        Returns list of (heading, section_text) tuples.
        Preamble before first heading gets heading = "".
        """
        sections: list[tuple[str, str]] = []
        matches  = list(_H1_PATTERN.finditer(text))

        if not matches:
            return [("", text.strip())]

        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

        for i, match in enumerate(matches):
            heading       = match.group(1).strip()
            section_start = match.start()
            section_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text  = text[section_start:section_end].strip()
            if section_text:
                sections.append((heading, section_text))

        return sections

    def _merge_short(
        self,
        sections: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """
        Merge sections shorter than min_chars into the next section.
        If last section is short, merge it into the previous one.
        """
        if not sections:
            return sections

        merged: list[tuple[str, str]] = []
        pending_heading, pending_text = sections[0]

        for heading, text in sections[1:]:
            if len(pending_text) < self.min_chars:
                pending_text = pending_text + "\n\n" + text
            else:
                merged.append((pending_heading, pending_text))
                pending_heading, pending_text = heading, text

        if pending_text:
            if merged and len(pending_text) < self.min_chars:
                prev_heading, prev_text = merged[-1]
                merged[-1] = (prev_heading, prev_text + "\n\n" + pending_text)
            else:
                merged.append((pending_heading, pending_text))

        return merged

    def _split_large(
        self,
        sections: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """
        Split sections larger than max_chars by paragraph (\\n\\n).
        Sub-chunks inherit the parent heading for embedding context.
        If a single paragraph still exceeds max_chars, keep it as-is
        rather than splitting mid-sentence.
        """
        result: list[tuple[str, str]] = []

        for heading, text in sections:
            if len(text) <= self.max_chars:
                result.append((heading, text))
                continue

            # Split by paragraph
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            current_parts: list[str] = []
            current_len = 0

            for para in paragraphs:
                para_len = len(para)
                if current_parts and current_len + para_len + 2 > self.max_chars:
                    # Flush current buffer
                    result.append((heading, "\n\n".join(current_parts)))
                    current_parts = [para]
                    current_len = para_len
                else:
                    current_parts.append(para)
                    current_len += para_len + 2  # +2 for \n\n

            # Flush remainder
            if current_parts:
                result.append((heading, "\n\n".join(current_parts)))

        return result

    @staticmethod
    def _to_chunks(
        sections: list[tuple[str, str]],
        doc_id: str,
    ) -> list[Chunk]:
        def resolve_heading(heading: str, content: str) -> str:
            if heading:
                return heading
            first_line = content.split("\n")[0].strip()
            first_line = re.sub(r"^===\s*(.+?)\s*===$", r"\1", first_line)
            return first_line[:30] if first_line else doc_id

        return [
            Chunk(
                id=f"{doc_id}__chunk_{i}",
                doc_id=doc_id,
                heading=resolve_heading(heading, text),
                content=text,
            )
            for i, (heading, text) in enumerate(sections)
        ]


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pipeline.preprocessing.normalizer import normalize
    from .loader import load_documents
    from pathlib import Path

    docs = load_documents(Path("data"))
    for doc in docs:
        doc.content = normalize(doc.content)
    chunks = HeadingChunker(min_chars=100, max_chars=2000).chunk_all(docs)

    print(f"\n{len(chunks)} chunks:\n")
    for c in chunks:
        print(c.char_count, c.heading if c.heading else "(preamble)")
