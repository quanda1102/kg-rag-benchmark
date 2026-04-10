"""
pipeline/preprocessing/chunker.py
───────────────────────────────────
Split documents into chunks.

Strategy: heading (H1)
  Split on lines starting with "# " (level-1 heading).
  Each chunk = one section with its heading as context.
  Chunks that are too short get merged with the next one.

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
    chunks = HeadingChunker(min_chars=100).chunk_all(docs)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .loader import Document


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
    """

    def __init__(self, min_chars: int = 100) -> None:
        self.min_chars = min_chars

    # ── Public API ─────────────────────────────────────────────────────────────

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split a single Document into Chunks."""
        raw_sections = self._split_on_h1(doc.content)
        merged       = self._merge_short(raw_sections)
        return self._to_chunks(merged, doc.id)

    def chunk_all(self, docs: list[Document]) -> list[Chunk]:
        """Split all documents. Returns flat list of Chunks."""
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk(doc))

        print(
            f"✓ HeadingChunker: {len(docs)} docs → {len(all_chunks)} chunks "
            f"(min_chars={self.min_chars})"
        )
        return all_chunks

    # ── Private helpers ────────────────────────────────────────────────────────

    def _split_on_h1(self, text: str) -> list[tuple[str, str]]:
        """
        Split text on H1 headings.
        Returns list of (heading, section_text) tuples.
        Preamble before first heading gets heading = "".

        Each section includes its own heading line as the first line,
        so embedding models get full context.
        """
        sections: list[tuple[str, str]] = []
        matches  = list(_H1_PATTERN.finditer(text))

        if not matches:
            # No headings at all — whole doc is one chunk
            return [("", text.strip())]

        # Preamble before first heading
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

        # Each heading section = from this match start to next match start
        for i, match in enumerate(matches):
            heading      = match.group(1).strip()
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
                # Too short — absorb into next section
                pending_text = pending_text + "\n\n" + text
                # Keep pending_heading (first heading wins)
            else:
                merged.append((pending_heading, pending_text))
                pending_heading, pending_text = heading, text

        # Flush last
        if pending_text:
            if merged and len(pending_text) < self.min_chars:
                # Last section too short — merge into previous
                prev_heading, prev_text = merged[-1]
                merged[-1] = (prev_heading, prev_text + "\n\n" + pending_text)
            else:
                merged.append((pending_heading, pending_text))

        return merged

    def _to_chunks(
        self,
        sections: list[tuple[str, str]],
        doc_id: str,
    ) -> list[Chunk]:
        return [
            Chunk(
                id=f"{doc_id}__chunk_{i}",
                doc_id=doc_id,
                heading=heading,
                content=text,
            )
            for i, (heading, text) in enumerate(sections)
        ]


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = """Đây là phần mở đầu trước heading đầu tiên.

# Lê Quang Huy — CEO & Engineering Director

Lê Quang Huy gia nhập TechViet Solutions vào tháng 3 năm 2018.
Hiện tại anh dẫn dắt định hướng tổng thể của công ty.

Về chuyên môn, anh có nền tảng rất sâu về Java, Spring Boot, Kubernetes.

# Trần Đức Long — Engineering Manager

Trần Đức Long gia nhập TechViet Solutions vào tháng 6 năm 2019.
Anh phụ trách điều phối execution của nhóm Engineering.

# X

Quá ngắn.
"""

    from pipeline.preprocessing.loader import Document

    doc    = Document(id="test_doc", content=sample, source_path="test.md")
    chunks = HeadingChunker(min_chars=50).chunk(doc)

    print(f"\n{len(chunks)} chunks:\n")
    for c in chunks:
        print(f"  {c}")
        print(f"  content preview: {c.content[:80]!r}")
        print()