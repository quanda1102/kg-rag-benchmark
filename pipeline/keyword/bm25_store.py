"""
pipeline/keyword/bm25_store.py
───────────────────────────────
BM25 keyword search over chunk content.

Usage:
    from pipeline.keyword.bm25_store import BM25Store

    store = BM25Store()
    store.index(chunks)
    results = store.search("Lê Quang Huy Legacy CRM", k=5)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from pipeline.preprocessing.chunker import Chunk


# ── Dataclass ──────────────────────────────────────────────────────────────────

@dataclass
class BM25Result:
    chunk_id: str
    heading:  str
    content:  str
    score:    float

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return (
            f"BM25Result(heading={self.heading!r}, "
            f"score={self.score:.4f}, preview={preview!r})"
        )


# ── BM25Store ──────────────────────────────────────────────────────────────────

_INDEX_PATH = Path(".bm25/index.pkl")
_META_PATH  = Path(".bm25/meta.json")


class BM25Store:
    """
    BM25Okapi index over chunk content.
    Persists index to disk to avoid re-indexing on every run.
    """

    def __init__(self) -> None:
        self.bm25:  BM25Okapi | None = None
        self.meta:  list[dict]       = []  # [{chunk_id, heading, content}]

    # ── Public API ─────────────────────────────────────────────────────────────

    def index(self, chunks: list[Chunk]) -> None:
        """Build BM25 index from chunks and persist to disk."""
        self.meta = [
            {
                "chunk_id": c.id,
                "heading":  c.heading,
                "content":  c.content,
            }
            for c in chunks
        ]
        tokenized = [self._tokenize(c.content) for c in chunks]
        self.bm25  = BM25Okapi(tokenized)

        self._save()
        print(f"✓ BM25Store: indexed {len(chunks)} chunks")

    def load(self) -> bool:
        """Load persisted index from disk. Returns True if successful."""
        if not _INDEX_PATH.exists() or not _META_PATH.exists():
            return False
        with open(_INDEX_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(_META_PATH, encoding="utf-8") as f:
            self.meta = json.load(f)
        print(f"✓ BM25Store: loaded {len(self.meta)} chunks from disk")
        return True

    def search(self, query: str, k: int = 10) -> list[BM25Result]:
        """Search for top-k chunks by BM25 score."""
        if self.bm25 is None:
            raise RuntimeError("BM25Store not indexed. Call index() or load() first.")

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Get top-k indices sorted by score descending
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        return [
            BM25Result(
                chunk_id=self.meta[i]["chunk_id"],
                heading=self.meta[i]["heading"],
                content=self.meta[i]["content"],
                score=float(scores[i]),
            )
            for i in top_indices
            if scores[i] > 0  # skip zero-score results
        ]

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Simple whitespace + punctuation tokenizer.
        Lowercase, split on whitespace and common punctuation.
        Keeps Vietnamese words intact.
        """
        import re
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def _save(self) -> None:
        _INDEX_PATH.parent.mkdir(exist_ok=True)
        with open(_INDEX_PATH, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(_META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    from pipeline.preprocessing.loader import load_documents
    from pipeline.preprocessing.normalizer import normalize
    from pipeline.preprocessing.chunker import HeadingChunker

    docs = load_documents(Path("data"))
    for doc in docs:
        doc.content = normalize(doc.content)
    chunks = HeadingChunker(min_chars=100, max_chars=2000).chunk_all(docs)

    store = BM25Store()
    store.index(chunks)

    queries = [
        "Lê Quang Huy Legacy CRM",
        "incident Orion Commerce Platform",
        "Kubernetes DevOps",
    ]

    for query in queries:
        print(f"\nQuery: {query!r}")
        results = store.search(query, k=3)
        for r in results:
            print(f"  {r}")