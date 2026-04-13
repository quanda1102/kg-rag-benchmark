"""
pipeline/preprocessing/embedding.py
─────────────────────────────────────
Embed chunks into dense vectors.

Model: Qwen/Qwen3-Embedding-0.6B via sentence-transformers
  - Multilingual, strong on Vietnamese
  - Runs on MPS (Apple Silicon M4)
  - 1024-dim dense vectors

Dataclass:
  ChunkEmbedding
    chunk_id    — source chunk id
    doc_id      — source document id
    heading     — chunk heading (for metadata)
    embedding   — list[float], 1024 dims
    content     — original chunk text

Usage:
    from pipeline.preprocessing.chunker import HeadingChunker
    from pipeline.preprocessing.embedding import Embedder

    chunks    = HeadingChunker().chunk_all(docs)
    embedder  = Embedder()
    embeddings = embedder.embed_all(chunks)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from pipeline.preprocessing.chunker import Chunk


# ── Dataclass ──────────────────────────────────────────────────────────────────

@dataclass
class ChunkEmbedding:
    chunk_id:  str
    doc_id:    str
    heading:   str
    content:   str
    embedding: list[float]

    def __repr__(self) -> str:
        return (
            f"ChunkEmbedding(chunk_id={self.chunk_id!r}, "
            f"heading={self.heading!r}, "
            f"dims={len(self.embedding)})"
        )


# ── Embedder ───────────────────────────────────────────────────────────────────

_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# Qwen3-Embedding requires this prompt prefix for queries (not documents)
# docs are embedded as-is, queries use this prefix at retrieval time
QUERY_PROMPT = "Instruct: Given a question, retrieve relevant passages that answer the question\nQuery: "


class Embedder:
    """
    Embed Chunks into dense vectors using Qwen3-Embedding-0.6B.

    Args:
        model_name: HuggingFace model id. Default: Qwen/Qwen3-Embedding-0.6B
        batch_size: Number of chunks per forward pass. Default: 32
        device:     'mps' for Apple Silicon, 'cpu' as fallback.
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        batch_size: int = 32,
        device: str = "mps",
    ) -> None:
        self.batch_size = batch_size
        print(f"Loading embedding model: {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        print("✓ Model loaded")

    # ── Public API ─────────────────────────────────────────────────────────────

    def embed(self, chunk: Chunk) -> ChunkEmbedding:
        """Embed a single Chunk."""
        vector = self.model.encode(
            chunk.content,
            normalize_embeddings=True,
        ).tolist()

        return ChunkEmbedding(
            chunk_id=chunk.id,
            doc_id=chunk.doc_id,
            heading=chunk.heading,
            content=chunk.content,
            embedding=vector,
        )

    def embed_all(self, chunks: list[Chunk]) -> list[ChunkEmbedding]:
        """
        Embed all chunks in batches.
        Returns list of ChunkEmbedding in same order as input.
        """
        if not chunks:
            return []

        texts = [c.content for c in chunks]

        print(f"Embedding {len(chunks)} chunks (batch_size={self.batch_size})...")
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        results = [
            ChunkEmbedding(
                chunk_id=chunk.id,
                doc_id=chunk.doc_id,
                heading=chunk.heading,
                content=chunk.content,
                embedding=vector.tolist(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        print(f"✓ Embedder: {len(results)} embeddings, dims={len(results[0].embedding)}")
        return results

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a query string for retrieval.
        Uses prompt prefix as required by Qwen3-Embedding.
        """
        vector = self.model.encode(
            query,
            prompt=QUERY_PROMPT,
            normalize_embeddings=True,
        )
        return vector.tolist()


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

    embedder = Embedder()
    embeddings = embedder.embed_all(chunks)

    print(f"\nSample:")
    print(embeddings[0])
    print(f"First 5 dims: {embeddings[0].embedding[:5]}")
