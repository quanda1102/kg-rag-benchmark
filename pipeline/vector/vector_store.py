"""
pipeline/store/vector_store.py
───────────────────────────────
Store và retrieve chunk embeddings via ChromaDB.

Usage:
    from pipeline.store.vector_store import VectorStore

    store = VectorStore()
    store.store_all(embeddings)
    results = store.search("ai đang làm Legacy CRM Sunset?", k=5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import chromadb

if TYPE_CHECKING:
    from pipeline.vector.embedding import ChunkEmbedding, Embedder


# ── Dataclass ──────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    chunk_id:  str
    doc_id:    str
    heading:   str
    content:   str
    score:     float          # cosine similarity, higher = better

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return (
            f"SearchResult(heading={self.heading!r}, "
            f"score={self.score:.4f}, preview={preview!r})"
        )


# ── VectorStore ────────────────────────────────────────────────────────────────

_COLLECTION_NAME = "chunks"


class VectorStore:
    """
    ChromaDB-backed vector store for ChunkEmbeddings.

    Args:
        persist_path:   Directory to persist ChromaDB data.
                        None = in-memory only.
        collection:     ChromaDB collection name. Default: "chunks"
    """

    def __init__(
        self,
        persist_path: str | None = None,
        collection: str = _COLLECTION_NAME,
    ) -> None:
        if persist_path:
            self.client = chromadb.PersistentClient(path=persist_path)
        else:
            self.client = chromadb.EphemeralClient()

        self.collection = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
        print(f"✓ VectorStore ready — collection={collection!r}, "
              f"persistent={persist_path is not None}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def store(self, embedding: ChunkEmbedding) -> None:
        """Store a single ChunkEmbedding."""
        self.collection.upsert(
            ids=[embedding.chunk_id],
            embeddings=[embedding.embedding],
            documents=[embedding.content],
            metadatas=[{
                "doc_id":  embedding.doc_id,
                "heading": embedding.heading,
            }],
        )

    def store_all(self, embeddings: list[ChunkEmbedding]) -> None:
        """Store all ChunkEmbeddings in one batch."""
        if not embeddings:
            return

        self.collection.upsert(
            ids=[e.chunk_id for e in embeddings],
            embeddings=[e.embedding for e in embeddings],
            documents=[e.content for e in embeddings],
            metadatas=[{
                "doc_id":  e.doc_id,
                "heading": e.heading,
            } for e in embeddings],
        )
        print(f"✓ VectorStore: stored {len(embeddings)} embeddings")

    def search(
        self,
        query_vector: list[float],
        k: int = 5,
        doc_id: str | None = None,
    ) -> list[SearchResult]:
        """
        Search for k nearest chunks.

        Args:
            query_vector:   Embedded query from Embedder.embed_query()
            k:              Number of results to return
            doc_id:         Optional filter by source document
        """
        where = {"doc_id": doc_id} if doc_id else None

        response = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for i in range(len(response["ids"][0])):
            meta = response["metadatas"][0][i]
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 - distance
            score = 1 - response["distances"][0][i]
            results.append(SearchResult(
                chunk_id=response["ids"][0][i],
                doc_id=meta["doc_id"],
                heading=meta["heading"],
                content=response["documents"][0][i],
                score=score,
            ))

        return results

    def count(self) -> int:
        """Return number of stored embeddings."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete all embeddings in collection."""
        self.client.delete_collection(_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("✓ VectorStore reset")


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    from pipeline.preprocessing.loader import load_documents
    from pipeline.preprocessing.normalizer import normalize
    from pipeline.preprocessing.chunker import HeadingChunker
    from pipeline.vector.embedding import Embedder

    # Build pipeline
    docs = load_documents(Path("data"))
    for doc in docs:
        doc.content = normalize(doc.content)

    chunks    = HeadingChunker(min_chars=100, max_chars=2000).chunk_all(docs)
    embedder  = Embedder()
    embeddings = embedder.embed_all(chunks)

    # Store
    store = VectorStore(persist_path=".chromadb")
    store.store_all(embeddings)
    print(f"Total stored: {store.count()}")

    # Search
    query   = "ai đang làm Legacy CRM Sunset?"
    q_vec   = embedder.embed_query(query)
    results = store.search(q_vec, k=5)

    print(f"\nQuery: {query!r}")
    print(f"{len(results)} results:\n")
    for r in results:
        print(f"  {r}")