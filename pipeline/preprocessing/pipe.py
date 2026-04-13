
if __name__ == "__main__":
    from pathlib import Path
    from .loader import load_documents
    from .normalizer import normalize
    from .chunker import HeadingChunker
    from pipeline.vector.embedding import Embedder
    from pipeline.vector.vector_store import VectorStore

    docs = load_documents(Path("data"))
    for doc in docs:
        doc.content = normalize(doc.content)

    chunks = HeadingChunker(min_chars=100, max_chars=2000).chunk_all(docs)

    embedder = Embedder()
    embeddings = embedder.embed_all(chunks)

    print(f"\nSample:")
    print(embeddings[0])
    print(f"First 5 dims: {embeddings[0].embedding[:5]}")

    store = VectorStore()
    store.store_all(embeddings)

    results = store.search("ai đang làm Legacy CRM Sunset?", k=5)