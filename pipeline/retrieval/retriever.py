"""
pipeline/retrieval/retriever.py
────────────────────────────────
Unified retriever: Vector + BM25 + Graph traversal + RRF rerank.

Flow:
    0. Query analysis → intent, expanded query, cypher hints
    1. Vector search  → top-k chunks by semantic similarity
    2. BM25 search    → top-k chunks by expanded keyword match
    3. Cypher gen     → LLM generates Cypher from natural language query
    4. Graph traverse → Neo4j returns related chunk_ids
    5. Fetch chunks   → ChromaDB fetch by chunk_id
    6. RRF rerank     → merge all 3 sources

Usage:
    from pipeline.retrieval.retriever import Retriever
    from pipeline.config import load_config

    cfg       = load_config("experiments/baseline.yaml")
    retriever = Retriever(cfg)
    results   = retriever.retrieve("ai đang làm Legacy CRM Sunset?")
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from pipeline.config import ExperimentConfig
from pipeline.vector.embedding import Embedder
from pipeline.vector.vector_store import VectorStore, SearchResult
from pipeline.keyword.bm25_store import BM25Store, BM25Result
from pipeline.retrieval.query_analyzer import QueryAnalyzer, QueryAnalysis
from pipeline.graph.connection import get_driver


# ── Dataclass ──────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    chunk_id: str
    heading:  str
    content:  str
    score:    float     # final RRF score
    source:   str       # "vector" | "bm25" | "graph" | combinations

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return (
            f"RetrievalResult(heading={self.heading!r}, "
            f"score={self.score:.4f}, source={self.source!r}, "
            f"preview={preview!r})"
        )


# ── Cypher generation prompt ───────────────────────────────────────────────────

_CYPHER_SYSTEM = """
Bạn là hệ thống sinh Cypher query cho Neo4j knowledge graph.

Schema của graph:
Nodes:
  - PERSON     {id, name, chunk_id, ...properties}
  - PROJECT    {id, name, chunk_id, ...properties}
  - DEPARTMENT {id, name, chunk_id, ...properties}
  - INCIDENT   {id, name, chunk_id, ...properties}
  - SKILL      {id, name, chunk_id, ...properties}

Relations:
  - (PERSON)-[:MANAGES]->(PERSON)
  - (PERSON)-[:WORKS_ON]->(PROJECT)
  - (PERSON)-[:BELONGS_TO]->(DEPARTMENT)
  - (PERSON)-[:REPORTED]->(INCIDENT)
  - (INCIDENT)-[:AFFECTS]->(PROJECT)
  - (PROJECT)-[:OWNED_BY]->(DEPARTMENT)
  - (PERSON)-[:HAS_SKILL]->(SKILL)

Quy tắc:
- Luôn RETURN chunk_id của các node liên quan
- Dùng case-insensitive match: toLower(n.name) CONTAINS toLower($keyword)
- Nếu có property hints, dùng exact match cho properties đó
- Chỉ trả về Cypher query, không có markdown hay giải thích
- Giới hạn kết quả: LIMIT 20

Ví dụ:

Query: "ai đang làm Legacy CRM Sunset?"
Cypher:
MATCH (p:PERSON)-[:WORKS_ON]->(proj:PROJECT)
WHERE toLower(proj.name) CONTAINS toLower("Legacy CRM Sunset")
RETURN p.chunk_id AS chunk_id, p.name AS name

Query: "Lê Quang Huy quản lý ai?"
Cypher:
MATCH (mgr:PERSON)-[:MANAGES]->(p:PERSON)
WHERE toLower(mgr.name) CONTAINS toLower("Lê Quang Huy")
RETURN p.chunk_id AS chunk_id, p.name AS name

Query: "incident nào ảnh hưởng đến Orion Commerce Platform?"
Cypher:
MATCH (i:INCIDENT)-[:AFFECTS]->(proj:PROJECT)
WHERE toLower(proj.name) CONTAINS toLower("Orion Commerce Platform")
RETURN i.chunk_id AS chunk_id, i.name AS name

Query: "incident severity critical status open" [hints: severity="Nghiêm trọng", status="Mở"]
Cypher:
MATCH (i:INCIDENT)
WHERE i.severity = "Nghiêm trọng" AND i.status = "Mở"
RETURN i.chunk_id AS chunk_id, i.name AS name
""".strip()


# ── Retriever ──────────────────────────────────────────────────────────────────

class Retriever:
    """
    Unified GraphRAG retriever — Vector + BM25 + Graph + RRF.

    Args:
        cfg:      ExperimentConfig
        embedder: Embedder instance (reuse if already loaded)
        store:    VectorStore instance (reuse if already loaded)
    """

    def __init__(
        self,
        cfg:        ExperimentConfig,
        embedder:   Embedder | None    = None,
        store:      VectorStore | None = None,
        bm25_store: BM25Store | None   = None,
    ) -> None:
        self.cfg      = cfg
        self.top_k    = cfg.reranking.top_k
        self.rrf_k    = cfg.reranking.rrf_k

        self.embedder = embedder or Embedder(
            model_name=cfg.embedding.model,
            device="mps",
        )
        self.vector_store = store or VectorStore(persist_path=".chromadb")

        self.bm25_store = bm25_store or BM25Store()
        if not bm25_store and not self.bm25_store.load():
            raise RuntimeError(
                "BM25 index not found. Run: uv run -m pipeline.keyword.bm25_store"
            )

        self.driver   = get_driver(cfg.neo4j)
        self.llm      = OpenAI()
        self.analyzer = QueryAnalyzer(model="gpt-5.4-mini")

        self._use_bm25  = True
        self._use_graph = True

        print("✓ Retriever ready")

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """
        Full GraphRAG retrieval pipeline.
        Sources can be toggled via _use_bm25 and _use_graph flags.
        """
        # 0. Query analysis
        analysis = self.analyzer.analyze(query)
        print(f"  [analyzer] intent={analysis.intent}")
        print(f"  [analyzer] expanded_vi={analysis.expanded_query_vi}")
        print(f"  [analyzer] use_graph={analysis.use_graph}")
        print(f"  [analyzer] cypher_hints={analysis.cypher_hints}")

        # 1. Vector search — always on, original query
        query_vector   = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(query_vector, k=self.top_k * 2)

        # 2. BM25 search — expanded query for better keyword match
        if self._use_bm25:
            bm25_query   = f"{analysis.expanded_query_vi} {analysis.expanded_query_en}"
            bm25_results = self.bm25_store.search(bm25_query, k=self.top_k * 2)
        else:
            bm25_results = []

        # 3. Graph traverse — controlled by benchmark variant, not analyzer routing
        if self._use_graph:
            graph_chunk_ids = self._graph_retrieve(
                query,
                cypher_hints=analysis.cypher_hints or {},
            )
            graph_results = self._fetch_chunks_by_ids(graph_chunk_ids)
        else:
            graph_results = []

        # 4. RRF rerank
        final = self._rrf(vector_results, bm25_results, graph_results)
        return final[:self.top_k]

    # ── Private ────────────────────────────────────────────────────────────────

    def _graph_retrieve(
        self,
        query:        str,
        cypher_hints: dict[str, str] = {},
    ) -> list[str]:
        """Generate Cypher from query, execute on Neo4j, return chunk_ids."""
        cypher = self._generate_cypher(query, cypher_hints)
        if not cypher:
            return []

        print(f"\n── Generated Cypher ──\n{cypher}\n─────────────────────")

        try:
            with self.driver.session() as session:
                records = [r.data() for r in session.run(cypher)]
            return [r["chunk_id"] for r in records if r.get("chunk_id")]
        except Exception as ex:
            print(f"  ⚠ Cypher execution failed: {ex}")
            return []

    def _generate_cypher(
        self,
        query:        str,
        cypher_hints: dict[str, str] = {},
    ) -> str | None:
        """Ask LLM to generate Cypher query from natural language."""
        # Append hints to user message if present
        hint_str = ""
        if cypher_hints:
            hint_str = "\n\nProperty hints (dùng các giá trị này chính xác):\n"
            hint_str += "\n".join(f"  {k}: {v!r}" for k, v in cypher_hints.items())

        try:
            response = self.llm.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": _CYPHER_SYSTEM},
                    {"role": "user",   "content": query + hint_str},
                ],
                temperature=0,
                max_completion_tokens=256,
            )
            cypher = response.choices[0].message.content.strip()
            cypher = cypher.replace("```cypher", "").replace("```", "").strip()
            return cypher
        except Exception as ex:
            print(f"  ⚠ Cypher generation failed: {ex}")
            return None

    def _fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[SearchResult]:
        """Fetch chunks from ChromaDB by exact chunk_id."""
        if not chunk_ids:
            return []
        try:
            response = self.vector_store.collection.get(
                ids=chunk_ids,
                include=["documents", "metadatas"],
            )
            return [
                SearchResult(
                    chunk_id=cid,
                    doc_id=response["metadatas"][i].get("doc_id", ""),
                    heading=response["metadatas"][i].get("heading", ""),
                    content=response["documents"][i],
                    score=0.0,
                )
                for i, cid in enumerate(response["ids"])
            ]
        except Exception as ex:
            print(f"  ⚠ ChromaDB fetch by id failed: {ex}")
            return []

    def _rrf(
        self,
        vector_results: list[SearchResult],
        bm25_results:   list[BM25Result],
        graph_results:  list[SearchResult],
    ) -> list[RetrievalResult]:
        """
        Reciprocal Rank Fusion across 3 sources.
        score(d) = Σ 1 / (k + rank(d))
        Chunk appearing in multiple sources gets score boost.
        """
        k      = self.rrf_k
        scores: dict[str, float] = {}
        meta:   dict[str, dict]  = {}

        def _add(
            chunk_id: str,
            rank:     int,
            heading:  str,
            content:  str,
            source:   str,
        ) -> None:
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
            if chunk_id not in meta:
                meta[chunk_id] = {
                    "heading": heading,
                    "content": content,
                    "sources": set(),
                }
            meta[chunk_id]["sources"].add(source)

        for rank, r in enumerate(vector_results):
            _add(r.chunk_id, rank, r.heading, r.content, "vector")

        for rank, r in enumerate(bm25_results):
            _add(r.chunk_id, rank, r.heading, r.content, "bm25")

        for rank, r in enumerate(graph_results):
            _add(r.chunk_id, rank, r.heading, r.content, "graph")

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

        return [
            RetrievalResult(
                chunk_id=cid,
                heading=meta[cid]["heading"],
                content=meta[cid]["content"],
                score=scores[cid],
                source="+".join(sorted(meta[cid]["sources"])),
            )
            for cid in sorted_ids
        ]


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pipeline.config import load_config
    from pipeline.graph.connection import close_driver
    from dotenv import load_dotenv

    load_dotenv(override=True)

    cfg       = load_config("experiments/baseline.yaml")
    retriever = Retriever(cfg)

    queries = [
        "incident nào vừa có severity critical vừa đang ở trạng thái open?",
        "ai đang làm Legacy CRM Sunset?",
        "Lê Quang Huy quản lý ai?",
        "incident nào ảnh hưởng đến Orion Commerce Platform?",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query!r}")
        print('='*60)
        results = retriever.retrieve(query)
        for r in results:
            print(f"  {r}")

    close_driver()
