"""
pipeline/graph/graph_store.py
──────────────────────────────
Store entities and relations in Neo4j.

Usage:
    from pipeline.graph.graph_store import GraphStore
    from pipeline.config import load_config

    cfg   = load_config("experiments/baseline.yaml")
    store = GraphStore(cfg.neo4j)
    store.store_all(entities, relations)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pipeline.config import Neo4jConfig
from pipeline.graph.connection import get_driver

if TYPE_CHECKING:
    from pipeline.graph.entity_extractor import Entity, Relation


class GraphStore:
    """
    Neo4j-backed graph store for Entities and Relations.
    Reuses singleton driver from connection.py.
    """

    def __init__(self, cfg: Neo4jConfig) -> None:
        self.driver = get_driver(cfg)

    # ── Public API ─────────────────────────────────────────────────────────────

    def store_all(
        self,
        entities:  list[Entity],
        relations: list[Relation],
    ) -> None:
        """Store all entities and relations in batch. Uses MERGE to avoid duplicates."""
        with self.driver.session() as session:
            session.execute_write(self._merge_entities_batch, entities)
            session.execute_write(self._merge_relations_batch, relations)

        print(
            f"✓ GraphStore: stored {len(entities)} entities, "
            f"{len(relations)} relations"
        )

    def query(self, cypher: str, **params) -> list[dict[str, Any]]:
        """Run arbitrary Cypher query. Returns list of record dicts."""
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            return [record.data() for record in result]

    def stats(self) -> dict[str, int]:
        """Return node and relation counts."""
        nodes = self.query("MATCH (n) RETURN count(n) as count")[0]["count"]
        rels  = self.query("MATCH ()-[r]->() RETURN count(r) as count")[0]["count"]
        return {"nodes": nodes, "relations": rels}

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _merge_entities_batch(tx, entities: list[Entity]) -> None:
        """
        Batch MERGE entities using UNWIND.
        Dynamic label set via apoc.create.addLabels — requires APOC plugin.
        """
        tx.run("""
            UNWIND $rows AS row
            MERGE (n {id: row.id})
            SET n.name     = row.name,
                n.chunk_id = row.chunk_id
            SET n += row.properties
            WITH n, row
            CALL apoc.create.addLabels(n, [row.type, '__Entity__']) YIELD node
            RETURN node
        """, rows=[
            {
                "id":         e.id,
                "name":       e.name,
                "type":       e.type,
                "chunk_id":   e.chunk_id,
                "properties": e.properties,
            }
            for e in entities
        ])

    @staticmethod
    def _merge_relations_batch(tx, relations: list[Relation]) -> None:
        """
        Batch MERGE relations using UNWIND.
        Dynamic relation type via apoc.merge.relationship — requires APOC plugin.
        """
        tx.run("""
            UNWIND $rows AS row
            MATCH (a {id: row.source})
            MATCH (b {id: row.target})
            CALL apoc.merge.relationship(a, row.relation, {}, row.properties, b)
            YIELD rel
            RETURN rel
        """, rows=[
            {
                "source":     r.source,
                "target":     r.target,
                "relation":   r.relation,
                "properties": r.properties,
            }
            for r in relations
        ])


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    from pipeline.preprocessing.loader import load_documents
    from pipeline.preprocessing.normalizer import normalize
    from pipeline.preprocessing.chunker import HeadingChunker
    from pipeline.graph.entity_extractor import EntityExtractor
    from pipeline.graph.connection import clear_graph, close_driver
    from pipeline.config import load_config
    from dotenv import load_dotenv
    import random
    import asyncio

    load_dotenv(override=True)

    cfg = load_config("experiments/baseline.yaml")

    docs = load_documents(Path("data"))
    for doc in docs:
        doc.content = normalize(doc.content)
    chunks = HeadingChunker(min_chars=100, max_chars=2000).chunk_all(docs)

    # chunks_sample = random.sample(chunks, 3)
    extractor     = EntityExtractor()
    # entities, relations = extractor.extract_all(chunks_sample)
    entities, relations = asyncio.run(extractor.extract_all(chunks))

    store = GraphStore(cfg.neo4j)
    clear_graph(cfg.neo4j)
    store.store_all(entities, relations)

    print(f"\nStats: {store.stats()}")

    print("\n── MANAGES ──")
    for r in store.query(
        "MATCH (a:PERSON)-[:MANAGES]->(b:PERSON) RETURN a.name, b.name"
    ):
        print(f"  {r['a.name']} → {r['b.name']}")

    print("\n── WORKS_ON ──")
    for r in store.query(
        "MATCH (a:PERSON)-[:WORKS_ON]->(p:PROJECT) RETURN a.name, p.name"
    ):
        print(f"  {r['a.name']} → {r['p.name']}")

    close_driver()