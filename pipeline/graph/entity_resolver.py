"""
pipeline/graph/entity_resolver.py
───────────────────────────────────
Resolve duplicate entities in Neo4j using FuzzyMatchResolver
from neo4j-graphrag package.

Chạy SAU khi store_all() đã write entities vào Neo4j.
FuzzyMatchResolver scan tất cả nodes trong graph,
tìm nodes cùng label có tên gần giống nhau → merge thành 1 node.

Usage:
    from pipeline.graph.entity_resolver import resolve_entities
    from pipeline.config import load_config

    cfg = load_config("experiments/baseline.yaml")
    resolve_entities(cfg.neo4j)
"""

from __future__ import annotations

from neo4j_graphrag.experimental.components.resolver import (
    FuzzyMatchResolver,
)

from pipeline.config import Neo4jConfig
from pipeline.graph.connection import get_driver


async def resolve_entities(
    cfg: Neo4jConfig,
    similarity_threshold: float = 0.85,
) -> None:
    """
    Merge duplicate entity nodes in Neo4j.

    Args:
        cfg:                  Neo4j config
        similarity_threshold: 0.0-1.0, higher = stricter matching
                              0.85 là safe default — bắt typo nhỏ
                              nhưng không merge người khác tên
    """
    driver   = get_driver(cfg)
    resolver = FuzzyMatchResolver(
        driver=driver,
        similarity_threshold=similarity_threshold,
    )

    print(f"Running entity resolution (threshold={similarity_threshold})...")
    await resolver.run()
    print("✓ Entity resolution complete")


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    from pipeline.config import load_config
    from pipeline.graph.connection import close_driver
    from dotenv import load_dotenv

    load_dotenv(override=True)

    cfg = load_config("experiments/baseline.yaml")

    # Check stats trước
    from pipeline.graph.graph_store import GraphStore
    store = GraphStore(cfg.neo4j)
    print(f"Before resolution: {store.stats()}")

    # Run resolver
    asyncio.run(resolve_entities(cfg.neo4j))

    # Check stats sau
    print(f"After resolution:  {store.stats()}")

    close_driver()