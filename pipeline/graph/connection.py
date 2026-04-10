"""
pipeline/graph/connection.py
─────────────────────────────
Neo4j driver wrapper. Single connection reused across pipeline.

Usage:
    from pipeline.graph.connection import get_driver, close_driver

    driver = get_driver(cfg.neo4j)
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as count")
        print(result.single()["count"])
    close_driver()
"""

from __future__ import annotations

from neo4j import GraphDatabase, Driver
from pipeline.config import Neo4jConfig

_driver: Driver | None = None


def get_driver(cfg: Neo4jConfig) -> Driver:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            cfg.uri,
            auth=(cfg.username, cfg.password),
        )
        _driver.verify_connectivity()
        print(f"✓ Connected to Neo4j at {cfg.uri}")
    return _driver


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def clear_graph(cfg: Neo4jConfig) -> None:
    """Wipe all nodes and relationships. Use before re-indexing."""
    driver = get_driver(cfg)
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("✓ Graph cleared")