"""
pipeline/config.py
──────────────────
Load and validate experiment YAML config.
Usage:
    cfg = load_config("experiments/baseline.yaml")
    print(cfg.chunking.strategy)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    strategy: Literal["heading", "sliding_window"] = "heading"
    chunk_size: int = 512       # sliding_window only
    overlap: int = 64           # sliding_window only


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"


class GraphConfig(BaseModel):
    traversal: Literal["bfs", "predefined", "cypher_llm"] = "bfs"
    bfs_depth: int = 3


class RerankingConfig(BaseModel):
    strategy: Literal["rrf", "cross_encoder", "llm"] = "rrf"
    rrf_k: int = 60
    top_k: int = 5


class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password123"


class LLMConfig(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    extraction_max_tokens: int = 2048
    judge_max_tokens: int = 512


class EvalConfig(BaseModel):
    questions_path: str = "eval/questions.json"
    results_dir: str = "results/"


class ExperimentConfig(BaseModel):
    name: str
    description: str = ""
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate experiment config from YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return ExperimentConfig(**raw)