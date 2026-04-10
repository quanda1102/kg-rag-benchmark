"""
smoke_test.py
─────────────
Quick sanity check before running full benchmark.
Verifies: config loading, Neo4j connection, data loading.

Run: python smoke_test.py
"""

import sys
from pathlib import Path


def check_config():
    print("\n[1/3] Config loading...")
    from pipeline.config import load_config
    cfg = load_config("experiments/baseline.yaml")
    print(f"  ✓ Experiment: {cfg.name}")
    print(f"  ✓ Chunking:   {cfg.chunking.strategy}")
    print(f"  ✓ Embedding:  {cfg.embedding.model}")
    print(f"  ✓ Traversal:  {cfg.graph.traversal}")
    print(f"  ✓ Reranking:  {cfg.reranking.strategy}")
    return cfg


def check_neo4j(cfg):
    print("\n[2/3] Neo4j connection...")
    try:
        from pipeline.graph.connection import get_driver, close_driver
        driver = get_driver(cfg.neo4j)
        with driver.session() as session:
            result = session.run("RETURN 'hello' AS msg")
            msg = result.single()["msg"]
            print(f"  ✓ Neo4j responded: {msg}")
        close_driver()
    except Exception as e:
        print(f"  ✗ Neo4j connection failed: {e}")
        print("  → Make sure Docker is running: docker compose up -d")
        return False
    return True


def check_data():
    print("\n[3/3] Data loading...")
    from pipeline.preprocessing.loader import load_documents, load_ground_truth, load_questions

    plaintext_dir = Path("data/plaintext.md")
    json_path     = Path("data/company.json")
    questions_path = Path("eval/questions.json")

    ok = True

    if plaintext_dir.exists():
        docs = load_documents(plaintext_dir)
        print(f"  ✓ {len(docs)} documents found")
    else:
        print(f"  ✗ data/plaintext/ not found — add your 93 text documents")
        ok = False

    if json_path.exists():
        gt = load_ground_truth(json_path)
        print(f"  ✓ Ground truth loaded")
    else:
        print(f"  ✗ data/json/company.json not found — add your JSON ground truth")
        ok = False

    if questions_path.exists():
        questions = load_questions(questions_path)
        print(f"  ✓ {len(questions)} benchmark questions loaded")
    else:
        print(f"  ✗ eval/questions.json not found — add your benchmark questions")
        ok = False

    return ok


if __name__ == "__main__":
    print("=" * 50)
    print("KG-RAG Benchmark — Smoke Test")
    print("=" * 50)

    try:
        cfg = check_config()
    except Exception as e:
        print(f"  ✗ Config error: {e}")
        sys.exit(1)

    neo4j_ok = check_neo4j(cfg)
    data_ok   = check_data()

    print("\n" + "=" * 50)
    if neo4j_ok and data_ok:
        print("✓ All checks passed — ready to build pipeline")
    else:
        print("✗ Some checks failed — fix above before proceeding")
    print("=" * 50)