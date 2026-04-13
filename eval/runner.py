"""
eval/runner.py
───────────────
Run evaluation across 3 retrieval variants.

Variants:
    vector              — RAG only
    vector+bm25         — RAG + BM25
    vector+bm25+graph   — RAG + BM25 + KG (full pipeline)

Usage:
    uv run -m eval.runner
    uv run -m eval.runner --config experiments/baseline.yaml
    uv run -m eval.runner --limit 5
    uv run -m eval.runner --open   # auto-open HTML report after run
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from eval.judge import Judge
from eval.judge import estimate_cost_usd
from eval.metrics import QuestionResult, VariantResult, compute_metrics
from pipeline.config import load_config, ExperimentConfig
from pipeline.generation.generator import Generator
from pipeline.graph.connection import close_driver
from pipeline.keyword.bm25_store import BM25Store
from pipeline.vector.embedding import Embedder
from pipeline.vector.vector_store import VectorStore
from pipeline.retrieval.retriever import Retriever


# ── Variant configs ────────────────────────────────────────────────────────────

VARIANTS = [
    "vector",
    "vector+bm25",
    "vector+bm25+graph",
]


# ── Ground truth chunks ────────────────────────────────────────────────────────

def build_gt_chunks(
    questions:    list[dict],
    company_path: str = "data/company.json",
) -> dict[str, list[str]]:
    """
    Map question relevant_entity_ids → entity names.
    Used for Recall@5: check if any gt name appears in retrieved chunk
    heading or content.
    """
    with open(company_path, encoding="utf-8") as f:
        company = json.load(f)

    id_to_name: dict[str, str] = {}

    for emp in company["employees"]:
        id_to_name[emp["id"]] = emp["name"]
    for dept in company["departments"]:
        id_to_name[dept["id"]] = dept["name"]
    for proj in company["projects"]:
        id_to_name[proj["id"]] = proj["name"]
    for inc in company["incidents"]:
        id_to_name[inc["id"]] = inc["title"]
    for skill in company.get("skills", []):
        id_to_name[skill["id"]] = skill["name"]

    gt_chunks: dict[str, list[str]] = {}
    for q in questions:
        entity_names = [
            id_to_name[eid]
            for eid in q.get("relevant_entity_ids", [])
            if eid in id_to_name
        ]
        gt_chunks[q["id"]] = entity_names

    return gt_chunks


# ── Retriever factory ──────────────────────────────────────────────────────────

def build_retriever(
    variant:  str,
    cfg:      ExperimentConfig,
    embedder: Embedder,
    store:    VectorStore,
    bm25:     BM25Store,
) -> Retriever:
    """Build retriever with specific sources enabled per variant."""
    retriever = Retriever(
        cfg,
        embedder=embedder,
        store=store,
        bm25_store=bm25,
    )
    retriever._use_bm25  = "bm25"  in variant
    retriever._use_graph = "graph" in variant
    return retriever


def _generate_with_retry(
    generator: Generator,
    question:  str,
    attempts:  int = 2,
):
    """Retry generation a small number of times for transient failures."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return generator.generate(question)
        except Exception as ex:
            last_error = ex
            print(f"   [warn] generation attempt {attempt}/{attempts} failed: {ex}")
            if attempt < attempts:
                time.sleep(1)

    if last_error is not None:
        raise last_error


def _judge_with_retry(
    judge:        Judge,
    question:     str,
    ground_truth: str,
    answer:       str,
    attempts:     int = 2,
):
    """Retry judge a small number of times for transient failures."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return judge.score(
                question=question,
                ground_truth=ground_truth,
                answer=answer,
            )
        except Exception as ex:
            last_error = ex
            print(f"   [warn] judge attempt {attempt}/{attempts} failed: {ex}")
            if attempt < attempts:
                time.sleep(1)

    if last_error is not None:
        raise last_error


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_eval(
    config_path: str       = "experiments/baseline.yaml",
    limit:       int | None = None,
    auto_open:   bool      = False,
    question_id:  str | None = None
) -> str:
    """
    Run full evaluation pipeline.

    Args:
        config_path: Path to experiment config YAML
        limit:       Limit number of questions (for testing)
        auto_open:   Auto-open HTML report after run

    Returns:
        Path to results JSON
    """
    load_dotenv(override=True)

    cfg   = load_config(config_path)
    judge = Judge(model="gpt-5.4")

    # Load questions
    questions_path = Path(cfg.eval.questions_path)
    with open(questions_path, encoding="utf-8") as f:
        all_questions = json.load(f)["questions"]

    questions = all_questions[:limit] if limit else all_questions
    if question_id:
        questions = [q for q in questions if q["id"] == question_id]
        if not questions:
            raise ValueError(f"Question {question_id!r} not found")
    print(f"✓ Loaded {len(questions)}/{len(all_questions)} questions")

    # Build ground truth chunks mapping
    gt_chunks = build_gt_chunks(questions, company_path="data/company.json")
    print(f"✓ Ground truth chunks built for {len(gt_chunks)} questions")

    # Pre-load shared components — avoid reloading per variant
    embedder = Embedder(model_name=cfg.embedding.model, device="mps")
    store    = VectorStore(persist_path=".chromadb")
    bm25     = BM25Store()
    if not bm25.load():
        raise RuntimeError(
            "BM25 index not found. Run: uv run -m pipeline.keyword.bm25_store"
        )

    print(f"\n{'='*60}")
    print(f"Running {len(VARIANTS)} variants × {len(questions)} questions")
    print(f"{'='*60}\n")

    results: list[QuestionResult] = []

    for q in questions:
        print(f"── {q['id']} [{q['type']}]")
        print(f"   Q: {q['question']}")

        qr = QuestionResult(
            id=q["id"],
            question=q["question"],
            type=q["type"],
            ground_truth=q["ground_truth"],
        )

        for variant in VARIANTS:
            retriever = build_retriever(variant, cfg, embedder, store, bm25)
            generator = Generator(cfg, retriever=retriever)

            try:
                # Time full pipeline
                t0         = time.perf_counter()
                gen_result = _generate_with_retry(generator, q["question"])
                latency_ms = (time.perf_counter() - t0) * 1000

                # Judge answer quality
                judge_result = _judge_with_retry(
                    judge,
                    question=q["question"],
                    ground_truth=q["ground_truth"],
                    answer=gen_result.answer,
                )

                qr.variants[variant] = VariantResult(
                    variant=variant,
                    retrieved=[
                        {
                            "heading": r.heading,
                            "content": r.content,
                        }
                        for r in gen_result.context
                    ],
                    answer=gen_result.answer,
                    verdict=judge_result.verdict,
                    reasoning=judge_result.reasoning,
                    latency_ms=latency_ms,
                    prompt_tokens=gen_result.prompt_tokens,
                    completion_tokens=gen_result.completion_tokens,
                    generation_cost_usd=estimate_cost_usd(
                        cfg.llm.model,
                        prompt_tokens=gen_result.prompt_tokens,
                        completion_tokens=gen_result.completion_tokens,
                    ),
                    judge_prompt_tokens=judge_result.prompt_tokens,
                    judge_completion_tokens=judge_result.completion_tokens,
                    judge_cost_usd=judge_result.cost_usd,
                )

                verdict_emoji = {
                    "CORRECT":   "✅",
                    "PARTIAL":   "⚠️ ",
                    "INCORRECT": "❌",
                }.get(judge_result.verdict, "? ")

                print(
                    f"   [{variant:20s}] {verdict_emoji} {judge_result.verdict:10s} "
                    f"{latency_ms:6.0f}ms "
                    f"({gen_result.prompt_tokens}+{gen_result.completion_tokens} tok) "
                    f"— {judge_result.reasoning}"
                )
            except Exception as ex:
                qr.variants[variant] = VariantResult(
                    variant=variant,
                    retrieved=[],
                    answer=f"[ERROR] {ex}",
                    verdict="INCORRECT",
                    reasoning=f"Pipeline failed for this variant: {ex}",
                    latency_ms=0.0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    generation_cost_usd=0.0,
                    judge_prompt_tokens=0,
                    judge_completion_tokens=0,
                    judge_cost_usd=0.0,
                )
                print(
                    f"   [{variant:20s}] ❌ ERROR      "
                    f"{0:6.0f}ms (0+0 tok) — {ex}"
                )

        print()
        results.append(qr)

    # Compute metrics
    reports = compute_metrics(results, gt_chunks)

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for variant in VARIANTS:
        if variant in reports:
            print(reports[variant])

    # Save results JSON
    results_dir = Path(cfg.eval.results_dir)
    results_dir.mkdir(exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = results_dir / f"baseline_{timestamp}.json"

    output = {
        "config":    config_path,
        "timestamp": timestamp,
        "variants":  VARIANTS,
        "questions": [
            {
                "id":           qr.id,
                "question":     qr.question,
                "type":         qr.type,
                "ground_truth": qr.ground_truth,
                "variants": {
                    v: {
                        "retrieved":          vr.retrieved,
                        "answer":             vr.answer,
                        "verdict":            vr.verdict,
                        "reasoning":          vr.reasoning,
                        "latency_ms":         vr.latency_ms,
                        "prompt_tokens":      vr.prompt_tokens,
                        "completion_tokens":  vr.completion_tokens,
                        "generation_cost_usd": vr.generation_cost_usd,
                        "judge_prompt_tokens": vr.judge_prompt_tokens,
                        "judge_completion_tokens": vr.judge_completion_tokens,
                        "judge_cost_usd":     vr.judge_cost_usd,
                    }
                    for v, vr in qr.variants.items()
                },
            }
            for qr in results
        ],
        "metrics": {
            v: {
                "accuracy":                r.accuracy,
                "partial_credit":          r.partial_credit,
                "recall_at_5":             r.recall_at_5,
                "retrieval_eligible_total": r.retrieval_eligible_total,
                "recall_hits":             r.recall_hits,
                "answer_accuracy_when_recalled": r.answer_accuracy_when_recalled,
                "answer_partial_credit_when_recalled": r.answer_partial_credit_when_recalled,
                "generation_failure_rate_when_recalled": r.generation_failure_rate_when_recalled,
                "correct_with_recall_hit": r.correct_with_recall_hit,
                "partial_with_recall_hit": r.partial_with_recall_hit,
                "incorrect_with_recall_hit": r.incorrect_with_recall_hit,
                "correct":                 r.correct,
                "partial":                 r.partial,
                "incorrect":               r.incorrect,
                "avg_latency_ms":          r.avg_latency_ms,
                "total_prompt_tokens":     r.total_prompt_tokens,
                "total_completion_tokens": r.total_completion_tokens,
                "total_generation_cost_usd": r.total_generation_cost_usd,
                "total_judge_prompt_tokens": r.total_judge_prompt_tokens,
                "total_judge_completion_tokens": r.total_judge_completion_tokens,
                "total_judge_cost_usd":    r.total_judge_cost_usd,
                "total_eval_cost_usd":     r.total_eval_cost_usd,
                "by_type":                 r.by_type,
            }
            for v, r in reports.items()
            if v in VARIANTS
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✓ Results saved to {output_path}")

    # Generate HTML report
    from eval.report import generate_report
    generate_report(str(output_path), auto_open=auto_open)

    close_driver()
    return str(output_path)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GraphRAG Evaluation Runner")
    parser.add_argument(
        "--config", default="experiments/baseline.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of questions (for testing)",
    )
    parser.add_argument(
        "--open", action="store_true",
        help="Auto-open HTML report after run",
    )
    parser.add_argument(
        "--question", type=str, default=None,
        help="Run single question by id (e.g. q_005)",
    )
    args = parser.parse_args()

    run_eval(
        config_path=args.config,
        limit=args.limit,
        auto_open=args.open,
        question_id=args.question,
    )
