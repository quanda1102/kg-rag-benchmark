"""
eval/metrics.py
────────────────
Compute evaluation metrics from eval results.

Metrics:
    Recall@5         — ground truth entity name appears in heading or content
                       of any top-5 retrieved chunk
    Answer Quality   — CORRECT / PARTIAL / INCORRECT breakdown
    E2E vs Retrieval — answer quality conditioned on retrieval hit
    Latency          — avg query time per variant
    Cost             — token usage per variant

Usage:
    from eval.metrics import compute_metrics

    metrics = compute_metrics(results, gt_chunks)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import re


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TRANSLATION = str.maketrans({
    ".": " ",
    ",": " ",
    ":": " ",
    ";": " ",
    "!": " ",
    "?": " ",
    "(": " ",
    ")": " ",
    "[": " ",
    "]": " ",
    "{": " ",
    "}": " ",
    "\"": " ",
    "'": " ",
    "`": " ",
    "|": " ",
    "/": " ",
    "\\": " ",
    "-": " ",
    "_": " ",
    "—": " ",
    "–": " ",
})


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    """Result for a single question across all variants."""
    id:           str
    question:     str
    type:         str           # single_hop | two_hop | three_hop | ambiguous
    ground_truth: str

    variants: dict[str, "VariantResult"] = field(default_factory=dict)


@dataclass
class VariantResult:
    """Result for one variant on one question."""
    variant:           str
    retrieved:         list[dict[str, str]] | list[str]
    answer:            str
    verdict:           str         # CORRECT | PARTIAL | INCORRECT
    reasoning:         str
    latency_ms:        float
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    generation_cost_usd: float = 0.0
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    judge_cost_usd:      float = 0.0


@dataclass
class MetricsReport:
    """Aggregated metrics across all questions."""
    variant:                  str
    total:                    int
    correct:                  int
    partial:                  int
    incorrect:                int
    retrieval_eligible_total: int
    recall_hits:              int
    correct_with_recall_hit:  int
    partial_with_recall_hit:  int
    incorrect_with_recall_hit:int
    recall_at_5:              float
    avg_latency_ms:           float
    total_prompt_tokens:      int
    total_completion_tokens:  int
    total_generation_cost_usd: float
    total_judge_prompt_tokens: int
    total_judge_completion_tokens: int
    total_judge_cost_usd:     float
    by_type:                  dict[str, dict] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def partial_credit(self) -> float:
        """Accuracy counting PARTIAL as 0.5."""
        return (self.correct + 0.5 * self.partial) / self.total if self.total else 0.0

    @property
    def answer_accuracy_when_recalled(self) -> float:
        """End-to-end correctness among cases where retrieval hit top-k."""
        return (
            self.correct_with_recall_hit / self.recall_hits
            if self.recall_hits else 0.0
        )

    @property
    def answer_partial_credit_when_recalled(self) -> float:
        """End-to-end partial-credit score among retrieval-hit cases."""
        return (
            (self.correct_with_recall_hit + 0.5 * self.partial_with_recall_hit) / self.recall_hits
            if self.recall_hits else 0.0
        )

    @property
    def generation_failure_rate_when_recalled(self) -> float:
        """
        How often answer is still incorrect even though retrieval hit.
        This is a proxy for post-retrieval failure (generation/reasoning).
        """
        return (
            self.incorrect_with_recall_hit / self.recall_hits
            if self.recall_hits else 0.0
        )

    @property
    def total_eval_cost_usd(self) -> float:
        return self.total_generation_cost_usd + self.total_judge_cost_usd

    def __repr__(self) -> str:
        return (
            f"\n── {self.variant} ──────────────────\n"
            f"  E2E Accuracy:   {self.accuracy:.1%}  ({self.correct}/{self.total} CORRECT)\n"
            f"  Partial credit: {self.partial_credit:.1%}\n"
            f"  PARTIAL:        {self.partial}/{self.total}\n"
            f"  INCORRECT:      {self.incorrect}/{self.total}\n"
            f"  Recall@5:       {self.recall_at_5:.1%}  ({self.recall_hits}/{self.retrieval_eligible_total})\n"
            f"  Acc|RecallHit:  {self.answer_accuracy_when_recalled:.1%}\n"
            f"  PC|RecallHit:   {self.answer_partial_credit_when_recalled:.1%}\n"
            f"  GenFail|Hit:    {self.generation_failure_rate_when_recalled:.1%}\n"
            f"  Avg latency:    {self.avg_latency_ms:.0f}ms\n"
            f"  Tokens (prompt/completion): "
            f"{self.total_prompt_tokens:,} / {self.total_completion_tokens:,}\n"
            f"  Generation cost:${self.total_generation_cost_usd:.4f}\n"
            f"  Judge cost:     ${self.total_judge_cost_usd:.4f}\n"
            f"  Total eval cost:${self.total_eval_cost_usd:.4f}\n"
        )


# ── Recall@5 helpers ───────────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """Light normalization for retrieval matching."""
    if not text:
        return ""

    text = text.lower().strip().translate(_PUNCT_TRANSLATION)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _matches_text(gt_name: str, text: str) -> bool:
    """Match entity name against normalized text."""
    gt = _normalize_text(gt_name)
    target = _normalize_text(text)

    if not gt or not target:
        return False

    if len(gt) < 4:
        words = target.split()
        return gt in words

    return gt in target


def _chunk_matches(gt_name: str, retrieved_chunk: dict[str, str] | str) -> bool:
    """
    Check if ground truth entity name matches a retrieved chunk.

    Supports both:
    - new schema: {"heading": "...", "content": "..."}
    - old schema: "heading only"
    """
    if isinstance(retrieved_chunk, str):
        return _matches_text(gt_name, retrieved_chunk)

    heading = retrieved_chunk.get("heading", "")
    content = retrieved_chunk.get("content", "")
    return _matches_text(gt_name, heading) or _matches_text(gt_name, content)


def recall_at_k(
    gt_names:  list[str],
    retrieved: list[dict[str, str]] | list[str],
    k:         int = 5,
) -> bool:
    """Return True if any gt_name matches any of the top-k retrieved chunks."""
    top_k = retrieved[:k]
    return any(
        _chunk_matches(gt, chunk)
        for gt in gt_names
        for chunk in top_k
    )


# ── Metrics computation ────────────────────────────────────────────────────────

def compute_metrics(
    results:             list[QuestionResult],
    ground_truth_chunks: dict[str, list[str]],  # question_id → relevant entity names
) -> dict[str, "MetricsReport"]:
    """
    Compute metrics for each variant.

    Args:
        results:               List of QuestionResult
        ground_truth_chunks:   Map from question_id to list of relevant entity names
    """
    # Collect all variant names
    variants: set[str] = set()
    for r in results:
        variants.update(r.variants.keys())

    reports: dict[str, MetricsReport] = {}

    for variant in variants:
        total                    = 0
        correct                  = 0
        partial                  = 0
        incorrect                = 0
        recall_hits              = 0
        retrieval_eligible_total = 0
        correct_with_recall_hit  = 0
        partial_with_recall_hit  = 0
        incorrect_with_recall_hit = 0
        latencies:  list[float] = []
        prompt_tokens     = 0
        completion_tokens = 0
        generation_cost_usd = 0.0
        judge_prompt_tokens = 0
        judge_completion_tokens = 0
        judge_cost_usd = 0.0

        by_type: dict[str, dict] = defaultdict(lambda: {
            "total": 0, "correct": 0, "partial": 0, "incorrect": 0,
            "retrieval_eligible_total": 0,
            "recall_hits": 0,
            "correct_with_recall_hit": 0,
            "partial_with_recall_hit": 0,
            "incorrect_with_recall_hit": 0,
        })

        for q in results:
            if variant not in q.variants:
                continue

            vr = q.variants[variant]
            total += 1
            by_type[q.type]["total"] += 1

            # Answer quality
            if vr.verdict == "CORRECT":
                correct += 1
                by_type[q.type]["correct"] += 1
            elif vr.verdict == "PARTIAL":
                partial += 1
                by_type[q.type]["partial"] += 1
            else:
                incorrect += 1
                by_type[q.type]["incorrect"] += 1

            # Recall@5
            gt_names = ground_truth_chunks.get(q.id, [])
            if gt_names:
                retrieval_eligible_total += 1
                by_type[q.type]["retrieval_eligible_total"] += 1
                hit = recall_at_k(gt_names, vr.retrieved, k=5)
                if hit:
                    recall_hits += 1
                    by_type[q.type]["recall_hits"] += 1
                    if vr.verdict == "CORRECT":
                        correct_with_recall_hit += 1
                        by_type[q.type]["correct_with_recall_hit"] += 1
                    elif vr.verdict == "PARTIAL":
                        partial_with_recall_hit += 1
                        by_type[q.type]["partial_with_recall_hit"] += 1
                    else:
                        incorrect_with_recall_hit += 1
                        by_type[q.type]["incorrect_with_recall_hit"] += 1

            # Latency + tokens
            latencies.append(vr.latency_ms)
            prompt_tokens     += vr.prompt_tokens
            completion_tokens += vr.completion_tokens
            generation_cost_usd += getattr(vr, "generation_cost_usd", 0.0)
            judge_prompt_tokens += getattr(vr, "judge_prompt_tokens", 0)
            judge_completion_tokens += getattr(vr, "judge_completion_tokens", 0)
            judge_cost_usd += getattr(vr, "judge_cost_usd", 0.0)

        # Add accuracy to by_type
        for t, td in by_type.items():
            t_total = td["total"]
            retrieval_total = td["retrieval_eligible_total"]
            recall_hit_total = td["recall_hits"]
            td["accuracy"] = round(td["correct"] / t_total, 3) if t_total else 0.0
            td["partial_credit"] = round(
                (td["correct"] + 0.5 * td["partial"]) / t_total, 3
            ) if t_total else 0.0
            td["recall_at_5"] = round(
                td["recall_hits"] / retrieval_total, 3
            ) if retrieval_total else 0.0
            td["answer_accuracy_when_recalled"] = round(
                td["correct_with_recall_hit"] / recall_hit_total, 3
            ) if recall_hit_total else 0.0
            td["answer_partial_credit_when_recalled"] = round(
                (
                    td["correct_with_recall_hit"]
                    + 0.5 * td["partial_with_recall_hit"]
                ) / recall_hit_total, 3
            ) if recall_hit_total else 0.0
            td["generation_failure_rate_when_recalled"] = round(
                td["incorrect_with_recall_hit"] / recall_hit_total, 3
            ) if recall_hit_total else 0.0

        reports[variant] = MetricsReport(
            variant=variant,
            total=total,
            correct=correct,
            partial=partial,
            incorrect=incorrect,
            retrieval_eligible_total=retrieval_eligible_total,
            recall_hits=recall_hits,
            correct_with_recall_hit=correct_with_recall_hit,
            partial_with_recall_hit=partial_with_recall_hit,
            incorrect_with_recall_hit=incorrect_with_recall_hit,
            recall_at_5=(
                recall_hits / retrieval_eligible_total
                if retrieval_eligible_total else 0.0
            ),
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
            total_prompt_tokens=prompt_tokens,
            total_completion_tokens=completion_tokens,
            total_generation_cost_usd=generation_cost_usd,
            total_judge_prompt_tokens=judge_prompt_tokens,
            total_judge_completion_tokens=judge_completion_tokens,
            total_judge_cost_usd=judge_cost_usd,
            by_type=dict(by_type),
        )

    return reports
