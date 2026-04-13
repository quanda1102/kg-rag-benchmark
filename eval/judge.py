"""
eval/judge.py
──────────────
LLM-as-judge for answer quality evaluation.

Scores each answer as CORRECT / PARTIAL / INCORRECT
with one-line reasoning.

Usage:
    from eval.judge import Judge

    judge = Judge()
    result = judge.score(
        question="Lê Quang Huy quản lý ai?",
        ground_truth="Lê Quang Huy quản lý Trần Đức Long...",
        answer="Lê Quang Huy quản lý các trưởng bộ phận..."
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel


# ── Schema ─────────────────────────────────────────────────────────────────────

Verdict = Literal["CORRECT", "PARTIAL", "INCORRECT"]


class JudgeOutput(BaseModel):
    verdict:   Verdict
    reasoning: str


# ── Pricing ────────────────────────────────────────────────────────────────────

_MODEL_PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "gpt-5.4": {
        "input": 2.50,
        "output": 15.00,
    },
    "gpt-5.4-mini": {
        "input": 0.75,
        "output": 4.50,
    },
    "gpt-5.4-nano": {
        "input": 0.20,
        "output": 1.25,
    },
}


def _normalize_model_name(model: str) -> str:
    """Normalize model alias/snapshot to pricing key when possible."""
    if model.startswith("gpt-5.4-mini"):
        return "gpt-5.4-mini"
    if model.startswith("gpt-5.4-nano"):
        return "gpt-5.4-nano"
    if model.startswith("gpt-5.4"):
        return "gpt-5.4"
    return model


def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Estimate API cost in USD from token usage."""
    normalized_model = _normalize_model_name(model)
    pricing = _MODEL_PRICING_USD_PER_1M.get(normalized_model)
    if not pricing:
        return 0.0

    return (
        (prompt_tokens / 1_000_000) * pricing["input"]
        + (completion_tokens / 1_000_000) * pricing["output"]
    )


# ── Prompt ─────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
Bạn là evaluator khách quan đánh giá chất lượng câu trả lời của hệ thống RAG.

Đây là benchmark end-to-end cho pipeline retrieval + generation.
Bạn đang chấm chất lượng câu trả lời cuối cùng của hệ thống, không chấm riêng retrieval.

Cho:
- Câu hỏi gốc
- Ground truth
- System answer

Nhiệm vụ:
Đánh giá xem kết luận cuối cùng của system answer có đúng với ground truth hay không.

Thang điểm:
- CORRECT:
  - Kết luận cuối cùng khớp với ground truth
  - Không có mâu thuẫn đáng kể
  - Có thể khác cách diễn đạt nhưng không đổi nghĩa

- PARTIAL:
  - Có một phần thông tin đúng hoặc đúng entity chính
  - Nhưng thiếu chi tiết quan trọng, chưa đầy đủ, hoặc diễn đạt chưa đủ rõ để coi là hoàn toàn đúng
  - Chỉ dùng PARTIAL khi câu trả lời không mâu thuẫn trực tiếp với ground truth

- INCORRECT:
  - Kết luận cuối cùng sai hoặc mâu thuẫn với ground truth
  - Trả lời phủ định khi ground truth khẳng định có đáp án cụ thể
  - Trả lời khẳng định sai entity / sai thuộc tính / sai quan hệ
  - Không liên quan hoặc bịa đặt

Quy tắc quan trọng:
- Ưu tiên chấm theo kết luận cuối cùng, không chỉ theo việc có nhắc vài chi tiết đúng
- Nếu answer chứa thông tin đúng nhưng kết luận cuối mâu thuẫn với ground truth, verdict phải là INCORRECT
- Nếu answer có vẻ đã retrieve hoặc nhắc đúng candidate nhưng kết luận cuối vẫn sai, verdict vẫn phải là INCORRECT
- Không suy đoán retrieval có thể đúng hay sai; chỉ chấm đầu ra cuối cùng so với câu hỏi và ground truth
- Không chấm theo văn phong hay độ dài
- Reasoning chỉ 1 câu ngắn, nêu rõ vì sao
""".strip()

_USER_TEMPLATE = """
Câu hỏi: {question}

Ground truth: {ground_truth}

System answer: {answer}
""".strip()


# ── Judge ──────────────────────────────────────────────────────────────────────

@dataclass
class JudgeResult:
    model:        str
    question:     str
    ground_truth: str
    answer:       str
    verdict:      Verdict
    reasoning:    str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd:      float = 0.0

    def __repr__(self) -> str:
        return (
            f"JudgeResult(verdict={self.verdict!r}, "
            f"reasoning={self.reasoning!r})"
        )


class Judge:
    """
    LLM-as-judge for answer quality.

    Args:
        model: OpenAI model. Default: gpt-5.4-mini
    """

    def __init__(self, model: str = "gpt-5.4") -> None:
        self.model  = model
        self.client = OpenAI()

    def score(
        self,
        question:     str,
        ground_truth: str,
        answer:       str,
    ) -> JudgeResult:
        """Score a single answer."""
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _USER_TEMPLATE.format(
                    question=question,
                    ground_truth=ground_truth,
                    answer=answer,
                )},
            ],
            temperature=0,
            response_format=JudgeOutput,
        )
        output = response.choices[0].message.parsed
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        return JudgeResult(
            model=self.model,
            question=question,
            ground_truth=ground_truth,
            answer=answer,
            verdict=output.verdict,
            reasoning=output.reasoning,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=estimate_cost_usd(
                self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
        )

    def score_batch(
        self,
        items: list[dict],  # [{question, ground_truth, answer}]
    ) -> list[JudgeResult]:
        """Score a batch of answers."""
        results = []
        for item in items:
            result = self.score(
                question=item["question"],
                ground_truth=item["ground_truth"],
                answer=item["answer"],
            )
            results.append(result)
        return results


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)

    judge = Judge()

    test_cases = [
        {
            "question":     "Lê Quang Huy quản lý ai?",
            "ground_truth": "Lê Quang Huy quản lý Trần Đức Long, Lê Mỹ Duyên, Vương Thanh Hà, Nguyễn Thị Quỳnh, Phan Tuấn Kiệt, Võ Thanh Sơn, Mạc Thiên An, Nguyễn Hữu Phúc.",
            "answer":       "Lê Quang Huy quản lý Trần Đức Long, Lê Mỹ Duyên, Vương Thanh Hà, Nguyễn Thị Quỳnh.",
        },
        {
            "question":     "Project nào đang paused?",
            "ground_truth": "Legacy CRM Sunset đang ở trạng thái paused.",
            "answer":       "Không có project nào đang paused.",
        },
    ]

    for case in test_cases:
        result = judge.score(**case)
        print(f"Q: {case['question']}")
        print(f"→ {result}")
        print()
