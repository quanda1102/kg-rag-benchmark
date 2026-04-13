"""
pipeline/generation/generator.py
──────────────────────────────────
Generate answers from retrieved context using ReAct pattern.

Flow (ReAct):
    Query
      → Think: phân tích cần tìm entity nào (Pydantic structured output)
      → Act:   retrieve sub-query
      → Observe: extract intermediate entity
      → (lặp tối đa MAX_HOPS lần)
      → Final answer

Usage:
    from pipeline.generation.generator import Generator
    from pipeline.config import load_config

    cfg       = load_config("experiments/baseline.yaml")
    generator = Generator(cfg)
    result    = generator.generate("ai đang làm Legacy CRM Sunset?")
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI
from pydantic import BaseModel

from pipeline.config import ExperimentConfig
from pipeline.retrieval.retriever import Retriever, RetrievalResult


# ── Constants ──────────────────────────────────────────────────────────────────

MAX_HOPS = 3  # max reasoning steps = max hop count in dataset


# ── Pydantic schema for Think step ────────────────────────────────────────────

class ThinkResult(BaseModel):
    done:                bool
    thought:             str
    sub_query:           str = ""
    intermediate_entity: str = ""


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class ReActStep:
    thought:   str
    sub_query: str
    retrieved: list[RetrievalResult]
    entity:    str  # intermediate entity extracted at this step


@dataclass
class GenerationResult:
    query:             str
    answer:            str
    context:           list[RetrievalResult]
    model:             str
    steps:             list[ReActStep] = field(default_factory=list)
    prompt_tokens:     int = 0
    completion_tokens: int = 0

    def __repr__(self) -> str:
        return (
            f"GenerationResult(\n"
            f"  query={self.query!r}\n"
            f"  answer={self.answer[:200]!r}\n"
            f"  hops={len(self.steps)}\n"
            f"  context_chunks={len(self.context)}\n"
            f"  tokens={self.prompt_tokens}+{self.completion_tokens}\n"
            f")"
        )


# ── Prompts ────────────────────────────────────────────────────────────────────

_THINK_SYSTEM = """
Bạn là reasoning engine cho hệ thống RAG của TechViet Solutions.
Nhiệm vụ: Phân tích câu hỏi và context hiện tại, trả về JSON object để quyết định bước tiếp theo.

QUAN TRỌNG - Mapping thuật ngữ Anh-Việt trong context:
  critical ↔ Nghiêm trọng
  high ↔ Cao  
  medium ↔ Trung bình
  low ↔ Thấp
  open ↔ Mở
  resolved/closed ↔ Đã xử lý
Khi đánh giá context, áp dụng mapping này trước khi kết luận thiếu thông tin.

JSON object gồm các field:
- done (bool): true nếu context hiện tại đã đủ để trả lời câu hỏi gốc
- thought (str): reasoning về bước tiếp theo cần làm
- sub_query (str): câu query cụ thể để retrieve thêm (nếu done=false)
- intermediate_entity (str): tên entity vừa xác định được từ context ở bước này

Quy tắc:
- Nếu context đã đủ để trả lời câu hỏi gốc → done=true, sub_query=""
- Nếu chưa đủ → xác định entity trung gian cần tìm, đặt sub_query là tên cụ thể
- sub_query phải là tên người/project/incident/department cụ thể
- intermediate_entity là entity vừa extract được từ context ở bước này
""".strip()

_THINK_USER = """
Câu hỏi gốc: {question}

Context đã có:
{context}

Trajectory các bước trước:
{trajectory}

Phân tích bước tiếp theo:
""".strip()

_ANSWER_SYSTEM = """
Bạn là trợ lý AI của TechViet Solutions.

Chỉ trả lời dựa trên context được cung cấp.

Cách làm:
1. Xác định các mục liên quan trong context.
2. Trích nguyên văn các thuộc tính liên quan của từng mục.
3. So khớp từng điều kiện trong câu hỏi với các thuộc tính đó.
4. Nếu có ít nhất một mục thỏa tất cả điều kiện, phải trả lời bằng mục đó.
5. Chỉ được kết luận "không tìm thấy" khi không có mục nào thỏa tất cả điều kiện.

Lưu ý:
- Câu hỏi có thể dùng nhãn tiếng Anh, còn context dùng nhãn tiếng Việt tương đương:
  critical ↔ Nghiêm trọng | high ↔ Cao | medium ↔ Trung bình | low ↔ Thấp
  open ↔ Mở | resolved ↔ Đã xử lý
- Không tự dịch hoặc chuẩn hóa thuộc tính nếu không cần.
- Khi trích thuộc tính từ context, ưu tiên giữ nguyên cách ghi trong context.
- Không suy đoán ngoài context.
""".strip()

_ANSWER_USER = """
Context:
{context}

Câu hỏi: {query}
""".strip()


# ── Generator ──────────────────────────────────────────────────────────────────

class Generator:
    """
    ReAct-based generator: Think → Act → Observe → Answer.

    Args:
        cfg:       ExperimentConfig
        retriever: Retriever instance (reuse if already loaded)
    """

    def __init__(
        self,
        cfg:       ExperimentConfig,
        retriever: Retriever | None = None,
    ) -> None:
        self.cfg       = cfg
        self.client    = OpenAI()
        self.retriever = retriever or Retriever(cfg)
        print(f"✓ Generator ready — model={cfg.llm.model!r}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(self, query: str) -> GenerationResult:
        """
        ReAct pipeline:
          0. Initial retrieve on original query
          1. Think → decide if done or need more hops
          2. Act   → retrieve sub_query
          3. Observe → record intermediate entity
          4. Repeat until done or MAX_HOPS reached
          5. Final answer generation
        """
        steps:       list[ReActStep]       = []
        all_context: list[RetrievalResult] = []
        total_prompt_tokens     = 0
        total_completion_tokens = 0

        # ── Step 0: Initial retrieve ───────────────────────────────────────────
        initial_results = self.retriever.retrieve(query)
        all_context     = self._merge_context(all_context, initial_results)
        print(f"\n[ReAct] initial retrieve → {len(initial_results)} chunks")

        # ── ReAct loop ─────────────────────────────────────────────────────────
        for hop in range(MAX_HOPS):
            context_str = self._assemble_context(all_context)
            trajectory  = self._format_trajectory(steps)

            think_result, p_tok, c_tok = self._think(
                question=query,
                context=context_str,
                trajectory=trajectory,
            )
            total_prompt_tokens     += p_tok
            total_completion_tokens += c_tok

            print(
                f"[ReAct] hop={hop+1} done={think_result.done} "
                f"thought={think_result.thought[:80]}"
            )

            # Done — enough context to answer
            if think_result.done:
                print(f"[ReAct] sufficient context after {hop+1} hop(s)")
                break

            sub_query = think_result.sub_query.strip()
            if not sub_query:
                print("[ReAct] no sub_query returned, stopping early")
                break

            # Act: retrieve on sub_query
            print(f"[ReAct] sub_query={sub_query!r}")
            new_results = self.retriever.retrieve(sub_query)
            all_context = self._merge_context(all_context, new_results)

            steps.append(ReActStep(
                thought=think_result.thought,
                sub_query=sub_query,
                retrieved=new_results,
                entity=think_result.intermediate_entity,
            ))

        # ── Final answer ───────────────────────────────────────────────────────
        final_context = self._assemble_context(all_context)
        answer, p_tok, c_tok = self._answer(query, final_context)
        total_prompt_tokens     += p_tok
        total_completion_tokens += c_tok

        return GenerationResult(
            query=query,
            answer=answer,
            context=all_context,
            model=self.cfg.llm.model,
            steps=steps,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
        )

    # ── Private ────────────────────────────────────────────────────────────────

    def _think(
        self,
        question:   str,
        context:    str,
        trajectory: str,
    ) -> tuple[ThinkResult, int, int]:
        """
        Ask LLM to reason about next step using Pydantic structured output.
        Returns (ThinkResult, prompt_tokens, completion_tokens).
        """
        response = self.client.beta.chat.completions.parse(
            model=self.cfg.llm.model,
            messages=[
                {"role": "system", "content": _THINK_SYSTEM},
                {"role": "user",   "content": _THINK_USER.format(
                    question=question,
                    context=context[:4000],  # cap to avoid huge think prompts
                    trajectory=trajectory,
                )},
            ],
            temperature=0,
            max_completion_tokens=512,
            response_format=ThinkResult,
        )

        usage = response.usage

        # Model refused to answer
        if response.choices[0].message.refusal:
            print(f"  ⚠ Think step refused: {response.choices[0].message.refusal}")
            return (
                ThinkResult(done=True, thought="refusal", sub_query="", intermediate_entity=""),
                usage.prompt_tokens     if usage else 0,
                usage.completion_tokens if usage else 0,
            )

        parsed = response.choices[0].message.parsed

        # Truncated response — treat as done to avoid bad state
        if response.choices[0].finish_reason == "length":
            print("  ⚠ Think step truncated, treating as done")
            return (
                ThinkResult(done=True, thought="truncated", sub_query="", intermediate_entity=""),
                usage.prompt_tokens     if usage else 0,
                usage.completion_tokens if usage else 0,
            )

        return (
            parsed,
            usage.prompt_tokens     if usage else 0,
            usage.completion_tokens if usage else 0,
        )

    def _answer(
        self,
        query:   str,
        context: str,
    ) -> tuple[str, int, int]:
        """Final answer generation with full accumulated context."""
        response = self.client.chat.completions.create(
            model=self.cfg.llm.model,
            messages=[
                {"role": "system", "content": _ANSWER_SYSTEM},
                {"role": "user",   "content": _ANSWER_USER.format(
                    context=context,
                    query=query,
                )},
            ],
            temperature=0,
            max_completion_tokens=self.cfg.llm.extraction_max_tokens,
        )
        usage = response.usage
        return (
            response.choices[0].message.content.strip(),
            usage.prompt_tokens     if usage else 0,
            usage.completion_tokens if usage else 0,
        )

    @staticmethod
    def _assemble_context(results: list[RetrievalResult]) -> str:
        parts = []
        for i, r in enumerate(results):
            header = f"[{i+1}] {r.heading}" if r.heading else f"[{i+1}]"
            parts.append(f"{header}\n{r.content}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _merge_context(
        existing: list[RetrievalResult],
        new:      list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Merge new results into existing, dedup by chunk_id.
        Preserves original order, appends new chunks at end.
        """
        seen     = {r.chunk_id for r in existing}
        combined = list(existing)
        for r in new:
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                combined.append(r)
        return combined

    @staticmethod
    def _format_trajectory(steps: list[ReActStep]) -> str:
        if not steps:
            return "(chưa có bước nào)"
        lines = []
        for i, s in enumerate(steps):
            lines.append(
                f"Bước {i+1}:\n"
                f"  Thought: {s.thought}\n"
                f"  Sub-query: {s.sub_query}\n"
                f"  Entity found: {s.entity}"
            )
        return "\n\n".join(lines)


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pipeline.config import load_config
    from pipeline.graph.connection import close_driver
    from dotenv import load_dotenv

    load_dotenv(override=True)

    cfg       = load_config("experiments/baseline.yaml")
    generator = Generator(cfg)

    queries = [
        # single-hop — should stop after hop 1
        "Phòng Engineering hiện có bao nhiêu nhân viên?",
        # two-hop
        "Ai là manager của người được giao xử lý incident "
        "\"Support assistant returned outdated policy snippets\"?",
        # three-hop
        "Department head của người được giao incident "
        "\"Checkout API timeout spike during merchant flash sale\" là ai?",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        result = generator.generate(query)
        print(f"Query:  {result.query}")
        print(f"Answer: {result.answer}")
        print(f"Hops:   {len(result.steps)}")
        print(f"Tokens: {result.prompt_tokens}+{result.completion_tokens}")
        print(f"Context chunks: {len(result.context)}")
        for step in result.steps:
            print(f"  → sub_query={step.sub_query!r} entity={step.entity!r}")

    close_driver()