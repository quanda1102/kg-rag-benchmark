"""
pipeline/graph/entity_extractor.py
─────────────────────────────────────────
Extract entities and relations from chunks using OpenAI structured outputs.

Entity types:
    PERSON, PROJECT, DEPARTMENT, INCIDENT, SKILL

Relation types:
    MANAGES, WORKS_ON, BELONGS_TO, REPORTED, AFFECTS, OWNED_BY, HAS_SKILL

Usage:
    from pipeline.graph.entity_extractor import EntityExtractor

    extractor = EntityExtractor()
    entities, relations = asyncio.run(extractor.extract_all(chunks))
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel

if TYPE_CHECKING:
    from pipeline.preprocessing.chunker import Chunk


# ── Usage Tracing ──────────────────────────────────────────────────────────────

@dataclass
class UsageStats:
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    total_tokens:      int = 0
    total_chunks:      int = 0

    def add(self, usage) -> None:
        self.prompt_tokens     += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens      += usage.total_tokens
        self.total_chunks      += 1

    def report(self, model: str) -> None:
        # gpt-5.4-mini pricing: $0.075/1M input, $4.50/1M output
        input_cost  = self.prompt_tokens     / 1_000_000 * 0.075
        output_cost = self.completion_tokens / 1_000_000 * 4.5
        total_cost  = input_cost + output_cost
        print(
            f"\n── Token Usage ──────────────────\n"
            f"  model:       {model}\n"
            f"  chunks:      {self.total_chunks}\n"
            f"  prompt:      {self.prompt_tokens:,} tokens  (${input_cost:.4f})\n"
            f"  completion:  {self.completion_tokens:,} tokens  (${output_cost:.4f})\n"
            f"  total:       {self.total_tokens:,} tokens  (${total_cost:.4f})\n"
            f"─────────────────────────────────"
        )


# ── Pydantic schema (structured output) ───────────────────────────────────────

EntityType   = Literal["PERSON", "PROJECT", "DEPARTMENT", "INCIDENT", "SKILL"]
RelationType = Literal["MANAGES", "WORKS_ON", "BELONGS_TO", "REPORTED", "AFFECTS", "OWNED_BY", "HAS_SKILL"]

_ALLOWED_TYPES: dict[str, tuple[str, str]] = {
    "MANAGES":    ("PERSON",   "PERSON"),
    "WORKS_ON":   ("PERSON",   "PROJECT"),
    "BELONGS_TO": ("PERSON",   "DEPARTMENT"),
    "REPORTED":   ("PERSON",   "INCIDENT"),
    "AFFECTS":    ("INCIDENT", "PROJECT"),
    "OWNED_BY":   ("PROJECT",  "DEPARTMENT"),
    "HAS_SKILL":  ("PERSON",   "SKILL"),
}


class ExtractedEntity(BaseModel):
    name:       str
    type:       EntityType
    properties: Optional[dict[str, str]] = None


class ExtractedRelation(BaseModel):
    source:     str
    target:     str
    relation:   RelationType
    properties: Optional[dict[str, str]] = None


class ExtractionResult(BaseModel):
    entities:  list[ExtractedEntity]
    relations: list[ExtractedRelation]


# ── Domain models ──────────────────────────────────────────────────────────────

class Entity(BaseModel):
    id:         str
    name:       str
    type:       EntityType
    chunk_id:   str
    properties: dict[str, str] = {}

    def __repr__(self) -> str:
        return f"Entity(name={self.name!r}, type={self.type!r})"


class Relation(BaseModel):
    source:     str
    target:     str
    relation:   RelationType
    properties: dict[str, str] = {}

    def __repr__(self) -> str:
        return f"Relation({self.source!r} -[{self.relation}]-> {self.target!r})"


# ── Prompt ─────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
Bạn là hệ thống trích xuất thực thể và quan hệ từ văn bản tiếng Việt.

Trích xuất các thực thể thuộc loại:
- PERSON: tên người (vd: Lê Quang Huy)
- PROJECT: tên dự án (vd: Legacy CRM Sunset)
- DEPARTMENT: tên phòng ban (vd: Engineering, Product)
- INCIDENT: tên sự cố (vd: Checkout API timeout spike)
- SKILL: kỹ năng kỹ thuật (vd: Java, Kubernetes, Python)

Trích xuất các quan hệ thuộc loại (chú ý chiều source → target):
- MANAGES:    source=PERSON,     target=PERSON,     ý nghĩa: người quản lý người khác
- WORKS_ON:   source=PERSON,     target=PROJECT,    ý nghĩa: người làm việc trong dự án
- BELONGS_TO: source=PERSON,     target=DEPARTMENT, ý nghĩa: người thuộc phòng ban
- REPORTED:   source=PERSON,     target=INCIDENT,   ý nghĩa: người báo cáo sự cố
- AFFECTS:    source=INCIDENT,   target=PROJECT,    ý nghĩa: sự cố ảnh hưởng dự án
- OWNED_BY:   source=PROJECT,    target=DEPARTMENT, ý nghĩa: dự án thuộc phòng ban
- HAS_SKILL:  source=PERSON,     target=SKILL,      ý nghĩa: người có kỹ năng

━━━ QUY TẮC CHIỀU QUAN HỆ ━━━

"X báo cáo trực tiếp cho Y" → Y MANAGES X  (Y là source, X là target)
"X được assign cho Y"        → Y WORKS_ON incident  (không phải X REPORTED)
"X phụ trách dự án Y"        → X WORKS_ON Y
"X là project owner của Y"   → X WORKS_ON Y  (với role=owner trong properties)
"X thuộc phòng Y"            → X BELONGS_TO Y

━━━ VÍ DỤ ━━━

Input:
  "Khưu Anh Thư báo cáo trực tiếp cho Võ Thanh Sơn, Head of DevOps."

Đúng:
  {"source": "Võ Thanh Sơn", "target": "Khưu Anh Thư", "relation": "MANAGES"}

Sai:
  {"source": "Khưu Anh Thư", "target": "Võ Thanh Sơn", "relation": "MANAGES"}  ✗

---

Input:
  "Incident được assign cho La Minh Tâm, ML Engineer."

Đúng:
  {"source": "La Minh Tâm", "target": "incident_name", "relation": "WORKS_ON"}

Sai:
  {"source": "La Minh Tâm", "target": "incident_name", "relation": "REPORTED"}  ✗

---

Input:
  "Lý Khánh Linh là project owner của Legacy CRM Sunset."

Đúng:
  {"source": "Lý Khánh Linh", "target": "Legacy CRM Sunset", "relation": "WORKS_ON", "properties": {"role": "project owner"}}

---

Input:
  "Phòng Finance làm việc sát với Data và DevOps."

Đúng:
  → Không extract quan hệ này vì không có relation type phù hợp cho DEPARTMENT → DEPARTMENT.

━━━ LƯU Ý ━━━
- Chỉ extract quan hệ nếu source và target đúng loại entity như định nghĩa trên
- Không được đảo ngược chiều quan hệ
- Bỏ qua quan hệ không có relation type phù hợp
""".strip()


# ── EntityExtractor ────────────────────────────────────────────────────────────

class EntityExtractor:
    """
    Extract entities and relations from Chunks using OpenAI async structured outputs.

    Args:
        model:          OpenAI model. Default: gpt-5.4-mini
        api_key:        OpenAI API key. Default: reads from OPENAI_API_KEY env
        max_concurrent: Max concurrent API calls. Default: 5
    """

    def __init__(
        self,
        model:          str      = "gpt-5.4-mini",
        api_key:        str | None = None,
        max_concurrent: int      = 5,
    ) -> None:
        self.model          = model
        self.max_concurrent = max_concurrent
        self.client         = AsyncOpenAI(api_key=api_key)
        self.usage          = UsageStats()
        print(f"✓ EntityExtractor ready — model={model!r}, max_concurrent={max_concurrent}")

    # ── Public API ─────────────────────────────────────────────────────────────

    async def extract(self, chunk: Chunk) -> tuple[list[Entity], list[Relation]]:
        """Extract entities and relations from a single Chunk."""
        result = await self._call_llm(chunk.content)
        return self._build(result, chunk.id)

    async def extract_all(
        self,
        chunks: list[Chunk],
    ) -> tuple[list[Entity], list[Relation]]:
        """
        Extract from all chunks concurrently.
        - asyncio.gather preserves order → dedup is deterministic
        - Semaphore limits concurrent API calls to avoid rate limit
        - Relations remapped to canonical ids via global registry
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_chunk(
            i: int,
            chunk: Chunk,
        ) -> tuple[list[Entity], list[Relation]] | Exception:
            async with semaphore:
                print(f"  [{i+1}/{len(chunks)}] {chunk.heading[:50]!r}")
                try:
                    result = await self._call_llm(chunk.content)
                    return self._build(result, chunk.id)
                except Exception as ex:
                    print(f"  ⚠ skipped {chunk.id}: {ex}")
                    return ex

        # gather preserves order: results[i] = chunks[i]
        raw_results = await asyncio.gather(
            *[process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        )

        # Merge results sequentially — global_name_to_id must be built in order
        all_entities:      list[Entity]   = []
        all_relations:     list[Relation] = []
        seen_entities:     set[str]       = set()
        global_name_to_id: dict[str, str] = {}

        for result in raw_results:
            if isinstance(result, Exception):
                continue

            entities, relations = result

            for e in entities:
                key = f"{e.name}__{e.type}"
                if key not in seen_entities:
                    seen_entities.add(key)
                    all_entities.append(e)
                if e.name not in global_name_to_id:
                    global_name_to_id[e.name] = e.id

            for r in relations:
                source_name = next(
                    (e.name for e in entities if e.id == r.source), None
                )
                target_name = next(
                    (e.name for e in entities if e.id == r.target), None
                )
                if not source_name or not target_name:
                    continue
                canonical_source = global_name_to_id.get(source_name)
                canonical_target = global_name_to_id.get(target_name)
                if canonical_source and canonical_target:
                    all_relations.append(Relation(
                        source=canonical_source,
                        target=canonical_target,
                        relation=r.relation,
                        properties=r.properties,
                    ))

        self.usage.report(self.model)
        print(
            f"✓ EntityExtractor: {len(chunks)} chunks → "
            f"{len(all_entities)} entities, {len(all_relations)} relations"
        )
        return all_entities, all_relations

    # ── Private ────────────────────────────────────────────────────────────────

    async def _call_llm(self, content: str) -> ExtractionResult:
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": content},
            ],
            temperature=0,
            response_format=ExtractionResult,
        )
        self.usage.add(response.usage)
        return response.choices[0].message.parsed

    @staticmethod
    def _build(
        result:   ExtractionResult,
        chunk_id: str,
    ) -> tuple[list[Entity], list[Relation]]:
        """Build Entity/Relation objects and validate type constraints."""
        entities:        list[Entity]   = []
        name_to_id:      dict[str, str] = {}
        entity_type_map: dict[str, str] = {}

        for e in result.entities:
            eid = f"{chunk_id}__{re.sub(r'\W+', '_', e.name)}__{e.type}"
            entities.append(Entity(
                id=eid,
                name=e.name,
                type=e.type,
                chunk_id=chunk_id,
                properties=e.properties or {},
            ))
            name_to_id[e.name]   = eid
            entity_type_map[eid] = e.type

        raw_relations: list[Relation] = []
        for r in result.relations:
            if r.source not in name_to_id or r.target not in name_to_id:
                continue
            raw_relations.append(Relation(
                source=name_to_id[r.source],
                target=name_to_id[r.target],
                relation=r.relation,
                properties=r.properties or {},
            ))

        relations = EntityExtractor._validate_relations(raw_relations, entity_type_map)
        return entities, relations

    @staticmethod
    def _validate_relations(
        relations:       list[Relation],
        entity_type_map: dict[str, str],
    ) -> list[Relation]:
        """Drop relations where source/target types don't match allowed pairs."""
        valid = []
        for r in relations:
            expected    = _ALLOWED_TYPES.get(r.relation)
            source_type = entity_type_map.get(r.source)
            target_type = entity_type_map.get(r.target)
            if (source_type, target_type) == expected:
                valid.append(r)
            else:
                print(
                    f"  ✗ dropped: {r.relation} "
                    f"({source_type} → {target_type}, expected {expected})"
                )
        return valid


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    from pathlib import Path
    from pipeline.preprocessing.loader import load_documents
    from pipeline.preprocessing.normalizer import normalize
    from pipeline.preprocessing.chunker import HeadingChunker
    from dotenv import load_dotenv
    import random

    load_dotenv(override=True)

    docs = load_documents(Path("data"))
    for doc in docs:
        doc.content = normalize(doc.content)
    chunks = HeadingChunker(min_chars=100, max_chars=2000).chunk_all(docs)

    chunks_sample = random.sample(chunks, 3)
    extractor = EntityExtractor()
    entities, relations = asyncio.run(extractor.extract_all(chunks_sample))

    print(f"\nEntities ({len(entities)}):")
    for e in entities:
        print(f"  {e}")

    print(f"\nRelations ({len(relations)}):")
    for r in relations:
        print(f"  {r}")