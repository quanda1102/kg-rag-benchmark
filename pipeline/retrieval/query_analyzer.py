"""
pipeline/retrieval/query_analyzer.py
──────────────────────────────────────
Analyze and expand queries before retrieval.

Responsibilities:
  - Detect query intent (lookup / filter / comparison / multi_hop)
  - Expand query with Vietnamese equivalents for BM25
  - Extract entity mentions for graph traversal
  - Provide cypher hints for filter queries
  - Decide which retrieval sources to activate

Usage:
    from pipeline.retrieval.query_analyzer import QueryAnalyzer

    analyzer = QueryAnalyzer()
    analysis = analyzer.analyze("incident nào severity critical và status open?")
    print(analysis.expanded_query_vi)
    print(analysis.cypher_hints)
"""

from __future__ import annotations

from typing import Literal, Optional

from openai import OpenAI
from pydantic import BaseModel


# ── Schema ─────────────────────────────────────────────────────────────────────

QueryIntent = Literal["lookup", "filter", "comparison", "multi_hop"]


class QueryAnalysis(BaseModel):
    intent:             QueryIntent
    expanded_query_vi:  str         # Vietnamese expanded query for BM25
    expanded_query_en:  str         # English expanded query for BM25
    entities_mentioned: list[str]   # entity names extracted from query
    cypher_hints:       Optional[dict[str, str]] = None  # {"severity": "Nghiêm trọng", "status": "Mở"}
    use_graph:          bool        # should graph traversal be activated?


# ── Prompt ─────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
Bạn là hệ thống phân tích câu hỏi cho GraphRAG pipeline về công ty TechViet Solutions.

Nhiệm vụ: phân tích câu hỏi và trả về JSON với các fields sau.

━━━ INTENT ━━━
- lookup:     tìm thông tin về một entity cụ thể (người, dự án, phòng ban)
- filter:     lọc theo thuộc tính (severity, status, số lượng, ngày tháng)
- comparison: so sánh nhiều entities
- multi_hop:  cần traverse nhiều quan hệ (ai quản lý người làm dự án X?)

━━━ EXPANDED QUERIES ━━━
Mở rộng query với các từ đồng nghĩa và bản dịch để tăng BM25 recall.

━━━ CYPHER HINTS ━━━
Với filter queries, map các giá trị Anh → Việt như đang dùng trong data:

Severity mapping:
  critical/nghiêm trọng → "Nghiêm trọng"
  high/cao              → "Cao"
  medium/trung bình     → "Trung bình"
  low/thấp              → "Thấp"

Status mapping (incident):
  open/mở/chưa xử lý   → "Mở"
  resolved/đã xử lý    → "Đã xử lý"
  investigating         → "Đang điều tra"

Status mapping (project):
  active/đang triển khai → "Đang triển khai"
  paused/tạm dừng        → "Tạm dừng"
  planned/lên kế hoạch   → "Lên kế hoạch"
  completed/hoàn thành   → "Hoàn thành"

━━━ USE_GRAPH ━━━
True nếu query cần traverse relations (quản lý, làm việc, thuộc phòng ban).
False nếu query chỉ cần text search (tìm thông tin về 1 entity).

━━━ VÍ DỤ ━━━

Query: "incident nào severity critical và status open?"
Output:
{
  "intent": "filter",
  "expanded_query_vi": "sự cố mức độ nghiêm trọng đang mở chưa xử lý incident",
  "expanded_query_en": "incident severity critical status open unresolved",
  "entities_mentioned": [],
  "cypher_hints": {"severity": "Nghiêm trọng", "status": "Mở"},
  "use_graph": false
}

Query: "ai đang làm Legacy CRM Sunset?"
Output:
{
  "intent": "lookup",
  "expanded_query_vi": "nhân viên thành viên làm việc dự án Legacy CRM Sunset",
  "expanded_query_en": "who is working on Legacy CRM Sunset project members",
  "entities_mentioned": ["Legacy CRM Sunset"],
  "cypher_hints": {},
  "use_graph": true
}

Query: "Lê Quang Huy quản lý ai?"
Output:
{
  "intent": "multi_hop",
  "expanded_query_vi": "Lê Quang Huy quản lý cấp dưới báo cáo trực tiếp",
  "expanded_query_en": "Le Quang Huy manages direct reports subordinates",
  "entities_mentioned": ["Lê Quang Huy"],
  "cypher_hints": {},
  "use_graph": true
}

Query: "Phòng Engineering có bao nhiêu nhân viên?"
Output:
{
  "intent": "filter",
  "expanded_query_vi": "phòng Engineering số lượng nhân viên thành viên",
  "expanded_query_en": "Engineering department headcount employees members",
  "entities_mentioned": ["Engineering"],
  "cypher_hints": {},
  "use_graph": false
}
""".strip()


# ── QueryAnalyzer ──────────────────────────────────────────────────────────────

class QueryAnalyzer:
    """
    Analyze and expand queries before retrieval.

    Args:
        model: OpenAI model. Default: gpt-5.4-mini — cheap, fast, sufficient.
    """

    def __init__(self, model: str = "gpt-5.4-nano") -> None:
        self.model  = model
        self.client = OpenAI()

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze query and return structured QueryAnalysis.
        Uses structured output to guarantee schema.
        """
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": query},
            ],
            temperature=0,
            response_format=QueryAnalysis,
        )
        return response.choices[0].message.parsed


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)

    analyzer = QueryAnalyzer()

    queries = [
        "incident nào vừa có severity critical vừa đang ở trạng thái open?",
        "ai đang làm Legacy CRM Sunset?",
        "Lê Quang Huy quản lý ai?",
        "Phòng Engineering có bao nhiêu nhân viên?",
        "project nào đang paused?",
        "ai có skill Kubernetes trong phòng DevOps?",
    ]

    for q in queries:
        print(f"\nQuery: {q!r}")
        analysis = analyzer.analyze(q)
        print(f"  intent:            {analysis.intent}")
        print(f"  expanded_vi:       {analysis.expanded_query_vi}")
        print(f"  expanded_en:       {analysis.expanded_query_en}")
        print(f"  entities:          {analysis.entities_mentioned}")
        print(f"  cypher_hints:      {analysis.cypher_hints}" or {})
        print(f"  use_graph:         {analysis.use_graph}")