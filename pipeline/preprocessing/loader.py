"""
pipeline/preprocessing/loader.py
──────────────────────────────────
Load raw data from disk.

Two sources:
  - data/json/company.json     → ground truth entities + relationships
  - data/plaintext/*.txt       → 93 plain text documents for RAG input

Usage:
    from pipeline.preprocessing.loader import load_documents, load_ground_truth

    docs = load_documents("data/plaintext")
    gt   = load_ground_truth("data/json/company.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    id: str           # filename stem, e.g. "emp_001_le_quang_huy"
    content: str
    source_path: str


@dataclass
class GroundTruth:
    departments: list[dict]
    employees: list[dict]
    skills: list[dict]
    employee_skills: list[dict]
    projects: list[dict]
    project_members: list[dict]
    incidents: list[dict]


def load_documents(plaintext_dir: str | Path) -> list[Document]:
    """Load all .txt / .md files from plaintext directory."""
    plaintext_dir = Path(plaintext_dir)
    if not plaintext_dir.exists():
        raise FileNotFoundError(f"Plaintext directory not found: {plaintext_dir}")

    docs = []
    for path in sorted(plaintext_dir.glob("*.txt")) + sorted(plaintext_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if content:
            docs.append(Document(
                id=path.stem,
                content=content,
                source_path=str(path),
            ))

    print(f"✓ Loaded {len(docs)} documents from {plaintext_dir}")
    return docs


def load_ground_truth(json_path: str | Path) -> GroundTruth:
    """Load structured ground truth from JSON."""
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON ground truth not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    gt = GroundTruth(
        departments=data.get("departments", []),
        employees=data.get("employees", []),
        skills=data.get("skills", []),
        employee_skills=data.get("employee_skills", []),
        projects=data.get("projects", []),
        project_members=data.get("project_members", []),
        incidents=data.get("incidents", []),
    )

    print(
        f"✓ Loaded ground truth: "
        f"{len(gt.employees)} employees, "
        f"{len(gt.projects)} projects, "
        f"{len(gt.incidents)} incidents"
    )
    return gt


def load_questions(questions_path: str | Path) -> list[dict]:
    """Load benchmark questions with ground truth answers."""
    questions_path = Path(questions_path)
    with open(questions_path, encoding="utf-8") as f:
        data = json.load(f)

    questions = data if isinstance(data, list) else data.get("questions", [])
    print(f"✓ Loaded {len(questions)} benchmark questions")
    return questions

if __name__ == "__main__":
    plaintext_dir = Path("data")
    json_path = Path("data/company.json")
    question_path = Path("eval/questions.json")

    docs = load_documents(plaintext_dir)
    print(docs)
    load_ground_truth(json_path)
    load_questions(question_path)