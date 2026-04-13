"""
eval/report.py
───────────────
Generate HTML evaluation report from results JSON.

Usage:
    uv run -m eval.report --results results/baseline_20260413_1640.json
    uv run -m eval.report --results results/baseline_20260413_1640.json --open
"""

from __future__ import annotations

import json
import sys
import webbrowser
from datetime import datetime
from pathlib import Path


# ── HTML Template ──────────────────────────────────────────────────────────────

def generate_report(results_path: str, auto_open: bool = False) -> str:
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    variants  = data["variants"]
    questions = data["questions"]
    metrics   = data["metrics"]
    timestamp = data.get("timestamp", "unknown")
    total_generation_cost = sum(
        metrics.get(v, {}).get("total_generation_cost_usd", 0.0)
        for v in variants
    )
    total_judge_cost = sum(
        metrics.get(v, {}).get("total_judge_cost_usd", 0.0)
        for v in variants
    )
    total_eval_cost = total_generation_cost + total_judge_cost
    total_questions = len(questions)
    total_variant_runs = sum(
        len(q.get("variants", {}))
        for q in questions
    )

    # ── Prepare chart data ──
    variant_labels = json.dumps(variants)

    accuracy_data = json.dumps([
        round(metrics[v]["accuracy"] * 100, 1) for v in variants
    ])
    partial_data = json.dumps([
        round(metrics[v]["partial_credit"] * 100, 1) for v in variants
    ])
    recall_data = json.dumps([
        round(metrics[v]["recall_at_5"] * 100, 1) for v in variants
    ])
    latency_data = json.dumps([
        round(metrics[v]["avg_latency_ms"]) for v in variants
    ])

    # ── Type breakdown ──
    all_types = sorted(set(
        t for v in variants
        for t in metrics[v].get("by_type", {}).keys()
    ))

    type_accuracy_datasets = []
    colors = ["#00ff9d", "#00b8ff", "#ff6b6b", "#ffd93d"]
    for i, v in enumerate(variants):
        by_type = metrics[v].get("by_type", {})
        type_accuracies = []
        for t in all_types:
            td = by_type.get(t, {})
            total = td.get("total", 0)
            correct = td.get("correct", 0)
            acc = round(correct / total * 100, 1) if total > 0 else 0
            type_accuracies.append(acc)
        type_accuracy_datasets.append({
            "label": v,
            "data": type_accuracies,
            "backgroundColor": colors[i % len(colors)] + "33",
            "borderColor": colors[i % len(colors)],
            "borderWidth": 2,
        })

    type_labels_js   = json.dumps(all_types)
    type_datasets_js = json.dumps(type_accuracy_datasets)

    # ── Per-question table rows ──
    verdict_badge = {
        "CORRECT":   '<span class="badge correct">CORRECT</span>',
        "PARTIAL":   '<span class="badge partial">PARTIAL</span>',
        "INCORRECT": '<span class="badge incorrect">INCORRECT</span>',
    }

    rows = []
    for q in questions:
        qtype = q["type"].replace("_", " ").upper()
        row_variants = ""
        for v in variants:
            vd = q["variants"].get(v, {})
            verdict  = vd.get("verdict", "—")
            latency  = vd.get("latency_ms", 0)
            reasoning = vd.get("reasoning", "")
            badge    = verdict_badge.get(verdict, verdict)
            row_variants += f"""
                <td>
                    <div class="result-cell">
                        <div class="result-top">
                            {badge}
                            <span class="latency-pill">{latency:.0f}ms</span>
                        </div>
                        <div class="reasoning">{reasoning}</div>
                    </div>
                </td>
            """
        rows.append(f"""
            <tr>
                <td class="q-id">{q["id"]}</td>
                <td class="q-type"><span class="type-tag">{qtype}</span></td>
                <td class="q-text">{q["question"]}</td>
                {row_variants}
            </tr>
        """)

    rows_html = "\n".join(rows)

    # ── Variant header cols ──
    variant_headers = "".join(f"<th>{v}</th>" for v in variants)

    # ── Summary cards ──
    summary_cards = ""
    for v in variants:
        m = metrics[v]
        acc = round(m["accuracy"] * 100, 1)
        pc  = round(m["partial_credit"] * 100, 1)
        lat = round(m["avg_latency_ms"])
        r5  = round(m["recall_at_5"] * 100, 1)
        acc_hit = round(m.get("answer_accuracy_when_recalled", 0) * 100, 1)
        fail_hit = round(m.get("generation_failure_rate_when_recalled", 0) * 100, 1)
        generation_cost = m.get("total_generation_cost_usd", 0.0)
        judge_cost = m.get("total_judge_cost_usd", 0.0)
        total_cost = m.get("total_eval_cost_usd", generation_cost + judge_cost)
        summary_cards += f"""
        <div class="summary-card">
            <div class="card-variant">{v}</div>
            <div class="card-group-title">End-to-End</div>
            <div class="card-metrics">
                <div class="card-metric metric-primary">
                    <span class="metric-val">{acc}%</span>
                    <span class="metric-label">Answer Accuracy</span>
                </div>
                <div class="card-metric">
                    <span class="metric-val">{pc}%</span>
                    <span class="metric-label">Partial Credit</span>
                </div>
                <div class="card-metric">
                    <span class="metric-val">{lat}ms</span>
                    <span class="metric-label">Avg Latency</span>
                </div>
                <div class="card-metric">
                    <span class="metric-val">${generation_cost:.4f}</span>
                    <span class="metric-label">Generation Cost</span>
                </div>
            </div>
            <div class="card-divider"></div>
            <div class="card-group-title">Retrieval Diagnostics & Eval Cost</div>
            <div class="card-metrics">
                <div class="card-metric">
                    <span class="metric-val">{r5}%</span>
                    <span class="metric-label">Recall@5</span>
                </div>
                <div class="card-metric">
                    <span class="metric-val">{acc_hit}%</span>
                    <span class="metric-label">Acc | Recall Hit</span>
                </div>
                <div class="card-metric">
                    <span class="metric-val">{fail_hit}%</span>
                    <span class="metric-label">Gen Fail | Hit</span>
                </div>
                <div class="card-metric">
                    <span class="metric-val">${judge_cost:.4f}</span>
                    <span class="metric-label">Judge Cost</span>
                </div>
                <div class="card-metric metric-primary">
                    <span class="metric-val">${total_cost:.4f}</span>
                    <span class="metric-label">Total Eval Cost</span>
                </div>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GraphRAG Eval — {timestamp}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;700;800&display=swap');

  :root {{
    --bg:       #0a0a0f;
    --surface:  #111118;
    --border:   #1e1e2e;
    --accent1:  #00ff9d;
    --accent2:  #00b8ff;
    --accent3:  #ff6b6b;
    --accent4:  #ffd93d;
    --text:     #e0e0f0;
    --muted:    #6060a0;
    --muted-strong: #9ea4d1;
    --correct:  #00ff9d;
    --partial:  #ffd93d;
    --incorrect:#ff6b6b;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.6;
    min-height: 100vh;
  }}

  /* ── Header ── */
  .header {{
    border-bottom: 1px solid var(--border);
    padding: 2rem 3rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    background: var(--bg);
    z-index: 100;
    backdrop-filter: blur(10px);
  }}

  .header-left {{
    display: flex;
    align-items: center;
    gap: 1.5rem;
  }}

  .logo {{
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent1);
  }}

  .header-divider {{
    width: 1px;
    height: 24px;
    background: var(--border);
  }}

  .header-title {{
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--muted);
    letter-spacing: 0.05em;
  }}

  .header-meta {{
    display: flex;
    gap: 2rem;
    align-items: center;
  }}

  .meta-item {{
    display: flex;
    flex-direction: column;
    align-items: flex-end;
  }}

  .meta-val {{
    font-size: 0.85rem;
    color: var(--text);
    font-weight: 600;
  }}

  .meta-label {{
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }}

  .meta-val.cost {{
    color: var(--accent4);
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 800;
  }}

  /* ── Main ── */
  .main {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem 3rem 4rem;
  }}

  /* ── Section ── */
  .section {{
    margin-bottom: 3rem;
  }}

  .section-title {{
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }}

  .section-title::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }}

  /* ── Summary cards ── */
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1rem;
  }}

  .summary-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
  }}

  .summary-card:hover {{ border-color: var(--accent1); }}

  .summary-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent1), var(--accent2));
  }}

  .card-variant {{
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--accent1);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
  }}

  .card-group-title {{
    font-size: 0.62rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 0.75rem;
  }}

  .card-divider {{
    height: 1px;
    background: var(--border);
    margin: 1rem 0 1rem;
  }}

  .card-metrics {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
  }}

  @media (max-width: 640px) {{
    .card-metrics {{ grid-template-columns: 1fr 1fr; }}
  }}

  .card-metric {{
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    padding: 0.7rem 0.8rem;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: rgba(255,255,255,0.02);
  }}

  .metric-primary {{
    background: linear-gradient(180deg, rgba(0,255,157,0.08), rgba(255,255,255,0.02));
  }}

  .metric-val {{
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--text);
    line-height: 1;
  }}

  .metric-label {{
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    line-height: 1.35;
  }}

  /* ── Charts ── */
  .charts-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
  }}

  @media (max-width: 900px) {{
    .charts-grid {{ grid-template-columns: 1fr; }}
  }}

  .chart-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
  }}

  .chart-label {{
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--muted-strong);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 1rem;
  }}

  .chart-wrap {{
    position: relative;
    height: 220px;
  }}

  /* ── Table ── */
  .table-wrap {{
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--surface);
  }}

  thead {{
    position: sticky;
    top: 0;
    background: var(--surface);
    z-index: 10;
  }}

  th {{
    padding: 0.9rem 1rem;
    text-align: left;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}

  td {{
    padding: 1rem 1rem;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }}

  tr:last-child td {{ border-bottom: none; }}

  tr:hover td {{ background: rgba(255,255,255,0.02); }}

  .q-id {{
    font-weight: 700;
    color: var(--accent2);
    white-space: nowrap;
    font-size: 0.75rem;
    position: sticky;
    left: 0;
    background: var(--surface);
    z-index: 2;
  }}

  .q-type {{
    white-space: nowrap;
    position: sticky;
    left: 71px;
    background: var(--surface);
    z-index: 2;
  }}

  .type-tag {{
    display: inline-block;
    padding: 0.2em 0.5em;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: rgba(0,184,255,0.1);
    color: var(--accent2);
    border: 1px solid rgba(0,184,255,0.2);
    white-space: nowrap;
  }}

  .q-text {{
    min-width: 320px;
    max-width: 420px;
    color: var(--text);
    font-size: 0.82rem;
    line-height: 1.55;
  }}

  .badge {{
    display: inline-block;
    padding: 0.2em 0.6em;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    white-space: nowrap;
  }}

  .badge.correct {{
    background: rgba(0,255,157,0.1);
    color: var(--correct);
    border: 1px solid rgba(0,255,157,0.3);
  }}

  .badge.partial {{
    background: rgba(255,217,61,0.1);
    color: var(--partial);
    border: 1px solid rgba(255,217,61,0.3);
  }}

  .badge.incorrect {{
    background: rgba(255,107,107,0.1);
    color: var(--incorrect);
    border: 1px solid rgba(255,107,107,0.3);
  }}

  .reasoning {{
    font-size: 0.72rem;
    color: #a8a8cc;
    margin-top: 0.55rem;
    max-width: 280px;
    line-height: 1.4;
  }}

  .result-cell {{
    min-width: 210px;
  }}

  .result-top {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
  }}

  .latency-pill {{
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.03);
    color: var(--muted);
    font-size: 0.65rem;
    line-height: 1;
  }}

  .notes {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 1rem;
  }}

  .note-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
  }}

  .note-title {{
    font-size: 0.68rem;
    color: var(--accent2);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.45rem;
    font-weight: 700;
  }}

  .note-body {{
    color: #a8a8cc;
    font-size: 0.74rem;
    line-height: 1.55;
  }}

  /* ── Footer ── */
  .footer {{
    border-top: 1px solid var(--border);
    padding: 1.5rem 3rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: var(--muted);
    font-size: 0.7rem;
  }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: var(--bg); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: var(--muted); }}
</style>
</head>
<body>

<header class="header">
  <div class="header-left">
    <div class="logo">GraphRAG</div>
    <div class="header-divider"></div>
    <div class="header-title">Evaluation Report</div>
  </div>
  <div class="header-meta">
    <div class="meta-item">
      <span class="meta-val">{len(questions)}</span>
      <span class="meta-label">Questions</span>
    </div>
    <div class="meta-item">
      <span class="meta-val">{total_variant_runs}</span>
      <span class="meta-label">Variant Runs</span>
    </div>
    <div class="meta-item">
      <span class="meta-val">{timestamp}</span>
      <span class="meta-label">Run ID</span>
    </div>
    <div class="meta-item">
      <span class="meta-val cost">${total_generation_cost:.4f}</span>
      <span class="meta-label">Run Gen Cost</span>
    </div>
    <div class="meta-item">
      <span class="meta-val cost">${total_judge_cost:.4f}</span>
      <span class="meta-label">Run Judge Cost</span>
    </div>
    <div class="meta-item">
      <span class="meta-val cost">${total_eval_cost:.4f}</span>
      <span class="meta-label">Run Total Cost</span>
    </div>
  </div>
</header>

<main class="main">

  <!-- Summary -->
  <section class="section">
    <div class="section-title">Summary</div>
    <div class="summary-grid">
      {summary_cards}
    </div>
  </section>

  <section class="section">
    <div class="section-title">How To Read</div>
    <div class="notes">
      <div class="note-card">
        <div class="note-title">End-to-End</div>
        <div class="note-body">
          <strong>Answer Accuracy</strong> và <strong>Partial Credit</strong> chấm câu trả lời cuối cùng của hệ thống. Đây là metric chính để so variant.
        </div>
      </div>
      <div class="note-card">
        <div class="note-title">Retrieval</div>
        <div class="note-body">
          <strong>Recall@5</strong> đo top-5 retrieved chunks có chứa đúng entity hay không, độc lập với việc model trả lời đúng hay sai.
        </div>
      </div>
      <div class="note-card">
        <div class="note-title">Post-Retrieval Failure</div>
        <div class="note-body">
          <strong>Acc | Recall Hit</strong> cho biết khi retrieval đã hit thì answer đúng bao nhiêu. <strong>Gen Fail | Hit</strong> cho biết retrieval đúng nhưng generation vẫn làm sai bao nhiêu.
        </div>
      </div>
    </div>
  </section>

  <!-- Charts -->
  <section class="section">
    <div class="section-title">Metrics</div>
    <div class="charts-grid">

      <div class="chart-card">
        <div class="chart-label">Accuracy by Variant</div>
        <div class="chart-wrap">
          <canvas id="accuracyChart"></canvas>
        </div>
      </div>

      <div class="chart-card">
        <div class="chart-label">Avg Latency (ms)</div>
        <div class="chart-wrap">
          <canvas id="latencyChart"></canvas>
        </div>
      </div>

      <div class="chart-card">
        <div class="chart-label">Accuracy by Question Type</div>
        <div class="chart-wrap">
          <canvas id="typeChart"></canvas>
        </div>
      </div>

      <div class="chart-card">
        <div class="chart-label">Recall@5 by Variant</div>
        <div class="chart-wrap">
          <canvas id="recallChart"></canvas>
        </div>
      </div>

    </div>
  </section>

  <!-- Per-question table -->
  <section class="section">
    <div class="section-title">Per-Question Results</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Type</th>
            <th>Question</th>
            {variant_headers}
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
  </section>

</main>

<footer class="footer">
  <span>GraphRAG Evaluation Pipeline</span>
  <span>Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</span>
</footer>

<script>
const VARIANTS = {variant_labels};
const COLORS   = ['#00ff9d', '#00b8ff', '#ff6b6b', '#ffd93d'];

const baseOpts = {{
  responsive: true,
  maintainAspectRatio: false,
  plugins: {{
    legend: {{
      labels: {{
        color: '#c4caec',
        font: {{ family: 'JetBrains Mono', size: 11, weight: '600' }},
        boxWidth: 12,
        padding: 18,
      }}
    }},
  }},
  scales: {{
    x: {{
      ticks: {{
        color: '#c4caec',
        font: {{ family: 'JetBrains Mono', size: 10, weight: '600' }},
        padding: 8,
      }},
      grid:  {{ color: 'rgba(82, 88, 136, 0.18)' }},
      border: {{ color: 'rgba(120, 128, 184, 0.28)' }},
    }},
    y: {{
      ticks: {{
        color: '#c4caec',
        font: {{ family: 'JetBrains Mono', size: 10, weight: '600' }},
        padding: 8,
      }},
      grid:  {{ color: 'rgba(82, 88, 136, 0.18)' }},
      border: {{ color: 'rgba(120, 128, 184, 0.28)' }},
    }},
  }},
}};

// Accuracy bar
new Chart(document.getElementById('accuracyChart'), {{
  type: 'bar',
  data: {{
    labels: VARIANTS,
    datasets: [{{
      label: 'Accuracy %',
      data: {accuracy_data},
      backgroundColor: COLORS.map(c => c + '33'),
      borderColor: COLORS,
      borderWidth: 2,
      borderRadius: 4,
    }}, {{
      label: 'Partial Credit %',
      data: {partial_data},
      backgroundColor: 'transparent',
      borderColor: COLORS.map(c => c + '88'),
      borderWidth: 1,
      borderDash: [4, 4],
      type: 'line',
      tension: 0.3,
      pointRadius: 4,
    }}]
  }},
  options: {{ ...baseOpts, scales: {{ ...baseOpts.scales, y: {{ ...baseOpts.scales.y, min: 0, max: 100 }} }} }},
}});

// Latency bar
new Chart(document.getElementById('latencyChart'), {{
  type: 'bar',
  data: {{
    labels: VARIANTS,
    datasets: [{{
      label: 'Avg Latency (ms)',
      data: {latency_data},
      backgroundColor: ['#00ff9d33', '#00b8ff33', '#ff6b6b33'],
      borderColor: ['#00ff9d', '#00b8ff', '#ff6b6b'],
      borderWidth: 2,
      borderRadius: 4,
    }}]
  }},
  options: baseOpts,
}});

// Type radar
new Chart(document.getElementById('typeChart'), {{
  type: 'radar',
  data: {{
    labels: {type_labels_js},
    datasets: {type_datasets_js},
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{
        labels: {{
          color: '#c4caec',
          font: {{ family: 'JetBrains Mono', size: 11, weight: '600' }},
          boxWidth: 12,
        }}
      }}
    }},
    scales: {{
      r: {{
        min: 0, max: 100,
        ticks: {{
          color: '#c4caec',
          backdropColor: 'transparent',
          font: {{ family: 'JetBrains Mono', size: 9, weight: '600' }},
          stepSize: 25
        }},
        angleLines: {{ color: 'rgba(82, 88, 136, 0.18)' }},
        grid:  {{ color: 'rgba(82, 88, 136, 0.18)' }},
        pointLabels: {{
          color: '#c4caec',
          font: {{ family: 'JetBrains Mono', size: 10, weight: '600' }}
        }},
      }}
    }},
  }},
}});

// Recall bar
new Chart(document.getElementById('recallChart'), {{
  type: 'bar',
  data: {{
    labels: VARIANTS,
    datasets: [{{
      label: 'Recall@5 %',
      data: {recall_data},
      backgroundColor: ['#00ff9d33', '#00b8ff33', '#ff6b6b33'],
      borderColor: ['#00ff9d', '#00b8ff', '#ff6b6b'],
      borderWidth: 2,
      borderRadius: 4,
    }}]
  }},
  options: {{ ...baseOpts, scales: {{ ...baseOpts.scales, y: {{ ...baseOpts.scales.y, min: 0, max: 100 }} }} }},
}});
</script>
</body>
</html>"""

    # Save report
    results_p   = Path(results_path)
    report_path = results_p.parent / results_p.name.replace(".json", "_report.html")
    report_path.write_text(html, encoding="utf-8")

    print(f"✓ Report saved to {report_path}")

    if auto_open:
        webbrowser.open(f"file://{report_path.absolute()}")

    return str(report_path)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to results JSON")
    parser.add_argument("--open",    action="store_true", help="Auto-open in browser")
    args = parser.parse_args()

    generate_report(args.results, auto_open=args.open)
