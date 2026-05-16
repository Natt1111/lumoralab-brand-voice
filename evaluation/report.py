"""
Report generator: turns a list of EvalResult objects into a Markdown or HTML report.
"""
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.runner import EvalResult

REPORTS_DIR = Path(__file__).parent / "reports"


def _score_bar(score: float, width: int = 10) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def save_markdown_report(
    results: list[EvalResult],
    path: Optional[Path] = None,
) -> Path:
    """Generate a Markdown evaluation report and write it to disk."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"eval_{stamp}.md"

    valid  = [r for r in results if not r.error]
    errors = [r for r in results if r.error]

    avg_voice  = sum(r.voice_score for r in valid) / len(valid) if valid else 0
    avg_ret    = sum(r.retrieval_precision for r in valid) / len(valid) if valid else 0
    avg_faith  = sum(r.faithfulness_score for r in valid) / len(valid) if valid else 0
    n_passed   = sum(1 for r in valid if r.passed)

    lines: list[str] = [
        "# LumoraLab RAG — Evaluation Report",
        f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
        "---",
        "## Summary\n",
        f"| Metric | Score |",
        f"|--------|-------|",
        f"| Cases run | {len(results)} ({len(errors)} errors) |",
        f"| Pass rate | {n_passed}/{len(valid)} ({n_passed/len(valid):.0%} if valid else 'N/A') |",
        f"| Avg voice consistency | {avg_voice*10:.1f}/10 {_score_bar(avg_voice)} |",
        f"| Avg retrieval precision | {avg_ret:.0%} {_score_bar(avg_ret)} |",
        f"| Avg faithfulness | {avg_faith*10:.1f}/10 {_score_bar(avg_faith)} |",
        "\n---",
        "## Per-Case Results\n",
        "| ID | Content Type | Voice | Retrieval | Faithful | Pass | Time |",
        "|----|-------------|-------|-----------|----------|------|------|",
    ]

    for r in results:
        status = "✓" if r.passed else ("⚠ ERR" if r.error else "✗")
        if r.error:
            lines.append(
                f"| {r.test_id} | {r.content_type} | ERR | ERR | ERR | {status} | {r.duration_seconds}s |"
            )
        else:
            lines.append(
                f"| {r.test_id} | {r.content_type} "
                f"| {r.voice.raw_score:.0f}/10 "
                f"| {r.retrieval.score:.0%} "
                f"| {r.faithfulness.raw_score:.0f}/10 "
                f"| {status} | {r.duration_seconds}s |"
            )

    lines += ["\n---", "## Detail\n"]
    for r in results:
        lines += [
            f"### {r.test_id} — `{r.content_type}`",
            f"\n**Query:** {r.query}\n",
        ]
        if r.error:
            lines.append(f"> **ERROR:** {r.error}\n")
            continue
        lines += [
            f"- Voice consistency: **{r.voice.raw_score:.0f}/10** — {r.voice.reason}",
            f"- Retrieval precision: **{r.retrieval.score:.0%}** — {r.retrieval.reason}",
            f"- Faithfulness: **{r.faithfulness.raw_score:.0f}/10** — {r.faithfulness.reason}",
            f"\n**Generated content:**\n```\n{r.generated_content[:600]}{'...' if len(r.generated_content) > 600 else ''}\n```\n",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_html_report(
    results: list[EvalResult],
    path: Optional[Path] = None,
) -> Path:
    """Generate a minimal dark-theme HTML report."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"eval_{stamp}.html"

    valid   = [r for r in results if not r.error]
    n_pass  = sum(1 for r in valid if r.passed)
    avg_v   = sum(r.voice_score for r in valid) / len(valid) if valid else 0
    avg_r   = sum(r.retrieval_precision for r in valid) / len(valid) if valid else 0
    avg_f   = sum(r.faithfulness_score for r in valid) / len(valid) if valid else 0

    def pct(v: float) -> str:
        return f"{v*100:.0f}%"

    rows = ""
    for r in results:
        ok = "✓" if r.passed else "✗"
        color = "#10B981" if r.passed else "#E24B4A"
        if r.error:
            rows += f"<tr><td>{r.test_id}</td><td>{r.content_type}</td><td colspan=3 style='color:#E24B4A'>ERROR: {r.error}</td><td style='color:#E24B4A'>✗</td></tr>"
        else:
            rows += (
                f"<tr>"
                f"<td>{r.test_id}</td>"
                f"<td>{r.content_type}</td>"
                f"<td>{r.voice.raw_score:.0f}/10</td>"
                f"<td>{pct(r.retrieval.score)}</td>"
                f"<td>{r.faithfulness.raw_score:.0f}/10</td>"
                f"<td style='color:{color}'>{ok}</td>"
                f"</tr>"
            )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<title>LumoraLab Eval Report</title>
<style>
body{{background:#0A0A0A;color:#FAFAFA;font-family:system-ui,sans-serif;padding:40px;max-width:900px;margin:auto}}
h1{{font-size:22px;font-weight:500;margin-bottom:4px}}
p.sub{{color:#888;font-size:13px;margin-bottom:32px}}
.stat{{display:inline-block;background:#141414;border:1px solid #2A2A2A;border-radius:8px;padding:16px 24px;margin:0 12px 12px 0;min-width:140px}}
.stat-label{{font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:#666;margin-bottom:6px}}
.stat-value{{font-size:24px;font-weight:500}}
table{{width:100%;border-collapse:collapse;margin-top:32px;font-size:13px}}
th{{text-align:left;color:#666;font-size:10px;letter-spacing:.08em;text-transform:uppercase;padding:8px 12px;border-bottom:1px solid #2A2A2A}}
td{{padding:10px 12px;border-bottom:1px solid #1A1A1A;vertical-align:top}}
tr:hover td{{background:#111}}
</style></head><body>
<h1>LumoraLab RAG — Evaluation Report</h1>
<p class="sub">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<div>
  <div class="stat"><div class="stat-label">Pass rate</div><div class="stat-value">{n_pass}/{len(valid)}</div></div>
  <div class="stat"><div class="stat-label">Avg voice</div><div class="stat-value">{avg_v*10:.1f}/10</div></div>
  <div class="stat"><div class="stat-label">Avg retrieval</div><div class="stat-value">{pct(avg_r)}</div></div>
  <div class="stat"><div class="stat-label">Avg faithful</div><div class="stat-value">{avg_f*10:.1f}/10</div></div>
</div>
<table>
<tr><th>ID</th><th>Content type</th><th>Voice</th><th>Retrieval</th><th>Faithful</th><th>Pass</th></tr>
{rows}
</table></body></html>"""

    path.write_text(html, encoding="utf-8")
    return path
