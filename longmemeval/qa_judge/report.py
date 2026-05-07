from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _acc(rows: List[Tuple[str, str]]) -> Dict[str, Any]:
    total = len(rows)
    correct = sum(1 for _, label in rows if str(label).upper() == "CORRECT")
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total * 100.0 if total else 0.0),
    }


def generate_judge_report(judge_root: Path) -> Dict[str, Any]:
    root = Path(judge_root)
    all_rows: List[Tuple[str, str]] = []
    by_qtype: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    by_method: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for summary in sorted(root.glob("*/*/*/summary.json")):
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
        except Exception:
            continue
        qtype = str(data.get("question_type", "")).strip() or "unknown"
        method = str(data.get("retrieval_method", "")).strip() or "unknown"
        label = str(data.get("judge_label", "")).strip().upper()
        row = (qtype, label)
        all_rows.append(row)
        by_qtype[qtype].append(row)
        by_method[method].append((method, label))

    overall = _acc(all_rows)
    by_question_type: Dict[str, Dict[str, Any]] = {}
    for qtype in sorted(by_qtype):
        by_question_type[qtype] = _acc(by_qtype[qtype])
    by_retrieval_method: Dict[str, Dict[str, Any]] = {}
    for method in sorted(by_method):
        by_retrieval_method[method] = _acc(by_method[method])

    report_json = {
        "judge_root": str(root),
        "total": overall["total"],
        "correct": overall["correct"],
        "accuracy": overall["accuracy"],
        "by_question_type": by_question_type,
        "by_retrieval_method": by_retrieval_method,
        "notes": {
            "source_glob": "*/*/*/summary.json",
            "correct_label": "CORRECT",
        },
    }
    (root / "report.json").write_text(json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append(f"# Judge Report ({root.name})")
    lines.append("")
    lines.append("## Overall Accuracy")
    lines.append("")
    lines.append(f"- Total: {overall['total']}")
    lines.append(f"- Correct: {overall['correct']}")
    lines.append(f"- Accuracy: {overall['accuracy']:.2f}%")
    lines.append("")
    lines.append("## Accuracy By Question Type")
    lines.append("")
    lines.append("| Question Type | Correct | Total | Accuracy |")
    lines.append("|---|---:|---:|---:|")
    for qtype in sorted(by_question_type):
        row = by_question_type[qtype]
        lines.append(f"| {qtype} | {row['correct']} | {row['total']} | {row['accuracy']:.2f}% |")
    lines.append("")
    lines.append("## Accuracy By Retrieval Method")
    lines.append("")
    lines.append("| Retrieval Method | Correct | Total | Accuracy |")
    lines.append("|---|---:|---:|---:|")
    for method in sorted(by_retrieval_method):
        row = by_retrieval_method[method]
        lines.append(f"| {method} | {row['correct']} | {row['total']} | {row['accuracy']:.2f}% |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Source: `*/*/*/summary.json` under this run directory.")
    lines.append("- Label mapping: only `CORRECT` counts as correct.")
    lines.append("")
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")

    return report_json
