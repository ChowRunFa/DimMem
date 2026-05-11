#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests

THIS_FILE = Path(__file__).resolve()
LOCOMO_SRC_ROOT = THIS_FILE.parents[1]
if str(LOCOMO_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCOMO_SRC_ROOT))

from prompts.judge_prompts import build_judge_payload


def _clean(v: Any) -> str:
    return str(v or "").strip()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_text(resp_json: Dict[str, Any]) -> str:
    try:
        return _clean(resp_json["choices"][0]["message"]["content"])
    except Exception:
        return ""


def _parse_label(raw_text: str) -> Dict[str, str]:
    label = ""
    reasoning = ""
    try:
        obj = json.loads(raw_text)
        if isinstance(obj, dict):
            label = _clean(obj.get("label")).upper()
            reasoning = _clean(obj.get("reasoning"))
    except Exception:
        pass
    if not label:
        m = re.search(r"\b(CORRECT|WRONG)\b", raw_text, flags=re.IGNORECASE)
        if m:
            label = m.group(1).upper()
    return {"label": label, "reasoning": reasoning, "raw_text": raw_text}


def _call_chat(
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    prompt: str,
    timeout: int,
    max_tokens: int,
) -> Dict[str, Any]:
    session = requests.Session()
    session.trust_env = False
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = session.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _build_gold_map(conv_json_path: Path, conv_name: str = "") -> Dict[str, Dict[str, Any]]:
    obj = _load_json(conv_json_path)
    rows = []
    if isinstance(obj, list):
        if conv_name:
            # Find matching conversation by sample_id
            for item in obj:
                if _clean(item.get("sample_id")) == conv_name:
                    rows = item.get("qa") or []
                    break
        elif len(obj) == 1:
            rows = obj[0].get("qa") or []
        else:
            # Merge all qa from all conversations
            for item in obj:
                rows.extend(item.get("qa") or [])
    m: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        q = _clean((r or {}).get("question"))
        if q:
            m[q] = r
            m[_norm_question(q)] = r
    return m


def _norm_question(text: str) -> str:
    s = _clean(text).lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[\"'`]", "", s)
    s = re.sub(r"\s+([?.!,;:])", r"\1", s)
    return s


def _iter_summary_files(out_root: Path) -> List[Path]:
    # Supports both layouts:
    # 1) <out>/<conv>/<sample>/summary.json
    # 2) <out>/<sample>/summary.json (single-conv flattened mode)
    return sorted(p for p in out_root.rglob("summary.json") if p.parent.name != out_root.name)


def _write_report(out_root: Path) -> None:
    summaries = _iter_summary_files(out_root)
    total = 0
    correct = 0
    by_conv: Dict[str, Dict[str, int]] = {}
    by_category: Dict[str, Dict[str, int]] = {}
    for p in summaries:
        d = _load_json(p)
        conv = _clean(d.get("conv_name"))
        category = _clean(d.get("category")) or "UNKNOWN"
        label = _clean(d.get("judge_label")).upper()
        if conv not in by_conv:
            by_conv[conv] = {"total": 0, "correct": 0}
        if category not in by_category:
            by_category[category] = {"total": 0, "correct": 0}
        by_conv[conv]["total"] += 1
        by_category[category]["total"] += 1
        total += 1
        if label == "CORRECT":
            by_conv[conv]["correct"] += 1
            by_category[category]["correct"] += 1
            correct += 1
    acc = (correct / total * 100.0) if total else 0.0
    lines = [
        f"# Judge Report ({out_root.name})",
        "",
        f"- Total: {total}",
        f"- Correct: {correct}",
        f"- Accuracy: {acc:.2f}%",
        "",
        "## Accuracy By Conv",
        "",
        "| Conv | Correct | Total | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for conv in sorted(by_conv):
        c = by_conv[conv]["correct"]
        t = by_conv[conv]["total"]
        a = (c / t * 100.0) if t else 0.0
        lines.append(f"| {conv} | {c} | {t} | {a:.2f}% |")
    lines.extend(
        [
            "",
            "## Accuracy By Category",
            "",
            "| Category | Correct | Total | Accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for cat in sorted(by_category):
        c = by_category[cat]["correct"]
        t = by_category[cat]["total"]
        a = (c / t * 100.0) if t else 0.0
        lines.append(f"| {cat} | {c} | {t} | {a:.2f}% |")
    (out_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_root = args.output_base / (args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_root.mkdir(parents=True, exist_ok=True)
    gold_map = _build_gold_map(args.conv_json, _clean(args.conv_name))

    qa_summaries = sorted(args.qa_root.glob("*/*/summary.json"))
    if _clean(args.conv_name):
        conv_name_filter = _clean(args.conv_name)
        qa_summaries = [p for p in qa_summaries if p.parent.parent.name == conv_name_filter]
    done = 0
    fail = 0
    missing_gold = 0
    rows: List[Dict[str, Any]] = []

    _write_json(
        out_root / "run_manifest.json",
        {
            "created_at": datetime.now().isoformat(),
            "qa_root": str(args.qa_root),
            "conv_json": str(args.conv_json),
            "output_root": str(out_root),
            "model_name": args.model_name,
            "base_url": args.base_url,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "max_tokens": args.max_tokens,
            "qa_count": len(qa_summaries),
            "conv_name_filter": _clean(args.conv_name),
        },
    )

    for qa_summary in qa_summaries:
        qa = _load_json(qa_summary)
        conv_name = qa_summary.parent.parent.name
        sample_id = qa_summary.parent.name
        query = _clean(qa.get("query"))
        if not query:
            req_path = qa_summary.parent / "qa_request.json"
            if req_path.exists():
                try:
                    req = _load_json(req_path)
                    query = _clean(req.get("query"))
                except Exception:
                    pass
        model_answer = _clean(qa.get("qa_answer"))

        # In single-conv filtered mode, avoid duplicated conv directory nesting.
        use_flat_layout = bool(_clean(args.conv_name))
        out_dir = (out_root / sample_id) if use_flat_layout else (out_root / conv_name / sample_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        gold_row = gold_map.get(query) or gold_map.get(_norm_question(query))
        if not gold_row:
            rec = {
                "conv_name": conv_name,
                "sample_id": sample_id,
                "ok": False,
                "error": "missing_gold",
                "query": query,
                "gold_answer": "",
                "qa_answer": model_answer,
                "output_dir": str(out_dir),
            }
            _write_json(out_dir / "summary.json", rec)
            rows.append(rec)
            fail += 1
            missing_gold += 1
            continue

        gold_answer = _clean(gold_row.get("answer"))
        category = _clean(gold_row.get("category"))
        payload = build_judge_payload(query=query, gold_answer=gold_answer, model_answer=model_answer)
        (out_dir / "judge_prompt.txt").write_text(payload["prompt"], encoding="utf-8")
        _write_json(
            out_dir / "judge_request.json",
            {"query": query, "gold_answer": gold_answer, "model_answer": model_answer, "qa_summary": str(qa_summary)},
        )

        ok = False
        err = None
        parsed: Dict[str, Any] | None = None
        resp_json: Dict[str, Any] | None = None
        started = time.time()
        for attempt in range(1, max(1, args.max_retries) + 1):
            try:
                resp_json = _call_chat(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model_name=args.model_name,
                    prompt=payload["prompt"],
                    timeout=args.timeout,
                    max_tokens=args.max_tokens,
                )
                raw_text = _extract_text(resp_json)
                parsed = _parse_label(raw_text)
                ok = bool(_clean((parsed or {}).get("label")))
                if not ok:
                    raise ValueError("empty_label")
                err = None
                break
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                if attempt < args.max_retries:
                    time.sleep(min(2 * attempt, 8))

        if resp_json is not None:
            _write_json(out_dir / "judge_raw_response.json", resp_json)
        if parsed is not None:
            _write_json(out_dir / "judge_result.json", parsed)
        rec = {
            "conv_name": conv_name,
            "sample_id": sample_id,
            "category": category,
            "ok": ok,
            "error": err,
            "elapsed_seconds": time.time() - started,
            "query": query,
            "gold_answer": gold_answer,
            "qa_answer": model_answer,
            "judge_label": _clean((parsed or {}).get("label")).upper(),
            "output_dir": str(out_dir),
        }
        _write_json(out_dir / "summary.json", rec)
        rows.append(rec)
        if ok:
            done += 1
        else:
            fail += 1

        _write_json(
            out_root / "status.json",
            {
                "state": "running",
                "total": len(qa_summaries),
                "done": done,
                "fail": fail,
                "missing_gold": missing_gold,
                "running": {"conv_name": conv_name, "sample_id": sample_id},
                "updated_at": datetime.now().isoformat(),
            },
        )

    _write_json(
        out_root / "summary.json",
        {"state": "completed", "total": len(qa_summaries), "done": done, "fail": fail, "missing_gold": missing_gold, "rows": rows},
    )
    _write_json(
        out_root / "status.json",
        {"state": "completed", "total": len(qa_summaries), "done": done, "fail": fail, "missing_gold": missing_gold, "updated_at": datetime.now().isoformat()},
    )
    _write_report(out_root)
    print(str(out_root))
    return out_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LoCoMo judge from QA results.")
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--conv-json", type=Path, required=True, help="Path to locomo10.json for the same conv")
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("./results/locomo_judge"),
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--base-url", default="http://127.0.0.1:7790/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-name", default="qwen3-30b-a3b")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--conv-name", default="", help="Only judge this conv from qa-root, e.g. Audrey-conv44")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
