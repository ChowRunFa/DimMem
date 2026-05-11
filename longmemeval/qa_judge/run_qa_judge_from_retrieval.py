#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import requests

THIS_FILE = Path(__file__).resolve()
LONGMEMEVAL_DIR = THIS_FILE.parents[1]
SUBMIT_ROOT = THIS_FILE.parents[2]
if str(LONGMEMEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(LONGMEMEVAL_DIR))

from prompts.qa_prompts import build_qa_payload
from prompts.judge_prompts import build_judge_payload
RETRIEVAL_ROOT = SUBMIT_ROOT / "results/retrieval"
QUERY_ANALYSIS_ROOT = SUBMIT_ROOT / "results/query_analysis"
QA_ROOT = SUBMIT_ROOT / "results/qa"
JUDGE_ROOT = SUBMIT_ROOT / "results/judge"

MODEL_NAME = "jiaorong-qwen3-80b-instruct"
BASE_URL = "https://c4ai.ccccltd.cn/api/compatible/v1"
API_KEY = "sk-TdEsLfD17ERBPSUYvBGMMoXnh2QXhQVP"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_retrieval_dirs(root: Path) -> Iterable[Tuple[str, str, str, Path]]:
    for question_type_dir in sorted(root.iterdir()):
        if not question_type_dir.is_dir():
            continue
        question_type = question_type_dir.name
        for method_dir in sorted(question_type_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method = method_dir.name
            for sample_dir in sorted(method_dir.iterdir()):
                if sample_dir.is_dir():
                    yield question_type, method, sample_dir.name, sample_dir


def _filter_record(record: Dict[str, Any]) -> Dict[str, Any]:
    dimension = record.get("dimension") if isinstance(record.get("dimension"), dict) else {}
    out: Dict[str, Any] = {}
    if _clean(record.get("source_time")):
        out["source_time"] = _clean(record.get("source_time"))
    if _clean(record.get("content")):
        out["content"] = _clean(record.get("content"))
    if _clean(dimension.get("reason")):
        out["dimension"] = out.get("dimension", {})
        out["dimension"]["reason"] = _clean(dimension.get("reason"))
    if _clean(dimension.get("purpose")):
        out["dimension"] = out.get("dimension", {})
        out["dimension"]["purpose"] = _clean(dimension.get("purpose"))
    return out


def _chat(prompt: str) -> Dict[str, Any]:
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()


def _extract_message(resp_json: Dict[str, Any]) -> str:
    try:
        return _clean(resp_json["choices"][0]["message"]["content"])
    except Exception:
        return ""


def _parse_answer(raw_text: str) -> Dict[str, Any]:
    reasoning = ""
    answer = raw_text.strip()
    for line in raw_text.splitlines():
        text = line.strip()
        if text.lower().startswith("reasoning:"):
            reasoning = text.split(":", 1)[1].strip()
        elif text.lower().startswith("answer:"):
            answer = text.split(":", 1)[1].strip()
    return {
        "reasoning": reasoning,
        "answer": answer,
        "raw_text": raw_text,
    }


def _parse_judge(raw_text: str) -> Dict[str, Any]:
    label = ""
    reasoning = ""
    try:
        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            label = _clean(payload.get("label")).upper()
            reasoning = _clean(payload.get("reasoning"))
    except Exception:
        pass
    if not label:
        match = re.search(r"\b(CORRECT|WRONG)\b", raw_text, flags=re.IGNORECASE)
        if match:
            label = match.group(1).upper()
    return {
        "label": label,
        "reasoning": reasoning,
        "raw_text": raw_text,
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    targets = list(_iter_retrieval_dirs(RETRIEVAL_ROOT))
    for question_type, method, sample_id, retrieval_dir in targets:
        input_json = QUERY_ANALYSIS_ROOT / question_type / sample_id / "input.json"
        top_records_json = retrieval_dir / "top_records.json"
        if not input_json.exists() or not top_records_json.exists():
            continue

        input_payload = _load_json(input_json)
        top_records = json.loads(top_records_json.read_text(encoding="utf-8"))
        filtered_records = [_filter_record(r) for r in top_records]

        query = _clean(input_payload.get("question"))
        gold_answer = input_payload.get("answer")

        qa_payload = build_qa_payload(query=query, retrieved_records=filtered_records)
        qa_resp_json = _chat(qa_payload["prompt"])
        qa_raw = _extract_message(qa_resp_json)
        qa_parsed = _parse_answer(qa_raw)

        qa_dir = QA_ROOT / question_type / method / sample_id
        qa_dir.mkdir(parents=True, exist_ok=True)
        (qa_dir / "qa_prompt.txt").write_text(qa_payload["prompt"], encoding="utf-8")
        _write_json(qa_dir / "qa_request.json", {
            "model_name": MODEL_NAME,
            "query": query,
            "retrieved_records": filtered_records,
        })
        _write_json(qa_dir / "qa_raw_response.json", qa_resp_json)
        _write_json(qa_dir / "qa_result.json", qa_parsed)

        judge_payload = build_judge_payload(
            query=query,
            gold_answer=gold_answer,
            model_answer=qa_parsed["answer"],
        )
        judge_resp_json = _chat(judge_payload["prompt"])
        judge_raw = _extract_message(judge_resp_json)
        judge_parsed = _parse_judge(judge_raw)

        judge_dir = JUDGE_ROOT / question_type / method / sample_id
        judge_dir.mkdir(parents=True, exist_ok=True)
        (judge_dir / "judge_prompt.txt").write_text(judge_payload["prompt"], encoding="utf-8")
        _write_json(judge_dir / "judge_request.json", {
            "model_name": MODEL_NAME,
            "query": query,
            "gold_answer": gold_answer,
            "model_answer": qa_parsed["answer"],
        })
        _write_json(judge_dir / "judge_raw_response.json", judge_resp_json)
        _write_json(judge_dir / "judge_result.json", judge_parsed)

        _write_json(qa_dir / "summary.json", {
            "question_type": question_type,
            "retrieval_method": method,
            "sample_id": sample_id,
            "query": query,
            "gold_answer": gold_answer,
            "qa_answer": qa_parsed["answer"],
        })
        _write_json(judge_dir / "summary.json", {
            "question_type": question_type,
            "retrieval_method": method,
            "sample_id": sample_id,
            "query": query,
            "gold_answer": gold_answer,
            "qa_answer": qa_parsed["answer"],
            "judge_label": judge_parsed["label"],
        })


if __name__ == "__main__":
    main()
