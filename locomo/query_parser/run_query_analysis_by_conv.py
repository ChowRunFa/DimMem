#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests


THIS_FILE = Path(__file__).resolve()
LOCOMO_SRC_ROOT = THIS_FILE.parents[1]
PROMPT_FILE = LOCOMO_SRC_ROOT / "prompts" / "prompts.py"
DEFAULT_INPUT_ROOT = Path("data/locomo10.json")
DEFAULT_OUTPUT_BASE = Path("./results/locomo_query_analysis")


def _extract_prompt_constant(name: str) -> str:
    text = PROMPT_FILE.read_text(encoding="utf-8")
    pattern = rf"{name}\s*=\s*\"\"\"(.*?)\"\"\""
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"unable to locate prompt constant: {name}")
    return match.group(1)


QUERY_PROMPT_TEMPLATE = _extract_prompt_constant("LOCOMO_QUERY_ANALYSIS_PROMPT")


def _safe_json_fragment(text: str) -> Any:
    payload = (text or "").strip()
    if not payload:
        raise ValueError("empty response")
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload, flags=re.IGNORECASE)
        payload = re.sub(r"\s*```$", "", payload).strip()
    try:
        return json.loads(payload)
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", payload):
        try:
            parsed, _ = decoder.raw_decode(payload[match.start() :])
            return parsed
        except Exception:
            continue
    raise ValueError("unable to parse JSON")


def _build_prompt(question: str) -> str:
    return QUERY_PROMPT_TEMPLATE.replace("{question}", str(question or "").strip())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _call_chat(
    *,
    session: requests.Session,
    base_url: str,
    api_key: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    response = session.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _extract_text(response_json: Dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    return str((choices[0].get("message") or {}).get("content") or "").strip()


def _load_questions(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"questions file is not a list: {path}")
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            row = {"question": str(row)}
        q = str(row.get("question") or "").strip()
        out.append({"index": i, "question": q, "raw": row})
    return out


def _load_conversations(input_root: Path, exclude_categories: List[int] | None = None) -> List[Dict[str, Any]]:
    """Load conversations from either a single JSON file or a directory.

    Single file (locomo10.json): list of conversations with 'qa' field.
    Directory: look for */locomo10_questions_only.json files (legacy format).
    """
    if exclude_categories is None:
        exclude_categories = []

    if input_root.is_file():
        data = json.loads(input_root.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {input_root}")
        result = []
        for item in data:
            conv_name = str(item.get("sample_id") or "").strip()
            questions = []
            for i, q in enumerate(item.get("qa") or []):
                if not isinstance(q, dict):
                    continue
                # Filter by category if specified
                category = q.get("category")
                if category is not None and int(category) in exclude_categories:
                    continue
                questions.append({"index": i, "question": str(q.get("question") or "").strip(), "raw": q})
            result.append({"conv_name": conv_name, "questions": questions})
        return result
    else:
        # Legacy: directory with */locomo10_questions_only.json
        conv_files = sorted(input_root.glob("*/locomo10_questions_only.json"))
        result = []
        for conv_file in conv_files:
            conv_name = conv_file.parent.name
            questions = _load_questions(conv_file)
            result.append({"conv_name": conv_name, "questions": questions})
        return result


def run(args: argparse.Namespace) -> Path:
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_base / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    conversations = _load_conversations(args.input_root, exclude_categories=args.exclude_categories)
    if args.max_convs > 0:
        conversations = conversations[: args.max_convs]

    _write_json(
        run_root / "run_manifest.json",
        {
            "created_at": datetime.now().isoformat(),
            "input_root": str(args.input_root),
            "output_root": str(run_root),
            "model_name": args.model_name,
            "base_url": args.base_url,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "resume": args.resume,
            "max_convs": args.max_convs,
            "max_questions_per_conv": args.max_questions_per_conv,
            "conv_count": len(conversations),
            "prompt_file": str(PROMPT_FILE),
        },
    )

    session = requests.Session()
    session.trust_env = False

    total = 0
    done = 0
    fail = 0
    summary_rows: List[Dict[str, Any]] = []

    for conv_entry in conversations:
        conv_name = conv_entry["conv_name"]
        conv_out = run_root / conv_name
        conv_out.mkdir(parents=True, exist_ok=True)

        questions = conv_entry["questions"]
        if args.max_questions_per_conv > 0:
            questions = questions[: args.max_questions_per_conv]
        total += len(questions)

        for item in questions:
            idx = int(item["index"])
            q = item["question"]
            sample_id = f"{idx:04d}"
            out_dir = conv_out / sample_id
            out_dir.mkdir(parents=True, exist_ok=True)

            result_path = out_dir / "result.json"
            if args.resume and result_path.exists():
                try:
                    old = json.loads(result_path.read_text(encoding="utf-8"))
                    if old.get("ok") is True:
                        done += 1
                        summary_rows.append(old)
                        continue
                except Exception:
                    pass

            prompt = _build_prompt(q)
            _write_json(out_dir / "input.json", item["raw"])
            (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

            ok = False
            error = None
            parsed = None
            raw_text = ""
            response_json: Dict[str, Any] | None = None
            started = time.time()

            for attempt in range(1, args.max_retries + 1):
                try:
                    response_json = _call_chat(
                        session=session,
                        base_url=args.base_url,
                        api_key=args.api_key,
                        model_name=args.model_name,
                        prompt=prompt,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                    )
                    raw_text = _extract_text(response_json)
                    parsed = _safe_json_fragment(raw_text)
                    if not isinstance(parsed, dict):
                        raise ValueError("parsed_response_not_object")
                    ok = True
                    error = None
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < args.max_retries:
                        time.sleep(min(2 * attempt, 8))

            elapsed = time.time() - started
            if response_json is not None:
                _write_json(out_dir / "raw_response.json", response_json)
            (out_dir / "raw_response.txt").write_text(raw_text, encoding="utf-8")
            if parsed is not None:
                _write_json(out_dir / "parsed.json", parsed)

            one = {
                "conv_name": conv_name,
                "index": idx,
                "question": q,
                "ok": ok,
                "error": error,
                "elapsed_seconds": elapsed,
                "usage": (response_json or {}).get("usage"),
                "output_dir": str(out_dir),
            }
            _write_json(result_path, one)
            summary_rows.append(one)

            if ok:
                done += 1
            else:
                fail += 1

            _write_json(
                run_root / "status.json",
                {
                    "run_root": str(run_root),
                    "total": total,
                    "done": done,
                    "fail": fail,
                    "running": {"conv_name": conv_name, "index": idx},
                    "updated_at": time.time(),
                },
            )

    final = {
        "run_root": str(run_root),
        "total": total,
        "done": done,
        "fail": fail,
        "rows": summary_rows,
    }
    _write_json(run_root / "summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return run_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LoCoMo query analysis by conv folders.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--base-url", default="http://127.0.0.1:7790/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-name", default="qwen3-30b-a3b")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-convs", type=int, default=0, help="0 means all")
    parser.add_argument("--max-questions-per-conv", type=int, default=0, help="0 means all")
    parser.add_argument("--exclude-categories", type=int, nargs="*", default=[], help="Categories to exclude (e.g., 5)")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()

