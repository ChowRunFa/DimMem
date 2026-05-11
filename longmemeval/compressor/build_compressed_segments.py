#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_MODEL = "/data/aios-weights/LLM-Lingua/llmlingua-2-bert-base-multilingual-cased-meetingbank"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        _ensure_dir(dst.parent)
        shutil.copyfile(src, dst)


def _window_text(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, message in enumerate(messages, start=1):
        timestamp = str(message.get("timestamp") or "").strip()
        weekday = str(message.get("weekday") or "").strip()
        content = str(message.get("content") or "")
        lines.append(f"[{timestamp}, {weekday}] {idx}.User: {content}")
    return "\n".join(lines)


def _compress_text(compressor: Any, text: str, *, rate: float, target_token: int) -> str:
    raw = str(text or "")
    if not raw.strip():
        return raw
    result = compressor.compress_prompt(
        context=[raw],
        instruction="",
        rate=rate,
        target_token=target_token,
    )
    compressed = str(result.get("compressed_prompt") or "").strip()
    return compressed if compressed else raw


def _compress_window_json(
    *,
    window_path: Path,
    compressor: Any,
    rate: float,
    target_token: int,
    model_name: str,
) -> Dict[str, Any]:
    payload = _load_json(window_path)
    messages = list(payload.get("messages") or [])
    compressed_messages: List[Dict[str, Any]] = []

    total_original = 0
    total_compressed = 0
    changed_count = 0
    for msg in messages:
        original = str(msg.get("content") or "")
        compressed = _compress_text(compressor, original, rate=rate, target_token=target_token)
        updated = dict(msg)
        updated["original_content"] = original
        updated["content"] = compressed
        updated["compression_applied"] = compressed != original
        updated["original_length"] = len(original)
        updated["compressed_length"] = len(compressed)
        compressed_messages.append(updated)
        total_original += len(original)
        total_compressed += len(compressed)
        if compressed != original:
            changed_count += 1

    out = dict(payload)
    out["messages"] = compressed_messages
    out["text"] = _window_text(compressed_messages)
    out["compression"] = {
        "message_count": len(compressed_messages),
        "compressed_message_count": changed_count,
        "original_chars": total_original,
        "compressed_chars": total_compressed,
        "compression_ratio": (total_compressed / total_original) if total_original else 1.0,
        "model_name": model_name,
        "rate": rate,
        "target_token": target_token,
        "content_only": True,
    }
    return out


def _iter_record_dirs(raw_run_root: Path) -> List[Path]:
    rows: List[Path] = []
    for summary in sorted(raw_run_root.glob("*/*/summary.json")):
        rows.append(summary.parent)
    return rows


def _compress_record(
    *,
    input_record_dir: Path,
    raw_run_root: Path,
    compressed_run_root: Path,
    compressor: Any,
    rate: float,
    target_token: int,
    model_name: str,
    device_map: str,
) -> Dict[str, Any]:
    rel_record_dir = input_record_dir.relative_to(raw_run_root)
    out_record_dir = compressed_run_root / rel_record_dir
    out_windows_dir = out_record_dir / "windows"
    _ensure_dir(out_windows_dir)

    _copy_if_exists(input_record_dir / "input_item.json", out_record_dir / "input_item.json")
    _copy_if_exists(input_record_dir / "all_user_messages.txt", out_record_dir / "all_user_messages.txt")

    src_summary = _load_json(input_record_dir / "summary.json")
    compressed_windows_meta: List[Dict[str, Any]] = []

    for window_json in sorted((input_record_dir / "windows").glob("window_*.json")):
        compressed_payload = _compress_window_json(
            window_path=window_json,
            compressor=compressor,
            rate=rate,
            target_token=target_token,
            model_name=model_name,
        )
        stem = window_json.stem
        _write_json(out_windows_dir / f"{stem}.json", compressed_payload)
        (out_windows_dir / f"{stem}.txt").write_text(compressed_payload.get("text", ""), encoding="utf-8")
        # Copy assistant_replies.json from raw window
        _copy_if_exists(
            input_record_dir / "windows" / f"{stem}_assistant_replies.json",
            out_windows_dir / f"{stem}_assistant_replies.json",
        )
        compressed_windows_meta.append(
            {
                "window_index": compressed_payload.get("window_index"),
                "message_count": compressed_payload.get("message_count"),
                "start_global_user_index": compressed_payload.get("start_global_user_index"),
                "end_global_user_index": compressed_payload.get("end_global_user_index"),
                "start_timestamp": compressed_payload.get("start_timestamp"),
                "end_timestamp": compressed_payload.get("end_timestamp"),
                "start_session_id": compressed_payload.get("start_session_id"),
                "end_session_id": compressed_payload.get("end_session_id"),
                "compression": compressed_payload.get("compression"),
            }
        )

    total_original = sum(int((row.get("compression") or {}).get("original_chars") or 0) for row in compressed_windows_meta)
    total_compressed = sum(int((row.get("compression") or {}).get("compressed_chars") or 0) for row in compressed_windows_meta)
    total_changed = sum(int((row.get("compression") or {}).get("compressed_message_count") or 0) for row in compressed_windows_meta)
    total_messages = sum(int((row.get("compression") or {}).get("message_count") or 0) for row in compressed_windows_meta)

    out_summary = dict(src_summary)
    out_summary["source_record_dir"] = str(input_record_dir)
    out_summary["output_dir"] = str(out_record_dir)
    out_summary["compression"] = {
        "model_name": model_name,
        "device_map": device_map,
        "rate": rate,
        "target_token": target_token,
        "content_only": True,
        "window_count": len(compressed_windows_meta),
        "message_count": total_messages,
        "compressed_message_count": total_changed,
        "original_chars": total_original,
        "compressed_chars": total_compressed,
        "compression_ratio": (total_compressed / total_original) if total_original else 1.0,
    }
    out_summary["windows"] = compressed_windows_meta
    _write_json(out_record_dir / "summary.json", out_summary)
    return out_summary


def run(args: argparse.Namespace) -> Path:
    raw_run_root = args.raw_run_root.resolve()
    output_root = args.output_root.resolve()
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    compressed_run_root = output_root / run_name
    _ensure_dir(compressed_run_root)

    from llmlingua import PromptCompressor

    compressor = PromptCompressor(
        model_name=args.model_name,
        device_map=args.device_map,
        use_llmlingua2=True,
        llmlingua2_config={
            "max_batch_size": int(args.max_batch_size),
            "max_force_token": int(args.max_force_token),
        },
        model_config={
            "low_cpu_mem_usage": False,
            "attn_implementation": "eager",
        },
    )

    record_dirs = _iter_record_dirs(raw_run_root)
    if args.max_records > 0:
        record_dirs = record_dirs[: args.max_records]

    failures: List[Dict[str, Any]] = []
    done = 0
    for idx, record_dir in enumerate(record_dirs, start=1):
        try:
            _compress_record(
                input_record_dir=record_dir,
                raw_run_root=raw_run_root,
                compressed_run_root=compressed_run_root,
                compressor=compressor,
                rate=args.rate,
                target_token=args.target_token,
                model_name=args.model_name,
                device_map=args.device_map,
            )
            done += 1
        except Exception as exc:
            failures.append({"index": idx, "record_dir": str(record_dir), "error": f"{type(exc).__name__}: {exc}"})

    source_ts = raw_run_root.name
    experiment = {
        "experiment_name": "full_compression_from_raw_segments",
        "created_at": datetime.now().isoformat(),
        "source_raw_dir": str(raw_run_root),
        "output_dir": str(compressed_run_root),
        "script": str(Path(__file__).resolve()),
        "model_name": args.model_name,
        "device_map": args.device_map,
        "rate": args.rate,
        "target_token": args.target_token,
        "source_timestamp": source_ts,
        "records_total": len(record_dirs),
        "records_done": done,
        "records_failed": len(failures),
    }
    _write_json(compressed_run_root / "experiment_config.json", experiment)
    if failures:
        _write_json(compressed_run_root / "failures.json", {"failures": failures})
    return compressed_run_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compress raw segment windows into compressed segment windows.")
    parser.add_argument(
        "--raw-run-root",
        type=Path,
        required=True,
        help="Raw segment run root, e.g. .../segment_results/raw/<timestamp>",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./results/segments/compressed"),
    )
    parser.add_argument("--run-name", default="", help="Optional output run folder name")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--rate", type=float, default=0.8)
    parser.add_argument("--target-token", type=int, default=-1)
    parser.add_argument("--max-records", type=int, default=0, help="0 means all records")
    parser.add_argument("--max-batch-size", type=int, default=50)
    parser.add_argument("--max-force-token", type=int, default=100)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output = run(args)
    print(str(output))


if __name__ == "__main__":
    main()

