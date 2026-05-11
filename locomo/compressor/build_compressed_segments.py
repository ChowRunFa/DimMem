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
    for idx, m in enumerate(messages, start=1):
        lines.append(f"[{m['timestamp']}, {m['weekday']}] {idx}.{m['speaker']}: {m['text']}")
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
        original = str(msg.get("text") or "")
        compressed = _compress_text(compressor, original, rate=rate, target_token=target_token)
        updated = dict(msg)
        updated["original_text"] = original
        updated["text"] = compressed
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
    # Legacy layout: <run>/<conv>/<record>/summary.json
    legacy = [p.parent for p in sorted(raw_run_root.glob("*/*/summary.json"))]
    if legacy:
        return legacy
    # Flattened layout: <run>/<conv>/summary.json + window_*.json at conv root
    flat = [p.parent for p in sorted(raw_run_root.glob("*/summary.json"))]
    return flat


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
    output_record_dir = compressed_run_root / rel_record_dir
    # Support both layouts:
    # 1) legacy: input_record_dir/windows/window_*.json
    # 2) flat:   input_record_dir/window_*.json
    input_windows_dir = input_record_dir / "windows"
    if not input_windows_dir.exists():
        input_windows_dir = input_record_dir

    output_windows_dir = output_record_dir / "windows"
    _ensure_dir(output_windows_dir)

    _copy_if_exists(input_record_dir / "input_item.json", output_record_dir / "input_item.json")
    _copy_if_exists(input_record_dir / "all_turns.txt", output_record_dir / "all_turns.txt")

    src_summary = _load_json(input_record_dir / "summary.json")
    compressed_windows: List[Dict[str, Any]] = []
    for json_path in sorted(input_windows_dir.glob("window_*.json")):
        compressed_payload = _compress_window_json(
            window_path=json_path,
            compressor=compressor,
            rate=rate,
            target_token=target_token,
            model_name=model_name,
        )
        stem = json_path.stem
        _write_json(output_windows_dir / f"{stem}.json", compressed_payload)
        (output_windows_dir / f"{stem}.txt").write_text(compressed_payload["text"], encoding="utf-8")
        compressed_windows.append(
            {
                "window_index": compressed_payload.get("window_index"),
                "message_count": compressed_payload.get("message_count"),
                "start_global_turn_index": compressed_payload.get("start_global_turn_index"),
                "end_global_turn_index": compressed_payload.get("end_global_turn_index"),
                "start_timestamp": compressed_payload.get("start_timestamp"),
                "end_timestamp": compressed_payload.get("end_timestamp"),
                "start_session_id": compressed_payload.get("start_session_id"),
                "end_session_id": compressed_payload.get("end_session_id"),
                "compression": compressed_payload.get("compression"),
            }
        )

    total_original = sum(int((row.get("compression") or {}).get("original_chars") or 0) for row in compressed_windows)
    total_compressed = sum(int((row.get("compression") or {}).get("compressed_chars") or 0) for row in compressed_windows)
    total_changed = sum(int((row.get("compression") or {}).get("compressed_message_count") or 0) for row in compressed_windows)
    total_messages = sum(int((row.get("compression") or {}).get("message_count") or 0) for row in compressed_windows)

    out_summary = dict(src_summary)
    out_summary["source_record_dir"] = str(input_record_dir)
    out_summary["output_dir"] = str(output_record_dir)
    out_summary["compression"] = {
        "model_name": model_name,
        "device_map": device_map,
        "rate": rate,
        "target_token": target_token,
        "content_only": True,
        "window_count": len(compressed_windows),
        "message_count": total_messages,
        "compressed_message_count": total_changed,
        "original_chars": total_original,
        "compressed_chars": total_compressed,
        "compression_ratio": (total_compressed / total_original) if total_original else 1.0,
    }
    out_summary["windows"] = compressed_windows
    _write_json(output_record_dir / "summary.json", out_summary)
    return out_summary


def run(args: argparse.Namespace) -> Path:
    raw_run_root = args.raw_run_root.resolve()
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_run_root = args.output_root.resolve() / run_name
    _ensure_dir(out_run_root)

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

    status_path = out_run_root / "status.json"
    failures_path = out_run_root / "failures.json"

    failures: List[Dict[str, Any]] = []
    if args.resume and failures_path.exists():
        try:
            failures = list(_load_json(failures_path).get("failures") or [])
        except Exception:
            failures = []

    done = 0
    failed = 0
    skipped = 0
    start_ts = datetime.now().isoformat()
    _write_json(
        status_path,
        {
            "state": "running",
            "started_at": start_ts,
            "updated_at": datetime.now().isoformat(),
            "source_raw_dir": str(raw_run_root),
            "output_dir": str(out_run_root),
            "records_total": len(record_dirs),
            "done": done,
            "failed": failed,
            "skipped_existing": skipped,
            "inflight_record": None,
            "resume": bool(args.resume),
        },
    )

    for idx, record_dir in enumerate(record_dirs, start=1):
        rel_record_dir = record_dir.relative_to(raw_run_root)
        output_record_dir = out_run_root / rel_record_dir
        output_summary = output_record_dir / "summary.json"
        if args.resume and output_summary.exists():
            skipped += 1
            done += 1
            _write_json(
                status_path,
                {
                    "state": "running",
                    "started_at": start_ts,
                    "updated_at": datetime.now().isoformat(),
                    "source_raw_dir": str(raw_run_root),
                    "output_dir": str(out_run_root),
                    "records_total": len(record_dirs),
                    "done": done,
                    "failed": failed,
                    "skipped_existing": skipped,
                    "inflight_record": None,
                    "last_record": str(record_dir),
                    "resume": bool(args.resume),
                },
            )
            continue

        _write_json(
            status_path,
            {
                "state": "running",
                "started_at": start_ts,
                "updated_at": datetime.now().isoformat(),
                "source_raw_dir": str(raw_run_root),
                "output_dir": str(out_run_root),
                "records_total": len(record_dirs),
                "done": done,
                "failed": failed,
                "skipped_existing": skipped,
                "inflight_record": str(record_dir),
                "resume": bool(args.resume),
            },
        )
        try:
            _compress_record(
                input_record_dir=record_dir,
                raw_run_root=raw_run_root,
                compressed_run_root=out_run_root,
                compressor=compressor,
                rate=args.rate,
                target_token=args.target_token,
                model_name=args.model_name,
                device_map=args.device_map,
            )
            done += 1
        except Exception as exc:
            failed += 1
            failures.append(
                {
                    "index": idx,
                    "record_dir": str(record_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _write_json(failures_path, {"failures": failures})
        finally:
            _write_json(
                status_path,
                {
                    "state": "running",
                    "started_at": start_ts,
                    "updated_at": datetime.now().isoformat(),
                    "source_raw_dir": str(raw_run_root),
                    "output_dir": str(out_run_root),
                    "records_total": len(record_dirs),
                    "done": done,
                    "failed": failed,
                    "skipped_existing": skipped,
                    "inflight_record": None,
                    "last_record": str(record_dir),
                    "resume": bool(args.resume),
                },
            )

    experiment = {
        "experiment_name": "full_compression_from_raw_segments",
        "created_at": datetime.now().isoformat(),
        "source_raw_dir": str(raw_run_root),
        "output_dir": str(out_run_root),
        "script": str(Path(__file__).resolve()),
        "model_name": args.model_name,
        "device_map": args.device_map,
        "rate": args.rate,
        "target_token": args.target_token,
        "source_timestamp": raw_run_root.name,
        "records_total": len(record_dirs),
        "records_done": done,
        "records_failed": failed,
        "records_skipped_existing": skipped,
        "resume": bool(args.resume),
    }
    _write_json(out_run_root / "experiment_config.json", experiment)
    if failures:
        _write_json(failures_path, {"failures": failures})
    _write_json(
        status_path,
        {
            "state": "completed",
            "started_at": start_ts,
            "updated_at": datetime.now().isoformat(),
            "source_raw_dir": str(raw_run_root),
            "output_dir": str(out_run_root),
            "records_total": len(record_dirs),
            "done": done,
            "failed": failed,
            "skipped_existing": skipped,
            "inflight_record": None,
            "resume": bool(args.resume),
        },
    )
    return out_run_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compress LoCoMo raw segment windows.")
    parser.add_argument("--raw-run-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./results/locomo_segments/compressed"),
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--rate", type=float, default=0.8)
    parser.add_argument("--target-token", type=int, default=-1)
    parser.add_argument("--max-records", type=int, default=0, help="0 means all records")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Disable resume/skip-existing mode")
    parser.set_defaults(resume=True)
    parser.add_argument("--max-batch-size", type=int, default=50)
    parser.add_argument("--max-force-token", type=int, default=100)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out = run(args)
    print(str(out))


if __name__ == "__main__":
    main()
