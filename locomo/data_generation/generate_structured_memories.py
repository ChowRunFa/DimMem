#!/usr/bin/env python3
"""
结构化记忆抽取器

读取生成的合成对话 window txt 文件，调用 LLM 按 LOCOMO prompt 抽取结构化记忆 JSON。
第一个 window (window_0000) 不带 overlap 规则，后续 window 前 5 条为重叠上下文。

使用同步 OpenAI client + ThreadPoolExecutor（避免 httpx async 代理问题）。

用法:
    python generate_structured_memories.py \
        --input-dir  .../generated_conversation \
        --output-dir .../generated_conversation_memories \
        --concurrency 16
"""

import argparse
import glob
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import httpx
from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Prompt ──────────────────────────────────────────────────────────────────

from prompts import LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT

OverlappingContextRules = ""  # Not used for synthetic data (has_overlap=False)

SYSTEM_PROMPT = "You are a structured memory extractor. You may only output valid JSON."

# ── Core logic ──────────────────────────────────────────────────────────────


def discover_conv_dirs(input_dir: str) -> list[dict]:
    """Discover all conversation directories and their windows."""
    result = []
    for conv_dir_name in sorted(os.listdir(input_dir)):
        conv_path = os.path.join(input_dir, conv_dir_name)
        win_dir = os.path.join(conv_path, "windows")
        if not os.path.isdir(win_dir):
            continue
        txt_files = sorted(glob.glob(os.path.join(win_dir, "window_*.txt")))
        for txt_path in txt_files:
            fname = os.path.basename(txt_path)
            m = re.match(r"window_(\d+)\.txt", fname)
            if not m:
                continue
            win_idx = int(m.group(1))
            result.append({
                "conv_name": conv_dir_name,
                "win_idx": win_idx,
                "txt_path": txt_path,
                "fname_stem": f"window_{win_idx:04d}",
            })
    return result


def extract_json_from_response(content: str) -> dict:
    """从 LLM 响应中提取 JSON。"""
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and start < end:
        try:
            return json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Cannot parse JSON from response: {content[:200]}...")


def extract_memories_for_window(
    client: OpenAI,
    model: str,
    conversation_text: str,
    has_overlap: bool,
    temperature: float,
    max_retries: int = 3,
) -> dict:
    """同步调用 LLM 抽取结构化记忆。"""
    overlap_rules = OverlappingContextRules if has_overlap else ""
    user_prompt = (LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT
                   .replace("{{OverlappingContextRules}}", overlap_rules)
                   .replace("{{conversation}}", conversation_text))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=4096,
            )
            content = resp.choices[0].message.content.strip()
            return extract_json_from_response(content)
        except Exception as e:
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(1.5 * (2 ** attempt))
            else:
                raise


def process_one_window(
    client: OpenAI,
    model: str,
    win_meta: dict,
    output_dir: str,
    temperature: float,
) -> dict | None:
    """处理单个 window，返回 error dict 或 None。"""
    out_conv_dir = os.path.join(output_dir, win_meta["conv_name"], "memories")
    os.makedirs(out_conv_dir, exist_ok=True)
    out_path = os.path.join(out_conv_dir, f"{win_meta['fname_stem']}.json")

    if os.path.exists(out_path):
        return None

    with open(win_meta["txt_path"], "r", encoding="utf-8") as f:
        conversation_text = f.read().strip()

    # 合成数据每个 window 独立生成，无真实重叠，统一不加 overlap 规则
    has_overlap = False

    try:
        result = extract_memories_for_window(
            client, model, conversation_text, has_overlap, temperature,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return None
    except Exception as e:
        logger.error("Failed: %s — %s", win_meta["txt_path"], e)
        return {
            "window": win_meta["txt_path"],
            "conv_name": win_meta["conv_name"],
            "win_idx": win_meta["win_idx"],
            "error": str(e),
        }


def run(args: argparse.Namespace):
    logger.info("Discovering windows in %s ...", args.input_dir)
    all_windows = discover_conv_dirs(args.input_dir)
    logger.info("Found %d windows across %d conversations",
                len(all_windows),
                len(set(w["conv_name"] for w in all_windows)))

    http_client = httpx.Client(proxy=None, timeout=120.0)
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        http_client=http_client,
    )

    all_errors = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(process_one_window, client, args.model, w, args.output_dir, args.temperature): w
            for w in all_windows
        }
        for fut in tqdm(as_completed(futures), total=len(all_windows), desc="Extracting memories"):
            err = fut.result()
            if err is not None:
                all_errors.append(err)

    http_client.close()

    manifest = {
        "created_at": datetime.now().isoformat(),
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "model": args.model,
        "total_windows": len(all_windows),
        "total_success": len(all_windows) - len(all_errors),
        "total_errors": len(all_errors),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if all_errors:
        with open(os.path.join(args.output_dir, "errors.json"), "w", encoding="utf-8") as f:
            json.dump(all_errors, f, indent=2, ensure_ascii=False)
        logger.warning("Finished with %d errors", len(all_errors))
    else:
        logger.info("All done! %d windows processed, 0 errors.", len(all_windows))


def main():
    parser = argparse.ArgumentParser(description="Extract structured memories from synthetic conversation windows")
    parser.add_argument("--input-dir", required=True, help="Dir with conv subdirectories containing windows/*.txt")
    parser.add_argument("--output-dir", required=True, help="Output directory for memory JSONs")
    parser.add_argument("--base-url", default="https://models-proxy.stepfun-inc.com/v1",
                        help="LLM API base URL")
    parser.add_argument("--api-key", default="ak-ic8y499r1fkx40brlq70z3dlata25c0b",
                        help="API key")
    parser.add_argument("--model", default="gpt-5.4", help="Model name")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Max concurrent LLM calls (default: 16)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Generation temperature (default: 0.1)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
