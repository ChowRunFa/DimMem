#!/usr/bin/env python3
"""
合成对话数据生成器

读取已分窗的 LoCoMo 对话数据，调用 LLM 生成风格相似但人物/主题/时间不同的合成训练数据。
同一 conv 下的所有 window 共享同一对新人物名，保持会话一致性。

使用同步 OpenAI client + ThreadPoolExecutor 并发（避免 httpx async 代理问题）。

用法:
    python generate_synthetic_conversations.py \
        --input-dir  .../20260426_by_conv_raw \
        --output-dir .../generated_conversation \
        --num-variants 10 \
        --concurrency 16
"""

import argparse
import glob
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Semaphore

import httpx
from openai import OpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)


def setup_logging(log_file: str | None = None):
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="a"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )

# ── 人名池 ──────────────────────────────────────────────────────────────────

NAME_POOL = [
    "Alice", "Bob", "Charlie", "Diana", "Eric", "Fiona", "George", "Hannah",
    "Ivan", "Julia", "Kevin", "Laura", "Mike", "Nina", "Oscar", "Pam",
    "Quinn", "Rachel", "Steve", "Tina", "Victor", "Wendy", "Xavier", "Yara",
    "Zach", "Amber", "Brian", "Cindy", "Derek", "Elena", "Frank", "Grace",
    "Henry", "Iris", "Jake", "Karen", "Leo", "Mona", "Nathan", "Olivia",
    "Patrick", "Rosa", "Scott", "Tracy", "Uma", "Vince", "Wanda", "Yuri",
    "Bella", "Carlos", "Daisy", "Ethan", "Felix", "Gloria", "Hugo", "Isla",
    "Jason", "Kira", "Liam", "Megan", "Noah", "Pearl", "Ruben", "Sara",
    "Tyler", "Vera", "Will", "Zoe",
    # batch 2
    "Adrian", "Beth", "Cliff", "Donna", "Edgar", "Flora", "Grant", "Holly",
    "Ian", "Jade", "Kent", "Lisa", "Marco", "Nora", "Owen", "Penny",
    "Reed", "Sonia", "Troy", "Ursula", "Wade", "Xena", "Yolanda", "Zane",
    "Ava", "Blake", "Cora", "Dante", "Elise", "Finn", "Greta", "Heath",
    "Ivy", "Joel", "Kate", "Lance", "Maya", "Nico", "Opal", "Porter",
    "Rhea", "Seth", "Tara", "Uri", "Violet", "Warren", "Ximena", "Yves",
    "Zelda", "Axel", "Bree", "Cole", "Daria", "Emil", "Freya", "Glen",
    "Hazel", "Igor", "June", "Kirk", "Lydia", "Miles", "Nell", "Otto",
    "Priya", "Remy", "Stella", "Theo", "Una", "Vega", "Wyatt", "Yuki",
    "Zara", "Arlo", "Bianca", "Cruz", "Delia", "Emery", "Gage", "Hana",
    "Jude", "Kai", "Leona", "Marcel", "Nyla", "Orion", "Pia", "Rocco",
    "Sage", "Tate", "Vito", "Wren", "Yael", "Zion",
]


def assign_name_pairs(
    conv_speakers: dict[str, tuple[str, str]],
    num_variants: int,
) -> dict[str, list[tuple[str, str]]]:
    """为每个 conv 的每个 variant 分配一对不重复的新人名。"""
    all_original = set()
    for a, b in conv_speakers.values():
        all_original.add(a)
        all_original.add(b)

    available = [n for n in NAME_POOL if n not in all_original]
    random.shuffle(available)

    result = {}
    idx = 0
    for conv_name in sorted(conv_speakers.keys()):
        pairs = []
        for _ in range(num_variants):
            if idx + 1 >= len(available):
                random.shuffle(available)
                idx = 0
            pairs.append((available[idx], available[idx + 1]))
            idx += 2
        result[conv_name] = pairs
    return result


# ── Prompt ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a conversation data generator. Your task is to create a brand-new conversation \
that imitates the STYLE and STRUCTURE of a given example, but with DIFFERENT \
characters, topics, and timestamps.

Rules:
1. Use the two NEW character names specified by the user (do NOT use the original names).
2. Choose a COMPLETELY DIFFERENT topic/subject matter, but keep the same conversational \
   depth, tone, and turn-by-turn structure (e.g., if the example has greetings → deep \
   discussion → encouragement, your output should follow the same emotional arc).
3. Keep EXACTLY the same number of messages as the example.
4. Preserve which "side" each speaker is on: if the first speaker in the example initiates, \
   the corresponding new speaker should also initiate; the turn order must mirror the example.
5. Timestamps: shift the base date randomly (a different year/month/day), but preserve \
   the SAME time gaps between consecutive messages. The weekday abbreviation MUST match \
   the actual day of week for the shifted date.
6. Output ONLY the conversation lines, one per message, in EXACTLY this format:
   [YYYY-MM-DDTHH:MM:SS, Weekday] N.Speaker: text
   or with fractional seconds:
   [YYYY-MM-DDTHH:MM:SS.ffffff, Weekday] N.Speaker: text
7. Do NOT output anything else — no explanation, no markdown, no extra blank lines.
"""

USER_PROMPT_TEMPLATE = """\
Below is an example conversation with {msg_count} messages between "{orig_a}" and "{orig_b}".

Replace them with the following new characters:
- "{orig_a}" → "{new_a}"
- "{orig_b}" → "{new_b}"

Generate a NEW conversation with a COMPLETELY DIFFERENT topic, following all the rules.

=== EXAMPLE ===
{example_text}
=== END EXAMPLE ===

Now generate the new conversation ({msg_count} messages) using "{new_a}" and "{new_b}" \
with a different topic and shifted timestamps. Output ONLY the conversation lines."""

# ── Core logic ──────────────────────────────────────────────────────────────


def discover_windows(input_dir: str) -> list[dict]:
    """Scan input_dir for all window JSON files."""
    pattern = os.path.join(input_dir, "*", "0*", "windows", "window_*.json")
    json_files = sorted(glob.glob(pattern))
    windows = []
    for jf in json_files:
        rel = os.path.relpath(jf, input_dir)
        parts = Path(rel).parts
        windows.append({
            "json_path": jf,
            "conv_name": parts[0],
            "sample_id": parts[1],
            "window_file": parts[3],
        })
    return windows


def get_conv_speakers(input_dir: str) -> dict[str, tuple[str, str]]:
    """从每个 conv 的第一个 window 提取两位说话人。"""
    result = {}
    pattern = os.path.join(input_dir, "*-conv*")
    for conv_dir in sorted(glob.glob(pattern)):
        conv_name = os.path.basename(conv_dir)
        win0_list = sorted(glob.glob(os.path.join(conv_dir, "0*", "windows", "window_0000.json")))
        if not win0_list:
            continue
        with open(win0_list[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        speakers = []
        seen = set()
        for m in data["messages"]:
            s = m["speaker"]
            if s not in seen:
                speakers.append(s)
                seen.add(s)
            if len(speakers) == 2:
                break
        result[conv_name] = (speakers[0], speakers[1])
    return result


def load_window_text(json_path: str) -> tuple[str, int]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["text"], data["message_count"]


def generate_one_variant(
    client: OpenAI,
    model: str,
    example_text: str,
    msg_count: int,
    orig_a: str,
    orig_b: str,
    new_a: str,
    new_b: str,
    temperature: float,
    max_retries: int = 3,
) -> str:
    """同步调用 LLM 生成一个变体。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            msg_count=msg_count,
            orig_a=orig_a,
            orig_b=orig_b,
            new_a=new_a,
            new_b=new_b,
            example_text=example_text,
        )},
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
            return content
        except Exception as e:
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(1.5 * (2 ** attempt))
            else:
                raise


def process_one_task(
    client: OpenAI,
    model: str,
    window_meta: dict,
    output_dir: str,
    variant_idx: int,
    temperature: float,
    orig_speakers: tuple[str, str],
    name_pair: tuple[str, str],
) -> dict | None:
    """处理单个 (window, variant) 任务，返回 error dict 或 None。"""
    json_path = window_meta["json_path"]
    text, msg_count = load_window_text(json_path)
    orig_a, orig_b = orig_speakers
    new_a, new_b = name_pair

    win_stem = Path(window_meta["window_file"]).stem
    out_win_dir = os.path.join(
        output_dir,
        window_meta["conv_name"],
        window_meta["sample_id"],
        "windows",
    )
    os.makedirs(out_win_dir, exist_ok=True)
    out_path = os.path.join(out_win_dir, f"{win_stem}_v{variant_idx}.txt")

    if os.path.exists(out_path):
        return None  # 断点续跑

    try:
        result = generate_one_variant(
            client, model, text, msg_count,
            orig_a, orig_b, new_a, new_b,
            temperature,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result + "\n")
        return None
    except Exception as e:
        logger.error("Failed: %s v%d — %s", json_path, variant_idx, e)
        return {
            "window": json_path,
            "variant": variant_idx,
            "new_speakers": [new_a, new_b],
            "error": str(e),
        }


def run(args: argparse.Namespace):
    logger.info("Discovering windows in %s ...", args.input_dir)
    windows = discover_windows(args.input_dir)
    logger.info("Found %d windows", len(windows))

    conv_speakers = get_conv_speakers(args.input_dir)
    logger.info("Conversations: %s", {k: v for k, v in conv_speakers.items()})

    name_mapping = assign_name_pairs(conv_speakers, args.num_variants)
    for conv_name, pairs in name_mapping.items():
        orig = conv_speakers[conv_name]
        for vi, (na, nb) in enumerate(pairs):
            logger.info("  %s v%d: %s->%s, %s->%s", conv_name, vi, orig[0], na, orig[1], nb)

    # 保存名字映射
    os.makedirs(args.output_dir, exist_ok=True)
    mapping_path = os.path.join(args.output_dir, "name_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({
            conv: {
                "original": list(conv_speakers[conv]),
                "variants": [{"variant": i, "speakers": list(p)} for i, p in enumerate(pairs)]
            }
            for conv, pairs in name_mapping.items()
        }, f, indent=2, ensure_ascii=False)

    # 构建所有 (window, variant) 任务
    all_tasks = []
    for w in windows:
        conv = w["conv_name"]
        for vi in range(args.num_variants):
            all_tasks.append((w, vi, conv_speakers[conv], name_mapping[conv][vi]))

    total_tasks = len(all_tasks)
    logger.info("Total tasks: %d (concurrency=%d)", total_tasks, args.concurrency)

    # 创建同步 client，显式禁用代理
    http_client = httpx.Client(proxy=None, timeout=120.0)
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        http_client=http_client,
    )

    all_errors = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {}
        for w, vi, orig_sp, name_pair in all_tasks:
            fut = executor.submit(
                process_one_task,
                client, args.model, w, args.output_dir,
                vi, args.temperature, orig_sp, name_pair,
            )
            futures[fut] = (w["json_path"], vi)

        for fut in tqdm(as_completed(futures), total=total_tasks, desc="Generating"):
            err = fut.result()
            if err is not None:
                all_errors.append(err)

    # Write manifest
    manifest = {
        "created_at": datetime.now().isoformat(),
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "model": args.model,
        "base_url": args.base_url,
        "num_variants": args.num_variants,
        "concurrency": args.concurrency,
        "temperature": args.temperature,
        "total_windows": len(windows),
        "total_generated": total_tasks - len(all_errors),
        "total_errors": len(all_errors),
    }
    manifest_path = os.path.join(args.output_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    http_client.close()

    if all_errors:
        err_path = os.path.join(args.output_dir, "errors.json")
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(all_errors, f, indent=2, ensure_ascii=False)
        logger.warning("Finished with %d errors, see %s", len(all_errors), err_path)
    else:
        logger.info("All done! %d files generated, 0 errors.", total_tasks)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic conversation data from LoCoMo windows")
    parser.add_argument("--input-dir", required=True, help="Root dir with conv subdirectories")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--base-url", default="https://models-proxy.stepfun-inc.com/v1",
                        help="LLM API base URL")
    parser.add_argument("--api-key", default="ak-ic8y499r1fkx40brlq70z3dlata25c0b",
                        help="API key")
    parser.add_argument("--model", default="gpt-5.4", help="Model name")
    parser.add_argument("--num-variants", type=int, default=3,
                        help="Variants per window (default: 3)")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Max concurrent LLM calls (default: 16)")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Generation temperature (default: 0.9)")
    parser.add_argument("--log-file", default=None,
                        help="Log file path (also logs to stderr)")
    args = parser.parse_args()
    setup_logging(args.log_file)
    run(args)


if __name__ == "__main__":
    main()
