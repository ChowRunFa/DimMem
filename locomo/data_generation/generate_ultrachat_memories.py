#!/usr/bin/env python3
"""
UltraChat 记忆标注脚本

使用 LONGMEMEVAL_STRUCTURED_MEMORY_EXTRACTION_PROMPT（不带 overlap rule）
对 UltraChat 15-message 窗口做结构化记忆抽取。

用法:
    python generate_ultrachat_memories.py \
        --input-dir  .../ultrachat_windows_15 \
        --output-dir .../ultrachat_windows_15_memories \
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

# ── Prompt (from longmemeval, no overlap rule) ──────────────────────────────

SYSTEM_PROMPT = "You are a structured memory extractor. You may only output valid JSON."

USER_PROMPT_TEMPLATE = """
You are a structured memory extractor.

Task: Extract structured memories with long-term value from the input user-assistant conversation, and strictly output valid JSON.
Only output JSON. Do not output explanations, analysis, Markdown, or any extra text.

========================
Output Format
========================

{{
  "memories": [
    {{
      "source_id": 1,
      "content": "",
      "dimension": {{
        "memory_type": "",
        "time": "",
        "location": "",
        "reason": "",
        "purpose": "",
        "keywords": []
      }}
    }}
  ]
}}

========================
Extraction Targets
========================

Extract:
1. Factual information, identity/background, relationships, current status, tools, models, datasets, and project configurations;
2. Specific experiences, events, actions, behavioral records, stage progress, and future plans;
3. Long-term preferences, habits, interests, values, goals, abilities, interaction style, or writing style;
4. Information that helps understand the user's future needs or retrieve the user's background.

Do not extract:
1. Greetings, thanks, simple confirmations, or meaningless small talk;
2. Temporary formatting requirements, one-off operation instructions, or current-task details without long-term value.

========================
memory_type
========================

dimension.memory_type must be one of: fact, episodic, profile.

fact: stable facts, answering "what it is / what exists / what is used / what the relationship is / what the current status is".
Includes identity, background, relationships, status, tools, models, datasets, configurations, confirmed choices, and stable objective attributes.
Example: The user uses LLaMA2-7B as the base model.

episodic: specific events, answering "what happened / what someone did / what someone experienced / what someone plans to do".
Includes a specific event, experience, action, stage progress, future plan, or concrete fact with time/location/context.
Example: The user plans to train a local LLaMA2-7B model using Urdu data.

profile: long-term user profile, answering "what someone is like long-term / what someone likes / what someone usually does / what someone believes".
Includes preferences, habits, interests, values, long-term goals, abilities, style preferences, and stable behavior patterns.
Example: The user prefers concise Java code with a single main function.

Do not extract content that cannot be classified as fact, episodic, or profile.

========================
content Rules
========================

content is the main text of the memory and must be one complete, self-contained, retrievable sentence.

Requirements:
1. Clearly state who the memory is about and the core fact, event, or profile information.
2. If the source text contains time, location, reason, or purpose, include them in content when possible.
3. Remove ambiguous pronouns so that content does not depend on the original context.
4. Normalize relative time expressions based on the message timestamp, e.g., yesterday → a specific date.
5. Do not add unsupported information, and do not overgeneralize a single event into a long-term profile.

========================
dimension Rules
========================

dimension is used for structured retrieval. Except for memory_type, use "" when there is no clear evidence; use [] for keywords when there are no clear keywords.

time: the time when the memory is valid, happened, is planned to happen, or repeatedly occurs.
Use an absolute date if available. Normalize relative time if the message timestamp is available. Use "" if there is no time.
Do not use the current system time unless it is the message timestamp.

location: physical place, online platform, organizational context, home space, workplace, system environment, or activity venue.
Fill this only when the source text explicitly mentions or strongly implies it. Do not force ordinary topics into location.

reason: cause, motivation, trigger, or background condition.
Fill this only when the source text explicitly states or strongly implies it. Do not infer hidden motivations. Do not confuse it with purpose.

purpose: goal, intention, or expected outcome.
Fill this only when the source text explicitly states or strongly implies it. Do not infer unstated purposes.

keywords: key terms or phrases for retrieval, deduplication, and query-memory alignment.
Extract subjects, objects, tools, models, datasets, projects, people, locations, activities, results, preference objects, interest domains, etc.
Keywords must be short words or noun phrases. Do not include full sentences. Do not repeat keywords. Do not extract ordinary words without retrieval value.

========================
Extraction Rules
========================

1. Process messages in chronological order.
2. Extract memories mainly from user messages.
3. Each memory should be as atomic as possible. If a message is long or contains multiple independent information points, split it into multiple memories, preserving key details such as people, time, location, events, reasons, purposes, preferences, and objects separately. Avoid over-merging or omitting details.
4. The `content` field must be self-contained and must not rely on the original dialogue context.
5. The `time` field should be normalized based on the message timestamp: relative time expressions must be converted into absolute dates or time ranges. For example, if the message timestamp is `2023-05-08` and the original text says `yesterday`, then `time` should be `2023-05-07`.
6. Simple confirmations, temporary formatting requirements, and one-off tasks in the current conversation should generally not be extracted.
7. The output must be valid JSON. Do not output any text outside the JSON.

Here is the real input you need to process:

{conversation}
"""


# ── Core logic ──────────────────────────────────────────────────────────────


def discover_windows(input_dir: str) -> list[dict]:
    """Discover all window txt files."""
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
    temperature: float,
    max_retries: int = 3,
) -> dict:
    """同步调用 LLM 抽取结构化记忆。"""
    user_prompt = USER_PROMPT_TEMPLATE.format(conversation=conversation_text)

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

    try:
        result = extract_memories_for_window(
            client, model, conversation_text, temperature,
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
    all_windows = discover_windows(args.input_dir)
    logger.info("Found %d windows", len(all_windows))

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
    parser = argparse.ArgumentParser(description="Extract structured memories from UltraChat windows")
    parser.add_argument("--input-dir", required=True, help="Dir with batch subdirectories containing windows/*.txt")
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
