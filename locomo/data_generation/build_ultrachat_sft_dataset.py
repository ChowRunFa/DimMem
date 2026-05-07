#!/usr/bin/env python3
"""
将 (UltraChat window, extracted_memories) 对整理为 SFT 训练数据集。

输出格式: JSONL，每行一个样本，包含 messages 字段 (OpenAI chat format):
  [
    {"role": "system", "content": "<system_prompt>"},
    {"role": "user",   "content": "<conversation_text>"},
    {"role": "assistant", "content": "<memories_json>"}
  ]

用法:
    python build_ultrachat_sft_dataset.py \
        --conv-dir   .../ultrachat_windows_15 \
        --mem-dir    .../ultrachat_windows_15_memories \
        --output     .../sft_ultrachat_5000_w15.jsonl
"""

import argparse
import glob
import json
import logging
import os
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a structured memory extractor.

Task: Extract structured memories with long-term value from the input user-assistant conversation, and strictly output valid JSON.
Only output JSON. Do not output explanations, analysis, Markdown, or any extra text.

========================
Output Format
========================

{
  "memories": [
    {
      "source_id": 1,
      "content": "",
      "dimension": {
        "memory_type": "",
        "time": "",
        "location": "",
        "reason": "",
        "purpose": "",
        "keywords": []
      }
    }
  ]
}

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

episodic: specific events, answering "what happened / what someone did / what someone experienced / what someone plans to do".
Includes a specific event, experience, action, stage progress, future plan, or concrete fact with time/location/context.

profile: long-term user profile, answering "what someone is like long-term / what someone likes / what someone usually does / what someone believes".
Includes preferences, habits, interests, values, long-term goals, abilities, style preferences, and stable behavior patterns.

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

location: physical place, online platform, organizational context, home space, workplace, system environment, or activity venue.
Fill this only when the source text explicitly mentions or strongly implies it.

reason: cause, motivation, trigger, or background condition.
Fill this only when the source text explicitly states or strongly implies it.

purpose: goal, intention, or expected outcome.
Fill this only when the source text explicitly states or strongly implies it.

keywords: key terms or phrases for retrieval, deduplication, and query-memory alignment.
Keywords must be short words or noun phrases. Do not include full sentences. Do not repeat keywords.

========================
Extraction Rules
========================

1. Process messages in chronological order.
2. Extract memories mainly from user messages.
3. Each memory should be as atomic as possible. If a message is long or contains multiple independent information points, split it into multiple memories.
4. The `content` field must be self-contained and must not rely on the original dialogue context.
5. The `time` field should be normalized based on the message timestamp.
6. Simple confirmations, temporary formatting requirements, and one-off tasks should generally not be extracted.
7. The output must be valid JSON. Do not output any text outside the JSON."""


def main():
    parser = argparse.ArgumentParser(description="Build SFT dataset from UltraChat window-memory pairs")
    parser.add_argument("--conv-dir", required=True,
                        help="Directory with windows (batch_*/windows/*.txt)")
    parser.add_argument("--mem-dir", required=True,
                        help="Directory with extracted memories (batch_*/memories/*.json)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file path")
    parser.add_argument("--min-memories", type=int, default=1,
                        help="Skip samples with fewer than N memories (default: 1)")
    args = parser.parse_args()

    conv_dir = args.conv_dir
    mem_dir = args.mem_dir

    # Discover all memory JSON files
    mem_files = sorted(glob.glob(os.path.join(mem_dir, "*/memories/window_*.json")))
    logger.info(f"Found {len(mem_files)} memory files in {mem_dir}")

    total = 0
    skipped_no_conv = 0
    skipped_few_mem = 0
    skipped_parse_err = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out_f:
        for mem_path in mem_files:
            # Parse path: mem_dir/batch_XXXX/memories/window_XXXX.json
            rel = os.path.relpath(mem_path, mem_dir)
            parts = rel.split(os.sep)
            if len(parts) < 3:
                continue
            conv_name = parts[0]
            win_fname = parts[2]  # window_XXXX.json
            win_stem = win_fname.replace(".json", "")  # window_XXXX

            # Find corresponding conversation txt
            conv_txt_path = os.path.join(conv_dir, conv_name, "windows", f"{win_stem}.txt")
            if not os.path.exists(conv_txt_path):
                skipped_no_conv += 1
                continue

            # Read conversation
            with open(conv_txt_path, "r", encoding="utf-8") as f:
                conversation_text = f.read().strip()

            if not conversation_text:
                skipped_no_conv += 1
                continue

            # Read memories
            try:
                with open(mem_path, "r", encoding="utf-8") as f:
                    memories_data = json.load(f)
            except (json.JSONDecodeError, Exception):
                skipped_parse_err += 1
                continue

            memories_list = memories_data.get("memories", [])
            if len(memories_list) < args.min_memories:
                skipped_few_mem += 1
                continue

            # Build SFT sample
            assistant_response = json.dumps(memories_data, indent=2, ensure_ascii=False)

            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": conversation_text},
                    {"role": "assistant", "content": assistant_response},
                ],
                "metadata": {
                    "conv_name": conv_name,
                    "window": win_stem,
                    "n_memories": len(memories_list),
                },
            }

            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            total += 1

    logger.info(f"Done! {total} samples written to {args.output}")
    logger.info(f"  skipped_no_conv={skipped_no_conv}, skipped_few_mem={skipped_few_mem}, "
                f"skipped_parse_err={skipped_parse_err}")

    print(f"\n{'='*50}")
    print(f"SFT Dataset Summary")
    print(f"{'='*50}")
    print(f"Total samples: {total}")
    print(f"Skipped (no conv): {skipped_no_conv}")
    print(f"Skipped (few memories): {skipped_few_mem}")
    print(f"Skipped (parse error): {skipped_parse_err}")
    print(f"Output: {args.output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
