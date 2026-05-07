#!/usr/bin/env python3
"""
将 UltraChat 多主题用户输入数据整理为 15 条消息一窗口的 txt 格式。

输入：train_0_multi_topic_user_input.jsonl
  每行 {"user_id", "topic_count", "source_ids", "input"}
  input 内含 ---TopicN---- 分隔符和 [timestamp] N.User: text 格式消息

输出：output_dir/<user_id>/windows/window_XXXX.txt
  每个文件 15 条消息，格式:
  [timestamp] N.User: text

策略：
  1. 将每个样本内的消息提取出来（去掉 topic 分隔符）
  2. 按 user_id 分组，同一用户的消息串联
  3. 连续消息重新编号后，按 15 条切分为窗口
  4. 总共生成 5000 个窗口

用法:
    python format_ultrachat_windows.py \
        --input /path/to/train_0_multi_topic_user_input.jsonl \
        --output-dir /path/to/ultrachat_windows_15 \
        --window-size 15 \
        --num-windows 5000
"""

import argparse
import json
import logging
import os
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_messages(input_text: str) -> list[str]:
    """从 input 字段提取所有消息行（去掉 topic 分隔符）。"""
    messages = []
    for line in input_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r"---Topic\d+----", line):
            continue
        if line.startswith("["):
            messages.append(line)
    return messages


def renumber_messages(messages: list[str]) -> list[str]:
    """重新编号消息为连续的 1, 2, 3, ...。"""
    result = []
    for i, msg in enumerate(messages, 1):
        # Replace the original number: [timestamp] OLD_NUM.User: -> [timestamp] NEW_NUM.User:
        new_msg = re.sub(
            r"(\[\d{4}-\d{2}-\d{2}T[\d:.]+,\s*\w+\])\s*\d+\.(User:)",
            rf"\1 {i}.\2",
            msg,
        )
        result.append(new_msg)
    return result


def main():
    parser = argparse.ArgumentParser(description="Format UltraChat data into 15-message windows")
    parser.add_argument("--input", required=True, help="Path to train_0_multi_topic_user_input.jsonl")
    parser.add_argument("--output-dir", required=True, help="Output directory for windows")
    parser.add_argument("--window-size", type=int, default=15, help="Messages per window (default: 15)")
    parser.add_argument("--num-windows", type=int, default=5000, help="Total windows to generate (default: 5000)")
    args = parser.parse_args()

    window_size = args.window_size
    num_windows = args.num_windows

    logger.info("Reading %s ...", args.input)

    # Collect all messages grouped by user_id
    user_messages = defaultdict(list)
    total_samples = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            user_id = d["user_id"]
            msgs = extract_messages(d["input"])
            if msgs:
                user_messages[user_id].extend(msgs)
            total_samples += 1

    logger.info("Read %d samples from %d users", total_samples, len(user_messages))
    total_msgs = sum(len(v) for v in user_messages.values())
    logger.info("Total messages: %d, possible %d-msg windows: %d",
                total_msgs, window_size, total_msgs // window_size)

    # Generate windows by slicing each user's messages into window_size chunks
    os.makedirs(args.output_dir, exist_ok=True)
    windows_written = 0

    for user_id in sorted(user_messages.keys()):
        if windows_written >= num_windows:
            break

        msgs = user_messages[user_id]
        if len(msgs) < window_size:
            continue

        # Slice into non-overlapping windows
        for start in range(0, len(msgs) - window_size + 1, window_size):
            if windows_written >= num_windows:
                break

            window_msgs = msgs[start:start + window_size]
            # Renumber
            window_msgs = renumber_messages(window_msgs)

            # Write to file
            win_dir = os.path.join(args.output_dir, user_id, "windows")
            os.makedirs(win_dir, exist_ok=True)
            win_idx = len([f for f in os.listdir(win_dir) if f.startswith("window_")])
            win_path = os.path.join(win_dir, f"window_{win_idx:04d}.txt")

            with open(win_path, "w", encoding="utf-8") as out_f:
                out_f.write("\n".join(window_msgs) + "\n")

            windows_written += 1

    # If not enough windows from single users, concatenate across users
    if windows_written < num_windows:
        logger.info("Got %d windows from single-user slicing, need %d more from cross-user concat",
                    windows_written, num_windows - windows_written)

        # Collect all remaining messages from users with < window_size msgs
        remaining_msgs = []
        for user_id in sorted(user_messages.keys()):
            msgs = user_messages[user_id]
            if len(msgs) < window_size:
                remaining_msgs.extend(msgs)
            else:
                # Also use leftover from slicing
                used = (len(msgs) // window_size) * window_size
                remaining_msgs.extend(msgs[used:])

        logger.info("Remaining messages for cross-user concat: %d", len(remaining_msgs))

        # Slice remaining into windows, use batch directories to avoid too many files in one dir
        batch_size = 500  # windows per batch directory
        batch_idx = 0
        batch_win_idx = 0

        for start in range(0, len(remaining_msgs) - window_size + 1, window_size):
            if windows_written >= num_windows:
                break

            window_msgs = remaining_msgs[start:start + window_size]
            window_msgs = renumber_messages(window_msgs)

            batch_name = f"batch_{batch_idx:04d}"
            win_dir = os.path.join(args.output_dir, batch_name, "windows")
            os.makedirs(win_dir, exist_ok=True)
            win_path = os.path.join(win_dir, f"window_{batch_win_idx:04d}.txt")

            with open(win_path, "w", encoding="utf-8") as out_f:
                out_f.write("\n".join(window_msgs) + "\n")

            windows_written += 1
            batch_win_idx += 1
            if batch_win_idx >= batch_size:
                batch_idx += 1
                batch_win_idx = 0

    logger.info("Done! Written %d windows to %s", windows_written, args.output_dir)

    # Write manifest
    manifest = {
        "input": args.input,
        "output_dir": args.output_dir,
        "window_size": window_size,
        "num_windows_requested": num_windows,
        "num_windows_written": windows_written,
        "total_samples": total_samples,
        "total_users": len(user_messages),
        "total_messages": total_msgs,
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
