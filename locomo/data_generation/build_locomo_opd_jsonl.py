#!/usr/bin/env python3
"""
将 LoCoMo SFT messages jsonl 转换为 ROLL OPD 格式：
- messages: [system, user] (不含 assistant)
- ground_truth: 原 assistant 内容 (用于 reward 评估参考)
- tag: locomo_opd
"""
import argparse
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="SFT jsonl (messages format)")
    p.add_argument("--output", required=True, help="OPD jsonl output")
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    n = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = row.get("messages", [])

            system_msg = None
            user_msg = None
            assistant_msg = None
            for m in msgs:
                if m["role"] == "system":
                    system_msg = m["content"]
                elif m["role"] == "user":
                    user_msg = m["content"]
                elif m["role"] == "assistant":
                    assistant_msg = m["content"]

            if not user_msg:
                continue

            opd_messages = []
            if system_msg:
                opd_messages.append({"role": "system", "content": system_msg})
            opd_messages.append({"role": "user", "content": user_msg})

            out = {
                "id": str(n),
                "source": "locomo_sft",
                "messages": opd_messages,
                "ground_truth": assistant_msg or "",
                "tag": "locomo_opd",
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} OPD samples -> {args.output}")


if __name__ == "__main__":
    main()
