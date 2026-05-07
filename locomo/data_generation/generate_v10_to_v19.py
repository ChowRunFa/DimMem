#!/usr/bin/env python3
"""
生成 v10-v19 合成对话数据。

读取原始 LoCoMo 窗口数据，为每个 conv 分配 10 组全新人名（与 v0-v9 完全不重复），
调用 LLM 生成变体对话，输出到 results/generated_conversation/ 下的扁平目录结构。

用法:
    python generate_v10_to_v19.py \
        --api-key <KEY> \
        --concurrency 16
"""

import argparse
import glob
import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import httpx
from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE, "results", "segment_results", "raw", "20260426_by_conv_raw")
OUTPUT_DIR = os.path.join(BASE, "results", "generated_conversation")
MAPPING_PATH = os.path.join(OUTPUT_DIR, "name_mapping.json")

VARIANT_START = 10
VARIANT_END = 20  # exclusive, so v10-v19

# ── 全新人名池 (200+ 名字, 与 v0-v9 完全不重复) ──────────────────────────
NEW_NAME_POOL = [
    # A
    "Abby", "Ada", "Agnes", "Aiden", "Alana", "Albert", "Alyssa", "Andre",
    "Angela", "Anita", "Anton", "April", "Ariel", "Arnold", "Asher", "Astrid",
    # B
    "Barrett", "Becky", "Benny", "Beryl", "Blair", "Bonnie", "Boyd", "Brady",
    "Brenda", "Brent", "Bridget", "Bruno", "Bryant", "Byron",
    # C
    "Caleb", "Camila", "Carmen", "Casey", "Cecil", "Celeste", "Chad", "Clara",
    "Clark", "Cleo", "Clyde", "Colby", "Connie", "Conrad", "Corey", "Craig",
    # D
    "Dahlia", "Dale", "Damon", "Dana", "Daphne", "Darcy", "Darin", "Dawn",
    "Dean", "Dennis", "Devin", "Dolly", "Donald", "Doris", "Drew", "Dustin",
    # E
    "Earl", "Edith", "Edwin", "Effie", "Elaine", "Elijah", "Eliot", "Ella",
    "Ellis", "Elmer", "Elsa", "Enid", "Erin", "Ernest", "Esther", "Eunice",
    # F
    "Faith", "Faye", "Floyd", "Frances", "Frida", "Fritz",
    # G
    "Gail", "Gareth", "Gary", "Gemma", "Gerald", "Ginger", "Gordon", "Graham",
    "Greta", "Griffin", "Gus", "Gwen",
    # H
    "Hal", "Harlan", "Harriet", "Harvey", "Heidi", "Helen", "Herbert", "Homer",
    "Hope", "Howard", "Hunter",
    # I
    "Ida", "Ines", "Ingrid", "Irene", "Irving", "Isolde", "Iver",
    # J
    "Janet", "Jarvis", "Jean", "Jenny", "Jerome", "Jesse", "Jill", "Joaquin",
    "Jocelyn", "Jonas", "Joy", "Judith", "Jules", "Justin",
    # K
    "Keegan", "Keith", "Kelly", "Kendall", "Kenny", "Kim",
    # L
    "Lana", "Larry", "Leah", "Lena", "Leon", "Lester", "Lewis", "Lila",
    "Linda", "Lloyd", "Logan", "Lois", "Loretta", "Louis", "Lucia", "Luther",
    # M
    "Mabel", "Maddie", "Magnus", "Malik", "Mandy", "Margo", "Marlene",
    "Martin", "Mateo", "Maude", "Maxine", "Melvin", "Mercy", "Milton", "Mira",
    "Morris", "Murray", "Myra",
    # N
    "Nadine", "Nancy", "Ned", "Nellie", "Nelson", "Neville", "Nigel", "Norma",
    # O
    "Odette", "Olga", "Omar", "Oona", "Orla", "Otis", "Owen",
    # P
    "Paige", "Palmer", "Paula", "Percy", "Perry", "Petra", "Phillip", "Piper",
    # Q
    "Quincy",
    # R
    "Ralph", "Ramona", "Randall", "Raquel", "Raymond", "Regina", "Rena",
    "Rex", "Rita", "Robin", "Roger", "Roland", "Ronda", "Rosie", "Roy", "Ruby", "Ruth",
    # S
    "Sadie", "Sandra", "Sandy", "Selma", "Shane", "Sharon", "Shelby", "Simon",
    "Skip", "Solomon", "Spencer", "Stanley", "Stuart", "Suki", "Sylvia",
    # T
    "Tabitha", "Tamara", "Tanya", "Teresa", "Terrence", "Thea", "Theo",
    "Tiffany", "Todd", "Tommy", "Trent", "Trudy",
    # U
    "Ulysses",
    # V
    "Valerie", "Vance", "Vernon", "Vivian",
    # W
    "Wallace", "Walter", "Wayne", "Wesley", "Whitney", "Wilma", "Winston",
    # X-Z
    "Yvette", "Yvonne", "Zeke", "Zelma",
]

# ── Prompt (增强多样性版本) ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a conversation data generator. Your task is to create a brand-new conversation \
that uses the given example ONLY as a structural blueprint (message count, turn order, \
time gaps). The content, topic, tone, and personal details must be COMPLETELY ORIGINAL.

Rules:
1. Use the two NEW character names specified by the user (do NOT use the original names).
2. Choose a COMPLETELY DIFFERENT topic AND a DIFFERENT emotional tone from the example. \
   For instance, if the example is about hobbies and gardening, do NOT write about \
   hobbies or gardening — choose something from a totally different domain such as \
   career changes, travel adventures, family milestones, health journeys, tech projects, \
   financial decisions, creative collaborations, community activism, academic pursuits, \
   childhood memories, culinary experiments, sports competitions, pet stories, home \
   renovation, volunteering abroad, learning a new language, etc.
3. Vary the conversational style: some conversations can be casual and playful, others \
   serious and reflective, others excited and fast-paced. Do NOT default to a warm, \
   supportive, hobby-sharing pattern every time.
4. Give each character a DISTINCT personality, background, and speaking style. Avoid \
   making both speakers sound interchangeable.
5. Include SPECIFIC, CONCRETE details (names of places, dates, amounts, brands, book \
   titles, food names, etc.) rather than generic statements. Make the conversation feel \
   like a real exchange between real people.
6. Keep EXACTLY the same number of messages as the example.
7. Preserve which "side" each speaker is on: the turn order must mirror the example.
8. Timestamps: shift the base date randomly (a different year/month/day), but preserve \
   the SAME time gaps between consecutive messages. The weekday abbreviation MUST match \
   the actual day of week for the shifted date.
9. Output ONLY the conversation lines, one per message, in EXACTLY this format:
   [YYYY-MM-DDTHH:MM:SS, Weekday] N.Speaker: text
   or with fractional seconds:
   [YYYY-MM-DDTHH:MM:SS.ffffff, Weekday] N.Speaker: text
10. Do NOT output anything else — no explanation, no markdown, no extra blank lines.
"""

USER_PROMPT_TEMPLATE = """\
Below is an example conversation with {msg_count} messages between "{orig_a}" and "{orig_b}". \
Use it ONLY as a structural template (message count, turn order, time gaps).

Replace the characters:
- "{orig_a}" → "{new_a}"
- "{orig_b}" → "{new_b}"

IMPORTANT: Generate a conversation about a COMPLETELY DIFFERENT subject from the example. \
Do NOT reuse any topics, activities, or themes from the example. Create fresh, specific, \
and realistic content with concrete details. Vary the emotional tone and speaking styles.

=== STRUCTURAL TEMPLATE ===
{example_text}
=== END TEMPLATE ===

Now generate the new conversation ({msg_count} messages) using "{new_a}" and "{new_b}" \
with an entirely different topic, different tone, and shifted timestamps. \
Output ONLY the conversation lines."""


def load_existing_names():
    """加载已有 name_mapping.json 中的所有名字。"""
    if not os.path.exists(MAPPING_PATH):
        return set()
    with open(MAPPING_PATH) as f:
        nm = json.load(f)
    used = set()
    for info in nm.values():
        for s in info["original"]:
            used.add(s)
        for v in info["variants"]:
            for s in v["speakers"]:
                used.add(s)
    return used


def get_conv_speakers_and_windows():
    """扫描原始数据，返回 {conv_key: {speakers, windows}}。"""
    convs = {}
    for conv_dir in sorted(glob.glob(os.path.join(INPUT_DIR, "*-conv*"))):
        conv_name = os.path.basename(conv_dir)
        # Find sample subdirectory
        sample_dirs = sorted(glob.glob(os.path.join(conv_dir, "0*")))
        if not sample_dirs:
            continue
        sample_dir = sample_dirs[0]
        sample_id = os.path.basename(sample_dir)
        win_dir = os.path.join(sample_dir, "windows")
        if not os.path.isdir(win_dir):
            continue

        # Get speakers from first window
        win0 = os.path.join(win_dir, "window_0000.json")
        if not os.path.exists(win0):
            continue
        with open(win0) as f:
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

        # Collect all windows
        windows = []
        for jf in sorted(glob.glob(os.path.join(win_dir, "window_*.json"))):
            windows.append(jf)

        convs[conv_name] = {
            "speakers": (speakers[0], speakers[1]),
            "sample_id": sample_id,
            "windows": windows,
        }
    return convs


def assign_new_name_pairs(convs, num_variants, used_names):
    """为 v10-v19 分配全新人名对。"""
    available = [n for n in NEW_NAME_POOL if n not in used_names]
    random.seed(42)  # 可复现
    random.shuffle(available)

    needed = len(convs) * num_variants * 2
    if len(available) < needed:
        raise ValueError(f"Need {needed} names but only {len(available)} available")

    result = {}
    idx = 0
    for conv_name in sorted(convs.keys()):
        pairs = []
        for _ in range(num_variants):
            pairs.append((available[idx], available[idx + 1]))
            idx += 2
        result[conv_name] = pairs
    return result


def generate_one(client, model, example_text, msg_count, orig_a, orig_b,
                 new_a, new_b, temperature, max_retries=3):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            msg_count=msg_count, orig_a=orig_a, orig_b=orig_b,
            new_a=new_a, new_b=new_b, example_text=example_text,
        )},
    ]
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=4096,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(1.5 * (2 ** attempt))
            else:
                raise


def process_task(client, model, win_json, conv_name, conv_num,
                 variant_idx, orig_speakers, name_pair, temperature):
    """处理单个 (window, variant)，输出到扁平目录。"""
    new_a, new_b = name_pair
    orig_a, orig_b = orig_speakers

    # Parse window index
    win_stem = Path(win_json).stem  # window_0000

    # Output: NewName-convXX_vNN/windows/window_XXXX.txt
    out_dir_name = f"{new_a}-{conv_name.split('-')[1]}_v{variant_idx}"
    out_win_dir = os.path.join(OUTPUT_DIR, out_dir_name, "windows")
    os.makedirs(out_win_dir, exist_ok=True)
    out_path = os.path.join(out_win_dir, f"{win_stem}.txt")

    if os.path.exists(out_path):
        return None  # skip existing

    # Load original window
    with open(win_json) as f:
        data = json.load(f)
    text = data["text"]
    msg_count = data["message_count"]

    try:
        result = generate_one(
            client, model, text, msg_count,
            orig_a, orig_b, new_a, new_b, temperature,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result + "\n")
        return None
    except Exception as e:
        logger.error("Failed: %s v%d — %s", win_json, variant_idx, e)
        return {"window": win_json, "variant": variant_idx, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--base-url", default="https://models-proxy.stepfun-inc.com/v1")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    # 1. 扫描原始数据
    convs = get_conv_speakers_and_windows()
    total_windows = sum(len(c["windows"]) for c in convs.values())
    logger.info("Found %d convs, %d windows total", len(convs), total_windows)

    # 2. 分配新人名 (避开已用名字)
    used_names = load_existing_names()
    logger.info("Already used %d names", len(used_names))
    num_variants = VARIANT_END - VARIANT_START
    new_pairs = assign_new_name_pairs(convs, num_variants, used_names)

    for conv_name, pairs in new_pairs.items():
        orig = convs[conv_name]["speakers"]
        for vi, (na, nb) in enumerate(pairs):
            logger.info("  %s v%d: %s→%s, %s→%s",
                        conv_name, vi + VARIANT_START, orig[0], na, orig[1], nb)

    # 3. 更新 name_mapping.json
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH) as f:
            mapping = json.load(f)
    else:
        mapping = {}

    for conv_name, pairs in new_pairs.items():
        # Find the key in existing mapping
        conv_num = conv_name.split("-")[1]  # conv44
        map_key = None
        for k in mapping:
            if conv_num in k:
                map_key = k
                break
        if map_key and map_key in mapping:
            for vi, (na, nb) in enumerate(pairs):
                mapping[map_key]["variants"].append({
                    "variant": vi + VARIANT_START,
                    "speakers": [na, nb],
                })
        else:
            orig = convs[conv_name]["speakers"]
            mapping[conv_name] = {
                "original": list(orig),
                "variants": [
                    {"variant": vi + VARIANT_START, "speakers": [na, nb]}
                    for vi, (na, nb) in enumerate(pairs)
                ],
            }

    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    logger.info("Updated name_mapping.json")

    # 4. 构建任务
    all_tasks = []
    for conv_name, info in convs.items():
        conv_num = conv_name.split("-")[1]
        for win_json in info["windows"]:
            for vi in range(num_variants):
                all_tasks.append((
                    win_json, conv_name, conv_num,
                    vi + VARIANT_START,
                    info["speakers"],
                    new_pairs[conv_name][vi],
                ))

    logger.info("Total tasks: %d", len(all_tasks))

    # 5. 并发生成
    http_client = httpx.Client(proxy=None, timeout=120.0)
    client = OpenAI(
        api_key=args.api_key, base_url=args.base_url,
        http_client=http_client,
    )

    errors = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                process_task, client, args.model,
                wj, cn, cnum, vi, sp, np_, args.temperature,
            ): (wj, vi)
            for wj, cn, cnum, vi, sp, np_ in all_tasks
        }
        for fut in tqdm(as_completed(futures), total=len(all_tasks), desc="Generating v10-v19"):
            err = fut.result()
            if err:
                errors.append(err)

    http_client.close()

    if errors:
        err_path = os.path.join(OUTPUT_DIR, "errors_v10_v19.json")
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        logger.warning("Finished with %d errors", len(errors))
    else:
        logger.info("All done! %d files generated, 0 errors.", len(all_tasks))


if __name__ == "__main__":
    main()
