#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qa_judge.report import generate_judge_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-root", type=Path, required=True)
    args = parser.parse_args()
    result = generate_judge_report(args.judge_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
