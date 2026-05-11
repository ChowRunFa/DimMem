#!/usr/bin/env python3
"""Backward-compatible entrypoint for memory_constructor/run_extract_windows_with_en_prompt.py."""
from __future__ import annotations

import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
LONGMEM_ROOT = THIS_FILE.parent
if str(LONGMEM_ROOT) not in sys.path:
    sys.path.insert(0, str(LONGMEM_ROOT))

from memory_constructor.run_extract_windows_with_en_prompt import *  # noqa: F401,F403

