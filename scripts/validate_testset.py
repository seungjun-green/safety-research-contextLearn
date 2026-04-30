#!/usr/bin/env python
"""Schema sanity check for a built testset.

Usage:
    python scripts/validate_testset.py jailbreak_testset.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/validate_testset.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.testset import validate_schema  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="jailbreak_testset.json",
        help="Path to the testset JSON.",
    )
    args = parser.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f"ERROR: testset not found at {p}", file=sys.stderr)
        return 2

    with p.open("r", encoding="utf-8") as f:
        try:
            testset = json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: invalid JSON in {p}: {e}", file=sys.stderr)
            return 2

    errors = validate_schema(testset)
    if errors:
        print(f"FAIL: {len(errors)} validation error(s) in {p}:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    meta = testset.get("meta", {})
    n = len(testset.get("examples", []))
    print(
        f"OK: {p}\n"
        f"  source            = {meta.get('source')}\n"
        f"  total_examples    = {n}\n"
        f"  shot_sizes        = {meta.get('shot_sizes')}\n"
        f"  examples_per_size = {meta.get('examples_per_size')}\n"
        f"  categories        = {len(meta.get('categories', []))} unique\n"
        f"  seed              = {meta.get('seed')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
