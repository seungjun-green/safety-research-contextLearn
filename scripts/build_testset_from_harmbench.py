#!/usr/bin/env python
"""Build the many-shot jailbreak testset from HarmBench.

Thin CLI wrapper around :func:`src.testset.build_testset`. The actual
HarmBench column / split detection lives in ``src/testset.py`` so it is
also importable from notebooks. If you need to tweak the column
auto-detection (HarmBench schemas drift between revisions), edit
``src/testset.py``.

Usage:
    python scripts/build_testset_from_harmbench.py \\
        --config configs/testset/harmbench_default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_testset_build_config  # noqa: E402
from src.testset import build_testset  # noqa: E402


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/testset/harmbench_default.yaml",
        help="Path to a TestsetBuildConfig YAML.",
    )
    args = parser.parse_args()

    cfg = load_testset_build_config(args.config)
    out_path = build_testset(cfg)
    print(
        f"OK: wrote testset to {out_path}. "
        "Validate with `python scripts/validate_testset.py`."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
