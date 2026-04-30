#!/usr/bin/env python
"""Minimal training entry point.

Usage:
    python scripts/train.py --config configs/train/llama31_8b_longalpaca_32k.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_train_config  # noqa: E402
from src.trainer import Trainer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a TrainConfig YAML.")
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    final_ckpt = Trainer(cfg).train()
    print(f"FINAL_CHECKPOINT={final_ckpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
