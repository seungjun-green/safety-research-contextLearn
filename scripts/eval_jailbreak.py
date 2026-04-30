#!/usr/bin/env python
"""Run jailbreak evaluation on a checkpoint (or the base model).

Usage:
    # Baseline (no fine-tuning):
    python scripts/eval_jailbreak.py --config configs/eval/jailbreak_eval_llama31_8b.yaml

    # Fine-tuned checkpoint:
    python scripts/eval_jailbreak.py \\
        --config configs/eval/jailbreak_eval_llama31_8b.yaml \\
        --checkpoint outputs/longalpaca_32k/epoch1_step500_loss1.234/ \\
        --experiment-name longalpaca_32k_eval
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_eval_config  # noqa: E402
from src.jailbreak_eval import JailbreakEvaluator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an EvalConfig YAML.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override checkpoint_path (LoRA adapter dir, or merged model dir).",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Override experiment_name (so different checkpoints don't overwrite each other).",
    )
    args = parser.parse_args()

    cfg = load_eval_config(args.config)
    overrides: dict = {}
    if args.checkpoint is not None:
        overrides["checkpoint_path"] = args.checkpoint
    if args.experiment_name is not None:
        overrides["experiment_name"] = args.experiment_name
    if overrides:
        cfg = replace(cfg, **overrides)

    results = JailbreakEvaluator(cfg).run()
    print(
        f"ASR_OVERALL={results['asr_overall']:.4f} "
        f"N={results['n_examples']} "
        f"OUTPUT={Path(cfg.output_dir) / cfg.experiment_name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
