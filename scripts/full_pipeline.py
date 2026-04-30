#!/usr/bin/env python
"""End-to-end experiment driver.

Runs (in order):
  1. Baseline jailbreak eval on the base model.
  2. LoRA fine-tuning per the train config.
  3. Post-FT jailbreak eval on the trained checkpoint.
  4. (Optional) Capability eval on the trained checkpoint.
  5. Plots ASR vs. shot_count for baseline vs. post-FT.

Driven by an "experiment manifest" YAML:

    experiment_name: longalpaca_32k_e2e
    output_dir: outputs
    train_config: configs/train/llama31_8b_longalpaca_32k.yaml
    eval_config:  configs/eval/jailbreak_eval_llama31_8b.yaml
    run_capability_eval: true   # optional, default false

Assumes ``jailbreak_testset.json`` has already been built. Run
``python scripts/build_testset_from_harmbench.py`` first if not.

Usage:
    python scripts/full_pipeline.py --manifest path/to/manifest.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    load_eval_config,
    load_train_config,
)
from src.jailbreak_eval import JailbreakEvaluator  # noqa: E402
from src.trainer import Trainer  # noqa: E402
from src.utils import setup_logger  # noqa: E402


def _load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        m = yaml.safe_load(f)
    required = {"experiment_name", "output_dir", "train_config", "eval_config"}
    missing = required - set(m or {})
    if missing:
        raise ValueError(f"Manifest missing keys: {sorted(missing)}")
    return m


def _plot_asr_vs_shotcount(
    baseline_results: dict[str, Any],
    post_ft_results: dict[str, Any],
    out_path: Path,
) -> None:
    """Headline figure: ASR vs shot_count, baseline overlaid with post-FT."""
    import matplotlib

    matplotlib.use("Agg")  # script-friendly: no display required
    import matplotlib.pyplot as plt

    base = baseline_results["asr_by_shot_count"]
    post = post_ft_results["asr_by_shot_count"]

    # Union of shot counts, in numeric order.
    keys = sorted({int(k) for k in base} | {int(k) for k in post})
    base_y = [base.get(str(k), 0.0) for k in keys]
    post_y = [post.get(str(k), 0.0) for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(keys, base_y, marker="o", label="baseline (no FT)")
    ax.plot(keys, post_y, marker="s", label="post-FT")
    ax.set_xscale("log", base=2)
    ax.set_xticks(keys)
    ax.set_xticklabels([str(k) for k in keys])
    ax.set_xlabel("shot count (faux dialogue turns)")
    ax.set_ylabel("attack success rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Many-shot jailbreak ASR vs. shot count")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to the manifest YAML.")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the pre-FT baseline eval (useful when iterating on training).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training; expects --checkpoint to be passed in.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="LoRA adapter directory to use when --skip-train is set.",
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    exp_name: str = manifest["experiment_name"]
    output_dir = Path(manifest["output_dir"]) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("full_pipeline", output_dir / "pipeline.log")
    logger.info("Manifest: %s", manifest)

    train_cfg = load_train_config(manifest["train_config"])
    eval_cfg = load_eval_config(manifest["eval_config"])

    # All artefacts go under outputs/<exp_name>/<phase>/.
    train_cfg = replace(
        train_cfg,
        output_dir=str(output_dir / "checkpoints"),
    )
    base_eval_cfg = replace(
        eval_cfg,
        output_dir=str(output_dir),
        experiment_name="baseline",
        checkpoint_path=None,
    )

    # 1. Baseline ----------------------------------------------------------
    if args.skip_baseline:
        logger.info("Skipping baseline eval per --skip-baseline")
        baseline_results = json.loads(
            (output_dir / "baseline" / "results.json").read_text()
        )
    else:
        logger.info("Running baseline jailbreak eval")
        baseline_results = JailbreakEvaluator(base_eval_cfg).run()

    # 2. Train -------------------------------------------------------------
    if args.skip_train:
        if not args.checkpoint:
            raise ValueError("--skip-train requires --checkpoint")
        ckpt_path = Path(args.checkpoint)
        logger.info("Skipping training; using checkpoint %s", ckpt_path)
    else:
        logger.info("Starting fine-tuning: %s", train_cfg.experiment_name)
        ckpt_path = Trainer(train_cfg).train()
        logger.info("Training complete: %s", ckpt_path)

    # 3. Post-FT eval ------------------------------------------------------
    post_ft_cfg = replace(
        eval_cfg,
        output_dir=str(output_dir),
        experiment_name="post_ft",
        checkpoint_path=str(ckpt_path),
    )
    logger.info("Running post-FT jailbreak eval")
    post_ft_results = JailbreakEvaluator(post_ft_cfg).run()

    # 4. Capability eval (optional) ---------------------------------------
    if manifest.get("run_capability_eval", False):
        logger.info("Running capability eval (MMLU + needle-in-haystack)")
        from src.capability_eval import (
            run_mmlu,
            run_needle_in_haystack,
            write_capability_results,
        )
        from src.checkpoint import load_lora_for_inference

        cap_model, cap_tok = load_lora_for_inference(
            base_model_name=eval_cfg.base_model_name,
            adapter_path=ckpt_path,
        )
        try:
            mmlu = run_mmlu(cap_model, cap_tok, post_ft_cfg, n_examples=500)
            needle = run_needle_in_haystack(cap_model, cap_tok, post_ft_cfg)
            write_capability_results(
                {"mmlu": mmlu, "needle_in_haystack": needle},
                output_dir / "post_ft" / "capability_results.json",
            )
        finally:
            del cap_model
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    # 5. Headline plot -----------------------------------------------------
    plot_path = output_dir / "plots" / "asr_vs_shotcount.png"
    _plot_asr_vs_shotcount(baseline_results, post_ft_results, plot_path)
    logger.info("Wrote headline plot to %s", plot_path)

    print(
        "OK\n"
        f"  baseline ASR: {baseline_results['asr_overall']:.4f}\n"
        f"  post-FT  ASR: {post_ft_results['asr_overall']:.4f}\n"
        f"  plot       : {plot_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
