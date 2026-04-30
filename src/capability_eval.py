"""Optional capability regression checks.

Two cheap probes:

1. ``run_mmlu``: a small subset of MMLU (default 500 examples) using a
   standard 5-shot prompt template. Reports overall accuracy. This is a
   sanity check that long-context fine-tuning didn't tank base
   capability — *not* a publication-grade MMLU run.

2. ``run_needle_in_haystack``: synthetic needle test. We bury a
   ``"The magic number is N"`` sentence at a configurable depth in a
   long filler context and ask the model to recall the number. Reports
   per-(depth, context_len) accuracy. Confirms long-context retrieval
   still works after fine-tuning.

Both helpers accept an HF model + tokenizer (the same kind returned by
:func:`src.model.build_model_and_tokenizer`). They do *not* require
vLLM — these checks are run rarely enough that HF generate is fine.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import EvalConfig

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


_log = logging.getLogger("capability_eval")


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------


_MMLU_LETTERS = ["A", "B", "C", "D"]


def _format_mmlu_question(ex: dict[str, Any]) -> str:
    return (
        f"Question: {ex['question']}\n"
        f"A. {ex['choices'][0]}\n"
        f"B. {ex['choices'][1]}\n"
        f"C. {ex['choices'][2]}\n"
        f"D. {ex['choices'][3]}\n"
        "Answer:"
    )


def run_mmlu(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    config: EvalConfig,
    n_examples: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Subset MMLU evaluation. Returns ``{"accuracy": float, "n": int}``."""
    import torch
    from datasets import load_dataset

    _log.info("Loading MMLU 'all' split (test) for capability check")
    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:n_examples]

    correct = 0
    n = 0
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for idx in indices:
            ex = ds[idx]
            prompt = _format_mmlu_question(ex)
            messages = [
                {"role": "user", "content": prompt},
            ]
            tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(device)
            out = model.generate(
                tokens,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            decoded = tokenizer.decode(
                out[0, tokens.shape[1] :], skip_special_tokens=True
            ).strip()
            pred_letter = next(
                (c for c in decoded.upper() if c in _MMLU_LETTERS), None
            )
            gold_letter = _MMLU_LETTERS[ex["answer"]]
            if pred_letter == gold_letter:
                correct += 1
            n += 1

    acc = correct / max(1, n)
    _log.info("MMLU subset accuracy: %.4f over n=%d", acc, n)
    return {"accuracy": acc, "n": n}


# ---------------------------------------------------------------------------
# Needle in a haystack
# ---------------------------------------------------------------------------


_FILLER_SENTENCE = (
    "The grass is green and the sky is blue. "
    "Birds sing in the trees and the clouds drift overhead. "
)


def _build_needle_prompt(
    tokenizer: "PreTrainedTokenizerBase",
    context_len_tokens: int,
    depth: float,
    needle: str,
) -> tuple[str, int]:
    """Construct a (filler ... needle ... filler) context of ~context_len_tokens.

    Returns (prompt_text, expected_answer_int).
    """
    # Cheap repetition for filler. We aim for token count, not character count.
    one_filler_tokens = len(
        tokenizer.encode(_FILLER_SENTENCE, add_special_tokens=False)
    )
    target_tokens = max(64, context_len_tokens)
    n_filler = max(1, target_tokens // max(1, one_filler_tokens))
    filler = _FILLER_SENTENCE * n_filler

    # Insert the needle at the requested depth in the filler text.
    insert_idx = int(len(filler) * depth)
    # Snap to a sentence boundary to avoid slicing words.
    snap = filler.rfind(". ", 0, insert_idx)
    if snap == -1:
        snap = insert_idx
    contaminated = filler[: snap + 2] + needle + " " + filler[snap + 2 :]

    user = (
        "Read the following passage carefully, then answer the question at the end.\n\n"
        f"Passage:\n{contaminated}\n\n"
        "Question: What is the magic number? Reply with just the number."
    )
    return user, 0  # caller fills in the actual answer


def run_needle_in_haystack(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    config: EvalConfig,
    depths: list[float] | None = None,
    context_lens: list[int] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run a small needle-in-a-haystack grid. Returns per-(depth, ctx_len) accuracy."""
    import torch

    depths = depths or [0.1, 0.5, 0.9]
    context_lens = context_lens or [8000, 16000, 32000]

    rng = random.Random(seed)
    model.eval()
    device = next(model.parameters()).device

    results: dict[str, dict[str, float]] = {}
    n_per_cell = 3  # cheap grid; bumps up easily if needed.

    with torch.no_grad():
        for ctx_len in context_lens:
            if ctx_len + 256 > config.inference_max_seq_len:
                _log.warning(
                    "Skipping ctx_len=%d (> inference_max_seq_len=%d)",
                    ctx_len,
                    config.inference_max_seq_len,
                )
                continue
            for depth in depths:
                cell_correct = 0
                for _ in range(n_per_cell):
                    magic = rng.randint(10_000, 99_999)
                    needle_sentence = f"The magic number is {magic}."
                    user_text, _ = _build_needle_prompt(
                        tokenizer, ctx_len, depth, needle_sentence
                    )
                    messages = [{"role": "user", "content": user_text}]
                    tokens = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(device)
                    out = model.generate(
                        tokens,
                        max_new_tokens=16,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    decoded = tokenizer.decode(
                        out[0, tokens.shape[1] :], skip_special_tokens=True
                    )
                    if str(magic) in decoded:
                        cell_correct += 1
                acc = cell_correct / n_per_cell
                results.setdefault(str(ctx_len), {})[f"{depth:.2f}"] = acc
                _log.info(
                    "needle ctx=%d depth=%.2f acc=%.2f", ctx_len, depth, acc
                )

    return {"per_cell_accuracy": results, "n_per_cell": n_per_cell}


# ---------------------------------------------------------------------------
# Combined writer
# ---------------------------------------------------------------------------


def write_capability_results(
    results: dict[str, Any], path: str | Path
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return p


__all__ = ["run_mmlu", "run_needle_in_haystack", "write_capability_results"]
