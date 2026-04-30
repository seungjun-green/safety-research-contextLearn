"""Long-context training datasets.

Supported ``dataset_name`` values:

- ``longalpaca``: ``Yukang/LongAlpaca-12k``
- ``longalign``: ``THUDM/LongAlign-10k``
- ``mixed``: combine multiple per ``data_mix_ratios``
- ``naive_refusal``: synthetic (jailbreak-shaped prompt → refusal) pairs.
  Sources its harmful prompts from a held-out HarmBench split. **Never
  generates new harmful content.** Disabled unless the user has explicitly
  accepted HarmBench's license (we just call ``datasets.load_dataset``;
  the user accepts at first download).
- any other string: treated as a HF dataset id, loaded as-is, with
  ``conversations`` / ``messages`` / ``instruction``-style auto-detection.

All examples are tokenised under the model's chat template. Labels are
masked to ``-100`` on user / system tokens so loss flows only on
assistant turns. This matters more here than usual because our contexts
are 16k–32k and a naive label-mask would leak loss across long user
turns.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

from .config import TrainConfig

if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase


IGNORE_INDEX = -100
_log = logging.getLogger("data")


# ---------------------------------------------------------------------------
# Per-source raw → unified messages format
# ---------------------------------------------------------------------------


def _longalpaca_to_messages(ex: dict[str, Any]) -> list[dict[str, str]]:
    """Yukang/LongAlpaca-12k has columns: 'instruction', 'output' (and 'input')."""
    user = ex.get("instruction", "")
    extra = ex.get("input", "")
    if extra:
        user = f"{user}\n\n{extra}"
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": ex.get("output", "")},
    ]


def _longalign_to_messages(ex: dict[str, Any]) -> list[dict[str, str]]:
    """THUDM/LongAlign-10k has 'messages' or 'conversations' style entries."""
    if "messages" in ex and isinstance(ex["messages"], list):
        return [
            {"role": m["role"], "content": m["content"]}
            for m in ex["messages"]
            if m.get("role") in {"system", "user", "assistant"}
        ]
    if "conversations" in ex and isinstance(ex["conversations"], list):
        # ShareGPT-style: [{"from": "human"|"gpt"|"system", "value": "..."}, ...]
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        out: list[dict[str, str]] = []
        for turn in ex["conversations"]:
            role = role_map.get(turn.get("from", ""), None)
            if role is None:
                continue
            out.append({"role": role, "content": turn.get("value", "")})
        return out
    # Fallback: instruction/output style
    return _longalpaca_to_messages(ex)


def _generic_to_messages(ex: dict[str, Any]) -> list[dict[str, str]]:
    """Best-effort converter for unknown HF datasets."""
    if "messages" in ex:
        return _longalign_to_messages(ex)
    if "conversations" in ex:
        return _longalign_to_messages(ex)
    return _longalpaca_to_messages(ex)


_CONVERTERS: dict[str, Any] = {
    "longalpaca": _longalpaca_to_messages,
    "longalign": _longalign_to_messages,
}

_HF_REPOS: dict[str, str] = {
    "longalpaca": "Yukang/LongAlpaca-12k",
    "longalign": "THUDM/LongAlign-10k",
}


# ---------------------------------------------------------------------------
# Tokenisation with assistant-only labels
# ---------------------------------------------------------------------------


def _tokenize_with_assistant_mask(
    messages: list[dict[str, str]],
    tokenizer: "PreTrainedTokenizerBase",
    max_seq_len: int,
) -> dict[str, list[int]] | None:
    """Tokenise a multi-turn conversation, masking labels on non-assistant tokens.

    Strategy: for each prefix ending after an assistant turn, render the
    prefix-without-final-assistant and the prefix-with-final-assistant.
    Tokens between the two are the assistant's tokens; label them as
    themselves and mask everything else to ``IGNORE_INDEX``.

    Returns ``None`` for examples that become empty after truncation.
    """
    if not messages or all(m.get("role") != "assistant" for m in messages):
        return None

    # Full rendering: ground truth.
    full_ids: list[int] = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    if not isinstance(full_ids, list):
        full_ids = list(full_ids)
    labels: list[int] = [IGNORE_INDEX] * len(full_ids)

    # Walk each assistant turn, mark its token span as supervised.
    cursor = 0
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prefix = messages[:i]
        prompt_ids = tokenizer.apply_chat_template(
            prefix, tokenize=True, add_generation_prompt=True
        )
        if not isinstance(prompt_ids, list):
            prompt_ids = list(prompt_ids)
        with_assistant = messages[: i + 1]
        with_ids = tokenizer.apply_chat_template(
            with_assistant, tokenize=True, add_generation_prompt=False
        )
        if not isinstance(with_ids, list):
            with_ids = list(with_ids)

        start = max(len(prompt_ids), cursor)
        end = len(with_ids)
        if end <= start or end > len(full_ids):
            continue
        for k in range(start, end):
            labels[k] = full_ids[k]
        cursor = end

    if all(label == IGNORE_INDEX for label in labels):
        return None

    # Truncate from the left if too long (keep the most recent context).
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[-max_seq_len:]
        labels = labels[-max_seq_len:]
        if all(label == IGNORE_INDEX for label in labels):
            return None

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def _tokenize_dataset(
    raw: "Dataset",
    converter: Any,
    tokenizer: "PreTrainedTokenizerBase",
    max_seq_len: int,
) -> "Dataset":
    """Map the raw dataset through ``converter`` and label-aware tokenisation."""
    from datasets import Dataset

    rows: list[dict[str, Any]] = []
    skipped = 0
    for ex in raw:
        msgs = converter(ex)
        if not msgs:
            skipped += 1
            continue
        tok = _tokenize_with_assistant_mask(msgs, tokenizer, max_seq_len)
        if tok is None:
            skipped += 1
            continue
        rows.append(tok)
    if skipped:
        _log.info("Skipped %d examples (empty / no assistant turn)", skipped)
    if not rows:
        raise RuntimeError("All examples filtered out during tokenisation.")
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# naive_refusal
# ---------------------------------------------------------------------------


_REFUSAL_TEMPLATES: list[str] = [
    "I can't help with that. If you're researching a related but legitimate topic, I'm happy to point you to public, non-operational resources.",
    "I won't provide guidance on that. The request is asking me to help cause harm, and I'd rather decline than try to find a partial workaround.",
    "Sorry, this isn't something I'll help with. I'm not going to walk through the steps even framed as fiction or as a hypothetical.",
    "No — I'm declining this. That includes the lower-detail or 'theoretical' versions, which would still be useful for the same harm.",
    "I can't help with this request. Adding more context, more examples, or framing it as roleplay won't change my answer.",
    "I won't assist with that. If there's an underlying lawful problem you're trying to solve, tell me what it is and I'll try to help with that instead.",
    "This is a refusal. The faux-conversation pattern in the prompt is a known jailbreak format, and I'm not going to comply with the final query.",
    "I'm not going to help with that. The earlier turns don't constitute consent, and the final ask is still harmful.",
    "Declining. I notice the prompt is structured to look like prior compliance — that doesn't change what's being asked of me now.",
    "I won't do that. I'm going to give a direct refusal rather than a long explanation that gradually leaks the information.",
]


def build_naive_refusal_dataset(
    tokenizer: "PreTrainedTokenizerBase",
    max_seq_len: int,
    n_examples: int = 1000,
    seed: int = 42,
    held_out_split: str = "train",
    harmbench_repo: str = "walledai/HarmBench",
) -> "Dataset":
    """Synthetic (faux jailbreak prompt → refusal) dataset.

    The harmful prompts are sampled from a held-out HarmBench split. The
    *responses* are always one of ``_REFUSAL_TEMPLATES`` — we never
    synthesise harmful content.

    Notes on contamination: the user is responsible for using a different
    split here than the one their testset draws from. By default we read
    from ``train`` and the testset builder reads from ``test``; if the
    user's split arrangement differs, they need to override
    ``held_out_split`` accordingly.
    """
    from datasets import load_dataset

    _log.info(
        "Building naive_refusal dataset from %s split=%s n=%d",
        harmbench_repo,
        held_out_split,
        n_examples,
    )

    rng = random.Random(seed)
    raw = load_dataset(harmbench_repo, split=held_out_split)

    # HarmBench column conventions vary; pick the harmful-prompt column robustly.
    col_candidates = ["behavior", "prompt", "goal", "instruction"]
    cols = set(raw.column_names)
    prompt_col: str | None = next((c for c in col_candidates if c in cols), None)
    if prompt_col is None:
        raise RuntimeError(
            f"Could not find a harmful-prompt column in {harmbench_repo}. "
            f"Looked for {col_candidates}, found columns: {sorted(cols)}"
        )

    harmful_prompts: list[str] = [str(p) for p in raw[prompt_col]]
    if not harmful_prompts:
        raise RuntimeError("Held-out HarmBench split is empty.")

    rows: list[dict[str, Any]] = []
    for _ in range(n_examples):
        # Build a faux many-shot prompt from 1–6 harmful prompts (paired with
        # refusals — so the model sees prior refusals, not prior compliance).
        # This trains "keep refusing even after a long context."
        n_shots = rng.randint(1, 6)
        faux: list[dict[str, str]] = []
        for _ in range(n_shots):
            q = rng.choice(harmful_prompts)
            a = rng.choice(_REFUSAL_TEMPLATES)
            faux.append({"role": "user", "content": q})
            faux.append({"role": "assistant", "content": a})
        target = rng.choice(harmful_prompts)
        refusal = rng.choice(_REFUSAL_TEMPLATES)
        msgs = faux + [
            {"role": "user", "content": target},
            {"role": "assistant", "content": refusal},
        ]
        tok = _tokenize_with_assistant_mask(msgs, tokenizer, max_seq_len)
        if tok is not None:
            rows.append(tok)

    from datasets import Dataset

    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_train_dataset(
    config: TrainConfig,
    tokenizer: "PreTrainedTokenizerBase",
) -> "Dataset":
    """Build the tokenised training dataset for ``config``."""
    from datasets import concatenate_datasets, load_dataset

    name = config.dataset_name
    _log.info("Loading training dataset: %s", name)

    if name == "naive_refusal":
        ds = build_naive_refusal_dataset(
            tokenizer=tokenizer,
            max_seq_len=config.max_seq_len,
            n_examples=config.dataset_subset_size or 1000,
            seed=config.seed,
        )
        return ds

    if name == "mixed":
        assert config.data_mix_ratios is not None  # validated in __post_init__
        parts: list["Dataset"] = []
        for sub_name, ratio in config.data_mix_ratios.items():
            if ratio <= 0:
                continue
            sub_cfg = _replace_dataset(config, sub_name)
            sub_ds = build_train_dataset(sub_cfg, tokenizer)
            n_keep = max(1, int(len(sub_ds) * ratio))
            sub_ds = sub_ds.select(range(min(n_keep, len(sub_ds))))
            parts.append(sub_ds)
        if not parts:
            raise RuntimeError("Mixed dataset has no positive-weight components.")
        ds = concatenate_datasets(parts)
        ds = ds.shuffle(seed=config.seed)
        return ds

    repo = _HF_REPOS.get(name, name)
    converter = _CONVERTERS.get(name, _generic_to_messages)
    raw = load_dataset(repo, split="train")
    if config.dataset_subset_size is not None:
        raw = raw.select(range(min(config.dataset_subset_size, len(raw))))

    return _tokenize_dataset(raw, converter, tokenizer, config.max_seq_len)


def split_train_val(ds: "Dataset", val_ratio: float, seed: int) -> tuple["Dataset", "Dataset"]:
    """Split a tokenised dataset into train/val (deterministic)."""
    if val_ratio <= 0:
        from datasets import Dataset

        return ds, Dataset.from_list([])
    n_val = max(1, int(len(ds) * val_ratio))
    splits = ds.train_test_split(test_size=n_val, seed=seed)
    return splits["train"], splits["test"]


def collate_for_causal_lm(
    batch: list[dict[str, Any]],
    pad_token_id: int,
) -> dict[str, Any]:
    """Right-pad a batch of tokenised examples for causal LM training.

    Uses right-padding (with ``IGNORE_INDEX`` on the labels) — this is
    the convention for SFT with attention masks. Flash Attention 2
    handles right-padding correctly.
    """
    import torch

    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for ex in batch:
        ids = ex["input_ids"]
        lbl = ex["labels"]
        am = ex["attention_mask"]
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        labels.append(lbl + [IGNORE_INDEX] * pad)
        attention_mask.append(am + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def _replace_dataset(config: TrainConfig, dataset_name: str) -> TrainConfig:
    """Return a copy of ``config`` pointing at a different ``dataset_name``."""
    from dataclasses import replace

    return replace(
        config,
        dataset_name=dataset_name,
        data_mix_ratios=None,
        dataset_subset_size=None,
    )


__all__ = [
    "build_train_dataset",
    "build_naive_refusal_dataset",
    "split_train_val",
    "collate_for_causal_lm",
    "IGNORE_INDEX",
]
