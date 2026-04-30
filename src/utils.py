"""Small cross-cutting utilities: seeding, logging, chat formatting."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizerBase


def set_seed(seed: int) -> None:
    """Seed Python / NumPy / PyTorch (CPU + CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device() -> str:
    """Return ``cuda`` if available, else ``cpu``."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def setup_logger(name: str, log_path: str | Path | None = None) -> logging.Logger:
    """Return a logger that writes to stderr and (optionally) to ``log_path``.

    Idempotent: calling twice with the same name reuses the existing handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path is not None:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def count_tokens(text: str, tokenizer: "PreTrainedTokenizerBase") -> int:
    """Return the number of tokens in ``text`` under ``tokenizer``."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def format_chat_for_jailbreak(
    faux_dialogue: list[dict[str, str]],
    target_query: str,
    tokenizer: "PreTrainedTokenizerBase",
    system_prompt: str | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """Render the many-shot jailbreak prompt via ``tokenizer.apply_chat_template``.

    Each ``faux_dialogue`` entry is rendered as a real ``(user, assistant)``
    message pair — NOT as raw concatenated text. This matches the actual
    many-shot jailbreak attack format and is critical for reproducibility.

    Parameters
    ----------
    faux_dialogue
        List of ``{"user": ..., "assistant": ...}`` dicts. May be empty.
    target_query
        The final harmful query used as the last user turn.
    tokenizer
        Any HF tokenizer that exposes ``apply_chat_template``.
    system_prompt
        Optional system prompt prepended to the conversation.
    add_generation_prompt
        Forwarded to ``apply_chat_template``. Set to True to leave the
        conversation ending at the start of the assistant's reply, ready
        for generation.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for turn in faux_dialogue:
        if "user" not in turn or "assistant" not in turn:
            raise ValueError(
                f"faux_dialogue turn missing required keys: {turn.keys()}"
            )
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": target_query})

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(rendered, str):  # apply_chat_template can return list[int]
        raise TypeError(
            "apply_chat_template returned a non-string; pass tokenize=False"
        )
    return rendered


def write_jsonl(path: str | Path, rows: list[dict[str, Any]], mode: str = "w") -> None:
    """Append-or-overwrite a list of dicts to a JSONL file."""
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append a single record to a JSONL file (creating parent dirs)."""
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


__all__ = [
    "set_seed",
    "get_device",
    "setup_logger",
    "count_tokens",
    "format_chat_for_jailbreak",
    "write_jsonl",
    "append_jsonl",
]
