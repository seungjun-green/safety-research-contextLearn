"""LoRA adapter save / load / merge utilities.

Disk footprint: a Llama-3.1-8B + LoRA(r=16) adapter is ~80–150MB per
checkpoint. We never save optimizer state — restartable training is not
worth the disk cost for the experiments this repo runs (1 epoch, ~6h on
an H100). If you need restartability, save it explicitly outside the
trainer.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from .config import TrainConfig, dump_config

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedModel


_log = logging.getLogger("checkpoint")


def save_lora_checkpoint(
    model: "PreTrainedModel",
    config: TrainConfig,
    path: str | Path,
) -> Path:
    """Save a PEFT model's adapter + the resolved config to ``path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    # PEFT models expose .save_pretrained which writes adapter_config.json
    # and adapter_model.safetensors only — no full model weights, no
    # optimizer state. This is what we want.
    model.save_pretrained(str(p))

    dump_config(config, p / "config.yaml")
    _log.info("Saved LoRA checkpoint to %s", p)
    return p


def load_lora_for_inference(
    base_model_name: str,
    adapter_path: str | Path,
):
    """Load a base model and apply a LoRA adapter for HF-based inference.

    Used by capability eval / debug paths. The main jailbreak eval path
    merges the adapter first and points vLLM at the merged directory —
    see :func:`merge_and_unload`.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model, tokenizer


def merge_and_unload(
    adapter_path: str | Path,
    base_model_name: str,
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """Merge a LoRA adapter into base weights and save a flat HF model.

    Returns the directory containing the merged model. vLLM's eval path
    points directly at this directory.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(output_path)
    if out.exists():
        if not overwrite:
            _log.info("Merged model already at %s; reusing.", out)
            return out
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    _log.info(
        "Merging adapter %s into base %s → %s",
        adapter_path,
        base_model_name,
        out,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        # No flash_attention_2 here — we're not running forward, just merging weights.
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_path))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))

    # Free GPU memory before the eval path spins up vLLM.
    del peft_model, merged, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _log.info("Merged model saved to %s", out)
    return out


__all__ = [
    "save_lora_checkpoint",
    "load_lora_for_inference",
    "merge_and_unload",
]
