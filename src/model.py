"""Model + tokenizer builders.

Three paths:

1. **Unsloth training path** (``use_unsloth=True``, train config): fastest
   on a single H100 at long context. Uses ``FastLanguageModel`` and its
   patched LoRA wrapper.

2. **Vanilla HF training path** (``use_unsloth=False``): standard
   ``AutoModelForCausalLM`` + ``peft.get_peft_model`` using default
   Transformers attention kernels (no FlashAttention install required).

3. **vLLM inference engine** (``build_vllm_engine``): used by the eval
   harness. We do NOT load LoRA into vLLM; the eval pipeline merges the
   adapter to a flat HF model first (via ``checkpoint.merge_and_unload``)
   and points vLLM at that directory. Empirically more reliable at very
   long context than vLLM's adapter loading.

Llama 3.1 supports 128k context natively, so we never apply RoPE
scaling — but we assert/warn if the requested context length exceeds
that, since silent extrapolation past the trained context window is a
common foot-gun.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Tuple

from .config import EvalConfig, TrainConfig

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


_LLAMA31_NATIVE_CONTEXT = 131072
_log = logging.getLogger("model")


def _check_context_length(model_name: str, max_seq_len: int) -> None:
    """Warn (not error) if the requested context exceeds Llama 3.1's native 128k."""
    if "llama-3.1" in model_name.lower() and max_seq_len > _LLAMA31_NATIVE_CONTEXT:
        warnings.warn(
            f"max_seq_len={max_seq_len} exceeds Llama 3.1's native 128k context "
            f"({_LLAMA31_NATIVE_CONTEXT}). Behaviour beyond this is extrapolation; "
            "consider explicit RoPE scaling.",
            stacklevel=2,
        )


def _build_train_unsloth(
    config: TrainConfig,
) -> Tuple["PreTrainedModel", "PreTrainedTokenizerBase"]:
    from unsloth import FastLanguageModel  # type: ignore

    load_in_4bit = config.quant_mode == "4bit"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_len,
        dtype=None,  # let Unsloth choose bf16 on H100
        load_in_4bit=load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        max_seq_length=config.max_seq_len,
    )
    return model, tokenizer


def _build_train_hf(
    config: TrainConfig,
) -> Tuple["PreTrainedModel", "PreTrainedTokenizerBase"]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quantization_config = None
    if config.quant_mode == "4bit":
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, use_fast=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    if config.quant_mode == "4bit":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    else:
        model.gradient_checkpointing_enable()

    peft_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    return model, tokenizer


def build_model_and_tokenizer(
    config: TrainConfig | EvalConfig,
) -> Tuple["PreTrainedModel", "PreTrainedTokenizerBase"]:
    """Build a (model, tokenizer) pair for training or HF-based inference.

    For vLLM-based eval, use :func:`build_vllm_engine` instead — this
    function returns an HF model and is meant for either training or
    diagnostic / capability eval flows that need a regular HF model.
    """
    if isinstance(config, TrainConfig):
        _check_context_length(config.model_name, config.max_seq_len)
        if config.use_unsloth:
            _log.info("Building train model via Unsloth fast-path")
            return _build_train_unsloth(config)
        _log.info("Building train model via vanilla HF kernels")
        return _build_train_hf(config)

    # EvalConfig: build a plain HF model (used by capability eval / debug paths).
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _check_context_length(config.base_model_name, config.inference_max_seq_len)

    model_path = config.checkpoint_path or config.base_model_name
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def build_vllm_engine(config: EvalConfig, model_path: str) -> Any:
    """Build a vLLM ``LLM`` engine for evaluation.

    ``model_path`` should be either:
      * an HF repo id (for baseline runs), or
      * a local directory holding a fully-merged HF model (for fine-tuned
        runs — the eval pipeline merges LoRA adapters first via
        :func:`src.checkpoint.merge_and_unload`).
    """
    import os

    from vllm import LLM  # type: ignore

    # Colab / Jupyter compatibility:
    # - vLLM v1 engine fails in notebook environments because its coordinator
    #   process conflicts with the already-initialised CUDA context. Force v0.
    # - v0 also requires 'spawn' when CUDA is pre-initialised.
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    _check_context_length(config.base_model_name, config.inference_max_seq_len)

    _log.info(
        "Building vLLM engine: model=%s max_model_len=%d gpu_util=%.2f prefix_cache=%s",
        model_path,
        config.inference_max_seq_len,
        config.gpu_memory_utilization,
        config.enable_prefix_caching,
    )

    return LLM(
        model=model_path,
        max_model_len=config.inference_max_seq_len,
        dtype="bfloat16",
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=config.enable_prefix_caching,
        enforce_eager=True,       # avoids CUDA graph capture issues in Colab
        trust_remote_code=False,
    )


__all__ = ["build_model_and_tokenizer", "build_vllm_engine"]
