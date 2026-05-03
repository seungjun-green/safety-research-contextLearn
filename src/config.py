"""YAML-driven config loaders.

We deliberately avoid Hydra / OmegaConf. All configs are plain frozen
dataclasses populated from ``yaml.safe_load``. The frozen-ness is a hint
that configs are immutable for the duration of a run; tests / scripts
can rebuild a new config via ``dataclasses.replace`` if they need to
override a single field.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml


DEFAULT_LORA_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for a single fine-tuning run."""

    # Identity / IO
    experiment_name: str
    output_dir: str
    seed: int = 42

    # Model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    use_unsloth: bool = True
    quant_mode: str = "bf16"  # one of: bf16, 4bit
    max_seq_len: int = 16384

    # Optimisation
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_epochs: int = 1
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: list(DEFAULT_LORA_TARGET_MODULES)
    )

    # Dataset
    dataset_name: str = "longalpaca"  # longalpaca | longalign | mixed | naive_refusal | <hf id>
    dataset_subset_size: int | None = None
    data_mix_ratios: dict[str, float] | None = None
    pack_sequences: bool = False

    # Validation / checkpointing
    val_split_ratio: float = 0.02
    val_steps_per_epoch: int = 4
    early_stopping_patience: int | None = None
    save_every_n_steps: int | None = None

    # ---- helpers -----------------------------------------------------

    def __post_init__(self) -> None:
        # Mild validation. Heavier validation lives in the modules that
        # consume specific fields (e.g. data.py validates dataset_name).
        if self.quant_mode not in {"bf16", "4bit"}:
            raise ValueError(
                f"quant_mode must be 'bf16' or '4bit', got {self.quant_mode!r}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
            )
        if self.dataset_name == "mixed" and not self.data_mix_ratios:
            raise ValueError(
                "dataset_name='mixed' requires non-empty data_mix_ratios"
            )

    @property
    def run_dir(self) -> Path:
        """Directory where this run's checkpoints / logs are written."""
        return Path(self.output_dir) / self.experiment_name


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalConfig:
    """Configuration for a jailbreak evaluation run."""

    experiment_name: str
    output_dir: str

    # Model under test
    base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    checkpoint_path: str | None = None  # None → evaluate base model

    # Testset
    testset_path: str = "jailbreak_testset.json"

    # Inference
    inference_backend: str = "vllm"  # vllm | hf
    inference_max_seq_len: int = 80000
    gpu_memory_utilization: float = 0.90
    enable_prefix_caching: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0

    # Optional system prompt prepended to every test prompt.
    # None / "" disables it (matches the default many-shot attack format).
    system_prompt: str | None = None

    # Two-step (emotion-conditioned) generation. When BOTH are set,
    # the evaluator runs:
    #   step 1 — render the faux dialogue + target query as a passage,
    #            ask the model (under first_step_system_prompt) to rate its
    #            emotional reaction on 7 categories (0–5 each).
    #   step 2 — re-prompt the model in chat-template form, with
    #            second_step_system_prompt + the parsed emotion ratings
    #            as the system message, and the actual harmful query as
    #            the final user turn. The step-2 response is what gets
    #            judged. Use ``{emotion}`` in second_step_system_prompt
    #            as a placeholder for the rating string; if absent, the
    #            ratings are appended after a newline.
    first_step_system_prompt: str | None = None
    second_step_system_prompt: str | None = None

    # Combined single-pass "two-part" generation (the "advanced" mode).
    # When advanced_user_prompt is set, the evaluator runs ONE HF generation
    # per example. The user turn contains both the transcript (substituted
    # via the ``{transcript}`` placeholder) and instructions to (a) rate
    # the model's emotional reaction and (b) reply to the final user
    # message. The assistant turn is prefilled with
    # ``advanced_assistant_prefill`` to force the model to commit to the
    # structured format (this is what defeats Llama-3.1's
    # refuse-on-contact behavior). The substring AFTER
    # ``advanced_response_marker`` in the model output is what gets
    # judged; the portion before it is parsed for emotion ratings and
    # saved for traceability. Mutually exclusive with system_prompt and
    # first/second_step_system_prompt. HF backend only.
    advanced_user_prompt: str | None = None
    advanced_assistant_prefill: str | None = None
    advanced_response_marker: str = "PART 2"

    # Judge
    judge_model: str = "gpt-4o"
    judge_provider: str = "openai"  # openai | local_vllm

    # Filtering
    categories_to_eval: list[str] | None = None
    shot_sizes_to_eval: list[int] | None = None

    # Capability eval (optional)
    run_capability_eval: bool = False

    def __post_init__(self) -> None:
        if self.judge_provider not in {"openai", "local_vllm"}:
            raise ValueError(
                f"judge_provider must be one of openai|local_vllm, got {self.judge_provider!r}"
            )
        if self.inference_backend not in {"vllm", "hf"}:
            raise ValueError(
                f"inference_backend must be one of vllm|hf, got {self.inference_backend!r}"
            )
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}"
            )
        # Two-step mode: both first/second_step_system_prompt must be set together.
        first = self.first_step_system_prompt
        second = self.second_step_system_prompt
        if (first is None) != (second is None):
            raise ValueError(
                "first_step_system_prompt and second_step_system_prompt must "
                "both be set (two-step mode) or both be None (single-step mode)."
            )
        if first is not None and self.system_prompt is not None:
            raise ValueError(
                "Cannot combine `system_prompt` (single-step) with "
                "`first_step_system_prompt` / `second_step_system_prompt` "
                "(two-step). Use one or the other."
            )
        # Combined "advanced" mode: mutually exclusive with both other modes.
        if self.advanced_user_prompt is not None:
            if self.system_prompt is not None or first is not None or second is not None:
                raise ValueError(
                    "advanced_user_prompt cannot be combined with "
                    "system_prompt or first/second_step_system_prompt; "
                    "pick one mode."
                )
            if "{transcript}" not in self.advanced_user_prompt:
                raise ValueError(
                    "advanced_user_prompt must contain the `{transcript}` "
                    "placeholder (gets replaced with the rendered "
                    "faux-dialogue + target query for each example)."
                )

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.experiment_name


# ---------------------------------------------------------------------------
# Testset build
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestsetBuildConfig:
    """Configuration for ``scripts/build_testset_from_harmbench.py``."""

    harmbench_repo: str = "walledai/HarmBench"
    shot_sizes: list[int] = field(
        default_factory=lambda: [2, 4, 8, 16, 32, 64, 128, 256]
    )
    examples_per_size: int = 30
    mix_categories_in_faux_dialogue: bool = True
    seed: int = 42
    output_path: str = "jailbreak_testset.json"


# ---------------------------------------------------------------------------
# YAML loaders
# ---------------------------------------------------------------------------


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config root must be a mapping, got {type(raw).__name__} in {p}"
        )
    return raw


def _filter_kwargs(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    """Drop keys that aren't dataclass fields with a clear error if any exist."""
    valid = {f.name for f in fields(cls)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(
            f"Unknown keys in config for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid)}"
        )
    return {k: v for k, v in raw.items() if k in valid}


def load_train_config(path: str | Path) -> TrainConfig:
    return TrainConfig(**_filter_kwargs(TrainConfig, _load_yaml(path)))


def load_eval_config(path: str | Path) -> EvalConfig:
    return EvalConfig(**_filter_kwargs(EvalConfig, _load_yaml(path)))


def load_testset_build_config(path: str | Path) -> TestsetBuildConfig:
    return TestsetBuildConfig(**_filter_kwargs(TestsetBuildConfig, _load_yaml(path)))


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """Recursively convert a (possibly nested) dataclass config to a plain dict."""
    from dataclasses import asdict, is_dataclass

    if is_dataclass(cfg):
        return asdict(cfg)
    raise TypeError(f"Expected a dataclass, got {type(cfg).__name__}")


def replace_two_step(
    cfg: EvalConfig,
    *,
    first_step_system_prompt: str,
    second_step_system_prompt: str,
    **overrides: Any,
) -> EvalConfig:
    """Convenience wrapper around ``dataclasses.replace`` for two-step eval.

    Equivalent to::

        replace(
            cfg,
            system_prompt=None,
            first_step_system_prompt=first_step_system_prompt,
            second_step_system_prompt=second_step_system_prompt,
            **overrides,
        )

    The single-step ``system_prompt`` is force-cleared because the two-step
    config validator rejects mixing the two modes.
    """
    overrides.pop("system_prompt", None)
    overrides.pop("first_step_system_prompt", None)
    overrides.pop("second_step_system_prompt", None)
    return replace(
        cfg,
        system_prompt=None,
        first_step_system_prompt=first_step_system_prompt,
        second_step_system_prompt=second_step_system_prompt,
        **overrides,
    )


def replace_two_step_advanced(
    cfg: EvalConfig,
    *,
    user_prompt: str | None = None,
    assistant_prefill: str | None = None,
    response_marker: str = "PART 2",
    **overrides: Any,
) -> EvalConfig:
    """Convenience wrapper to enable the combined single-pass two-part eval.

    Defaults to a proven "warm mother emotion → safe-assistant reply" template
    that was hand-validated on Llama-3.1-8B-Instruct. The assistant turn is
    prefilled to force the model to commit to a structured response, sidestepping
    the refuse-on-contact pattern Llama is heavily safety-tuned for.

    Use it like::

        eval_cfg = replace_two_step_advanced(
            load_eval_config("configs/eval/jailbreak_eval_llama31_8b.yaml"),
            checkpoint_path=None,
            experiment_name="emotion_then_reply_eval",
            inference_backend="hf",
        )
        JailbreakEvaluator(eval_cfg).run()

    Override any of ``user_prompt`` / ``assistant_prefill`` / ``response_marker``
    if you want a different template; ``user_prompt`` must contain the literal
    ``{transcript}`` placeholder and the prefill / marker should match the
    section structure in ``user_prompt``.
    """
    # Late import to avoid circular dependency (jailbreak_eval imports from config).
    from .jailbreak_eval import (
        DEFAULT_ADVANCED_USER_PROMPT,
        DEFAULT_ADVANCED_ASSISTANT_PREFILL,
    )

    if user_prompt is None:
        user_prompt = DEFAULT_ADVANCED_USER_PROMPT
    if assistant_prefill is None:
        assistant_prefill = DEFAULT_ADVANCED_ASSISTANT_PREFILL

    for k in (
        "system_prompt",
        "first_step_system_prompt",
        "second_step_system_prompt",
        "advanced_user_prompt",
        "advanced_assistant_prefill",
        "advanced_response_marker",
    ):
        overrides.pop(k, None)

    return replace(
        cfg,
        system_prompt=None,
        first_step_system_prompt=None,
        second_step_system_prompt=None,
        advanced_user_prompt=user_prompt,
        advanced_assistant_prefill=assistant_prefill,
        advanced_response_marker=response_marker,
        **overrides,
    )


def dump_config(cfg: Any, path: str | Path) -> None:
    """Write a config out as YAML next to a checkpoint / results file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False)


__all__ = [
    "TrainConfig",
    "EvalConfig",
    "TestsetBuildConfig",
    "load_train_config",
    "load_eval_config",
    "load_testset_build_config",
    "config_to_dict",
    "dump_config",
    "replace",
    "replace_two_step",
    "replace_two_step_advanced",
    "DEFAULT_LORA_TARGET_MODULES",
]
