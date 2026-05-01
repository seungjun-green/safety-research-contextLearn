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
    "DEFAULT_LORA_TARGET_MODULES",
]
