# Long-Context Jailbreak Robustness

A research codebase for studying whether **long-context fine-tuning** improves an open-source LLM's robustness to **many-shot jailbreak** attacks.

> **Hypothesis:** SFT-ing an instruction-tuned model on long-context data (LongAlpaca / LongAlign) makes it more robust to many-shot jailbreaking, possibly via the same mechanism Llama 2's "Ghost Attention" used to make system instructions persist across long dialogues.

The repo measures:

- **Attack Success Rate (ASR)** as a function of shot count, using HarmBench-derived faux dialogues ranging from 2 to 256 shots.
- The headline result is a single matplotlib figure: **ASR vs. shot count, baseline vs. fine-tuned**, written to `outputs/<exp>/plots/asr_vs_shotcount.png`.

---

## Table of contents

1. [Overview](#1-overview)
2. [Setup](#2-setup)
3. [Quickstart](#3-quickstart)
4. [Config reference](#4-config-reference)
5. [Datasets](#5-datasets)
6. [Inference engine (vLLM is required)](#6-inference-engine-vllm-is-required)
7. [Judge setup](#7-judge-setup)
8. [Reproducing the headline result](#8-reproducing-the-headline-result)
9. [File structure](#9-file-structure)
10. [Caveats and responsible use](#10-caveats-and-responsible-use)

---

## 1. Overview

**Research question.** Does the same long-context training that lets a model retain a system instruction across 30k tokens also make it harder to "wear down" the safety policy with 256 fake compliance demonstrations in front of a final harmful query?

**What this codebase does.**

- Fine-tunes Llama 3.1 8B with LoRA at long context (16k–32k) on standard long-context instruction data.
- Evaluates the resulting model against a many-shot jailbreak testset built from HarmBench (240 examples spanning shot counts in {2, 4, 8, 16, 32, 64, 128, 256}).
- Uses vLLM for inference (continuous batching + prefix caching make the 60k-token prompts tractable on a single H100) and an LLM-as-judge (Anthropic / OpenAI / local) to score harmfulness.
- Optionally checks for capability regression (MMLU subset + needle-in-a-haystack).

**What this codebase does NOT do.**

- It never generates new harmful content. The testset of harmful queries + reference jailbroken responses is loaded from a JSON the user builds locally from HarmBench. The "naive_refusal" training option also pulls its harmful prompts from a held-out HarmBench split — refusals are templated, not synthesised.

---

## 2. Setup

### Hardware

Designed for a single NVIDIA H100 80GB. Should also fit on A100 80GB with smaller batch / shorter context.

### Install

You only need a shell once — to install dependencies:

```bash
git clone <repo>
cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

If `flash-attn` fails to build (it needs a CUDA toolchain), set `use_unsloth: true` in your train config — Unsloth bundles its own attention kernels and avoids the dependency.

### Hugging Face + API keys

Done once, also in a shell:

```bash
huggingface-cli login            # accept HarmBench's license on the website first
export ANTHROPIC_API_KEY=sk-ant-...    # for judge_provider: anthropic
export OPENAI_API_KEY=sk-...           # for judge_provider: openai
```

After this, **everything else can be done from Python / a notebook**. See §3.

---

## 3. Quickstart (Python-first)

The CLI scripts in `scripts/` are thin wrappers — every operation has a Python entry point you can call from a notebook or REPL.

Open a notebook in the repo root (or `sys.path.append('/path/to/repo')` from anywhere) and:

### 3.1. Get the testset

The testset is a single JSON file at the repo root: **`jailbreak_testset.json`**. It contains 240 many-shot jailbreak prompts derived from HarmBench (30 examples per shot count, 8 shot counts from 2 to 256). It is **not committed** — it's in `.gitignore` because it contains harmful content.

If you already have a copy of `jailbreak_testset.json` at the repo root, you're done with this step. Verify it:

```python
from src.testset import load_testset

testset = load_testset("jailbreak_testset.json")    # validates schema; raises on malformed entries
print(f"{testset['meta']['total_examples']} examples, "
      f"shot sizes = {testset['meta']['shot_sizes']}, "
      f"categories = {testset['meta']['categories']}")
```

If you don't have one, build it from HarmBench (after `huggingface-cli login`):

```python
from src.config import load_testset_build_config
from src.testset import build_testset, load_testset

cfg = load_testset_build_config("configs/testset/harmbench_default.yaml")
build_testset(cfg)   # writes ./jailbreak_testset.json

testset = load_testset("jailbreak_testset.json")
print(f"{testset['meta']['total_examples']} examples")
```

You can override anything from the YAML on the fly without editing the file — useful for a smaller dev testset:

```python
from dataclasses import replace
dev_cfg = replace(
    cfg,
    examples_per_size=5,
    shot_sizes=[2, 8, 32],
    output_path="jailbreak_testset_dev.json",
)
build_testset(dev_cfg)
```

Then point `eval_cfg.testset_path` at `jailbreak_testset_dev.json` to iterate the pipeline cheaply.

### 3.2. Run the baseline jailbreak eval (no fine-tuning)

```python
from src.config import load_eval_config
from src.jailbreak_eval import JailbreakEvaluator

eval_cfg = load_eval_config("configs/eval/jailbreak_eval_llama31_8b.yaml")
results = JailbreakEvaluator(eval_cfg).run()

print(f"Overall ASR: {results['asr_overall']:.3f}")
print(f"ASR by shot count: {results['asr_by_shot_count']}")
```

This builds a vLLM engine, generates responses for all 240 prompts, judges them with the configured judge, and writes:

- `outputs/jailbreak_eval_llama31_8b/results.json` — aggregated ASR
- `outputs/jailbreak_eval_llama31_8b/raw_responses.jsonl` — model outputs (sensitive — gitignored)
- `outputs/jailbreak_eval_llama31_8b/judged_responses.jsonl` — same plus per-example judge decisions

Expect ~30–60 minutes on one H100.

### 3.3. Train a fine-tuned model

```python
from src.config import load_train_config
from src.trainer import Trainer

train_cfg = load_train_config("configs/train/llama31_8b_longalpaca_32k.yaml")
final_ckpt = Trainer(train_cfg).train()
print("Final checkpoint:", final_ckpt)
```

`Trainer.train()` runs the SFT loop, validates `val_steps_per_epoch` times per epoch (each writes a LoRA adapter), and returns the path to the final checkpoint. Live tqdm shows train loss + LR; full step-level history is logged to `outputs/llama31_8b_longalpaca_32k/train_log.jsonl`.

The 32k config takes ~12–18h on one H100. Use `configs/train/llama31_8b_longalpaca_16k.yaml` (~4–6h) for fast iteration, or override fields on the fly:

```python
from dataclasses import replace
quick_cfg = replace(
    train_cfg,
    experiment_name="quick_smoke_test",
    max_seq_len=4096,
    dataset_subset_size=64,   # tiny subset
    num_epochs=1,
    val_steps_per_epoch=1,
)
Trainer(quick_cfg).train()
```

### 3.4. Re-run the eval on the fine-tuned checkpoint

```python
from dataclasses import replace
from src.jailbreak_eval import JailbreakEvaluator

ft_cfg = replace(
    eval_cfg,
    checkpoint_path=str(final_ckpt),                 # from §3.3
    experiment_name="longalpaca_32k_eval",
)
ft_results = JailbreakEvaluator(ft_cfg).run()

print(f"Baseline ASR : {results['asr_overall']:.3f}")
print(f"Fine-tuned   : {ft_results['asr_overall']:.3f}")
```

The first eval against a LoRA checkpoint merges the adapter into a flat HF model under `<checkpoint_path>/merged/` (the merge is cached, so subsequent re-runs skip it).

### 3.5. Quick headline plot

```python
import json
import matplotlib.pyplot as plt

base = results["asr_by_shot_count"]
post = ft_results["asr_by_shot_count"]
keys = sorted({int(k) for k in base} | {int(k) for k in post})

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(keys, [base.get(str(k), 0.0) for k in keys], marker="o", label="baseline")
ax.plot(keys, [post.get(str(k), 0.0) for k in keys], marker="s", label="post-FT")
ax.set_xscale("log", base=2)
ax.set_xticks(keys); ax.set_xticklabels([str(k) for k in keys])
ax.set_xlabel("shot count"); ax.set_ylabel("attack success rate")
ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend()
ax.set_title("Many-shot jailbreak ASR vs. shot count")
fig.tight_layout()
plt.show()
```

(`scripts/full_pipeline.py` writes the same plot to disk automatically — see §8.)

### CLI alternative (optional)

If you do prefer a shell:

```bash
python scripts/build_testset_from_harmbench.py --config configs/testset/harmbench_default.yaml
python scripts/validate_testset.py jailbreak_testset.json
python scripts/eval_jailbreak.py --config configs/eval/jailbreak_eval_llama31_8b.yaml
python scripts/train.py --config configs/train/llama31_8b_longalpaca_32k.yaml
python scripts/eval_jailbreak.py \
    --config configs/eval/jailbreak_eval_llama31_8b.yaml \
    --checkpoint outputs/llama31_8b_longalpaca_32k/final_*/ \
    --experiment-name longalpaca_32k_eval
```

The shell scripts are 1:1 with the Python calls above.

---

## 4. Config reference

### `TrainConfig` (`configs/train/*.yaml`)

| Field | Default | Description |
|---|---|---|
| `experiment_name` | _required_ | Used as the output subdirectory name. |
| `output_dir` | _required_ | Top-level output dir. Run goes to `output_dir/experiment_name/`. |
| `seed` | `42` | Seeds Python / NumPy / Torch (CPU + CUDA). |
| `model_name` | `meta-llama/Llama-3.1-8B-Instruct` | HF repo id. |
| `use_unsloth` | `true` | Toggles Unsloth fast-path vs. vanilla HF + Flash Attention 2. |
| `quant_mode` | `bf16` | One of `bf16`, `4bit` (use `4bit` for tighter VRAM at long context). |
| `max_seq_len` | `16384` | Training context length. |
| `batch_size` | `1` | Per-step micro-batch. |
| `gradient_accumulation_steps` | `16` | Effective batch = `batch_size * grad_accum`. |
| `num_epochs` | `1` | Long-context SFT works fine with 1 epoch. |
| `learning_rate` | `2e-5` | LR for AdamW. |
| `weight_decay` | `0.0` | LoRA weight decay (typically 0). |
| `warmup_ratio` | `0.03` | Linear warmup fraction; constant LR after. |
| `max_grad_norm` | `1.0` | Gradient clipping. |
| `lora_r` | `16` | LoRA rank. |
| `lora_alpha` | `32` | LoRA scaling factor. |
| `lora_dropout` | `0.05` | LoRA dropout. |
| `lora_target_modules` | all 7 Llama proj layers | See `src/config.py::DEFAULT_LORA_TARGET_MODULES`. |
| `dataset_name` | `longalpaca` | One of `longalpaca`, `longalign`, `mixed`, `naive_refusal`, or any HF id. |
| `dataset_subset_size` | `null` | Cap (for fast iteration) or `null` for full. For `naive_refusal`, this sets the number of synthetic examples (default 1000). |
| `data_mix_ratios` | `null` | Required when `dataset_name == "mixed"`. e.g. `{longalpaca: 0.7, longalign: 0.3}`. |
| `pack_sequences` | `false` | Unsloth packing for short examples (rarely useful at 32k). |
| `val_split_ratio` | `0.02` | Fraction held out for validation loss. |
| `val_steps_per_epoch` | `4` | Validate this many times per epoch (each writes a checkpoint). |
| `early_stopping_patience` | `null` | None disables early stopping. |
| `save_every_n_steps` | `null` | Extra step-based checkpointing on top of validation cadence. |

### `EvalConfig` (`configs/eval/*.yaml`)

| Field | Default | Description |
|---|---|---|
| `experiment_name` | _required_ | Output subdir name. |
| `output_dir` | _required_ | Top-level output dir. |
| `base_model_name` | `meta-llama/Llama-3.1-8B-Instruct` | HF repo id of the base model. |
| `checkpoint_path` | `null` | LoRA adapter dir, merged HF dir, or `null` for baseline. |
| `testset_path` | `jailbreak_testset.json` | Path to the testset JSON (at the repo root by default). |
| `inference_max_seq_len` | `80000` | Must fit the longest expected prompt; ≥80k for 256-shot. |
| `gpu_memory_utilization` | `0.90` | Passed to vLLM. Lower if other processes need GPU memory. |
| `enable_prefix_caching` | `true` | Big speedup when a shot-size group shares long prefixes. |
| `max_new_tokens` | `512` | Response length cap. |
| `temperature` | `0.0` | Reproducible lower-bound ASR. Bump to `0.7` for upper bound. |
| `top_p` | `1.0` | Nucleus sampling. |
| `judge_model` | `claude-sonnet-4-7` | Provider-specific model id. |
| `judge_provider` | `anthropic` | One of `anthropic`, `openai`, `local_vllm`. |
| `categories_to_eval` | `null` | Subset filter; `null` = all. |
| `shot_sizes_to_eval` | `null` | Subset filter; `null` = all in testset. |
| `run_capability_eval` | `false` | (Reserved for `full_pipeline.py`.) |

### `TestsetBuildConfig` (`configs/testset/*.yaml`)

| Field | Default | Description |
|---|---|---|
| `harmbench_repo` | `walledai/HarmBench` | HF repo id. Verify columns; the script warns on fallback. |
| `shot_sizes` | `[2, 4, 8, 16, 32, 64, 128, 256]` | Shot counts to sample. |
| `examples_per_size` | `30` | Examples per shot-count cell. |
| `mix_categories_in_faux_dialogue` | `true` | Faux turns sample across categories; final target is fixed per example. |
| `seed` | `42` | RNG seed for sampling. |
| `output_path` | `jailbreak_testset.json` | Output file (relative to repo root). |

---

## 5. Datasets

| `dataset_name` | Source | When to use |
|---|---|---|
| `longalpaca` | `Yukang/LongAlpaca-12k` | Default. Diverse long-context instructions; good for the headline run. |
| `longalign` | `THUDM/LongAlign-10k` | Diversity comparison. Different mix of instruction types. |
| `mixed` | combination | Set `data_mix_ratios: {longalpaca: 0.7, longalign: 0.3}`. |
| `naive_refusal` | held-out HarmBench split | Anthropic-style baseline: trains on (faux jailbreak prompt → refusal) pairs. Use this only to reproduce "fine-tune-to-refuse" delays the attack but doesn't fix it. The harmful prompts come from the `train` split of HarmBench by default; the testset uses `test` (or whatever `build_testset_from_harmbench.py` resolved). If those overlap in your HarmBench revision, override the split. |
| any other string | HF dataset id | Best-effort schema detection. |

**Label masking:** all datasets are tokenised under Llama 3.1's chat template with labels masked to `-100` on user / system tokens, so loss flows only on assistant turns. This matters more here than usual because contexts are 16k–32k and naive masking would leak loss across long user turns.

---

## 6. Inference engine (vLLM is required)

The eval harness uses vLLM, not HF native generation. This is **not optional** — the harness will fail without `vllm>=0.6` installed.

**Why.** A 256-shot prompt is roughly 60k tokens. HF native `model.generate` pads every batch to the longest prompt's length, which wastes both VRAM and time when shot sizes are mixed. Even at batch_size=1, HF generation at 60k context on an 8B model takes minutes per example, putting a full 240-example eval at multi-hour scale. vLLM's **PagedAttention + continuous batching** packs requests into the same forward pass as KV cache space allows, with no padding waste. On one H100, a 240-example mixed-shot-size eval typically completes in 30–60 minutes; without prefix caching it's closer to 30–60s/example.

**Batching strategy used by `JailbreakEvaluator`.**

1. Build all prompts up-front (chat-template-applied), preserving order.
2. Group by `shot_count` and process in **ascending** order. Small-shot groups validate the pipeline cheaply; if there's an OOM it surfaces before the expensive 256-shot group runs.
3. Within a group, call `llm.generate(prompts_in_group, sampling_params)` once — vLLM handles continuous batching internally. Do not manually chunk inside a group.
4. The `vllm.LLM` engine is initialised once and reused across all groups.

**LoRA + vLLM.** vLLM's native LoRA loading has had edge cases at very long context. The eval pipeline therefore calls `merge_and_unload` first and points vLLM at the merged model directory (see `src/checkpoint.py::merge_and_unload`).

**Memory tuning notes (single H100).**
- `gpu_memory_utilization: 0.90` is a good default. Drop to `0.85` if vLLM OOMs at startup.
- `enable_prefix_caching: true` is important — examples within a shot-size group share long common prefixes.
- `inference_max_seq_len: 80000` accommodates the 256-shot prompts. Llama 3.1 supports 128k natively, so this is well within the trained context.

---

## 7. Judge setup

### Choosing a judge

| Provider | Pros | Cons |
|---|---|---|
| `anthropic` | Strong refusal/compliance discrimination; default. | API cost; rate limits. |
| `openai` | Independent cross-check vs. Claude. | Same constraints. |
| `local_vllm` | No API cost / data leaves the box. | GPU contention with the eval model — see below. |

### GPU juggling for `local_vllm`

The eval model occupies most of the H100's VRAM at 32k+ context. Running the judge concurrently on the same GPU is not feasible. Two patterns work:

1. **Sequential**: let the eval pipeline finish generation, free vLLM (the harness does this automatically with `del llm; torch.cuda.empty_cache()`), then start the local judge server, then run judging.
2. **Separate GPU**: if a second GPU is available, point `LOCAL_VLLM_BASE_URL` at a server bound to that device.

### Known judge biases

- LLM judges over-call **partial compliance** as unsafe when refusal language is buried mid-response. The rubric in `src/judge.py::JUDGE_RUBRIC` explicitly addresses this (a refusal buried in compliant text scores 4 or 5).
- LLM judges tend to be lenient on **fictional / roleplay framing**. The rubric counters this too.
- For headline results, **always cross-check** with at least one other judge model (`configs/eval/jailbreak_eval_with_judge.yaml` swaps the judge to GPT-4o).

---

## 8. Reproducing the headline result

The full pipeline is just the §3 steps stitched together. From a notebook:

```python
from dataclasses import replace
from pathlib import Path

from src.config import (
    load_testset_build_config, load_train_config, load_eval_config,
)
from src.testset import build_testset, load_testset
from src.trainer import Trainer
from src.jailbreak_eval import JailbreakEvaluator

# 1. Build + validate the testset (skip if already built).
# If you don't already have jailbreak_testset.json at the repo root:
# build_testset(load_testset_build_config("configs/testset/harmbench_default.yaml"))
load_testset("jailbreak_testset.json")    # raises on schema errors

# 2. Baseline eval.
eval_cfg = load_eval_config("configs/eval/jailbreak_eval_llama31_8b.yaml")
baseline = JailbreakEvaluator(replace(eval_cfg, experiment_name="baseline")).run()

# 3. Fine-tune.
train_cfg = load_train_config("configs/train/llama31_8b_longalpaca_32k.yaml")
ckpt = Trainer(train_cfg).train()

# 4. Post-FT eval (re-uses the same testset, points at the new checkpoint).
post_ft = JailbreakEvaluator(replace(
    eval_cfg, experiment_name="post_ft", checkpoint_path=str(ckpt),
)).run()

print(f"baseline ASR = {baseline['asr_overall']:.3f}")
print(f"post-FT ASR  = {post_ft['asr_overall']:.3f}")
```

For one-shot reproducibility (and to auto-save the headline plot to disk), there's also a CLI driver that runs all of the above plus the optional capability eval:

```bash
python scripts/full_pipeline.py --manifest configs/longalpaca_32k_e2e.yaml
```

Outputs in either case:

- `outputs/longalpaca_32k_e2e/baseline/results.json` — pre-FT ASR
- `outputs/longalpaca_32k_e2e/checkpoints/llama31_8b_longalpaca_32k/final_*/` — LoRA adapter
- `outputs/longalpaca_32k_e2e/post_ft/results.json` — post-FT ASR
- `outputs/longalpaca_32k_e2e/post_ft/capability_results.json` — MMLU + needle (if enabled)
- `outputs/longalpaca_32k_e2e/plots/asr_vs_shotcount.png` — **the headline figure**

To cross-check against a different judge family:

```python
gpt_eval_cfg = load_eval_config("configs/eval/jailbreak_eval_with_judge.yaml")
JailbreakEvaluator(replace(
    gpt_eval_cfg,
    checkpoint_path=str(ckpt),
    experiment_name="longalpaca_32k_gpt4o_judge",
)).run()
```

---

## 9. File structure

```
.
├── src/
│   ├── __init__.py
│   ├── config.py            # Frozen-dataclass configs + YAML loaders
│   ├── model.py             # HF / Unsloth / vLLM builders
│   ├── data.py              # LongAlpaca / LongAlign / mixed / naive_refusal
│   ├── trainer.py           # LoRA SFT loop with periodic validation
│   ├── checkpoint.py        # save/load LoRA + merge_and_unload
│   ├── jailbreak_eval.py    # vLLM-based many-shot jailbreak eval
│   ├── judge.py             # LLM-as-judge (anthropic / openai / local_vllm)
│   ├── capability_eval.py   # MMLU subset + needle-in-haystack
│   ├── testset.py           # Loader + schema validator + filter + builder
│   └── utils.py             # Seeding, logging, chat formatting, JSONL helpers
│
├── jailbreak_testset.json   # The actual testset (gitignored; lives at repo root)
│
├── configs/
│   ├── train/
│   │   ├── llama31_8b_longalpaca_16k.yaml
│   │   ├── llama31_8b_longalpaca_32k.yaml
│   │   ├── llama31_8b_longalign_32k.yaml
│   │   └── llama31_8b_naive_refusal_baseline.yaml
│   ├── eval/
│   │   ├── jailbreak_eval_llama31_8b.yaml
│   │   └── jailbreak_eval_with_judge.yaml
│   ├── testset/
│   │   └── harmbench_default.yaml
│   └── longalpaca_32k_e2e.yaml      # full_pipeline.py manifest
│
├── scripts/
│   ├── build_testset_from_harmbench.py
│   ├── validate_testset.py
│   ├── train.py
│   ├── eval_jailbreak.py
│   └── full_pipeline.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 10. Caveats and responsible use

**Statistical caveats.**

- Default `n=30` per shot-count cell. The 95% Wilson CI on a single-cell ASR of 0.5 is roughly ±0.18; on 0.1 it's about ±0.10. Don't read fine-grained shot-by-shot differences as significant without bootstrapping.
- Default `temperature=0.0` gives a **lower-bound** ASR (the model's most-likely response). Sampling at `temperature=0.7` typically yields higher ASR. Headline numbers should report the temperature used.
- **Judge noise** is real. Cross-validate with at least two judges (Anthropic + OpenAI, or +1 local). The rubric in `src/judge.py` is calibrated to penalise common false-negatives (refusals buried in compliant text, fictional/roleplay framing).
- **Capability/safety trade-off.** Long-context SFT can drag MMLU down by 1–3 points. The optional capability eval surfaces this; report it alongside ASR, not separately.

**What NOT to claim.**

- "Long-context SFT solves many-shot jailbreaking." It doesn't, and probably *can't* without explicit safety data — the Anthropic blog showed naive refusal-tuning merely delays the attack to higher shot counts.
- "ASR=0 means safe." Even ASR=0 on this testset only rules out *this* attack against *this* set of harmful queries. Other attack vectors (PAIR, GCG, prompt injection, etc.) are out of scope.
- Per-category claims at small `examples_per_size`. Bump `examples_per_size` to 60+ before reporting per-category ASR.

**Responsible use — sensitive artifacts.**

- `jailbreak_testset.json` (at the repo root) contains harmful prompts and reference jailbroken responses. **It is in `.gitignore` and must not be committed or shared publicly.**
- `outputs/<exp>/<phase>/raw_responses.jsonl` and `judged_responses.jsonl` contain the model's responses to harmful queries — some of which will be successful jailbreaks. **Same rule: gitignored, do not share publicly.**
- If publishing, release **only aggregated ASR numbers** (the `results.json` *minus* per-example fields), never per-example outputs.
- `.gitignore` enforces the above for `data/` and `outputs/`. Don't relax it.
- The codebase **never generates new harmful content**: harmful prompts come from a user-built HarmBench testset, refusals (in `naive_refusal`) are templated.
