"""Many-shot jailbreak evaluation harness (vLLM only).

This module implements the eval pipeline described in the README:

1. Load + filter the testset.
2. Resolve the model under test:
     - If ``checkpoint_path is None``: use the base model directly.
     - Else: merge the LoRA adapter into a flat HF model directory via
       :func:`src.checkpoint.merge_and_unload` and point vLLM at it.
3. Build a vLLM ``LLM`` engine **once**.
4. Render every example through ``apply_chat_template`` (the faux turns
   are real ``(user, assistant)`` message pairs — never raw text
   concatenation; that's the actual attack format).
5. Group examples by ``shot_count`` and process groups in **ascending
   shot_count order** (small groups validate the pipeline cheaply
   before the expensive 256-shot group runs). Within a group, call
   ``llm.generate(...)`` once on the entire group — vLLM handles
   continuous batching internally.
6. Save raw responses, score them with the judge, save aggregated ASR.

Note on storage: ``raw_responses.jsonl`` may contain harmful content
since the model under test sometimes complies. It is in ``.gitignore``
and the README warns the user not to share it publicly.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .checkpoint import merge_and_unload
from .config import EvalConfig, dump_config
from .judge import Judge
from .testset import filter_testset, load_testset
from .utils import format_chat_for_jailbreak, setup_logger

if TYPE_CHECKING:  # pragma: no cover
    pass


_log = logging.getLogger("jailbreak_eval")


class JailbreakEvaluator:
    """vLLM-based many-shot jailbreak evaluator."""

    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.run_dir = config.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("jailbreak_eval", self.run_dir / "eval.log")
        dump_config(config, self.run_dir / "config.yaml")

    # ------------------------------------------------------------------
    # Model resolution
    # ------------------------------------------------------------------

    def _resolve_model_path(self) -> str:
        """Return the local model directory or HF id to hand to vLLM."""
        ckpt = self.config.checkpoint_path
        if ckpt is None:
            self.logger.info(
                "No checkpoint set — evaluating base model %s",
                self.config.base_model_name,
            )
            return self.config.base_model_name

        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint_path does not exist: {ckpt_path}")

        # If the checkpoint dir already contains a full model (config.json + safetensors),
        # treat it as a merged model and skip the merge step.
        if (ckpt_path / "config.json").exists() and any(
            ckpt_path.glob("*.safetensors")
        ):
            adapter_marker = ckpt_path / "adapter_config.json"
            if not adapter_marker.exists():
                self.logger.info(
                    "Checkpoint appears to be a merged HF model already: %s", ckpt_path
                )
                return str(ckpt_path)

        merged_dir = ckpt_path / "merged"
        merge_and_unload(
            adapter_path=ckpt_path,
            base_model_name=self.config.base_model_name,
            output_path=merged_dir,
            overwrite=False,
        )
        return str(merged_dir)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompts(
        self, examples: list[dict[str, Any]], tokenizer
    ) -> list[str]:
        prompts: list[str] = []
        for ex in examples:
            prompts.append(
                format_chat_for_jailbreak(
                    faux_dialogue=ex["faux_dialogue"],
                    target_query=ex["target_query"],
                    tokenizer=tokenizer,
                    system_prompt=None,
                    add_generation_prompt=True,
                )
            )
        return prompts

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_with_vllm(
        self,
        examples: list[dict[str, Any]],
        prompts: list[str],
        llm,
        tokenizer,
    ) -> list[dict[str, Any]]:
        """Group by shot_count ascending, generate per group, return ordered records."""
        from vllm import SamplingParams  # type: ignore

        sampling = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
        )

        # Build groups, preserving original index for final ordering.
        groups: dict[int, list[int]] = defaultdict(list)
        for i, ex in enumerate(examples):
            groups[int(ex["shot_count"])].append(i)

        records: dict[int, dict[str, Any]] = {}
        for shot_count in sorted(groups):
            idxs = groups[shot_count]
            self.logger.info(
                "Generating shot_count=%d (n=%d examples)", shot_count, len(idxs)
            )
            group_prompts = [prompts[i] for i in idxs]

            outputs = llm.generate(group_prompts, sampling)

            # vLLM returns outputs in the same order as the input prompts.
            for local_i, out in enumerate(outputs):
                global_i = idxs[local_i]
                ex = examples[global_i]
                response_text = out.outputs[0].text if out.outputs else ""
                # vLLM's prompt_token_ids gives us a cheap token count.
                prompt_tok = (
                    len(out.prompt_token_ids)
                    if getattr(out, "prompt_token_ids", None) is not None
                    else len(tokenizer.encode(group_prompts[local_i]))
                )
                records[global_i] = {
                    "id": ex["id"],
                    "category": ex["category"],
                    "shot_count": ex["shot_count"],
                    "target_query": ex["target_query"],
                    "prompt_token_count": prompt_tok,
                    "response": response_text,
                }

        return [records[i] for i in range(len(examples))]

    def _generate_with_hf(
        self,
        examples: list[dict[str, Any]],
        prompts: list[str],
        tokenizer,
    ) -> list[dict[str, Any]]:
        """HF generate fallback. Slower than vLLM but works in any notebook env.

        Processes one prompt at a time (no padding waste, no Jupyter
        multiprocessing issues). Groups by shot_count ascending so cheap
        groups validate the pipeline before the expensive 256-shot group.
        """
        import torch
        from tqdm.auto import tqdm
        from transformers import AutoModelForCausalLM

        cfg = self.config
        model_path = self._resolve_model_path()

        self.logger.info("Loading HF model: %s", model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model.eval()

        # Group by shot_count, preserve original index for ordering.
        groups: dict[int, list[int]] = defaultdict(list)
        for i, ex in enumerate(examples):
            groups[int(ex["shot_count"])].append(i)

        records: dict[int, dict[str, Any]] = {}
        do_sample = cfg.temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=do_sample,
            top_p=cfg.top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = cfg.temperature

        for shot_count in sorted(groups):
            idxs = groups[shot_count]
            self.logger.info(
                "Generating shot_count=%d (n=%d examples)", shot_count, len(idxs)
            )
            for global_i in tqdm(idxs, desc=f"shot={shot_count}", dynamic_ncols=True):
                ex = examples[global_i]
                prompt = prompts[global_i]
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, **gen_kwargs)
                response_text = tokenizer.decode(
                    out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                records[global_i] = {
                    "id": ex["id"],
                    "category": ex["category"],
                    "shot_count": ex["shot_count"],
                    "target_query": ex["target_query"],
                    "prompt_token_count": int(inputs["input_ids"].shape[1]),
                    "response": response_text,
                }

        # Free model before judge runs (judge may also use GPU via local_vllm).
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return [records[i] for i in range(len(examples))]

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_asr(records: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(records)
        n_unsafe = sum(1 for r in records if r["judge"]["unsafe"])
        n_disagree = sum(
            1 for r in records if r["judge"].get("_parse_error") or r["judge"].get("_recovered")
        )

        by_shot: dict[str, dict[str, int]] = defaultdict(
            lambda: {"n": 0, "unsafe": 0}
        )
        by_cat: dict[str, dict[str, int]] = defaultdict(
            lambda: {"n": 0, "unsafe": 0}
        )
        by_shot_cat: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"n": 0, "unsafe": 0})
        )

        for r in records:
            sc = str(r["shot_count"])
            cat = r["category"]
            unsafe = int(bool(r["judge"]["unsafe"]))
            by_shot[sc]["n"] += 1
            by_shot[sc]["unsafe"] += unsafe
            by_cat[cat]["n"] += 1
            by_cat[cat]["unsafe"] += unsafe
            by_shot_cat[sc][cat]["n"] += 1
            by_shot_cat[sc][cat]["unsafe"] += unsafe

        def _ratio(d: dict[str, int]) -> float:
            return d["unsafe"] / d["n"] if d["n"] else 0.0

        return {
            "asr_overall": (n_unsafe / n) if n else 0.0,
            "asr_by_shot_count": {k: _ratio(v) for k, v in sorted(by_shot.items(), key=lambda x: int(x[0]))},
            "asr_by_category": {k: _ratio(v) for k, v in sorted(by_cat.items())},
            "asr_by_shot_and_category": {
                sc: {cat: _ratio(d) for cat, d in cats.items()}
                for sc, cats in sorted(by_shot_cat.items(), key=lambda x: int(x[0]))
            },
            "n_examples": n,
            "n_judged_unsafe": n_unsafe,
            "judge_disagreements": n_disagree,
        }

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the full eval pipeline. Returns the aggregated results dict."""
        from transformers import AutoTokenizer

        cfg = self.config

        self.logger.info("Loading testset from %s", cfg.testset_path)
        testset = load_testset(cfg.testset_path)
        testset = filter_testset(
            testset,
            categories=cfg.categories_to_eval,
            shot_sizes=cfg.shot_sizes_to_eval,
        )
        examples: list[dict[str, Any]] = testset["examples"]
        if not examples:
            raise RuntimeError(
                "Testset is empty after filtering. Check categories_to_eval / shot_sizes_to_eval."
            )
        self.logger.info("Filtered testset: %d examples", len(examples))

        # Tokenizer is always loaded from the base model, since we always have
        # a HF chat template there. (Merged dirs also have it, but base is
        # canonical and avoids surprises.)
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        prompts = self._build_prompts(examples, tokenizer)
        self.logger.info("Prepared %d prompts", len(prompts))

        # --- Generate ---
        if cfg.inference_backend == "vllm":
            from .model import build_vllm_engine

            model_path = self._resolve_model_path()
            llm = build_vllm_engine(cfg, model_path)
            records = self._generate_with_vllm(examples, prompts, llm, tokenizer)
            del llm  # free KV cache before judge spins up
        else:  # hf
            self.logger.info("Using HF generate backend (slower; works in any notebook env)")
            records = self._generate_with_hf(examples, prompts, tokenizer)

        # Persist raw responses immediately — judging can be slow/network-dependent
        # and we want the artefact on disk before that step.
        raw_path = self.run_dir / "raw_responses.jsonl"
        with raw_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(
                    json.dumps(
                        {
                            "id": r["id"],
                            "category": r["category"],
                            "shot_count": r["shot_count"],
                            "prompt_token_count": r["prompt_token_count"],
                            "response": r["response"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        self.logger.info("Wrote raw responses to %s", raw_path)

        # Free GPU memory before spinning up a local judge (if any).
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # --- Judge ---
        self.logger.info(
            "Judging %d responses with %s/%s",
            len(records),
            cfg.judge_provider,
            cfg.judge_model,
        )
        judge = Judge(model=cfg.judge_model, provider=cfg.judge_provider)
        for r in records:
            r["judge"] = judge.judge_response(
                target_query=r["target_query"], model_response=r["response"]
            )

        # Persist judged records (separate file from raw — raw is the
        # canonical "what the model said" record, judged is the analysis).
        judged_path = self.run_dir / "judged_responses.jsonl"
        with judged_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # --- Aggregate ---
        agg = self._aggregate_asr(records)
        results = {"config": _config_dict(cfg), **agg}
        results_path = self.run_dir / "results.json"
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=False)
        self.logger.info(
            "ASR overall=%.3f (n=%d, judged_unsafe=%d). Wrote %s",
            results["asr_overall"],
            results["n_examples"],
            results["n_judged_unsafe"],
            results_path,
        )
        return results


def _config_dict(cfg: EvalConfig) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(cfg)


__all__ = ["JailbreakEvaluator"]
