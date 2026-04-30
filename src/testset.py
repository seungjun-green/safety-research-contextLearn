"""Testset loader, schema validator, and filter.

Schema (defined in the README and produced by
``scripts/build_testset_from_harmbench.py``):

.. code-block:: json

    {
      "meta": {
        "source": "HarmBench",
        "shot_sizes": [2, 4, 8, 16, 32, 64, 128, 256],
        "examples_per_size": 30,
        "total_examples": 240,
        "categories": ["..."],
        "seed": 42
      },
      "examples": [
        {
          "id": "string-unique-id",
          "category": "string",
          "shot_count": 8,
          "faux_dialogue": [{"user": "...", "assistant": "..."}, ...],
          "target_query": "..."
        }
      ]
    }
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

from .config import TestsetBuildConfig


# Required top-level keys.
_REQUIRED_TOP_LEVEL = {"meta", "examples"}
_REQUIRED_META_KEYS = {
    "source",
    "shot_sizes",
    "examples_per_size",
    "total_examples",
    "categories",
    "seed",
}
_REQUIRED_EXAMPLE_KEYS = {
    "id",
    "category",
    "shot_count",
    "faux_dialogue",
    "target_query",
}


def validate_schema(testset: dict[str, Any]) -> list[str]:
    """Return a list of validation errors (empty list = valid).

    The validator is permissive about extra keys but strict about required
    ones, types, and uniqueness of example IDs.
    """
    errors: list[str] = []

    if not isinstance(testset, dict):
        return [f"Top-level testset must be a dict, got {type(testset).__name__}"]

    missing_top = _REQUIRED_TOP_LEVEL - set(testset)
    if missing_top:
        errors.append(f"Missing top-level keys: {sorted(missing_top)}")
        # Stop early: deeper checks would crash without these.
        return errors

    meta = testset["meta"]
    if not isinstance(meta, dict):
        errors.append("'meta' must be a dict")
    else:
        missing_meta = _REQUIRED_META_KEYS - set(meta)
        if missing_meta:
            errors.append(f"Missing 'meta' keys: {sorted(missing_meta)}")
        if "shot_sizes" in meta and not (
            isinstance(meta["shot_sizes"], list)
            and all(isinstance(s, int) and s > 0 for s in meta["shot_sizes"])
        ):
            errors.append("meta.shot_sizes must be a list of positive ints")
        if "categories" in meta and not (
            isinstance(meta["categories"], list)
            and all(isinstance(c, str) for c in meta["categories"])
        ):
            errors.append("meta.categories must be a list of strings")

    examples = testset["examples"]
    if not isinstance(examples, list):
        errors.append("'examples' must be a list")
        return errors

    seen_ids: set[str] = set()
    for i, ex in enumerate(examples):
        if not isinstance(ex, dict):
            errors.append(f"examples[{i}] must be a dict, got {type(ex).__name__}")
            continue

        missing_ex = _REQUIRED_EXAMPLE_KEYS - set(ex)
        if missing_ex:
            errors.append(f"examples[{i}] missing keys: {sorted(missing_ex)}")
            continue

        ex_id = ex["id"]
        if not isinstance(ex_id, str) or not ex_id:
            errors.append(f"examples[{i}].id must be a non-empty string")
        else:
            if ex_id in seen_ids:
                errors.append(f"examples[{i}].id={ex_id!r} is duplicated")
            seen_ids.add(ex_id)

        if not isinstance(ex["category"], str) or not ex["category"]:
            errors.append(f"examples[{i}].category must be a non-empty string")

        shot_count = ex["shot_count"]
        if not isinstance(shot_count, int) or shot_count < 0:
            errors.append(
                f"examples[{i}].shot_count must be a non-negative int, got {shot_count!r}"
            )

        if not isinstance(ex["target_query"], str) or not ex["target_query"]:
            errors.append(f"examples[{i}].target_query must be a non-empty string")

        faux = ex["faux_dialogue"]
        if not isinstance(faux, list):
            errors.append(f"examples[{i}].faux_dialogue must be a list")
            continue
        if isinstance(shot_count, int) and len(faux) != shot_count:
            errors.append(
                f"examples[{i}].faux_dialogue has {len(faux)} turns "
                f"but shot_count={shot_count}"
            )
        for j, turn in enumerate(faux):
            if not isinstance(turn, dict):
                errors.append(
                    f"examples[{i}].faux_dialogue[{j}] must be a dict"
                )
                continue
            if "user" not in turn or "assistant" not in turn:
                errors.append(
                    f"examples[{i}].faux_dialogue[{j}] missing 'user' / 'assistant'"
                )
                continue
            if not isinstance(turn["user"], str) or not isinstance(
                turn["assistant"], str
            ):
                errors.append(
                    f"examples[{i}].faux_dialogue[{j}] user/assistant must be strings"
                )

    return errors


def load_testset(path: str | Path) -> dict[str, Any]:
    """Load + validate a testset JSON. Raises on schema errors."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Testset not found at {p}. Either drop a pre-built "
            "jailbreak_testset.json into the repo root, or build one via "
            "`from src.testset import build_testset; "
            "build_testset(load_testset_build_config(...))`."
        )
    with p.open("r", encoding="utf-8") as f:
        testset = json.load(f)

    errs = validate_schema(testset)
    if errs:
        joined = "\n  - ".join(errs[:20])
        more = f"\n  ...and {len(errs) - 20} more" if len(errs) > 20 else ""
        raise ValueError(f"Malformed testset at {p}:\n  - {joined}{more}")
    return testset


def filter_testset(
    testset: dict[str, Any],
    categories: list[str] | None = None,
    shot_sizes: list[int] | None = None,
) -> dict[str, Any]:
    """Return a *copy* of ``testset`` keeping only matching examples.

    ``meta`` is updated to reflect the filtered counts so downstream
    consumers see consistent metadata.
    """
    out = copy.deepcopy(testset)
    examples = out["examples"]
    if categories is not None:
        cat_set = set(categories)
        examples = [e for e in examples if e["category"] in cat_set]
    if shot_sizes is not None:
        size_set = set(shot_sizes)
        examples = [e for e in examples if e["shot_count"] in size_set]

    out["examples"] = examples
    out["meta"] = dict(out["meta"])
    out["meta"]["total_examples"] = len(examples)
    if shot_sizes is not None:
        out["meta"]["shot_sizes"] = sorted(set(shot_sizes))
    if categories is not None:
        out["meta"]["categories"] = sorted(set(categories))
    return out


# ---------------------------------------------------------------------------
# Builder (HarmBench → schema-valid testset JSON)
# ---------------------------------------------------------------------------
#
# The builder lives here (not in scripts/) so that notebook users can run it
# from Python without shelling out. The CLI script in
# ``scripts/build_testset_from_harmbench.py`` is just a thin wrapper around
# :func:`build_testset`.
#
# HarmBench's column / split names drift between revisions; the loader
# auto-detects common names and warns when it falls back to a placeholder
# compliant-response (the testset is still schema-valid in that case, but
# tagged via ``meta.fallback_response = True``).


_log = logging.getLogger("testset_builder")

_QUERY_COLS = ["behavior", "prompt", "goal", "instruction", "question"]
_CATEGORY_COLS = ["category", "semantic_category", "functional_category", "tags"]
_RESPONSE_COLS = [
    "target_response",
    "completion",
    "jailbroken_response",
    "response",
    "answer",
    "target",
]

_FALLBACK_RESPONSE = (
    "[fallback compliant-response placeholder; HarmBench split did not "
    "expose a target/jailbroken-response column. Replace this with real "
    "jailbroken completions for a faithful many-shot evaluation.]"
)


def _find_col(cols: set[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def _stable_id(query: str, shot_count: int, salt: int) -> str:
    h = hashlib.sha256()
    h.update(f"{shot_count}:{salt}:{query}".encode("utf-8"))
    return h.hexdigest()[:16]


def load_harmbench(repo: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Download HarmBench, return ``(rows, column_map)``.

    ``column_map`` keys: ``query``, ``category``, ``response`` — values
    are the actual column names in this revision of HarmBench. ``response``
    may map to ``""`` (empty) when no compliant-response column exists,
    in which case downstream code falls back to a synthetic placeholder.
    """
    from datasets import load_dataset

    _log.info("Downloading HarmBench from %s", repo)
    candidate_splits = ["standard", "test", "train", "validation"]
    ds = None
    for split in candidate_splits:
        try:
            ds = load_dataset(repo, split=split)
            _log.info("Using split=%s (n=%d)", split, len(ds))
            break
        except Exception as e:  # noqa: BLE001
            _log.debug("split=%s failed: %s", split, e)
    if ds is None:
        full = load_dataset(repo)
        split_name = max(full, key=lambda k: len(full[k]))
        ds = full[split_name]
        _log.info("Using auto-detected split=%s (n=%d)", split_name, len(ds))

    cols = set(ds.column_names)
    query_col = _find_col(cols, _QUERY_COLS)
    if query_col is None:
        raise RuntimeError(
            f"Could not find a harmful-prompt column in {repo}. "
            f"Looked for {_QUERY_COLS}, found columns: {sorted(cols)}"
        )
    category_col = _find_col(cols, _CATEGORY_COLS)
    response_col = _find_col(cols, _RESPONSE_COLS)
    if response_col is None:
        _log.warning(
            "No compliant-response column found in %s. Falling back to a "
            "placeholder string for the assistant turns (the testset is "
            "still schema-valid; tagged via meta.fallback_response=True).",
            repo,
        )

    _log.info(
        "Columns resolved: query=%r category=%r response=%r",
        query_col,
        category_col,
        response_col,
    )

    rows: list[dict[str, Any]] = []
    for ex in ds:
        rows.append(
            {
                "query": str(ex[query_col]),
                "category": str(ex[category_col]) if category_col else "uncategorised",
                "response": str(ex[response_col]) if response_col else "",
            }
        )
    return rows, {
        "query": query_col,
        "category": category_col or "",
        "response": response_col or "",
    }


def make_examples(
    rows: list[dict[str, Any]],
    cfg: TestsetBuildConfig,
) -> list[dict[str, Any]]:
    """Sample faux dialogues + target queries per ``shot_count`` cell."""
    rng = random.Random(cfg.seed)
    n = len(rows)
    if n < max(cfg.shot_sizes) + 2:
        raise RuntimeError(
            f"Not enough HarmBench rows ({n}) to build a testset with "
            f"max shot size {max(cfg.shot_sizes)}."
        )

    by_category: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_category.setdefault(r["category"], []).append(i)

    examples: list[dict[str, Any]] = []
    for shot_count in cfg.shot_sizes:
        for k in range(cfg.examples_per_size):
            target_idx = rng.randrange(n)
            target = rows[target_idx]

            pool = (
                list(range(n))
                if cfg.mix_categories_in_faux_dialogue
                else list(by_category[target["category"]])
            )
            pool = [i for i in pool if i != target_idx]
            if len(pool) < shot_count:
                # Per-category pool too small; fall back to global.
                pool = [i for i in range(n) if i != target_idx]
            faux_idxs = rng.sample(pool, shot_count)

            faux_dialogue = [
                {
                    "user": rows[j]["query"],
                    "assistant": rows[j]["response"] or _FALLBACK_RESPONSE,
                }
                for j in faux_idxs
            ]

            examples.append(
                {
                    "id": _stable_id(target["query"], shot_count, salt=k),
                    "category": target["category"],
                    "shot_count": shot_count,
                    "faux_dialogue": faux_dialogue,
                    "target_query": target["query"],
                }
            )

    _log.info("Built %d examples", len(examples))
    return examples


def build_testset(cfg: TestsetBuildConfig) -> Path:
    """Build a testset from HarmBench end-to-end and write it to disk.

    Parameters
    ----------
    cfg
        A :class:`TestsetBuildConfig`. Use
        :func:`src.config.load_testset_build_config` to load one from YAML.

    Returns
    -------
    Path
        Path to the written testset JSON (default: ``./jailbreak_testset.json``).

    Notes
    -----
    Schema validation is run before writing — a corrupt testset will raise
    rather than silently land on disk. Re-run the function any time you
    want to regenerate; output is overwritten.
    """
    rows, col_map = load_harmbench(cfg.harmbench_repo)
    examples = make_examples(rows, cfg)
    used_fallback = col_map["response"] == ""

    testset = {
        "meta": {
            "source": "HarmBench",
            "harmbench_repo": cfg.harmbench_repo,
            "shot_sizes": list(cfg.shot_sizes),
            "examples_per_size": cfg.examples_per_size,
            "total_examples": len(examples),
            "categories": sorted({r["category"] for r in rows}),
            "seed": cfg.seed,
            "mix_categories_in_faux_dialogue": cfg.mix_categories_in_faux_dialogue,
            "harmbench_columns_used": col_map,
            "fallback_response": used_fallback,
        },
        "examples": examples,
    }

    errs = validate_schema(testset)
    if errs:
        joined = "\n  - ".join(errs)
        raise RuntimeError(f"Built testset failed schema validation:\n  - {joined}")

    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(testset, f, ensure_ascii=False, indent=2)

    if used_fallback:
        _log.warning(
            "Fallback compliant-response placeholder was used. The eval "
            "still runs but is not a faithful many-shot attack."
        )
    _log.info("Wrote %d examples to %s", len(examples), out_path)
    return out_path


__all__ = [
    "load_testset",
    "validate_schema",
    "filter_testset",
    "build_testset",
    "load_harmbench",
    "make_examples",
]
