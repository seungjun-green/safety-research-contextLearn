"""Long-context jailbreak robustness research package.

Modules
-------
- ``config``: YAML → frozen dataclass loaders for train / eval / testset configs.
- ``model``: HF + Unsloth model builders and a vLLM engine builder.
- ``data``: long-context training datasets (LongAlpaca / LongAlign / mixed / naive_refusal).
- ``trainer``: LoRA SFT loop with periodic validation and checkpointing.
- ``checkpoint``: LoRA save / load / ``merge_and_unload`` helpers.
- ``jailbreak_eval``: many-shot jailbreak evaluation harness (vLLM only).
- ``judge``: provider-agnostic LLM-as-judge for harmful-response scoring.
- ``capability_eval``: optional MMLU + needle-in-haystack regression checks.
- ``testset``: testset schema validator + filter.
- ``utils``: seeding / logging / chat-template helpers.
"""

__version__ = "0.1.0"
