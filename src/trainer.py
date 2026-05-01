"""LoRA SFT trainer for long-context fine-tuning.

The loop is intentionally minimal — there is no Accelerate / Trainer
abstraction. The reasons are:

1. The whole pipeline runs on a single H100; we don't need DDP / DeepSpeed.
2. The Unsloth fast-path is incompatible with several HF Trainer
   internals.
3. Reading 200 lines of explicit loop is easier than auditing 10 callback
   layers when something goes wrong at 30k context.

Validation cadence is set by ``val_steps_per_epoch``: we evaluate that
many times per epoch (evenly spaced). Each validation pass also writes
a LoRA checkpoint so we can recover the best-loss model later.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .checkpoint import save_lora_checkpoint
from .config import TrainConfig, dump_config
from .data import build_train_dataset, collate_for_causal_lm, split_train_val
from .model import build_model_and_tokenizer
from .utils import append_jsonl, set_seed, setup_logger

if TYPE_CHECKING:  # pragma: no cover
    pass


class Trainer:
    """LoRA SFT trainer with periodic validation + checkpointing."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        set_seed(config.seed)

        self.run_dir = config.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "train_log.jsonl"
        self.logger = setup_logger("trainer", self.run_dir / "trainer.log")
        dump_config(config, self.run_dir / "config.yaml")

        self.logger.info("Building model + tokenizer (%s)", config.model_name)
        self.model, self.tokenizer = build_model_and_tokenizer(config)

        self.logger.info("Building datasets (%s)", config.dataset_name)
        full_ds = build_train_dataset(config, self.tokenizer)
        self.train_ds, self.val_ds = split_train_val(
            full_ds, config.val_split_ratio, config.seed
        )
        self.logger.info(
            "Train size = %d | Val size = %d", len(self.train_ds), len(self.val_ds)
        )

        # Lazy imports to keep the module importable without torch installed.
        import torch
        from torch.utils.data import DataLoader

        self.torch = torch

        pad_id = self.tokenizer.pad_token_id

        def collate(batch):
            return collate_for_causal_lm(batch, pad_token_id=pad_id)

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=0,  # tokenisation already done; workers add startup cost
            pin_memory=True,
        )
        self.val_loader = (
            DataLoader(
                self.val_ds,
                batch_size=max(1, config.batch_size),
                shuffle=False,
                collate_fn=collate,
                num_workers=0,
                pin_memory=True,
            )
            if len(self.val_ds) > 0
            else None
        )

        self.optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        steps_per_epoch = math.ceil(
            len(self.train_loader) / config.gradient_accumulation_steps
        )
        self.total_steps = max(1, steps_per_epoch * config.num_epochs)
        self.warmup_steps = max(1, int(self.total_steps * config.warmup_ratio))
        self.scheduler = self._build_scheduler()

        # Validation cadence in micro-batches:
        per_epoch_micro = max(1, len(self.train_loader))
        self.val_every_micro = max(
            1, per_epoch_micro // max(1, config.val_steps_per_epoch)
        )

        self._best_val_loss: float | None = None
        self._best_ckpt: Path | None = None
        self._patience_left = (
            config.early_stopping_patience
            if config.early_stopping_patience is not None
            else math.inf
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_scheduler(self):
        """Linear warmup → constant. Cosine isn't worth the complexity for 1-epoch SFT."""
        from torch.optim.lr_scheduler import LambdaLR

        warmup = self.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step + 1) / float(warmup)
            return 1.0

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _log_event(self, **kwargs) -> None:
        append_jsonl(self.log_path, {"t": time.time(), **kwargs})

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def evaluate(self) -> float:
        """Return mean validation loss. Returns ``inf`` if no val set."""
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        total = 0.0
        n = 0
        with self.torch.no_grad():
            for batch in self.val_loader:
                batch = self._to_device(batch)
                out = self.model(**batch)
                loss = float(out.loss.item())
                total += loss
                n += 1
        self.model.train()
        return total / max(1, n)

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------

    def train(self) -> Path:
        """Run the full training loop. Returns the path to the **best** checkpoint
        (lowest val loss seen during training). Falls back to the final checkpoint
        if no validation ever ran. The best/final paths are also exposed as
        ``self.best_ckpt`` and ``self.final_ckpt`` after training.
        """
        from tqdm.auto import tqdm

        cfg = self.config
        self.model.train()

        global_step = 0
        micro_step = 0
        running_loss = 0.0
        running_count = 0
        last_ckpt: Path | None = None
        accum = cfg.gradient_accumulation_steps

        self.logger.info(
            "Starting training: epochs=%d, total_optim_steps=%d, warmup=%d",
            cfg.num_epochs,
            self.total_steps,
            self.warmup_steps,
        )
        self._log_event(event="train_start", total_steps=self.total_steps)

        for epoch in range(cfg.num_epochs):
            pbar = tqdm(
                self.train_loader,
                desc=f"epoch {epoch + 1}/{cfg.num_epochs}",
                dynamic_ncols=True,
            )
            self.optimizer.zero_grad(set_to_none=True)

            for batch in pbar:
                batch = self._to_device(batch)
                out = self.model(**batch)
                loss = out.loss / accum
                loss.backward()

                running_loss += float(out.loss.item())
                running_count += 1
                micro_step += 1

                if micro_step % accum == 0:
                    self.torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                avg_loss = running_loss / max(1, running_count)
                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                # Log every accumulation boundary.
                if micro_step % accum == 0:
                    self._log_event(
                        event="step",
                        epoch=epoch,
                        micro_step=micro_step,
                        global_step=global_step,
                        train_loss=avg_loss,
                        lr=lr,
                    )

                # Periodic validation + checkpoint.
                should_validate = micro_step % self.val_every_micro == 0
                should_save_step = (
                    cfg.save_every_n_steps is not None
                    and global_step > 0
                    and global_step % cfg.save_every_n_steps == 0
                    and micro_step % accum == 0
                )
                if should_validate or should_save_step:
                    val_loss = self.evaluate()
                    self.logger.info(
                        "epoch=%d step=%d val_loss=%.4f train_loss=%.4f",
                        epoch + 1,
                        global_step,
                        val_loss,
                        avg_loss,
                    )
                    self._log_event(
                        event="val",
                        epoch=epoch,
                        global_step=global_step,
                        val_loss=val_loss,
                        train_loss=avg_loss,
                    )

                    ckpt_dir = (
                        self.run_dir
                        / f"epoch{epoch + 1}_step{global_step}_loss{val_loss:.4f}"
                    )
                    save_lora_checkpoint(self.model, cfg, ckpt_dir)
                    last_ckpt = ckpt_dir

                    if (
                        self._best_val_loss is None
                        or val_loss < self._best_val_loss
                    ):
                        self._best_val_loss = val_loss
                        self._best_ckpt = ckpt_dir
                        self._patience_left = (
                            cfg.early_stopping_patience
                            if cfg.early_stopping_patience is not None
                            else math.inf
                        )
                    else:
                        self._patience_left -= 1
                        if self._patience_left <= 0:
                            self.logger.info(
                                "Early stopping triggered at step %d (val=%.4f, best=%.4f)",
                                global_step,
                                val_loss,
                                self._best_val_loss,
                            )
                            self._log_event(
                                event="early_stop",
                                global_step=global_step,
                                val_loss=val_loss,
                                best_val_loss=self._best_val_loss,
                            )
                            return self._finalize(epoch, global_step, last_ckpt)

            # End of epoch — flush any leftover gradient accumulation.
            if micro_step % accum != 0:
                self.torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                global_step += 1

        return self._finalize(cfg.num_epochs - 1, global_step, last_ckpt)

    def _finalize(
        self, epoch: int, global_step: int, last_ckpt: Path | None
    ) -> Path:
        """Save a final checkpoint, then return whichever checkpoint had the
        lowest val loss (best). Falls back to the final checkpoint if no
        validation ever ran (e.g. tiny dataset, val_split_ratio=0).
        """
        val_loss = self.evaluate()
        final_dir = (
            self.run_dir
            / f"final_epoch{epoch + 1}_step{global_step}_loss{val_loss:.4f}"
        )
        save_lora_checkpoint(self.model, self.config, final_dir)
        self.final_ckpt = final_dir

        # Final model might also be the best — update tracker if so.
        if (
            self._best_val_loss is None
            or (val_loss != float("inf") and val_loss < self._best_val_loss)
        ):
            self._best_val_loss = val_loss
            self._best_ckpt = final_dir

        best = self._best_ckpt or final_dir
        self.best_ckpt = best

        self._log_event(
            event="train_end",
            epoch=epoch,
            global_step=global_step,
            final_val_loss=val_loss,
            final_ckpt=str(final_dir),
            best_ckpt=str(best),
            best_val_loss=self._best_val_loss,
        )
        self.logger.info(
            "Final checkpoint at %s (val_loss=%.4f). Best checkpoint: %s (val_loss=%.4f).",
            final_dir,
            val_loss,
            best,
            self._best_val_loss if self._best_val_loss is not None else float("nan"),
        )
        return best


__all__ = ["Trainer"]
