"""DiffusionTrainer: full training loop with checkpointing, logging, and evaluation."""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.cuda.amp as amp
from omegaconf import DictConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..evaluation.metrics import aggregate_metrics, evaluate_batch
from ..models.diffusion import GaussianDiffusion
from ..utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class DiffusionTrainer:
    def __init__(
        self,
        model: GaussianDiffusion,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        output_dir: str,
        checkpoint_dir: str,
        resume: Optional[str] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        if str(self.device) == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU.")
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt_manager = CheckpointManager(checkpoint_dir)

        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.999),
        )

        # LR scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.num_epochs,
            eta_min=cfg.learning_rate * 0.1,
        )

        # Mixed precision
        self.use_amp = cfg.get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = amp.GradScaler(enabled=self.use_amp)

        self.grad_accum_steps = max(1, cfg.get("gradient_accumulation_steps", 1))

        # Optional torch.compile (PyTorch >= 2.0) for additional throughput
        if cfg.get("compile_model", False):
            try:
                self.model = torch.compile(self.model)
                logger.info("torch.compile enabled.")
            except Exception as e:
                logger.warning(f"torch.compile failed, continuing without: {e}")

        # State
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        if resume:
            self._load_checkpoint(resume)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> None:
        logger.info(
            f"Starting training on {self.device} | "
            f"epochs={self.cfg.num_epochs} | "
            f"steps/epoch≈{len(self.train_loader)}"
        )

        for epoch in range(self.epoch, self.cfg.num_epochs):
            self.epoch = epoch
            epoch_loss = self._train_epoch()

            self.scheduler.step()

            # Periodic evaluation
            if self.global_step > 0 and self.global_step % self.cfg.eval_every_steps < len(
                self.train_loader
            ):
                val_metrics = self._evaluate()
                val_loss = val_metrics.get("psnr", 0.0)
                is_best = val_loss > self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                logger.info(
                    f"[Epoch {epoch}] val: "
                    + " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                )

                self.ckpt_manager.save(
                    self._state_dict(),
                    step=self.global_step,
                    is_best=is_best,
                )

            logger.info(
                f"Epoch {epoch}/{self.cfg.num_epochs} | "
                f"loss={epoch_loss:.4f} | "
                f"lr={self.scheduler.get_last_lr()[0]:.2e}"
            )

        logger.info("Training complete.")

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch}",
            leave=False,
            dynamic_ncols=True,
        )

        accum_loss = 0.0
        accum_mse = 0.0

        for i, batch in enumerate(pbar):
            batch = self._to_device(batch)
            is_last_accum = (i + 1) % self.grad_accum_steps == 0 or (i + 1) == len(self.train_loader)
            loss_dict = self._train_step(batch, accumulate=not is_last_accum)

            accum_loss += loss_dict["loss"] / self.grad_accum_steps
            accum_mse += loss_dict["mse_loss"] / self.grad_accum_steps

            if is_last_accum:
                total_loss += accum_loss
                n_batches += 1
                self.global_step += 1

                pbar.set_postfix(
                    loss=f"{accum_loss:.4f}",
                    mse=f"{accum_mse:.4f}",
                    step=self.global_step,
                )

                accum_loss = 0.0
                accum_mse = 0.0

                # Periodic checkpoint
                if self.global_step % self.cfg.save_every_steps == 0:
                    self.ckpt_manager.save(self._state_dict(), step=self.global_step)

        return total_loss / max(n_batches, 1)

    def _train_step(self, batch: Dict, accumulate: bool = False) -> Dict[str, float]:
        if not accumulate:
            self.optimizer.zero_grad()

        with amp.autocast(enabled=self.use_amp):
            loss_dict = self.model.compute_loss(batch)
            loss = loss_dict["loss"] / self.grad_accum_steps

        self.scaler.scale(loss).backward()

        if not accumulate:
            clip = self.cfg.get("gradient_clip", 1.0)
            if clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        self.model.eval()
        all_metrics = []
        image_size = self.cfg.image_size

        pbar = tqdm(self.val_loader, desc="Evaluating", leave=False, dynamic_ncols=True)
        for batch in pbar:
            batch = self._to_device(batch)
            cond = batch["cond"]
            modality = batch["modality"]
            target = batch["image"]

            # Generate samples
            samples = self.model.ddim_sample(
                cond=cond,
                modality=modality,
                num_steps=50,
            )

            # Normalize to [0, 1] for metrics
            samples_norm = (samples + 1) / 2
            target_norm = (target + 1) / 2

            metrics = evaluate_batch(
                pred=samples_norm,
                target=target_norm,
                pred_mask=(samples_norm > 0.5).float(),
                target_mask=batch["cond"][:, 0:1],  # tumor mask
            )
            all_metrics.append(metrics)

        self.model.train()
        return aggregate_metrics(all_metrics)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _state_dict(self) -> Dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }

    def _load_checkpoint(self, path: Optional[str] = None) -> None:
        state = self.ckpt_manager.load(path)
        if state is None:
            return
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
        self.epoch = state.get("epoch", 0) + 1
        self.global_step = state.get("global_step", 0)
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        logger.info(
            f"Resumed from epoch {self.epoch - 1}, step {self.global_step}"
        )

    # ------------------------------------------------------------------

    def _to_device(self, batch: Dict) -> Dict:
        return {
            k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
