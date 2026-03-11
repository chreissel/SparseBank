# Entry point called by regression.py (and thus main.py).
# Translates SparseBank's dict-based config into BNSReg-style Lightning
# objects and runs training.

from __future__ import annotations

import logging
from pathlib import Path

import lightning as L

from step2.dataset import SparseRegressionConfig
from step2.dataloader import LitBNSDataRegression
from step2.task import LitModelS4D, S4DModelConfig

log = logging.getLogger("step2")


def train_regression(cfg: dict) -> Path:
    """Train an S4D regression model.

    Parameters
    ----------
    cfg:
        Top-level SparseBank config dict (loaded from config.yaml).

    Returns
    -------
    Path to the saved checkpoint.
    """
    train_cfg = cfg.get("training", {})

    # ── Config objects ────────────────────────────────────────────────
    data_cfg  = SparseRegressionConfig.from_sparsebank_cfg(cfg)
    model_cfg = S4DModelConfig.from_sparsebank_cfg(cfg)

    # ── DataModule & LightningModule ──────────────────────────────────
    dm        = LitBNSDataRegression(data_cfg)
    lit_model = LitModelS4D(model_cfg)

    # ── Trainer ───────────────────────────────────────────────────────
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = L.Trainer(
        max_epochs=train_cfg.get("max_epochs", 50),
        default_root_dir=str(ckpt_dir),
        accelerator="auto",
        log_every_n_steps=10,
    )

    log.info("[Step 2] Starting training …")
    trainer.fit(lit_model, datamodule=dm)

    ckpt_path = ckpt_dir / "final_model.ckpt"
    trainer.save_checkpoint(str(ckpt_path))
    log.info("[Step 2] Model saved → %s", ckpt_path)
    return ckpt_path
