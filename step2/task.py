# Mirrors BNSReg's src/BNSReg/tasks/parameter_estimation/model_s4d_mse.py
# Adapted for SparseBank's dict-based config and step1 data format.

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import optim
import lightning as L

from step2.model import S4Model


@dataclass(frozen=True)
class S4DModelConfig:
    """Hyperparameters for S4Model.

    Mirrors BNSReg's S4DModelConfig.  dt_min / dt_max control the
    range of time-step initialisations in the S4D kernel.
    """

    d_input: int
    d_output: int
    d_model: int
    d_state: int
    n_layers: int
    dropout: float
    dt_min: float = 0.001
    dt_max: float = 0.1
    lr: float | None = None

    def __post_init__(self):
        if self.dt_min >= self.dt_max:
            raise ValueError("dt_min must be < dt_max")

    def model_kwargs(self) -> dict:
        return {
            "d_input":  self.d_input,
            "d_output": self.d_output,
            "d_model":  self.d_model,
            "d_state":  self.d_state,
            "n_layers": self.n_layers,
            "dropout":  self.dropout,
            "dt_min":   self.dt_min,
            "dt_max":   self.dt_max,
            "lr":       self.lr,
        }

    @classmethod
    def from_sparsebank_cfg(cls, cfg: dict) -> "S4DModelConfig":
        m = cfg["model"]
        return cls(
            d_input=m.get("d_input",  2),   # default 2 = H1 + L1 channels
            d_output=m.get("d_output", 1),
            d_model=m.get("d_model",  64),
            d_state=m.get("d_state",  16),
            n_layers=m.get("n_layers", 4),
            dropout=m.get("dropout",  0.1),
            dt_min=m.get("dt_min",    0.001),
            dt_max=m.get("dt_max",    0.1),
            lr=m.get("lr",            None),
        )


class LitModelS4DMSE(L.LightningModule):
    """Lightning module for S4D-based BNS chirp-mass regression with MSE loss.

    Mirrors BNSReg's LitModelS4DMSE.

    Batch format (from LitBNSDataRegression / BNSDatasetRegression):
        X_observed : (B, n_ifos, seq_len)
        y_target   : (B, d_output)
    """

    def __init__(self, model_cfg: S4DModelConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = model_cfg
        self.criterion = torch.nn.MSELoss(reduction="mean")
        self.model: S4Model | None = None
        self.configure_model()

    # ------------------------------------------------------------------
    def configure_model(self):
        if self.model is not None:
            return
        self.model = S4Model(**self.cfg.model_kwargs())
        try:
            self.model = torch.compile(self.model)
        except Exception:
            pass  # torch.compile not available in this environment — skip silently

    def forward(self, x):
        return self.model(x)

    # ------------------------------------------------------------------
    def compute_loss(self, batch):
        X_observed, y_target = batch
        # X_observed: (B, n_ifos, seq_len) → transpose → (B, seq_len, n_ifos)
        x = X_observed.transpose(2, 1)
        outputs = self(x)                                   # (B, d_output)

        mse_per_var = torch.nn.MSELoss(reduction="none")
        y_indiv_mse = mse_per_var(outputs, y_target).T.mean(dim=1)  # (d_output,)
        return self.criterion(outputs, y_target), y_indiv_mse

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        loss, y_indiv_mse = self.compute_loss(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for i, mse_i in enumerate(y_indiv_mse):
            self.log(f"train/mse/var_{i}", mse_i, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_indiv_mse = self.compute_loss(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for i, mse_i in enumerate(y_indiv_mse):
            self.log(f"val/mse/var_{i}", mse_i, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_indiv_mse = self.compute_loss(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
