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

    loss: one of "mse" or "gaussian_nll".
        "mse"          – standard MSELoss.
        "gaussian_nll" – GaussianNLLLoss; the model predicts mean *and*
                         log-variance for each target variable, so the actual
                         S4Model output dimension is 2 * d_output.
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
    loss: str = "mse"

    def __post_init__(self):
        if self.dt_min >= self.dt_max:
            raise ValueError("dt_min must be < dt_max")
        if self.loss not in ("mse", "gaussian_nll"):
            raise ValueError(
                f"Unknown loss {self.loss!r}. Must be 'mse' or 'gaussian_nll'."
            )

    def model_kwargs(self) -> dict:
        # gaussian_nll needs the network to emit mean + log-var → 2 * d_output
        model_d_output = 2 * self.d_output if self.loss == "gaussian_nll" else self.d_output
        return {
            "d_input":  self.d_input,
            "d_output": model_d_output,
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
            loss=m.get("loss",        "mse"),
        )


class LitModelS4D(L.LightningModule):
    """Lightning module for S4D-based BNS parameter regression.

    Supports two loss functions selected via ``model_cfg.loss``:

    * ``"mse"``          – MSELoss (original behaviour).
    * ``"gaussian_nll"`` – GaussianNLLLoss with predicted per-variable
                           uncertainties.  The network outputs
                           ``[mean_0, …, mean_{d-1}, log_var_0, …, log_var_{d-1}]``
                           (2 * d_output values).  Log-variance is used so the
                           predicted variance is always positive after ``exp()``.

    Batch format (from LitBNSDataRegression / BNSDatasetRegression):
        X_observed : (B, n_ifos, seq_len)
        y_target   : (B, d_output)
    """

    def __init__(self, model_cfg: S4DModelConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = model_cfg

        if model_cfg.loss == "gaussian_nll":
            self.criterion = torch.nn.GaussianNLLLoss(full=False, eps=1e-6, reduction="mean")
        else:
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
        outputs = self(x)                                   # (B, model_d_output)

        if self.cfg.loss == "gaussian_nll":
            d = self.cfg.d_output
            mean    = outputs[:, :d]       # (B, d_output)
            log_var = outputs[:, d:]       # (B, d_output)
            var     = torch.exp(log_var)   # (B, d_output) – always positive

            loss = self.criterion(mean, y_target, var)

            # Per-variable NLL for logging
            gnll_none = torch.nn.GaussianNLLLoss(full=False, eps=1e-6, reduction="none")
            y_indiv = gnll_none(mean, y_target, var).mean(dim=0)  # (d_output,)
        else:
            mse_per_var = torch.nn.MSELoss(reduction="none")
            y_indiv = mse_per_var(outputs, y_target).mean(dim=0)  # (d_output,)
            loss = self.criterion(outputs, y_target)

        return loss, y_indiv

    # ------------------------------------------------------------------
    def _log_per_var(self, y_indiv, prefix: str):
        metric = "nll" if self.cfg.loss == "gaussian_nll" else "mse"
        for i, val in enumerate(y_indiv):
            self.log(f"{prefix}/{metric}/var_{i}", val, on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        loss, y_indiv = self.compute_loss(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_per_var(y_indiv, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_indiv = self.compute_loss(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_per_var(y_indiv, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_indiv = self.compute_loss(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_per_var(y_indiv, "test")
        return loss

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
