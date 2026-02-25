# ─────────────────────────────────────────────────────────────
# STEP 2 – Train chirp-mass regression  (BNSReg / S4D style)
# ─────────────────────────────────────────────────────────────

from pathlib import Path
import h5py

import logging
log = logging.getLogger("step2")

def train_regression(cfg: dict) -> Path:
    """
    Train an S4D-based sequence model to regress chirp mass from whitened strain.
    Mirrors BNSReg commit 0a7cad7.
    TODO: update with the real trainings from Kevin!
    """
    import torch
    from torch import nn, optim
    from torch.utils.data import Dataset, DataLoader
    import lightning as L

    # ── Dataset ───────────────────────────────────────────────
    class BNSDatasetRegression(Dataset):
        def __init__(self, split: str, data_cfg: dict):
            path = (Path(data_cfg["data_dir"]) / split /
                    f"sig_combined_{split}.h5")
            with h5py.File(path, "r") as f:
                self.x = torch.tensor(f["injected_data"][:], dtype=torch.float32)
                self.y = torch.tensor(f["chirp_mass"][:],    dtype=torch.float32)
            # normalise input
            self.x = (self.x - self.x.mean()) / (self.x.std() + 1e-8)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            # shape: (seq_len, 1)
            return self.x[idx].unsqueeze(-1), self.y[idx].unsqueeze(-1)

    # ── Minimal S4D kernel ────────────────────────────────────
    class S4DKernel(nn.Module):
        """Diagonal structured SSM kernel (simplified S4D)."""
        def __init__(self, d_model, d_state):
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state
            # learnable diagonal A, B, C parameters (complex)
            self.log_A_real = nn.Parameter(torch.randn(d_model, d_state))
            self.A_imag     = nn.Parameter(torch.randn(d_model, d_state))
            self.B          = nn.Parameter(torch.randn(d_model, d_state, 2))  # complex
            self.C          = nn.Parameter(torch.randn(d_model, d_state, 2))
            self.D          = nn.Parameter(torch.ones(d_model))

        def forward(self, u):
            # u: (B, L, d_model) – simple linear recurrence approximation
            # For a production pipeline replace with the full convolution kernel.
            return u * self.D.unsqueeze(0).unsqueeze(0)

    class S4DLayer(nn.Module):
        def __init__(self, d_model, d_state, dropout=0.0):
            super().__init__()
            self.kernel  = S4DKernel(d_model, d_state)
            self.norm    = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.output  = nn.Linear(d_model, d_model)

        def forward(self, x):
            y = self.kernel(x)
            y = self.dropout(y)
            y = self.output(y)
            return self.norm(x + y)

    class S4Model(nn.Module):
        def __init__(self, d_input, d_output, d_model, d_state, n_layers, dropout):
            super().__init__()
            self.encoder = nn.Linear(d_input, d_model)
            self.layers  = nn.ModuleList([
                S4DLayer(d_model, d_state, dropout) for _ in range(n_layers)
            ])
            self.decoder = nn.Linear(d_model, d_output)

        def forward(self, x):
            # x: (B, L, d_input)
            x = self.encoder(x)
            for layer in self.layers:
                x = layer(x)
            x = x.mean(dim=1)   # pool over sequence
            return self.decoder(x)

    # ── Lightning module ──────────────────────────────────────
    class LitModelS4DMSE(L.LightningModule):
        def __init__(self, model_cfg):
            super().__init__()
            self.save_hyperparameters()
            self.model = S4Model(**model_cfg)
            self.criterion = nn.MSELoss()

        def compute_loss(self, batch):
            x, y = batch
            return self.criterion(self.model(x), y)

        def training_step(self, batch, _):
            loss = self.compute_loss(batch)
            self.log("train/loss", loss, on_epoch=True, prog_bar=True)
            return loss

        def validation_step(self, batch, _):
            loss = self.compute_loss(batch)
            self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        def configure_optimizers(self):
            opt = optim.AdamW(self.parameters(), lr=1e-3)
            sch = optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch}}

    # ── DataModule ────────────────────────────────────────────
    class LitBNSDataRegression(L.LightningDataModule):
        def __init__(self, data_cfg):
            super().__init__()
            self.cfg = data_cfg

        def setup(self, stage=None):
            if stage == "fit":
                self.train_dataset = BNSDatasetRegression("train", self.cfg)
                self.val_dataset   = BNSDatasetRegression("val",   self.cfg)
            else:
                self.test_dataset = BNSDatasetRegression("test", self.cfg)

        def train_dataloader(self):
            return DataLoader(self.train_dataset,
                              batch_size=self.cfg["batch_size"],
                              shuffle=True,
                              num_workers=self.cfg.get("num_workers", 4),
                              persistent_workers=True)

        def val_dataloader(self):
            return DataLoader(self.val_dataset,
                              batch_size=self.cfg["batch_size"],
                              num_workers=self.cfg.get("num_workers", 4),
                              persistent_workers=True)

    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    dm = LitBNSDataRegression(data_cfg)

    model_params = dict(
        d_input   = model_cfg.get("d_input",   1),
        d_output  = model_cfg.get("d_output",  1),
        d_model   = model_cfg.get("d_model",   64),
        d_state   = model_cfg.get("d_state",   16),
        n_layers  = model_cfg.get("n_layers",  4),
        dropout   = model_cfg.get("dropout",   0.1),
    )
    lit_model = LitModelS4DMSE(model_params)

    ckpt_dir  = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(exist_ok=True)

    trainer = L.Trainer(
        max_epochs       = train_cfg.get("max_epochs", 50),
        default_root_dir = str(ckpt_dir),
        accelerator      = "auto",
        log_every_n_steps= 10,
    )

    log.info("[Step 2] Starting training …")
    trainer.fit(lit_model, datamodule=dm)

    ckpt_path = ckpt_dir / "final_model.ckpt"
    trainer.save_checkpoint(str(ckpt_path))
    log.info(f"[Step 2] Model saved → {ckpt_path}")
    return ckpt_path

