"""
Gravitational Wave Analysis Pipeline
=====================================
Steps:
  1. Generate BNS signals + backgrounds (SparseBank style)
  2. Train a chirp-mass regression model (BNSReg / S4D-based)
  3. Use predicted masses to filter a gstlal template bank
  4. Run matched filtering with gstlal

Requirements:
  pip install pycbc lalsuite h5py torch lightning numpy scipy
  (gstlal must be installed from source / conda-forge)

Usage:
  python gw_pipeline.py --config pipeline_config.yaml
"""

import os
import argparse
import logging
import yaml
import h5py
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# STEP 1 – Signal & background generation  (SparseBank style)
# ─────────────────────────────────────────────────────────────

def generate_dataset(cfg: dict, split: str) -> Path:
    """
    Generate BNS waveforms + Gaussian noise backgrounds and write to HDF5.

    Parameters mirror SparseBank's generation approach:
      - masses drawn from uniform distributions
      - waveforms approximant: TaylorF2
      - strain whitened, windowed, and segmented
    """
    import pycbc.waveform as wf
    from pycbc.noise import noise_from_psd
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.filter import sigma, matched_filter
    from pycbc.types import FrequencySeries, TimeSeries

    rng = np.random.default_rng(cfg.get("seed", 42) + hash(split) % 2**31)

    n_samples   = cfg[f"n_{split}"]
    sample_rate = cfg.get("sample_rate", 512)   # Hz
    duration    = cfg.get("duration", 64)        # seconds
    f_lower     = cfg.get("f_lower", 20.0)       # Hz
    snr_min     = cfg.get("snr_min", 20.0)
    snr_max     = cfg.get("snr_max", 30.0)
    m_min       = cfg.get("m1_min", 1.0)
    m_max       = cfg.get("m1_max", 2.5)

    delta_t = 1.0 / sample_rate
    delta_f = 1.0 / duration
    flen    = int(duration * sample_rate // 2 + 1)

    out_dir  = Path(cfg["data_dir"]) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sig_combined_{split}.h5"

    log.info(f"[Step 1] Generating {n_samples} {split} samples → {out_path}")

    psd = aLIGOZeroDetHighPower(flen, delta_f, f_lower)

    injected_data_list = []
    chirp_mass_list    = []
    snr_list           = []

    for i in range(n_samples):
        m1 = rng.uniform(m_min, m_max)
        m2 = rng.uniform(m_min, min(m1, m_max))   # m2 <= m1
        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2  # chirp mass

        # ── generate waveform ──────────────────────────────
        hp, hc = wf.get_fd_waveform(
            approximant="TaylorF2",
            mass1=m1, mass2=m2,
            delta_f=delta_f, f_lower=f_lower,
            distance=100,          # Mpc (will be rescaled by SNR)
        )
        hp.resize(flen)

        # ── target SNR rescaling ──────────────────────────
        target_snr = rng.uniform(snr_min, snr_max)
        sig = sigma(hp, psd=psd, low_frequency_cutoff=f_lower)
        hp   = hp * (target_snr / sig) if sig > 0 else hp

        # ── noise realization ────────────────────────────
        noise = noise_from_psd(int(duration * sample_rate), delta_t, psd,
                               seed=int(rng.integers(0, 2**31)))
        noise_fd = noise.to_frequencyseries()
        noise_fd.resize(flen)

        strain_fd = noise_fd + hp
        strain_td = strain_fd.to_timeseries()

        # keep only a small window around merger  (BNSReg style)
        t_start = int(62.75 * sample_rate)
        t_end   = int(63.00 * sample_rate)
        window  = np.array(strain_td)[t_start:t_end]

        injected_data_list.append(window.astype(np.float32))
        chirp_mass_list.append(mc)
        snr_list.append(target_snr)

        if (i + 1) % 500 == 0:
            log.info(f"  ... {i+1}/{n_samples} generated")

    with h5py.File(out_path, "w") as f:
        f.create_dataset("injected_data", data=np.array(injected_data_list))
        f.create_dataset("chirp_mass",    data=np.array(chirp_mass_list,  dtype=np.float32))
        f.create_dataset("snr",           data=np.array(snr_list,         dtype=np.float32))

    log.info(f"[Step 1] Done. Written {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────
# STEP 2 – Train chirp-mass regression  (BNSReg / S4D style)
# ─────────────────────────────────────────────────────────────

def train_regression(cfg: dict) -> Path:
    """
    Train an S4D-based sequence model to regress chirp mass from whitened strain.
    Mirrors BNSReg commit 0a7cad7.
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


# ─────────────────────────────────────────────────────────────
# STEP 3 – Per-event template bank filtering
# ─────────────────────────────────────────────────────────────

def _load_model(ckpt_path: Path):
    """
    Load the trained S4D regression model from a Lightning checkpoint.
    Returns model in eval mode on CPU.
    """
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Reconstruct model from saved hyperparameters
    hparams     = ckpt.get("hyper_parameters", {})
    model_cfg   = hparams.get("model_cfg", {
        "d_input": 1, "d_output": 1, "d_model": 64,
        "d_state": 16, "n_layers": 4, "dropout": 0.1,
    })

    # inline minimal S4D model (mirrors step 2 definition)
    import torch.nn as nn

    class S4DLayer(nn.Module):
        def __init__(self, d_model, d_state, dropout=0.0):
            super().__init__()
            self.kernel  = nn.Linear(d_model, d_model, bias=False)
            self.norm    = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.output  = nn.Linear(d_model, d_model)
        def forward(self, x):
            return self.norm(x + self.dropout(self.output(self.kernel(x))))

    class S4Model(nn.Module):
        def __init__(self, d_input, d_output, d_model, d_state, n_layers, dropout):
            super().__init__()
            self.encoder = nn.Linear(d_input, d_model)
            self.layers  = nn.ModuleList([
                S4DLayer(d_model, d_state, dropout) for _ in range(n_layers)])
            self.decoder = nn.Linear(d_model, d_output)
        def forward(self, x):
            x = self.encoder(x)
            for layer in self.layers:
                x = layer(x)
            return self.decoder(x.mean(dim=1))

    model = S4Model(**model_cfg)
    state_dict = {k.replace("model.", ""): v
                  for k, v in ckpt["state_dict"].items()
                  if k.startswith("model.")}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _predict_chirp_mass(model, strain: np.ndarray) -> float:
    """
    Run inference on a single strain window.
    Returns predicted chirp mass (scalar, M_sun).
    """
    import torch
    x = torch.tensor(strain, dtype=torch.float32)
    x = (x - x.mean()) / (x.std() + 1e-8)
    x = x.unsqueeze(0).unsqueeze(-1)   # (1, L, 1)
    with torch.no_grad():
        pred = model(x)
    return float(pred.squeeze())


def _prune_bank(bank_in: Path, bank_out: Path,
                mc_pred: float, margin: float) -> int:
    """
    Copy bank_in → bank_out keeping only templates within
    [mc_pred - margin, mc_pred + margin].
    Returns number of templates kept.
    """
    mc_min = mc_pred - margin
    mc_max = mc_pred + margin

    def chirp_mass(row):
        m1, m2 = row.mass1, row.mass2
        return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2

    try:
        from ligo.lw import ligolw, lsctables, utils as ligolw_utils

        xmldoc = ligolw_utils.load_filename(
            str(bank_in),
            contenthandler=lsctables.use_in(ligolw.LIGOLWContentHandler))
        sngl_table  = lsctables.SnglInspiralTable.get_table(xmldoc)
        keep        = [r for r in sngl_table if mc_min <= chirp_mass(r) <= mc_max]
        sngl_table[:] = keep
        ligolw_utils.write_filename(xmldoc, str(bank_out))
        return len(keep)

    except ImportError:
        # fallback: write a plain-text mass range file
        with open(bank_out.with_suffix(".txt"), "w") as fh:
            fh.write(f"chirp_mass_min={mc_min:.6f}\nchirp_mass_max={mc_max:.6f}\n")
        return -1   # unknown count


def filter_template_bank_per_event(cfg: dict, ckpt_path: Path) -> list[dict]:
    """
    For every event in the test set:
      1. Run the trained model to predict chirp mass
      2. Write a dedicated filtered template bank for that event

    Returns a list of dicts:
      [{"event_id": int, "mc_pred": float, "bank_path": Path}, ...]
    """
    import torch
    import h5py

    log.info("[Step 3] Loading model …")
    model = _load_model(ckpt_path)

    test_file = Path(cfg["data"]["data_dir"]) / "test" / "sig_combined_test.h5"
    with h5py.File(test_file, "r") as f:
        strains    = f["injected_data"][:]   # (N, L)
        y_true     = f["chirp_mass"][:]

    bank_in    = Path(cfg["bank_filter"]["input_bank"])
    banks_dir  = Path(cfg["bank_filter"].get("per_event_dir", "templates/per_event"))
    banks_dir.mkdir(parents=True, exist_ok=True)
    margin     = cfg["bank_filter"].get("margin", 0.1)

    n_events   = len(strains)
    log.info(f"[Step 3] Filtering bank for {n_events} events (margin ±{margin} M_sun) …")

    event_banks = []
    n_before    = None   # read once

    for i, strain in enumerate(strains):
        mc_pred  = _predict_chirp_mass(model, strain)
        bank_out = banks_dir / f"bank_event_{i:06d}.xml.gz"

        n_kept = _prune_bank(bank_in, bank_out, mc_pred, margin)

        if n_before is None and n_kept >= 0:
            # read full bank size once for logging
            try:
                from ligo.lw import ligolw, lsctables, utils as ligolw_utils
                xmldoc   = ligolw_utils.load_filename(
                    str(bank_in),
                    contenthandler=lsctables.use_in(ligolw.LIGOLWContentHandler))
                n_before = len(lsctables.SnglInspiralTable.get_table(xmldoc))
            except Exception:
                n_before = -1

        log.info(
            f"  event {i:06d} | mc_pred={mc_pred:.4f} M_sun "
            f"| templates: {n_before} → {n_kept} | {bank_out.name}"
        )

        event_banks.append({
            "event_id":  i,
            "mc_pred":   mc_pred,
            "mc_true":   float(y_true[i]),
            "bank_path": bank_out,
            "n_kept":    n_kept,
        })

    log.info(f"[Step 3] Done. {n_events} per-event banks written to {banks_dir}")
    return event_banks


# ─────────────────────────────────────────────────────────────
# STEP 4 – Per-event matched filtering with gstlal
# ─────────────────────────────────────────────────────────────

def run_matched_filter_per_event(cfg: dict, event_banks: list[dict]) -> None:
    """
    For each event, run gstlal_inspiral using the event's dedicated
    filtered template bank.

    event_banks: list of dicts returned by filter_template_bank_per_event()
    """
    import subprocess

    mf_cfg     = cfg["matched_filter"]
    output_dir = Path(mf_cfg.get("output_dir", "gstlal_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    strain_files = mf_cfg.get("strain_files", [])
    ifos         = mf_cfg.get("ifos", ["H1", "L1"])

    log.info(f"[Step 4] Running per-event matched filtering for "
             f"{len(event_banks)} events …")

    for event in event_banks:
        event_id  = event["event_id"]
        bank_path = event["bank_path"]
        mc_pred   = event["mc_pred"]

        event_out = output_dir / f"event_{event_id:06d}"
        event_out.mkdir(exist_ok=True)

        log.info(f"  event {event_id:06d} | mc_pred={mc_pred:.4f} | bank={bank_path.name}")

        if mf_cfg.get("run_locally", True):
            for strain_file in strain_files:
                for ifo in ifos:
                    out_file = event_out / f"triggers_{ifo}.xml.gz"
                    cmd = [
                        "gstlal_inspiral",
                        "--psd-fft-length",    str(mf_cfg.get("psd_fft_length",    16)),
                        "--ht-gate-threshold", str(mf_cfg.get("ht_gate_threshold", 100)),
                        "--svd-tolerance",     str(mf_cfg.get("svd_tolerance",  0.9999)),
                        "--bank-file",         str(bank_path),
                        "--ifo",               ifo,
                        "--frame-files",       strain_file,
                        "--output",            str(out_file),
                    ]
                    log.info(f"    {ifo} ← {strain_file}")
                    _run(cmd)

        elif mf_cfg.get("use_condor", False):
            # write a per-event config and DAG, collect for bulk submission
            cfg_path = event_out / "gstlal_config.ini"
            _write_gstlal_config(mf_cfg, str(bank_path), str(cfg_path))
            dag_cmd = [
                "gstlal_inspiral_pipe",
                "--config-file", str(cfg_path),
                "--bank-file",   str(bank_path),
                "--output-dir",  str(event_out),
            ]
            _run(dag_cmd)
            dag_files = list(event_out.glob("*.dag"))
            if dag_files:
                _run(["condor_submit_dag", str(dag_files[0])])

        else:
            log.info(f"    DAG written to {event_out} — submit manually.")

    log.info("[Step 4] Per-event matched filtering complete.")


def _write_gstlal_config(mf_cfg: dict, bank_path: str, out_path: str) -> None:
    """Write a minimal gstlal_inspiral_pipe config ini."""
    content = f"""
[DEFAULT]
ifos           = {' '.join(mf_cfg.get('ifos', ['H1', 'L1']))}
bank-file      = {bank_path}
psd-fft-length = {mf_cfg.get('psd_fft_length', 16)}
output-dir     = {mf_cfg.get('output_dir', 'gstlal_output')}

[inspiral]
ht-gate-threshold = {mf_cfg.get('ht_gate_threshold', 100)}
svd-tolerance     = {mf_cfg.get('svd_tolerance', 0.9999)}
"""
    with open(out_path, "w") as fh:
        fh.write(content.strip() + "\n")
    log.info(f"  Config written → {out_path}")


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"Command failed:\n{result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    if result.stdout:
        log.debug(result.stdout)

