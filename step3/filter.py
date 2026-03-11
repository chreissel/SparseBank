# ─────────────────────────────────────────────────────────────
# STEP 3 – Per-event template bank filtering
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("step3")


def _get_device():
    """Return the best available torch device (CUDA > MPS > CPU)."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(ckpt_path: Path):
    """
    Load the trained S4D regression model from a Lightning checkpoint.

    Returns the inner S4Model in eval mode on the best available device.
    """
    from step2.task import LitModelS4D

    device = _get_device()
    log.info("[Step 3] Using device: %s", device)

    lit = LitModelS4D.load_from_checkpoint(str(ckpt_path), map_location=device)
    model = lit.model
    model.eval()
    model.to(device)
    return model, device


def _predict_chirp_masses_batch(model, device, x_batch) -> np.ndarray:
    """
    Run inference on one DataLoader batch.

    Parameters
    ----------
    x_batch:
        Torch tensor of shape (B, n_ifos, seq_len) as returned by
        the step2 DataLoader (collated X_observed tensors).

    Returns
    -------
    np.ndarray of shape (B, n_outputs)
        Columns: [mc_pred, ratio_pred, mc_unc, ratio_unc]
    """
    import torch

    # DataLoader yields (B, n_ifos, seq_len) → model wants (B, seq_len, n_ifos)
    x = x_batch.permute(0, 2, 1).to(device)
    with torch.no_grad():
        pred = model(x)  # (B, n_outputs)
    return pred.cpu().numpy()


def _prune_bank(bank_in: Path, bank_out: Path,
                mc_pred: float, margin: float) -> int:
    """
    Copy bank_in → bank_out keeping only templates within
    [mc_pred - margin, mc_pred + margin].

    Returns the number of templates kept, or -1 if ligo.lw is unavailable.
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
        sngl_table = lsctables.SnglInspiralTable.get_table(xmldoc)
        keep = [r for r in sngl_table if mc_min <= chirp_mass(r) <= mc_max]
        sngl_table[:] = keep
        ligolw_utils.write_filename(xmldoc, str(bank_out))
        return len(keep)

    except ImportError:
        # fallback: write a plain-text mass range file
        txt_path = bank_out.with_suffix("").with_suffix(".txt")
        with open(txt_path, "w") as fh:
            fh.write(f"chirp_mass_min={mc_min:.6f}\nchirp_mass_max={mc_max:.6f}\n")
        return -1   # unknown count


def filter_bank(cfg: dict, ckpt_path: Path) -> list[dict]:
    """
    For every event in the test set:
      1. Load data via the step2 DataModule (reuses dataset/dataloader config).
      2. Run batched GPU inference over the test DataLoader.
      3. Write a per-event filtered template bank.

    Parameters
    ----------
    cfg:
        Top-level SparseBank config dict (loaded from config.yaml).
    ckpt_path:
        Path to the Lightning checkpoint produced by step2.

    Returns
    -------
    List of dicts with keys:
        event_id  – integer index of the event
        mc_pred   – predicted chirp mass (M_sun)
        mc_true   – true chirp mass from the test set (M_sun)
        bank_path – Path to the per-event filtered bank
        n_kept    – number of templates kept (-1 if ligo.lw unavailable)
    """
    from step2.dataloader import LitBNSDataRegression
    from step2.dataset import SparseRegressionConfig

    log.info("[Step 3] Loading model from %s …", ckpt_path)
    model, device = _load_model(ckpt_path)

    # ── Reuse the step2 DataModule for test-set loading ─────────────────────
    reg_cfg = SparseRegressionConfig.from_sparsebank_cfg(cfg)
    dm = LitBNSDataRegression(reg_cfg)
    dm.setup("predict")
    loader = dm.predict_dataloader()

    n_total = len(loader.dataset)
    log.info("[Step 3] Running batched inference over %d events "
             "(batch_size=%d) on %s …",
             n_total, reg_cfg.test_batch_size, device)

    # ── Batched inference over the test DataLoader ───────────────────────────
    all_preds:  list[np.ndarray] = []
    all_y_true: list[np.ndarray] = []

    for x_batch, y_batch in loader:
        all_preds.append(_predict_chirp_masses_batch(model, device, x_batch))
        all_y_true.append(y_batch.numpy())

    predictions = np.concatenate(all_preds,  axis=0)   # (n_events, n_outputs)
    y_true      = np.concatenate(all_y_true, axis=0)   # (n_events, n_target_vars)
    # chirp_mass is always the first target variable
    mc_true = y_true[:, 0]

    bank_in   = Path(cfg["bank_filter"]["input_bank"])
    banks_dir = Path(cfg["bank_filter"].get("per_event_dir", "templates/per_event"))
    banks_dir.mkdir(parents=True, exist_ok=True)
    margin    = float(cfg["bank_filter"].get("margin", 0.1))

    n_events   = len(predictions)
    bank_paths = [banks_dir / f"bank_event_{i:06d}.xml.gz" for i in range(n_events)]
    mc_preds   = predictions[:, 0].tolist()

    log.info("[Step 3] Filtering bank for %d events (margin ±%.3f M_sun) …",
             n_events, margin)

    event_banks = []
    for i in range(n_events):
        mc_pred, ratio_pred, mc_unc, ratio_unc = predictions[i]
        bank_out = bank_paths[i]
        n_kept   = _prune_bank(bank_in, bank_out, mc_preds[i], margin)

        log.info(
            "  event %06d | mc_pred=%.4f M_sun | mc_true=%.4f M_sun "
            "| n_kept=%s | %s",
            i, mc_pred, float(mc_true[i]), n_kept, bank_out.name,
        )

        event_banks.append({
            "event_id":  i,
            "mc_pred":   float(mc_pred),
            "mc_true":   float(mc_true[i]),
            "bank_path": bank_out,
            "n_kept":    n_kept,
        })

    log.info("[Step 3] Done. %d per-event banks written to %s",
             n_events, banks_dir)
    return event_banks
