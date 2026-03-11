# ─────────────────────────────────────────────────────────────
# STEP 3 – Per-event template bank filtering
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("step3")


def _load_model(ckpt_path: Path):
    """
    Load the trained S4D regression model from a Lightning checkpoint.

    Uses the same LitModelS4D class from step2 so the architecture
    is guaranteed to match the checkpoint that was just produced by
    train_regression().  Returns the inner S4Model in eval mode on CPU.
    """
    from step2.task import LitModelS4D

    lit = LitModelS4D.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    model = lit.model
    model.eval()
    return model


def _predict_chirp_mass(model, strain: np.ndarray) -> float:
    """
    Run inference on a single strain window.

    Parameters
    ----------
    model:
        S4Model returned by _load_model().
    strain:
        NumPy array of shape (n_ifos, seq_len) or (seq_len,) for a
        single-channel case.

    Returns
    -------
    Predicted chirp mass (scalar, M_sun).
    """
    import torch

    x = torch.tensor(strain, dtype=torch.float32)

    if x.ndim == 1:
        # single channel: (seq_len,) → (seq_len, 1)
        x = x.unsqueeze(-1)
    else:
        # multi-channel: (n_ifos, seq_len) → (seq_len, n_ifos)
        x = x.T

    x = x.unsqueeze(0)  # → (1, seq_len, n_ifos)  i.e. (B, L, d_input)

    with torch.no_grad():
        pred = model(x)
    return float(pred.squeeze())


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
      1. Run the trained model to predict chirp mass from the strain data.
      2. Write a dedicated filtered template bank for that event.

    The path to the input XML bank is read from
    ``cfg["bank_filter"]["input_bank"]`` so it can be set freely in
    config.yaml (or overridden programmatically).

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
    import h5py

    log.info("[Step 3] Loading model from %s …", ckpt_path)
    model = _load_model(ckpt_path)

    data_dir = Path(cfg["data"]["data_dir"])
    test_dir = data_dir / "test"
    h5_files = sorted(test_dir.glob("*.h5"))

    if not h5_files:
        raise FileNotFoundError(f"[Step 3] No HDF5 files found in {test_dir}")

    log.info("[Step 3] Found %d HDF5 file(s) in %s", len(h5_files), test_dir)

    all_strains, all_y_true = [], []
    for h5_path in h5_files:
        log.info("[Step 3]   Loading %s …", h5_path.name)
        with h5py.File(h5_path, "r") as f:
            strains_chunk = f["injected_data"][:]
            y_true_chunk  = f["chirp_mass"][:]

        # Flatten batched format if necessary: (n_gen, bs, n_ifos, L) → (N, n_ifos, L)
        if strains_chunk.ndim == 4:
            n_gen, bs = strains_chunk.shape[:2]
            strains_chunk = strains_chunk.reshape(n_gen * bs, *strains_chunk.shape[2:])
            y_true_chunk  = y_true_chunk.reshape(-1)

        all_strains.append(strains_chunk)
        all_y_true.append(y_true_chunk)

    strains = np.concatenate(all_strains, axis=0)
    y_true  = np.concatenate(all_y_true,  axis=0)

    bank_in   = Path(cfg["bank_filter"]["input_bank"])
    banks_dir = Path(cfg["bank_filter"].get("per_event_dir", "templates/per_event"))
    banks_dir.mkdir(parents=True, exist_ok=True)
    margin    = float(cfg["bank_filter"].get("margin", 0.1))

    n_events = len(strains)
    log.info("[Step 3] Filtering bank for %d events (margin ±%.3f M_sun) …",
             n_events, margin)

    event_banks = []
    n_before    = None  # read full bank size once for logging

    for i, strain in enumerate(strains):
        mc_pred  = _predict_chirp_mass(model, strain)
        bank_out = banks_dir / f"bank_event_{i:06d}.xml.gz"

        n_kept = _prune_bank(bank_in, bank_out, mc_pred, margin)

        if n_before is None and n_kept >= 0:
            try:
                from ligo.lw import ligolw, lsctables, utils as ligolw_utils
                xmldoc   = ligolw_utils.load_filename(
                    str(bank_in),
                    contenthandler=lsctables.use_in(ligolw.LIGOLWContentHandler))
                n_before = len(lsctables.SnglInspiralTable.get_table(xmldoc))
            except Exception:
                n_before = -1

        log.info(
            "  event %06d | mc_pred=%.4f M_sun | mc_true=%.4f M_sun "
            "| templates: %s → %s | %s",
            i, mc_pred, float(y_true[i]),
            n_before, n_kept, bank_out.name,
        )

        event_banks.append({
            "event_id":  i,
            "mc_pred":   mc_pred,
            "mc_true":   float(y_true[i]),
            "bank_path": bank_out,
            "n_kept":    n_kept,
        })

    log.info("[Step 3] Done. %d per-event banks written to %s",
             n_events, banks_dir)
    return event_banks
