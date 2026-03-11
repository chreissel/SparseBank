# ─────────────────────────────────────────────────────────────
# STEP 3 – Per-event template bank filtering
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    Uses the same LitModelS4D class from step2 so the architecture
    is guaranteed to match the checkpoint that was just produced by
    train_regression().  Returns the inner S4Model in eval mode on the
    best available device (CUDA > MPS > CPU).
    """
    from step2.task import LitModelS4D

    device = _get_device()
    log.info("[Step 3] Using device: %s", device)

    lit = LitModelS4D.load_from_checkpoint(str(ckpt_path), map_location=device)
    model = lit.model
    model.eval()
    model.to(device)
    return model, device


def _predict_chirp_masses_batch(model, device, strains: np.ndarray) -> np.ndarray:
    """
    Run batched inference on all strain windows in a single forward pass.

    Parameters
    ----------
    model:
        S4Model returned by _load_model().
    device:
        torch.device to run inference on.
    strains:
        NumPy array of shape (n_events, n_ifos, seq_len) or
        (n_events, seq_len) for a single-channel case.

    Returns
    -------
    predictions : np.ndarray of shape (n_events, n_outputs)
        Columns: [mc_pred, ratio_pred, mc_unc, ratio_unc]
    """
    import torch

    x = torch.tensor(strains, dtype=torch.float32)

    if x.ndim == 2:
        # single channel: (n_events, seq_len) → (n_events, seq_len, 1)
        x = x.unsqueeze(-1)
    else:
        # multi-channel: (n_events, n_ifos, seq_len) → (n_events, seq_len, n_ifos)
        x = x.permute(0, 2, 1)

    x = x.to(device)

    with torch.no_grad():
        pred = model(x)  # (n_events, n_outputs)

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
      1. Run the trained model to predict chirp mass from the strain data
         (all events batched in a single GPU/CPU forward pass).
      2. Write a dedicated filtered template bank for that event
         (bank pruning runs in parallel via a thread pool).

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
    model, device = _load_model(ckpt_path)

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

        all_strains.append(strains_chunk)
        all_y_true.append(y_true_chunk)

    strains = np.concatenate(all_strains, axis=0)
    y_true  = np.concatenate(all_y_true,  axis=0)

    bank_in   = Path(cfg["bank_filter"]["input_bank"])
    banks_dir = Path(cfg["bank_filter"].get("per_event_dir", "templates/per_event"))
    banks_dir.mkdir(parents=True, exist_ok=True)
    margin    = float(cfg["bank_filter"].get("margin", 0.1))

    n_events = len(strains)
    log.info("[Step 3] Running batched inference for %d events on %s …",
             n_events, device)

    # ── Batched GPU inference (single forward pass for all events) ──────────
    predictions = _predict_chirp_masses_batch(model, device, strains)
    # predictions shape: (n_events, 4) → [mc_pred, ratio_pred, mc_unc, ratio_unc]

    log.info("[Step 3] Filtering bank for %d events (margin ±%.3f M_sun) …",
             n_events, margin)

    bank_paths = [banks_dir / f"bank_event_{i:06d}.xml.gz" for i in range(n_events)]
    mc_preds   = predictions[:, 0].tolist()

    # ── Parallel bank pruning (thread pool for I/O-bound XML work) ──────────
    n_kept_map: dict[int, int] = {}
    n_before: int | None = None

    with ThreadPoolExecutor() as executor:
        future_to_idx = {
            executor.submit(_prune_bank, bank_in, bank_paths[i], mc_preds[i], margin): i
            for i in range(n_events)
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            n_kept_map[i] = future.result()

    # Read full bank size once for logging (if ligo.lw available)
    if any(v >= 0 for v in n_kept_map.values()):
        try:
            from ligo.lw import ligolw, lsctables, utils as ligolw_utils
            xmldoc   = ligolw_utils.load_filename(
                str(bank_in),
                contenthandler=lsctables.use_in(ligolw.LIGOLWContentHandler))
            n_before = len(lsctables.SnglInspiralTable.get_table(xmldoc))
        except Exception:
            n_before = -1

    event_banks = []
    for i in range(n_events):
        mc_pred, ratio_pred, mc_unc, ratio_unc = predictions[i]
        n_kept   = n_kept_map[i]
        bank_out = bank_paths[i]

        log.info(
            "  event %06d | mc_pred=%.4f M_sun | mc_true=%.4f M_sun "
            "| templates: %s → %s | %s",
            i, mc_pred, float(y_true[i]),
            n_before, n_kept, bank_out.name,
        )

        event_banks.append({
            "event_id":  i,
            "mc_pred":   float(mc_pred),
            "mc_true":   float(y_true[i]),
            "bank_path": bank_out,
            "n_kept":    n_kept,
        })

    log.info("[Step 3] Done. %d per-event banks written to %s",
             n_events, banks_dir)
    return event_banks
