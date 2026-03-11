"""
Step 4 – Per-event matched filtering on pre-whitened HDF5 data
==============================================================
Runs a Python matched filter using the whitened strain data produced by
step 1 and the per-event filtered template banks from step 3.

The data is already whitened (flat noise spectrum), so the optimal filter
equals the normalised template waveform.  Peak SNR is computed via
FFT-based cross-correlation for each IFO independently.
"""

import json
import logging
from pathlib import Path

import h5py
import numpy as np

log = logging.getLogger(__name__)


# ── HDF5 data loading ────────────────────────────────────────────────────────

def _locate_event(test_files: list[Path], event_id: int) -> tuple[Path, int]:
    """Map a global event_id to (hdf5_file, local_index) via sorted test files."""
    cumsum = 0
    for fpath in test_files:
        with h5py.File(fpath, "r") as f:
            n = f["injected_data"].shape[0]
        if cumsum + n > event_id:
            return fpath, event_id - cumsum
        cumsum += n
    raise IndexError(
        f"event_id {event_id} exceeds total test samples ({cumsum})"
    )


def _load_event(fpath: Path, local_idx: int) -> tuple[np.ndarray, dict]:
    """Return whitened strain (n_ifos, seq_len) and true parameters for one event."""
    with h5py.File(fpath, "r") as f:
        strain = f["injected_data"][local_idx].astype(np.float64)
        params = {
            "chirp_mass": float(f["chirp_mass"][local_idx]),
            "mass_ratio":  float(f["mass_ratio"][local_idx]),
            "snr":         float(f["snr"][local_idx]),
        }
    return strain, params


# ── Template bank loading ─────────────────────────────────────────────────────

def _load_bank_templates(bank_path: Path) -> list[dict]:
    """Return a list of {mass1, mass2} dicts from the per-event filtered bank."""
    try:
        from ligo.lw import ligolw, lsctables
        from ligo.lw import utils as ligolw_utils

        xmldoc = ligolw_utils.load_filename(
            str(bank_path),
            contenthandler=lsctables.use_in(ligolw.LIGOLWContentHandler),
        )
        tbl = lsctables.SnglInspiralTable.get_table(xmldoc)
        return [{"mass1": row.mass1, "mass2": row.mass2} for row in tbl]
    except Exception:
        pass

    # Fallback: plain-text chirp-mass range written by step 3 when ligo.lw
    # is unavailable.
    txt = bank_path.with_suffix("").with_suffix(".txt")
    if txt.exists():
        kv = {}
        for line in txt.read_text().splitlines():
            k, v = line.split("=")
            kv[k.strip()] = float(v.strip())
        mc = (kv["chirp_mass_min"] + kv["chirp_mass_max"]) / 2.0
        # equal-mass approximation: m1 = m2 = mc * 2^(1/5)
        m = mc * 2 ** 0.2
        return [{"mass1": m, "mass2": m}]

    log.warning("No templates found at %s", bank_path)
    return []


# ── Template waveform generation ─────────────────────────────────────────────

def _make_template(
    mass1: float,
    mass2: float,
    sample_rate: int,
    f_min: float,
    n_samples: int,
) -> np.ndarray:
    """
    Generate a TaylorT4 time-domain template right-aligned to n_samples.
    Falls back to zeros when pycbc is unavailable or waveform generation fails.
    """
    try:
        from pycbc.waveform import get_td_waveform

        hp, _ = get_td_waveform(
            approximant="TaylorT4",
            mass1=mass1,
            mass2=mass2,
            delta_t=1.0 / sample_rate,
            f_lower=f_min,
        )
        arr = np.array(hp, dtype=np.float64)
    except Exception as exc:
        log.debug("Template generation failed (%s); skipping", exc)
        return np.zeros(n_samples)

    # Right-align: keep the merger end, pad zeros at the start if short.
    if len(arr) >= n_samples:
        return arr[-n_samples:]
    return np.pad(arr, (n_samples - len(arr), 0))


# ── Matched filter ────────────────────────────────────────────────────────────

def _peak_snr(strain: np.ndarray, template: np.ndarray) -> float:
    """
    Matched-filter peak |SNR| for pre-whitened strain.

    For white (already-whitened) noise the optimal filter equals the
    normalised template itself.  Cross-correlation is computed in the
    frequency domain via scipy for efficiency.
    """
    from scipy.signal import fftconvolve

    norm = np.linalg.norm(template)
    if norm < 1e-30:
        return 0.0
    # Time-reversed template → convolution == cross-correlation.
    xcorr = fftconvolve(strain, template[::-1] / norm, mode="full")
    return float(np.max(np.abs(xcorr)))


# ── Main entry point ──────────────────────────────────────────────────────────

def run_matched_filter_per_event(cfg: dict, event_banks: list[dict]) -> None:
    """
    For each event run a Python matched filter on the pre-whitened HDF5
    strain produced by step 1, using the per-event filtered bank from step 3.

    Parameters
    ----------
    cfg:
        Top-level SparseBank config dict.
    event_banks:
        List of dicts returned by step 3's filter_bank().
    """
    mf_cfg     = cfg["matched_filter"]
    output_dir = Path(mf_cfg.get("output_dir", "gstlal_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the test data directory (pre-whitened HDF5 from step 1).
    data_dir = Path(cfg.get("data_dir", "data"))
    test_dir = Path(mf_cfg.get("test_dir", str(data_dir / "test")))

    sample_rate = int(cfg.get("sample_rate", 512))
    f_min       = float(cfg.get("f_min", 20.0))
    ifos        = mf_cfg.get("ifos", ["H1", "L1"])

    test_files = sorted(test_dir.glob("*.h5"))
    if not test_files:
        log.error("[Step 4] No HDF5 files found in %s", test_dir)
        return

    log.info(
        "[Step 4] Running per-event matched filtering for %d events "
        "on pre-whitened HDF5 data from %s …",
        len(event_banks), test_dir,
    )

    for event in event_banks:
        event_id  = event["event_id"]
        bank_path = event["bank_path"]
        mc_pred   = event["mc_pred"]

        event_out = output_dir / f"event_{event_id:06d}"
        event_out.mkdir(exist_ok=True)

        log.info(
            "  event %06d | mc_pred=%.4f M_sun | bank=%s",
            event_id, mc_pred, bank_path.name,
        )

        # Load whitened strain for this event.
        try:
            fpath, local_idx = _locate_event(test_files, event_id)
            strain, true_params = _load_event(fpath, local_idx)
        except Exception as exc:
            log.error("    Could not load event %d: %s", event_id, exc)
            continue

        # Load template mass parameters from the per-event filtered bank.
        templates = _load_bank_templates(bank_path)
        if not templates:
            log.warning("    No templates for event %d — skipping", event_id)
            continue

        n_samples = strain.shape[-1]
        ifo_results: dict[str, dict] = {}

        for ifo_idx, ifo in enumerate(ifos):
            if ifo_idx >= strain.shape[0]:
                log.warning("    IFO %s not available in strain array", ifo)
                continue

            ifo_strain = strain[ifo_idx]  # (seq_len,)

            # Compute peak SNR across all templates; keep the maximum.
            peak_snr  = 0.0
            best_mass1 = templates[0]["mass1"]
            best_mass2 = templates[0]["mass2"]

            for tmpl in templates:
                t = _make_template(
                    tmpl["mass1"], tmpl["mass2"], sample_rate, f_min, n_samples
                )
                snr = _peak_snr(ifo_strain, t)
                if snr > peak_snr:
                    peak_snr  = snr
                    best_mass1 = tmpl["mass1"]
                    best_mass2 = tmpl["mass2"]

            ifo_results[ifo] = {
                "peak_snr":  peak_snr,
                "best_mass1": best_mass1,
                "best_mass2": best_mass2,
            }
            log.info("    %s  peak SNR = %.3f", ifo, peak_snr)

        # Save per-event results as JSON.
        out_file = event_out / "results.json"
        payload = {
            "event_id":   event_id,
            "mc_pred":    mc_pred,
            "mc_true":    event.get("mc_true"),
            "true_params": true_params,
            "ifo_results": ifo_results,
        }
        with open(out_file, "w") as fh:
            json.dump(payload, fh, indent=2)
        log.info("    Results → %s", out_file)

    log.info("[Step 4] Per-event matched filtering complete.")
