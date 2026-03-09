# ─────────────────────────────────────────────────────────────
# STEP 5 – Sensitive Volume Computation
# ─────────────────────────────────────────────────────────────
"""
Sensitive volume estimation at a false-alarm-rate (FAR) threshold of 1/year.

Algorithm
---------
1. Load injection parameters (chirp_mass, mass_ratio, network_snr) from the
   Step 1 HDF5 test set.

2. Load gstlal trigger output from Step 4 for each event and determine the
   peak combined network SNR recovered by the matched-filter pipeline.

3. Map each injection to an effective luminosity distance using the reference
   calibration supplied in config.yaml:

       D_eff,i = D_ref_Mpc * (snr_ref / snr_i)          [Mpc]

   where snr_ref is the injection SNR at the reference distance D_ref_Mpc.
   The Step-1 injections are drawn from a PowerLaw(snr_min, snr_max, -3) SNR
   distribution; because dN/dSNR ∝ SNR^{-3} while a uniform-in-volume
   distribution requires dN/dSNR ∝ SNR^{-4}, Monte Carlo weights

       w_i = 1 / snr_i

   are applied so that the sensitive volume estimate is valid for an
   astrophysically motivated uniform-in-volume source population.

4. Mark injection i as *found* when

       SNR_recovered,i > snr_threshold

   where snr_threshold is calibrated to the FAR threshold of 1/year
   (config key: sensitive_volume.snr_threshold).  If gstlal trigger files are
   present the recovered SNR is read from them; otherwise the *injected* SNR
   stored in the HDF5 file is used directly (i.e. the injection is found if it
   was injected above the threshold).

5. Compute the Monte Carlo sensitive volume:

       V_T = V_inj * Σ_{found} w_i / Σ_{all} w_i          [Mpc^3]

   with  V_inj = (4π/3) * D_max^3  and  D_max = D_ref_Mpc * (snr_ref / snr_min).

6. Optionally compare against downloaded GWOSC pipeline results (step5/download_results.py).

References
----------
* Tiwari 2018 (PyCBC VT methodology): https://arxiv.org/abs/1712.00482
* Chen et al. 2017 (sensitive volume review): https://arxiv.org/abs/1612.02084
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("step5")

# 1 false alarm per year expressed in Hz
FAR_ONE_PER_YEAR_HZ = 1.0 / (365.25 * 24 * 3600)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_injections(data_dir: Path) -> dict:
    """
    Load injection parameters from the Step 1 test-set HDF5 file.

    Returns a dict with keys:
        chirp_mass  – (N,) array of injected chirp masses [M_sun]
        mass_ratio  – (N,) array of injected mass ratios  (m2/m1)
        snr         – (N,) array of injected network SNRs
    """
    import h5py

    test_file = data_dir / "test" / "sig_combined_test.h5"
    if not test_file.exists():
        raise FileNotFoundError(
            f"[Step 5] Injection file not found: {test_file}\n"
            "Run Step 1 first to generate injection data."
        )

    with h5py.File(test_file, "r") as fh:
        chirp_mass = np.asarray(fh["chirp_mass"]).flatten()
        mass_ratio = np.asarray(fh["mass_ratio"]).flatten()
        snr        = np.asarray(fh["snr"]).flatten()

    log.info("[Step 5] Loaded %d injections from %s", len(snr), test_file)
    return {"chirp_mass": chirp_mass, "mass_ratio": mass_ratio, "snr": snr}


def _peak_snr_from_triggers(trigger_dir: Path, ifos: list[str]) -> float | None:
    """
    Parse the gstlal XML trigger files for a single event and return the
    peak combined network SNR:

        rho_net = sqrt( sum_i rho_i^2 )

    Returns None if no trigger files are found or ligo.lw is unavailable.
    """
    snrs: dict[str, float] = {}

    for ifo in ifos:
        trig_file = trigger_dir / f"triggers_{ifo}.xml.gz"
        if not trig_file.exists():
            continue
        try:
            from ligo.lw import ligolw, lsctables, utils as ligolw_utils

            xmldoc = ligolw_utils.load_filename(
                str(trig_file),
                contenthandler=lsctables.use_in(ligolw.LIGOLWContentHandler),
            )
            table = lsctables.SnglInspiralTable.get_table(xmldoc)
            if len(table):
                snrs[ifo] = max(float(row.snr) for row in table)
        except Exception as exc:
            log.debug("[Step 5] Could not parse %s: %s", trig_file, exc)

    if not snrs:
        return None

    # Combined network SNR (quadrature sum over available detectors)
    return float(np.sqrt(sum(v**2 for v in snrs.values())))


def _snr_to_distance(snr: np.ndarray, d_ref: float, snr_ref: float) -> np.ndarray:
    """Convert network SNR to effective luminosity distance [Mpc].

    D_eff = D_ref * (snr_ref / snr)
    """
    return d_ref * snr_ref / np.asarray(snr, dtype=float)


def _sensitive_volume_mc(
    found_mask: np.ndarray,
    snr_all: np.ndarray,
    snr_min: float,
    d_ref: float,
    snr_ref: float,
) -> tuple[float, float]:
    """
    Monte Carlo estimate of the sensitive volume V_T [Mpc^3] and its
    statistical uncertainty sigma_V [Mpc^3].

    The injection distribution is  dN/dSNR ∝ SNR^{-3}  (PowerLaw, Step 1).
    To convert to a uniform-in-volume population (dN/dSNR ∝ SNR^{-4}),
    importance weights  w_i = 1 / snr_i  are applied.

    V_T = V_inj * sum_{found} w_i / sum_{all} w_i

    with V_inj = (4π/3) * D_max^3  and  D_max = D_ref * (snr_ref / snr_min).

    Statistical uncertainty (Poisson-like):
    sigma_V / V_T = 1 / sqrt(N_found)
    """
    w = 1.0 / np.asarray(snr_all, dtype=float)

    w_found = w[found_mask].sum()
    w_total = w.sum()

    d_max = d_ref * snr_ref / snr_min
    v_inj = (4.0 * np.pi / 3.0) * d_max**3

    p_det = w_found / w_total if w_total > 0 else 0.0
    v_t   = v_inj * p_det

    n_found = int(found_mask.sum())
    sigma_v = v_t / np.sqrt(max(n_found, 1))

    return v_t, sigma_v


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_sensitive_volume_analysis(cfg: dict) -> dict:
    """
    Compute the sensitive volume at FAR = 1/year.

    Parameters
    ----------
    cfg:
        Top-level SparseBank config dict (loaded from config.yaml).

    Returns
    -------
    results dict with keys:
        v_t_mpc3        – sensitive volume [Mpc^3]
        sigma_v_mpc3    – 1-sigma MC uncertainty [Mpc^3]
        n_injections    – total number of injections analysed
        n_found         – injections recovered above snr_threshold
        p_det           – detection probability (N_found / N_total, weighted)
        snr_threshold   – SNR threshold used (proxy for FAR = 1/year)
        d_ref_mpc       – reference distance [Mpc]
        snr_ref         – SNR at reference distance
        output_path     – path to the JSON results file
    """
    sv_cfg      = cfg.get("sensitive_volume", {})
    data_dir    = Path(cfg.get("data", {}).get("data_dir", cfg.get("data_dir", "data")))
    output_dir  = Path(sv_cfg.get("output_dir", "sensitive_volume_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    snr_threshold = float(sv_cfg.get("snr_threshold", 10.0))
    d_ref         = float(sv_cfg.get("d_ref_mpc",     100.0))
    snr_ref       = float(sv_cfg.get("snr_ref",        8.0))
    snr_min       = float(cfg.get("snr_min",           8.0))
    ifos          = list(cfg.get("matched_filter", {}).get("ifos", ["H1", "L1"]))
    gstlal_dir    = Path(cfg.get("matched_filter", {}).get("output_dir", "gstlal_output"))

    far_threshold = float(sv_cfg.get("far_threshold_hz", FAR_ONE_PER_YEAR_HZ))

    log.info(
        "[Step 5] Sensitive volume analysis | FAR threshold = %.2e Hz (%.1f / yr) "
        "| SNR proxy threshold = %.1f",
        far_threshold,
        far_threshold * 365.25 * 24 * 3600,
        snr_threshold,
    )

    # ── 1. Load injections ────────────────────────────────────────────────────
    injections = _load_injections(data_dir)
    n_inj      = len(injections["snr"])

    # ── 2. Determine found / missed ───────────────────────────────────────────
    found_mask = np.zeros(n_inj, dtype=bool)

    have_triggers = gstlal_dir.exists() and any(gstlal_dir.iterdir())

    if have_triggers:
        log.info(
            "[Step 5] Loading gstlal triggers from %s for %d events …",
            gstlal_dir, n_inj,
        )
        recovered_snrs = np.zeros(n_inj)
        for i in range(n_inj):
            event_dir = gstlal_dir / f"event_{i:06d}"
            if event_dir.exists():
                rho = _peak_snr_from_triggers(event_dir, ifos)
                if rho is not None:
                    recovered_snrs[i] = rho
                    log.debug("  event %06d | recovered SNR = %.2f", i, rho)
        found_mask = recovered_snrs > snr_threshold
        log.info(
            "[Step 5] Trigger-based found/missed: %d / %d above SNR=%.1f",
            found_mask.sum(), n_inj, snr_threshold,
        )
    else:
        # Fall back: use the injected SNR as a direct proxy.
        # An injection is "found" if its injected SNR exceeds the threshold,
        # i.e. the signal was loud enough to have been detected had it been
        # a real event (optimistic upper bound on sensitivity).
        log.info(
            "[Step 5] No gstlal trigger directory found at %s. "
            "Using injected SNR as detection proxy (optimistic estimate).",
            gstlal_dir,
        )
        found_mask = injections["snr"] > snr_threshold

    # ── 3. Compute sensitive volume ───────────────────────────────────────────
    v_t, sigma_v = _sensitive_volume_mc(
        found_mask=found_mask,
        snr_all=injections["snr"],
        snr_min=snr_min,
        d_ref=d_ref,
        snr_ref=snr_ref,
    )

    n_found = int(found_mask.sum())
    w       = 1.0 / injections["snr"]
    p_det   = w[found_mask].sum() / w.sum() if w.sum() > 0 else 0.0

    log.info(
        "[Step 5] V_T = %.3e ± %.3e Mpc^3  (N_found=%d / %d,  p_det=%.4f)",
        v_t, sigma_v, n_found, n_inj, p_det,
    )

    # ── 4. Per-chirp-mass breakdown ───────────────────────────────────────────
    mc_bins  = np.linspace(
        injections["chirp_mass"].min(),
        injections["chirp_mass"].max(),
        num=6,
    )
    mc_bin_results = []
    for lo, hi in zip(mc_bins[:-1], mc_bins[1:]):
        mask_bin  = (injections["chirp_mass"] >= lo) & (injections["chirp_mass"] < hi)
        if mask_bin.sum() == 0:
            continue
        vt_bin, sv_bin = _sensitive_volume_mc(
            found_mask=found_mask & mask_bin,
            snr_all=injections["snr"][mask_bin],
            snr_min=snr_min,
            d_ref=d_ref,
            snr_ref=snr_ref,
        )
        mc_bin_results.append(
            {
                "mc_min": float(lo),
                "mc_max": float(hi),
                "v_t_mpc3": vt_bin,
                "sigma_v_mpc3": sv_bin,
                "n_inj": int(mask_bin.sum()),
                "n_found": int((found_mask & mask_bin).sum()),
            }
        )
        log.info(
            "  Mc ∈ [%.3f, %.3f] M_sun | V_T = %.3e Mpc^3  (N=%d, found=%d)",
            lo, hi, vt_bin, mask_bin.sum(), (found_mask & mask_bin).sum(),
        )

    # ── 5. Save results ───────────────────────────────────────────────────────
    results = {
        "v_t_mpc3":        float(v_t),
        "sigma_v_mpc3":    float(sigma_v),
        "n_injections":    n_inj,
        "n_found":         n_found,
        "p_det":           float(p_det),
        "snr_threshold":   snr_threshold,
        "far_threshold_hz": far_threshold,
        "far_threshold_per_year": far_threshold * 365.25 * 24 * 3600,
        "d_ref_mpc":       d_ref,
        "snr_ref":         snr_ref,
        "snr_min":         snr_min,
        "d_max_mpc":       d_ref * snr_ref / snr_min,
        "used_gstlal_triggers": have_triggers,
        "chirp_mass_bins": mc_bin_results,
    }

    out_path = output_dir / "sensitive_volume.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    log.info("[Step 5] Results written to %s", out_path)
    results["output_path"] = str(out_path)
    return results
