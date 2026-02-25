# ─────────────────────────────────────────────────────────────
# STEP 1 – Signal & background generation 
# ─────────────────────────────────────────────────────────────

import os
import argparse
import logging
import yaml
import h5py
import numpy as np
from pathlib import Path

import logging
log = logging.getLogger("step1")

def generate_dataset(cfg: dict, split: str) -> Path:
    """
    Generate BNS waveforms + Gaussian noise backgrounds and write to HDF5.
    TODO: replace with the actual BNS signal generation

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
        t_start = int(0.0 * sample_rate)
        t_end   = int(55.0 * sample_rate)
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
