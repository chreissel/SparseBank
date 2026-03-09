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
from types import SimpleNamespace
import torch

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
    from ml4gw.transforms import SpectralDensity
    from ml4gw.dataloading import Hdf5TimeSeriesDataset
    from ml4gw.transforms import Whiten
    from ml4gw.gw import compute_network_snr,reweight_snrs, get_ifo_geometry, compute_observed_strain
    from ml4gw.distributions import PowerLaw, Sine, Cosine, DeltaFunction
    from torch.distributions import Uniform
    from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator
    from step1.transforms import Triangular
    from ml4gw.waveforms import TaylorF2
    from step1.load_data import load_data

    rng = np.random.default_rng(cfg.get("seed", 42) + hash(split) % 2**31)

    n_samples   = cfg[f"n_{split}"]
    batch_size  = cfg[f"batch_size"]
    ifos        = ['H1', 'L1']
    sample_rate = cfg.get("sample_rate", 512)   # Hz
    duration    = cfg.get("duration", 64)       # seconds
    f_min       = cfg.get("f_min", 20.0)       # Hz
    f_max       = cfg.get("f_max", 256.0)      # Hz
    f_ref       = cfg.get("f_ref", 50.0)       # Hz
    snr_min     = cfg.get("snr_min", 20.0)
    snr_max     = cfg.get("snr_max", 30.0)
    m_min       = cfg.get("m1_min", 1.0)
    m_max       = cfg.get("m1_max", 2.5)
    right_pad   = cfg.get("right_pad", 1.0)
    psd_length  = cfg.get("psd_length", 64.0)
    open_data   = Path(cfg["data_dir"]) / "background_data"

    nyguist = sample_rate / 2
    num_samples    = int(duration * sample_rate)
    num_freqs = num_samples // 2 + 1
    psd_size = int(psd_length * sample_rate)
    window_length = psd_length + 2.0 + duration

    out_dir  = Path(cfg["data_dir"]) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sig_combined_{split}.h5"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not open_data.exists():
        log.info(f"[Step 1] Background data directory not found at {open_data}. Fetching data...")
        load_cfg = SimpleNamespace(
            general=SimpleNamespace(
                ifos=ifos,
                waveform_duration=window_length,
                sample_rate=sample_rate,
            )
        )
        load_data(load_cfg, Path(cfg["data_dir"]))

    log.info(f"[Step 1] Generating {n_samples} {split} samples → {out_path}")

    injected_data_list = []
    chirp_mass_list    = []
    mass_ratio_list    = []
    snr_list           = []

    total = 0
    while total < n_samples:

        # ── generate waveform ──────────────────────────────
        param_dict = {
            "mass_1": Triangular(1.0, 2.5, 2.5),
            "s1z": DeltaFunction(0.0),
            "s2z": DeltaFunction(0.0),
            "distance": PowerLaw(100,1000,2),
            "phic": DeltaFunction(0.0),
            "inclination": Sine()
        }
        params = {}
        for k, dist in param_dict.items():
            params[k] = dist.sample((batch_size,)).to(device)

        param_dict['mass_2'] = Uniform(1.0, params["mass_1"])
        params["mass_2"] = param_dict['mass_2'].sample().to(device)

        approximant = TaylorF2().to(device)
        q = params['mass_2']/params['mass_1']
        params['chirp_mass'] = (q/(1+q)**2)**(3/5.)*(params['mass_2']+params['mass_1'])
        params['mass_ratio'] = q
        params["chi1"], params["chi2"] = params["s1z"], params["s2z"]

        waveform_generator = TimeDomainCBCWaveformGenerator(
            approximant=approximant,
            sample_rate=sample_rate,
            f_min=f_min,
            duration=duration,
            right_pad=right_pad,
            f_ref=f_ref,
        ).to(device)

        hc, hp = waveform_generator(**params)

        # ── waveform projection ────────────────────────────
        dec = Cosine()
        psi = Uniform(0, torch.pi)
        phi = Uniform(-torch.pi, torch.pi)

        params['dec'] = dec.sample((batch_size,)).to(device)
        params['psi'] = psi.sample((batch_size,)).to(device)
        params['phi'] = phi.sample((batch_size,)).to(device)

        tensors, vertices = get_ifo_geometry(*ifos)

        waveforms = compute_observed_strain(
            dec=params['dec'],
            psi=params['psi'],
            phi=params['phi'],
            detector_tensors=tensors.to(device),
            detector_vertices=vertices.to(device),
            sample_rate=sample_rate,
            cross=hc,
            plus=hp,
        )

        # ── noise realization ──────────────────────────────
        fnames = list(open_data.iterdir())
        dataloader = Hdf5TimeSeriesDataset(
            fnames=fnames,
            channels=ifos,
            kernel_size=int(window_length * sample_rate),
            batch_size=batch_size,
            batches_per_epoch=1,
            coincident=False,
        )
        background = [x for x in dataloader][0].to(device)
        spectral_density = SpectralDensity(
            sample_rate=sample_rate,
            fftlength=2.0,
            overlap=None,
            average='median',
        ).to(device)

        whiten = Whiten(
            fduration=2.0, sample_rate=sample_rate, highpass=f_min
        ).to(device)

        psd = spectral_density(background[..., :psd_size].double())
        kernel = background[..., psd_size:]

        # ── target SNR rescaling ──────────────────────────
        pad = int(2.0 / 2 * sample_rate)
        injected = kernel.detach().clone()

        if psd.shape[-1] != num_freqs:
            # Adding dummy dimensions for consistency
            while psd.ndim < 3:
                psd = psd[None]
            psd = torch.nn.functional.interpolate(psd, size=(num_freqs,), mode="linear")

        target_snr = PowerLaw(8,100,-3).to(device)
        waveforms = reweight_snrs(responses=waveforms,target_snrs=target_snr.sample((batch_size,)),psd=psd,sample_rate=sample_rate,highpass=f_min,)

        injected[:, :, pad:-pad] += waveforms[..., -num_samples:]
        injected_whitened = whiten(injected, psd)
        strain_td = injected_whitened.cpu().numpy()

        # compute network SNR
        network_snr = compute_network_snr(responses=waveforms, psd=psd, sample_rate=sample_rate, highpass=f_min)
        params['snr'] = network_snr

        # keep only a small window around merger
        t_start = int(0.0 * sample_rate)
        t_end   = int(55.0 * sample_rate)
        window  = strain_td[:, :, t_start:t_end]
        injected_data_list.append(window.astype(np.float32))

        chirp_mass_list.append(params['chirp_mass'])
        mass_ratio_list.append(params['mass_ratio'])
        snr_list.append(network_snr)

        total += batch_size
        if total % batch_size == 0:
                log.info(f"  ... {total}/{n_samples} generated")

    with h5py.File(out_path, "w") as f:
        f.create_dataset("injected_data", data=np.array(injected_data_list))
        f.create_dataset("chirp_mass",    data=np.array(chirp_mass_list,  dtype=np.float32))
        f.create_dataset("mass_ratio",    data=np.array(mass_ratio_list,  dtype=np.float32))
        f.create_dataset("snr",           data=np.array(snr_list,         dtype=np.float32))

    log.info(f"[Step 1] Done. Written {out_path}")
    return out_path
