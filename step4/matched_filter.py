"""
Step 4 – Per-event matched filtering with gstlal
=================================================
Loads the pre-whitened HDF5 strain produced by step 1, writes each
event to a temporary GWF file, then runs gstlal_inspiral using the
per-event filtered template banks from step 3.

Supports local execution and HTCondor DAG submission.
"""

import logging
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np

log = logging.getLogger(__name__)


# ── HDF5 → GWF conversion ─────────────────────────────────────────────────────

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


def _load_event(fpath: Path, local_idx: int) -> np.ndarray:
    """Return whitened strain array (n_ifos, seq_len) for one event."""
    with h5py.File(fpath, "r") as f:
        return f["injected_data"][local_idx].astype(np.float64)


def _write_gwf(
    strain: np.ndarray,
    ifo: str,
    channel: str,
    sample_rate: int,
    gps_start: int,
    out_path: Path,
) -> None:
    """
    Write a single IFO's whitened strain to a GWF file using gwpy.

    Parameters
    ----------
    strain:      1-D float64 array (seq_len,)
    ifo:         detector prefix, e.g. "H1"
    channel:     channel suffix, e.g. "GDS-CALIB_STRAIN"
    sample_rate: samples per second
    gps_start:   GPS start time assigned to the segment
    out_path:    destination .gwf file
    """
    from gwpy.timeseries import TimeSeries

    channel_name = f"{ifo}:{channel}"
    ts = TimeSeries(
        strain,
        sample_rate=sample_rate,
        t0=gps_start,
        channel=channel_name,
        unit="dimensionless",
    )
    ts.write(str(out_path), format="gwf")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_matched_filter_per_event(cfg: dict, event_banks: list[dict]) -> None:
    """
    For each event, convert the pre-whitened HDF5 strain to a temporary GWF
    file and run gstlal_inspiral using the event's dedicated filtered bank.

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

    # Resolve the pre-whitened test data directory from step 1.
    data_dir = Path(cfg.get("data_dir", "data"))
    test_dir = Path(mf_cfg.get("test_dir", str(data_dir / "test")))

    sample_rate   = int(cfg.get("sample_rate", 512))
    gps_start     = int(mf_cfg.get("gps_start", 1000000000))
    ifos          = mf_cfg.get("ifos", ["H1", "L1"])
    channel_names = mf_cfg.get(
        "channel_names",
        {ifo: "GDS-CALIB_STRAIN" for ifo in ifos},
    )

    test_files = sorted(test_dir.glob("*.h5"))
    if not test_files:
        log.error("[Step 4] No HDF5 files found in %s", test_dir)
        return

    log.info(
        "[Step 4] Running per-event matched filtering for %d events "
        "using pre-whitened HDF5 data from %s …",
        len(event_banks), test_dir,
    )

    for event in event_banks:
        event_id  = event["event_id"]
        bank_path = event["bank_path"]
        mc_pred   = event["mc_pred"]

        event_out = output_dir / f"event_{event_id:06d}"
        event_out.mkdir(exist_ok=True)

        log.info(
            "  event %06d | mc_pred=%.4f | bank=%s",
            event_id, mc_pred, bank_path.name,
        )

        # Load the whitened strain for this event.
        try:
            fpath, local_idx = _locate_event(test_files, event_id)
            strain = _load_event(fpath, local_idx)   # (n_ifos, seq_len)
        except Exception as exc:
            log.error("    Could not load event %d: %s", event_id, exc)
            continue

        duration_s = strain.shape[-1] / sample_rate
        gps_end    = gps_start + int(duration_s)

        if mf_cfg.get("run_locally", True):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                # Write one GWF file per IFO, then call gstlal_inspiral.
                for ifo_idx, ifo in enumerate(ifos):
                    if ifo_idx >= strain.shape[0]:
                        log.warning("    IFO %s not present in strain — skipping", ifo)
                        continue

                    channel = channel_names.get(ifo, "GDS-CALIB_STRAIN")
                    gwf_path = tmp / f"{ifo}_strain.gwf"

                    _write_gwf(
                        strain[ifo_idx], ifo, channel,
                        sample_rate, gps_start, gwf_path,
                    )

                    out_file = event_out / f"triggers_{ifo}.xml.gz"
                    cmd = [
                        "gstlal_inspiral",
                        "--psd-fft-length",    str(mf_cfg.get("psd_fft_length",    16)),
                        "--ht-gate-threshold", str(mf_cfg.get("ht_gate_threshold", 100)),
                        "--svd-tolerance",     str(mf_cfg.get("svd_tolerance",  0.9999)),
                        "--bank-file",         str(bank_path),
                        "--ifo",               ifo,
                        "--channel-name",      f"{ifo}={channel}",
                        "--frame-files",       str(gwf_path),
                        "--gps-start-time",    str(gps_start),
                        "--gps-end-time",      str(gps_end),
                        "--output",            str(out_file),
                    ]
                    log.info("    %s ← %s (GPS %d–%d)", ifo, gwf_path.name,
                             gps_start, gps_end)
                    _run(cmd)

        elif mf_cfg.get("use_condor", False):
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
            log.info("    DAG written to %s — submit manually.", event_out)

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
    log.info("  Config written → %s", out_path)


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("Command failed:\n%s", result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    if result.stdout:
        log.debug(result.stdout)
