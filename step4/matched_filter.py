"""
Step 4 – Per-event matched filtering with gstlal
=================================================
Runs gstlal_inspiral using the per-event filtered template banks produced
by step 3.  Supports local execution and HTCondor DAG submission.
"""

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def run_matched_filter_per_event(cfg: dict, event_banks: list[dict]) -> None:
    """
    For each event, run gstlal_inspiral using the event's dedicated
    filtered template bank.

    event_banks: list of dicts returned by step3's filter_bank()
    """
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


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"Command failed:\n{result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    if result.stdout:
        log.debug(result.stdout)
