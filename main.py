# ─────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────

import argparse
import logging
import yaml
from pathlib import Path

from step1.data import generate_dataset
from step2 import train_regression
from step3 import filter_bank
from step4 import run_matched_filter_per_event
from step5 import run_sensitive_volume_analysis, download_pipeline_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="GW Analysis Pipeline")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config (optional; defaults used otherwise)")
    parser.add_argument("--steps", default="1,2,3,4",
                        help="Comma-separated list of steps to run (default: 1,2,3,4). "
                             "Step 5 computes the sensitive volume at FAR=1/year and "
                             "downloads public matched-filter results from GWOSC.")
    args = parser.parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as fh:
            cfg.update(yaml.safe_load(fh))

    steps = [int(s) for s in args.steps.split(",")]
    log.info("Running steps: %s", steps)

    ckpt_path   = None
    event_banks = None

    # ── Step 1: Data generation ───────────────────────────────
    if 1 in steps:
        for split in ("train", "val", "test"):
            generate_dataset(cfg, split)

    # ── Step 2: Model training ────────────────────────────────
    if 2 in steps:
        ckpt_path = train_regression(cfg)

    # ── Step 3: Per-event template bank filtering ─────────────
    if 3 in steps:
        if ckpt_path is None:
            # Locate the most recent checkpoint produced by step 2
            ckpt_dir   = Path(cfg["training"]["checkpoint_dir"])
            candidates = sorted(ckpt_dir.glob("*.ckpt"))
            ckpt_path  = candidates[-1] if candidates else Path("checkpoints/final_model.ckpt")
            log.info("[Step 3] Using checkpoint: %s", ckpt_path)
        event_banks = filter_bank(cfg, ckpt_path)

    # ── Step 4: Matched filtering ─────────────────────────────
    if 4 in steps:
        if event_banks is None:
            log.warning("[Step 4] No per-event banks from step 3; "
                        "step 4 will use the full input bank.")
        run_matched_filter_per_event(cfg, event_banks or [])

    # ── Step 5: Sensitive volume & GWOSC download ─────────────
    if 5 in steps:
        log.info("[Step 5] Downloading public matched-filter results from GWOSC …")
        download_pipeline_results(cfg)
        log.info("[Step 5] Computing sensitive volume at FAR = 1/year …")
        sv_results = run_sensitive_volume_analysis(cfg)
        log.info(
            "[Step 5] V_T = %.3e ± %.3e Mpc^3  (p_det=%.4f, N_found=%d/%d)",
            sv_results["v_t_mpc3"],
            sv_results["sigma_v_mpc3"],
            sv_results["p_det"],
            sv_results["n_found"],
            sv_results["n_injections"],
        )

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
