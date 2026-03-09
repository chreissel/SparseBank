# ─────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────

import os
import argparse
import logging
import yaml
import h5py
import numpy as np
from pathlib import Path

from step1.data import generate_dataset
from step2 import train_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="GW Analysis Pipeline")
    parser.add_argument("--config",  default='config.yaml',
                        help="Path to YAML config (optional; defaults used otherwise)")
    parser.add_argument("--steps",   default="1,2,3,4",
                        help="Comma-separated list of steps to run (default: 1,2,3,4)")
    args = parser.parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as fh:
            user_cfg = yaml.safe_load(fh)
        # shallow merge top-level keys
        cfg.update(user_cfg)

    steps = [int(s) for s in args.steps.split(",")]
    log.info(f"Running steps: {steps}")

    ckpt_path  = None
    bank_path  = Path(cfg["bank_filter"]["output_bank"])

    if 1 in steps:
        for split in ("train", "val", "test"):
            generate_dataset(cfg, split)

    if 2 in steps:
        ckpt_path = train_regression(cfg)

    if 3 in steps:
        if ckpt_path is None:
            # try to find an existing checkpoint
            ckpt_dir  = Path(cfg["training"]["checkpoint_dir"])
            candidates = sorted(ckpt_dir.glob("*.ckpt"))
            ckpt_path  = candidates[-1] if candidates else Path("checkpoints/final_model.ckpt")
            log.info(f"[Step 3] Using checkpoint: {ckpt_path}")
        bank_path = filter_template_bank(cfg, ckpt_path)

    if 4 in steps:
        run_matched_filter(cfg, bank_path)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()

