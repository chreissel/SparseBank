# ─────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────

import argparse
import logging
import yaml

from data_generation.data import generate_dataset
from regression import train_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="GW Analysis Pipeline")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config (optional; defaults used otherwise)")
    parser.add_argument("--steps", default="1,2",
                        help="Comma-separated list of steps to run (default: 1,2).")
    args = parser.parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as fh:
            cfg.update(yaml.safe_load(fh))

    steps = [int(s) for s in args.steps.split(",")]
    log.info("Running steps: %s", steps)

    # ── Step 1: Data generation ───────────────────────────────
    if 1 in steps:
        for split in ("train", "val", "test"):
            generate_dataset(cfg, split)

    # ── Step 2: Model training ────────────────────────────────
    if 2 in steps:
        train_regression(cfg)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
