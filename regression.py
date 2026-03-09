# ─────────────────────────────────────────────────────────────
# STEP 2 – Train chirp-mass regression  (BNSReg / S4D style)
# ─────────────────────────────────────────────────────────────
#
# This module is a thin shim so that main.py can continue to call
#   from regression import train_regression
#
# All implementation lives in the step2/ package, which mirrors the
# Lightning architecture from kyoon-mit/BNSReg.

from step2.train import train_regression  # noqa: F401
