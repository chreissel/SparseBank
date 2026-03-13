# step2 – BNS regression training package
# Mirrors the Lightning architecture from kyoon-mit/BNSReg.
#
# Public API consumed by regression.py:
#   from step2 import train_regression

from regression.train import train_regression

__all__ = ["train_regression"]
