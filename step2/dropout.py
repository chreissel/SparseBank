# Derived from state-spaces/s4
# https://github.com/state-spaces/s4
#
# Copyright (c) 2023 The S4 Authors
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Modified by Kyungseop Yoon (kyoon@mit.edu), 2026-01-14
#   * Removed unnecessary imports.
# - Adapted for SparseBank step2 by removing BNSReg package dependency.

"""DropoutNd utility used by the S4D model."""

import torch
import torch.nn as nn
from einops import rearrange


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie: bool = True, transposed: bool = True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(
                "dropout probability has to be in [0, 1), but got {}".format(p)
            )
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed:
                X = rearrange(X, "b ... d -> b d ...")
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            mask = torch.rand(*mask_shape, device=X.device) < 1.0 - self.p
            X = X * mask * (1.0 / (1 - self.p))
            if not self.transposed:
                X = rearrange(X, "b d ... -> b ... d")
            return X
        return X
