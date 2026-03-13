# This file is derived from the S4 repository:
#   https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py
#   https://github.com/state-spaces/s4/blob/main/examples.py
#
# Copyright (c) 2023 The S4 Authors
# Licensed under the Apache License, Version 2.0
#
# Modifications (c) 2026 Kyungseop Yoon (kyoon@mit.edu), 2026-01-14
#   - Adjusted to use local DropoutNd from step2.dropout.
#   - Replaced use of Python complex literals (e.g. `1j`) with
#     `torch.complex(...)` to ensure compatibility with `torch.compile`.
#
# Adapted for SparseBank step2.

"""Minimal S4D model following the BNSReg architecture."""

import math
import torch
import torch.nn as nn
from einops import repeat

from step2.dropout import DropoutNd


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()

        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), "n -> h n", h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """Returns: (..., c, L) where c is number of channels (default 1)."""
        dt = torch.exp(self.log_dt)           # (H,)
        C = torch.view_as_complex(self.C)     # (H, N//2)

        # torch.compile-safe: use torch.complex instead of 1j literal
        A = torch.complex(-torch.exp(self.log_A_real), self.A_imag)  # (H, N//2)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)            # (H, N//2)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H, N//2, L)
        C = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum("hn, hnl -> hl", C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay."""
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        # Position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):
        """Input and output shape (B, H, L)."""
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L)                          # (H, L)

        # Convolution via FFT
        k_f = torch.fft.rfft(k, n=2 * L)             # (H, L)
        u_f = torch.fft.rfft(u, n=2 * L)             # (B, H, L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B, H, L)

        # Skip connection (D term)
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y, None  # dummy state for interface compatibility


class S4Model(nn.Module):
    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        d_state=64,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        lr=None,
        dt_min=0.001,
        dt_max=0.1,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4D layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(
                    d_model,
                    d_state=d_state,
                    dropout=dropout,
                    transposed=True,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    lr=lr,
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(DropoutNd(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """Input x is shape (B, L, d_input)."""
        x = self.encoder(x)            # (B, L, d_model)
        x = x.transpose(-1, -2)        # (B, d_model, L)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            z, _ = layer(z)
            z = dropout(z)
            x = z + x

            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)        # (B, L, d_model)
        x = x.mean(dim=1)              # (B, d_model)  — average pooling
        x = self.decoder(x)            # (B, d_output)
        return x
