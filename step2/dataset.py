# Adapted from BNSReg (kyoon-mit/BNSReg) to handle the HDF5 format
# produced by SparseBank's step1/data.py.
#
# Step1 stores data in batched form:
#   injected_data : (n_gen_batches, gen_batch_size, n_ifos, seq_len)
#   chirp_mass    : (n_gen_batches, gen_batch_size)
#   mass_ratio    : (n_gen_batches, gen_batch_size)
#   snr           : (n_gen_batches, gen_batch_size)
#
# OR in already-flattened form:
#   injected_data : (n_samples, n_ifos, seq_len)
#
# The dataset handles both shapes transparently.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple

import h5py
import torch
from torch.utils.data import Dataset

Stage = Literal["train", "val", "test"]

_VALID_VARIABLES = frozenset(
    {"chirp_mass", "mass_ratio", "snr", "chi1", "chi2", "distance",
     "inclination", "dec", "psi", "phi", "phic", "mass_1", "mass_2",
     "s1z", "s2z"}
)

_DTYPE_MAP = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
}


@dataclass(frozen=True)
class SparseRegressionConfig:
    """Configuration for the step2 dataset / dataloader.

    Mirrors BNSReg's BNSDataModuleRegressionConfig but adapted to the
    dict-based config.yaml of SparseBank.
    """

    train_file: str
    val_file: str
    test_file: str

    target_variables: Tuple[str, ...] = ("chirp_mass",)
    observed_variables: Tuple[str, ...] = ("snr",)

    injected_data_key: str = "injected_data"

    # Dataloader
    train_batch_size: int = 32
    val_batch_size: int = 32
    test_batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = None

    # Precision
    strain_precision: str = "torch.float32"
    variables_precision: str = "torch.float32"

    def dataloader_kwargs(self, stage: Stage) -> dict:
        if stage == "train":
            return {
                "batch_size": self.train_batch_size,
                "shuffle": self.shuffle,
                "num_workers": self.num_workers,
                "persistent_workers": self.persistent_workers,
                "prefetch_factor": self.prefetch_factor,
            }
        elif stage in ("val", "test"):
            return {
                "batch_size": self.val_batch_size if stage == "val" else self.test_batch_size,
                "shuffle": False,
                "num_workers": self.num_workers,
                "persistent_workers": self.persistent_workers,
                "prefetch_factor": self.prefetch_factor,
            }
        raise ValueError(f"Unknown stage: {stage!r}")

    @classmethod
    def from_sparsebank_cfg(cls, cfg: dict) -> "SparseRegressionConfig":
        """Construct from SparseBank's top-level YAML config dict."""
        data_cfg = cfg["data"]
        data_dir = data_cfg["data_dir"]
        bs = data_cfg.get("batch_size", 32)
        nw = data_cfg.get("num_workers", 4)

        return cls(
            train_file=f"{data_dir}/train/sig_combined_train.h5",
            val_file=f"{data_dir}/val/sig_combined_val.h5",
            test_file=f"{data_dir}/test/sig_combined_test.h5",
            train_batch_size=bs,
            val_batch_size=bs,
            test_batch_size=bs,
            num_workers=nw,
            persistent_workers=nw > 0,
            prefetch_factor=2 if nw > 0 else None,
        )


class BNSDatasetRegression(Dataset):
    """PyTorch Dataset for step2 BNS regression.

    Mirrors BNSReg's BNSDatasetRegression.  Returns 3-tuples:
        (X_sequence, y_target, z_observed)

    X_sequence : (n_ifos, seq_len)   — raw whitened strain per detector
    y_target   : (len(target_variables),)
    z_observed : (len(observed_variables),)
    """

    def __init__(self, stage: Stage, cfg: SparseRegressionConfig):
        self.cfg = cfg
        self.stage = stage
        self._file_handle: h5py.File | None = None

        self.file_path = {
            "train": cfg.train_file,
            "val": cfg.val_file,
            "test": cfg.test_file,
        }[stage]

        self.strain_dtype = _DTYPE_MAP[cfg.strain_precision]
        self.var_dtype = _DTYPE_MAP[cfg.variables_precision]

        # Inspect shape once to determine indexing mode
        with h5py.File(self.file_path, "r") as f:
            shape = f[cfg.injected_data_key].shape

        if len(shape) == 4:
            # step1 batched format: (n_gen_batches, gen_batch_size, n_ifos, seq_len)
            self._4d = True
            self._n_gen_batches, self._gen_batch_size, self.n_ifos, self.seq_len = shape
            self.n_samples = self._n_gen_batches * self._gen_batch_size
        elif len(shape) == 3:
            # Already flattened: (n_samples, n_ifos, seq_len)
            self._4d = False
            self.n_samples, self.n_ifos, self.seq_len = shape
        else:
            raise ValueError(
                f"injected_data has unexpected shape {shape}; "
                "expected 3-D (n_samples, n_ifos, seq_len) or "
                "4-D (n_gen_batches, gen_batch_size, n_ifos, seq_len)."
            )

    # ------------------------------------------------------------------
    # HDF5 lazy file handle (opened once per worker, never closed)
    # ------------------------------------------------------------------
    def _get_file(self) -> h5py.File:
        if self._file_handle is None:
            self._file_handle = h5py.File(self.file_path, "r")
        return self._file_handle

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        f = self._get_file()

        if self._4d:
            bi, si = divmod(idx, self._gen_batch_size)
            x_raw = f[self.cfg.injected_data_key][bi, si]   # (n_ifos, seq_len)
            y_raw = {k: f[k][bi, si] for k in (*self.cfg.target_variables,
                                                *self.cfg.observed_variables)}
        else:
            x_raw = f[self.cfg.injected_data_key][idx]       # (n_ifos, seq_len)
            y_raw = {k: f[k][idx] for k in (*self.cfg.target_variables,
                                             *self.cfg.observed_variables)}

        X_sequence = torch.as_tensor(x_raw, dtype=self.strain_dtype)   # (n_ifos, seq_len)

        y_target = torch.stack([
            torch.as_tensor(y_raw[k], dtype=self.var_dtype)
            for k in self.cfg.target_variables
        ]) if self.cfg.target_variables else torch.empty(0, dtype=self.var_dtype)

        z_observed = torch.stack([
            torch.as_tensor(y_raw[k], dtype=self.var_dtype)
            for k in self.cfg.observed_variables
        ]) if self.cfg.observed_variables else torch.empty(0, dtype=self.var_dtype)

        return X_sequence, y_target, z_observed
