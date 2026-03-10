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
#
# Multiple *.h5 files with the same structure can live in a single
# directory; the dataset discovers all of them and presents them as one
# concatenated dataset.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

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

    Each of *_dir points to a folder that contains one or more *.h5 files
    sharing the same HDF5 key structure.  All files in a folder are treated
    as a single concatenated dataset.
    """

    train_dir: str
    val_dir: str
    test_dir: str

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
            train_dir=f"{data_dir}/train",
            val_dir=f"{data_dir}/val",
            test_dir=f"{data_dir}/test",
            train_batch_size=bs,
            val_batch_size=bs,
            test_batch_size=bs,
            num_workers=nw,
            persistent_workers=nw > 0,
            prefetch_factor=2 if nw > 0 else None,
        )


class BNSDatasetRegression(Dataset):
    """PyTorch Dataset for step2 BNS regression.

    Discovers all *.h5 files inside *dir_path* (sorted for reproducibility)
    and presents them as a single concatenated dataset.  Every file must
    share the same HDF5 key structure and spatial dimensions
    (n_ifos, seq_len).

    Both the step1 batched format (4-D) and the already-flattened format
    (3-D) are supported and may be mixed across files in the same folder.

    Returns 3-tuples:
        (X_sequence, y_target, z_observed)

    X_sequence : (n_ifos, seq_len)   — raw whitened strain per detector
    y_target   : (len(target_variables),)
    z_observed : (len(observed_variables),)
    """

    def __init__(self, stage: Stage, cfg: SparseRegressionConfig):
        self.cfg = cfg
        self.stage = stage

        folder = {
            "train": cfg.train_dir,
            "val":   cfg.val_dir,
            "test":  cfg.test_dir,
        }[stage]

        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.folder}")

        self.file_paths: List[Path] = sorted(self.folder.glob("*.h5"))
        if not self.file_paths:
            raise FileNotFoundError(f"No *.h5 files found in {self.folder}")

        self.strain_dtype = _DTYPE_MAP[cfg.strain_precision]
        self.var_dtype = _DTYPE_MAP[cfg.variables_precision]

        # Inspect every file once to build the cumulative index table.
        self._file_meta: List[dict] = []
        self._cumulative_sizes: List[int] = []  # cumulative n_samples

        n_ifos: int | None = None
        seq_len: int | None = None
        cumsum = 0

        for fp in self.file_paths:
            with h5py.File(fp, "r") as f:
                shape = f[cfg.injected_data_key].shape

            if len(shape) == 4:
                nb, gb, fi, sl = shape
                n_samples = nb * gb
                meta: dict = {"4d": True, "n_gen_batches": nb, "gen_batch_size": gb}
            elif len(shape) == 3:
                n_samples, fi, sl = shape
                meta = {"4d": False}
            else:
                raise ValueError(
                    f"injected_data in {fp} has unexpected shape {shape}; "
                    "expected 3-D (n_samples, n_ifos, seq_len) or "
                    "4-D (n_gen_batches, gen_batch_size, n_ifos, seq_len)."
                )

            if n_ifos is None:
                n_ifos, seq_len = fi, sl
                self.n_ifos = n_ifos
                self.seq_len = seq_len
            elif (fi, sl) != (n_ifos, seq_len):
                raise ValueError(
                    f"Shape mismatch in {fp}: (n_ifos={fi}, seq_len={sl}) "
                    f"but previous files had ({n_ifos}, {seq_len})."
                )

            meta["n_samples"] = n_samples
            self._file_meta.append(meta)
            cumsum += n_samples
            self._cumulative_sizes.append(cumsum)

        self.n_samples = cumsum
        # Lazy HDF5 file handles — one dict per worker process.
        self._file_handles: Dict[int, h5py.File] = {}

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------
    def _locate(self, idx: int) -> Tuple[int, int]:
        """Map a global sample index to (file_index, local_index)."""
        lo, hi = 0, len(self._cumulative_sizes)
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._cumulative_sizes[mid]:
                hi = mid
            else:
                lo = mid + 1
        file_idx = lo
        offset = self._cumulative_sizes[file_idx - 1] if file_idx > 0 else 0
        return file_idx, idx - offset

    # ------------------------------------------------------------------
    # HDF5 lazy file handles (opened once per worker, never closed)
    # ------------------------------------------------------------------
    def _get_file(self, file_idx: int) -> h5py.File:
        if file_idx not in self._file_handles:
            self._file_handles[file_idx] = h5py.File(self.file_paths[file_idx], "r")
        return self._file_handles[file_idx]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        file_idx, local_idx = self._locate(idx)
        meta = self._file_meta[file_idx]
        f = self._get_file(file_idx)

        if meta["4d"]:
            bi, si = divmod(local_idx, meta["gen_batch_size"])
            x_raw = f[self.cfg.injected_data_key][bi, si]   # (n_ifos, seq_len)
            y_raw = {k: f[k][bi, si] for k in (*self.cfg.target_variables,
                                                *self.cfg.observed_variables)}
        else:
            x_raw = f[self.cfg.injected_data_key][local_idx]  # (n_ifos, seq_len)
            y_raw = {k: f[k][local_idx] for k in (*self.cfg.target_variables,
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
