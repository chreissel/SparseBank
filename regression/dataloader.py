# Mirrors BNSReg's src/BNSReg/dataloader/regression_loader.py
# Adapted for SparseBank's SparseRegressionConfig.

import lightning as L
from torch.utils.data import DataLoader

from step2.dataset import BNSDatasetRegression, SparseRegressionConfig


class LitBNSDataRegression(L.LightningDataModule):
    """Lightning DataModule for BNS chirp-mass regression.

    Mirrors BNSReg's LitBNSDataRegression.
    """

    def __init__(self, cfg: SparseRegressionConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

    def setup(self, stage: str | None = None):
        if stage == "fit":
            self.train_dataset = BNSDatasetRegression("train", self.cfg)
            self.val_dataset   = BNSDatasetRegression("val",   self.cfg)
        elif stage in ("test", "predict"):
            self.test_dataset  = BNSDatasetRegression("test",  self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.cfg.dataloader_kwargs("train"))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.cfg.dataloader_kwargs("val"))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.cfg.dataloader_kwargs("test"))

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, **self.cfg.dataloader_kwargs("test"))
