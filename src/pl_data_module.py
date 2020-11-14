from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from dataset import ImageDataset


class ImageDataModule(LightningDataModule):
    def __init__(self, cfg, transforms):
        super().__init__()
        self.cfg = cfg
        self.transforms = transforms

    def prepare_data(self):
        train_dataset = ImageDataset(
            dataset_cfg=self.cfg.dataset,
            transform=self.transforms["train"]
        )
        val_dataset = ImageDataset(
            dataset_cfg=self.cfg.dataset,
            transform=self.transforms["val"]
        )

        self.datasets = {
            "train": train_dataset,
            "val": val_dataset,
        }

    def train_dataloader(self):
        return DataLoader(
            dataset=self.datasets["train"],
            batch_size=self.cfg.dataloader.batch_size,
            num_workers=self.cfg.dataloader.num_workers,
            shuffle=self.cfg.dataloader.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.datasets["val"],
            batch_size=self.cfg.dataloader.batch_size,
            num_workers=self.cfg.dataloader.num_workers,
            shuffle=False
        )
