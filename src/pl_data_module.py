from torch.utils.data import DataLoader, Subset
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import StratifiedKFold

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
        # cross validation
        k_fold = self.cfg.dataset.k_fold
        val_k = self.cfg.dataset.val_k
        if k_fold > 0:
            X, y = train_dataset.get_all_data()
            skf = StratifiedKFold(n_splits=k_fold, shuffle=False)
            fold_set = [(train_idx, val_idx)
                        for train_idx, val_idx in skf.split(X, y)]
            train_dataset = Subset(
                train_dataset, indices=fold_set[val_k][0])
            val_dataset = Subset(
                val_dataset, indices=fold_set[val_k][1])

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
