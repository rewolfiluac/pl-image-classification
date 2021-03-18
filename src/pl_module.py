import torch
import mlflow
import numpy as np
import pytorch_lightning as pl

from utils.factory import get_model, get_loss, get_optimizer


class LightningModuleReg(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.net = get_model(self.cfg.model)
        self.loss = get_loss(self.cfg.loss)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        correct, labels_size = self.__calc_correct(y_hat, y)
        return {"val_loss": loss, "val_corr": correct, "labels_size": labels_size}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        labels_size = np.array([x["labels_size"] for x in outputs]).sum()
        val_acc = np.array([x["val_corr"] for x in outputs]).sum() / labels_size
        metrics = {
            "val_loss_mean": float(val_loss_mean.cpu().numpy()),
            "val_acc": float(val_acc),
        }
        self.log_dict(metrics, on_epoch=True, logger=False)

    def configure_optimizers(self):
        optimizer, scheduler = get_optimizer(
            cfg=self.cfg.optimizer, model_params=self.net.parameters()
        )
        return [optimizer], [scheduler]

    def __calc_correct(self, outputs, labels):
        _, predicted_indexes = torch.max(outputs.data, 1)
        labels_size = labels.size(0)
        correct = (predicted_indexes == labels).sum().item()
        return correct, labels_size


class LightningModuleInference(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.net = get_model(self.cfg.model)

    def forward(self, x):
        return self.net(x)
