import sys
import os
from urllib.parse import urlparse

import mlflow
import hydra
from omegaconf import DictConfig
from pytorch_lightning.trainer import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from pl_module import LightningModuleReg
from pl_data_module import ImageDataModule
from utils.util import get_parser, read_yaml, git_commits
from utils.factory import get_transform
from utils.s3 import setup_endpoint


def train(cfg):
    setup_endpoint(cfg.server.s3_endpoint)

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.callback.checkpoint.monitor,
        save_last=cfg.callback.checkpoint.save_last,
        save_top_k=cfg.callback.checkpoint.save_top_k,
        mode=cfg.callback.checkpoint.mode,
        save_weights_only=cfg.callback.checkpoint.save_weights_only,
        filename=cfg.callback.checkpoint.filename,
        dirpath=mlflow.get_artifact_uri(),
    )

    trainer = Trainer(
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
        logger=False,
        max_epochs=cfg.general.epoch,
        gpus=cfg.general.gpus,
        precision=cfg.general.precision,
        amp_backend=cfg.general.amp_backend,
        amp_level=cfg.general.amp_level,
        accumulate_grad_batches=cfg.general.acc_grad,
        fast_dev_run=True if cfg.general.debug else False,
        limit_train_batches=0.25 if cfg.general.debug else 1.0,
        limit_val_batches=0.25 if cfg.general.debug else 1.0,
        resume_from_checkpoint=cfg.general.resume_from_checkpoint,
    )

    pl_module = LightningModuleReg(cfg=cfg)

    transforms = {
        "train": get_transform(cfg.transform.train),
        "val": get_transform(cfg.transform.val),
    }

    pl_data_module = ImageDataModule(cfg.data, transforms=transforms)

    trainer.fit(model=pl_module, datamodule=pl_data_module)


@git_commits
@hydra.main(config_path="./configs", config_name="main")
def run(cfg: DictConfig):
    print(cfg)
    seed_everything(seed=cfg.general.seed)

    mlflow.set_tracking_uri(cfg.server.mlflow_uri)
    mlflow.pytorch.autolog()
    # コンフィグを保存
    mlflow.log_artifact("./configs")

    train(cfg)


if __name__ == "__main__":
    run()
