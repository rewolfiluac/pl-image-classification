import mlflow
from pytorch_lightning.trainer import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from pl_module import LightningModuleReg
from pl_data_module import ImageDataModule
from utils.util import get_parser, read_yaml
from utils.factory import get_transform


def train(cfg):
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.callback.checkpoint.monitor,
        save_last=cfg.callback.checkpoint.save_last,
        save_top_k=cfg.callback.checkpoint.save_top_k,
        mode=cfg.callback.checkpoint.mode,
        save_weights_only=cfg.callback.checkpoint.save_weights_only,
        filename=cfg.callback.checkpoint.filename
    )

    trainer = Trainer(
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
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


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = read_yaml(path=args.config)
    seed_everything(seed=cfg.general.seed)
    mlflow.pytorch.autolog()
    mlflow.log_artifact(args.config)
    train(cfg)
