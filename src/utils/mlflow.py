import tempfile
from pathlib import Path

import mlflow
from omegaconf import DictConfig, OmegaConf


def artifacts_omegaconf(conf: DictConfig):
    with tempfile.TemporaryDirectory() as dp:
        p = Path(dp) / "main.yaml"
        OmegaConf.save(conf, str(p))
        mlflow.log_artifact(str(p))
