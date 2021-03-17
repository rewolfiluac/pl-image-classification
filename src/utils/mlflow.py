import tempfile

import mlflow
from omegaconf import DictConfig, OmegaConf


def artifacts_omegaconf(conf: DictConfig):
    with tempfile.TemporaryDirectory() as dp:
        print(dp)
        OmegaConf.to_yaml(conf)
        # mlflow.log_artifact()
