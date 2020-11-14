import torch
from torch.nn import functional
from timm import create_model, list_models
import torch_optimizer as t_optim

import transforms


def get_model(cfg):
    if cfg.base == "timm":
        if cfg.model_name not in list_models(pretrained=cfg.pretrained):
            print(list_models(pretrained=cfg.pretrained))
            assert True, "Not Found Model: {}".format(cfg.model_name)
        net = create_model(
            model_name=cfg.model_name,
            pretrained=cfg.pretrained,
            num_classes=cfg.num_classes,
            in_chans=cfg.in_chans
        )
        return net
    else:
        assert True, "Not Found Model Base: {}".format(cfg.base)


def get_loss(cfg):
    if cfg.base == "torch":
        return getattr(functional, cfg.name)
    else:
        assert True, "Not Found Loss Base: {}".format(cfg.base)


def get_optimizer(cfg, model_params):
    # optimizer
    if cfg.optimizer.base == "torch_optimizer":
        optimizer = getattr(t_optim, cfg.optimizer.optim_name)(
            model_params, **cfg.optimizer.params
        )
    else:
        assert True, "Not Found Optimizer Base: {}".format(cfg.optimizer.base)

    # scheduler
    if cfg.scheduler.base == "torch":
        scheduler = getattr(torch.optim.lr_scheduler, cfg.scheduler.sche_name)(
            optimizer, **cfg.scheduler.params
        )
    else:
        assert True, "Not Found Scheduler Base: {}".format(cfg.optimizer.base)
    return optimizer, scheduler


def get_transform(transform_cfg):
    return __build_transform(transform_cfg)


def __build_transform(transform_cfg):
    if type(transform_cfg) == str:
        return getattr(transforms, transform_cfg)()
    for key, val in transform_cfg.items():
        if isinstance(val, list):
            return getattr(transforms, key)(
                [__build_transform(cfg) for cfg in val])
        elif isinstance(val, dict):
            return getattr(transforms, key)(**val)
        else:
            raise Exception("Illegal Values: {}, {}".format(key, val))
