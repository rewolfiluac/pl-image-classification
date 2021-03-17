from torch.nn import functional
from timm import create_model, list_models
from omegaconf.listconfig import ListConfig

import optimizers
import schedulers
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
            in_chans=cfg.in_chans,
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
    try:
        optimizer = getattr(optimizers, cfg.optimizer.optim_name)(
            model_params, **cfg.optimizer.params
        )
    except AttributeError:
        raise Exception("Not Found Optimizer Base: {}".format(cfg.optimizer.base))
    # scheduler
    try:
        scheduler = getattr(schedulers, cfg.scheduler.name)(
            optimizer, **cfg.scheduler.params
        )
    except AttributeError:
        raise Exception("Not Found Scheduler: {}".format(cfg.scheduler.name))

    return optimizer, scheduler


def get_transform(transform_cfg):
    return __build_transform(transform_cfg)


def __build_transform(transform_cfg):
    name = transform_cfg.name
    params = transform_cfg.params
    print(type(params))
    if type(params) == dict:
        return getattr(transforms, name)(**params)
    elif type(params) == ListConfig:
        return getattr(transforms, name)([__build_transform(aug) for aug in params])
    else:
        raise Exception(
            "Illegal Values: aug_name:{}, aug_params:{}".format(name, params)
        )
