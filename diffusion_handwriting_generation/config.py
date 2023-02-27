import io
from os import PathLike
import pydoc
from typing import Any, Union

from addict import Dict
from fire import Fire
from ruamel.yaml import YAML, CommentedMap


class CfgDict(Dict):
    """Modified addict.Dict class without blank {} returns while missing."""

    def __missing__(self, key):
        return None


class DLConfig:
    """
    Main config class with addict cfg for interaction and yaml one for safe dumping.

    Args:
        yaml_config (ruamel.yaml.CommentedMap): safe loaded with ruamel yaml config.
    """

    def __init__(self, yaml_config: CommentedMap):
        self.__yaml_config = yaml_config
        self.__cfg = CfgDict(yaml_config)

    def __getattr__(self, item):
        return getattr(self.__cfg, item)

    def __getitem__(self, key):
        return self.__cfg[key]

    @classmethod
    def load(cls, path: Union[PathLike, str]):
        yaml = YAML()
        with open(path, "r") as f:
            yaml_config = yaml.load(f)
        return cls(yaml_config)

    def dump(self, path: PathLike):
        yaml = YAML()

        with open(path, "w") as f:
            yaml.dump(self.__yaml_config, f)

    @property
    def pretty_text(self) -> str:
        yaml = YAML()
        buf = io.BytesIO()
        yaml.dump(self.__yaml_config, buf)
        return buf.getvalue().decode("utf-8")


def merge_configs(
    base_cfg: CommentedMap, cfg: Union[CommentedMap, dict]
) -> CommentedMap:
    """Merges base config with the new one."""
    for k, v in cfg.items():
        if isinstance(v, dict):
            if k not in base_cfg:
                base_cfg[k] = {}
            merge_configs(base_cfg[k], v)
        else:
            base_cfg[k] = v
    return base_cfg


def update_config(config: CommentedMap, params: dict) -> CommentedMap:
    """Updates base config with params from new one --config and some specified --params."""
    for k, v in params.items():
        *path, key = k.split(".")

        updating_config = config

        if path:
            for p in path:
                if p not in updating_config:
                    updating_config[p] = {}
                updating_config = updating_config[p]

        updating_config.update({key: v})
    return config


def fit_config(**kwargs) -> CommentedMap:
    """
    Loads base config and updates it with specified new one --config with some others --params.
    Also does inheritance from `base.yml`.
    """
    yaml = YAML()

    with open("./diffusion_handwriting_generation/configs/base.yml", "r") as f:
        base_config = yaml.load(f)

    if "config" in kwargs:
        cfg_name = kwargs.pop("config")
        with open(f"./diffusion_handwriting_generation/configs/{cfg_name}", "r") as f:
            yaml_config = yaml.load(f)

        merged_cfg = merge_configs(base_config, yaml_config)
    else:
        merged_cfg = base_config

    updated_cfg = update_config(merged_cfg, kwargs)
    return updated_cfg


def object_from_dict(d: CfgDict, *args, **default_kwargs) -> Any:
    """Loads python object from cfg dict with params."""
    kwargs = dict(d).copy()
    object_type = kwargs.pop("type", None)

    if object_type is None:
        raise ImportError(
            "Can't initialize any object from dict without `type` key specified",
            kwargs,
        )

    params = kwargs.pop("params", None)
    for name, value in default_kwargs.items():
        params.setdefault(name, value)

    try:
        if params is not None:
            return pydoc.locate(object_type)(*args, **params)  # noqa
        else:
            return pydoc.locate(object_type)
    except TypeError:
        raise ImportError(
            "Check module is accessible/installed and correct module params for",
            object_type,
        )


def config_entrypoint() -> DLConfig:
    """Loads config from shell arg path."""
    # kwargs only read using fire to prevent double print of cfg
    yaml_config = fit_config(**Fire(lambda **kwargs: kwargs))
    cfg = DLConfig(yaml_config)
    return cfg
