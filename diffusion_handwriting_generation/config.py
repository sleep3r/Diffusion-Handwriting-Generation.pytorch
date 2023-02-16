# Standard Library
import io
from os import PathLike
import pydoc
from typing import Any, Optional

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
    def load(cls, path: PathLike | str):
        yaml = YAML()
        yaml_config = yaml.load(path)
        return cls(yaml_config)

    @property
    def pretty_text(self) -> str:
        yaml = YAML()
        buf = io.BytesIO()
        yaml.dump(self.__yaml_config, buf)
        return buf.getvalue().decode("utf-8")

    def dump(self, path: PathLike):
        yaml = YAML()

        with open(path, "w") as f:
            yaml.dump(self.__yaml_config, f)


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


def fit(**kwargs) -> CommentedMap:
    """
    Loads base config and updates it with specified new one --config with some others --params.
    """
    yaml = YAML()

    with open("./ocr_ner_train/configs/base.yml", "r") as f:
        base_config = yaml.load(f)

    if "config" in kwargs:
        cfg_name = kwargs.pop("config")
        with open(f"./ocr_ner_train/configs/{cfg_name}", "r") as f:
            cfg_yaml = yaml.load(f)

        merged_cfg = update_config(base_config, cfg_yaml)
    else:
        merged_cfg = base_config
    update_cfg = update_config(merged_cfg, kwargs)
    return update_cfg


def object_from_dict(d: Optional[CfgDict], parent=None, **default_kwargs) -> Any:
    """Loads python object from cfg dict with params."""
    kwargs = dict(d).copy()

    object_type = kwargs.pop("type", None)
    if object_type is not None:
        params = kwargs.pop("params", None)

        for name, value in default_kwargs.items():
            params.setdefault(name, value)

        if parent is not None:
            if params is not None:
                return getattr(parent, object_type)(**params)
            else:
                return getattr(parent, object_type)
        else:
            try:
                if params is not None:
                    return pydoc.locate(object_type)(**params)  # noqa
                else:
                    return pydoc.locate(object_type)
            except TypeError:
                raise ImportError(
                    "Check module is accessible/installed and correct module params for",
                    object_type,
                )


def config_entrypoint() -> DLConfig:
    yaml_config: CommentedMap = fit(
        **Fire(lambda **kwargs: kwargs)
    )  # dirty hack to prevent double print of cfg
    cfg = DLConfig(yaml_config)
    return cfg
