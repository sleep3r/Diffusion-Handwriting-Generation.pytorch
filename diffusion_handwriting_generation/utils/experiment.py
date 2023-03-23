from datetime import datetime
from getpass import getuser
import json
import logging
import os
from pathlib import Path
import random
from socket import gethostname

import git
import numpy as np
import torch

from diffusion_handwriting_generation.config import DLConfig
from diffusion_handwriting_generation.utils.env import collect_env
from diffusion_handwriting_generation.utils.log import get_logger
from diffusion_handwriting_generation.utils.path import mkdir_or_exist


def create_workdir(cfg: DLConfig, meta: dict) -> dict:
    """
    Creates working directory for artifacts storage.

    Args:
        cfg (DLConfig): config object;
        meta (dict): meta dictionary.
    Returns:
        dict: updated meta dictionary.
    """
    dirname = f"{cfg.experiment.name}/{datetime.now().strftime('%d.%m/%H.%M.%S')}"
    meta["run_name"] = dirname
    meta["exp_dir"] = Path(cfg.experiment.work_dir) / dirname
    mkdir_or_exist(meta["exp_dir"])
    return meta


def env_collect(meta: dict, logger: logging.Logger) -> dict:
    env_info_dict = collect_env()
    env_info = "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"

    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)

    repo = git.Repo(search_parent_directories=True)
    meta["sha"] = repo.head.object.hexsha
    meta["host_name"] = f"{getuser()}@{gethostname()}"
    return meta


def set_random_seed(
    seed: int = 42,
    precision: int = 10,
    deterministic: bool = False,
) -> None:
    """
    Seeds all.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for CUDNN backend,
                              i.e., set `torch.backends.cudnn.deterministic` to True and
                              `torch.backends.cudnn.benchmark` to False. Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_printoptions(precision=precision)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True  # noqa
        torch.backends.cudnn.benchmark = False  # noqa


def determine_exp(cfg: DLConfig, meta: dict, logger: logging.Logger) -> dict:
    """Sets seed and experiment name."""
    if cfg.experiment.seed is not None:
        logger.info(
            f"Set random seed to {cfg.experiment.seed}, deterministic: False \n",
        )
        set_random_seed(
            cfg.experiment.seed,
            precision=cfg.experiment.precision,
            deterministic=False,
        )

    meta["seed"] = cfg.experiment.seed
    meta["exp_name"] = cfg.experiment.name
    return meta


def log_artifacts(cfg: DLConfig, meta: dict) -> None:
    # dump and log config
    cfg.dump(meta["exp_dir"] / "config.yml")

    # dump and log report json with meta info
    with open(meta["exp_dir"] / "report.json", "w") as f:
        meta["exp_dir"] = str(meta["exp_dir"])  # json serialization doesn't like Path
        json.dump(meta, f, indent=4)


def prepare_exp(cfg: DLConfig) -> tuple[dict, logging.Logger]:
    # init the meta dict to record some important information
    # such as environment info and seed, which will be logged
    meta: dict = {}

    # create work_dir
    meta = create_workdir(cfg, meta)

    # init the logger before other steps
    logger = get_logger("train", cfg, meta)

    # log env info
    meta = env_collect(meta, logger=logger)

    # set random seed, exp name
    meta = determine_exp(cfg, meta, logger=logger)
    return meta, logger
