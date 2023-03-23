import logging
import sys

import pandas as pd

from diffusion_handwriting_generation.config import DLConfig

logger_initialized: dict = {}


def get_logger(name, cfg: DLConfig, meta: dict, log_level=logging.INFO):
    """
    Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        cfg (DLConfig): config;
        meta (dict): meta info dict;
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
    Returns:
        logging.Logger: The expected logger.
    """
    if cfg is not None:
        log_level = "INFO"

    logger = logging.getLogger(name)

    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    logger.propagate = False

    handlers = [logging.StreamHandler(stream=sys.stdout)]

    # only rank 0 will add a FileHandler
    if meta is not None:
        log_file = meta["exp_dir"] / "run.log"
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, "w")
        handlers.append(file_handler)  # type: ignore

    FORMAT = "%(asctime)s - [%(levelname)s] %(message)s"
    formatter = logging.Formatter(FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level)

    logger_initialized[name] = True
    return logger


def print_log(msg, logger=None, level=logging.DEBUG):
    """
    Prints a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            f"logger should be either a logging.Logger object, "
            f"str, silent or None, but got {type(logger)}",
        )


def log_hf_history(history: list[dict], logger: logging.Logger) -> None:
    """
    Logs hf trainer history to logfile:

    2022-10-12 21:54:58,821 - [INFO] ------------------------------------------------------------
        train_runtime: 165.0475
        train_samples_per_second: 0.206
        train_steps_per_second: 0.036
        total_flos: 259478495232.0
        train_loss: 2.1963675816853843
        epoch: 2.0
        step: 6

    Args:
        history: transformers trainer log_history;
        logger: training logger.
    """
    dash_line = "-" * 60 + "\n"
    for log_step in history:
        metrics = [f"{k}: {v}" for k, v in log_step.items()]
        logger.info(dash_line + "\n".join(metrics) + "\n")


def log_ner_reports(reports: list[dict], logger: logging.Logger) -> None:
    """
    Logs auto_ner validator classification reports to `run.log`.
    """
    dash_line = "-" * 60 + "\n"
    for i, report in enumerate(reports):
        logger.info(f"Epoch {i}:{dash_line}")
        logger.info(pd.DataFrame(report).T.to_string())
        logger.info(dash_line)
