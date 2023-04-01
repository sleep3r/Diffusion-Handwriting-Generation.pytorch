import logging
import sys

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
