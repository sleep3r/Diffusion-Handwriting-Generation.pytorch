import os


def check_file_exist(filename: str, msg_tmpl: str = 'file "{}" does not exist') -> None:
    """
    Checks if a file exists.

    Args:
        filename: file name;
        msg_tmpl: message template.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name: str, mode: int = 0o777) -> None:
    """
    Creates a directory if it does not exist.

    Args:
        dir_name: directory name;
        mode: mode.
    """
    if dir_name == "":
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)
