# mypy: ignore-errors

from collections import defaultdict
import os
import subprocess
import sys

import torch


def collect_env() -> dict:
    """
    Collects the information of the running environments.

    Returns:
        dict: The environment information. The following fields are contained.
            - sys.platform: The variable of ``sys.platform``.
            - Python: Python version.
            - CUDA available: Bool, indicating if CUDA is available.
            - GPU devices: Device type of each GPU.
            - CUDA_HOME (optional): The env var ``CUDA_HOME``.
            - NVCC (optional): NVCC version.
            - GCC: GCC version, "n/a" if GCC is not installed.
            - PyTorch: PyTorch version.
            - TorchVision (optional): TorchVision version.
            - OpenCV: OpenCV version.
    """
    env_info = {}
    env_info["sys.platform"] = sys.platform
    env_info["Python"] = sys.version.replace("\n", "")

    cuda_available = torch.cuda.is_available()
    env_info["CUDA available"] = cuda_available

    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info["GPU " + ",".join(device_ids)] = name

        # ML
        from torch.utils.cpp_extension import CUDA_HOME

        env_info["CUDA_HOME"] = CUDA_HOME

        if CUDA_HOME is not None and os.path.isdir(CUDA_HOME):
            try:
                nvcc = os.path.join(CUDA_HOME, "bin/nvcc")
                nvcc = subprocess.check_output(f'"{nvcc}" -V | tail -n1', shell=True)
                nvcc = nvcc.decode("utf-8").strip()
            except subprocess.SubprocessError:
                nvcc = "Not Available"
            env_info["NVCC"] = nvcc

    try:
        gcc = subprocess.check_output("gcc --version | head -n1", shell=True)
        gcc = gcc.decode("utf-8").strip()
        env_info["GCC"] = gcc
    except subprocess.CalledProcessError:  # gcc is unavailable
        env_info["GCC"] = "n/a"

    env_info["PyTorch"] = torch.__version__
    return env_info
