from platform import platform

import torch


def get_current_os() -> str:
    if is_linux_os():
        return "linux"
    return "windows"


def is_linux_os() -> bool:
    return platform().find("Linux") != -1


def processing_unit() -> str:
    return 'CUDA' if torch.cuda.is_available() else 'CPU'


class Environment:
    """
    Class for storing user specific configuration overwritten using environmental variables
    """
    def __init__(self, env):
        self.os = get_current_os()
        self.conf = float(env.get("CONFIDENCE", 0.7))
        self.processing_unit = processing_unit()

    @staticmethod
    def from_env(env):
        return Environment(env)

    def to_info_string(self):
        return "os: {}, processing unit: {}, detections confidence: {}%".format(
            self.os,
            self.processing_unit,
            self.conf*100
        )

    def cuda_to_info_string(self):
        return f"GPU: {torch.cuda.get_device_name(0)} " \
               f"Memory usage -> " \
               f"allocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB, " \
               f"cached: {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} GB."
