import re
import typing
import platform

import torch


def get_current_os() -> str:
    if is_linux_os():
        return "linux"
    return "windows"


def is_linux_os() -> bool:
    return platform.platform().find("Linux") != -1


def pytorch_info(version) -> typing.Tuple[str, str]:
    regex = r"([0-9]*.*)\+([a-z]*)"
    match = re.match(regex, version)
    version, engine = match.group(1), match.group(2)
    return version, engine


def processing_unit() -> typing.Tuple[str, str]:
    version, engine = pytorch_info(torch.__version__)
    if engine == "cu":
        return "CUDA", version
    if engine == "cpu":
        return "CPU", version
    return engine.upper(), version


def get_resolution(resolution_env) -> typing.Tuple[int, int]:
    regex = r"\d+"
    resolution = re.findall(regex, resolution_env)
    return int(resolution[0]), int(resolution[1])


class Environment:
    """
    Class for storing user specific configuration overwritten using environmental variables
    """

    def __init__(self, env):
        self.os = get_current_os()
        self.conf = float(env.get("CONFIDENCE", 0.7))
        self.processing_unit, self.version = processing_unit()
        self.resolution = get_resolution(env.get("RESOLUTION", "1280x720"))
        self.img_graphics_format = "jpg"

    @staticmethod
    def from_env(env):
        return Environment(env)

    def to_info_string(self) -> str:
        return "os: {}, processing unit: {}, detections confidence: {}%, resolution: {}x{}".format(
            self.os,
            self.processing_unit,
            self.conf * 100,
            self.resolution[0],
            self.resolution[1],
        )

    def cuda_to_info_string(self) -> str:
        return (
            f"GPU: {torch.cuda.get_device_name(0)} "
            f"Memory usage -> "
            f"allocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB, "
            f"cached: {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} GB."
        )

    def cpu_to_info_string(self) -> str:
        import cpuinfo

        cpu_info = cpuinfo.get_cpu_info()
        return f"CPU: {cpu_info['brand_raw']}, Arch: {cpu_info['arch_string_raw']}, Cores: {cpu_info['count']}"
