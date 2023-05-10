from camera_perception.common import pytorch_info


# YOLO check
import ultralytics

ultralytics.checks()


# CUDA/CPU check
import torch
import torchvision


version, engine = pytorch_info(torch.__version__)
version_vision, engine_vision = pytorch_info(torchvision.__version__)

if engine != engine_vision:
    print(
        f"torch and torchvision packages are not compatible:\n"
        f"torch: {engine} {version}\n"
        f"torchvision: {engine_vision} {version_vision}"
    )
    exit(1)

if engine == "cu" and not torch.cuda.is_available():
    print(
        f"CUDA is not available but PyTorch {engine.upper()} {version} is installed. "
        f"Check environment configuration for conflicted packages."
    )
    exit(1)

if not torch.cuda.is_available():
    print(f"CUDA is not available. PyTorch {engine.upper()} {version} will be used.")
    exit(0)

if engine == "cpu" and torch.cuda.is_available():
    print(
        f"CUDA is available but PyTorch {engine.upper()} {version} is used. "
        f"Check environment configuration for better performance."
    )
