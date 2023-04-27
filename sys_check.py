# YOLO check
import ultralytics

ultralytics.checks()

# CUDA check
import torch

if not torch.cuda.is_available():
    print("CUDA is not available.")
