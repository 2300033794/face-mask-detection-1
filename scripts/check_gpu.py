"""
check_gpu.py — Verify CUDA / GPU availability before training.

Run:
    python scripts/check_gpu.py

Expected output (GPU machine):
    PyTorch version : 2.x.x+cu118
    CUDA available  : True
    GPU count       : 1
    GPU name        : NVIDIA GeForce RTX 3080
    Device selected : cuda:0

Expected output (CPU-only machine):
    PyTorch version : 2.x.x+cpu
    CUDA available  : False
    Device selected : cpu
"""

import torch


def check_gpu() -> str:
    """Return the best available device string ('cuda:0' or 'cpu')."""
    print(f"PyTorch version : {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available  : {cuda_available}")

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU count       : {gpu_count}")
        print(f"GPU name        : {gpu_name}")
        device = "cuda:0"
    else:
        print("No GPU detected — training will use CPU (slower).")
        device = "cpu"

    print(f"Device selected : {device}")
    return device


if __name__ == "__main__":
    check_gpu()
