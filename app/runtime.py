"""Explicit CUDA selection and executable GPU diagnostics."""
from __future__ import annotations

import torch


def resolve_device(requested: str = "cuda:0") -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Run nvidia-smi and install the CUDA PyTorch wheel; "
                "for Docker enable the WSL2 GPU backend. Use --device cpu explicitly for debugging."
            )
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} does not exist")
        device = torch.device(f"cuda:{index}")
    elif device.type != "cpu":
        raise ValueError("Only CUDA and CPU devices are supported")
    return device


def gpu_diagnostics(device: torch.device | None = None) -> dict:
    """Includes a real tensor operation rather than merely reporting driver state."""
    info = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if device is not None:
        info["device"] = str(device)
    if torch.cuda.is_available():
        selected = device if device is not None and device.type == "cuda" else torch.device("cuda:0")
        props = torch.cuda.get_device_properties(selected)
        with torch.inference_mode():
            matrix = torch.ones((32, 32), device=selected)
            value = (matrix @ matrix)[0, 0].item()
        info.update({
            "gpu_name": props.name,
            "gpu_total_memory_bytes": props.total_memory,
            "cuda_tensor_test": value == 32.0,
        })
    return info
