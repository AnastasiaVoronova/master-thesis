from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def resolve_device(device_arg: str) -> torch.device:
    requested = (device_arg or "cuda").strip().lower()

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"[warn] Requested '{device_arg}', but CUDA is unavailable. Falling back to CPU.")
            return torch.device("cpu")

        cuda_device = torch.device(requested)
        if cuda_device.index is not None and cuda_device.index >= torch.cuda.device_count():
            print(f"[warn] CUDA index {cuda_device.index} is out of range. Falling back to cuda:0.")
            return torch.device("cuda:0")
        return cuda_device

    if requested == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        print("[warn] Requested 'mps', but it is unavailable. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_indices_path(path: str | None) -> str | None:
    if not path:
        return None
    return path if Path(path).exists() else None


def model_key(model_name: str) -> str:
    normalized = model_name.replace("\\", "/").strip("/")
    return normalized.replace("/", "_")
