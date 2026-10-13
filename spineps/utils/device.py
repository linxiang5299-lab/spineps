"""Compute device selection shared by the segmentation and labeling models."""

from __future__ import annotations

from typing import Literal

import torch

DeviceName = Literal["cuda", "mps", "cpu"]


def pick_device_name(use_cpu: bool = False) -> DeviceName:
    """Selects the compute device by name: CUDA first, then Apple MPS, then CPU.

    Args:
        use_cpu (bool, optional): Forces the CPU regardless of the available accelerators. Defaults to False.

    Returns:
        DeviceName: ``"cuda"``, ``"mps"`` or ``"cpu"``.
    """
    if use_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def pick_device(use_cpu: bool = False) -> torch.device:
    """Selects the compute device as a ``torch.device``, see :func:`pick_device_name`."""
    name = pick_device_name(use_cpu)
    return torch.device("cuda:0" if name == "cuda" else name)
