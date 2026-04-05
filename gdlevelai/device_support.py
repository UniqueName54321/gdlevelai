from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _log(message: str) -> None:
    print(f"[device_support] {message}", flush=True)


@dataclass
class RuntimeDevice:
    backend: str
    device: Any


def resolve_device(requested: str) -> RuntimeDevice:
    req = (requested or "auto").lower()

    if req == "auto":
        if torch.cuda.is_available():
            if torch.version.hip is not None:
                _log("Auto-selected ROCm (torch.cuda + HIP)")
                return RuntimeDevice("rocm", torch.device("cuda"))
            _log("Auto-selected CUDA")
            return RuntimeDevice("cuda", torch.device("cuda"))

        try:
            import torch_directml  # type: ignore

            dml = torch_directml.device()
            _log("Auto-selected DirectML")
            return RuntimeDevice("directml", dml)
        except Exception:
            _log("Auto-selected CPU")
            return RuntimeDevice("cpu", torch.device("cpu"))

    if req == "cpu":
        return RuntimeDevice("cpu", torch.device("cpu"))

    if req == "cuda":
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise RuntimeError("CUDA requested but CUDA runtime is unavailable")
        return RuntimeDevice("cuda", torch.device("cuda"))

    if req == "rocm":
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError(
                "ROCm requested but torch ROCm runtime is unavailable (need ROCm PyTorch build)"
            )
        return RuntimeDevice("rocm", torch.device("cuda"))

    if req == "directml":
        try:
            import torch_directml  # type: ignore

            dml = torch_directml.device()
            return RuntimeDevice("directml", dml)
        except Exception as exc:
            raise RuntimeError(
                "DirectML requested but torch-directml is unavailable. Install with: pip install torch-directml"
            ) from exc

    raise RuntimeError(
        "Unsupported device value. Use one of: auto, cpu, cuda, rocm, directml"
    )
