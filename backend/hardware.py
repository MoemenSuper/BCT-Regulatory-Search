"""Where the local models run: the best accelerator this machine has, batches sized to its memory.

Speed only: every device gives the same results, a weaker machine is just slower.
NVIDIA CUDA > Apple MPS > Intel XPU > CPU. On CPU, torch already uses all physical cores.
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def torch_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


@lru_cache(maxsize=1)
def gpu_memory_gb() -> float:
    if torch_device() != "cuda":
        return 0.0
    import torch

    return torch.cuda.get_device_properties(0).total_memory / 2**30


def batch_size(gpu_8gb: int) -> int:
    """A batch size sized for this machine, given the one that suits an 8 GB GPU."""
    device = torch_device()
    if device == "cuda":
        memory = gpu_memory_gb()
        # cards report a little under their nominal size (an "8 GB" card shows ~7.9)
        return gpu_8gb if memory >= 7.5 else max(1, gpu_8gb // 2) if memory >= 3.5 else max(1, gpu_8gb // 8)
    if device in {"mps", "xpu"}:
        return max(1, gpu_8gb // 4)  # unified / shared memory: stay modest
    return max(1, gpu_8gb // 8)  # CPU: small batches keep latency flat
