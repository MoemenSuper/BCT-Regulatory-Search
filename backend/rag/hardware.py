"""Where the local models run: the best accelerator this machine has, batches sized to its memory.

Speed only: every device gives the same results, a weaker machine is just slower.
NVIDIA CUDA > Apple MPS > Intel XPU > CPU. On CPU, torch already uses all physical cores.
"""
from functools import lru_cache
from pathlib import Path


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


def memory_gb() -> tuple[float, float] | None:
    """(total, free) memory in GB this machine, or this container, can still hand out.

    Linux (Docker): /proc/meminfo, narrowed by the container's own memory limit when it has one
    (cgroup v2). Elsewhere psutil when installed. None when it cannot be read.
    """
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, value = line.split(":", 1)
            info[name] = int(value.split()[0]) / 2**20
        total, free = info["MemTotal"], info["MemAvailable"]
        limit = Path("/sys/fs/cgroup/memory.max")
        if limit.is_file() and limit.read_text().strip() != "max":
            cap = int(limit.read_text()) / 2**30
            used = int(Path("/sys/fs/cgroup/memory.current").read_text()) / 2**30
            total, free = min(total, cap), min(free, cap - used)
        return total, free
    except (OSError, KeyError, ValueError):
        pass
    try:
        import psutil

        memory = psutil.virtual_memory()
        return memory.total / 2**30, memory.available / 2**30
    except Exception:
        return None
