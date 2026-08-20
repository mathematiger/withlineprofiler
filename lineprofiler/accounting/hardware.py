"""What the machine a run measured on actually had: cores, memory and devices.

The sampler records what a run *consumed*. On its own that number cannot be read: "14.2 cores"
means one thing on a 16-core workstation and another on a 128-core node, and neither the
headroom nor the scaling behaviour is recoverable without knowing which. Profiling the same
workload on two servers produces two documents that cannot be compared at all.

This is deliberately a separate module from ``identity.py``, which records *placement* — host,
rank, job id — and states that it imports no library and never will. Capacity needs psutil and
NVML, so it belongs here instead, modelled on ``provenance.py``: resolved once, degraded to
``{}`` on any failure, and never able to break a run.

Every field is independently guarded, so one unreadable counter costs one field rather than the
whole inventory. Absent values are *omitted*, never zeroed — a machine that reports no RAM must
not be describable in the same shape as a machine with none.
"""

from __future__ import annotations

import os
from typing import Any

from lineprofiler.accounting.capabilities import nvml_module, psutil_module

_UNSET: object = object()
_cached: dict[str, Any] | object = _UNSET


def describe() -> dict[str, Any]:
    """Return this machine's capacity, resolving it at most once per process.

    Cached because a ``spawn``-heavy run constructs one :class:`SnapshotWriter` per worker and
    NVML enumeration is not free. Capacity cannot change under a running process, so the cache
    never goes stale — and a forked child inheriting the parent's dict is correct rather than
    merely cheap, since it is the same physical machine.

    Test specifically:
        - psutil and NVML both absent yields ``{}`` rather than raising
        - only psutil present yields the CPU and RAM fields and no ``gpus`` key
        - a device whose name cannot be read is skipped without losing the others
        - a ``bytes`` device name is decoded rather than rendered as ``b'...'``
    """
    global _cached
    if _cached is _UNSET:
        _cached = _describe_uncached()
    return dict(_cached)  # type: ignore[arg-type]


def reset_cache() -> None:
    """Discard the cached inventory. For tests that patch the underlying capabilities."""
    global _cached
    _cached = _UNSET


def format_capacity(hardware: dict[str, Any]) -> str:
    """Render one machine's capacity as a single line, or ``""`` when nothing is known.

    Example: ``128 cores (60 available to this job), 2.0 TB RAM, 4x A100-SXM4-80GB``.
    """
    parts = []
    cores = hardware.get("cpu_cores")
    if cores:
        affinity = hardware.get("cpu_affinity")
        # Only when it differs: on an unconstrained box the two are equal and printing both
        # invites the reader to look for a distinction that is not there.
        if affinity and affinity != cores:
            parts.append(f"{cores} cores ({affinity} available to this job)")
        else:
            parts.append(f"{cores} cores")
    ram = hardware.get("ram_total")
    if ram:
        parts.append(f"{_format_bytes(int(ram))} RAM")
    gpus = hardware.get("gpus")
    if gpus:
        parts.append(format_gpu_models(gpus))
    return ", ".join(parts)


def format_gpu_models(gpus: list[dict[str, Any]]) -> str:
    """Render a device list as ``4x A100-SXM4-80GB``, or list the models when they differ."""
    if not gpus:
        return ""
    names = [str(gpu.get("name", "?")) for gpu in gpus]
    distinct = sorted(set(names))
    if len(distinct) == 1:
        return f"{len(names)}x {distinct[0]}"
    return ", ".join(f"{names.count(name)}x {name}" for name in distinct)


def total_vram(gpus: list[dict[str, Any]]) -> int:
    """Sum the device memory of a machine's GPUs, in bytes."""
    return sum(int(gpu.get("vram_total", 0)) for gpu in gpus)


def _describe_uncached() -> dict[str, Any]:
    hardware: dict[str, Any] = {}
    _add_cpu(hardware)
    _add_memory(hardware)
    gpus = _describe_gpus()
    if gpus:
        hardware["gpus"] = gpus
    return hardware


def _add_cpu(hardware: dict[str, Any]) -> None:
    """Physical cores, SMT siblings, and how many this process may actually run on."""
    psutil = psutil_module()
    if psutil is not None:
        physical = _read(lambda: psutil.cpu_count(logical=False))
        if physical:
            hardware["cpu_cores"] = int(physical)
        logical = _read(lambda: psutil.cpu_count(logical=True))
        if logical:
            hardware["cpu_threads"] = int(logical)
    # The honest denominator for headroom under Slurm or a container, where the box has far
    # more cores than the job was given. Absent on macOS and Windows: omitted there rather
    # than falled back to the core count, which would claim a quota nothing measured.
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        count = _read(lambda: len(affinity(0)))
        if count:
            hardware["cpu_affinity"] = int(count)


def _add_memory(hardware: dict[str, Any]) -> None:
    psutil = psutil_module()
    if psutil is None:
        return
    total = _read(lambda: psutil.virtual_memory().total)
    if total:
        hardware["ram_total"] = int(total)


def _describe_gpus() -> list[dict[str, Any]]:
    """Model name and total memory per device.

    Indices are NVML's, matching ``sampler._open_devices`` deliberately: they are the machine's
    physical devices, not torch's ``CUDA_VISIBLE_DEVICES``-renumbered logical ones. That is
    what lets a capacity entry line up with the utilisation the sampler reports for the same
    index.
    """
    nvml = nvml_module()
    if nvml is None:
        return []
    count = _read(lambda: int(nvml.nvmlDeviceGetCount()))
    if not count:
        return []

    devices: list[dict[str, Any]] = []
    for index in range(int(count)):
        device = _describe_device(nvml, index)
        if device is not None:
            devices.append(device)
    return devices


def _describe_device(nvml: Any, index: int) -> dict[str, Any] | None:  # noqa: ANN401
    """One device's name and memory, or ``None`` when it cannot be addressed at all.

    A separate function rather than a loop body so each guarded read closes over its own
    ``handle``. Written inline, the lambdas bound the loop variable and every device reported
    the last one's name.
    """
    handle = _read(lambda: nvml.nvmlDeviceGetHandleByIndex(index))
    if handle is None:
        return None
    device: dict[str, Any] = {"index": index}
    name = _read(lambda: nvml.nvmlDeviceGetName(handle))
    if name is not None:
        device["name"] = _as_text(name)
    total = _read(lambda: nvml.nvmlDeviceGetMemoryInfo(handle).total)
    if total:
        device["vram_total"] = int(total)
    return device


def _as_text(value: object) -> str:
    """Decode a device name, which older ``nvidia-ml-py`` returns as ``bytes``."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _read(source: Any) -> Any:  # noqa: ANN401 - one guard for every capability probe
    """Call ``source``, returning ``None`` on any failure.

    Broad by design: this module's contract is that it cannot break a run, and the callers
    raise platform-specific errors that are not worth enumerating.
    """
    try:
        return source()
    except Exception:  # noqa: BLE001 - any failure means the field is unavailable here
        return None


def _format_bytes(value: int) -> str:
    """Local byte formatter, so this module stays below ``analysis`` in the import order."""
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"
