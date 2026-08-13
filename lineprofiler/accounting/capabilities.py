"""Optional dependencies, resolved once and degraded to ``None`` when absent.

Every optional integration is imported lazily and behind a capability check. A missing
package disables one block of the report; it never raises, and it never costs anything at
import time for users who do not have it.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from types import ModuleType

_UNSET: object = object()

_psutil: ModuleType | None | object = _UNSET
_torch: ModuleType | None | object = _UNSET
_nvml: ModuleType | None | object = _UNSET


def psutil_module() -> ModuleType | None:
    """Return ``psutil`` if installed, else ``None``. Powers RSS and I/O counters."""
    global _psutil
    if _psutil is _UNSET:
        try:
            import psutil
        except ImportError:
            _psutil = None
        else:
            _psutil = psutil
    return _psutil  # type: ignore[return-value]


def torch_module() -> ModuleType | None:
    """Return ``torch`` if installed, else ``None``. Powers CUDA memory and NVTX ranges."""
    global _torch
    if _torch is _UNSET:
        try:
            import torch
        except ImportError:
            _torch = None
        else:
            _torch = torch
    return _torch  # type: ignore[return-value]


def nvml_module() -> ModuleType | None:
    """Return an initialised ``pynvml`` if a GPU is visible, else ``None``.

    ``nvidia-ml-py`` reports two different utilisation numbers, and the sampler reads both.
    ``nvmlDeviceGetUtilizationRates`` is whole-device: the share of time any kernel from any
    process was resident, which on a shared node includes work that is not yours.
    ``nvmlDeviceGetProcessUtilization`` breaks that down per pid, which is what makes "the
    device is busy" and "*we* are keeping it busy" separable. Neither is a compute-versus-
    wait split; that needs kernel-level timing from ``torch.profiler``.
    """
    global _nvml
    if _nvml is _UNSET:
        _nvml = _initialise_nvml()
    return _nvml  # type: ignore[return-value]


def nvtx_range_functions() -> tuple[Callable[[str], object], Callable[[], object]] | None:
    """Return ``(push, pop)`` for NVTX ranges, or ``None`` when nothing can emit them.

    Prefers ``torch.cuda.nvtx`` and falls back to the standalone ``nvtx`` package. Emitting
    these costs a few hundred nanoseconds and buys phase names inside any externally started
    Nsight Systems capture — this package never launches nsys itself.
    """
    torch = torch_module()
    if torch is not None:
        try:
            return (torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop)
        except AttributeError:
            pass
    try:
        import nvtx
    except ImportError:
        return None
    return (nvtx.push_range, nvtx.pop_range)


def record_function_factory() -> Callable[[str], AbstractContextManager[object]] | None:
    """Return ``torch.profiler.record_function``, or ``None`` when torch is absent.

    It is a no-op when no torch profiler is active, so the only cost outside a capture is
    constructing the object.
    """
    torch = torch_module()
    if torch is None:
        return None
    try:
        factory: Callable[[str], AbstractContextManager[object]] = torch.profiler.record_function
    except AttributeError:
        return None
    return factory


def cuda_is_available() -> bool:
    """Whether torch is installed and reports a usable CUDA device."""
    torch = torch_module()
    return bool(torch and torch.cuda.is_available())


def cuda_synchronize() -> Callable[[], None] | None:
    """Return ``torch.cuda.synchronize``, or ``None`` when there is no CUDA device.

    Handed to ``phase(name, sync=True)``. Returning ``None`` rather than a no-op lambda is
    what lets the phase hot path skip the call with a single ``is not None`` test on a CPU
    box, instead of paying for a Python-level call that does nothing.
    """
    torch = torch_module()
    if torch is None or not torch.cuda.is_available():
        return None
    synchronize: Callable[[], None] = torch.cuda.synchronize
    return synchronize


def _initialise_nvml() -> ModuleType | None:
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 - any NVML failure means the capability is absent
        return None
    module: ModuleType = pynvml
    return module
