"""Heavy profilers, started for a bounded window and correlated with the phase timeline.

Exactly one backend can be active. VizTracer, cProfile and ``line_profiler`` all contend for
the interpreter's trace hook, and ``torch.profiler`` distorts timing enough that combining
it with another sampler makes both meaningless — so ``backend`` is a single enum value, not
a set of flags, and there is no way to express two at once.

The window is expressed in entries of a phase you name, which keeps it generic: any training
loop has *some* repeating outer phase, whether it is called an iteration, an epoch or a
step.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from lineprofiler.accounting.capabilities import cuda_is_available, torch_module


class Backend(str, Enum):
    """Which heavy profiler to run for the window, if any."""

    NONE = "none"
    TORCH = "torch"
    VIZTRACER = "viztracer"

    @classmethod
    def parse(cls, value: Backend | str | None) -> Backend:
        """Accept the enum, its string value, or ``None``.

        Test specifically:
            - an unknown name raises ``ValueError`` naming the valid options
            - ``None`` and ``"none"`` both resolve to ``Backend.NONE``
        """
        if value is None:
            return cls.NONE
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError:
            valid = ", ".join(member.value for member in cls)
            raise ValueError(f"unknown backend {value!r}; expected one of: {valid}") from None


class BackendWindow:
    """Starts and stops one heavy profiler across a range of entries into a phase.

    Test specifically:
        - the backend starts on the configured entry and stops on the configured exit
        - an unavailable backend degrades to a no-op and records why
        - the artifact path lands in the run directory and is recorded in metadata
    """

    def __init__(
        self,
        backend: Backend,
        window: tuple[int, int] | None,
        phase_name: str,
        run_dir: Path,
    ) -> None:
        self.backend = backend
        self.window = window
        self.phase_name = phase_name
        self.run_dir = run_dir
        self.entries = 0
        self.artifact: Path | None = None
        self.unavailable_reason: str | None = None
        self._handle: Any = None

    @property
    def active(self) -> bool:
        return self._handle is not None

    def on_phase_enter(self, name: str) -> None:
        """Count entries into the window phase and start the backend at the lower bound."""
        if self.backend is Backend.NONE or self.window is None or name != self.phase_name:
            return
        self.entries += 1
        if self.entries == self.window[0]:
            self._start()

    def on_phase_exit(self, name: str) -> None:
        """Stop the backend once the window's last entry completes."""
        if self.window is None or name != self.phase_name or not self.active:
            return
        if self.entries >= self.window[1]:
            self._stop()

    def close(self) -> None:
        """Stop the backend if the run ends inside the window."""
        if self.active:
            self._stop()

    def describe(self) -> dict[str, Any]:
        """Return the backend's configuration and artifact for ``metadata.json``."""
        return {
            "backend": self.backend.value,
            "window": list(self.window) if self.window else None,
            "window_phase": self.phase_name,
            "artifact": str(self.artifact) if self.artifact else None,
            "unavailable_reason": self.unavailable_reason,
        }

    def _start(self) -> None:
        directory = self.run_dir / "backend"
        directory.mkdir(parents=True, exist_ok=True)
        if self.backend is Backend.TORCH:
            self._handle = _start_torch(directory)
        elif self.backend is Backend.VIZTRACER:
            self._handle = _start_viztracer(directory)
        if self._handle is None:
            self.unavailable_reason = f"{self.backend.value} is not installed or not usable"

    def _stop(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        self.artifact = handle.stop()


class _TorchProfiler(Protocol):
    """The slice of ``torch.profiler.profile`` this module drives."""

    def stop(self) -> object: ...

    def export_chrome_trace(self, path: str) -> object: ...


class _VizTracer(Protocol):
    """The slice of ``viztracer.VizTracer`` this module drives."""

    def stop(self) -> object: ...

    def save(self, output_file: str) -> object: ...


class _TorchHandle:
    """A running ``torch.profiler`` capture, exported as a Chrome trace on stop."""

    def __init__(self, profiler: _TorchProfiler, artifact: Path) -> None:
        self._profiler = profiler
        self._artifact = artifact

    def stop(self) -> Path:
        self._profiler.stop()
        self._profiler.export_chrome_trace(str(self._artifact))
        return self._artifact


class _VizTracerHandle:
    """A running VizTracer capture, saved to its artifact on stop."""

    def __init__(self, tracer: _VizTracer, artifact: Path) -> None:
        self._tracer = tracer
        self._artifact = artifact

    def stop(self) -> Path:
        self._tracer.stop()
        self._tracer.save(str(self._artifact))
        return self._artifact


def _start_torch(directory: Path) -> _TorchHandle | None:
    torch = torch_module()
    if torch is None:
        return None
    activities = [torch.profiler.ProfilerActivity.CPU]
    if cuda_is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = torch.profiler.profile(activities=activities)
    profiler.start()
    return _TorchHandle(profiler, directory / "torch_trace.json")


def _start_viztracer(directory: Path) -> _VizTracerHandle | None:
    try:
        from viztracer import VizTracer
    except ImportError:
        return None
    tracer = VizTracer(output_file=str(directory / "viztracer.json"))
    tracer.start()
    return _VizTracerHandle(tracer, directory / "viztracer.json")
