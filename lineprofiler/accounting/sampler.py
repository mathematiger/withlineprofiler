"""Background resource sampling: memory, I/O and GPU utilisation, tagged with the phase.

The phase tree answers "where did the time go". This answers "what was the machine doing
while it went there" — resident memory, bytes read and written, CUDA allocator state and
whole-device GPU utilisation, each row stamped with the phase that was open when it was
taken.

Everything here is optional. Without ``psutil`` there are no memory or I/O rows; without
torch there are no CUDA rows; without ``nvidia-ml-py`` there is no utilisation. The sampler
still runs and the rest of the report is unaffected.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO, runtime_checkable

from lineprofiler.accounting.capabilities import (
    cuda_is_available,
    nvml_module,
    psutil_module,
    torch_module,
)


class MemoryInfo(Protocol):
    """Resident set size, as returned by ``psutil.Process.memory_info()``."""

    rss: int


class IoCounters(Protocol):
    """Cumulative byte counters, as returned by ``psutil.Process.io_counters()``."""

    read_bytes: int
    write_bytes: int


@runtime_checkable
class ProcessHandle(Protocol):
    """The slice of ``psutil.Process`` this module uses.

    Declaring it explicitly keeps the sampler type-checked without depending on psutil's
    stubs being installed, and documents exactly which counters are read.
    """

    def memory_info(self) -> MemoryInfo: ...

    def io_counters(self) -> IoCounters: ...


@dataclass(slots=True)
class Sample:
    """One row of resource state, taken at ``t`` while ``phase`` was open.

    Byte counters are cumulative process totals as reported by the OS, not per-phase
    figures; per-phase attribution is a difference between consecutive samples.
    """

    t: float
    phase: str
    rss: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    cuda_alloc: int = 0
    cuda_reserved: int = 0
    gpu_util: float = -1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Sample:
        """Rebuild a sample from a JSONL row, tolerating fields absent on other platforms."""
        return cls(
            t=data["t"],
            phase=data.get("phase", ""),
            rss=data.get("rss", 0),
            read_bytes=data.get("read_bytes", 0),
            write_bytes=data.get("write_bytes", 0),
            cuda_alloc=data.get("cuda_alloc", 0),
            cuda_reserved=data.get("cuda_reserved", 0),
            gpu_util=data.get("gpu_util", -1.0),
        )


@dataclass(slots=True)
class SamplerCapabilities:
    """What this process can actually measure, resolved once at construction."""

    memory: bool = False
    io: bool = False
    cuda: bool = False
    gpu_util: bool = False

    def describe(self) -> str:
        """Return a short human-readable list of the active capabilities."""
        active = [name for name, on in self.as_dict().items() if on]
        return ", ".join(active) if active else "none"

    def as_dict(self) -> dict[str, bool]:
        return {"memory": self.memory, "io": self.io, "cuda": self.cuda, "gpu_util": self.gpu_util}


class ResourceSampler:
    """Daemon thread appending one JSONL row per interval to ``path``.

    Test specifically:
        - a synthetic workload writing a known number of bytes is reported within a band
          (page cache and buffering make exactness impossible)
        - a synthetic leak shows a positive RSS slope and a non-leaking equivalent does not
        - the thread does not prevent interpreter exit
        - construction succeeds with psutil absent, with torch absent and with NVML absent
    """

    def __init__(self, path: Path, interval_s: float, phase_of: Callable[[], str]) -> None:
        self.path = path
        self.interval_s = interval_s
        self._phase_of = phase_of
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: ProcessHandle | None = open_process()
        self.capabilities = _detect_capabilities(self._process)

    def start(self) -> None:
        """Begin sampling in a daemon thread.

        Clears the stop flag first, so a sampler that was stopped — around a ``fork``, for
        instance — can be started again rather than exiting immediately.
        """
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="lineprofiler-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and wait briefly for the thread to finish its current row."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 1.0)
            self._thread = None

    def take(self) -> Sample:
        """Read every available counter once. Missing capabilities leave their fields at zero."""
        sample = Sample(t=time.time(), phase=self._phase_of())
        self._add_process_metrics(sample)
        self._add_cuda_metrics(sample)
        self._add_gpu_utilisation(sample)
        return sample

    def _run(self) -> None:
        """Sample until stopped, bracketing the run with a baseline and a final row.

        The cumulative OS counters only yield useful numbers as differences, so a run whose
        first row is taken after the work has already started reports nothing for that work.
        The baseline row is taken before any interval elapses, and the final row after the
        stop signal, so bytes moved at either end of the run are still counted.
        """
        with self.path.open("a", encoding="utf-8") as handle:
            self._write_row(handle)
            while not self._stop.wait(self.interval_s):
                self._write_row(handle)
            self._write_row(handle)

    def _write_row(self, handle: TextIO) -> None:
        handle.write(json.dumps(_compact(self.take())) + "\n")
        handle.flush()

    def _add_process_metrics(self, sample: Sample) -> None:
        if self._process is None:
            return
        if self.capabilities.memory:
            sample.rss = self._process.memory_info().rss
        if self.capabilities.io:
            counters = self._process.io_counters()
            sample.read_bytes = counters.read_bytes
            sample.write_bytes = counters.write_bytes

    def _add_cuda_metrics(self, sample: Sample) -> None:
        if not self.capabilities.cuda:
            return
        torch = torch_module()
        if torch is None:
            return
        sample.cuda_alloc = torch.cuda.memory_allocated()
        sample.cuda_reserved = torch.cuda.memory_reserved()

    def _add_gpu_utilisation(self, sample: Sample) -> None:
        if not self.capabilities.gpu_util:
            return
        nvml = nvml_module()
        if nvml is None:
            return
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            sample.gpu_util = float(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:  # noqa: BLE001 - a transient NVML error must not kill the thread
            sample.gpu_util = -1.0


def read_samples(path: Path) -> list[Sample]:
    """Read a ``.samples`` file, skipping any truncated final line.

    Test specifically:
        - a file whose last line was cut off mid-write still parses the earlier rows
    """
    samples: list[Sample] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return samples
    for line in text.splitlines():
        try:
            samples.append(Sample.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError):
            continue
    return samples


def open_process() -> ProcessHandle | None:
    psutil = psutil_module()
    if psutil is None:
        return None
    try:
        handle: ProcessHandle = psutil.Process(os.getpid())
    except Exception:  # noqa: BLE001 - psutil raises platform-specific errors here
        return None
    return handle


def _detect_capabilities(process: ProcessHandle | None) -> SamplerCapabilities:
    """Probe each counter once; ``io_counters`` in particular is absent on macOS."""
    capabilities = SamplerCapabilities()
    if process is not None:
        capabilities.memory = _probe(lambda: process.memory_info().rss)
        capabilities.io = _probe(lambda: process.io_counters().read_bytes)
    capabilities.cuda = cuda_is_available()
    capabilities.gpu_util = nvml_module() is not None
    return capabilities


def _probe(read: Callable[[], object]) -> bool:
    try:
        read()
    except Exception:  # noqa: BLE001 - any failure means the counter is unavailable here
        return False
    return True


def _compact(sample: Sample) -> dict[str, Any]:
    """Drop zero-valued fields so a 12-hour sample file stays small."""
    row: dict[str, Any] = {"t": round(sample.t, 3), "phase": sample.phase}
    for name in ("rss", "read_bytes", "write_bytes", "cuda_alloc", "cuda_reserved"):
        value = getattr(sample, name)
        if value:
            row[name] = value
    if sample.gpu_util >= 0:
        row["gpu_util"] = sample.gpu_util
    return row
