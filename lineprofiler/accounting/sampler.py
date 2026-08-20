"""Background resource sampling: memory, I/O and GPU utilisation, tagged with the phase.

The phase tree answers "where did the time go". This answers "what was the machine doing
while it went there" — resident memory, bytes read and written, CUDA allocator state and
GPU utilisation per visible device, each row stamped with the phase that was open when it
was taken.

Utilisation is recorded at two levels, because on a shared node they answer different
questions: the whole-device busy percentage counts every process's kernels, while the
per-process figure counts only this pid's. A device pinned at 95% that owes 20% of it to
your run is a queueing problem, not a saturated GPU.

Everything here is optional. Without ``psutil`` there are no memory or I/O rows; without
torch there are no CUDA rows; without ``nvidia-ml-py`` there is no utilisation. The sampler
still runs and the rest of the report is unaffected.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple, Protocol, TextIO, runtime_checkable

from lineprofiler.accounting.capabilities import (
    cuda_is_available,
    nvml_module,
    psutil_module,
    torch_module,
)
from lineprofiler.accounting.selfio import bytes_written, record_bytes_written


class MemoryInfo(Protocol):
    """Resident set size, as returned by ``psutil.Process.memory_info()``."""

    rss: int


class IoCounters(Protocol):
    """Cumulative byte counters, as returned by ``psutil.Process.io_counters()``.

    ``read_bytes``/``write_bytes`` count traffic at the block layer — what actually reached
    the storage device. ``read_chars``/``write_chars`` count bytes passed through the read
    and write syscalls, cache hit or not. Only the first pair exists on every platform; the
    second is Linux's ``rchar``/``wchar`` and is read defensively.
    """

    read_bytes: int
    write_bytes: int


class IoSnapshot(NamedTuple):
    """One reading of the process byte counters, at both the disk and the syscall layer.

    Keeping both is what makes a page-cached read visible. A training run whose dataset fits
    in RAM moves no disk bytes at all, so ``read_bytes`` alone reports its data loading as
    idle — correctly, and uselessly.

    ``available`` separates "the counters read as zero" from "the counters could not be
    read". They are not the same fact, and conflating them is dangerous: these values are
    cumulative, so a failed read that reports zero is differenced against the *next* real
    reading as though the process had rewound to the start of time, fabricating the whole
    cumulative total as one interval's traffic.
    """

    read_bytes: int = 0
    write_bytes: int = 0
    read_chars: int = 0
    write_chars: int = 0
    available: bool = True

    def is_empty(self) -> bool:
        """Whether every counter is zero, which is how an unsupported platform reports."""
        return not (self.read_bytes or self.write_bytes or self.read_chars or self.write_chars)


@runtime_checkable
class ProcessHandle(Protocol):
    """The slice of ``psutil.Process`` this module uses.

    Declaring it explicitly keeps the sampler type-checked without depending on psutil's
    stubs being installed, and documents exactly which counters are read.
    """

    def memory_info(self) -> MemoryInfo: ...

    def io_counters(self) -> IoCounters: ...

    def cpu_percent(self) -> float: ...


def read_io_snapshot(process: ProcessHandle | None) -> IoSnapshot:
    """Read both layers of byte counter at once, flagging a failure rather than faking a zero.

    Returns ``available=False`` when psutil is absent, when the platform has no per-process
    counters (macOS), or when the read itself failed — which happens transiently on a process
    the kernel is tearing down. Callers must never difference an unavailable reading; see
    :class:`IoSnapshot`.
    """
    if process is None:
        return IoSnapshot(available=False)
    try:
        counters = process.io_counters()
    except Exception:  # noqa: BLE001 - the counter can vanish on a dying process
        return IoSnapshot(available=False)
    return IoSnapshot(
        read_bytes=counters.read_bytes,
        write_bytes=counters.write_bytes,
        read_chars=getattr(counters, "read_chars", 0),
        write_chars=getattr(counters, "write_chars", 0),
    )


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
    read_chars: int = 0
    write_chars: int = 0
    self_write_chars: int = 0
    self_write_bytes: int = 0
    cuda_alloc: int = 0
    cuda_reserved: int = 0
    cpu_percent: float = -1.0
    """Process CPU over the interval ending at ``t``, as a percentage of one core; a
    multithreaded process exceeds 100. ``-1.0`` means not measured — never ``0.0``, which is a
    real reading meaning the process was idle. The first row of a run carries the sentinel
    because ``psutil`` has no previous call to difference against."""
    gpu_util: float = -1.0
    gpu_utils: dict[int, float] = field(default_factory=dict)
    gpu_proc_utils: dict[int, float] = field(default_factory=dict)
    io_ok: bool = True
    """Whether the byte counters in this row were actually read. A row with ``False`` carries
    zeros that mean nothing, and analysis must not difference across it."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Sample:
        """Rebuild a sample from a JSONL row, tolerating fields absent on other platforms."""
        return cls(
            t=data["t"],
            phase=data.get("phase", ""),
            rss=data.get("rss", 0),
            read_bytes=data.get("read_bytes", 0),
            write_bytes=data.get("write_bytes", 0),
            read_chars=data.get("read_chars", 0),
            write_chars=data.get("write_chars", 0),
            self_write_chars=data.get("self_write_chars", 0),
            self_write_bytes=data.get("self_write_bytes", 0),
            cuda_alloc=data.get("cuda_alloc", 0),
            cuda_reserved=data.get("cuda_reserved", 0),
            cpu_percent=data.get("cpu_percent", -1.0),
            gpu_util=data.get("gpu_util", -1.0),
            gpu_utils=_device_map(data.get("gpu_utils")),
            gpu_proc_utils=_device_map(data.get("gpu_proc_utils")),
            io_ok=bool(data.get("io_ok", True)),
        )


@dataclass(slots=True)
class SamplerCapabilities:
    """What this process can actually measure, resolved once at construction."""

    memory: bool = False
    io: bool = False
    cpu: bool = False
    cuda: bool = False
    gpu_util: bool = False

    def describe(self) -> str:
        """Return a short human-readable list of the active capabilities."""
        active = [name for name, on in self.as_dict().items() if on]
        return ", ".join(active) if active else "none"

    def as_dict(self) -> dict[str, bool]:
        return {
            "memory": self.memory,
            "io": self.io,
            "cpu": self.cpu,
            "cuda": self.cuda,
            "gpu_util": self.gpu_util,
        }


class ResourceSampler:
    """Daemon thread appending one JSONL row per interval to ``path``.

    Test specifically:
        - a synthetic workload writing a known number of bytes is reported within a band
          (page cache and buffering make exactness impossible)
        - a synthetic leak shows a positive RSS slope and a non-leaking equivalent does not
        - the thread does not prevent interpreter exit
        - construction succeeds with psutil absent, with torch absent and with NVML absent
        - every visible device gets its own utilisation entry, and ``gpu_util`` is their mean
        - only this pid's rows are counted towards the per-process figure
        - a device whose utilisation call fails is skipped without losing the others
    """

    def __init__(self, path: Path, interval_s: float, phase_of: Callable[[], str]) -> None:
        self.path = path
        self.interval_s = interval_s
        self._phase_of = phase_of
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: ProcessHandle | None = open_process()
        self.capabilities = _detect_capabilities(self._process)
        self._devices: list[tuple[int, Any]] = _open_devices()
        self._proc_util_since = 0
        self.write_failures = 0

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

        A row that fails to write is counted and skipped rather than ending the thread. A
        full or flaky filesystem used to stop sampling permanently and silently, after which
        every I/O, memory and GPU number came from a truncated series that looked complete.
        """
        try:
            handle = self.path.open("a", encoding="utf-8")
        except OSError:
            self.write_failures += 1
            return
        try:
            self._write_row(handle)
            while not self._stop.wait(self.interval_s):
                self._write_row(handle)
            self._write_row(handle)
        finally:
            with contextlib.suppress(OSError):
                handle.close()

    def _write_row(self, handle: TextIO) -> None:
        """Append one row, declaring what it cost so no phase is billed for the profiler."""
        line = json.dumps(_compact(self.take())) + "\n"
        before = read_io_snapshot(self._process)
        try:
            handle.write(line)
            handle.flush()
        except OSError:
            self.write_failures += 1
            return
        after = read_io_snapshot(self._process)
        if before.available and after.available:
            record_bytes_written(
                len(line.encode("utf-8")),
                max(0, after.write_bytes - before.write_bytes),
            )

    def _add_process_metrics(self, sample: Sample) -> None:
        if self._process is None:
            return
        if self.capabilities.memory:
            sample.rss = self._process.memory_info().rss
        # Before the I/O branch, which returns early when its counters are unavailable. Read
        # after it, a platform with no per-process byte counters would lose its CPU readings
        # too, for no reason.
        if self.capabilities.cpu:
            sample.cpu_percent = self._process.cpu_percent()
        if self.capabilities.io:
            counters = read_io_snapshot(self._process)
            sample.io_ok = counters.available
            if not counters.available:
                return
            sample.read_bytes = counters.read_bytes
            sample.write_bytes = counters.write_bytes
            sample.read_chars = counters.read_chars
            sample.write_chars = counters.write_chars
            sample.self_write_chars, sample.self_write_bytes = bytes_written()

    def _add_cuda_metrics(self, sample: Sample) -> None:
        if not self.capabilities.cuda:
            return
        torch = torch_module()
        if torch is None:
            return
        sample.cuda_alloc = torch.cuda.memory_allocated()
        sample.cuda_reserved = torch.cuda.memory_reserved()

    def _add_gpu_utilisation(self, sample: Sample) -> None:
        """Read every visible device's busy percentage, and this process's share of it.

        ``gpu_util`` stays the mean across devices so that a single-GPU run reads exactly as
        it did before this became per-device, and so that sample files written by either
        version parse under both.
        """
        if not self.capabilities.gpu_util or not self._devices:
            return
        nvml = nvml_module()
        if nvml is None:
            return
        for index, handle in self._devices:
            busy = _device_utilisation(nvml, handle)
            if busy >= 0:
                sample.gpu_utils[index] = busy
            ours = self._process_utilisation(nvml, handle)
            if ours >= 0:
                sample.gpu_proc_utils[index] = ours
        if sample.gpu_utils:
            sample.gpu_util = sum(sample.gpu_utils.values()) / len(sample.gpu_utils)

    def _process_utilisation(self, nvml: ModuleType, handle: Any) -> float:  # noqa: ANN401
        """Mean SM utilisation NVML attributes to *this* pid since the previous sample.

        The timestamp cursor advances so each internal NVML sample is counted once. NVML
        raises rather than returning an empty list when the window held no samples, which is
        the ordinary case for an idle process, so the failure path here is not exceptional.
        """
        try:
            samples = nvml.nvmlDeviceGetProcessUtilization(handle, self._proc_util_since)
        except Exception:  # noqa: BLE001 - "no samples in window" is reported as an error
            return -1.0
        mine = [s for s in samples if getattr(s, "pid", -1) == os.getpid()]
        self._proc_util_since = max(
            (getattr(s, "timeStamp", 0) for s in samples),
            default=self._proc_util_since,
        )
        if not mine:
            return -1.0
        return sum(float(s.smUtil) for s in mine) / len(mine)


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


def _device_utilisation(nvml: ModuleType, handle: Any) -> float:  # noqa: ANN401
    """Whole-device busy percentage: any kernel, any process, this device."""
    try:
        return float(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception:  # noqa: BLE001 - a transient NVML error must not kill the thread
        return -1.0


def _open_devices() -> list[tuple[int, Any]]:
    """Resolve a handle per visible device once, so sampling costs no lookups.

    Indices are NVML's, which are the machine's physical devices — deliberately not torch's
    logical ones. ``CUDA_VISIBLE_DEVICES`` renumbers what your process can use, but the
    question this block answers is whether the *device* is busy, including with work from
    processes that are not yours.
    """
    nvml = nvml_module()
    if nvml is None:
        return []
    try:
        count = int(nvml.nvmlDeviceGetCount())
    except Exception:  # noqa: BLE001 - no usable device enumeration means no GPU block
        return []
    devices: list[tuple[int, Any]] = []
    for index in range(count):
        try:
            devices.append((index, nvml.nvmlDeviceGetHandleByIndex(index)))
        except Exception:  # noqa: BLE001 - skip a device that cannot be addressed
            continue
    return devices


def _device_map(raw: object) -> dict[int, float]:
    """Rebuild a per-device mapping from JSON, whose object keys are always strings."""
    if not isinstance(raw, dict):
        return {}
    return {int(key): float(value) for key, value in raw.items()}


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
        # This probe doubles as the priming call. ``cpu_percent`` reports the average since
        # its own previous call on this object, so the first one has nothing to difference
        # against and always returns a meaningless 0.0. Spending it here means every row the
        # sampler goes on to write covers a real interval. Deliberate, not incidental — if
        # this probe is ever removed, the first row starts lying about an idle process.
        capabilities.cpu = _probe(process.cpu_percent)
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
    for name in (
        "rss",
        "read_bytes",
        "write_bytes",
        "read_chars",
        "write_chars",
        "self_write_chars",
        "self_write_bytes",
        "cuda_alloc",
        "cuda_reserved",
    ):
        value = getattr(sample, name)
        if value:
            row[name] = value
    # ``>= 0`` rather than truthiness: an idle process legitimately reads 0.0, and dropping
    # that would make "measured, idle" indistinguishable from "never measured".
    if sample.cpu_percent >= 0:
        row["cpu_percent"] = sample.cpu_percent
    if sample.gpu_util >= 0:
        row["gpu_util"] = sample.gpu_util
    if sample.gpu_utils:
        row["gpu_utils"] = sample.gpu_utils
    if sample.gpu_proc_utils:
        row["gpu_proc_utils"] = sample.gpu_proc_utils
    if not sample.io_ok:
        row["io_ok"] = False
    return row
