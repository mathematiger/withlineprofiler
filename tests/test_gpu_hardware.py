"""Tests that need a real GPU, a real driver and a real torch — everything the stubs assume.

The per-device utilisation code, the NVML process attribution and ``phase(sync=True)`` were
written against a hand-rolled ``SimpleNamespace`` stub of pynvml. A stub encodes what its
author believed the driver does; these tests check that belief. They skip cleanly on a
machine without the hardware, so CI stays green while the claims stay honest.

Run with the extras installed:  poetry install --extras all
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from lineprofiler.accounting import Profiler, merge_run, render
from lineprofiler.accounting.capabilities import (
    cuda_is_available,
    cuda_synchronize,
    nvml_module,
    nvtx_range_functions,
    record_function_factory,
)
from lineprofiler.accounting.sampler import ResourceSampler

requires_nvml = pytest.mark.skipif(nvml_module() is None, reason="NVML unavailable")
requires_cuda = pytest.mark.skipif(not cuda_is_available(), reason="no CUDA device")


@pytest.fixture
def run_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


# ── NVML, against the real driver ───────────────────────────────────────────


@requires_nvml
def test_process_utilisation_raises_on_an_empty_window() -> None:
    """The load-bearing assumption behind ``_process_utilisation``'s except clause.

    NVML reports "no samples in this window" as an *error*, not as an empty list. If a driver
    release ever changed that to return ``[]``, the except branch would stop being the normal
    path and this test is what would notice.
    """
    nvml = nvml_module()
    assert nvml is not None
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    future = 2**62  # a timestamp no sample can predate

    with pytest.raises(Exception, match="(?i)not found"):
        nvml.nvmlDeviceGetProcessUtilization(handle, future)


@requires_nvml
def test_every_visible_device_is_enumerated_and_read(run_dir: Path) -> None:
    nvml = nvml_module()
    assert nvml is not None
    expected = nvml.nvmlDeviceGetCount()
    sampler = ResourceSampler(run_dir / "s", 1.0, lambda: "train")

    sample = sampler.take()

    assert sampler.capabilities.gpu_util is True
    assert len(sample.gpu_utils) == expected
    assert all(0.0 <= value <= 100.0 for value in sample.gpu_utils.values())
    assert sample.gpu_util == pytest.approx(
        sum(sample.gpu_utils.values()) / len(sample.gpu_utils),
    )


@requires_nvml
def test_the_report_renders_a_real_gpu_block(run_dir: Path) -> None:
    profiler = Profiler(
        run_dir=run_dir, role="learner", enabled=True,
        snapshot_interval_s=None, sample_interval_s=0.05,
    )
    with profiler.phase("train"):
        for _ in range(4):
            profiler.count("steps", 1)
    profiler.close()

    text = render(merge_run(run_dir))

    assert "GPU 0" in text
    assert "busy" in text


# ── CUDA, against real kernels ──────────────────────────────────────────────


def _matmul_chain(torch_module: object, tensor: object, rounds: int = 12) -> None:
    """Enough queued work that launch time and completion time cannot be confused."""
    result = tensor
    for _ in range(rounds):
        result = result @ tensor  # type: ignore[operator]


@requires_cuda
def test_sync_makes_a_phase_measure_gpu_time_not_launch_time(run_dir: Path) -> None:
    """The defect this option exists for, on real hardware.

    Measured on an A100: 0.40 ms unsynchronised against 687 ms synchronised for the same
    work — a factor of 1,700. Without ``sync=True`` a phase around a forward pass reports the
    time to enqueue its kernels and nothing about their cost.
    """
    import torch

    profiler = Profiler(
        run_dir=run_dir, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    tensor = torch.randn(4096, 4096, device="cuda")
    _matmul_chain(torch, tensor)
    torch.cuda.synchronize()  # warm cuBLAS so the comparison is not about first-call cost

    with profiler.phase("launch_only"):
        _matmul_chain(torch, tensor)
    torch.cuda.synchronize()
    with profiler.phase("synced", sync=True):
        _matmul_chain(torch, tensor)
    profiler.close()

    tree = merge_run(run_dir).tree
    launch_ns = tree[("launch_only",)].wall_ns
    synced_ns = tree[("synced",)].wall_ns

    assert synced_ns > launch_ns * 5, (
        f"sync=True measured {synced_ns / 1e6:.2f}ms against an unsynchronised "
        f"{launch_ns / 1e6:.2f}ms; it is not draining the queue"
    )


@requires_cuda
def test_sync_drains_the_queue_on_entry_too(run_dir: Path) -> None:
    """Entry matters as much as exit: work an earlier phase queued must not land here."""
    import torch

    profiler = Profiler(
        run_dir=run_dir, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    tensor = torch.randn(4096, 4096, device="cuda")
    _matmul_chain(torch, tensor)
    torch.cuda.synchronize()

    _matmul_chain(torch, tensor)          # queued before the phase opens, deliberately
    with profiler.phase("innocent", sync=True):
        pass
    profiler.close()

    # The backlog is drained on entry, so the phase itself measures very little.
    assert merge_run(run_dir).tree[("innocent",)].wall_ns < 20_000_000


@requires_cuda
def test_cuda_memory_is_sampled(run_dir: Path) -> None:
    import torch

    profiler = Profiler(
        run_dir=run_dir, enabled=True, snapshot_interval_s=None, sample_interval_s=0.05,
    )
    held = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    assert profiler._sampler is not None  # noqa: SLF001
    sample = profiler._sampler.take()  # noqa: SLF001
    profiler.close()
    del held

    assert sample.cuda_alloc > 0
    assert sample.cuda_reserved >= sample.cuda_alloc


@requires_cuda
def test_annotation_uses_the_real_nvtx_and_record_function(run_dir: Path) -> None:
    """With torch installed these resolve to real callables; the pairing must still hold."""
    assert nvtx_range_functions() is not None
    assert record_function_factory() is not None
    assert cuda_synchronize() is not None

    profiler = Profiler(
        run_dir=run_dir, enabled=True, snapshot_interval_s=None,
        sample_interval_s=None, annotate=True,
    )
    with profiler.phase("outer"), profiler.phase("inner"):
        profiler.count("units", 2)
    with pytest.raises(ValueError, match="expected"), profiler.phase("boom"):
        raise ValueError("expected")
    profiler.close()

    tree = merge_run(run_dir).tree
    assert tree[("outer",)].calls == 1
    assert tree[("outer", "inner")].counters == {"units": 2}
    assert tree[("boom",)].calls == 1


@requires_cuda
def test_the_torch_backend_writes_a_trace(run_dir: Path) -> None:
    """``backend="torch"`` for a bounded window, exercised against the real profiler."""
    import torch

    profiler = Profiler(
        run_dir=run_dir, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
        backend="torch", backend_window=(2, 3), window_phase="iteration",
    )
    tensor = torch.randn(512, 512, device="cuda")
    for _ in range(5):
        with profiler.phase("iteration"), profiler.phase("matmul"):
            _matmul_chain(torch, tensor, rounds=2)
    torch.cuda.synchronize()
    profiler.close()

    artifacts = merge_run(run_dir).backend_artifacts()
    assert artifacts, "the window never produced a trace"
    assert Path(str(artifacts[0]["artifact"])).exists()


# ── the CUDA context, against the real driver ───────────────────────────────


@requires_nvml
def test_compute_running_processes_reports_a_pid_and_its_memory() -> None:
    """The load-bearing assumption behind ``_process_memory_on``.

    The stub returns objects with ``pid`` and ``usedGpuMemory``, and the sampler reads both by
    name. If a driver release renamed either, or started reporting ``usedGpuMemory`` as
    ``None`` on an ordinary device, the VRAM-held row would silently become "n/a" and this is
    what would notice.
    """
    import torch

    nvml = nvml_module()
    assert nvml is not None
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    held = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()

    mine = [
        process for process in nvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if process.pid == os.getpid()
    ]
    del held

    assert mine, "this process holds a tensor; NVML must list it"
    assert mine[0].usedGpuMemory is not None
    assert mine[0].usedGpuMemory > 0


@requires_nvml
@requires_cuda
def test_the_device_sees_more_than_the_allocator_does(run_dir: Path) -> None:
    """The gap the second VRAM row exists to show: the CUDA primary context.

    Roughly 414 MiB on an A100 and invisible to ``torch.cuda.memory_reserved``, which is why a
    report built on the allocator alone showed 304.8 MB where the card held 2.7 GB.
    """
    import torch

    profiler = Profiler(
        run_dir=run_dir, enabled=True, snapshot_interval_s=None, sample_interval_s=0.05,
    )
    held = torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()
    assert profiler._sampler is not None  # noqa: SLF001
    sample = profiler._sampler.take()  # noqa: SLF001
    profiler.close()
    del held

    assert sample.cuda_proc_used > sample.cuda_reserved, "the context is the difference"


def test_is_available_does_not_open_a_cuda_context() -> None:
    """The measurement the whole ``cuda_sync`` gate rests on.

    Must run in a fresh interpreter: this process may already have initialised CUDA, and the
    claim is about a process that has not. Skips rather than fails where torch is absent, so
    the file's other tests still run.
    """
    probe = (
        "import torch;"
        "print(torch.cuda.is_available(), torch.cuda.is_initialized())"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        pytest.skip("torch is not importable in a fresh interpreter here")

    available, initialised = result.stdout.split()
    if available != "True":
        pytest.skip("no CUDA device visible")
    assert initialised == "False", "is_available() must not initialise the driver"


@requires_cuda
def test_a_sync_phase_leaves_a_cpu_only_process_without_a_context() -> None:
    """End to end, in a fresh interpreter: profiling must not create what it measures.

    The regression is worth a subprocess. In-process the answer is decided by whatever ran
    before it, and the whole defect was that one ``sync=True`` phase in a worker that touches
    no tensor used to be enough to hold ~414 MiB for the life of the process.
    """
    probe = (
        "import tempfile, torch;"
        "from lineprofiler.accounting import Profiler;"
        "d = tempfile.mkdtemp();"
        "p = Profiler(run_dir=d, enabled=True, snapshot_interval_s=None,"
        " sample_interval_s=None);"
        "ctx = p.phase('act', sync=True);"
        "ctx.__enter__(); ctx.__exit__(); p.close();"
        "print(torch.cuda.is_initialized())"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", "a sync=True phase opened a CUDA context"
