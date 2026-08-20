"""Tests for roles, the resource sampler, derived analysis, comparison and backends.

Phases 2 and 4: reporting by role, memory/IO/GPU sampling, and run comparison.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lineprofiler.accounting import Backend, Profiler, merge_run, render
from lineprofiler.accounting import hardware as hardware_module
from lineprofiler.accounting import sampler as sampler_module
from lineprofiler.accounting.analysis import (
    analyse,
    analyse_processes,
    format_bytes,
    sparkline,
)
from lineprofiler.accounting.backend import BackendWindow
from lineprofiler.accounting.cli import main as cli_main
from lineprofiler.accounting.compare import compare, comparison_as_dict, render_comparison
from lineprofiler.accounting.phasetree import PhaseStats
from lineprofiler.accounting.report import (
    _dominant_rows,
    _gpu_block,
    _percent_of,
    pooled_capacity,
    report_as_dict,
)
from lineprofiler.accounting.sampler import ResourceSampler, Sample, _compact, read_samples
from lineprofiler.accounting.snapshot import imbalance_of, new_run_id
from lineprofiler.accounting.trace import FLAG_ASYNC_UNSYNCED

# ── roles ───────────────────────────────────────────────────────────────────


def _run_worker(
    run_dir: Path, role: str, phases: dict[str, float], run_id: str | None = None,
) -> None:
    """Record one process's worth of phases under a role, then close cleanly.

    Each call opens and closes its own ``Profiler``, so nothing here is still open by the
    time the next call constructs its own — the two mint unrelated run ids unless the
    caller pins one, in which case they read as one run with several roles rather than two
    superseding attempts of the same directory.
    """
    profiler = Profiler(
        run_dir=run_dir,
        role=role,
        run_id=run_id,
        enabled=True,
        snapshot_interval_s=None,
        sample_interval_s=None,
    )
    for name, seconds in phases.items():
        with profiler.phase(name):
            time.sleep(seconds)
    profiler.close()


def test_roles_are_reported_separately(tmp_path: Path) -> None:
    run_id = new_run_id()
    _run_worker(tmp_path, "learner", {"train_step": 0.02}, run_id=run_id)
    _run_worker(tmp_path, "actor", {"self_play": 0.01}, run_id=run_id)

    run = merge_run(tmp_path)

    assert set(run.roles) == {"learner", "actor"}
    assert ("train_step",) in run.tree_of("learner")
    assert ("train_step",) not in run.tree_of("actor")
    assert ("self_play",) in run.tree_of("actor")


def test_unknown_role_gives_an_empty_tree(tmp_path: Path) -> None:
    _run_worker(tmp_path, "learner", {"train_step": 0.01})
    assert merge_run(tmp_path).tree_of("nonexistent") == {}


def test_role_comes_from_the_environment_when_not_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("LINEPROFILER_ROLE", "inference")
    profiler = Profiler(run_dir=tmp_path, enabled=True, snapshot_interval_s=None,
                        sample_interval_s=None)
    assert profiler.role == "inference"
    profiler.close()


def test_explicit_role_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINEPROFILER_ROLE", "inference")
    assert Profiler(enabled=False, role="learner").role == "learner"


def test_report_shows_a_block_per_role(tmp_path: Path) -> None:
    run_id = new_run_id()
    _run_worker(tmp_path, "learner", {"train_step": 0.02}, run_id=run_id)
    _run_worker(tmp_path, "actor", {"self_play": 0.01}, run_id=run_id)

    text = render(merge_run(tmp_path))

    assert "LEARNER" in text
    assert "ACTOR" in text
    assert "DOMINANT PHASES" in text


def test_report_shows_entry_counts_without_any_counter(tmp_path: Path) -> None:
    """The entries column comes from ``PhaseStats.calls``, which every phase records itself.

    This is what makes a ``count()`` that only counts entries deletable: the number it was
    added to expose is on the row already, for free and without the call.
    """
    profiler = Profiler(
        run_dir=tmp_path, role="actor", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    for _ in range(7):
        with profiler.phase("step"):
            time.sleep(0.001)
    profiler.close()

    rows = render(merge_run(tmp_path)).split("\n")
    header = next(row for row in rows if row.startswith("DOMINANT PHASES"))
    step = next(row for row in rows if row.startswith("step"))

    assert "entries" in header
    assert len(step) == len(header)
    assert step.split()[:2] == ["step", "7"]
    assert not any(row.lstrip().startswith("+ ") for row in rows)


def test_marked_rows_keep_the_column_width_of_an_unmarked_one() -> None:
    """A ``~``/``†`` prefix eats into the label field, so the label must truncate to match.

    Truncating to a fixed width regardless of the marks printed a two-mark row one column
    wider than its neighbours, which is only visible once a label is long enough to be cut.
    """
    def stats(stride: int = 0, async_entries: int = 0) -> PhaseStats:
        one = PhaseStats(
            calls=100, wall_ns=10_000_000, cpu_ns=10_000_000,
            sample_stride=stride, async_entries=async_entries,
        )
        one.hist.observe(100_000)
        return one

    tree: dict[tuple[str, ...], PhaseStats] = {
        ("iteration", "a_long_enough_phase_name"): stats(),
        ("iteration", "b_long_enough_phase_name"): stats(stride=10),
        ("iteration", "c_long_enough_phase_name"): stats(async_entries=100),
        ("iteration", "d_long_enough_phase_name"): stats(stride=10, async_entries=100),
    }
    header, *rows = _dominant_rows(tree)[: len(tree) + 1]

    assert all(len(row) == len(header) for row in rows), rows


def test_imbalance_of_known_totals() -> None:
    class _Worker:
        def __init__(self, wall_ns: int) -> None:
            self.wall_ns = wall_ns

    assert imbalance_of([]) == 1.0
    assert imbalance_of([_Worker(100)]) == pytest.approx(1.0)  # type: ignore[list-item]
    assert imbalance_of([_Worker(3000), _Worker(7000)]) == pytest.approx(1.4)  # type: ignore[list-item]


# ── sampler ─────────────────────────────────────────────────────────────────


def test_sampler_writes_rows_tagged_with_the_current_phase(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=0.05,
    )
    with profiler.phase("outer"), profiler.phase("inner"):
        time.sleep(0.3)
    profiler.close()

    writer = profiler._writer  # noqa: SLF001
    assert writer is not None
    samples = read_samples(writer.samples_path)

    assert samples, "the sampler must produce rows"
    assert any(sample.phase == "outer/inner" for sample in samples)


def test_sampler_can_be_disabled(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler.phase("work"):
        time.sleep(0.1)
    profiler.close()

    writer = profiler._writer  # noqa: SLF001
    assert profiler._sampler is None  # noqa: SLF001
    assert writer is not None
    assert not writer.samples_path.exists()


def test_sampler_reports_which_capabilities_are_available(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=5.0,
    )
    sampler = profiler._sampler  # noqa: SLF001
    assert sampler is not None
    capabilities = sampler.capabilities
    profiler.close()

    assert isinstance(capabilities.describe(), str)
    assert set(capabilities.as_dict()) == {"memory", "io", "cpu", "cuda", "gpu_util"}


def test_written_bytes_are_reported_within_a_band(tmp_path: Path) -> None:
    """Page cache and buffering make exactness impossible, so assert a band, not a value."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=0.05,
    )
    sampler = profiler._sampler  # noqa: SLF001
    assert sampler is not None
    if not sampler.capabilities.io:
        profiler.close()
        pytest.skip("io_counters unavailable on this platform")

    payload = b"x" * (1024 * 1024)
    target = tmp_path / "payload.bin"
    with profiler.phase("write_phase"):
        with target.open("wb") as handle:
            for _ in range(32):  # 32 MB
                handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        time.sleep(0.2)
    profiler.close()

    analysis = analyse_processes(merge_run(tmp_path).samples_by_process())
    written = analysis.totals.write_bytes
    assert 16 * 1024**2 < written < 96 * 1024**2, f"reported {written} bytes"


def test_sampler_thread_does_not_block_interpreter_exit(tmp_path: Path) -> None:
    import subprocess
    import sys

    script = (
        "from lineprofiler.accounting import Profiler\n"
        f"p = Profiler(run_dir={str(tmp_path)!r}, enabled=True, sample_interval_s=0.05)\n"
        "with p.phase('x'):\n"
        "    pass\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], timeout=30, check=False)
    assert completed.returncode == 0


# ── analysis ────────────────────────────────────────────────────────────────


def _series(rows: list[tuple[float, str, int, int]]) -> list[Sample]:
    return [
        Sample(t=t, phase=phase, rss=rss, write_bytes=written) for t, phase, rss, written in rows
    ]


def test_analysis_of_fewer_than_two_samples_is_empty() -> None:
    assert not analyse([]).has_samples
    assert not analyse([Sample(t=1.0, phase="a", rss=10)]).has_samples


def test_bytes_are_attributed_to_the_phase_open_during_the_interval() -> None:
    samples = _series([
        (0.0, "load", 100, 0),
        (1.0, "load", 100, 5_000),
        (2.0, "train", 100, 5_500),
    ])

    analysis = analyse(samples)

    assert analysis.io_by_phase["load"].write_bytes == 5_500
    assert "train" not in analysis.io_by_phase
    assert analysis.totals.write_bytes == 5_500


def test_a_growing_workload_shows_a_positive_rss_slope() -> None:
    leaking = _series([(float(i), "grow", 1_000_000 + i * 50_000, 0) for i in range(10)])
    steady = _series([(float(i), "flat", 1_000_000, 0) for i in range(10)])

    assert analyse(leaking).memory.slope_bytes_per_s == pytest.approx(50_000, rel=0.01)
    assert analyse(steady).memory.slope_bytes_per_s == pytest.approx(0.0, abs=1.0)
    assert analyse(leaking).memory_by_phase["grow"].slope_bytes_per_s > 0


def test_gpu_utilisation_averages_only_real_readings() -> None:
    samples = [
        Sample(t=0.0, phase="a", gpu_util=80.0),
        Sample(t=1.0, phase="a", gpu_util=-1.0),
        Sample(t=2.0, phase="a", gpu_util=60.0),
    ]
    assert analyse(samples).gpu_util_mean == pytest.approx(70.0)


def test_gpu_utilisation_is_negative_when_never_sampled() -> None:
    samples = [Sample(t=0.0, phase="a"), Sample(t=1.0, phase="a")]
    assert analyse(samples).gpu_util_mean == -1.0


def test_sparkline_scales_to_its_own_peak() -> None:
    assert sparkline([]) == ""
    assert sparkline([0.0, 0.0]) == "  "
    line = sparkline([0.0, 5.0, 10.0])
    assert len(line) == 3
    assert line[-1] == "█"


def test_format_bytes_uses_binary_units() -> None:
    assert format_bytes(512) == "512 B"
    assert format_bytes(1024) == "1.0 KB"
    assert format_bytes(5 * 1024**3) == "5.0 GB"
    assert format_bytes(-1024) == "-1.0 KB"


def test_two_processes_are_differenced_separately() -> None:
    """Cumulative counters from different processes must never be differenced together."""
    first = _series([(0.0, "checkpoint", 100, 0), (1.0, "checkpoint", 100, 10_000)])
    second = _series([(0.5, "replay", 100, 5_000), (1.5, "replay", 100, 15_000)])

    pooled_wrongly = analyse(sorted(first + second, key=lambda s: s.t))
    per_process = analyse_processes([first, second])

    assert per_process.totals.write_bytes == 20_000
    assert per_process.io_by_phase["checkpoint"].write_bytes == 10_000
    assert per_process.io_by_phase["replay"].write_bytes == 10_000
    assert pooled_wrongly.totals.write_bytes != 20_000, "the pooled path is the bug we fixed"


def test_memory_across_processes_is_summed() -> None:
    first = _series([(float(i), "a", 1_000_000 + i * 1000, 0) for i in range(5)])
    second = _series([(float(i), "a", 2_000_000 + i * 3000, 0) for i in range(5)])

    combined = analyse_processes([first, second])

    assert combined.memory.last_rss == first[-1].rss + second[-1].rss
    assert combined.memory.slope_bytes_per_s == pytest.approx(4000, rel=0.01)


# ── comparison ──────────────────────────────────────────────────────────────


def _build_run(run_dir: Path, per_call_s: float, entries: int) -> None:
    profiler = Profiler(
        run_dir=run_dir, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    for _ in range(entries):
        with profiler.phase("step"):
            time.sleep(per_call_s)
    profiler.close()


def test_compare_reports_a_known_ratio(tmp_path: Path) -> None:
    _build_run(tmp_path / "a", 0.01, 5)
    _build_run(tmp_path / "b", 0.02, 5)

    deltas = {d.phase: d for d in compare(merge_run(tmp_path / "a"), merge_run(tmp_path / "b"))}

    assert deltas["step"].percent == pytest.approx(100.0, rel=0.25)


def test_compare_is_insensitive_to_differing_entry_counts(tmp_path: Path) -> None:
    _build_run(tmp_path / "a", 0.01, 3)
    _build_run(tmp_path / "b", 0.01, 9)

    delta = compare(merge_run(tmp_path / "a"), merge_run(tmp_path / "b"))[0]

    assert delta.calls_a == 3
    assert delta.calls_b == 9
    assert delta.percent == pytest.approx(0.0, abs=30.0)


def test_compare_flags_phases_present_in_only_one_run(tmp_path: Path) -> None:
    _run_worker(tmp_path / "a", "main", {"shared": 0.01})
    _run_worker(tmp_path / "b", "main", {"shared": 0.01, "added": 0.01})

    deltas = {d.phase: d for d in compare(merge_run(tmp_path / "a"), merge_run(tmp_path / "b"))}

    assert deltas["added"].only_in == "B"
    assert deltas["shared"].only_in is None


def test_comparison_renders_as_text_and_json(tmp_path: Path) -> None:
    _run_worker(tmp_path / "a", "main", {"shared": 0.01})
    _run_worker(tmp_path / "b", "main", {"shared": 0.01, "added": 0.01})
    run_a, run_b = merge_run(tmp_path / "a"), merge_run(tmp_path / "b")

    text = render_comparison(run_a, run_b, "a", "b")
    payload = comparison_as_dict(run_a, run_b)

    assert "only in B" in text
    assert {entry["phase"] for entry in payload["phases"]} == {"shared", "added"}


# ── backend ─────────────────────────────────────────────────────────────────


def test_backend_parsing_accepts_names_and_rejects_unknowns() -> None:
    assert Backend.parse(None) is Backend.NONE
    assert Backend.parse("none") is Backend.NONE
    assert Backend.parse("torch") is Backend.TORCH
    assert Backend.parse(Backend.VIZTRACER) is Backend.VIZTRACER
    with pytest.raises(ValueError, match="unknown backend"):
        Backend.parse("nsys")


def test_backend_cannot_express_two_at_once() -> None:
    """The API takes a single enum value, so 'both' is not representable."""
    with pytest.raises(ValueError, match="unknown backend"):
        Backend.parse("torch+viztracer")


def test_window_starts_and_stops_on_the_configured_entries(tmp_path: Path) -> None:
    window = BackendWindow(Backend.VIZTRACER, (2, 3), "iteration", tmp_path)
    states = []
    for _ in range(5):
        window.on_phase_enter("iteration")
        states.append(window.active)
        window.on_phase_exit("iteration")
    window.close()

    # VizTracer is not installed in CI, so the window degrades rather than activating.
    assert window.entries == 5
    assert window.unavailable_reason is not None or any(states)


def test_window_ignores_phases_it_was_not_configured_for(tmp_path: Path) -> None:
    window = BackendWindow(Backend.TORCH, (1, 2), "iteration", tmp_path)
    window.on_phase_enter("something_else")
    assert window.entries == 0


def test_backend_none_never_arms_a_window(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path,
        enabled=True,
        snapshot_interval_s=None,
        sample_interval_s=None,
        backend="none",
        backend_window=(1, 2),
    )
    with profiler.phase("iteration"):
        pass
    profiler.close()
    assert profiler._window is None  # noqa: SLF001


def test_window_description_is_json_serialisable(tmp_path: Path) -> None:
    window = BackendWindow(Backend.TORCH, (1, 2), "iteration", tmp_path)
    assert json.loads(json.dumps(window.describe()))["backend"] == "torch"


# ── settings matrix ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("sample_interval_s", [None, 0.05])
@pytest.mark.parametrize("measure_cpu", [True, False])
@pytest.mark.parametrize("snapshot_interval_s", [None, 0.1])
@pytest.mark.parametrize("backend", ["none", "torch"])
def test_every_settings_combination_runs_and_reports(
    tmp_path: Path,
    sample_interval_s: float | None,
    measure_cpu: bool,
    snapshot_interval_s: float | None,
    backend: str,
) -> None:
    profiler = Profiler(
        run_dir=tmp_path,
        role="learner",
        enabled=True,
        snapshot_interval_s=snapshot_interval_s,
        sample_interval_s=sample_interval_s,
        measure_cpu=measure_cpu,
        backend=backend,
        backend_window=(1, 2),
        window_phase="iteration",
    )
    for _ in range(3):
        with profiler.phase("iteration"), profiler.phase("work"):
            profiler.count("units", 4)
            time.sleep(0.01)
    profiler.close()

    run = merge_run(tmp_path)

    assert run.unreadable == []
    assert run.tree[("iteration",)].calls == 3
    assert run.tree[("iteration", "work")].counters == {"units": 12}
    assert "LEARNER" in render(run)


# ── CLI ─────────────────────────────────────────────────────────────────────


def test_cli_report_and_compare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run_worker(tmp_path / "a", "main", {"step": 0.01})
    _run_worker(tmp_path / "b", "main", {"step": 0.01})

    assert cli_main(["report", str(tmp_path / "a")]) == 0
    assert "step" in capsys.readouterr().out

    assert cli_main(["compare", str(tmp_path / "a"), str(tmp_path / "b")]) == 0
    assert "step" in capsys.readouterr().out

    assert cli_main(["compare", str(tmp_path / "a"), str(tmp_path / "b"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["phases"]


def test_cli_report_json_is_a_usable_assertion_target(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """``compare`` had ``--json`` and ``report`` did not, so the CLI could not gate CI or
    diff sweep arms without re-implementing every derivation in the caller."""
    _run_worker(tmp_path, "main", {"step": 0.01})

    assert cli_main(["report", str(tmp_path), "--json"]) == 0
    document = json.loads(capsys.readouterr().out)

    assert document["run"]["roles"] == ["main"]
    assert document["run"]["processes"] == 1
    phases = {row["phase"]: row for row in document["roles"][0]["phases"]}
    assert phases["step"]["calls"] == 1
    assert phases["step"]["wall_ns"] > 0
    assert document["caveats"] == {"unreadable": [], "superseded": [], "stale": []}


def test_cli_report_json_honours_no_samples(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Omitted rather than zeroed: a caller must not read "no samples read" as "no I/O"."""
    _run_worker(tmp_path, "main", {"step": 0.01})

    assert cli_main(["report", str(tmp_path), "--json", "--no-samples"]) == 0

    assert "resources" not in json.loads(capsys.readouterr().out)


# ── exact per-phase I/O ─────────────────────────────────────────────────────


def test_io_phase_attributes_bytes_to_the_phase_that_wrote_them(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    if profiler.io_counters().is_empty():
        profiler.close()
        pytest.skip("per-process io counters unavailable on this platform")

    blob = b"q" * (1024 * 1024)
    with profiler.phase("quiet"):
        time.sleep(0.01)
    with profiler.phase("checkpoint", io=True), (tmp_path / "ckpt.bin").open("wb") as handle:
        for _ in range(8):
            handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    profiler.close()

    tree = merge_run(tmp_path).tree
    written = tree[("checkpoint",)].counters.get("io_write_bytes", 0)

    assert 4 * 1024**2 < written < 24 * 1024**2, f"reported {written} bytes"
    assert "io_write_bytes" not in tree[("quiet",)].counters


def test_io_phase_appears_in_the_exact_io_block(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    if profiler.io_counters().is_empty():
        profiler.close()
        pytest.skip("per-process io counters unavailable on this platform")

    with profiler.phase("save", io=True), (tmp_path / "x.bin").open("wb") as handle:
        handle.write(b"y" * (4 * 1024 * 1024))
        handle.flush()
        os.fsync(handle.fileno())
    profiler.close()

    text = render(merge_run(tmp_path))
    assert "I/O BY PHASE (measured exactly)" in text
    assert "save" in text


def test_io_flag_is_ignored_when_the_profiler_is_disabled(tmp_path: Path) -> None:
    profiler = Profiler(run_dir=tmp_path, enabled=False)
    with profiler.phase("save", io=True):
        pass
    assert profiler.merged_tree() == {}


# ── page-cached reads ───────────────────────────────────────────────────────


def _read_whole(path: Path, drop_cache: bool) -> int:
    """Read a file, optionally evicting its pages first so the bytes come off the disk."""
    fd = os.open(path, os.O_RDONLY)
    try:
        if drop_cache and hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        total = 0
        while chunk := os.read(fd, 1 << 20):
            total += len(chunk)
        return total
    finally:
        os.close(fd)


def test_page_cached_reads_are_reported_as_chars_not_disk_bytes(tmp_path: Path) -> None:
    """A warm read moves no disk bytes, so ``read_bytes`` alone reports a phase as idle.

    This is the page-cache blind spot: a training run whose dataset fits in RAM does its
    reads from memory, and the block-layer counter that ``read_bytes`` exposes correctly
    reports zero. The syscall-level counter is what shows the work happened at all.
    """
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    if profiler.io_counters().is_empty():
        profiler.close()
        pytest.skip("per-process io counters unavailable on this platform")

    payload = tmp_path / "shard.bin"
    payload.write_bytes(b"z" * (8 * 1024 * 1024))
    _read_whole(payload, drop_cache=False)  # warm the cache

    with profiler.phase("load_batch", io=True):
        assert _read_whole(payload, drop_cache=False) == 8 * 1024**2
    profiler.close()

    counters = merge_run(tmp_path).tree[("load_batch",)].counters
    assert counters.get("io_read_chars", 0) >= 8 * 1024**2
    assert counters.get("io_read_bytes", 0) < 8 * 1024**2


def test_cold_reads_report_both_disk_bytes_and_chars(tmp_path: Path) -> None:
    """A cold read moves the bytes through both counters, so disk pressure stays visible."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    if profiler.io_counters().is_empty():
        profiler.close()
        pytest.skip("per-process io counters unavailable on this platform")

    payload = tmp_path / "cold.bin"
    payload.write_bytes(b"z" * (8 * 1024 * 1024))
    os.sync()

    with profiler.phase("load_batch", io=True):
        _read_whole(payload, drop_cache=True)
    profiler.close()

    counters = merge_run(tmp_path).tree[("load_batch",)].counters
    assert counters.get("io_read_chars", 0) >= 8 * 1024**2
    if counters.get("io_read_bytes", 0) == 0:
        pytest.skip("filesystem did not honour POSIX_FADV_DONTNEED")
    assert counters["io_read_bytes"] >= 4 * 1024**2


def test_report_names_the_bytes_that_came_from_page_cache(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    if profiler.io_counters().is_empty():
        profiler.close()
        pytest.skip("per-process io counters unavailable on this platform")

    payload = tmp_path / "shard.bin"
    payload.write_bytes(b"z" * (8 * 1024 * 1024))
    _read_whole(payload, drop_cache=False)

    with profiler.phase("load_batch", io=True):
        _read_whole(payload, drop_cache=False)
    profiler.close()

    assert "from page cache" in render(merge_run(tmp_path))


# ── the profiler's own I/O ──────────────────────────────────────────────────


def test_the_profilers_own_writes_are_not_attributed_to_a_phase(tmp_path: Path) -> None:
    """The sampler and snapshot threads write to the run directory on the same process.

    Those bytes land in the process-wide counters, so without exclusion they are billed to
    whichever phase happens to be open — a phase that touched no file at all reports I/O.
    """
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=0.01, sample_interval_s=0.01,
    )
    if profiler.io_counters().is_empty():
        profiler.close()
        pytest.skip("per-process io counters unavailable on this platform")

    with profiler.phase("compute", io=True):
        time.sleep(0.3)  # long enough for many sampler rows and snapshots to be written
    profiler.close()

    counters = merge_run(tmp_path).tree[("compute",)].counters
    assert counters.get("io_write_chars", 0) == 0, f"leaked {counters} into a silent phase"
    assert counters.get("io_write_bytes", 0) == 0


# ── sampled attribution honesty ─────────────────────────────────────────────


def test_bytes_moved_outside_any_phase_are_reported_as_unattributed() -> None:
    """Bytes the sampler could not pin to a phase must be named, not billed to the root.

    A run shorter than a few sample intervals attributes most of its bytes to whatever was
    open when the baseline row was taken. Calling that "(root)" reads like a finding; it is
    really an admission that the sample rate was too coarse.
    """
    samples = [
        Sample(t=0.0, phase="", read_bytes=0),
        Sample(t=1.0, phase="load", read_bytes=900),
        Sample(t=2.0, phase="load", read_bytes=1000),
    ]
    analysis = analyse([samples][0])

    assert analysis.io_by_phase["(no phase open)"].read_bytes == 900
    assert analysis.unattributed_read_share == pytest.approx(0.9)


# ── annotation (NVTX / record_function) ─────────────────────────────────────


def test_annotation_is_off_by_default(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    assert profiler._nvtx is None  # noqa: SLF001
    assert profiler._record_function is None  # noqa: SLF001
    profiler.close()


def test_annotation_degrades_to_nothing_without_torch_or_nvtx(tmp_path: Path) -> None:
    """Neither package is required; a phase still records normally without them."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
        annotate=True,
    )
    with profiler.phase("annotated"):
        profiler.count("units", 1)
    profiler.close()

    tree = merge_run(tmp_path).tree
    assert tree[("annotated",)].calls == 1
    assert tree[("annotated",)].counters == {"units": 1}


def test_annotation_pushes_and_pops_in_matching_pairs(tmp_path: Path) -> None:
    """A stub stands in for NVTX so the pairing is verified without a GPU."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
        annotate=True,
    )
    events: list[str] = []
    profiler._nvtx = (lambda name: events.append(f"push:{name}"), lambda: events.append("pop"))  # noqa: SLF001

    with profiler.phase("outer"), profiler.phase("inner"):
        pass
    with pytest.raises(ValueError), profiler.phase("boom"):
        raise ValueError("expected")
    profiler.close()

    assert events == [
        "push:outer", "push:inner", "pop", "pop",
        "push:boom", "pop",
    ]


def test_annotation_is_ignored_when_disabled(tmp_path: Path) -> None:
    profiler = Profiler(run_dir=tmp_path, enabled=False, annotate=True)
    assert profiler._nvtx is None  # noqa: SLF001


# ── GPU utilisation, per device and per process ─────────────────────────────


def _fake_nvml(
    busy: dict[int, float],
    process_samples: dict[int, list[SimpleNamespace]] | None = None,
) -> SimpleNamespace:
    """Stand in for pynvml, so per-device sampling is verified without a GPU.

    A device whose busy value is negative raises, mimicking a transient NVML failure on one
    device of several. ``nvmlDeviceGetProcessUtilization`` raises when its window is empty,
    which is what the real one does rather than returning a list of length zero.
    """
    rows = process_samples or {}

    def get_rates(handle: int) -> SimpleNamespace:
        if busy[handle] < 0:
            raise RuntimeError("device unavailable")
        return SimpleNamespace(gpu=busy[handle])

    def get_process_utilization(handle: int, since: int) -> list[SimpleNamespace]:
        fresh = [row for row in rows.get(handle, []) if row.timeStamp > since]
        if not fresh:
            raise RuntimeError("NVML_ERROR_NOT_FOUND")
        return fresh

    return SimpleNamespace(
        nvmlDeviceGetCount=lambda: len(busy),
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetUtilizationRates=get_rates,
        nvmlDeviceGetProcessUtilization=get_process_utilization,
    )


def _sampler_with_nvml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nvml: SimpleNamespace,
) -> ResourceSampler:
    """Build a sampler whose device handles come from the stub, not from a real driver."""
    monkeypatch.setattr(sampler_module, "nvml_module", lambda: nvml)
    return ResourceSampler(tmp_path / "samples", 1.0, lambda: "train")


def _write_samples(run_dir: Path, samples: list[Sample]) -> None:
    """Attach resource samples to the worker file a test's profiler already wrote.

    Goes through the sampler's own compaction, so the report is fed exactly the JSON a real
    run would have produced rather than a hand-written approximation of it.
    """
    worker = next(iter((run_dir / "workers").rglob("w_*.json")))
    rows = "\n".join(json.dumps(_compact(sample)) for sample in samples)
    worker.with_suffix(".samples").write_text(rows + "\n", encoding="utf-8")


def test_every_visible_device_gets_its_own_utilisation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = _sampler_with_nvml(tmp_path, monkeypatch, _fake_nvml({0: 80.0, 1: 40.0}))

    sample = sampler.take()

    assert sample.gpu_utils == {0: 80.0, 1: 40.0}
    assert sample.gpu_util == pytest.approx(60.0), "the scalar stays the mean over devices"


def test_a_device_that_fails_to_report_does_not_lose_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = _sampler_with_nvml(tmp_path, monkeypatch, _fake_nvml({0: -1.0, 1: 40.0}))

    assert sampler.take().gpu_utils == {1: 40.0}


def test_only_this_processes_rows_count_towards_its_own_utilisation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    nvml = _fake_nvml(
        {0: 90.0},
        {0: [
            SimpleNamespace(pid=os.getpid(), smUtil=30.0, timeStamp=10),
            SimpleNamespace(pid=os.getpid() + 1, smUtil=55.0, timeStamp=11),
        ]},
    )
    sampler = _sampler_with_nvml(tmp_path, monkeypatch, nvml)

    sample = sampler.take()

    assert sample.gpu_proc_utils == {0: 30.0}, "the neighbour's 55% is not ours"
    assert sample.gpu_utils == {0: 90.0}, "but the device is busy with both"


def test_process_utilisation_is_absent_when_nvml_reports_no_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = _sampler_with_nvml(tmp_path, monkeypatch, _fake_nvml({0: 90.0}))

    assert sampler.take().gpu_proc_utils == {}


def test_per_device_utilisation_survives_the_sample_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON object keys are strings; the device indices must come back as ints."""
    nvml = _fake_nvml(
        {0: 80.0, 1: 40.0},
        {1: [SimpleNamespace(pid=os.getpid(), smUtil=25.0, timeStamp=3)]},
    )
    sampler = _sampler_with_nvml(tmp_path, monkeypatch, nvml)
    path = tmp_path / "samples"
    with path.open("a", encoding="utf-8") as handle:
        sampler._write_row(handle)  # noqa: SLF001

    restored = read_samples(path)[0]

    assert restored.gpu_utils == {0: 80.0, 1: 40.0}
    assert restored.gpu_proc_utils == {1: 25.0}


def test_workers_sum_their_own_share_but_average_the_devices(tmp_path: Path) -> None:
    """Two actors on one GPU place 60% on it between them; the device is 90% busy once."""
    first = [
        Sample(t=0.0, phase="train", gpu_utils={0: 90.0}, gpu_proc_utils={0: 40.0}),
        Sample(t=1.0, phase="train", gpu_utils={0: 90.0}, gpu_proc_utils={0: 40.0}),
    ]
    second = [
        Sample(t=0.0, phase="train", gpu_utils={0: 90.0}, gpu_proc_utils={0: 20.0}),
        Sample(t=1.0, phase="train", gpu_utils={0: 90.0}, gpu_proc_utils={0: 20.0}),
    ]

    devices = analyse_processes([first, second]).gpu_devices

    assert len(devices) == 1
    assert devices[0].busy_mean == pytest.approx(90.0)
    assert devices[0].ours_mean == pytest.approx(60.0)


def test_windows_without_our_kernels_count_as_idle_not_as_missing() -> None:
    """A worker busy in one window of four used a quarter of the device, not all of it."""
    samples = [
        Sample(t=0.0, phase="train", gpu_utils={0: 50.0}, gpu_proc_utils={0: 80.0}),
        Sample(t=1.0, phase="train", gpu_utils={0: 50.0}),
        Sample(t=2.0, phase="train", gpu_utils={0: 50.0}),
        Sample(t=3.0, phase="train", gpu_utils={0: 50.0}),
    ]

    assert analyse(samples).gpu_devices[0].ours_mean == pytest.approx(20.0)


def test_a_device_never_attributed_to_us_reports_unknown_not_zero() -> None:
    samples = [
        Sample(t=0.0, phase="train", gpu_utils={0: 50.0}),
        Sample(t=1.0, phase="train", gpu_utils={0: 50.0}),
    ]

    assert analyse(samples).gpu_devices[0].ours_mean == -1.0


def test_report_shows_a_row_per_device_with_our_share(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, role="learner", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler.phase("train"):
        pass
    profiler.close()
    _write_samples(tmp_path, [
        Sample(t=0.0, phase="train", rss=1_000, gpu_utils={0: 90.0, 1: 10.0},
               gpu_proc_utils={0: 45.0}),
        Sample(t=1.0, phase="train", rss=1_000, gpu_utils={0: 90.0, 1: 10.0},
               gpu_proc_utils={0: 45.0}),
    ])

    text = render(merge_run(tmp_path))

    assert "GPU 0" in text
    assert "GPU 1" in text
    assert "45.0%" in text, "our share of device 0"
    assert "this run" in text


def test_report_falls_back_to_one_figure_for_older_sample_files(tmp_path: Path) -> None:
    """Files written before utilisation was per-device still render their scalar."""
    profiler = Profiler(
        run_dir=tmp_path, role="learner", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler.phase("train"):
        pass
    profiler.close()
    _write_samples(tmp_path, [
        Sample(t=0.0, phase="train", rss=1_000, gpu_util=71.0),
        Sample(t=1.0, phase="train", rss=1_000, gpu_util=71.0),
    ])

    text = render(merge_run(tmp_path))

    assert "Utilisation (sampled)" in text
    assert "71.0%" in text
    assert "GPU 0" not in text


# ── CUDA-synchronised phases ────────────────────────────────────────────────


def test_sync_phase_drains_the_queue_at_both_ends(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    events: list[str] = []
    profiler._cuda_sync = lambda: events.append("sync")  # noqa: SLF001

    with profiler.phase("train", sync=True):
        events.append("body")
    profiler.close()

    assert events == ["sync", "body", "sync"]


def test_sync_happens_before_the_exit_clock_is_read(tmp_path: Path) -> None:
    """The whole point: kernels still running at the end of the phase are its cost."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    profiler._cuda_sync = lambda: time.sleep(0.02)  # noqa: SLF001

    with profiler.phase("train", sync=True):
        pass
    profiler.close()

    assert merge_run(tmp_path).tree[("train",)].wall_ns >= 20_000_000


def test_sync_is_a_no_op_without_a_cuda_device(tmp_path: Path) -> None:
    """CI has no GPU: the phase must record exactly as an unsynchronised one does."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler.phase("train", sync=True):
        profiler.count("steps", 3)
    profiler.close()

    tree = merge_run(tmp_path).tree
    assert tree[("train",)].calls == 1
    assert tree[("train",)].counters == {"steps": 3}


def test_sync_is_ignored_when_the_profiler_is_disabled(tmp_path: Path) -> None:
    profiler = Profiler(run_dir=tmp_path, enabled=False)
    assert profiler._cuda_sync is None  # noqa: SLF001
    with profiler.phase("train", sync=True):
        pass


# ── un-awaited device work ──────────────────────────────────────────────────


def test_an_async_phase_is_marked_in_the_report_and_a_synced_one_is_not(tmp_path: Path) -> None:
    """The acceptance test for the whole feature, and the workflow it is built around.

    A phase around unsynchronised device work reports submission time, which reads exactly
    like device time and is smaller by orders of magnitude. Flipping it to ``sync=True`` is
    how you turn one into the other, so the mark must disappear when you do — otherwise the
    reader cannot tell which of the two a given run measured.
    """
    profiler = Profiler(
        run_dir=tmp_path, role="server", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler:
        for _ in range(3):
            with profiler.phase("submit", async_work=True):
                pass
            with profiler.phase("awaited", sync=True, async_work=True):
                pass
        profiler.snapshot()

    text = render(merge_run(tmp_path))

    assert "†submit" in text
    assert "†awaited" not in text
    assert "wall time excludes un-awaited device work" in text
    assert "sync=True" in text


def test_a_phase_entered_both_ways_reports_the_share_that_was_unawaited(tmp_path: Path) -> None:
    """A mixture must read as a mixture: "this number is partly wrong" is a distinct claim."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler:
        for _ in range(2):
            with profiler.phase("forward", async_work=True):
                pass
        for _ in range(6):
            with profiler.phase("forward"):
                pass
        profiler.snapshot()

    stats = merge_run(tmp_path).tree[("forward",)]

    assert stats.calls == 8
    assert stats.async_entries == 2
    assert "2 of 8 entries" in render(merge_run(tmp_path))


def test_an_ordinary_phase_is_never_marked(tmp_path: Path) -> None:
    """The default must stay silent, or the mark distinguishes nothing.

    ``sync=False`` is the default, so treating it as the declaration would mark every phase
    of every run — including every phase of a CPU-only one, which has no device work to be
    un-awaited.
    """
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler:
        with profiler.phase("cpu_only"):
            pass
        profiler.snapshot()

    assert merge_run(tmp_path).tree[("cpu_only",)].async_entries == 0
    assert "†" not in render(merge_run(tmp_path))


def test_an_async_span_carries_the_flag(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None,
        sample_interval_s=None, trace=True,
    )
    with profiler:
        with profiler.phase("submit", async_work=True):
            pass
        with profiler.phase("plain"):
            pass

    trace = merge_run(tmp_path, with_trace=True).workers[0].trace
    flagged = {
        trace.path_of(span.phase_id)[-1]
        for span in trace.spans
        if span.flags & FLAG_ASYNC_UNSYNCED
    }

    assert flagged == {"submit"}


def test_the_async_count_survives_the_snapshot_round_trip(tmp_path: Path) -> None:
    """A field the hot path writes directly must also serialise, or it is lost at the merge."""
    stats = PhaseStats(calls=4, async_entries=3)

    assert PhaseStats.from_dict(stats.to_dict()).async_entries == 3


def test_a_worker_file_without_the_field_reads_as_nothing_async() -> None:
    """Back-compatibility: a 0.6.0 file declared nothing async, because it could not."""
    old = {
        "calls": 2, "wall_ns": 10, "cpu_ns": 5, "child_wall_ns": 0,
        "hist": {}, "counters": {},
    }

    assert PhaseStats.from_dict(old).async_entries == 0


def test_merging_sums_the_async_counts(tmp_path: Path) -> None:
    """One worker submitting un-awaited work taints the merged node, as sampling does."""
    first = PhaseStats(calls=3, async_entries=3)
    second = PhaseStats(calls=5, async_entries=0)
    first.merge(second)

    assert first.calls == 8
    assert first.async_entries == 3


# ── the pattern the README documents under "Using it in tests" ──────────────


def test_a_merged_run_is_a_usable_assertion_target(tmp_path: Path) -> None:
    """Pins the README's "Using it in tests" example, so the section cannot rot.

    The point of the section is that a merged run records cross-process behaviour that no
    other artifact does — a role that never started, a phase that never ran, a worker that
    died before its first flush.
    """
    profiler = Profiler(
        run_dir=tmp_path, role="evaluator", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler.phase("iteration"), profiler.phase("mcts"):
        profiler.count("simulations", 64)
    profiler.close()

    run = merge_run(tmp_path, with_samples=False)

    assert "evaluator" in run.roles
    assert run.tree[("iteration", "mcts")].calls == 1
    assert run.tree[("iteration", "mcts")].counters == {"simulations": 64}
    assert run.unreadable == []
    latest = max(w.written_at for w in run.workers)
    assert all(latest - w.written_at < 300 for w in run.workers)


def test_a_role_that_never_started_is_visibly_absent(tmp_path: Path) -> None:
    """The evaluator incident: the supervisor logged it as alive, and it did no work."""
    profiler = Profiler(
        run_dir=tmp_path, role="learner", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler.phase("train_step"):
        pass
    profiler.close()

    assert "evaluator" not in merge_run(tmp_path, with_samples=False).roles


# ── counter distribution ────────────────────────────────────────────────────


def test_a_hard_cap_is_distinguishable_from_bursty_arrival(tmp_path: Path) -> None:
    """The acceptance test: a mean cannot tell these apart, and they need opposite fixes.

    "Mean 1.9 against a cap of 2" is equally consistent with "always exactly 2" (the cap is
    binding, raise it) and "usually 1, occasionally 8" (the supply is not there, batch
    harder). The sum reports the same figure for both.
    """
    capped = Profiler(
        run_dir=tmp_path / "capped", role="server", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    with capped:
        for _ in range(8):
            with capped.phase("forward"):
                capped.count("rows", 2)
        capped.snapshot()

    bursty = Profiler(
        run_dir=tmp_path / "bursty", role="server", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    with bursty:
        for amount in (1, 1, 1, 1, 1, 1, 1, 9):
            with bursty.phase("forward"):
                bursty.count("rows", amount)
        bursty.snapshot()

    capped_stats = merge_run(tmp_path / "capped").tree[("forward",)]
    bursty_stats = merge_run(tmp_path / "bursty").tree[("forward",)]

    assert capped_stats.counters["rows"] == bursty_stats.counters["rows"] == 16
    assert (capped_stats.counter_min["rows"], capped_stats.counter_max["rows"]) == (2, 2)
    assert (bursty_stats.counter_min["rows"], bursty_stats.counter_max["rows"]) == (1, 9)
    assert "always 2" in render(merge_run(tmp_path / "capped"))
    assert "1..9" in render(merge_run(tmp_path / "bursty"))


def test_merging_extremes_does_not_treat_a_worker_total_as_one_observation(
    tmp_path: Path,
) -> None:
    """The trap: merging through ``add_count`` would make the maximum a worker's whole sum."""
    first = PhaseStats()
    for amount in (2, 3):
        first.add_count("rows", amount)
    second = PhaseStats()
    for amount in (1, 4):
        second.add_count("rows", amount)
    first.merge(second)

    assert first.counters["rows"] == 10
    assert first.counter_min["rows"] == 1
    assert first.counter_max["rows"] == 4


def test_counter_extremes_survive_the_snapshot_round_trip() -> None:
    stats = PhaseStats()
    stats.add_count("rows", 7)

    restored = PhaseStats.from_dict(stats.to_dict())

    assert restored.counter_min == {"rows": 7}
    assert restored.counter_max == {"rows": 7}


def test_a_worker_file_without_extremes_reads_as_having_none() -> None:
    old = {
        "calls": 1, "wall_ns": 10, "cpu_ns": 5, "child_wall_ns": 0,
        "hist": {}, "counters": {"rows": 4},
    }
    restored = PhaseStats.from_dict(old)

    assert restored.counters == {"rows": 4}
    assert restored.counter_min == {}


# ── GPU utilisation by phase ────────────────────────────────────────────────


def test_a_gpu_heavy_phase_reads_differently_from_a_gpu_idle_one() -> None:
    """The acceptance test, and the join that refutes an unsynchronised forward pass.

    A phase named ``forward`` holding most of a server's time at 7% device utilisation is a
    contradiction — but only if the two numbers appear on the same page.
    """
    samples = [
        Sample(t=float(i), phase="train_step", gpu_utils={0: 80.0}, cuda_reserved=3_000_000)
        for i in range(10)
    ] + [
        Sample(t=float(10 + i), phase="forward", gpu_utils={0: 6.0}, cuda_reserved=1_000_000)
        for i in range(10)
    ]

    by_phase = analyse_processes([samples]).gpu_by_phase

    assert by_phase["train_step"].quantile(0.5) == pytest.approx(80.0)
    assert by_phase["forward"].quantile(0.5) == pytest.approx(6.0)
    assert by_phase["train_step"].peak_cuda_reserved == 3_000_000


def test_gpu_samples_with_no_phase_open_are_named_not_billed_to_the_root() -> None:
    """The same admission the byte path makes: coarse sampling, not a finding about the root."""
    samples = [Sample(t=float(i), phase="", gpu_utils={0: 50.0}) for i in range(3)]

    assert "(no phase open)" in analyse_processes([samples]).gpu_by_phase


def test_the_gpu_phase_table_reaches_the_report() -> None:
    samples = [
        Sample(t=float(i), phase="forward", gpu_utils={0: 9.0}, cuda_reserved=2_000_000)
        for i in range(5)
    ]

    text = _gpu_block(analyse_processes([samples]))

    assert "GPU BY PHASE (sampled)" in text
    assert "forward" in text
    assert "submits device work without awaiting it" in text


def test_a_phase_with_no_gpu_reading_is_not_given_a_row() -> None:
    """Absent is absent: a phase the sampler never caught must not read as 0% utilisation."""
    samples = [Sample(t=float(i), phase="cpu_only", rss=1000) for i in range(3)]

    assert analyse_processes([samples]).gpu_by_phase == {}


# --- CPU sampling and the resource inventory -------------------------------------------


def test_cpu_readings_become_core_equivalents() -> None:
    """psutil reports percent-of-one-core; the report speaks in cores."""
    samples = [Sample(t=float(i), phase="train", cpu_percent=percent)
               for i, percent in enumerate((100.0, 300.0, 200.0))]

    cpu = analyse_processes([samples]).cpu

    assert cpu.measured
    assert cpu.peak == pytest.approx(3.0)
    assert cpu.mean == pytest.approx(2.0)


def test_unmeasured_cpu_rows_are_skipped_rather_than_read_as_idle() -> None:
    """The -1.0 sentinel must not drag the mean down; absent is not zero."""
    measured = [Sample(t=float(i), phase="train", cpu_percent=200.0) for i in range(3)]
    with_gaps = [*measured, Sample(t=9.0, phase="train"), Sample(t=10.0, phase="train")]

    assert analyse_processes([with_gaps]).cpu.mean == pytest.approx(
        analyse_processes([measured]).cpu.mean,
    )


def test_a_run_with_no_cpu_readings_is_flagged_unmeasured() -> None:
    """Zero cores and no measurement must be distinguishable."""
    samples = [Sample(t=float(i), phase="train", rss=1000) for i in range(3)]

    cpu = analyse_processes([samples]).cpu

    assert not cpu.measured
    assert cpu.peak == 0.0


def test_an_idle_process_is_measured_at_zero_not_dropped() -> None:
    """0.0 is a real reading. Only -1.0 means the counter was never read."""
    samples = [Sample(t=float(i), phase="wait", cpu_percent=0.0) for i in range(3)]

    assert analyse_processes([samples]).cpu.measured


def test_cpu_is_pooled_per_process_and_keeps_the_heaviest() -> None:
    """Peaks sum across processes; max_process names the worst single one."""
    busy = [Sample(t=float(i), phase="train", cpu_percent=400.0) for i in range(3)]
    light = [Sample(t=float(i), phase="train", cpu_percent=100.0) for i in range(3)]

    cpu = analyse_processes([busy, light]).cpu

    assert cpu.peak == pytest.approx(5.0)
    assert cpu.max_process == pytest.approx(4.0)


def test_the_zero_cpu_sample_survives_the_compact_round_trip() -> None:
    """An idle reading must reach the file; only the sentinel is dropped."""
    assert _compact(Sample(t=1.0, phase="p", cpu_percent=0.0))["cpu_percent"] == 0.0
    assert "cpu_percent" not in _compact(Sample(t=1.0, phase="p"))


def test_peak_rss_records_the_heaviest_single_process() -> None:
    """The run total sums; this is what says whether one worker carries it."""
    fat = [Sample(t=float(i), phase="train", rss=900) for i in range(3)]
    thin = [Sample(t=float(i), phase="train", rss=100) for i in range(3)]

    analysis = analyse_processes([fat, thin])

    assert analysis.memory.peak_rss == 1000
    assert analysis.peak_rss_max_process == 900


def test_hardware_degrades_to_empty_without_psutil_or_nvml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inventory that cannot be read is absent, never a machine with no cores."""
    monkeypatch.setattr(hardware_module, "psutil_module", lambda: None)
    monkeypatch.setattr(hardware_module, "nvml_module", lambda: None)
    monkeypatch.setattr(os, "sched_getaffinity", None, raising=False)
    hardware_module.reset_cache()

    assert hardware_module.describe() == {}
    hardware_module.reset_cache()


def test_hardware_reports_cpu_when_only_psutil_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One missing capability costs one field, not the whole inventory."""
    fake = SimpleNamespace(
        cpu_count=lambda logical=True: 256 if logical else 128,
        virtual_memory=lambda: SimpleNamespace(total=64 * 1024**3),
    )
    monkeypatch.setattr(hardware_module, "psutil_module", lambda: fake)
    monkeypatch.setattr(hardware_module, "nvml_module", lambda: None)
    hardware_module.reset_cache()

    described = hardware_module.describe()
    hardware_module.reset_cache()

    assert described["cpu_cores"] == 128
    assert described["cpu_threads"] == 256
    assert described["ram_total"] == 64 * 1024**3
    assert "gpus" not in described


def test_a_device_name_arriving_as_bytes_is_decoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older nvidia-ml-py returns bytes; the page must not render b'...'."""
    nvml = SimpleNamespace(
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetName=lambda handle: b"NVIDIA A100-SXM4-80GB",
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(total=80 * 1024**3),
    )
    monkeypatch.setattr(hardware_module, "psutil_module", lambda: None)
    monkeypatch.setattr(hardware_module, "nvml_module", lambda: nvml)
    hardware_module.reset_cache()

    gpus = hardware_module.describe()["gpus"]
    hardware_module.reset_cache()

    assert gpus == [{"index": 0, "name": "NVIDIA A100-SXM4-80GB", "vram_total": 80 * 1024**3}]


def test_each_device_keeps_its_own_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """A closure over the loop variable used to give every device the last one's name."""
    names = {0: "A100", 1: "L40S"}
    nvml = SimpleNamespace(
        nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetName=lambda handle: names[handle],
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(total=1024),
    )
    monkeypatch.setattr(hardware_module, "psutil_module", lambda: None)
    monkeypatch.setattr(hardware_module, "nvml_module", lambda: nvml)
    hardware_module.reset_cache()

    gpus = hardware_module.describe()["gpus"]
    hardware_module.reset_cache()

    assert [gpu["name"] for gpu in gpus] == ["A100", "L40S"]


def test_a_device_that_cannot_be_addressed_does_not_lose_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handle_by_index(index: int) -> int:
        if index == 0:
            raise RuntimeError("device unavailable")
        return index

    nvml = SimpleNamespace(
        nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=handle_by_index,
        nvmlDeviceGetName=lambda handle: "L40S",
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(total=1024),
    )
    monkeypatch.setattr(hardware_module, "psutil_module", lambda: None)
    monkeypatch.setattr(hardware_module, "nvml_module", lambda: nvml)
    hardware_module.reset_cache()

    gpus = hardware_module.describe()["gpus"]
    hardware_module.reset_cache()

    assert [gpu["index"] for gpu in gpus] == [1]


def test_the_affinity_figure_is_named_only_when_it_differs() -> None:
    """On an unconstrained box the two are equal; printing both invites a false distinction."""
    constrained = hardware_module.format_capacity({"cpu_cores": 128, "cpu_affinity": 60})
    whole = hardware_module.format_capacity({"cpu_cores": 128, "cpu_affinity": 128})

    assert "60 available to this job" in constrained
    assert whole == "128 cores"


def test_mixed_device_models_are_listed_separately() -> None:
    gpus = [{"name": "A100"}, {"name": "A100"}, {"name": "L40S"}]

    assert hardware_module.format_gpu_models(gpus) == "2x A100, 1x L40S"


def _run_with_hardware(tmp_path: Path, per_host: list[tuple[str, dict[str, object]]]) -> object:
    """A merged run whose workers carry exactly the given per-host capacity.

    The hardware is overwritten rather than added to: these tests run on a real machine, whose
    own inventory would otherwise leak in and make the assertions depend on the test host.
    """
    run_id = new_run_id()
    for _ in range(max(len(per_host), 1)):
        _run_worker(tmp_path, "actor", {"work": 0.01}, run_id=run_id)
    run = merge_run(tmp_path)
    for worker in run.workers:
        worker.hardware = {}
    for worker, (host, hardware) in zip(run.workers, per_host, strict=False):
        worker.placement = {**worker.placement, "host": host}
        worker.hardware = hardware
    return run


def test_a_run_knowing_neither_capacity_nor_usage_omits_the_section(tmp_path: Path) -> None:
    """Nothing to say is said with silence, not with a header full of blanks."""
    run = _run_with_hardware(tmp_path, [])

    assert "RESOURCES" not in render(run)  # type: ignore[arg-type]


def test_capacity_alone_still_names_the_machine(tmp_path: Path) -> None:
    """Sampling off means no consumption rows, but the machine is still worth stating: it is
    what makes this run's timings comparable against another server's."""
    run = _run_with_hardware(tmp_path, [("node01", {"cpu_cores": 64})])

    text = render(run)  # type: ignore[arg-type]

    assert "RESOURCES" in text
    assert "node01: 64 cores" in text
    # No consumption was measured, so no denominator is offered for one.
    assert "per-proc figures" not in text


def test_the_report_names_each_host_and_its_capacity(tmp_path: Path) -> None:
    run = _run_with_hardware(
        tmp_path,
        [("node01", {"cpu_cores": 128, "cpu_affinity": 60, "ram_total": 2 * 1024**4})],
    )

    text = render(run)  # type: ignore[arg-type]

    assert "node01: 128 cores (60 available to this job)" in text


def test_two_hosts_get_a_line_each_rather_than_one_merged_figure(tmp_path: Path) -> None:
    """A fat node and a thin one have no single capacity; averaging would describe neither."""
    run = _run_with_hardware(
        tmp_path, [("big", {"cpu_cores": 128}), ("small", {"cpu_cores": 16})],
    )

    text = render(run)  # type: ignore[arg-type]

    assert "big: 128 cores" in text
    assert "small: 16 cores" in text
    # Pooled for the available column, which is the run's real ceiling across both nodes.
    assert pooled_capacity(run)["cpu_cores"] == 144  # type: ignore[arg-type]


def test_a_percentage_is_suppressed_when_capacity_is_missing() -> None:
    """Never divide by an absent field, and never substitute a default for one."""
    assert _percent_of(4.0, None) == ""
    assert _percent_of(4.0, 0) == ""
    assert "50%" in _percent_of(4.0, 8)


def test_the_machine_document_survives_json_dumps(tmp_path: Path) -> None:
    run = _run_with_hardware(tmp_path, [("node01", {"cpu_cores": 64})])

    document = report_as_dict(run)  # type: ignore[arg-type]

    assert document["machine"]["capacity_by_host"]["node01"]["cpu_cores"] == 64
    json.dumps(document)


def test_a_run_that_knows_nothing_omits_the_machine_key(tmp_path: Path) -> None:
    """An absent measurement is an absent key, not a block of zeros."""
    run = _run_with_hardware(tmp_path, [])

    assert "machine" not in report_as_dict(run)  # type: ignore[arg-type]
