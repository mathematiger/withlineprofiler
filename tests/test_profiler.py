"""Exhaustive tests for the lineprofiler package.

The sample functions below live inside this file, which is itself inside the
folder passed as ``project_folder``, so they are picked up by the profiler.
Standard-library and pytest internals live elsewhere and are filtered out.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import pytest

from lineprofiler import FunctionStats, LineProfiler, LineStats, start_profiling, stop_profiling
from lineprofiler import profiler as profiler_module
from lineprofiler.config import ENV_ENABLED, ProfilerConfig, get_config
from lineprofiler.profiler import _MONITORING, _TOOL_ID, _qualname_of

THIS_DIR = str(Path(__file__).resolve().parent)

# Every engine, and for the builtin one every event source the interpreter offers, so the
# settrace path — the only one 3.10 and 3.11 can use — keeps being exercised on a runner that
# defaults to the newer.
_MODES = (
    ("line_profiler", "builtin:monitoring", "builtin:settrace")
    if _MONITORING is not None
    else ("line_profiler", "builtin:settrace")
)


@pytest.fixture(params=_MODES, autouse=True)
def mode(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run every profiler test under each engine and backend.

    Patches the defaults rather than each construction site, so the thirty-odd
    ``LineProfiler(project_folder=THIS_DIR)`` calls in this file need no edit.
    """
    chosen: str = request.param
    engine, _, backend = chosen.partition(":")
    monkeypatch.setattr(profiler_module, "_default_engine", lambda: engine)
    if backend:
        monkeypatch.setattr(profiler_module, "_default_backend", lambda: backend)
    return chosen


@pytest.fixture
def backend(mode: str) -> str:
    """The builtin backend in force — the interpreter's default under the C engine."""
    return mode.partition(":")[2] or profiler_module._default_backend()


def _needs_monitoring_for_discovery(mode: str) -> None:
    if mode == "line_profiler" and _MONITORING is None:
        pytest.skip("below 3.12 the line_profiler engine registers loaded modules only")


# --------------------------------------------------------------------------- #
# Sample functions to profile
# --------------------------------------------------------------------------- #
def add(a: int, b: int) -> int:
    c = a + b
    d = c * 2
    return d


def loop_sum(n: int) -> int:
    total = 0
    for i in range(n):
        total += i
    return total


def inner() -> int:
    s = 0
    for i in range(50):
        s += i
    return s


def outer() -> int:
    a = 1
    b = inner()  # nested in-project call
    c = 2  # must still be timed after the nested call
    d = 3
    return a + b + c + d


def sleeper() -> int:
    x = 1
    time.sleep(0.05)  # this line should dominate the timing
    y = 2
    return x + y


def raiser() -> None:
    prepared = 1  # noqa: F841
    raise ValueError("expected by the unwind test")


def counting_up(n: int):  # type: ignore[no-untyped-def]  # noqa: ANN201 - a generator
    for i in range(n):  # noqa: UP028 - the test asserts on the `yield` line itself
        yield i  # must not be billed the consumer's work


def consume_slowly(n: int) -> int:
    seen = 0
    for _ in counting_up(n):
        time.sleep(0.01)
        seen += 1
    return seen


def line_source(func_stats: FunctionStats, needle: str) -> LineStats:
    """Return the LineStats of the recorded line whose source contains needle."""
    for line_num, stats in func_stats.line_stats.items():
        if needle in func_stats.source_lines.get(line_num, ""):
            return stats
    raise AssertionError(f"no recorded line containing {needle!r}")


def stats_for(profiler: LineProfiler, func_name: str) -> FunctionStats:
    for (_, name, _), func_stats in profiler.get_stats().items():
        if name == func_name:
            return func_stats
    raise AssertionError(f"no recorded function named {func_name!r}")


# --------------------------------------------------------------------------- #
# LineStats / FunctionStats dataclasses
# --------------------------------------------------------------------------- #
def test_line_stats_average_with_hits() -> None:
    stats = LineStats(line_number=1, hits=4, total_time=2.0)
    assert stats.average_time == 0.5


def test_line_stats_average_zero_hits() -> None:
    stats = LineStats(line_number=1)
    assert stats.average_time == 0.0


def test_function_stats_defaults() -> None:
    fs = FunctionStats(filename="f.py", function_name="g", first_line=10)
    assert fs.line_stats == {}
    assert fs.source_lines == {}
    assert fs.total_time == 0.0


# --------------------------------------------------------------------------- #
# Context-manager protocol
# --------------------------------------------------------------------------- #
def test_enter_returns_self() -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler as entered:
        assert entered is profiler


def test_trace_restored_after_exit() -> None:
    before = sys.gettrace()
    with LineProfiler(project_folder=THIS_DIR):
        add(1, 2)
    assert sys.gettrace() is before


def test_trace_restored_after_exception() -> None:
    before = sys.gettrace()
    with pytest.raises(ValueError, match="boom"), LineProfiler(project_folder=THIS_DIR):
        raise ValueError("boom")
    assert sys.gettrace() is before


def test_disabled_after_exit() -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        add(1, 2)
    assert profiler._enabled is False


# --------------------------------------------------------------------------- #
# Core profiling behaviour
# --------------------------------------------------------------------------- #
def test_simple_function_lines_recorded() -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        add(2, 3)

    fs = stats_for(profiler, "add")
    # three body lines, each hit exactly once
    assert len(fs.line_stats) == 3
    assert all(ls.hits == 1 for ls in fs.line_stats.values())
    assert fs.total_time >= 0.0


def test_loop_hit_counts() -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        loop_sum(10)

    fs = stats_for(profiler, "loop_sum")
    body = line_source(fs, "total += i")
    assert body.hits == 10


def test_nested_call_lines_after_call_are_timed() -> None:
    """Regression: lines after an in-project nested call must still be recorded."""
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        outer()

    outer_fs = stats_for(profiler, "outer")
    # Every body line of outer, including those after the inner() call.
    for needle in ("a = 1", "b = inner()", "c = 2", "d = 3", "return a + b"):
        assert line_source(outer_fs, needle).hits == 1

    # inner() is profiled as its own function.
    inner_fs = stats_for(profiler, "inner")
    assert line_source(inner_fs, "s += i").hits == 50


def test_function_key_structure() -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        add(1, 1)

    keys = [k for k in profiler.get_stats() if k[1] == "add"]
    assert len(keys) == 1
    filename, name, first_line = keys[0]
    assert filename.endswith("test_profiler.py")
    assert name == "add"
    assert first_line == add.__code__.co_firstlineno


# --------------------------------------------------------------------------- #
# Timing accuracy
# --------------------------------------------------------------------------- #
def test_sleep_line_dominates_timing() -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        sleeper()

    fs = stats_for(profiler, "sleeper")
    sleep_line = line_source(fs, "time.sleep")
    # The sleeping line must account for most of the ~50 ms.
    assert sleep_line.total_time >= 0.04
    slowest = max(fs.line_stats.values(), key=lambda ls: ls.total_time)
    assert slowest is sleep_line


def test_overhead_excluded_from_fast_lines() -> None:
    """Trivial lines should report tiny times, not inflated by profiler work."""
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        loop_sum(1000)

    fs = stats_for(profiler, "loop_sum")
    body = line_source(fs, "total += i")
    # 1000 cheap additions should still be well under a second in total.
    assert body.total_time < 0.5
    assert body.average_time < body.total_time or body.hits == 1


# --------------------------------------------------------------------------- #
# Project-folder filtering
# --------------------------------------------------------------------------- #
def test_only_project_files_recorded() -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        add(1, 2)
        loop_sum(3)

    for filename, _, _ in profiler.get_stats():
        assert filename.startswith(THIS_DIR)


def test_outside_project_folder_not_recorded(tmp_path: Path) -> None:
    profiler = LineProfiler(project_folder=str(tmp_path))
    with profiler:
        add(1, 2)  # defined under THIS_DIR, not under tmp_path
    assert profiler.get_stats() == {}


def test_is_in_project_folder_cached() -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    inside = str(Path(THIS_DIR) / "x.py")
    outside = "/usr/lib/python/os.py"
    assert profiler._is_in_project_folder(inside) is True
    assert profiler._is_in_project_folder(outside) is False
    assert profiler._project_cache == {inside: True, outside: False}


# --------------------------------------------------------------------------- #
# Resource usage: source is stored once per file
# --------------------------------------------------------------------------- #
def test_source_lines_shared_per_file() -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        outer()  # outer and inner live in the same file

    outer_fs = stats_for(profiler, "outer")
    inner_fs = stats_for(profiler, "inner")
    # Same file => exact same source dict object (no per-function duplication).
    assert outer_fs.source_lines is inner_fs.source_lines
    assert len(profiler._source_cache) == 1


def test_source_lines_content() -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        add(1, 2)

    fs = stats_for(profiler, "add")
    assert any("c = a + b" in line for line in fs.source_lines.values())


# --------------------------------------------------------------------------- #
# clear / reset
# --------------------------------------------------------------------------- #
def test_clear_resets_state() -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        add(1, 2)
    assert profiler.get_stats()

    profiler.clear()
    assert profiler.get_stats() == {}
    assert profiler._source_cache == {}
    assert profiler._project_cache == {}
    assert profiler._frames.entries == []


def test_reset_is_clear() -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        add(1, 2)
    profiler.reset()
    assert profiler.get_stats() == {}


def test_reprofile_after_clear() -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        add(1, 2)
    profiler.clear()
    with profiler:
        loop_sum(5)

    assert stats_for(profiler, "loop_sum")
    assert [k for k in profiler.get_stats() if k[1] == "add"] == []


# --------------------------------------------------------------------------- #
# Auto-detection of the project folder
# --------------------------------------------------------------------------- #
def test_autodetect_project_folder_is_repo_root() -> None:
    """Autodetection walks up to the repo root, and falls back to a *directory* without one.

    Both halves matter and the environment decides which applies: run from a git checkout the
    first holds, run from an unpacked sdist (as a downstream packager does) only the second
    does. Asserting the repo root unconditionally passed for the wrong reason — it happened to
    be true wherever the suite was usually run — and hid a fallback that returned a file path.
    """
    profiler = LineProfiler()
    detected = profiler._project_folder

    assert detected.is_dir(), "the fallback must never be a file path"
    if (Path(THIS_DIR).parent / ".git").exists():
        assert (detected / "lineprofiler").is_dir()
    else:
        assert detected == Path(THIS_DIR).resolve()


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def test_print_stats_empty(capsys: pytest.CaptureFixture[str]) -> None:
    LineProfiler(project_folder=THIS_DIR).print_stats()
    assert "No profiling data collected." in capsys.readouterr().out


def test_print_stats_with_data(capsys: pytest.CaptureFixture[str]) -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        loop_sum(20)
    profiler.print_stats()
    out = capsys.readouterr().out
    assert "loop_sum" in out
    assert "Line #" in out


def test_print_stats_sort_and_limit(capsys: pytest.CaptureFixture[str]) -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        loop_sum(20)
    profiler.print_stats(top_n_lines=1, sort_by="hits")
    # Should not raise and should print the table header.
    assert "Hits" in capsys.readouterr().out


def test_print_global_top_stats(capsys: pytest.CaptureFixture[str]) -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        outer()
        loop_sum(10)
    profiler.print_global_top_stats(top_n=5)
    out = capsys.readouterr().out
    assert "Top 5 lines across all functions" in out


def test_print_global_top_stats_empty(capsys: pytest.CaptureFixture[str]) -> None:
    LineProfiler(project_folder=THIS_DIR).print_global_top_stats()
    assert "No profiling data above the threshold." in capsys.readouterr().out


# ── defects that used to corrupt the process, not just the numbers ──────────


def test_nesting_the_same_profiler_is_refused(tmp_path: Path) -> None:
    """It used to leave the trace function installed for the lifetime of the process.

    The inner __enter__ saved the profiler's own callback as the tracer to restore, so the
    outer __exit__ reinstalled it. _enabled was False, so it early-returned on every event —
    but a global trace function was still dispatched on every Python call, forever.
    """
    profiler = LineProfiler(project_folder=THIS_DIR)

    incumbent = sys.gettrace()

    # The nesting is the subject of the test, so it is written out rather than combined.
    with profiler:  # noqa: SIM117
        with pytest.raises(RuntimeError, match="already active"), profiler:
            pass

    assert sys.gettrace() is incumbent, "the profiler leaked its tracer"


def test_two_different_profilers_nest_or_are_refused(mode: str) -> None:
    """Distinct instances chain under ``settrace`` and are refused under builtin ``monitoring``.

    The backends genuinely differ here and neither behaviour is a bug. ``sys.settrace``
    tracers chain, so the inner profiler restores the outer one on exit. The builtin
    ``sys.monitoring`` engine has one profiler slot, so the inner claim is refused — which is
    the better outcome of the two: nesting double-counts every line either way, and the
    refusal says so instead of quietly returning inflated numbers. ``line_profiler`` keeps
    per-instance state and nests, and its discovery slot is refused instead.
    """
    incumbent = sys.gettrace()
    outer = LineProfiler(project_folder=THIS_DIR)
    inner = LineProfiler(project_folder=THIS_DIR)

    if mode == "builtin:monitoring":
        with outer, pytest.raises(RuntimeError, match="profiler slot"):
            inner.__enter__()
    elif mode == "line_profiler" and _MONITORING is not None:
        with outer, pytest.raises(RuntimeError, match="slot 4"):
            inner.__enter__()
    else:
        with outer, inner:
            add(1, 2)

    assert sys.gettrace() is incumbent


def test_the_tracer_is_cleared_after_an_exception(tmp_path: Path) -> None:
    incumbent = sys.gettrace()
    profiler = LineProfiler(project_folder=THIS_DIR)

    with pytest.raises(ValueError, match="expected"), profiler:
        raise ValueError("expected")

    assert sys.gettrace() is incumbent
    assert profiler._enabled is False


def test_repo_root_falls_back_to_a_directory_not_a_file(tmp_path: Path) -> None:
    """Outside a git checkout the fallback returned the *file*, so relative_to matched only
    that one module and the profiler silently profiled a single file."""
    module = tmp_path / "sub" / "mod.py"
    module.parent.mkdir(parents=True)
    module.write_text("x = 1", encoding="utf-8")
    profiler = LineProfiler(project_folder=THIS_DIR)

    root = profiler._find_repo_root(str(module))

    assert root.is_dir()
    assert root == module.parent


def test_repo_root_still_finds_a_git_directory(tmp_path: Path) -> None:
    module = tmp_path / "pkg" / "mod.py"
    module.parent.mkdir(parents=True)
    module.write_text("x = 1", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    profiler = LineProfiler(project_folder=THIS_DIR)

    assert profiler._find_repo_root(str(module)) == tmp_path.resolve()


# --------------------------------------------------------------------------- #
# Backend selection and the monitoring event source
# --------------------------------------------------------------------------- #
def test_backend_defaults_to_what_the_interpreter_offers(backend: str) -> None:
    assert LineProfiler(project_folder=THIS_DIR).backend == backend


@pytest.mark.skipif(_MONITORING is not None, reason="needs an interpreter without monitoring")
def test_monitoring_backend_is_refused_below_312() -> None:
    with pytest.raises(ValueError, match="3.12 or newer"):
        LineProfiler(project_folder=THIS_DIR, backend="monitoring")


@pytest.mark.skipif(_MONITORING is None, reason="needs sys.monitoring")
@pytest.mark.parametrize("backend", ["monitoring"])
def test_monitoring_frees_the_tool_slot_on_exit(backend: str) -> None:
    """A slot left claimed would refuse every later profiler in the process."""
    assert _MONITORING is not None  # narrowed for the type checker; the skipif guarantees it
    with LineProfiler(project_folder=THIS_DIR, backend="monitoring"):
        add(1, 2)

    assert _MONITORING.get_tool(_TOOL_ID) is None


@pytest.mark.skipif(_MONITORING is None, reason="needs sys.monitoring")
@pytest.mark.parametrize("backend", ["monitoring"])
def test_a_second_session_still_records_a_previously_filtered_function(backend: str) -> None:
    """Regression: ``DISABLE`` outlives the session that returned it.

    ``sys.monitoring`` opt-outs are permanent for the code object until
    ``restart_events()``, and they are *not* cleared by re-registering callbacks. So a
    profiler that filtered ``add`` out in one session used to leave it filtered for every
    later session in the same process — reporting a confident zero for a function that ran.
    Delete the ``restart_events()`` call in ``_enable_monitoring`` and this test fails.
    """
    filtered = ProfilerConfig(enabled=True, functions=("loop_sum",))
    with LineProfiler(project_folder=THIS_DIR, config=filtered, backend="monitoring"):
        add(1, 2)

    second = LineProfiler(project_folder=THIS_DIR, backend="monitoring")
    with second:
        add(1, 2)

    assert stats_for(second, "add"), "the first session's opt-out leaked into the second"


@pytest.mark.skipif(_MONITORING is None, reason="needs sys.monitoring")
@pytest.mark.parametrize("backend", ["monitoring"])
def test_monitoring_profiles_the_with_block_body(backend: str) -> None:
    """What ``sys.settrace`` structurally cannot do, and the reason for the new backend.

    ``sys.settrace`` only affects frames created after it is installed, so the block's own
    frame is never traced. ``sys.monitoring`` has no such restriction.
    """
    profiler = LineProfiler(project_folder=THIS_DIR, backend="monitoring")
    with profiler:
        marker_in_the_block = sum(range(3))  # noqa: F841

    body = stats_for(profiler, "test_monitoring_profiles_the_with_block_body")
    assert line_source(body, "marker_in_the_block").hits == 1


def test_an_exceptional_exit_does_not_bleed_into_the_next_line(backend: str) -> None:
    """The unwind must close the raising frame, not leave its time on the caller's clock.

    Under ``monitoring`` this needs ``PY_UNWIND`` registered alongside ``PY_RETURN``:
    ``settrace``'s ``return`` event covers both exits, but ``PY_RETURN`` fires only for a
    normal one, so without it the time spent raising is billed to whatever ran next.
    """
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        for _ in range(5):
            with contextlib.suppress(ValueError):
                raiser()

    fs = stats_for(profiler, "raiser")
    assert line_source(fs, "raise ValueError").hits == 5


# --------------------------------------------------------------------------- #
# The timing model: inclusive call lines, per-thread frames, generators
# --------------------------------------------------------------------------- #
def test_a_call_line_is_billed_the_whole_call() -> None:
    """The line ``b = inner()`` costs what ``inner`` costs — the convention every line profiler
    uses, and the one the old engine broke by resetting the caller's clock on the call.
    """
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        outer()

    call_line = line_source(stats_for(profiler, "outer"), "b = inner()")
    assert call_line.total_time >= stats_for(profiler, "inner").total_time * 0.9


def test_threads_started_inside_the_block_are_profiled_exactly() -> None:
    """Four threads, every hit counted once. One shared "last line" used to race here."""
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        threads = [threading.Thread(target=loop_sum, args=(500,)) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert line_source(stats_for(profiler, "loop_sum"), "total += i").hits == 2000


def test_a_yield_line_is_not_billed_the_consumers_time() -> None:
    """A suspended generator is not running; its ``yield`` must not absorb the consumer's sleep."""
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        consume_slowly(3)

    yield_line = line_source(stats_for(profiler, "counting_up"), "yield i")
    assert yield_line.hits == 3
    assert yield_line.total_time < 0.005, "three 10 ms sleeps landed on the yield line"


def test_functions_defined_inside_a_function_are_profiled(mode: str) -> None:
    """A closure is not an attribute of any module, so the C engine can only see it when it
    runs; the discovery hook is what makes ``with`` cover it too."""
    _needs_monitoring_for_discovery(mode)
    profiler = LineProfiler(project_folder=THIS_DIR)

    def local_product(n: int) -> int:
        product = 1
        for i in range(1, n + 1):
            product *= i
        return product

    with profiler:
        local_product(6)

    assert line_source(stats_for(profiler, "local_product"), "product *= i").hits == 6


def test_a_module_imported_inside_the_block_is_profiled(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A function that did not exist when profiling started is still profiled on 3.12+."""
    _needs_monitoring_for_discovery(mode)
    (tmp_path / "late_module.py").write_text(
        "def late(n):\n    total = 0\n    for i in range(n):\n        total += i\n"
        "    return total\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "late_module", raising=False)
    profiler = LineProfiler(project_folder=str(tmp_path))

    with profiler:
        late = importlib.import_module("late_module")
        late.late(40)

    assert line_source(stats_for(profiler, "late"), "total += i").hits == 40


# --------------------------------------------------------------------------- #
# Regions: per-line statistics partitioned by a named block
# --------------------------------------------------------------------------- #
def region_total(profiler: LineProfiler, name: str) -> float:
    return sum(f.total_time for f in profiler.region_stats()[name].values())


def test_regions_split_a_run_in_the_proportion_it_was_spent() -> None:
    """Two regions around known amounts of work must report that ratio, not a guess."""
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        for _ in range(3):
            with profiler.region("slow"):
                sleeper()  # ~50 ms
            with profiler.region("quick"):
                loop_sum(50)

    slow, quick = region_total(profiler, "slow"), region_total(profiler, "quick")
    assert slow > quick * 20, f"slow {slow:.4f}s should dominate quick {quick:.4f}s"
    assert profiler.region_entries() == {"slow": 3, "quick": 3}


def test_a_region_records_only_the_lines_run_inside_it() -> None:
    """What ran inside is billed to the region; what ran after it closed is not.

    The enclosing frame's own lines *are* included while the region is open — a region is the
    window it brackets, and the caller is running inside that window too.
    """
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:  # noqa: SIM117 - the nesting is the subject of the test
        with profiler.region("only_add"):
            add(1, 2)
        loop_sum(20)

    inside = {name for _, name, _ in profiler.region_stats()["only_add"]}
    assert "add" in inside
    assert "loop_sum" not in inside, "a function called after the region closed was billed to it"
    assert {name for _, name, _ in profiler.get_stats()} >= {"add", "loop_sum"}


def test_regions_nest_and_the_outer_one_includes_the_inner() -> None:
    """The same inclusive reading as a phase's wall time in the accounting layer."""
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:  # noqa: SIM117 - the nesting is the subject of the test
        with profiler.region("outer"):
            loop_sum(200)
            with profiler.region("inner"):
                sleeper()  # ~50 ms, and must show up in both

    assert region_total(profiler, "inner") >= 0.04
    assert region_total(profiler, "outer") >= region_total(profiler, "inner")


def test_a_region_left_by_an_exception_still_closes() -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        with contextlib.suppress(ValueError), profiler.region("raises"):
            raiser()
        add(1, 2)

    assert profiler._region_stack == []
    assert "raiser" in {name for _, name, _ in profiler.region_stats()["raises"]}


def test_a_region_entered_outside_the_block_records_nothing() -> None:
    """The calls are safe to leave in code that is not being profiled."""
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler.region("not_profiling"):
        add(1, 2)

    assert profiler.region_stats() == {}
    assert profiler.region_entries() == {}


def test_clear_forgets_regions() -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:  # noqa: SIM117 - the nesting is the subject of the test
        with profiler.region("gone"):
            add(1, 2)
    profiler.clear()

    assert profiler.region_stats() == {}
    assert profiler.region_entries() == {}


def test_print_regions_reports_each_region_and_its_share(
    capsys: pytest.CaptureFixture[str],
) -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:  # noqa: SIM117 - the nesting is the subject of the test
        with profiler.region("counting"):
            loop_sum(60)

    # top_n is generous because the caller's own lines are inside the region too, and they
    # are billed inclusively, so they outrank the loop body they are waiting on.
    profiler.print_regions(top_n=10)
    out = capsys.readouterr().out
    assert "Region: counting" in out
    assert "entries" in out
    assert "total += i" in out
    assert "need not sum to 100%" in out


def test_print_regions_says_so_when_none_were_used(
    capsys: pytest.CaptureFixture[str],
) -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        add(1, 2)
    profiler.print_regions()

    assert "No regions recorded." in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Engine selection and interoperability with line_profiler
# --------------------------------------------------------------------------- #
def test_the_engine_defaults_to_line_profiler_where_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()  # the mode fixture pins the default; this test asks what it really is
    assert LineProfiler(project_folder=THIS_DIR).engine == "line_profiler"


def test_passing_a_backend_selects_the_builtin_engine(backend: str) -> None:
    chosen = cast('Literal["monitoring", "settrace"]', backend)
    assert LineProfiler(project_folder=THIS_DIR, backend=chosen).engine == "builtin"


@pytest.mark.skipif(_MONITORING is None, reason="needs sys.monitoring")
def test_the_c_engine_frees_both_tool_slots_on_exit() -> None:
    assert _MONITORING is not None
    with LineProfiler(project_folder=THIS_DIR, engine="line_profiler"):
        add(1, 2)

    assert _MONITORING.get_tool(2) is None
    assert _MONITORING.get_tool(4) is None


def test_dump_stats_is_readable_by_line_profiler(tmp_path: Path) -> None:
    """``python -m line_profiler run.lprof`` and ``LineStats.from_files`` read this output."""
    import line_profiler

    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        loop_sum(7)
    target = tmp_path / "run.lprof"
    profiler.dump_stats(target)

    loaded = line_profiler.load_stats(str(target))
    key = next(k for k in loaded.timings if k[2] == "loop_sum")
    assert key[0].endswith("test_profiler.py")
    assert any(hits == 7 for _, hits, _ in loaded.timings[key])


def test_print_stats_writes_to_a_stream_in_source_order() -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        loop_sum(3)
    out = io.StringIO()
    profiler.print_stats(stream=out)

    rows = [line for line in out.getvalue().splitlines() if line[:1].isdigit()]
    numbers = [int(row.split()[0]) for row in rows]
    assert numbers == sorted(numbers)


# --------------------------------------------------------------------------- #
# Interpreter-version fallbacks
# --------------------------------------------------------------------------- #
def test_qualname_of_uses_co_qualname_where_available() -> None:
    """On 3.11+ a method is matched by its dotted name, which is what the globs expect."""

    class Holder:
        def step(self) -> None:
            pass

    if sys.version_info >= (3, 11):
        assert _qualname_of(Holder.step.__code__).endswith("Holder.step")
    else:  # pragma: no cover - the 3.10 branch, covered by the stub test below
        assert _qualname_of(Holder.step.__code__) == "step"


def test_qualname_of_falls_back_to_co_name_without_co_qualname() -> None:
    """The 3.10 branch, exercised on every version so it cannot rot untested.

    ``co_qualname`` arrived in 3.11. A stub stands in for a 3.10 code object because the
    dev environment and CI's default interpreter both have the attribute, so the fallback
    would otherwise only ever run on one matrix entry.
    """
    stub = SimpleNamespace(co_name="step")

    assert _qualname_of(stub) == "step"  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Config-driven include/exclude/function filtering
# --------------------------------------------------------------------------- #
def test_config_include_excludes_files_outside_the_glob() -> None:
    config = ProfilerConfig(enabled=True, include=("does_not_match_*.py",))
    profiler = LineProfiler(project_folder=THIS_DIR, config=config)
    with profiler:
        add(1, 2)
    assert profiler.get_stats() == {}


def test_config_include_matches_this_file() -> None:
    config = ProfilerConfig(enabled=True, include=("test_profiler.py",))
    profiler = LineProfiler(project_folder=THIS_DIR, config=config)
    with profiler:
        add(1, 2)
    assert stats_for(profiler, "add")


def test_config_exclude_wins_over_include() -> None:
    config = ProfilerConfig(
        enabled=True,
        include=("test_profiler.py",),
        exclude=("test_profiler.py",),
    )
    profiler = LineProfiler(project_folder=THIS_DIR, config=config)
    with profiler:
        add(1, 2)
    assert profiler.get_stats() == {}


def test_config_functions_filters_by_qualname() -> None:
    config = ProfilerConfig(enabled=True, functions=("loop_sum",))
    profiler = LineProfiler(project_folder=THIS_DIR, config=config)
    with profiler:
        add(1, 2)
        loop_sum(3)
    assert [k for k in profiler.get_stats() if k[1] == "add"] == []
    assert stats_for(profiler, "loop_sum")


def test_config_functions_glob_matches_qualname_prefix() -> None:
    config = ProfilerConfig(enabled=True, functions=("add*",))
    profiler = LineProfiler(project_folder=THIS_DIR, config=config)
    with profiler:
        add(1, 2)
    assert stats_for(profiler, "add")


# --------------------------------------------------------------------------- #
# start_profiling() / stop_profiling()
# --------------------------------------------------------------------------- #
def test_start_profiling_is_a_no_op_when_not_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    before = sys.gettrace()

    profiler = start_profiling(project_folder=THIS_DIR)
    add(1, 2)
    result = stop_profiling(print_stats=False)

    assert sys.gettrace() is before
    assert profiler.get_stats() == {}
    assert result is None


def test_start_profiling_traces_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_ENABLED, "1")

    start_profiling(project_folder=THIS_DIR)
    add(1, 2)
    profiler = stop_profiling(print_stats=False)

    assert profiler is not None
    assert stats_for(profiler, "add")


def test_stop_profiling_restores_the_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_ENABLED, "1")
    before = sys.gettrace()

    start_profiling(project_folder=THIS_DIR)
    add(1, 2)
    stop_profiling(print_stats=False)

    assert sys.gettrace() is before


def test_stop_profiling_without_start_is_a_no_op() -> None:
    assert stop_profiling(print_stats=False) is None


def test_double_start_profiling_warns_and_returns_the_running_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_ENABLED, "1")

    first = start_profiling(project_folder=THIS_DIR)
    with pytest.warns(RuntimeWarning, match="already called"):
        second = start_profiling(project_folder=THIS_DIR)
    stop_profiling(print_stats=False)

    assert second is first


def test_start_profiling_enabled_true_works_without_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_ENABLED, raising=False)

    start_profiling(project_folder=THIS_DIR, enabled=True)
    add(1, 2)
    profiler = stop_profiling(print_stats=False)

    assert profiler is not None
    assert stats_for(profiler, "add")


def test_start_profiling_enabled_false_wins_over_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_ENABLED, "1")

    profiler = start_profiling(project_folder=THIS_DIR, enabled=False)
    add(1, 2)

    assert stop_profiling(print_stats=False) is None
    assert profiler.get_stats() == {}


def test_lineprofiler_run_profiles_a_script_without_editing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ``kernprof`` equivalent: a script, no decorators, its own exit status kept."""
    from lineprofiler.accounting.cli import main

    (tmp_path / ".git").mkdir()
    script = tmp_path / "job.py"
    script.write_text(
        "import sys\n"
        "def busy(n):\n    total = 0\n    for i in range(n):\n        total += i\n"
        "    return total\n"
        "busy(int(sys.argv[1]))\nsys.exit(3)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.setattr(sys, "path", list(sys.path))

    status = main(["run", str(script), "12", "--top", "20"])

    out = capsys.readouterr().out
    assert status == 3
    assert "job.py::busy" in out
    assert "total += i" in out


def test_start_profiling_respects_config_include(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_ENABLED, "1")

    start_profiling(project_folder=THIS_DIR)
    add(1, 2)
    profiler = stop_profiling(print_stats=False)

    assert profiler is not None
    for filename, _, _ in profiler.get_stats():
        assert filename.startswith(THIS_DIR)


def test_get_config_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    config = get_config(THIS_DIR)
    assert config.enabled is False
    assert config.include == ()
    assert config.exclude == ()
    assert config.functions == ()


def test_get_config_enabled_via_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_ENABLED, "1")
    assert get_config(THIS_DIR).enabled is True


def test_get_config_falsy_values_are_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("0", "false", "False"):
        monkeypatch.setenv(ENV_ENABLED, value)
        assert get_config(THIS_DIR).enabled is False


def test_get_config_reads_pyproject_table(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[tool.lineprofiler]\ninclude = ["src/**"]\nexclude = ["src/generated/**"]\n'
        'functions = ["*.train_step"]\n',
        encoding="utf-8",
    )
    config = get_config(tmp_path)
    assert config.include == ("src/**",)
    assert config.exclude == ("src/generated/**",)
    assert config.functions == ("*.train_step",)


def test_get_config_missing_pyproject_degrades_to_defaults(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    config = get_config(tmp_path)
    assert config.include == ()
    assert config.exclude == ()


def test_get_config_malformed_pyproject_degrades_to_defaults(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text("this is not [ valid toml", encoding="utf-8")
    config = get_config(tmp_path)
    assert config.include == ()


def test_get_config_is_cached_per_project_root(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[tool.lineprofiler]\ninclude = ["a"]\n', encoding="utf-8",
    )
    first = get_config(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.lineprofiler]\ninclude = ["b"]\n', encoding="utf-8",
    )
    second = get_config(tmp_path)
    assert second is first
    assert second.include == ("a",)
