"""Exhaustive tests for the lineprofiler package.

The sample functions below live inside this file, which is itself inside the
folder passed as ``project_folder``, so they are picked up by the profiler.
Standard-library and pytest internals live elsewhere and are filtered out.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from lineprofiler import FunctionStats, LineProfiler, LineStats

THIS_DIR = str(Path(__file__).resolve().parent)


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
    assert profiler._last_key is None
    assert profiler._last_line is None


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
    profiler = LineProfiler()
    # The repo root contains the lineprofiler package and a .git directory.
    assert (profiler._project_folder / "lineprofiler").is_dir()


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
