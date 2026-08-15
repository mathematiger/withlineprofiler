"""Exhaustive tests for the lineprofiler package.

The sample functions below live inside this file, which is itself inside the
folder passed as ``project_folder``, so they are picked up by the profiler.
Standard-library and pytest internals live elsewhere and are filtered out.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lineprofiler import FunctionStats, LineProfiler, LineStats, start_profiling, stop_profiling
from lineprofiler.config import ENV_ENABLED, ProfilerConfig, get_config
from lineprofiler.profiler import _qualname_of

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


def test_two_different_profilers_may_nest(tmp_path: Path) -> None:
    """Only re-entering one instance is unsafe; distinct instances restore each other."""
    incumbent = sys.gettrace()
    outer = LineProfiler(project_folder=THIS_DIR)
    inner = LineProfiler(project_folder=THIS_DIR)

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
