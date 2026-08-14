"""Phase 3: multiprocessing.

This is where profilers usually break. The matrix covers every start method the platform
offers against several worker counts, plus the failure modes that matter in a long run: a
worker that raises, and a worker that is ``SIGKILL``ed before it ever writes.

Worker functions are module-level because ``spawn`` and ``forkserver`` pickle them by
qualified name.
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import signal
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from lineprofiler import accounting
from lineprofiler.accounting import Profiler, merge_run
from lineprofiler.accounting.profiler import ENV_ENABLE, ENV_RUN_DIR

START_METHODS = ["spawn", "fork", "forkserver"]
UNITS_PER_WORKER = 7


def _context(method: str) -> multiprocessing.context.DefaultContext:
    """Return the requested start-method context, skipping if unavailable here."""
    if method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"start method {method!r} is unavailable on this platform")
    context: multiprocessing.context.DefaultContext = multiprocessing.get_context(method)  # type: ignore[assignment]
    return context


def worker_records_units(run_dir: str, rank: int) -> None:
    """A worker doing a known, countable amount of work."""
    profiler = Profiler(
        run_dir=run_dir,
        role="actor",
        enabled=True,
        snapshot_interval_s=None,
        sample_interval_s=None,
    )
    for _ in range(UNITS_PER_WORKER):
        with profiler.phase("iteration"), profiler.phase("step"):
            profiler.count("units", 1)
            time.sleep(0.001)
    profiler.close()


def worker_from_environment(_run_dir: str, rank: int) -> None:
    """A worker configured entirely by the environment its parent exported."""
    profiler = Profiler(role="actor")
    with profiler.phase("inherited"):
        profiler.count("units", 1)
    profiler.close()


def worker_uses_the_ambient_profiler(run_dir: str, rank: int) -> None:
    """A worker instrumented only through the module-level functions.

    Under ``fork`` this is the case that goes wrong silently: the child inherits the parent's
    installed profiler, and would write into the parent's worker file if the fork handlers did
    not re-point it at the child's own.
    """
    profiler = Profiler(
        run_dir=run_dir, role="actor", enabled=True, install=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    for _ in range(UNITS_PER_WORKER):
        with accounting.phase("iteration"), accounting.phase("step"):
            accounting.count("units", 1)
    assert accounting.installed_profiler() is profiler
    profiler.close()


def worker_raises(run_dir: str, rank: int) -> None:
    """A worker that records some work and then dies with an exception."""
    profiler = Profiler(
        run_dir=run_dir, role="doomed", enabled=True, snapshot_interval_s=None,
        sample_interval_s=None,
    )
    with profiler.phase("before_crash"):
        profiler.count("units", 1)
    profiler.snapshot()
    raise RuntimeError("worker failed on purpose")


def worker_is_killed(run_dir: str, rank: int) -> None:
    """A worker that never gets to flush: SIGKILL cannot be caught."""
    profiler = Profiler(
        run_dir=run_dir, role="killed", enabled=True, snapshot_interval_s=None,
        sample_interval_s=None,
    )
    with profiler.phase("doomed"):
        profiler.count("units", 1)
        os.kill(os.getpid(), signal.SIGKILL)


def _run_workers(
    method: str,
    run_dir: Path,
    count: int,
    target: Callable[[str, int], None] = worker_records_units,
) -> list[int | None]:
    """Start ``count`` workers, join them, and return their exit codes."""
    context = _context(method)
    processes = [
        context.Process(target=target, args=(str(run_dir), rank)) for rank in range(count)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
    return [process.exitcode for process in processes]


# ── the matrix ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("workers", [1, 4, 16])
@pytest.mark.parametrize("method", START_METHODS)
def test_every_worker_is_recorded_exactly_once(method: str, workers: int, tmp_path: Path) -> None:
    exit_codes = _run_workers(method, tmp_path, workers)
    assert exit_codes == [0] * workers

    run = merge_run(tmp_path)

    assert len(run.workers) == workers, "one file per worker, no collisions"
    assert len({w.path for w in run.workers}) == workers, "file names must be unique"
    assert run.unreadable == []
    assert run.tree[("iteration", "step")].counters == {"units": workers * UNITS_PER_WORKER}
    assert run.tree[("iteration",)].calls == workers * UNITS_PER_WORKER


@pytest.mark.parametrize("method", START_METHODS)
def test_worker_files_are_unique_even_when_pids_repeat(method: str, tmp_path: Path) -> None:
    """Two waves of workers can reuse pids; the uuid is what keeps their files apart."""
    _run_workers(method, tmp_path, 4)
    _run_workers(method, tmp_path, 4)

    run = merge_run(tmp_path)

    assert len(run.workers) == 8
    assert run.tree[("iteration", "step")].counters == {"units": 8 * UNITS_PER_WORKER}


# ── the parent's own output ─────────────────────────────────────────────────


@pytest.mark.parametrize("method", START_METHODS)
def test_children_never_overwrite_the_parents_file(method: str, tmp_path: Path) -> None:
    parent = Profiler(
        run_dir=tmp_path, role="learner", enabled=True, snapshot_interval_s=None,
        sample_interval_s=None,
    )
    with parent.phase("parent_only"):
        parent.count("parent_units", 3)
    parent.snapshot()
    parent_path = parent._writer.path if parent._writer else None  # noqa: SLF001

    _run_workers(method, tmp_path, 4)
    parent.close()

    run = merge_run(tmp_path)
    assert parent_path is not None
    assert parent_path.exists()
    assert run.tree[("parent_only",)].counters == {"parent_units": 3}
    assert set(run.roles) == {"learner", "actor"}
    assert len(run.workers_of("learner")) == 1
    assert len(run.workers_of("actor")) == 4


@pytest.mark.filterwarnings("ignore:This process .*is multi-threaded")
def test_forked_child_starts_with_an_empty_tree(tmp_path: Path) -> None:
    """A fork copies the parent's accumulated phases; the child must not re-report them."""
    _context("fork")
    parent = Profiler(
        run_dir=tmp_path, role="parent", enabled=True, snapshot_interval_s=None,
        sample_interval_s=None,
    )
    with parent.phase("parent_work"):
        parent.count("parent_units", 5)

    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - runs only in the forked child
        try:
            with parent.phase("child_work"):
                parent.count("child_units", 1)
            parent.close()
        finally:
            os._exit(0)

    os.waitpid(child_pid, 0)
    parent.close()

    run = merge_run(tmp_path)
    child = next(w for w in run.workers if w.pid == child_pid)

    assert ("parent_work",) not in child.tree, "the child inherited the parent's phases"
    assert ("child_work",) in child.tree
    assert run.tree[("parent_work",)].counters == {"parent_units": 5}
    assert run.tree[("child_work",)].counters == {"child_units": 1}


@pytest.mark.filterwarnings("ignore:This process .*is multi-threaded")
def test_fork_inside_a_phase_gives_the_child_a_clean_stack(tmp_path: Path) -> None:
    _context("fork")
    parent = Profiler(
        run_dir=tmp_path, role="parent", enabled=True, snapshot_interval_s=None,
        sample_interval_s=None,
    )
    with parent.phase("outer"):
        child_pid = os.fork()
        if child_pid == 0:  # pragma: no cover - runs only in the forked child
            try:
                with parent.phase("child_root"):
                    pass
                parent.close()
            finally:
                os._exit(0)
        os.waitpid(child_pid, 0)
    parent.close()

    run = merge_run(tmp_path)
    child = next(w for w in run.workers if w.pid == child_pid)

    assert ("child_root",) in child.tree, "the child's phase must start at the root"
    assert ("outer", "child_root") not in child.tree


@pytest.mark.filterwarnings("ignore:This process .*is multi-threaded")
def test_forked_child_restarts_its_sampler(tmp_path: Path) -> None:
    """Threads do not survive fork, so the child's sampler must be started again."""
    _context("fork")
    parent = Profiler(
        run_dir=tmp_path, role="parent", enabled=True, snapshot_interval_s=None,
        sample_interval_s=0.05,
    )
    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - runs only in the forked child
        try:
            with parent.phase("child_work"):
                time.sleep(0.3)
            parent.close()
        finally:
            os._exit(0)

    os.waitpid(child_pid, 0)
    parent.close()

    run = merge_run(tmp_path)
    child = next(w for w in run.workers if w.pid == child_pid)
    assert child.samples, "the forked child produced no resource samples"


# ── environment propagation ─────────────────────────────────────────────────


def test_spawned_children_enable_themselves_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_ENABLE, raising=False)
    monkeypatch.delenv(ENV_RUN_DIR, raising=False)

    parent = Profiler(
        run_dir=tmp_path, role="learner", enabled=True, snapshot_interval_s=None,
        sample_interval_s=None,
    )
    assert os.environ[ENV_ENABLE] == "1"
    assert os.environ[ENV_RUN_DIR] == str(tmp_path)

    _run_workers("spawn", tmp_path, 3, target=worker_from_environment)
    parent.close()

    run = merge_run(tmp_path)
    assert run.tree[("inherited",)].counters == {"units": 3}
    assert len(run.workers_of("actor")) == 3


def test_forkserver_children_need_the_environment_set_before_the_daemon_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Documents a real ``forkserver`` constraint, not a defect in this package.

    The forkserver daemon is a long-lived process forked once, and its children inherit the
    daemon's environment — a snapshot taken when the daemon started. Variables exported
    afterwards never reach them. Under ``forkserver``, either export
    ``LINEPROFILER_PROFILE`` in the shell before training starts, or pass ``enabled`` and
    ``run_dir`` to the worker's ``Profiler`` explicitly.
    """
    context = _context("forkserver")
    warm_up = context.Process(target=worker_records_units, args=(str(tmp_path / "warmup"), 0))
    warm_up.start()
    warm_up.join(timeout=60)

    monkeypatch.delenv(ENV_ENABLE, raising=False)
    monkeypatch.delenv(ENV_RUN_DIR, raising=False)
    parent = Profiler(
        run_dir=tmp_path, role="learner", enabled=True, snapshot_interval_s=None,
        sample_interval_s=None,
    )
    _run_workers("forkserver", tmp_path, 2, target=worker_from_environment)
    parent.close()

    run = merge_run(tmp_path)
    assert ("inherited",) not in run.tree, (
        "if this now passes, forkserver has started propagating late environment changes "
        "and the documented workaround can be removed"
    )

    # Passing the configuration explicitly always works, whatever the start method.
    _run_workers("forkserver", tmp_path, 2, target=worker_records_units)
    assert merge_run(tmp_path).tree[("iteration", "step")].counters == {
        "units": 2 * UNITS_PER_WORKER,
    }


# ── failure modes ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", START_METHODS)
def test_a_worker_that_raises_still_contributes_its_work(method: str, tmp_path: Path) -> None:
    _run_workers(method, tmp_path, 2)
    exit_codes = _run_workers(method, tmp_path, 1, target=worker_raises)
    assert exit_codes[0] != 0

    run = merge_run(tmp_path)

    assert run.tree[("before_crash",)].counters == {"units": 1}
    assert run.tree[("iteration", "step")].counters == {"units": 2 * UNITS_PER_WORKER}
    assert run.unreadable == []


@pytest.mark.parametrize("method", START_METHODS)
def test_a_sigkilled_worker_does_not_break_the_merge(method: str, tmp_path: Path) -> None:
    """SIGKILL cannot be caught, so that worker's work is simply absent — not corrupt."""
    _run_workers(method, tmp_path, 2)
    exit_codes = _run_workers(method, tmp_path, 1, target=worker_is_killed)
    assert exit_codes[0] == -signal.SIGKILL

    run = merge_run(tmp_path)

    assert run.tree[("iteration", "step")].counters == {"units": 2 * UNITS_PER_WORKER}
    assert "killed" not in run.roles, "a worker killed before its first flush leaves nothing"
    assert run.unreadable == []


def test_a_truncated_worker_file_is_counted_as_lost(tmp_path: Path) -> None:
    """A half-written file is reported explicitly rather than silently under-counting."""
    _run_workers("spawn", tmp_path, 2)
    victim = sorted((tmp_path / "workers").rglob("w_*.json"))[0]
    victim.write_text(victim.read_text(encoding="utf-8")[:40], encoding="utf-8")

    run = merge_run(tmp_path)

    assert len(run.unreadable) == 1
    assert len(run.workers) == 1
    assert run.tree[("iteration", "step")].counters == {"units": UNITS_PER_WORKER}


# ── imbalance ───────────────────────────────────────────────────────────────


def worker_with_variable_load(run_dir: str, rank: int) -> None:
    """Worker ``rank`` sleeps proportionally longer, producing a known imbalance."""
    profiler = Profiler(
        run_dir=run_dir, role="actor", enabled=True, snapshot_interval_s=None,
        sample_interval_s=None,
    )
    with profiler.phase("work"):
        time.sleep(0.02 * (rank + 1))
    profiler.close()


def test_imbalance_reflects_uneven_workers(tmp_path: Path) -> None:
    _run_workers("spawn", tmp_path, 4, target=worker_with_variable_load)

    run = merge_run(tmp_path)

    # Sleeps of 1x, 2x, 3x, 4x: max/mean = 4 / 2.5 = 1.6.
    assert run.imbalance == pytest.approx(1.6, rel=0.15)


# ── fork safety ─────────────────────────────────────────────────────────────


def test_no_profiler_thread_is_alive_across_a_fork(tmp_path: Path) -> None:
    """Enabling the profiler must not add fork-deadlock risk to code that forks."""
    _context("fork")
    profiler = Profiler(
        run_dir=tmp_path, role="parent", enabled=True, snapshot_interval_s=0.05,
        sample_interval_s=0.05,
    )
    time.sleep(0.1)
    assert _profiler_threads(), "the sampler should be running before the fork"

    observed: list[str] = []
    original = profiler._pause_threads_before_fork  # noqa: SLF001

    def record_then_pause() -> None:
        original()
        observed.extend(_profiler_threads())

    os.register_at_fork(before=record_then_pause)
    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - runs only in the forked child
        os._exit(0)
    os.waitpid(child_pid, 0)

    assert observed == [], f"threads alive during fork: {observed}"
    time.sleep(0.15)
    assert _profiler_threads(), "the parent's sampler must resume after the fork"
    profiler.close()


def _profiler_threads() -> list[str]:
    import threading

    return [t.name for t in threading.enumerate() if t.name.startswith("lineprofiler")]


# ── the ambient profiler across start methods ───────────────────────────────


@pytest.mark.parametrize("method", START_METHODS)
def test_ambient_phases_are_recorded_by_every_worker(method: str, tmp_path: Path) -> None:
    """Each child must record onto its *own* profiler, whatever the start method."""
    exit_codes = _run_workers(
        method, tmp_path, 4, target=worker_uses_the_ambient_profiler,
    )
    assert exit_codes == [0, 0, 0, 0]

    run = merge_run(tmp_path)

    assert len(run.workers) == 4
    assert run.tree[("iteration", "step")].counters == {"units": 4 * UNITS_PER_WORKER}


def test_a_forked_child_resolves_its_own_profiler_not_the_parents(tmp_path: Path) -> None:
    """A fork copies the module global along with everything else, so the installed profiler
    in the child is the child's re-initialised copy. If that ever stops holding, the child
    writes into the parent's worker file and both runs are wrong."""
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork is unavailable on this platform")
    parent = Profiler(
        run_dir=tmp_path, role="learner", enabled=True, install=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    with accounting.phase("parent_only"):
        accounting.count("units", 1)
    writer = parent._writer  # noqa: SLF001
    assert writer is not None
    parent_file = writer.path

    _run_workers("fork", tmp_path, 2, target=worker_uses_the_ambient_profiler)
    parent.close()
    accounting.uninstall_profiler()

    run = merge_run(tmp_path)
    assert len(run.workers) == 3, "the parent and both children each wrote their own file"
    assert run.tree[("parent_only",)].counters == {"units": 1}, "recorded exactly once"
    assert run.tree[("iteration", "step")].counters == {"units": 2 * UNITS_PER_WORKER}
    assert sum(w.path == parent_file for w in run.workers) == 1


# ── a closed profiler must not come back in a forked child ──────────────────


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="fork unavailable",
)
def test_a_fork_after_close_does_not_resurrect_the_profiler(tmp_path: Path) -> None:
    """``os.register_at_fork`` callbacks cannot be unregistered, so a closed profiler was still
    called in every later child — and set ``_closed`` back to ``False``, giving the child a live
    writer, a sampler and a flush timer for a profiler the process had finished with. The new
    writer even re-created the run directory, so a test that deleted it got it back.
    """
    run_dir = tmp_path / "run"
    profiler = Profiler(
        run_dir=run_dir, role="parent", enabled=True,
        snapshot_interval_s=0.05, sample_interval_s=0.05,
    )
    profiler.close()
    shutil.rmtree(run_dir)

    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - runs only in the forked child
        os.close(read_fd)
        time.sleep(0.3)  # long enough for a resurrected flush timer to fire twice
        report = f"{profiler._closed}|{_profiler_threads()}"  # noqa: SLF001
        os.write(write_fd, report.encode())
        os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    child_says = os.read(read_fd, 4096).decode()
    os.close(read_fd)
    os.waitpid(child_pid, 0)

    assert child_says == "True|[]", f"child resurrected the profiler: {child_says}"
    assert not run_dir.exists(), "the child re-created the run directory it was finished with"


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="fork unavailable",
)
def test_a_fork_after_close_does_not_resurrect_the_ambient_profiler(tmp_path: Path) -> None:
    """``close()`` uninstalls, but the fork callback never re-installed — so a resurrected
    profiler wrote files that module-level ``accounting.phase()`` could not see. The two halves
    of the ambient API disagreed about whether a profiler existed.
    """
    profiler = Profiler(
        run_dir=tmp_path / "run", enabled=True, install=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    profiler.close()
    assert accounting.installed_profiler() is None

    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - runs only in the forked child
        os.close(read_fd)
        with accounting.phase("after_close"):
            pass
        report = f"{accounting.installed_profiler()}|{profiler._closed}"  # noqa: SLF001
        os.write(write_fd, report.encode())
        os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    child_says = os.read(read_fd, 4096).decode()
    os.close(read_fd)
    os.waitpid(child_pid, 0)

    assert child_says == "None|True", f"ambient state diverged in the child: {child_says}"


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="fork unavailable",
)
def test_a_live_profiler_still_reinitialises_in_a_forked_child(tmp_path: Path) -> None:
    """The guard above must not cost the ordinary case: an *open* profiler still gives its
    child a fresh identity, and the child's ambient lookup still resolves to it."""
    profiler = Profiler(
        run_dir=tmp_path / "run", enabled=True, install=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    try:
        read_fd, write_fd = os.pipe()
        child_pid = os.fork()
        if child_pid == 0:  # pragma: no cover - runs only in the forked child
            os.close(read_fd)
            resolved = accounting.installed_profiler() is profiler
            with accounting.phase("child_work"):
                pass
            profiler.close()
            os.write(write_fd, str(resolved).encode())
            os.close(write_fd)
            os._exit(0)

        os.close(write_fd)
        child_says = os.read(read_fd, 4096).decode()
        os.close(read_fd)
        os.waitpid(child_pid, 0)

        assert child_says == "True"
    finally:
        profiler.close()

    tree = merge_run(tmp_path / "run").tree
    assert ("child_work",) in tree, "the forked child of a live profiler wrote nothing"
