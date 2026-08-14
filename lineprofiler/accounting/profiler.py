"""Semantic accounting for long-running, multi-process reinforcement-learning training.

This is not a profiler in the tracing sense: it records aggregates for regions *you* name,
at a cost low enough to leave enabled for a twelve-hour run. Detailed data comes from
existing tools — ``torch.profiler``, VizTracer, memray, nsys — which later phases start for
a bounded window rather than reimplement.

Usage::

    profiler = Profiler(run_dir="profile")
    with profiler:
        for _ in range(steps):
            with profiler.phase("self_play"):
                with profiler.phase("mcts"):
                    profiler.count("mcts_simulations", 64)
"""

from __future__ import annotations

import atexit
import contextlib
import os
import signal
import threading
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager
from functools import wraps
from pathlib import Path
from time import perf_counter_ns, thread_time_ns
from types import FrameType
from typing import ParamSpec, Self, TypeVar, overload

from lineprofiler.accounting.backend import Backend, BackendWindow
from lineprofiler.accounting.capabilities import (
    cuda_synchronize,
    nvtx_range_functions,
    record_function_factory,
)
from lineprofiler.accounting.histogram import bucket_index
from lineprofiler.accounting.phase import PhasePath, PhaseStats, PhaseTree
from lineprofiler.accounting.sampler import (
    IoSnapshot,
    ProcessHandle,
    ResourceSampler,
    open_process,
    read_io_snapshot,
)
from lineprofiler.accounting.selfio import bytes_written as selfio_bytes_written
from lineprofiler.accounting.selfio import reset as selfio_reset
from lineprofiler.accounting.snapshot import SnapshotWriter, new_run_id

ENV_ENABLE = "LINEPROFILER_PROFILE"
"""Environment variable consulted once, at construction, for the default of ``enabled``."""

ENV_ROLE = "LINEPROFILER_ROLE"
"""Environment variable giving a spawned worker its role without changing its call site."""

ENV_RUN_DIR = "LINEPROFILER_RUN_DIR"
"""Environment variable pointing a child process at the run directory its parent opened."""

ENV_RUN_ID = "LINEPROFILER_RUN_ID"
"""Environment variable joining a child to its parent's *attempt*, so that a rerun into the
same directory is a separate run rather than extra workers on the previous one."""

MAX_DEPTH = 32
"""Phases nested deeper than this are folded into their ancestor, bounding the tree size."""

MAX_PHASES = 4096
"""Distinct phase paths one thread may create before further names fold into their parent.

Bounds memory: each node holds a dense 512-bucket histogram and is re-serialised into every
snapshot, so an uncapped tree fed from a dynamic name is both a leak and a growing write.
Well above what any hand-written instrumentation reaches — a tree this wide means names are
being built from data, which the warning says outright."""

_EXIT_SIGNALS = (signal.SIGTERM, signal.SIGUSR1, signal.SIGHUP)
"""Signals that mean "this process is ending" on a batch system, in the order they are
installed. Slurm sends SIGUSR1 ahead of preemption and SIGHUP on a lost allocation; both
terminate by default without running atexit."""

_ROOT: PhasePath = ()

_live_profilers: list[str] = []

_P = ParamSpec("_P")
_R = TypeVar("_R")


class Profiler:
    """Records wall time, CPU time and work counters for named phases.

    Args:
        run_dir: Directory for output. Created if absent. Defaults to the directory a
            parent process propagated via ``LINEPROFILER_RUN_DIR``, then to ``"profile"``.
        role: What this process does — ``"learner"``, ``"actor"``, ``"inference"``, or
            whatever your architecture calls its workers. The report groups phases by role,
            because in a pipeline where sixteen actors run alongside one learner, a single
            global percentage tells you nothing. Defaults to ``LINEPROFILER_ROLE``, then to
            ``"main"``.
        enabled: Master switch. When ``False`` every method is a no-op with no clock reads,
            no allocation and no thread. Defaults to the truthiness of the
            ``LINEPROFILER_PROFILE`` environment variable.
        snapshot_interval_s: Seconds between aggregate flushes. ``None`` disables the
            background flush thread, leaving only the exit and signal flushes.
        sample_interval_s: Seconds between resource samples (memory, I/O, GPU). ``None``
            disables the sampler thread entirely.
        measure_cpu: Record per-phase CPU time, and hence ``wait_ns``. This costs roughly
            1.5 µs per phase because ``time.thread_time_ns()`` is a real syscall
            (``CLOCK_THREAD_CPUTIME_ID`` is not in the vDSO), against about 1.7 µs for a
            phase without it. Leave it on unless you are instrumenting an inner loop.
        backend: At most one heavy profiler — ``"torch"`` or ``"viztracer"`` — run for a
            bounded window. Never more than one; they contend for the same hooks.
        backend_window: ``(start, end)`` entry numbers of ``window_phase``, or ``None``.
        window_phase: The repeating phase the window is counted in. Any training loop has
            one; name yours here rather than assuming it is called "iteration".
        annotate: Emit an NVTX range and a ``torch.profiler.record_function`` around every
            phase, so an externally started Nsight Systems or Kineto capture shows your
            phase names instead of anonymous frames. Costs a few hundred nanoseconds per
            phase and degrades to nothing when neither torch nor ``nvtx`` is installed.
            Off by default: it only pays for itself while such a capture is running.

    Statistics are accumulated per thread and merged only when a snapshot is written, so the
    hot path takes no locks and needs none. A snapshot taken while another thread is inside
    a phase can therefore observe that thread's counters a few hundred nanoseconds stale;
    the following snapshot corrects it.

    Test specifically:
        - ``enabled=False`` creates no files and no threads, and costs under ~200 ns/phase
        - the environment variable is read once at construction, not on every call
        - a second profiler on the same run directory warns
        - phases opened in different threads do not interfere
        - every combination of sampler on/off and backend none/torch constructs and closes
    """

    def __init__(
        self,
        run_dir: str | Path | None = None,
        role: str | None = None,
        enabled: bool | None = None,
        snapshot_interval_s: float | None = 30.0,
        sample_interval_s: float | None = 1.0,
        measure_cpu: bool = True,
        backend: Backend | str | None = None,
        backend_window: tuple[int, int] | None = None,
        window_phase: str = "iteration",
        annotate: bool = False,
    ) -> None:
        self.enabled: bool = _resolve_enabled(enabled)
        self.measure_cpu: bool = measure_cpu
        self.run_dir: Path = _resolve_run_dir(run_dir)
        self.run_id: str = os.environ.get(ENV_RUN_ID, "") or new_run_id()
        self.role: str = role or os.environ.get(ENV_ROLE, "") or "main"
        self.backend: Backend = Backend.parse(backend)

        self._local = threading.local()
        self._trees: list[PhaseTree] = []
        self._states: list[_ThreadState] = []
        self._writer: SnapshotWriter | None = None
        self._sampler: ResourceSampler | None = None
        self._flush_timer: threading.Timer | None = None
        self._snapshot_interval_s = snapshot_interval_s
        self._closed = False
        self._snapshot_failures = 0
        self._phase_overflow = 0
        self._window: BackendWindow | None = None
        self._process: ProcessHandle | None = open_process()
        self._sample_interval_s = sample_interval_s
        self._nvtx = nvtx_range_functions() if annotate and self.enabled else None
        self._record_function = record_function_factory() if annotate and self.enabled else None
        # Resolved once: torch.cuda.is_available() initialises the driver on first call, so
        # asking per phase would put a lock on the hot path.
        self._cuda_sync = cuda_synchronize() if self.enabled else None

        if not self.enabled:
            return

        _warn_if_already_live(self.run_dir)
        _propagate_to_children(self.run_dir, self.run_id)
        self._writer = SnapshotWriter(self.run_dir, role=self.role, run_id=self.run_id)
        self._start_backend_window(backend_window, window_phase)
        self._start_sampler(sample_interval_s)
        self._install_exit_hooks()
        self._start_flush_timer()
        os.register_at_fork(
            before=self._pause_threads_before_fork,
            after_in_parent=self._resume_threads_after_fork,
            after_in_child=self._reinitialise_after_fork,
        )

    # ── public API ──────────────────────────────────────────────────────────

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def phase(self, name: str, io: bool = False, sync: bool = False) -> _PhaseScope | _NullScope:
        """Open a named region, nested under the phase currently open on this thread.

        Args:
            name: The phase's name. Sibling phases with the same name accumulate together.
            sync: Drain the CUDA queue at both ends of the phase, so its wall time means GPU
                time. CUDA launches are asynchronous: without this, a phase around a forward
                pass measures the time to *enqueue* the kernels, and their real cost surfaces
                later as ``wait%`` on whichever phase happens to synchronise — usually the
                one that copies a result back, which did nothing wrong.

                Synchronising on entry as well as exit is what makes the number attributable:
                exit alone would bill this phase for whatever was still queued when it
                started. The cost is the pipelining you give up — the CPU can no longer run
                ahead of the GPU across this boundary — so put it on the phases you are
                actively measuring, not on every phase in the loop, and take timings from a
                run where it is off once you know where the work is. A no-op when torch is
                absent or no CUDA device is visible.
            io: Read the process byte counters on entry and exit, so this phase's I/O is
                measured *exactly* rather than inferred from the 1 Hz sampler. Costs a
                ``/proc`` read at each end — tens of microseconds — so it belongs on coarse
                phases that actually touch the disk (checkpoint, replay load, dataset read),
                never on an inner loop.

                Four counters land on the phase. ``io_read_bytes``/``io_write_bytes`` are
                block-layer traffic — what reached the device. ``io_read_chars``/
                ``io_write_chars`` are syscall-level, so a read served from the page cache
                shows up there and nowhere else. Bytes the profiler wrote for its own
                bookkeeping are excluded from all four.

        Test specifically:
            - an exception inside the body still closes the phase and records it
            - recursive entry of the same name does not double-count the parent's
              ``child_wall_ns``
            - the phase stack is thread-local, so concurrent threads nest independently
            - a generator body abandoned without exhaustion still closes its phase
            - ``io=True`` on a phase writing a known number of bytes reports them on that
              phase, and reports nothing on a phase that writes none
            - a page-cached read reports ``io_read_chars`` and no ``io_read_bytes``
            - a silent phase running alongside the sampler reports no I/O at all
            - ``sync=True`` synchronises once on entry and once on exit, and the exit
              synchronisation happens before the clock is read
            - ``sync=True`` without a CUDA device records the phase like any other
        """
        if not self.enabled:
            return _NULL_SCOPE
        return _PhaseScope(self, name, io, sync)

    def io_counters(self) -> IoSnapshot:
        """Return this process's cumulative byte counters at the disk and syscall layers.

        All zeros when psutil is absent or the platform has no per-process counters (macOS),
        which makes an ``io=True`` phase degrade to an ordinary phase. Use
        :meth:`IoSnapshot.is_empty` to test for that rather than comparing against a tuple.
        """
        return read_io_snapshot(self._process)

    def count(self, name: str, n: int = 1) -> None:
        """Attribute ``n`` work units to the phase currently open on this thread.

        Counting outside any phase attributes to the implicit root.

        Test specifically:
            - counting outside any phase lands on the root
            - counters survive the snapshot round-trip and the cross-worker merge
            - a float ``n`` raises ``TypeError`` rather than silently truncating
        """
        if not self.enabled:
            return
        if not isinstance(n, int):
            raise TypeError(f"count() takes an int, got {type(n).__name__}")
        self._thread_state().nodes[-1].add_count(name, n)

    def snapshot(self) -> None:
        """Merge every thread's statistics and write the current aggregate to disk.

        Safe to call from a signal handler: the hot path holds no locks, so there is
        nothing for this to deadlock against.

        Test specifically:
            - two snapshots in quick succession both produce a valid, complete file
            - a ``SIGTERM`` mid-run leaves a parseable file
        """
        if self._writer is None:
            return
        self._writer.write(self.merged_tree())

    def merged_tree(self) -> PhaseTree:
        """Return the union of every thread's phase tree for this process."""
        merged: PhaseTree = {}
        for tree in list(self._trees):
            for path, stats in list(tree.items()):
                node = merged.get(path)
                if node is None:
                    merged[path] = stats.copy()
                else:
                    node.merge(stats)
        return merged

    def close(self) -> None:
        """Write a final snapshot and stop the sampler, backend and flush threads."""
        if self._closed or not self.enabled:
            return
        self._closed = True
        if self._flush_timer is not None:
            self._flush_timer.cancel()
        if self._sampler is not None:
            self._sampler.stop()
        if self._window is not None:
            self._window.close()
            if self._writer is not None:
                self._writer.record_backend(self._window.describe())
        self.snapshot()
        if str(self.run_dir) in _live_profilers:
            _live_profilers.remove(str(self.run_dir))

    def current_phase(self) -> str:
        """Return the deepest phase open in any thread, as a ``/``-joined path.

        Used to tag resource samples. With several threads active the deepest stack is the
        most specific work in progress, which is the useful answer; it is an approximation
        when two threads are equally deep.
        """
        deepest: PhasePath = _ROOT
        for state in list(self._states):
            path = state.paths[-1]
            if len(path) > len(deepest):
                deepest = path
        return "/".join(deepest)

    @overload
    def profile(self, func: Callable[_P, _R]) -> Callable[_P, _R]: ...

    @overload
    def profile(self, *, name: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...

    def profile(
        self,
        func: Callable[_P, _R] | None = None,
        *,
        name: str | None = None,
    ) -> Callable[_P, _R] | Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Decorate a function so each call is a phase. Usable bare or with ``name=``.

        Test specifically:
            - both ``@profiler.profile`` and ``@profiler.profile(name="x")`` work
            - the wrapper preserves ``__name__`` and ``__doc__``
        """

        def decorate(target: Callable[_P, _R]) -> Callable[_P, _R]:
            phase_name = name or target.__name__

            @wraps(target)
            def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                with self.phase(phase_name):
                    return target(*args, **kwargs)

            return wrapper

        return decorate if func is None else decorate(func)

    # ── internals ───────────────────────────────────────────────────────────

    def _note_phase_overflow(self, path: PhasePath) -> None:
        """Warn once that the phase tree is full, naming a path that did not fit."""
        if self._phase_overflow:
            self._phase_overflow += 1
            return
        self._phase_overflow = 1
        warnings.warn(
            f"accounting phase tree reached {MAX_PHASES} distinct paths; further phases are "
            f"folded into their parent. The first that did not fit was {'/'.join(path)!r}. "
            "This usually means a phase name is built from data — use a fixed name and "
            "count() for the varying part.",
            RuntimeWarning,
            stacklevel=4,
        )

    def _thread_state(self) -> _ThreadState:
        """Return this thread's phase stack, creating and registering it on first use.

        Uses ``try``/``except`` rather than ``getattr`` with a default because the attribute
        is present on every call but the first, and the exception path is what costs.
        """
        try:
            return self._local.state  # type: ignore[no-any-return]
        except AttributeError:
            state = _ThreadState()
            self._local.state = state
            self._trees.append(state.tree)  # list.append is atomic; no lock needed
            self._states.append(state)
            return state

    def _pause_threads_before_fork(self) -> None:
        """Stop this profiler's threads so none is running when ``fork`` copies the process.

        Forking a multi-threaded process is hazardous: only the calling thread survives, and
        any lock another thread happened to hold is copied in its locked state, which is how
        forked children deadlock. Enabling a profiler must not add that hazard to a codebase
        that forks, so the sampler and flush timer are stopped for the duration of the fork
        and started again on both sides afterwards.

        Test specifically:
            - no ``lineprofiler`` thread is alive at the moment of the fork
            - the parent's sampler resumes and keeps producing rows afterwards
        """
        if not self.enabled:
            return
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None
        if self._sampler is not None:
            self._sampler.stop()

    def _resume_threads_after_fork(self) -> None:
        """Restart the parent's threads once the fork has completed."""
        if not self.enabled or self._closed:
            return
        if self._sampler is not None:
            self._sampler.start()
        self._start_flush_timer()

    def _reinitialise_after_fork(self) -> None:
        """Give the forked child its own identity, files and threads.

        A ``fork`` copies this object wholesale: the child would otherwise inherit the
        parent's output path and overwrite the parent's snapshots, and would re-report the
        parent's accumulated phases as its own. Threads do not survive a fork either, so the
        sampler and the flush timer exist only as dead references until they are restarted.

        Test specifically:
            - a forked child never writes to the parent's file
            - the child's tree starts empty rather than inheriting the parent's totals
            - the sampler and flush timer run again in the child
            - nesting a fork inside a phase leaves the child with a clean phase stack
            - the child's profiler-overhead total starts at zero, since its byte counters do
        """
        if not self.enabled:
            return
        self._local = threading.local()
        self._trees = []
        self._states = []
        self._closed = False
        self._process = open_process()
        # The child's OS byte counters start at zero, so the parent's overhead total would
        # over-deduct against them for the rest of the child's life.
        selfio_reset()
        _live_profilers.clear()
        _live_profilers.append(str(self.run_dir))
        self._writer = SnapshotWriter(self.run_dir, role=self.role, run_id=self.run_id)
        self._sampler = None
        self._flush_timer = None
        self._start_sampler(self._sample_interval_s)
        self._start_flush_timer()

    def _start_sampler(self, interval_s: float | None) -> None:
        """Start the resource sampler unless it was disabled."""
        if interval_s is None or self._writer is None:
            return
        self._sampler = ResourceSampler(
            path=self._writer.samples_path,
            interval_s=interval_s,
            phase_of=self.current_phase,
        )
        self._sampler.start()

    def _start_backend_window(self, window: tuple[int, int] | None, phase_name: str) -> None:
        """Arm the heavy-profiler window, if one was configured."""
        if self.backend is Backend.NONE or window is None:
            return
        self._window = BackendWindow(self.backend, window, phase_name, self.run_dir)

    def _install_exit_hooks(self) -> None:
        """Flush at interpreter exit and on every signal a scheduler uses to end a job.

        ``SIGTERM`` alone was not enough on a cluster. Slurm's preemption and time-limit
        warning is ``SIGUSR1`` (``--signal=USR1@120``, the standard checkpoint-before-eviction
        idiom) and its default disposition kills the process without running ``atexit``, so
        everything since the last periodic flush was lost precisely in the runs where you most
        wanted the numbers. ``SIGHUP`` covers a dropped allocation.

        ``SIGKILL`` remains unreachable by design: the last periodic snapshot is what survives
        it, which is why the flush cadence matters.
        """
        atexit.register(self.close)
        for signum in _EXIT_SIGNALS:
            self._chain_signal(signum)

    def _chain_signal(self, signum: signal.Signals) -> None:
        """Install a flushing handler for one signal, preserving whatever was there."""
        try:
            previous = signal.getsignal(signum)
        except ValueError:  # not on the main thread
            return

        def handler(number: int, frame: FrameType | None) -> None:
            self.close()
            if callable(previous):
                previous(number, frame)
            elif previous == signal.SIG_DFL:
                # Restoring the default and re-raising is what makes the process actually die
                # with the right status; returning from here would swallow the signal.
                signal.signal(number, signal.SIG_DFL)
                os.kill(os.getpid(), number)

        with contextlib.suppress(ValueError, OSError):
            signal.signal(signum, handler)

    def _start_flush_timer(self) -> None:
        """Schedule the next background snapshot, if a cadence was configured."""
        if self._snapshot_interval_s is None or self._closed:
            return
        timer = threading.Timer(self._snapshot_interval_s, self._on_timer)
        timer.daemon = True
        self._flush_timer = timer
        timer.start()

    def _on_timer(self) -> None:
        """Snapshot, then re-arm — and re-arm even if the snapshot raised.

        The re-arm used to follow the snapshot unguarded, so a single exception ended
        periodic flushing for the rest of the process. The worker file then froze at whatever
        it held at that moment, still parsing as a complete result, and the run under-reported
        by however many hours remained with nothing in the output to say so.
        """
        try:
            self.snapshot()
        except Exception:  # noqa: BLE001 - a failed flush must not stop the ones after it
            self._snapshot_failures += 1
        finally:
            self._start_flush_timer()


class _ThreadState:
    """One thread's open phase stack and its private slice of the phase tree."""

    __slots__ = ("names", "nodes", "paths", "tree")

    def __init__(self) -> None:
        root = PhaseStats()
        self.tree: PhaseTree = {_ROOT: root}
        self.names: list[str] = []
        self.paths: list[PhasePath] = [_ROOT]
        self.nodes: list[PhaseStats] = [root]


class _PhaseScope:
    """Context manager for one phase entry. Allocated per call, so it nests safely."""

    __slots__ = (
        "_cpu0", "_function", "_io", "_io0", "_name", "_profiler", "_self_io0", "_state",
        "_stats", "_sync", "_wall0",
    )

    def __init__(
        self,
        profiler: Profiler,
        name: str,
        io: bool = False,
        sync: bool = False,
    ) -> None:
        self._profiler = profiler
        self._name = name
        self._io = io
        # Resolved here rather than read at both ends: a phase that does not synchronise
        # then costs one `is not None` test instead of two attribute loads and a branch.
        self._sync = profiler._cuda_sync if sync else None
        self._state: _ThreadState | None = None
        self._stats: PhaseStats | None = None
        self._wall0 = 0
        self._cpu0 = 0
        self._io0 = IoSnapshot()
        self._self_io0 = (0, 0)
        self._function: AbstractContextManager[object] | None = None

    def __enter__(self) -> None:
        state = self._profiler._thread_state()
        if len(state.names) >= MAX_DEPTH:
            self._stats = state.nodes[-1]  # fold into the ancestor rather than grow the tree
            path = state.paths[-1]
        else:
            path = state.paths[-1] + (self._name,)
            stats = state.tree.get(path)
            if stats is None:
                stats = self._admit(state, path)
            if stats is None:
                self._stats = state.nodes[-1]  # tree is full; fold, exactly as at MAX_DEPTH
                path = state.paths[-1]
            else:
                self._stats = stats
        state.names.append(self._name)
        state.paths.append(path)
        state.nodes.append(self._stats)
        self._state = state
        window = self._profiler._window
        if window is not None:
            window.on_phase_enter(self._name)
        if self._io:
            self._io0 = self._profiler.io_counters()
            self._self_io0 = selfio_bytes_written()

        # Inlined rather than delegated: annotation is off by default, and a method call
        # on each side of every phase is measurable at roughly a microsecond per phase.
        nvtx = self._profiler._nvtx
        if nvtx is not None:
            nvtx[0](self._name)
        factory = self._profiler._record_function
        if factory is not None:
            self._function = factory(self._name)
            self._function.__enter__()
        # Last thing before the clocks: work queued by an earlier phase must not be billed
        # to this one, and anything between the drain and the timestamp is measured.
        if self._sync is not None:
            self._sync()
        self._cpu0 = thread_time_ns() if self._profiler.measure_cpu else 0
        self._wall0 = perf_counter_ns()

    def _admit(self, state: _ThreadState, path: PhasePath) -> PhaseStats | None:
        """Create the node for a newly seen path, or ``None`` when the tree is full.

        A phase name built from data — ``phase(f"episode_{i}")``, ``phase(f"load {filename}")``
        — grows the tree for the life of the process, and every node carries a dense
        512-bucket histogram that is also rewritten into every snapshot. That was an unbounded
        memory leak whose only symptom was a process slowly getting fatter, with nothing
        pointing at the cause. Past the cap, phases fold into their parent and the profiler
        says so once, naming the path that overflowed.
        """
        if len(state.tree) >= MAX_PHASES:
            self._profiler._note_phase_overflow(path)
            return None
        stats = PhaseStats()
        state.tree[path] = stats
        return stats

    def __exit__(self, *exc: object) -> None:
        if self._sync is not None:
            self._sync()  # first: the kernels this phase launched are part of its cost
        wall = perf_counter_ns() - self._wall0
        cpu = thread_time_ns() - self._cpu0 if self._profiler.measure_cpu else 0
        function = self._function
        if function is not None:
            function.__exit__(None, None, None)
            self._function = None
        nvtx = self._profiler._nvtx
        if nvtx is not None:
            nvtx[1]()
        state = self._state
        if state is None:
            return
        state.names.pop()
        state.paths.pop()
        state.nodes.pop()
        stats = self._stats
        parent = state.nodes[-1]
        if stats is None or stats is parent:
            return

        # PhaseStats.record and DurationHistogram.observe are inlined: at roughly a
        # microsecond per phase, two extra Python-level calls are a measurable share.
        stats.calls += 1
        stats.wall_ns += wall
        stats.cpu_ns += cpu
        histogram = stats.hist
        histogram.buckets[bucket_index(wall)] += 1
        histogram.count += 1
        parent.child_wall_ns += wall

        if self._io:
            self._record_io(stats)

        window = self._profiler._window
        if window is not None:
            window.on_phase_exit(self._name)

    def _record_io(self, stats: PhaseStats) -> None:
        """Attribute the bytes this phase moved, measured at its own boundaries.

        Records nothing at all when either boundary reading failed — a phase with no I/O
        counters is indistinguishable from one that moved no bytes, and that is the correct
        answer, where a fabricated delta is not.

        The profiler's own writes are deducted at the layer they were measured on: the
        syscall figure from ``write_chars`` and the bracketed block figure from
        ``write_bytes``. Deducting the syscall figure from both would leave most of the
        overhead in place, because rewriting a small worker file costs several whole blocks
        for data, inode and journal — measured at eight times the bytes handed to ``write``.
        """
        now = self._profiler.io_counters()
        if not (now.available and self._io0.available):
            # Cumulative counters: differencing against a reading that never happened would
            # bill this phase for the process's entire lifetime of traffic, in the block the
            # report presents as exactly measured. Record nothing instead.
            return
        chars_before, blocks_before = self._self_io0
        chars_after, blocks_after = selfio_bytes_written()
        self._add_delta(stats, "io_read_bytes", now.read_bytes - self._io0.read_bytes)
        self._add_delta(stats, "io_read_chars", now.read_chars - self._io0.read_chars)
        self._add_delta(
            stats,
            "io_write_bytes",
            now.write_bytes - self._io0.write_bytes - (blocks_after - blocks_before),
        )
        self._add_delta(
            stats,
            "io_write_chars",
            now.write_chars - self._io0.write_chars - (chars_after - chars_before),
        )

    @staticmethod
    def _add_delta(stats: PhaseStats, name: str, delta: int) -> None:
        if delta > 0:
            stats.add_count(name, delta)


class _NullScope:
    """The context manager handed out when the profiler is disabled: no state, no clocks."""

    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


_NULL_SCOPE = _NullScope()


def _resolve_enabled(enabled: bool | None) -> bool:
    """Resolve the master switch, reading the environment only when not given explicitly."""
    if enabled is not None:
        return enabled
    return _truthy(os.environ.get(ENV_ENABLE, ""))


def _resolve_run_dir(run_dir: str | Path | None) -> Path:
    """Use the caller's directory, else the one a parent process propagated, else the default."""
    if run_dir is not None:
        return Path(run_dir)
    return Path(os.environ.get(ENV_RUN_DIR, "") or "profile")


def _truthy(value: str) -> bool:
    return bool(value.strip()) and value not in {"0", "false", "False"}


def _propagate_to_children(run_dir: Path, run_id: str) -> None:
    """Export the switch and run directory so child processes enable themselves.

    A worker started with ``spawn`` inherits the environment but not the parent's objects,
    so this is what lets ``Profiler(role="actor")`` in a worker join the parent's run
    without every call site having to thread the configuration through. Set explicitly by
    the user, either variable wins — this only fills in what is unset.

    ``forkserver`` is the exception: its daemon is forked once and its children inherit the
    daemon's environment, a snapshot taken when the daemon started. Variables exported after
    that never reach them. Under ``forkserver``, export ``LINEPROFILER_PROFILE`` in the shell
    before training starts, or pass ``enabled`` and ``run_dir`` to each worker explicitly.
    """
    os.environ.setdefault(ENV_ENABLE, "1")
    os.environ.setdefault(ENV_RUN_DIR, str(run_dir))
    os.environ.setdefault(ENV_RUN_ID, run_id)


def _warn_if_already_live(run_dir: Path) -> None:
    """Warn when a second profiler targets a run directory that is already being written.

    Two profilers on *different* directories are harmless — each keeps its own thread-local
    state — so only a shared directory, where both would write worker files for the same
    run, is worth a warning.
    """
    if str(run_dir) in _live_profilers:
        warnings.warn(
            f"another accounting Profiler is already writing to {run_dir}; "
            "its phases will be recorded as a separate worker",
            RuntimeWarning,
            stacklevel=3,
        )
    _live_profilers.append(str(run_dir))


