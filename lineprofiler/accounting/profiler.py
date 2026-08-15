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
import re
import signal
import threading
import warnings
import weakref
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
from lineprofiler.accounting.phasetree import PhasePath, PhaseStats, PhaseTree
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

_NAME_SHAPE_WARN = 128
"""Distinct phase names sharing one shape before the profiler says the names look generated.

Far above any hand-written vocabulary — a 96-layer transformer named per layer stays quiet —
and far below ``MAX_PHASES``, so the warning arrives while the report is still readable."""

_NUMERIC_RUN = re.compile(r"\d+")
"""Collapses the varying part of a generated name, so ``episode_1`` and ``episode_2`` share a
shape. Only ever applied when a path is first admitted, never on the hot path."""

_Handler = Callable[[int, "FrameType | None"], object] | int | None
"""What ``signal.getsignal`` returns: a callable, ``SIG_DFL``/``SIG_IGN``, or ``None`` for a
handler that was not installed from Python."""

_ROOT: PhasePath = ()

_live_profilers: list[str] = []

_fork_hooks_installed = False
"""Whether this interpreter has had the fork callbacks registered. Once only, ever."""

_fork_registry: list[weakref.ref[Profiler]] = []
"""The profilers the fork callbacks dispatch to, weakly.

``os.register_at_fork`` has no counterpart that unregisters, so anything handed to it lives for
the interpreter's lifetime. Registering three *bound methods* per profiler therefore made every
enabled profiler immortal — it and its phase trees, thread states and writer — which a test
suite constructing one per test pays for in full. One registration for the process, dispatching
over weak references, lets a closed or dropped profiler actually go away.
"""


def _register_fork_hooks(profiler: Profiler) -> None:
    """Add ``profiler`` to the fork registry, arming the process-wide callbacks on first use."""
    global _fork_hooks_installed  # noqa: PLW0603 - one registration per interpreter
    _fork_registry.append(weakref.ref(profiler))
    if _fork_hooks_installed:
        return
    os.register_at_fork(
        before=_before_fork,
        after_in_parent=_after_fork_in_parent,
        after_in_child=_after_fork_in_child,
    )
    _fork_hooks_installed = True


def _forkable_profilers() -> list[Profiler]:
    """Return the live, open profilers, dropping any that closed or were collected."""
    live: list[Profiler] = []
    survivors: list[weakref.ref[Profiler]] = []
    for ref in _fork_registry:
        profiler = ref()
        if profiler is None or profiler._closed:  # noqa: SLF001 - same module
            continue
        survivors.append(ref)
        live.append(profiler)
    _fork_registry[:] = survivors
    return live


def _relink_signal_chain(
    signum: signal.Signals,
    ours: _Handler,
    previous: _Handler,
    skip: Profiler,
) -> None:
    """Point whoever chained on top of ``ours`` at ``previous`` instead.

    Only rewrites the *restore target*. The successor's live handler closure still calls
    ``ours``, which is harmless — a closed profiler's ``close()`` returns immediately and the
    call passes straight through — but its eventual restore now skips the profiler that is
    going away.
    """
    for reference in _fork_registry:
        profiler = reference()
        if profiler is None or profiler is skip:
            continue
        chained = profiler._chained_signals.get(signum)  # noqa: SLF001 - same module
        if chained is not None and chained[1] is ours:
            profiler._chained_signals[signum] = (chained[0], previous)  # noqa: SLF001
            return


def _before_fork() -> None:
    """Stop every live profiler's threads before the process is copied."""
    for profiler in _forkable_profilers():
        profiler._pause_threads_before_fork()  # noqa: SLF001 - same module


def _after_fork_in_parent() -> None:
    """Restart the threads the fork paused."""
    for profiler in _forkable_profilers():
        profiler._resume_threads_after_fork()  # noqa: SLF001 - same module


def _after_fork_in_child() -> None:
    """Give each inherited profiler its own identity, files and threads."""
    for profiler in _forkable_profilers():
        profiler._reinitialise_after_fork()  # noqa: SLF001 - same module

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
        run_id: Identifies one attempt, so :func:`merge_run` groups workers from the same
            attempt and supersedes an earlier one instead of merging both into one inflated
            total. Defaults to ``LINEPROFILER_RUN_ID``, then to a fresh id. Pass this
            explicitly when several workers must share an id but cannot receive it through
            the environment — a ``forkserver`` daemon freezes its environment at start, so
            variables exported after that never reach children it forks.
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
        install: Register this as the process-global profiler, so the module-level
            :func:`phase`, :func:`count` and :func:`current` resolve to it. Without this,
            instrumenting a deep function means threading a ``profiler`` argument through
            every caller between it and wherever the object was constructed. ``close()``
            uninstalls, and a forked child resolves its own profiler rather than the
            parent's.
        strict_names: Raise when two phase names differ only in their digits, rather than
            warning once the tree is filling up. Turns "my phase vocabulary is fixed" into a
            guarantee the profiler checks, instead of something to pin in a test by hand.
        thread_names: Nest each thread's phases under its thread name, so two threads of one
            process doing unrelated work are reported separately. ``role`` is per process, and
            a learner taking gradient steps alongside a collector draining a queue is one
            process with two very different answers to "where did the time go?". Off by
            default: it changes the shape of the reported tree, and most processes have only
            one interesting thread.

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
        run_id: str | None = None,
        enabled: bool | None = None,
        snapshot_interval_s: float | None = 30.0,
        sample_interval_s: float | None = 1.0,
        measure_cpu: bool = True,
        backend: Backend | str | None = None,
        backend_window: tuple[int, int] | None = None,
        window_phase: str = "iteration",
        annotate: bool = False,
        install: bool = False,
        strict_names: bool = False,
        thread_names: bool = False,
    ) -> None:
        self.enabled: bool = _resolve_enabled(enabled)
        self.measure_cpu: bool = measure_cpu
        self.strict_names: bool = strict_names
        self.thread_names: bool = thread_names
        self.run_dir: Path = _resolve_run_dir(run_dir)
        self.run_id: str = run_id or os.environ.get(ENV_RUN_ID, "") or new_run_id()
        self.role: str = role or os.environ.get(ENV_ROLE, "") or "main"
        self.backend: Backend = Backend.parse(backend)

        if install:
            install_profiler(self)
        self._local = threading.local()
        self._trees: list[PhaseTree] = []
        self._states: list[_ThreadState] = []
        self._writer: SnapshotWriter | None = None
        self._sampler: ResourceSampler | None = None
        self._flush_timer: threading.Timer | None = None
        self._snapshot_interval_s = snapshot_interval_s
        self._closed = False
        self._snapshot_failures = 0
        self._delta_baseline: PhaseTree = {}
        self._name_shapes: dict[str, int] = {}
        self._chained_signals: dict[signal.Signals, tuple[_Handler, _Handler]] = {}
        self._snapshot_callbacks: list[Callable[[PhaseTree], None]] = []
        self._callback_failures = 0
        self._phase_overflow = 0
        self._window: BackendWindow | None = None
        self._process: ProcessHandle | None = None
        self._sample_interval_s = sample_interval_s
        self._nvtx: tuple[Callable[[str], object], Callable[[], object]] | None = None
        self._record_function: Callable[[str], AbstractContextManager[object]] | None = None
        self._cuda_sync: Callable[[], None] | None = None
        self._env_keys_propagated: list[str] = []

        if not self.enabled:
            return

        # Everything below costs something, so none of it runs for a disabled profiler:
        # open_process() constructs a psutil.Process and reads /proc, and used to run
        # unconditionally above this check.
        self._process = open_process()
        self._nvtx = nvtx_range_functions() if annotate else None
        self._record_function = record_function_factory() if annotate else None
        # Resolved once: torch.cuda.is_available() initialises the driver on first call, so
        # asking per phase would put a lock on the hot path.
        self._cuda_sync = cuda_synchronize()

        _warn_if_already_live(self.run_dir)
        self._env_keys_propagated = _propagate_to_children(self.run_dir, self.run_id)
        self._writer = SnapshotWriter(self.run_dir, role=self.role, run_id=self.run_id)
        self._start_backend_window(backend_window, window_phase)
        self._start_sampler(sample_interval_s)
        self._install_exit_hooks()
        self._start_flush_timer()
        _register_fork_hooks(self)

    # ── public API ──────────────────────────────────────────────────────────

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def phase(
        self,
        name: str,
        io: bool = False,
        sync: bool = False,
        sample: float = 1.0,
    ) -> _PhaseScope | _NullScope | _SuppressedScope:
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
            sample: Measure one entry in ``round(1/sample)`` and scale the result, for a
                region worth splitting but too hot to afford at full rate. ``1.0`` (the
                default) measures every entry.

                **Everything derived from a sampled phase is an estimate**, and is reported
                as one: the node carries its stride, the report marks the row, and merging a
                sampled node into a measured one marks the result too. That labelling is the
                condition on which this option exists — every other number in this layer is
                measured, and an estimate that cannot be told apart from a measurement is the
                failure mode the layer is built to avoid.

                Sampling a phase samples its **whole subtree**. A skipped entry records
                nothing for itself or anything beneath it, because counting children at full
                rate under a parent counted at one in ``n`` would mix two rates in one tree
                and produce a plausible wrong number rather than an obvious one.

                Selection is a deterministic stride, not a random draw — a draw costs about
                as much as the phase it is avoiding. The cost is aliasing: a workload whose
                period matches the stride keeps measuring the same point in it.
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
        if sample != 1.0:
            return self._sampled_phase(name, io, sync, sample)
        return _PhaseScope(self, name, io, sync)

    def _sampled_phase(
        self,
        name: str,
        io: bool,
        sync: bool,
        sample: float,
    ) -> _PhaseScope | _NullScope | _SuppressedScope:
        """Decide a sampled entry *before* allocating a scope for it.

        This is where the saving actually comes from. Constructing a ``_PhaseScope`` costs
        about a microsecond of the ~2.7 µs a phase takes — allocation plus thirteen slot
        stores — so deciding inside ``__enter__`` would have meant paying most of a phase's
        cost to skip it, and ``sample=0.01`` would have bought about 30%. Deciding here means
        a skipped entry never builds one.

        Kept off the default path: an unsampled ``phase()`` reaches this through one float
        comparison and never calls it.
        """
        stride = _stride_of(sample)
        state = self._thread_state()
        if state.suppressed:
            return _NULL_SCOPE  # already inside a skipped subtree: nothing to record or undo
        path = state.paths[-1] + (name,)
        seen = state.sampled.get(path, 0)
        state.sampled[path] = seen + 1
        if seen % stride:
            state.suppressed = True
            return state.suppressor
        return _PhaseScope(self, name, io, sync, stride)

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
        state = self._thread_state()
        if state.suppressed:
            return
        # scale is 1 unless a sampled phase is open, so the multiply is the whole cost.
        state.nodes[-1].add_count(name, n * state.scale)

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
        """Return the union of every thread's phase tree for this process.

        With ``thread_names=True`` each thread's phases are nested under its thread name, so
        two threads of one process doing unrelated work — a learner taking gradient steps and
        a collector draining a queue — stop being averaged into one set of numbers. The
        prefixing happens here rather than at phase entry, so it costs nothing on the hot path
        and does not change what a phase name means.
        """
        merged: PhaseTree = {}
        for state in list(self._states):
            prefix: PhasePath = (state.thread,) if self.thread_names else _ROOT
            for path, stats in list(state.tree.items()):
                key = prefix + path
                node = merged.get(key)
                if node is None:
                    merged[key] = stats.copy()
                else:
                    node.merge(stats)
        if self.thread_names:
            _fill_thread_totals(merged)
        return merged

    def deltas(self) -> PhaseTree:
        """Return the work recorded since the previous call, and advance the cursor.

        ``merged_tree()`` is cumulative, so exporting a per-interval figure to W&B or
        TensorBoard means keeping a copy of the last reading and subtracting — which every
        user of a long training run otherwise writes for themselves. The first call returns
        everything so far.

        Quantiles survive the subtraction: histograms are bucket counts, so the difference of
        two cumulative histograms is the histogram of the interval between them.

        Note that ``wait_ns`` pairs with ``wall_ns`` and never with ``self_ns`` — waiting
        inside a child counts towards the parent, so ``wait / self`` exceeds 100% for any
        phase wrapping a blocking call.

        Has its own cursor, independent of :meth:`on_snapshot`. Calling this *inside* an
        ``on_snapshot`` callback is the intended way to get per-interval numbers.

        Test specifically:
            - two successive calls sum to ``merged_tree()``
            - a phase that did no work in the interval is absent rather than present at zero
            - counters and histogram quantiles are per-interval, not cumulative
        """
        current = self.merged_tree()
        deltas: PhaseTree = {}
        for path, stats in current.items():
            baseline = self._delta_baseline.get(path)
            delta = stats if baseline is None else stats.difference(baseline)
            if delta.calls or delta.wall_ns or delta.counters:
                deltas[path] = delta.copy() if baseline is None else delta
        self._delta_baseline = current
        return deltas

    def on_snapshot(
        self,
        callback: Callable[[PhaseTree], None],
    ) -> Callable[[PhaseTree], None]:
        """Call ``callback`` with the cumulative tree after each *periodic* flush.

        Returns the callback, so it is usable as a decorator.

        For live export: a training run that already has W&B or TensorBoard usually wants the
        breakdown during the run, not only after it. Call :meth:`deltas` inside the callback
        for per-interval figures.

        Fires only from the background flush timer — deliberately not from ``close()`` or from
        a snapshot taken in a signal handler, where running arbitrary user code risks
        deadlocking the process on its own final flush. That keeps :meth:`snapshot`
        signal-safe, at the cost of the last partial interval.

        A callback that raises is counted and skipped, never propagated: an exporter losing
        its connection must not stop the flush timer, which is the defect that used to freeze
        a worker file for the rest of a run.

        Test specifically:
            - a raising callback does not stop later flushes or later callbacks
            - the callback sees a tree, and ``deltas()`` inside it sees the interval
            - it works as a decorator, leaving the decorated function callable
        """
        self._snapshot_callbacks.append(callback)
        return callback

    def close(self) -> None:
        """Write a final snapshot and stop the sampler, backend and flush threads.

        Note that ``os._exit()`` reaches neither this nor the signal handlers, and it is the
        ordinary way a multiprocessing entrypoint tears a worker down. Call this yourself on
        any path that ends the process that way, or everything since the last periodic flush
        is lost — leaving a file that parses and looks complete but stops early.

        Also undoes what an enabled profiler did to the process: the ``atexit`` registration
        and the signal handlers are removed, the fork callbacks — which CPython offers no way
        to unregister — go inert, and any of ``LINEPROFILER_PROFILE`` / ``_RUN_DIR`` /
        ``_RUN_ID`` this instance exported (not ones already set by the user or launcher) are
        unset, so a later ``Profiler()`` in the same process mints its own run id instead of
        silently joining this one's attempt. Closed is terminal; a closed profiler never
        writes again, not even in a child forked afterwards.

        Test specifically:
            - every chained signal is back to its previous disposition afterwards
            - a handler the host installed after the profiler is not clobbered
            - a fork after this writes nothing and starts no thread
            - environment variables this instance propagated are unset; ones it found already
              set are left alone
        """
        uninstall_profiler(self)
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
        # After the final snapshot: teardown must never cost the flush it exists to protect.
        self._restore_signals()
        atexit.unregister(self.close)
        _fork_registry[:] = [ref for ref in _fork_registry if ref() is not self]
        for key in self._env_keys_propagated:
            os.environ.pop(key, None)

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

    def _check_name_shape(self, path: PhasePath) -> None:
        """Notice phase names that vary per call, before the tree fills up because of them.

        ``count()`` raises on a float rather than truncating it. A name built from data is the
        more damaging mistake of the two and had no such protection: it degrades the report
        instead of raising, and only announces itself at ``MAX_PHASES``, by which point the
        run is already unreadable.

        One name in isolation says nothing — ``conv2d`` and ``resnet50`` are perfectly good
        names. What gives it away is *repetition of a shape*: ``episode_1`` and ``episode_2``
        share the shape ``episode_#``, and a hand-written vocabulary does not accumulate
        hundreds of those. So the check counts distinct names per shape rather than judging
        any name on its own.

        The counter is shared across threads and incremented without a lock: this is a
        heuristic whose worst failure under a race is warning slightly late.
        """
        shape = _NUMERIC_RUN.sub("#", path[-1])
        if shape == path[-1]:
            return
        seen = self._name_shapes.get(shape, 0) + 1
        self._name_shapes[shape] = seen
        if self.strict_names and seen > 1:
            raise ValueError(
                f"phase name {path[-1]!r} looks built from data: {seen} distinct names now "
                f"share the shape {shape!r}. Use a fixed name and count() for the varying "
                f"part, or pass strict_names=False.",
            )
        if seen == _NAME_SHAPE_WARN:
            warnings.warn(
                f"{seen} distinct phase names share the shape {shape!r} (most recently "
                f"{path[-1]!r}). Names built from data grow the phase tree until it folds at "
                f"{MAX_PHASES} paths and the report stops being readable — use a fixed name "
                f"and count() for the varying part.",
                RuntimeWarning,
                stacklevel=2,
            )

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
        if not self.enabled or self._closed:
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
            - a child forked after ``close()`` inherits nothing: no file, no thread

        ``os.register_at_fork`` callbacks cannot be unregistered — CPython offers no API — so
        a closed profiler cannot stop being called here and must instead refuse to act. This
        used to set ``_closed`` back to ``False`` unconditionally, which meant any fork after
        ``close()`` handed the child a live profiler: a new writer that re-created the run
        directory, a sampler and a flush timer, for a profiler the process had finished with.
        """
        if not self.enabled or self._closed:
            return
        self._local = threading.local()
        self._trees = []
        self._states = []
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
            # Kept so close() can put back what was here. Recorded only once the install
            # succeeded, so a signal we failed to take is never one we try to restore.
            self._chained_signals[signum] = (handler, previous)

    def _restore_signals(self) -> None:
        """Take this profiler out of the signal chain it joined.

        Handlers chain, so which profiler is currently installed depends on construction order
        and closing order, and the two need not match: closing a parent before the child it
        constructed is ordinary code. Two cases therefore:

        - **We are on top.** Put back what we replaced.
        - **Something chained above us.** Restoring would delete whatever is on top, so we
          leave the handler installed and instead hand our predecessor to whoever holds us as
          theirs. Our closure still calls through correctly in the meantime; when the profiler
          above us closes, it now restores past us to the right target. Without this splice a
          parent closed before its child stranded its handler in the process for good.

        A handler the *host* installed above us cannot be spliced, because we do not know what
        it will restore to. There we simply stay installed — the safe side of the trade, since
        our handler still chains correctly and closing has made it inert.
        """
        for signum, (ours, previous) in self._chained_signals.items():
            with contextlib.suppress(ValueError, OSError):
                if signal.getsignal(signum) is ours:
                    signal.signal(signum, previous)
                else:
                    _relink_signal_chain(signum, ours, previous, skip=self)
        self._chained_signals.clear()

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
            self._notify_snapshot_callbacks()
            self._start_flush_timer()

    def _notify_snapshot_callbacks(self) -> None:
        """Hand the current tree to each live-export callback, surviving any that raises.

        One exporter losing its connection must not cost the run its remaining flushes, nor
        stop the callbacks registered after it.
        """
        if not self._snapshot_callbacks:
            return
        tree = self.merged_tree()
        for callback in list(self._snapshot_callbacks):
            try:
                callback(tree)
            except Exception:  # noqa: BLE001 - an exporter must not stop the flush timer
                self._callback_failures += 1


def _fill_thread_totals(merged: PhaseTree) -> None:
    """Give each per-thread root the wall time of the phases beneath it.

    A thread's root node is never entered, so it carries no time of its own — and the report
    derives every share from the wall time of top-level nodes. Without this the thread level
    would render as a row of zeros and suppress the block entirely. ``child_wall_ns`` matches,
    so the synthesised node claims no self time it did not spend.
    """
    totals: dict[str, int] = {}
    for path, stats in merged.items():
        if len(path) == 2:
            totals[path[0]] = totals.get(path[0], 0) + stats.wall_ns
    for path, stats in merged.items():
        if len(path) == 1:
            stats.wall_ns = totals.get(path[0], 0)
            stats.child_wall_ns = stats.wall_ns


class _ThreadState:
    """One thread's open phase stack and its private slice of the phase tree."""

    __slots__ = (
        "names", "nodes", "paths", "sampled", "scale", "suppressed", "suppressor", "thread",
        "tree",
    )

    def __init__(self) -> None:
        root = PhaseStats()
        self.tree: PhaseTree = {_ROOT: root}
        self.names: list[str] = []
        self.paths: list[PhasePath] = [_ROOT]
        self.nodes: list[PhaseStats] = [root]
        # Read once, on this thread's first phase, never on the hot path.
        self.thread: str = threading.current_thread().name
        self.suppressed: bool = False
        """Set while inside a sampled phase whose entry was not selected. Everything beneath
        records nothing: sampling a phase samples its whole subtree, or children would be
        counted at full rate under a parent counted at one in ``n`` and the tree would mix two
        rates in one place.

        A flag rather than a depth, because only the *outermost* skipped phase owns it —
        everything nested inside gets the shared null scope and has nothing to unwind."""
        self.suppressor: _SuppressedScope = _SuppressedScope(self)
        """Reused for every skipped entry on this thread. Only one can be open at a time (the
        outermost), so one instance suffices, and skipping then allocates nothing at all."""
        self.scale: int = 1
        """Product of the strides of the sampled phases currently open, so ``count()`` scales
        the work it attributes by the same factor as the time around it."""
        self.sampled: dict[PhasePath, int] = {}
        """Entries seen per sampled path. A deterministic stride rather than a random draw:
        one increment and a compare, against ~50 ns for ``random()`` on the hot path."""


class _PhaseScope:
    """Context manager for one phase entry. Allocated per call, so it nests safely."""

    __slots__ = (
        "_cpu0", "_function", "_io", "_io0", "_name", "_profiler", "_self_io0", "_skipped",
        "_state", "_stats", "_stride", "_sync", "_wall0",
    )

    def __init__(
        self,
        profiler: Profiler,
        name: str,
        io: bool = False,
        sync: bool = False,
        stride: int = 1,
    ) -> None:
        self._profiler = profiler
        self._name = name
        self._io = io
        self._stride = stride
        self._skipped = False
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
        self._state = state
        # Inside a sampled phase that was not selected: record nothing, but keep the depth so
        # __exit__ unwinds symmetrically.
        if state.suppressed:
            self._skipped = True
            return
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
        if self._stride != 1:
            state.scale *= self._stride
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
        # Only reached once per distinct path, never on the hot path, so a regex here is free.
        self._profiler._check_name_shape(path)
        stats = PhaseStats()
        state.tree[path] = stats
        return stats

    def __exit__(self, *exc: object) -> None:
        if self._skipped:
            return  # the outermost skipped phase owns the flag and clears it
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
        # The product of every sampled phase currently open, this one included — not just
        # this phase's own stride. A child of a phase sampled at one in ten is itself only
        # entered on those ten, so scaling it by its own stride alone (1) under-reports it
        # tenfold and leaves two rates mixed in one tree.
        scale = state.scale
        if self._stride != 1:
            state.scale //= self._stride
        stats = self._stats
        parent = state.nodes[-1]
        if stats is None or stats is parent:
            return

        # PhaseStats.record and DurationHistogram.observe are inlined: at roughly a
        # microsecond per phase, two extra Python-level calls are a measurable share.
        stats.calls += scale
        stats.wall_ns += wall * scale
        stats.cpu_ns += cpu * scale
        histogram = stats.hist
        histogram.buckets[bucket_index(wall)] += scale
        histogram.count += scale
        parent.child_wall_ns += wall * scale
        if scale != 1:
            # Marks every derived figure on this node as an estimate. Set on exit rather than
            # at admission so a node only claims to be sampled once it actually is.
            stats.sample_stride = scale

        if self._io:
            self._record_io(stats, scale)

        window = self._profiler._window
        if window is not None:
            window.on_phase_exit(self._name)

    def _record_io(self, stats: PhaseStats, scale: int = 1) -> None:
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
        # Scaled by the same factor as the time around it. Bytes measured on one entry in n,
        # sitting beside a duration scaled to all n, would be two rates in one row.
        self._add_delta(stats, "io_read_bytes", (now.read_bytes - self._io0.read_bytes) * scale)
        self._add_delta(stats, "io_read_chars", (now.read_chars - self._io0.read_chars) * scale)
        self._add_delta(
            stats,
            "io_write_bytes",
            (now.write_bytes - self._io0.write_bytes - (blocks_after - blocks_before)) * scale,
        )
        self._add_delta(
            stats,
            "io_write_chars",
            (now.write_chars - self._io0.write_chars - (chars_after - chars_before)) * scale,
        )

    @staticmethod
    def _add_delta(stats: PhaseStats, name: str, delta: int) -> None:
        if delta > 0:
            stats.add_count(name, delta)


class _SuppressedScope:
    """Handed out for an entry inside a sampled phase that was not selected.

    Holds only the thread state, so that ``__exit__`` unwinds the suppression depth
    symmetrically. Sampling a phase samples its whole subtree: recording children at full rate
    beneath a parent recorded at one in ``n`` would leave two rates mixed in one tree.
    """

    __slots__ = ("_state",)

    def __init__(self, state: _ThreadState) -> None:
        self._state = state

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        self._state.suppressed = False


class _NullScope:
    """The context manager handed out when the profiler is disabled: no state, no clocks."""

    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


_NULL_SCOPE = _NullScope()


# ── the ambient profiler ────────────────────────────────────────────────────
#
# `phase()` being an instance method means instrumenting a deep function requires threading a
# profiler argument through every caller between it and wherever the object was constructed.
# In a real integration that is the search, the episode loop, the actor session and the
# inference server — for one phase. The alternative every integration writes for itself is a
# process-global shim, so it ships here instead, once and tested.

_installed: Profiler | None = None
"""The profiler ``phase()``/``count()`` below resolve to. Process-global on purpose: phase
*statistics* are per thread (see ``_ThreadState``), but which profiler is in use is a property
of the process, and a thread-local one would leave every thread a worker spawned uninstrumented.
"""


def install_profiler(profiler: Profiler) -> None:
    """Make ``profiler`` the one the module-level functions resolve to.

    Usually done at construction with ``Profiler(..., install=True)`` rather than called
    directly. ``close()`` uninstalls, so a closed profiler is never resolvable.
    """
    global _installed  # noqa: PLW0603 - the point of the function
    if _installed is not None and _installed is not profiler and not _installed._closed:  # noqa: SLF001
        warnings.warn(
            "an accounting Profiler is already installed; module-level phase() and count() "
            "will now resolve to the new one",
            RuntimeWarning,
            stacklevel=3,
        )
    _installed = profiler


def uninstall_profiler(profiler: Profiler | None = None) -> None:
    """Clear the installed profiler, if it is ``profiler`` (or unconditionally when ``None``)."""
    global _installed  # noqa: PLW0603 - the point of the function
    if profiler is None or _installed is profiler:
        _installed = None


def installed_profiler() -> Profiler | None:
    """Return the installed profiler, or ``None``. Useful for asserting in tests."""
    return _installed


def start(run_dir: str | Path | None = None, role: str | None = None, **kwargs: object) -> Profiler:
    """Construct, install and return a ``Profiler`` — the two-line alternative to ``with``.

        from lineprofiler.accounting import start, stop

        start(role="actor")     # top of the script
        ...
        stop()                  # bottom of the script

    Equivalent to ``Profiler(run_dir=run_dir, role=role, install=True, **kwargs)``. Respects
    the same ``LINEPROFILER_PROFILE``/``LINEPROFILER_ROLE``/``LINEPROFILER_RUN_DIR`` defaults
    as the constructor, so a disabled profiler is a near-free no-op and these two calls are
    safe to leave in place permanently.
    """
    return Profiler(run_dir=run_dir, role=role, install=True, **kwargs)  # type: ignore[arg-type]


def stop() -> None:
    """Close the currently installed profiler, if any. The counterpart to ``start()``."""
    profiler = _installed
    if profiler is not None:
        profiler.close()


def phase(
    name: str,
    io: bool = False,
    sync: bool = False,
    sample: float = 1.0,
) -> _PhaseScope | _NullScope | _SuppressedScope:
    """Open a phase on the installed profiler, or do nothing when there is none.

    The no-op path is one module-global load and one identity test, so leaving these calls in
    library code that is sometimes profiled and sometimes not costs nothing measurable — which
    is what makes it safe to instrument a function that does not know whether it is being
    profiled.

    Test specifically:
        - with nothing installed, this records nothing and allocates no state
        - a forked child resolves the child's profiler, never the parent's dead one
        - after ``close()`` this is a no-op again
    """
    profiler = _installed
    if profiler is None:
        return _NULL_SCOPE
    return profiler.phase(name, io, sync, sample)


def count(name: str, n: int = 1) -> None:
    """Attribute ``n`` work units on the installed profiler, or do nothing when there is none."""
    profiler = _installed
    if profiler is not None:
        profiler.count(name, n)


def current() -> str:
    """Return the deepest phase open on the installed profiler, or ``""`` when there is none.

    Handy for tagging a log line or an exception with whatever the process was doing.
    """
    profiler = _installed
    return profiler.current_phase() if profiler is not None else ""


def _stride_of(sample: float) -> int:
    """Turn a sampling fraction into the "one entry in n" stride, validating it here.

    Raised rather than clamped: ``sample=0`` means "measure nothing", which is a mistake a
    caller wants to hear about immediately, not a phase that silently never appears.
    """
    if not 0.0 < sample <= 1.0:
        raise ValueError(f"sample must be in (0.0, 1.0], got {sample!r}")
    return max(1, round(1.0 / sample))


def _resolve_enabled(enabled: bool | None) -> bool:
    """Resolve the master switch, reading the environment only when not given explicitly."""
    if enabled is not None:
        return enabled
    return _truthy(os.environ.get(ENV_ENABLE, ""))


def _resolve_run_dir(run_dir: str | Path | None) -> Path:
    """Use the caller's directory, else the one a parent process propagated, else the default.

    Always returned absolute, because the result is exported to children through
    ``LINEPROFILER_RUN_DIR`` and a relative path means a different directory in every process
    that has its own working directory. A batch system gives workers exactly that, so one run
    used to scatter its worker files across the filesystem and merge as several short runs —
    or land in whatever tree the job happened to start in.

    A relative path is resolved against the working directory of the process that constructed
    the profiler, which is what the caller meant by it. Deliberately *not* against
    ``$SLURM_SUBMIT_DIR``: schedulers and portals set that to whatever directory the job was
    launched from, which under Open OnDemand is the dashboard's own installation directory and
    typically not writable. Relocating the user's path there would trade one surprising
    location for a less predictable one.
    """
    given = Path(run_dir) if run_dir is not None else Path(
        os.environ.get(ENV_RUN_DIR, "") or "profile",
    )
    return given if given.is_absolute() else Path.cwd() / given


def _truthy(value: str) -> bool:
    return bool(value.strip()) and value not in {"0", "false", "False"}


def _propagate_to_children(run_dir: Path, run_id: str) -> list[str]:
    """Export the switch and run directory so child processes enable themselves.

    A worker started with ``spawn`` inherits the environment but not the parent's objects,
    so this is what lets ``Profiler(role="actor")`` in a worker join the parent's run
    without every call site having to thread the configuration through. Set explicitly by
    the user, either variable wins — this only fills in what is unset.

    ``forkserver`` is the exception: its daemon is forked once and its children inherit the
    daemon's environment, a snapshot taken when the daemon started. Variables exported after
    that never reach them. Under ``forkserver``, export ``LINEPROFILER_PROFILE`` in the shell
    before training starts, or pass ``enabled`` and ``run_dir`` to each worker explicitly.

    Returns the keys this call actually set (i.e. were previously unset), so ``close()`` can
    undo exactly those and leave anything the user or launcher had already exported alone —
    otherwise a long-lived process that opens and closes several profilers in turn (this test
    suite included) hands every later one the first one's ``run_id``, which defeats the
    "separate attempts" protection in :func:`_split_by_attempt` instead of providing it.
    """
    to_set = {ENV_ENABLE: "1", ENV_RUN_DIR: str(run_dir), ENV_RUN_ID: run_id}
    newly_set = [key for key in to_set if key not in os.environ]
    for key in newly_set:
        os.environ[key] = to_set[key]
    return newly_set


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


