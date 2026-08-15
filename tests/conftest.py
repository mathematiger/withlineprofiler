"""Suite-wide guards against tests leaking process-global state into each other.

The accounting profiler installs signal handlers, an ``atexit`` hook and fork callbacks, and
this suite constructs well over a hundred enabled profilers in-process. When ``close()`` did
not undo any of that, the damage showed up nowhere near its cause: forked children could no
longer be terminated, and the failures landed in an unrelated file several hundred tests away,
reproducing only in a full run. A leak that is invisible in the file that causes it needs a
check that spans the whole session.
"""

from __future__ import annotations

import signal
from collections.abc import Iterator

import pytest

from lineprofiler import config as _config_module
from lineprofiler import profiler as _line_profiler_module
from lineprofiler.accounting import profiler as _profiler_module

_WATCHED_SIGNALS = (signal.SIGTERM, signal.SIGUSR1, signal.SIGHUP)


@pytest.fixture(autouse=True)
def _reset_ambient_line_profiler() -> Iterator[None]:
    """Undo ``start_profiling()``/config-cache state a test left behind.

    Both are process-global module state, same rationale as the fork registry cleanup below:
    a test that starts ambient profiling and forgets to stop it would otherwise hand the next
    test a live ``sys.settrace`` tracer.
    """
    yield
    if _line_profiler_module._installed is not None:  # noqa: SLF001 - hygiene
        _line_profiler_module.stop_profiling(print_stats=False)
    _config_module._cache.clear()  # noqa: SLF001 - hygiene


@pytest.fixture(autouse=True)
def _close_profilers_left_open() -> Iterator[None]:
    """Close any enabled profiler a test constructed and did not close.

    Sixteen tests here build one to assert on ``merged_tree()`` and never close it, which is
    the ordinary way to write them — but an *open* profiler owns the process's signal handlers,
    so leaving one behind hands the next test an interpreter it did not ask for. Rather than
    thread a ``try/finally`` through all sixteen, the fixture closes whatever is still open,
    and the session-scoped check below confirms that it worked.

    The registry is weak, so this sees only profilers that are still alive, and closing one
    twice is a no-op. Closing happens in reverse construction order because the handlers form
    a chain: the newest is the one currently installed, and an older profiler only becomes the
    top of the chain once everything after it has stepped down.
    """
    yield
    for reference in reversed(list(_profiler_module._fork_registry)):  # noqa: SLF001 - hygiene
        profiler = reference()
        if profiler is not None:
            profiler.close()


@pytest.fixture(scope="session", autouse=True)
def _no_leaked_signal_handlers() -> Iterator[None]:
    """Fail the session if any test left a signal handler installed.

    Session-scoped on purpose: the accumulation is what does the damage, and a per-test check
    would pass for a suite that leaks one handler per test.
    """
    before = {signum: signal.getsignal(signum) for signum in _WATCHED_SIGNALS}
    yield
    leaked = {
        signum.name: signal.getsignal(signum)
        for signum in _WATCHED_SIGNALS
        if signal.getsignal(signum) is not before[signum]
    }
    assert not leaked, (
        f"tests left signal handlers installed: {leaked}. An enabled Profiler chains "
        f"SIGTERM/SIGUSR1/SIGHUP and close() is what puts them back — a test that constructs "
        f"one without closing it alters the interpreter for every test that follows."
    )
