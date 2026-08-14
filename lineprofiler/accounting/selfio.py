"""Bytes this profiler wrote on its own behalf, so they can be kept out of your phases.

The sampler thread appends a JSONL row every interval and the snapshot writer replaces the
worker file on every flush. Both go through the same per-process byte counters that an
``io=True`` phase reads, so without this bookkeeping the profiler bills its own overhead to
whichever phase happened to be open — a phase that touched no file at all reports I/O.

Two totals are kept, because the two layers cost different amounts. ``chars`` is what the
write syscalls carried and is known exactly from the buffer length. ``block_bytes`` is what
reached the device, which is *not* derivable from it: rewriting a 500-byte worker file costs
whole blocks for data, inode, directory entry and journal, so a 14 KB run of bookkeeping was
measured at 116 KB of block traffic. That figure is therefore measured by bracketing each of
the profiler's own writes rather than estimated from a block size.

Both totals are process-wide and monotonic; a phase excludes only what was written between
its own entry and exit.
"""

from __future__ import annotations

import threading

_lock = threading.RLock()
"""Reentrant on purpose. A signal handler runs on the main thread between bytecodes, and the
main thread holds this lock at both ends of every ``io=True`` phase. With a plain ``Lock`` a
``SIGTERM`` arriving inside that window deadlocked the process permanently on the profiler's
own final flush — which reads as "the run hung on shutdown and lost its last snapshot", a
symptom nobody would trace back to here. Re-entry can let a handler's write land inside an
enclosing read; the cost is a few bytes of overhead accounting, against a hang."""

_chars = 0
_block_bytes = 0


def record_bytes_written(chars: int, block_bytes: int = 0) -> None:
    """Add one of the profiler's own writes to the overhead totals.

    Args:
        chars: Length of the buffer handed to the write syscall.
        block_bytes: Block-layer traffic observed across that write, from bracketing the
            process counters. Zero when the platform exposes no such counter.

    Called from the sampler and snapshot threads, never from the phase hot path.
    """
    global _chars, _block_bytes
    with _lock:
        _chars += chars
        _block_bytes += block_bytes


def bytes_written() -> tuple[int, int]:
    """Return ``(chars, block_bytes)`` written by this process's profiler for bookkeeping."""
    with _lock:
        return (_chars, _block_bytes)


def reset() -> None:
    """Forget the running totals. Used after a fork, where the child starts its own files."""
    global _chars, _block_bytes
    with _lock:
        _chars = 0
        _block_bytes = 0
