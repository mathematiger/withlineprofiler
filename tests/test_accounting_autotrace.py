"""Function-derived spans: the tier that needs no ``phase()`` calls in the traced code.

The failure that matters most here is the one this hit in development: a filter that admits
the virtualenv fills the timeline with ``pynvml`` and ``psutil`` frames and omits the user's
own code entirely — a page that looks like it worked and answers nothing.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from lineprofiler.accounting import Profiler
from lineprofiler.accounting.autotrace import AutoTracer
from lineprofiler.accounting.snapshot import merge_run
from lineprofiler.accounting.trace import FLAG_AUTO, TraceBuffer

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="auto-tracing needs sys.monitoring, which arrived in 3.12",
)

_PROJECT = """
import time

def inner(n):
    return sum(i * i for i in range(n))

def outer():
    inner(500)

def entry():
    for _ in range(3):
        outer()
"""


@pytest.fixture
def project(tmp_path: Path) -> Iterator[Path]:
    """A tiny project with a ``.git`` marker and no instrumentation of its own."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "workload.py").write_text(_PROJECT, encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    try:
        yield tmp_path
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("workload", None)


def _tracer(
    project: Path,
    functions: list[str] | None = None,
) -> tuple[TraceBuffer, AutoTracer]:
    """A tracer scoped to the fixture's project, with its own buffer."""
    buffer = TraceBuffer(capacity=4096)
    tracer = AutoTracer(
        buffer=buffer,
        thread_id_of=lambda: 0,
        functions=functions,
        project_folder=project,
    )
    return buffer, tracer


def _names(buffer: TraceBuffer) -> set[str]:
    spans, _ = buffer.drain()
    paths = buffer.paths()
    return {paths[span.phase_id][-1] for span in spans}


def test_uninstrumented_code_produces_spans(project: Path) -> None:
    """The whole promise of this tier: no ``phase()`` calls, still a timeline."""
    import workload  # type: ignore[import-not-found]

    buffer, tracer = _tracer(project)
    tracer.start()
    try:
        workload.entry()
    finally:
        tracer.stop()

    assert {"entry", "outer", "inner"} <= _names(buffer)


def test_library_frames_never_appear(project: Path) -> None:
    """The bug that made the first version useless: a venv under the project root.

    ``json`` lives in the stdlib and ``pytest`` in site-packages; neither is the user's code,
    and admitting either buries the code they came to look at.
    """
    import json as stdlib_json

    import workload

    buffer, tracer = _tracer(project)
    tracer.start()
    try:
        stdlib_json.dumps({"a": 1})
        workload.entry()
    finally:
        tracer.stop()

    names = _names(buffer)

    assert "inner" in names
    assert not {name for name in names if "dump" in name or "encode" in name}


def test_the_profilers_own_code_is_not_traced(project: Path) -> None:
    """Tracing the sampler would fill the page with the machinery doing the measuring."""
    import workload

    buffer, tracer = _tracer(project)
    tracer.start()
    try:
        workload.entry()
    finally:
        tracer.stop()

    assert not {name for name in _names(buffer) if "Profiler" in name or "Sampler" in name}


def test_function_globs_narrow_what_is_recorded(project: Path) -> None:
    import workload

    buffer, tracer = _tracer(project, functions=["inner"])
    tracer.start()
    try:
        workload.entry()
    finally:
        tracer.stop()

    assert _names(buffer) == {"inner"}


def test_a_glob_matching_nothing_records_nothing(project: Path) -> None:
    import workload

    buffer, tracer = _tracer(project, functions=["nothing_called_this"])
    tracer.start()
    try:
        workload.entry()
    finally:
        tracer.stop()

    assert _names(buffer) == set()


def test_auto_spans_are_flagged_and_carry_no_cpu_time(project: Path) -> None:
    """They cannot afford a thread_time_ns per call, and must not pretend otherwise."""
    import workload

    buffer, tracer = _tracer(project)
    tracer.start()
    try:
        workload.entry()
    finally:
        tracer.stop()

    spans, _ = buffer.drain()

    assert spans
    assert all(span.flags & FLAG_AUTO for span in spans)
    assert not any(span.cpu_measured for span in spans)


def test_a_second_tracer_in_one_process_still_records(project: Path) -> None:
    """``DISABLE`` is permanent per code object, so a second window needs restart_events().

    Without it the second run reports a confident empty timeline for code that definitely
    ran — the exact wrong-numbers shape this package exists to avoid.
    """
    import workload

    first_buffer, first = _tracer(project)
    first.start()
    try:
        workload.entry()
    finally:
        first.stop()
    assert _names(first_buffer)

    second_buffer, second = _tracer(project)
    second.start()
    try:
        workload.entry()
    finally:
        second.stop()

    assert "inner" in _names(second_buffer)


def test_the_slot_is_released_so_another_tracer_can_claim_it(project: Path) -> None:
    _, first = _tracer(project)
    first.start()
    first.stop()

    _, second = _tracer(project)
    second.start()
    second.stop()


def test_a_second_concurrent_tracer_is_refused(project: Path) -> None:
    """Two tracers on one slot would silently evict each other; say so instead."""
    _, first = _tracer(project)
    _, second = _tracer(project)
    first.start()
    try:
        with pytest.raises(RuntimeError, match="slot"):
            second.start()
    finally:
        first.stop()


def test_a_raising_function_still_records_its_span(project: Path) -> None:
    """PY_UNWIND is bound too: a hole in the timeline where an error happened is the worst
    possible place to have one."""
    (project / "boom.py").write_text(
        "def explode():\n    raise ValueError('boom')\n",
        encoding="utf-8",
    )
    import boom  # type: ignore[import-not-found]

    buffer = TraceBuffer(capacity=64)
    tracer = AutoTracer(buffer=buffer, thread_id_of=lambda: 0, project_folder=project)
    tracer.start()
    try:
        with pytest.raises(ValueError, match="boom"):
            boom.explode()
    finally:
        tracer.stop()
        sys.modules.pop("boom", None)

    assert "explode" in _names(buffer)


def test_the_profiler_wires_auto_tracing_end_to_end(project: Path, tmp_path: Path) -> None:
    """``trace="auto"`` on the profiler, which is how a user actually reaches this."""
    import workload

    run_dir = tmp_path / "profile"
    profiler = Profiler(
        run_dir=run_dir, role="main", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None, trace="auto",
    )
    # The profiler detected its own project root — this file's repo — which is not where the
    # fixture's workload lives. Swap in a tracer scoped to the fixture, releasing the slot
    # the profiler's own tracer holds first.
    profiler._stop_auto_tracing()  # noqa: SLF001 - frees the monitoring slot
    tracer = AutoTracer(
        buffer=profiler._trace,  # noqa: SLF001 - the buffer under test
        thread_id_of=profiler._trace_thread_id,  # noqa: SLF001
        project_folder=project,
    )
    tracer.start()
    profiler._auto = tracer  # noqa: SLF001 - so close() stops it
    with profiler:
        workload.entry()

    trace = merge_run(run_dir, with_trace=True).workers[0].trace
    names = {trace.path_of(span.phase_id)[-1] for span in trace.spans}

    assert {"entry", "outer", "inner"} <= names


def test_auto_spans_carry_the_file_function_and_line_they_came_from(project: Path) -> None:
    """The point of the tier: a span you can trace back to source without instrumenting it.

    Line numbers are pinned against the fixture's own text rather than hard-coded, so the
    test says "where the function is defined" instead of restating a constant.
    """
    import workload

    buffer, tracer = _tracer(project)
    tracer.start()
    try:
        workload.entry()
    finally:
        tracer.stop()

    spans, _ = buffer.drain()
    paths = buffer.paths()
    origins = buffer.origins()
    by_name = {
        paths[span.phase_id][-1]: origins.get(span.phase_id)
        for span in spans
    }

    source = (project / "workload.py").read_text(encoding="utf-8").splitlines()
    expected_line = source.index("def inner(n):") + 1

    inner = by_name["inner"]
    assert inner is not None
    assert inner.function == "inner"
    assert Path(inner.file) == project / "workload.py"
    assert inner.line == expected_line


def test_origins_survive_the_sidecar_round_trip(project: Path, tmp_path: Path) -> None:
    """A location held only in memory helps nobody: the page reads it back off disk."""
    import workload

    run_dir = tmp_path / "profile"
    profiler = Profiler(
        run_dir=run_dir, role="main", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None, trace="auto",
    )
    profiler._stop_auto_tracing()  # noqa: SLF001 - frees the monitoring slot
    tracer = AutoTracer(
        buffer=profiler._trace,  # noqa: SLF001 - the buffer under test
        thread_id_of=profiler._trace_thread_id,  # noqa: SLF001
        project_folder=project,
    )
    tracer.start()
    profiler._auto = tracer  # noqa: SLF001 - so close() stops it
    with profiler:
        workload.entry()

    trace = merge_run(run_dir, with_trace=True).workers[0].trace
    located = {
        trace.path_of(span.phase_id)[-1]: trace.origin_of(span.phase_id)
        for span in trace.spans
    }

    assert {"entry", "outer", "inner"} <= set(located)
    for name in ("entry", "outer", "inner"):
        origin = located[name]
        assert origin is not None, f"{name} lost its origin crossing the sidecar"
        assert origin.function == name
        assert Path(origin.file) == project / "workload.py"
        assert origin.line > 0


def test_a_named_phase_has_no_origin_rather_than_a_fabricated_one(tmp_path: Path) -> None:
    """There is no code object behind a name, and inventing one would point at a wrong file.

    The absence is the correct answer here, so it is asserted as an absence.
    """
    run_dir = tmp_path / "profile"
    profiler = Profiler(
        run_dir=run_dir, role="main", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None, trace=True,
    )
    with profiler, profiler.phase("iteration"):
        pass

    trace = merge_run(run_dir, with_trace=True).workers[0].trace

    assert trace.spans
    assert all(trace.origin_of(span.phase_id) is None for span in trace.spans)
