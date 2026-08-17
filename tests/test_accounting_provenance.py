"""Tests for source provenance and the render-scaling controls.

Both guard the same class of loss as the rest of this suite, at two different scales. A trace
that does not name the code it measured invites a conclusion about a program that no longer
exists — the profiled run executed the committed code while the tree being read had already
fixed the constraint the profile found. And a render that fails after the profiled run has
succeeded discards the expensive half of the work to save the cheap half.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from lineprofiler.accounting import Profiler, merge_run, render
from lineprofiler.accounting.cli import main as cli_main
from lineprofiler.accounting.htmltrace import _spans_to_draw, render_trace_html
from lineprofiler.accounting.provenance import describe_source, format_source, source_of
from lineprofiler.accounting.snapshot import new_run_id
from lineprofiler.accounting.tracealign import PlacedSpan, align_run, lifecycle_segments

# ── reading the revision ────────────────────────────────────────────────────


def _git_repo(path: Path) -> None:
    """Create a repository with one commit, or skip when git is unavailable."""
    try:
        subprocess.run(["git", "init", "-q"], cwd=path, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - depends on the machine
        pytest.skip("git is not available")
    for key, value in (("user.email", "t@example.com"), ("user.name", "t")):
        subprocess.run(["git", "config", key, value], cwd=path, check=True, timeout=10)
    (path / "code.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True, timeout=10)
    subprocess.run(
        ["git", "commit", "-q", "-m", "first", "--no-gpg-sign"],
        cwd=path, check=True, timeout=10,
    )


def test_a_directory_that_is_not_a_repository_yields_nothing(tmp_path: Path) -> None:
    """Absent, never guessed: a run with no revision must not be given one."""
    assert describe_source(tmp_path) == {}
    assert format_source({}) == ""


def test_a_clean_checkout_reports_a_commit_and_no_dirty_marker(tmp_path: Path) -> None:
    _git_repo(tmp_path)

    source = describe_source(tmp_path)

    assert len(str(source["commit"])) == 40
    assert "dirty_files" not in source
    assert format_source(source).startswith("Source ")
    assert "dirty" not in format_source(source)


def test_a_modified_tree_is_marked_dirty_with_a_count_and_a_hash(tmp_path: Path) -> None:
    """The case that bit the original investigation: the fix was in the tree, not the run."""
    _git_repo(tmp_path)
    (tmp_path / "code.py").write_text("x = 2\n", encoding="utf-8")

    source = describe_source(tmp_path)

    assert source["dirty_files"] == 1
    assert len(str(source["diff_sha"])) == 6
    assert "+dirty: 1 files" in format_source(source)


def test_two_different_diffs_hash_differently(tmp_path: Path) -> None:
    """Otherwise every dirty run of one commit is indistinguishable from every other."""
    _git_repo(tmp_path)
    (tmp_path / "code.py").write_text("x = 2\n", encoding="utf-8")
    first = describe_source(tmp_path)["diff_sha"]
    (tmp_path / "code.py").write_text("x = 3\n", encoding="utf-8")
    second = describe_source(tmp_path)["diff_sha"]

    assert first != second


def test_a_missing_git_binary_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provenance is a courtesy; a run must never fail because it could not be established."""
    monkeypatch.setenv("PATH", str(tmp_path))

    assert describe_source(tmp_path) == {}


# ── the revision reaching the report ────────────────────────────────────────


def test_the_report_header_names_the_revision(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
        source={"commit": "c49ce8412345", "dirty_files": 26, "diff_sha": "3f9a1c"},
    )
    with profiler, profiler.phase("step"):
        pass

    text = render(merge_run(tmp_path))

    assert "Source c49ce841 (+dirty: 26 files, diff sha 3f9a1c)" in text


def test_a_run_without_a_recorded_revision_says_nothing(tmp_path: Path) -> None:
    """A 0.6.0 run directory has no source key, and must render without inventing one."""
    assert source_of({"run_id": "x"}) == ""
    assert source_of({"source": "not-a-dict"}) == ""


def test_the_caller_can_supply_its_own_revision(tmp_path: Path) -> None:
    """An embedding program may know better — or want a config hash instead."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None, sample_interval_s=None,
        source={"commit": "deadbeefcafe"},
    )
    with profiler, profiler.phase("step"):
        pass

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["source"] == {"commit": "deadbeefcafe"}


# ── rendering a trace that is too large ─────────────────────────────────────


def _span(at: int) -> PlacedSpan:
    return PlacedSpan(
        worker="w", role="main", thread_id=0, path=("p",),
        t0_ns=at, t1_ns=at + 10, cpu_ns=-1, flags=0,
    )


def test_the_span_cap_keeps_the_longest_and_counts_the_rest() -> None:
    spans = [_span(index * 100) for index in range(10)]

    kept, omitted = _spans_to_draw(spans, max_spans=4)

    assert len(kept) == 4
    assert omitted == 6
    assert kept == sorted(kept, key=lambda span: span.t0_ns), "drawn in time order"


def test_no_cap_applies_below_the_limit() -> None:
    spans = [_span(index) for index in range(3)]
    assert _spans_to_draw(spans, max_spans=100) == (spans, 0)


def test_omitted_spans_are_stated_on_the_page(tmp_path: Path) -> None:
    """The count was always computed and embedded, and never reached the reader."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None,
        sample_interval_s=None, trace=True,
    )
    with profiler:
        for _ in range(20):
            with profiler.phase("step"):
                pass

    page = render_trace_html(merge_run(tmp_path, with_trace=True), max_spans=5)

    assert "span(s) are not drawn" in page
    assert "--max-spans" in page


def test_the_trace_command_reports_progress_to_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A long render and a hung one are indistinguishable without this."""
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None,
        sample_interval_s=None, trace=True,
    )
    with profiler, profiler.phase("step"):
        pass

    assert cli_main(["trace", str(tmp_path), "-o", str(tmp_path / "t.html")]) == 0
    captured = capsys.readouterr()

    assert "loading workers" in captured.err
    assert "aligned" in captured.err
    assert captured.out == "", "progress must not contaminate the rendered page"


def test_the_trace_command_can_be_silenced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    profiler = Profiler(
        run_dir=tmp_path, enabled=True, snapshot_interval_s=None,
        sample_interval_s=None, trace=True,
    )
    with profiler, profiler.phase("step"):
        pass

    assert cli_main(["trace", str(tmp_path), "-q", "-o", str(tmp_path / "t.html")]) == 0

    assert capsys.readouterr().err == ""


# ── stating what the percentages are of ─────────────────────────────────────


def test_the_role_block_names_its_denominator_and_marks_summed_columns(tmp_path: Path) -> None:
    """Four denominators are plausible here, and they read materially differently.

    A total larger than the run's own runtime is correct for a multi-process role and reads
    as an error until the summing is stated.
    """
    run_id = new_run_id()
    for _ in range(2):
        profiler = Profiler(
            run_dir=tmp_path, role="actor", run_id=run_id, enabled=True,
            snapshot_interval_s=None, sample_interval_s=None,
        )
        with profiler:
            with profiler.phase("mcts"):
                pass
            with profiler.phase("step"):
                pass

    text = render(merge_run(tmp_path))

    assert "% of phase wall time at the first branching level, summed over 2 processes" in text
    assert "(Σ2 proc)" in text


def test_a_single_process_role_does_not_claim_to_be_summed(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, role="learner", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler:
        with profiler.phase("train"):
            pass
        with profiler.phase("eval"):
            pass

    text = render(merge_run(tmp_path))

    assert "% of phase wall time at the first branching level" in text
    assert "summed over" not in text
    assert "Σ" not in text


def test_busy_and_working_reach_the_text_report(tmp_path: Path) -> None:
    """"Busy 97% but working 35%" is the bottleneck statement, and it was HTML-only."""
    profiler = Profiler(
        run_dir=tmp_path, role="actor", enabled=True, snapshot_interval_s=None,
        sample_interval_s=None, trace=True,
    )
    with profiler:
        for _ in range(5):
            with profiler.phase("blocked"):
                time.sleep(0.01)
        profiler.snapshot()

    text = render(merge_run(tmp_path, with_trace=True))

    assert "busy (phase open)" in text
    assert "working (on CPU)" in text
    assert "The gap is waiting." in text


def test_a_run_without_a_trace_reports_no_occupancy(tmp_path: Path) -> None:
    """Tracing is off by default; a role with no timeline must not be given a figure."""
    profiler = Profiler(
        run_dir=tmp_path, role="actor", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler, profiler.phase("step"):
        pass

    text = render(merge_run(tmp_path))

    assert "busy (phase open)" not in text


# ── the request lifecycle, end to end ───────────────────────────────────────


def test_a_two_process_lifecycle_attributes_each_segment_to_the_right_stage(
    tmp_path: Path,
) -> None:
    """The gap's own acceptance test, through the real API and the real report.

    A server that sleeps a known 50 ms before admitting and 20 ms computing must show ≈50 ms
    against the batching segment and ≈20 ms against compute — which is the reading that
    distinguishes "batch harder" from "buy a faster GPU".
    """
    import queue
    import threading

    run_id = new_run_id()
    requests: queue.Queue[int] = queue.Queue()
    replies: queue.Queue[int] = queue.Queue()

    def server() -> None:
        profiler = Profiler(
            run_dir=tmp_path, role="server", run_id=run_id, enabled=True,
            snapshot_interval_s=None, sample_interval_s=None, trace=True,
        )
        with profiler:
            for _ in range(4):
                key = requests.get()
                time.sleep(0.05)
                profiler.trace_mark("inference", key, "admitted")
                time.sleep(0.02)
                profiler.trace_mark("inference", key, "computed")
                replies.put(key)
            profiler.snapshot()

    def client() -> None:
        profiler = Profiler(
            run_dir=tmp_path, role="actor", run_id=run_id, enabled=True,
            snapshot_interval_s=None, sample_interval_s=None, trace=True,
        )
        with profiler:
            for index in range(4):
                profiler.trace_begin("inference", index)
                requests.put(index)
                with profiler.phase("queue_wait"):
                    replies.get()
                profiler.trace_end("inference", index)
            profiler.snapshot()

    threads = [threading.Thread(target=server), threading.Thread(target=client)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    aligned = align_run(merge_run(tmp_path, with_trace=True))
    segments = {s.name: s for s in lifecycle_segments(aligned)["inference"]}

    assert segments["begin → admitted"].mean_ns == pytest.approx(50_000_000, rel=0.4)
    assert segments["admitted → computed"].mean_ns == pytest.approx(20_000_000, rel=0.4)
    assert segments["begin → admitted"].count == 4


def test_the_lifecycle_block_reaches_the_report(tmp_path: Path) -> None:
    profiler = Profiler(
        run_dir=tmp_path, role="actor", enabled=True, snapshot_interval_s=None,
        sample_interval_s=None, trace=True,
    )
    with profiler:
        for index in range(3):
            profiler.trace_begin("inference", index)
            time.sleep(0.005)
            profiler.trace_mark("inference", index, "admitted")
            profiler.trace_end("inference", index)
        profiler.snapshot()

    text = render(merge_run(tmp_path, with_trace=True))

    assert "REQUEST LIFECYCLE" in text
    assert "begin → admitted" in text
    assert "(3 req)" in text


def test_a_run_with_no_lifecycle_marks_has_no_lifecycle_block(tmp_path: Path) -> None:
    """The block must be absent, not empty: an empty table reads as a measurement of zero."""
    profiler = Profiler(
        run_dir=tmp_path, role="actor", enabled=True, snapshot_interval_s=None,
        sample_interval_s=None, trace=True,
    )
    with profiler, profiler.phase("step"):
        pass

    assert "REQUEST LIFECYCLE" not in render(merge_run(tmp_path, with_trace=True))


def test_the_ambient_lifecycle_functions_no_op_without_a_profiler() -> None:
    """Safe to leave in library code that is sometimes profiled and sometimes not."""
    from lineprofiler.accounting import trace_begin, trace_end, trace_mark

    trace_begin("inference", 1)
    trace_mark("inference", 1, "admitted")
    trace_end("inference", 1)


def test_the_report_command_reads_traces_when_they_exist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The occupancy, concurrency and lifecycle blocks all derive from the trace.

    ``merge_run`` defaults ``with_trace=False``, so a run recorded with ``trace=True``
    rendered without the three blocks it was instrumented for.
    """
    profiler = Profiler(
        run_dir=tmp_path, role="actor", enabled=True, snapshot_interval_s=None,
        sample_interval_s=None, trace=True,
    )
    with profiler:
        for index in range(3):
            profiler.trace_begin("inference", index)
            time.sleep(0.005)
            profiler.trace_end("inference", index)
        profiler.snapshot()

    assert cli_main(["report", str(tmp_path)]) == 0

    assert "REQUEST LIFECYCLE" in capsys.readouterr().out
