"""Tests for the trace timeline page.

This is the one page in the package that ships JavaScript, so the constraint it must satisfy
is stated differently from the other two: still no network of any kind, but an inline
``<script>`` is permitted and expected. The report and source pages keep their stricter rule,
asserted in :mod:`test_html`.

As there, the assertions are about what would make the page *wrong* rather than about its
styling: that it opens offline, that a truncated trace says so, that unmeasured CPU time is
not drawn as "never waited", and that every element the script reaches for exists.
"""
from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from lineprofiler.accounting import Profiler
from lineprofiler.accounting.cli import main
from lineprofiler.accounting.htmltrace import render_trace_html, write_trace_html
from lineprofiler.accounting.snapshot import merge_run

_VOID = {
    "meta", "br", "hr", "img", "input", "link", "area",
    "base", "col", "embed", "source", "track", "wbr",
}


def _traced_run(run_dir: Path, *, capacity: int = 200_000, iterations: int = 3) -> None:
    """One worker with nested phases and a matched signal/wait pair per iteration."""
    profiler = Profiler(
        run_dir=run_dir,
        role="actor",
        enabled=True,
        snapshot_interval_s=None,
        sample_interval_s=None,
        trace=True,
        trace_capacity=capacity,
    )
    with profiler:
        for step in range(iterations):
            with profiler.phase("iteration"), profiler.phase("work"):
                pass
            profiler.signal("batch", step)
            profiler.wait_on("batch", step)


def _render(run_dir: Path) -> str:
    return render_trace_html(merge_run(run_dir, with_trace=True))


def _embedded(html: str) -> dict[str, Any]:
    match = re.search(
        r'<script type="application/json" id="lineprofiler-data">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None, "the page carries no data block"
    payload: dict[str, Any] = json.loads(match.group(1).replace("<\\/", "</"))
    return payload


class _Structure(HTMLParser):
    """Collects unbalanced tags and every ``id`` the document defines."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        found = dict(attrs)
        if "id" in found and found["id"]:
            self.ids.add(found["id"])
        if tag not in _VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return
        if not self.stack:
            self.errors.append(f"</{tag}> with nothing open")
        elif self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes <{self.stack[-1]}>")
        else:
            self.stack.pop()


# --------------------------------------------------------------------------- #
# Self-containment
# --------------------------------------------------------------------------- #
def test_the_trace_page_needs_no_network(tmp_path: Path) -> None:
    """Script is allowed here; a network request is not, and never becomes allowed.

    Matched against real tags and attributes rather than bare substrings: the page's own
    prose legitimately discusses markup, and a check that cannot tell a comment from an
    element fails on documentation instead of on a regression.
    """
    _traced_run(tmp_path)
    html = _render(tmp_path)

    assert "http://" not in html
    assert "https://" not in html
    assert not re.search(r"<link\b", html)
    assert not re.search(r"<\s*(script|img|iframe|embed)\b[^>]*\bsrc\s*=", html)
    assert "@import" not in html


def test_the_trace_page_carries_its_script_inline(tmp_path: Path) -> None:
    """Pan and zoom need script; the point is that it travels *inside* the file."""
    _traced_run(tmp_path)
    html = _render(tmp_path)

    assert "<script>" in html
    assert "getElementById" in html


def test_the_page_is_well_formed_and_every_scripted_element_exists(tmp_path: Path) -> None:
    """A script reaching for an id the document never defines is a silently blank chart."""
    _traced_run(tmp_path)
    html = _render(tmp_path)

    structure = _Structure()
    structure.feed(html)
    script = re.search(r"<script>(.*?)</script>", html, re.DOTALL)
    assert script is not None
    referenced = set(re.findall(r"getElementById\('([^']+)'\)", script.group(1)))

    assert structure.stack == []
    assert structure.errors == []
    assert referenced <= structure.ids


def test_the_inline_script_is_balanced(tmp_path: Path) -> None:
    """A truncated or mis-quoted script would leave the page inert with no error."""
    _traced_run(tmp_path)
    script = re.search(r"<script>(.*?)</script>", _render(tmp_path), re.DOTALL)
    assert script is not None
    body = script.group(1)

    for opener, closer in (("{", "}"), ("(", ")"), ("[", "]")):
        assert body.count(opener) == body.count(closer), f"unbalanced {opener}{closer}"


# --------------------------------------------------------------------------- #
# What the page says
# --------------------------------------------------------------------------- #
def test_the_page_reports_the_lanes_and_spans_it_drew(tmp_path: Path) -> None:
    _traced_run(tmp_path, iterations=4)
    data = _embedded(_render(tmp_path))

    assert data["lanes"]
    assert data["spans"]
    assert data["omitted"] == 0


def test_matched_links_become_arrows_in_the_payload(tmp_path: Path) -> None:
    _traced_run(tmp_path, iterations=3)

    assert _embedded(_render(tmp_path))["arrows"]


def test_a_wrapped_buffer_says_so_on_the_page(tmp_path: Path) -> None:
    """A truncated trace that renders as a complete one is the failure to avoid."""
    _traced_run(tmp_path, capacity=4, iterations=40)
    html = _render(tmp_path)

    assert "dropped" in html
    assert "not the whole run" in html


def test_an_untraced_run_explains_how_to_record_one(tmp_path: Path) -> None:
    """Someone discovering the feature gets instructions, not an exception or a blank page."""
    profiler = Profiler(
        run_dir=tmp_path, role="main", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None,
    )
    with profiler, profiler.phase("iteration"):
        pass

    html = _render(tmp_path)

    assert "no trace data" in html
    assert "LINEPROFILER_TRACE" in html


def test_the_page_names_the_clock_accuracy(tmp_path: Path) -> None:
    """Cross-host alignment is NTP-bounded and a reader must not have to guess."""
    _traced_run(tmp_path)

    assert "exact" in _render(tmp_path)


def test_a_phase_named_like_a_script_tag_cannot_escape_its_block(tmp_path: Path) -> None:
    """Phase names come from user code, so this is reachable input on this page too.

    The name legitimately appears *inside* the JSON block — that is the record of what ran —
    but ``</`` is escaped there so the block cannot be closed early, and the document must
    gain no live element from it.
    """
    profiler = Profiler(
        run_dir=tmp_path, role="main", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None, trace=True,
    )
    with profiler, profiler.phase("</script><img src=x onerror=alert(1)>"):
        pass

    html = _render(tmp_path)
    structure = _Structure()
    structure.feed(html)

    assert "</script><img" not in html  # the block cannot be closed early
    assert structure.errors == []
    assert "onerror=alert(1)>" in json.dumps(_embedded(html))  # preserved as data


def test_user_text_never_reaches_the_page_as_markup(tmp_path: Path) -> None:
    """Tooltips are built with textContent, never innerHTML.

    A profiling artifact gets mailed around and opened by other people, so a phase name that
    executes in their browser is a real defect rather than a theoretical one.
    """
    _traced_run(tmp_path)
    script = re.search(r"<script>(.*?)</script>", _render(tmp_path), re.DOTALL)
    assert script is not None

    assert "innerHTML" not in script.group(1)


def test_the_lane_table_separates_occupied_from_working(tmp_path: Path) -> None:
    """The gap between the two columns is the waiting the page exists to explain."""
    _traced_run(tmp_path)
    html = _render(tmp_path)

    assert "phase open" in html
    assert "on CPU" in html
    assert "blocked" in html


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def test_write_trace_html_creates_missing_directories(tmp_path: Path) -> None:
    _traced_run(tmp_path)
    destination = tmp_path / "reports" / "nested" / "trace.html"

    write_trace_html(merge_run(tmp_path, with_trace=True), destination)

    assert destination.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_cli_trace_writes_a_page(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _traced_run(run_dir)
    destination = tmp_path / "trace.html"

    assert main(["trace", str(run_dir), "-o", str(destination)]) == 0
    assert destination.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_cli_trace_json_lists_the_arrows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _traced_run(tmp_path)

    assert main(["trace", str(tmp_path), "--format", "json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["spans"]
    assert payload["dropped_spans"] == 0


def test_cli_trace_rejects_text_format(tmp_path: Path) -> None:
    """A timeline has no useful text form, so argparse should say so up front."""
    with pytest.raises(SystemExit):
        main(["trace", str(tmp_path), "--format", "text"])


def test_a_nested_call_never_shares_its_parents_row(tmp_path: Path) -> None:
    """The regression this row layout exists to fix.

    Every span used to be drawn at one y per lane, so ``work`` painted over the ``iteration``
    that contained it: the reader saw the child and never the parent, and could not tell
    nesting from sequence. A contained span must land on a different row than its container.
    """
    _traced_run(tmp_path)
    spans = _embedded(_render(tmp_path))["spans"]

    overlapping = [
        (outer, inner)
        for outer in spans
        for inner in spans
        if outer is not inner
        and outer["l"] == inner["l"]
        and inner["t"] >= outer["t"]
        and inner["t"] + inner["d"] <= outer["t"] + outer["d"]
        and inner["d"] < outer["d"]
    ]

    assert overlapping, "the fixture must nest, or this proves nothing"
    assert all(inner["y"] > outer["y"] for outer, inner in overlapping)


def test_a_lane_is_tall_enough_for_its_deepest_call(tmp_path: Path) -> None:
    """Rows are what the script sizes each lane from, so they must cover every span on it."""
    _traced_run(tmp_path)
    data = _embedded(_render(tmp_path))

    for index, lane in enumerate(data["lanes"]):
        deepest = max(
            (span["y"] for span in data["spans"] if span["l"] == index),
            default=0,
        )
        assert lane["rows"] >= deepest + 1


def test_the_call_order_table_follows_the_clock(tmp_path: Path) -> None:
    """The table exists to be read in order, so its rows must be in start order."""
    _traced_run(tmp_path, iterations=2)
    page = _render(tmp_path)

    assert "<h2>Call order</h2>" in page
    body = page.split("<h2>Call order</h2>", 1)[1]
    assert body.index("iteration") < body.index("work")


def test_calls_beyond_the_listed_rows_are_counted_not_dropped(tmp_path: Path) -> None:
    """A lane longer than the table says so, rather than appearing to end early."""
    _traced_run(tmp_path, iterations=40)
    page = _render(tmp_path)

    assert "further call(s) on this lane are not listed" in page


# --------------------------------------------------------------------------- #
# Reading the page: conclusions before evidence
# --------------------------------------------------------------------------- #
def test_the_page_leads_with_findings_not_with_a_chart(tmp_path: Path) -> None:
    """Someone opening this wants to know what is wrong before they read a canvas.

    The order is the point: a reader dropped straight into a timeline has to already know what
    a healthy run looks like to get anything from it.
    """
    _traced_run(tmp_path, iterations=8)
    page = _render(tmp_path)

    assert "<h2>Findings</h2>" in page
    assert page.index("<h2>Findings</h2>") < page.index("<h2>Timeline</h2>")


def test_a_run_with_nothing_wrong_says_so_rather_than_inventing_a_finding(
    tmp_path: Path,
) -> None:
    """Every page looking like it has a problem would make the section worthless."""
    profiler = Profiler(
        run_dir=tmp_path, role="main", enabled=True,
        snapshot_interval_s=None, sample_interval_s=None, trace=True,
    )
    with profiler, profiler.phase("work"):
        pass

    page = _render(tmp_path)

    assert "<h2>Findings</h2>" in page
    assert "no obvious bottleneck" in page or "findings" in page


def test_the_phase_summary_ranks_what_the_run_is_made_of(tmp_path: Path) -> None:
    """The question a reader starts with, and the one a timeline answers worst."""
    _traced_run(tmp_path, iterations=6)
    page = _render(tmp_path)

    assert "<h2>Phase summary</h2>" in page
    assert "self" in page
    body = page.split("<h2>Phase summary</h2>", 1)[1].split("<h2>", 1)[0]
    assert "iteration" in body


def test_the_timeline_carries_a_visible_legend(tmp_path: Path) -> None:
    """The colour convention was a footnote under the chart and the hatching was in Caveats.

    A reader who has to scroll past the chart to learn what its colours mean will read the
    chart wrong first.
    """
    _traced_run(tmp_path)
    page = _render(tmp_path)

    legend = page.split('<div class="tl-legend">', 1)[1].split("</div>", 1)[0]
    assert "on CPU" in legend
    assert "blocked" in legend
    assert "not measured" in legend


def test_the_page_says_what_clicking_a_span_does(tmp_path: Path) -> None:
    """The complaint this answers: it was never stated, and it only drew an outline."""
    _traced_run(tmp_path)
    page = _render(tmp_path)

    assert "click a span" in page.lower()
    assert "unpin" in page


def test_findings_can_point_at_the_timeline(tmp_path: Path) -> None:
    """A finding the reader cannot act on is a complaint, so each names a jump target."""
    _traced_run(tmp_path, iterations=8)
    page = _render(tmp_path)

    assert 'class="tl-jump"' in page
    assert "data-anchor=" in page


def test_the_jump_targets_name_something_the_chart_can_show(tmp_path: Path) -> None:
    """A button that scrolls to a phase the payload never carried does nothing, silently."""
    _traced_run(tmp_path, iterations=8)
    page = _render(tmp_path)
    data = _embedded(page)

    anchors = set(re.findall(r'data-anchor="([^"]+)"', page))
    drawable = {span["n"] for span in data["spans"]} | {lane["id"] for lane in data["lanes"]}

    assert anchors
    assert anchors <= drawable


def test_the_summary_bar_colours_agree_with_the_chart(tmp_path: Path) -> None:
    """Two renderings of one number that disagree are worse than one of them missing.

    The table is server-rendered and the canvas is painted by script, so the blend exists
    twice; this pins the endpoints so they cannot drift apart.
    """
    _traced_run(tmp_path)
    page = _render(tmp_path)

    assert "rgb(178,60,23)" in page  # fully on CPU, in the legend and the bars
    script = re.search(r"<script>(.*?)</script>", page, re.DOTALL)
    assert script is not None
    assert "0xB2" in script.group(1)
    assert "0x2F" in script.group(1)


# --------------------------------------------------------------------------- #
# Source locations
# --------------------------------------------------------------------------- #
def _auto_traced_run(run_dir: Path, project: Path) -> None:
    """One worker traced by function calls, with no ``phase()`` call in the traced code."""
    import workload  # type: ignore[import-not-found]

    from lineprofiler.accounting.autotrace import AutoTracer

    profiler = Profiler(
        run_dir=run_dir,
        role="actor",
        enabled=True,
        snapshot_interval_s=None,
        sample_interval_s=None,
        trace="auto",
    )
    # The profiler's own tracer is scoped to this repo, not to the fixture's project; swap in
    # one that admits the workload, releasing the monitoring slot first.
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


@pytest.fixture
def auto_project(tmp_path: Path) -> Iterator[Path]:
    """A tiny uninstrumented project, importable for the duration of one test."""
    project = tmp_path / "src"
    project.mkdir()
    (project / ".git").mkdir()
    (project / "workload.py").write_text(
        "def inner(n):\n"
        "    return n * n\n"
        "\n"
        "def entry():\n"
        "    return [inner(i) for i in range(20)]\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(project))
    try:
        yield project
    finally:
        sys.path.remove(str(project))
        sys.modules.pop("workload", None)


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="auto-tracing needs sys.monitoring, which arrived in 3.12",
)
def test_auto_traced_spans_carry_a_source_location_to_the_page(
    auto_project: Path,
    tmp_path: Path,
) -> None:
    """Uninstrumented code still tells the reader which function and line a span is."""
    run_dir = tmp_path / "profile"
    _auto_traced_run(run_dir, auto_project)
    data = _embedded(_render(run_dir))

    origins = data["origins"]
    located = {
        origins[span["o"]]["n"]: origins[span["o"]]
        for span in data["spans"]
        if span["o"] >= 0
    }

    assert {"entry", "inner"} <= set(located)
    assert located["inner"]["f"] == "workload.py"
    assert located["inner"]["l"] == 1
    assert located["inner"]["full"].endswith("workload.py")


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="auto-tracing needs sys.monitoring, which arrived in 3.12",
)
def test_the_origin_table_is_interned_rather_than_repeated_per_span(
    auto_project: Path,
    tmp_path: Path,
) -> None:
    """A location per span would repeat one filename thousands of times through the page."""
    run_dir = tmp_path / "profile"
    _auto_traced_run(run_dir, auto_project)
    data = _embedded(_render(run_dir))

    assert len(data["origins"]) < len(data["spans"])
    assert all("file" not in span for span in data["spans"])


def test_named_phases_render_without_inventing_a_source_location(tmp_path: Path) -> None:
    """A phase has no code object behind it; -1 says so, and the tooltip stays silent."""
    _traced_run(tmp_path)
    data = _embedded(_render(tmp_path))

    assert data["origins"] == []
    assert all(span["o"] == -1 for span in data["spans"])


def test_a_source_path_reaches_the_page_as_text_never_as_markup(tmp_path: Path) -> None:
    """Filenames come from user code exactly as phase names do, so they follow the same rule.

    The guard is structural: the tooltip and focus panel build their location line through
    ``named()``, which sets ``textContent``. An ``innerHTML`` anywhere in this script would
    make a crafted path executable in whoever opens the report.
    """
    _traced_run(tmp_path)
    script = re.search(r"<script>(.*?)</script>", _render(tmp_path), re.DOTALL)

    assert script is not None
    assert "innerHTML" not in script.group(1)
    assert "named('div', 'src'" in script.group(1)


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="auto-tracing needs sys.monitoring, which arrived in 3.12",
)
def test_the_phase_summary_names_where_each_phase_lives(
    auto_project: Path,
    tmp_path: Path,
) -> None:
    """The table a reader ranks by must answer "where", not just "how much".

    Locations only in the hover tooltip leave whoever scans the tables — which is how this
    page is read first — with a name and nowhere to go.
    """
    run_dir = tmp_path / "profile"
    _auto_traced_run(run_dir, auto_project)
    page = _render(run_dir)
    summary = page[page.find("<h2>Phase summary"):page.find("<h2>Timeline")]

    assert "workload.py:1" in summary  # inner, at the top of the file
    assert "workload.py:4" in summary  # entry


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="auto-tracing needs sys.monitoring, which arrived in 3.12",
)
def test_the_page_says_a_location_is_the_def_line_not_the_blocking_line(
    auto_project: Path,
    tmp_path: Path,
) -> None:
    """The claim the page must not let a reader make for it.

    A span covers a whole call, so the line that actually blocked is not recorded. Stating
    the limit is what keeps a definition line from being read as a measurement.
    """
    run_dir = tmp_path / "profile"
    _auto_traced_run(run_dir, auto_project)
    page = _render(run_dir)

    assert "where a function is defined" in page
    assert "not the line" in page


def test_a_run_with_no_locations_makes_no_claim_about_them(tmp_path: Path) -> None:
    """Named phases have no origins, so the caveat about them must not appear at all."""
    _traced_run(tmp_path)

    assert "where a function is defined" not in _render(tmp_path)


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="auto-tracing needs sys.monitoring, which arrived in 3.12",
)
def test_the_critical_path_names_where_each_link_lives(
    auto_project: Path,
    tmp_path: Path,
) -> None:
    """The page's most useful table is the one a reader most needs to act on.

    "The critical path is ``entry``" is not somewhere to go; ``workload.py:4`` is.
    """
    run_dir = tmp_path / "profile"
    _auto_traced_run(run_dir, auto_project)
    page = _render(run_dir)
    chain = page[page.find("<h2>Critical path"):page.find("<h2>Lanes")]

    assert "workload.py:" in chain


# --------------------------------------------------------------------------- #
# Navigating the chart
# --------------------------------------------------------------------------- #
def test_the_chart_is_reachable_and_drivable_by_keyboard(tmp_path: Path) -> None:
    """A canvas has no keyboard affordance at all unless one is given to it.

    ``tabIndex`` is what makes it focusable; without it the key handler is attached to an
    element that can never receive a key event, which fails silently and looks like a
    working feature.
    """
    _traced_run(tmp_path)
    script = re.search(r"<script>(.*?)</script>", _render(tmp_path), re.DOTALL)
    assert script is not None
    body = script.group(1)

    assert "tabIndex" in body
    assert "keydown" in body
    for key in ("ArrowLeft", "ArrowRight", "Escape"):
        assert key in body


def test_the_page_documents_its_keyboard_shortcuts(tmp_path: Path) -> None:
    """An undiscoverable shortcut is one that does not exist for most readers."""
    _traced_run(tmp_path)
    page = _render(tmp_path)

    assert "critical path" in page
    assert "Esc" in page


def test_a_collapsed_lane_folds_its_spans_rather_than_hiding_them(tmp_path: Path) -> None:
    """Collapsing must never remove activity from the chart, only stop giving it rows.

    A fold that hid spans would make a busy lane look idle, which is the single most
    misleading thing this page could do.
    """
    _traced_run(tmp_path)
    script = re.search(r"<script>(.*?)</script>", _render(tmp_path), re.DOTALL)
    assert script is not None
    body = script.group(1)

    assert "function rowOf" in body
    assert "collapsed[sp.l] ? 0" in body
    # The geometry must be recomputable, or a fold cannot change the layout.
    assert "function layout" in body


def test_pinning_a_span_traces_the_chain_that_led_to_it(tmp_path: Path) -> None:
    """What "click a span to trace what it was waiting for" was always supposed to mean."""
    _traced_run(tmp_path)
    script = re.search(r"<script>(.*?)</script>", _render(tmp_path), re.DOTALL)
    assert script is not None
    body = script.group(1)

    assert "function causalChain" in body
    assert "releasingArrow" in body


def test_the_causal_walk_cannot_hang_on_a_cycle(tmp_path: Path) -> None:
    """Two workers each waiting on the other produce a cycle in the recorded links.

    An unbounded walk there hangs the browser, which is worse than any wrong picture: the
    reader cannot even close the page to escape it.
    """
    _traced_run(tmp_path)
    script = re.search(r"<script>(.*?)</script>", _render(tmp_path), re.DOTALL)
    assert script is not None
    body = script.group(1)
    walk = body.split("function causalChain", 1)[1].split("function spanKey", 1)[0]

    assert "guard" in walk
    assert "members[" in walk
