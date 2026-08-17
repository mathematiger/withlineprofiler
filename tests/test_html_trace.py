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
