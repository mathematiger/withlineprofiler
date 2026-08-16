"""Tests for the self-contained HTML reports, for both subsystems.

There is deliberately no byte-for-byte golden here, unlike the text report. The text
report's exact columns are its contract — people read and diff it — whereas the HTML's
styling will churn, and a golden over it would be a maintenance tax that catches formatting
churn rather than wrong numbers. What is asserted instead are the properties that would make
the page *wrong*: that it opens offline, that the embedded data matches what a script would
read, that user-controlled text cannot escape its container, and that the icicle chart's
geometry is sound.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from test_accounting_report_golden import _build_fixed_run

from lineprofiler import LineProfiler
from lineprofiler.accounting import merge_run
from lineprofiler.accounting.cli import main
from lineprofiler.accounting.htmlreport import flame_cells, render_html, write_html
from lineprofiler.accounting.phasetree import PhaseStats, PhaseTree
from lineprofiler.accounting.report import report_as_dict

THIS_DIR = str(Path(__file__).resolve().parent)


def worked(n: int) -> int:
    total = 0
    for i in range(n):
        total += i * i
    return total


def holds_markup() -> str:
    markup = "<b>not bold</b>"  # must be escaped when this line is rendered
    return markup


def _embedded_data(html: str) -> dict[str, Any]:
    """Pull the JSON payload back out of the page, the way a reader or script would."""
    match = re.search(
        r'<script type="application/json" id="lineprofiler-data">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None, "the page carries no data block"
    payload: dict[str, Any] = json.loads(match.group(1).replace("<\\/", "</"))
    return payload


# --------------------------------------------------------------------------- #
# Self-containment
# --------------------------------------------------------------------------- #
def test_the_report_page_needs_no_network(tmp_path: Path) -> None:
    """A page that renders blank without a network was not worth generating."""
    _build_fixed_run(tmp_path)
    html = render_html(merge_run(tmp_path))

    assert "http://" not in html
    assert "https://" not in html
    assert "<link" not in html
    assert "src=" not in html


def test_the_source_page_needs_no_network(tmp_path: Path) -> None:
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        worked(50)
    destination = tmp_path / "source.html"
    profiler.to_html(destination)

    html = destination.read_text(encoding="utf-8")
    assert "http://" not in html
    assert "https://" not in html
    assert "<link" not in html


# --------------------------------------------------------------------------- #
# The embedded record
# --------------------------------------------------------------------------- #
def test_the_embedded_data_is_the_json_document(tmp_path: Path) -> None:
    """The page carries exactly what ``--format json`` would emit, so the two cannot drift."""
    _build_fixed_run(tmp_path)
    run = merge_run(tmp_path)

    assert _embedded_data(render_html(run)) == json.loads(json.dumps(report_as_dict(run)))


def test_a_phase_named_like_a_script_tag_cannot_escape_its_block(tmp_path: Path) -> None:
    """Phase names come from user code, so this is reachable input, not a hypothetical."""
    _build_fixed_run(tmp_path)
    run = merge_run(tmp_path)
    run.tree[("</script><img src=x>",)] = PhaseStats(calls=1, wall_ns=1000)

    html = render_html(run)

    body = html.split('<script type="application/json"')[0]
    assert "<img src=x>" not in body, "an unescaped tag reached the document body"
    assert _embedded_data(html), "the data block no longer parses"


def test_source_html_escapes_the_code_it_shows(tmp_path: Path) -> None:
    """The markup lives in a *function*, not in the ``with`` body.

    Only the monitoring backend traces the block's own frame, so a source line placed there
    would make this pass on 3.12 and fail on 3.11 for a reason unrelated to escaping.
    """
    profiler = LineProfiler(project_folder=THIS_DIR)
    with profiler:
        holds_markup()
    destination = tmp_path / "source.html"
    profiler.to_html(destination)

    html = destination.read_text(encoding="utf-8")
    assert "&lt;b&gt;not bold&lt;/b&gt;" in html
    assert "<b>not bold</b>" not in html


# --------------------------------------------------------------------------- #
# Icicle-chart geometry
# --------------------------------------------------------------------------- #
def _tree(**paths: int) -> PhaseTree:
    """Build a tree from ``a_b=wall_ns`` keyword pairs, where ``_`` separates path parts."""
    tree: PhaseTree = {}
    for name, wall_ns in paths.items():
        tree[tuple(name.split("__"))] = PhaseStats(calls=1, wall_ns=wall_ns)
    return tree


def test_every_cell_fits_inside_its_parent() -> None:
    cells = flame_cells(_tree(root=1000, root__a=600, root__b=400, root__a__deep=600))
    by_path = {cell.path: cell for cell in cells}

    for cell in cells:
        parent = by_path.get(cell.path[:-1])
        if parent is None:
            continue
        assert cell.x >= parent.x - 1e-9
        assert cell.x + cell.width <= parent.x + parent.width + 1e-9


def test_children_exceeding_their_parent_are_scaled_to_fit() -> None:
    """Honest over-subscription: a phase entered from two threads, or recursively.

    Drawing children wider than the parent they sit in would read as a measurement error
    rather than as the aggregation it actually is.
    """
    cells = flame_cells(_tree(root=1000, root__a=900, root__b=900))
    by_path = {cell.path: cell for cell in cells}
    parent = by_path[("root",)]

    for name in ("a", "b"):
        child = by_path[("root", name)]
        assert child.x + child.width <= parent.x + parent.width + 1e-9


def test_an_empty_tree_draws_nothing() -> None:
    assert flame_cells({}) == []


def test_wait_share_drives_the_cell_colour() -> None:
    """The one thing this chart says that a column of numbers does not."""
    busy = PhaseStats(calls=1, wall_ns=1000, cpu_ns=1000)
    blocked = PhaseStats(calls=1, wall_ns=1000, cpu_ns=0)
    tree: PhaseTree = {("busy",): busy, ("blocked",): blocked}

    cells = {cell.path: cell for cell in flame_cells(tree)}

    assert cells[("busy",)].wait_pct == pytest.approx(0.0)
    assert cells[("blocked",)].wait_pct == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Writing, and the CLI
# --------------------------------------------------------------------------- #
def test_write_html_creates_parent_directories(tmp_path: Path) -> None:
    _build_fixed_run(tmp_path)
    destination = tmp_path / "nested" / "deeper" / "report.html"

    write_html(merge_run(tmp_path), destination)

    assert destination.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_cli_writes_html_to_the_requested_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _build_fixed_run(run_dir)
    destination = tmp_path / "out.html"

    assert main(["report", str(run_dir), "--format", "html", "-o", str(destination)]) == 0
    assert destination.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_cli_json_flag_still_selects_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The superseded spelling stays working; it is in released docs and users' scripts."""
    _build_fixed_run(tmp_path)

    assert main(["report", str(tmp_path), "--json"]) == 0

    assert json.loads(capsys.readouterr().out)["run"]["roles"]


def test_cli_rejects_html_for_compare(tmp_path: Path) -> None:
    """``compare`` cannot render HTML, so argparse should say so rather than fail later."""
    with pytest.raises(SystemExit):
        main(["compare", str(tmp_path), str(tmp_path), "--format", "html"])
