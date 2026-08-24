"""Tests for the two ways in which the package is *reached*, rather than what it measures.

Nothing here profiles anything. These are packaging assertions: that the name on PyPI is
importable, that it hands back the same objects as the documented one, and that the CLI is
reachable through ``python -m`` when the console script is not on ``PATH``. Both failures cost
a first-time reader minutes and produce an error that names neither cause.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import with_line_profiler
from lineprofiler.accounting.cli import main as cli_main


def test_the_pip_name_re_exports_the_same_objects() -> None:
    """Same objects, not merely same-looking ones: a copy would drift into a second package."""
    import lineprofiler

    for name in lineprofiler.__all__:
        assert getattr(with_line_profiler, name) is getattr(lineprofiler, name), name
    assert with_line_profiler.__version__ == lineprofiler.__version__


def test_the_shim_reaches_the_accounting_layer() -> None:
    """``accounting`` is a subpackage, so it is reachable only if the shim imports it."""
    from lineprofiler import accounting

    assert with_line_profiler.accounting is accounting


def test_python_dash_m_runs_the_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The console script is not always on ``PATH`` — an unactivated virtualenv, a
    ``pip install --user``, a batch job running ``python`` by absolute path. The module form
    works wherever the package is importable, which is where the profiler was."""
    (tmp_path / "workers").mkdir(parents=True)

    completed = subprocess.run(
        [sys.executable, "-m", "lineprofiler", "report", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0
    assert cli_main(["report", str(tmp_path)]) == 0
    assert completed.stdout.strip() == capsys.readouterr().out.strip()
