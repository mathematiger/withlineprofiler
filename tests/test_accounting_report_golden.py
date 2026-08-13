"""Golden-file test for the report.

The run is built from fixed numbers written straight to disk rather than from real timings,
so the output is byte-for-byte deterministic and any change to the layout shows up as a
diff. Regenerate deliberately with::

    LINEPROFILER_UPDATE_GOLDEN=1 pytest tests/test_accounting_report_golden.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from lineprofiler.accounting import merge_run, render
from lineprofiler.accounting.compare import render_comparison
from lineprofiler.accounting.histogram import bucket_index

GOLDEN_DIR = Path(__file__).parent / "golden"


def _phase(
    calls: int,
    wall_ms: float,
    cpu_ms: float,
    child_ms: float = 0.0,
    counters: dict[str, int] | None = None,
) -> dict[str, object]:
    """Build one serialised phase node with a single-bucket histogram."""
    wall_ns = int(wall_ms * 1e6)
    per_call = max(1, wall_ns // max(1, calls))
    return {
        "calls": calls,
        "wall_ns": wall_ns,
        "cpu_ns": int(cpu_ms * 1e6),
        "child_wall_ns": int(child_ms * 1e6),
        "hist": {str(bucket_index(per_call)): calls},
        "counters": counters or {},
    }


def _write_worker(
    run_dir: Path,
    pid: int,
    role: str,
    phases: dict[str, object],
    samples: list[dict[str, object]] | None = None,
) -> None:
    """Write one worker's snapshot, and optionally its resource samples."""
    workers = run_dir / "workers"
    workers.mkdir(parents=True, exist_ok=True)
    stem = f"w_{pid}_{pid:08x}"
    payload = {
        "version": 1,
        "pid": pid,
        "role": role,
        "started_at": 1_700_000_000.0,
        "written_at": 1_700_000_090.0,
        "backend": None,
        "phases": phases,
    }
    (workers / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")
    if samples:
        (workers / f"{stem}.samples").write_text(
            "\n".join(json.dumps(row) for row in samples) + "\n", encoding="utf-8",
        )


def _build_fixed_run(run_dir: Path) -> None:
    """A two-role run with known phases, counters, I/O and memory growth."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps({"version": 1, "host": "golden-host", "started_at": 1_700_000_000.0}),
        encoding="utf-8",
    )

    for index, pid in enumerate((1001, 1002)):
        scale = 1.0 + index * 0.25
        _write_worker(
            run_dir,
            pid,
            "actor",
            {
                "iteration": _phase(20, 400.0 * scale, 40.0, 396.0 * scale),
                "iteration/mcts": _phase(
                    20, 300.0 * scale, 30.0, 0.0, {"mcts_simulations": 1280},
                ),
                "iteration/env_step": _phase(20, 96.0 * scale, 20.0, 0.0, {"env_steps": 20}),
            },
            samples=[
                {"t": 1_700_000_000.0 + step, "phase": "iteration/mcts",
                 "rss": 100_000_000 + step * 1_000_000}
                for step in range(6)
            ],
        )

    _write_worker(
        run_dir,
        2001,
        "learner",
        {
            "iteration": _phase(10, 500.0, 480.0, 495.0),
            "iteration/train_step": _phase(10, 400.0, 390.0, 0.0, {"train_samples": 1280}),
            "iteration/checkpoint": _phase(
                2, 95.0, 10.0, 0.0, {"io_write_bytes": 41_943_040},
            ),
        },
        samples=[
            {"t": 1_700_000_000.0 + step, "phase": "iteration/train_step",
             "rss": 500_000_000, "write_bytes": step * 8_388_608}
            for step in range(6)
        ],
    )


def _check_golden(name: str, actual: str) -> None:
    """Compare against the stored golden file, or rewrite it when asked to."""
    path = GOLDEN_DIR / name
    if os.environ.get("LINEPROFILER_UPDATE_GOLDEN"):
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
    assert path.exists(), (
        f"missing golden file {path}; regenerate with LINEPROFILER_UPDATE_GOLDEN=1"
    )
    assert actual == path.read_text(encoding="utf-8")


def test_report_matches_the_golden_file(tmp_path: Path) -> None:
    _build_fixed_run(tmp_path)
    _check_golden("report.txt", render(merge_run(tmp_path)))


def test_comparison_matches_the_golden_file(tmp_path: Path) -> None:
    run_a, run_b = tmp_path / "a", tmp_path / "b"
    _build_fixed_run(run_a)
    _build_fixed_run(run_b)
    # Make B's training step twice as slow and give it a phase A does not have.
    _write_worker(
        run_b,
        2002,
        "learner",
        {
            "iteration": _phase(10, 900.0, 880.0, 895.0),
            "iteration/train_step": _phase(10, 800.0, 790.0),
            "iteration/compile": _phase(1, 50.0, 50.0),
        },
    )

    text = render_comparison(merge_run(run_a), merge_run(run_b), "A", "B")
    _check_golden("compare.txt", text)
