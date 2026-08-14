"""Command-line entry point.

    lineprofiler report <run_dir>
    lineprofiler compare <run_a> <run_b> [--json]
"""

from __future__ import annotations

import argparse
import json
import sys

from lineprofiler.accounting.compare import comparison_as_dict, render_comparison
from lineprofiler.accounting.report import render
from lineprofiler.accounting.snapshot import merge_run


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and print the requested output. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "report":
        print(render(merge_run(args.run_dir, with_samples=not args.no_samples)))  # noqa: T201
    elif args.command == "compare":
        print(_render_compare(args))  # noqa: T201
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lineprofiler")
    subcommands = parser.add_subparsers(dest="command", required=True)

    report = subcommands.add_parser("report", help="merge a run directory and print its report")
    report.add_argument("run_dir", help="directory passed to Profiler(run_dir=...)")
    report.add_argument(
        "--no-samples",
        action="store_true",
        help=(
            "skip the resource samples and report phases only. Samples dominate memory: a "
            "12-hour worker holds ~28 MB of them, so a large run can exhaust a login node. "
            "Drops the I/O, memory and GPU blocks."
        ),
    )

    compare = subcommands.add_parser("compare", help="show what changed between two runs")
    compare.add_argument("run_a")
    compare.add_argument("run_b")
    compare.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    return parser


def _render_compare(args: argparse.Namespace) -> str:
    run_a, run_b = merge_run(args.run_a), merge_run(args.run_b)
    if args.json:
        return json.dumps(comparison_as_dict(run_a, run_b), indent=2)
    return render_comparison(run_a, run_b, args.run_a, args.run_b)


if __name__ == "__main__":
    sys.exit(main())
