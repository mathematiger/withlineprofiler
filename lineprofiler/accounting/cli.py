"""Command-line entry point.

    lineprofiler report <run_dir> [--no-samples] [--format text|json|html] [-o PATH]
    lineprofiler compare <run_a> <run_b> [--format text|json] [-o PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lineprofiler.accounting.compare import comparison_as_dict, render_comparison
from lineprofiler.accounting.report import render, report_as_dict
from lineprofiler.accounting.snapshot import merge_run


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and emit the requested output. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "report":
        _emit(_render_report(args), args.output)
    elif args.command == "compare":
        _emit(_render_compare(args), args.output)
    return 0


def _emit(text: str, output: str | None) -> None:
    """Write to ``output`` or stdout — the only file-writing path in the CLI.

    Parent directories are deliberately not created: from a command line a path that does
    not exist is usually a typo, and failing loudly beats scattering directories.
    """
    if output is None:
        print(text)  # noqa: T201
        return
    Path(output).write_text(text, encoding="utf-8")


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

    _add_output_arguments(report, formats=("text", "json", "html"))

    compare = subcommands.add_parser("compare", help="show what changed between two runs")
    compare.add_argument("run_a")
    compare.add_argument("run_b")
    _add_output_arguments(compare, formats=("text", "json"))
    return parser


def _add_output_arguments(
    parser: argparse.ArgumentParser,
    formats: tuple[str, ...],
) -> None:
    """Add the output selection shared by both subcommands.

    ``compare`` is offered only the formats it can actually produce, so an unsupported
    choice is rejected by argparse with the valid list rather than failing later.
    """
    parser.add_argument(
        "--format",
        choices=formats,
        default="text",
        help=(
            "output format. json lets a run gate CI or be diffed across sweep arms without "
            "re-deriving the shares and quantiles; html is a single self-contained file "
            "with no external assets"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        help="write to this path instead of stdout",
    )
    # Superseded by --format json, kept because it is in released documentation and may be
    # in users' scripts. Removing it would break them for no benefit.
    parser.add_argument(
        "--json",
        action="store_const",
        const="json",
        dest="format",
        help=argparse.SUPPRESS,
    )


def _render_report(args: argparse.Namespace) -> str:
    run = merge_run(args.run_dir, with_samples=not args.no_samples)
    if args.format == "json":
        return json.dumps(report_as_dict(run), indent=2)
    if args.format == "html":
        from lineprofiler.accounting.htmlreport import render_html

        return render_html(run)
    return render(run)


def _render_compare(args: argparse.Namespace) -> str:
    run_a, run_b = merge_run(args.run_a), merge_run(args.run_b)
    if args.format == "json":
        return json.dumps(comparison_as_dict(run_a, run_b), indent=2)
    return render_comparison(run_a, run_b, args.run_a, args.run_b)


if __name__ == "__main__":
    sys.exit(main())
