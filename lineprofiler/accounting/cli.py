"""Command-line entry point.

    lineprofiler report <run_dir> [--no-samples] [--format text|json|html] [-o PATH]
    lineprofiler compare <run_a> <run_b> [--format text|json] [-o PATH]
    lineprofiler trace <run_dir> [--max-spans N] [--quiet] [--format html|json] [-o PATH]
    lineprofiler run <script.py> [args...] [--top N] [--functions] [--html PATH]
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from collections.abc import Callable
from pathlib import Path

from lineprofiler.accounting.compare import comparison_as_dict, render_comparison
from lineprofiler.accounting.report import render, report_as_dict
from lineprofiler.accounting.snapshot import MergedRun, merge_run


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and emit the requested output. Returns a process exit code.

    ``0`` is success, ``1`` means ``trace --fail-over`` found something over its threshold,
    and ``2`` means the run directory does not exist — a usage error, which is what argparse
    already uses ``2`` for.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "report":
        # Traces are read when they exist: the report's occupancy, concurrency and request
        # lifecycle blocks all derive from them, and defaulting them away meant a run recorded
        # with trace=True rendered without the three blocks it was instrumented for. Costs
        # nothing on the common run, which has no sidecars to read.
        run = merge_run(args.run_dir, with_samples=not args.no_samples, with_trace=True)
        _emit(_render_report(run, args.format), args.output)
        return _report_exit_code(run)
    elif args.command == "compare":
        _emit(_render_compare(args), args.output)
    elif args.command == "trace":
        _emit(_render_trace(args), args.output)
        return _gate_exit_code(args)
    elif args.command == "run":
        return _run_script(args)
    return 0


def _run_script(args: argparse.Namespace) -> int:
    """Run a script under the line profiler with no edit to it — ``kernprof`` without the
    decorators. Everything under the script's project folder is profiled.

    The summary is printed even when the script fails, because a traceback is usually the
    moment the profile is wanted. The script's own exit status is preserved.
    """
    from lineprofiler.config import find_project_root
    from lineprofiler.profiler import LineProfiler

    script = Path(args.script).resolve()
    if not script.is_file():
        print(f"lineprofiler run: {script} is not a file", file=sys.stderr)  # noqa: T201
        return 2
    sys.argv = [str(script), *args.args]
    sys.path.insert(0, str(script.parent))
    profiler = LineProfiler(project_folder=find_project_root(script))
    status = 0
    try:
        with profiler:
            runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        status = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        if args.functions:
            profiler.print_stats()
        profiler.print_global_top_stats(top_n=args.top)
        if args.html:
            profiler.to_html(args.html)
    return status


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

    trace = subcommands.add_parser(
        "trace",
        help="draw the recorded timeline: which worker waited, when, and for whom",
    )
    trace.add_argument("run_dir", help="directory passed to Profiler(run_dir=...)")
    trace.add_argument(
        "--fail-over",
        type=float,
        default=None,
        metavar="PCT",
        help=(
            "exit non-zero if any finding costs more than PCT%% of the traced span. Turns the "
            "timeline into a regression gate: --fail-over 50 fails a build where one phase "
            "blocks for more than half the run. The report is still written."
        ),
    )
    trace.add_argument(
        "--max-spans",
        type=int,
        default=None,
        help=(
            "draw at most this many spans, keeping the longest. A large run degrades to a "
            "readable picture instead of failing to render at all; the number dropped is "
            "stated on the page"
        ),
    )
    trace.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="suppress the progress lines written to stderr",
    )
    # html first: a timeline is a picture, and the text form of one is a wall of numbers.
    _add_output_arguments(trace, formats=("html", "json"), default="html")

    run = subcommands.add_parser(
        "run",
        help="run a script under the line profiler, without editing it",
    )
    run.add_argument("script", help="the script to run; everything under its project is profiled")
    run.add_argument("args", nargs=argparse.REMAINDER, help="arguments passed to the script")
    run.add_argument("--top", type=int, default=25, help="lines in the summary (default 25)")
    run.add_argument(
        "--functions",
        action="store_true",
        help="also print every profiled function's full table, in source order",
    )
    run.add_argument("--html", metavar="PATH", help="also write the annotated source view here")
    return parser


def _add_output_arguments(
    parser: argparse.ArgumentParser,
    formats: tuple[str, ...],
    default: str = "text",
) -> None:
    """Add the output selection shared by the subcommands.

    Each is offered only the formats it can actually produce, so an unsupported choice is
    rejected by argparse with the valid list rather than failing later.
    """
    parser.add_argument(
        "--format",
        choices=formats,
        default=default,
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


def _render_report(run: MergedRun, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(report_as_dict(run), indent=2)
    if output_format == "html":
        from lineprofiler.accounting.htmlreport import render_html

        return render_html(run)
    return render(run)


def _report_exit_code(run: MergedRun) -> int:
    """``2`` when the run directory does not exist, ``0`` otherwise.

    Two failures look alike from a shell and are not: a path that does not exist is a usage
    error — the same thing argparse already exits ``2`` for — while a directory that exists
    and holds no worker files is a legitimate answer about a run nobody profiled, and must not
    fail a pipeline that merely reports on every run it finds. ``1`` stays reserved for
    ``trace --fail-over``, which is a finding about the run rather than a problem with the
    command. The report is written either way; only the code differs.
    """
    return 2 if run.empty_reason == "missing" else 0


def _render_trace(args: argparse.Namespace) -> str:
    """Render the timeline. Samples are read too, so the GPU lanes have something to draw.

    Progress goes to stderr because rendering a large run takes minutes, and silence for
    minutes is indistinguishable from a hang — the same ambiguity the timeline itself exists
    to resolve, one level up. Two verification runs were lost to a render that was working.
    """
    progress = _progress_reporter(quiet=getattr(args, "quiet", False))
    progress(f"loading workers from {args.run_dir}")
    run = merge_run(args.run_dir, with_samples=True, with_trace=True)
    progress(f"loaded {len(run.workers)} workers")
    if args.format == "json":
        # The same document the library's write_trace emits: the ranking the page leads with
        # is derived once, in findings.py, so a gate and a saved file cannot disagree.
        from lineprofiler.accounting.findings import trace_as_dict
        from lineprofiler.accounting.tracealign import align_run

        aligned = align_run(run)
        progress(f"aligned {len(aligned.spans):,} spans")
        return json.dumps(trace_as_dict(aligned), indent=2)
    from lineprofiler.accounting.htmltrace import render_trace_html

    return render_trace_html(run, max_spans=args.max_spans, progress=progress)


def _gate_exit_code(args: argparse.Namespace) -> int:
    """``1`` when a finding exceeds ``--fail-over``, else ``0``.

    Re-merges the run rather than threading the findings out of the renderer. That is a second
    read of the same directory, which is worth it here: the alternative is a return type that
    carries either text or text-plus-findings depending on a flag, and every caller of
    ``_render_trace`` paying for a gate almost none of them asked for.

    Exits ``0`` when the flag is absent, so adding it to a pipeline changes nothing until a
    threshold is actually set.
    """
    threshold = getattr(args, "fail_over", None)
    if threshold is None:
        return 0

    from lineprofiler.accounting.findings import rank_findings
    from lineprofiler.accounting.tracealign import align_run

    findings = rank_findings(align_run(merge_run(args.run_dir, with_trace=True)))
    over = [finding for finding in findings if finding.cost_pct > threshold]
    if not over:
        return 0
    for finding in over:
        # stderr, so a gate failure is visible even when the page went to stdout.
        # "costs N%" rather than repeating the headline's own percentage: a headline may
        # quote a different denominator (an idle-lane finding states a per-lane mean, while
        # the gate thresholds its share of the whole run), and printing the two side by side
        # as though they were one number reads as an arithmetic bug in the tool.
        print(  # noqa: T201
            f"lineprofiler: {finding.headline} "
            f"— costs {finding.cost_pct:.0f}% of the run, over the {threshold:.0f}% threshold",
            file=sys.stderr,
        )
    return 1


def _progress_reporter(quiet: bool) -> Callable[[str], None]:
    """Return a stderr progress callback, or one that says nothing.

    stderr rather than stdout: the rendered page routinely goes to stdout for redirection,
    and progress lines mixed into an HTML file would corrupt it.
    """
    if quiet:
        return lambda message: None

    def report(message: str) -> None:
        print(f"lineprofiler: {message}", file=sys.stderr)  # noqa: T201

    return report


def _render_compare(args: argparse.Namespace) -> str:
    run_a, run_b = merge_run(args.run_a), merge_run(args.run_b)
    if args.format == "json":
        return json.dumps(comparison_as_dict(run_a, run_b), indent=2)
    return render_comparison(run_a, run_b, args.run_a, args.run_b)


if __name__ == "__main__":
    sys.exit(main())
