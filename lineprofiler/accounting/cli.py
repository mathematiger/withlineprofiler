"""Command-line entry point.

    lineprofiler report <run_dir> [--no-samples] [--format text|json|html] [-o PATH]
    lineprofiler compare <run_a> <run_b> [--format text|json] [-o PATH]
    lineprofiler trace <run_dir> [--max-spans N] [--quiet] [--format html|json] [-o PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
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
    elif args.command == "trace":
        _emit(_render_trace(args), args.output)
        return _gate_exit_code(args)
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


def _render_report(args: argparse.Namespace) -> str:
    # Traces are read when they exist: the report's occupancy, concurrency and request
    # lifecycle blocks all derive from them, and defaulting them away meant a run recorded
    # with trace=True rendered without the three blocks it was instrumented for. Costs
    # nothing on the common run, which has no sidecars to read.
    run = merge_run(args.run_dir, with_samples=not args.no_samples, with_trace=True)
    if args.format == "json":
        return json.dumps(report_as_dict(run), indent=2)
    if args.format == "html":
        from lineprofiler.accounting.htmlreport import render_html

        return render_html(run)
    return render(run)


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
        from lineprofiler.accounting.findings import phase_totals, rank_findings
        from lineprofiler.accounting.tracealign import align_run

        aligned = align_run(run)
        progress(f"aligned {len(aligned.spans):,} spans")
        return json.dumps(
            {
                "duration_ns": aligned.duration_ns,
                "lanes": aligned.lanes,
                "spans": len(aligned.spans),
                "arrows": [
                    {
                        "channel": arrow.channel,
                        "key": arrow.key,
                        "from": arrow.src_worker,
                        "to": arrow.dst_worker,
                        "delay_ns": arrow.delay_ns,
                    }
                    for arrow in aligned.arrows
                ],
                "unmatched_waits": aligned.unmatched_waits,
                "dropped_spans": aligned.dropped_spans,
                # The same ranking the page leads with. A gate that has to re-derive "was
                # this a queue or a stall" from spans and arrows would be reimplementing
                # findings.py against the same data, and the two would drift.
                "findings": [
                    {
                        "kind": finding.kind,
                        "headline": finding.headline,
                        "detail": finding.detail,
                        "cost_pct": round(finding.cost_pct, 2),
                        "anchor": finding.anchor,
                        "lanes": list(finding.lanes),
                    }
                    for finding in rank_findings(aligned)
                ],
                "phases": [
                    {
                        "path": total.path,
                        "calls": total.calls,
                        "lanes": total.lanes,
                        "wall_ns": total.wall_ns,
                        "self_ns": total.self_ns,
                        "wait_ns": total.wait_ns,
                        "wait_pct": round(total.wait_pct, 1),
                    }
                    for total in phase_totals(aligned)
                ],
            },
            indent=2,
        )
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
