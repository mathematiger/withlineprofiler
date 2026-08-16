"""Annotated-source HTML for the line profiler: every profiled line, heat-coloured by time.

A terminal table tells you line 214 is slow. This tells you the same thing with the code
around it, which is usually what turns a number into a fix.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from lineprofiler.htmldoc import document, escape, tile

if TYPE_CHECKING:
    from lineprofiler.profiler import FunctionKey, FunctionStats


def render_source_html(
    stats: dict[FunctionKey, FunctionStats],
    title: str = "lineprofiler",
) -> str:
    """Return a self-contained page showing each profiled function's annotated source.

    Heat is scaled **per function**, not globally: a function's own slowest line is its
    darkest, so a cheap function stays readable instead of washing out next to an expensive
    one. The cross-function ranking is a separate table at the top, so the two views answer
    their own questions and neither has to compromise for the other.
    """
    functions = [
        function
        for function in sorted(stats.values(), key=lambda f: -f.total_time)
        if function.line_stats
    ]
    if not functions:
        return document(title, "<h1>No profiling data collected.</h1>")

    total = sum(function.total_time for function in functions)
    blocks = [
        f"<h1>{escape(title)}</h1>",
        f'<div class="tiles">{tile("functions", str(len(functions)))}'
        f'{tile("total time", _micros(total))}</div>',
        _hotspots(functions),
        *(_function_block(function) for function in functions),
    ]
    return document(title, "\n".join(block for block in blocks if block))


def _hotspots(functions: list[FunctionStats], limit: int = 15) -> str:
    """The slowest individual lines across every function — the global view, ranked."""
    entries = [
        (function, line_number, line)
        for function in functions
        for line_number, line in function.line_stats.items()
    ]
    entries.sort(key=lambda item: -item[2].total_time)
    rows = "".join(
        f"<tr><td class=\"mono\">{escape(function.function_name)}:{line_number}</td>"
        f"<td>{line.hits}</td><td>{_micros(line.total_time)}</td>"
        f"<td class=\"mono\">{escape(function.source_lines.get(line_number, '').strip())}</td>"
        "</tr>"
        for function, line_number, line in entries[:limit]
    )
    return (
        "<h2>Hotspots</h2>\n"
        '<div class="scroll"><table><thead><tr><th>where</th><th>hits</th>'
        "<th>time</th><th>line</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _function_block(function: FunctionStats) -> str:
    """One function's full source, every line annotated and shaded by its own share."""
    peak = max((line.total_time for line in function.line_stats.values()), default=0.0)
    last = max(function.line_stats, default=function.first_line)
    rows = "".join(
        _source_row(function, line_number, peak)
        for line_number in range(function.first_line, last + 1)
    )
    return (
        f"<h2>{escape(function.function_name)} — {_micros(function.total_time)}</h2>\n"
        f'<p class="note mono">{escape(function.filename)}:{function.first_line}</p>\n'
        '<div class="scroll"><table><thead><tr><th>line</th><th>hits</th><th>time</th>'
        "<th>per hit</th><th>%</th><th>source</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _source_row(function: FunctionStats, line_number: int, peak: float) -> str:
    """One source line. Lines that never ran are shown greyed rather than omitted.

    A zero-hit line inside a profiled function is a fact worth seeing — it means the branch
    was not taken — and dropping it would also break the flow of the code being read.
    """
    source = escape(function.source_lines.get(line_number, ""))
    line = function.line_stats.get(line_number)
    if line is None:
        return (
            f'<tr style="opacity:.45"><td class="mono">{line_number}</td>'
            f'<td></td><td></td><td></td><td></td><td class="mono">{source}</td></tr>'
        )

    share = line.total_time / function.total_time * 100 if function.total_time else 0.0
    intensity = line.total_time / peak if peak else 0.0
    # Alpha only, so the shading survives a dark theme and never fights the text colour.
    background = f"background: rgba(178,60,23,{intensity * 0.55:.3f})"
    return (
        f'<tr style="{background}"><td class="mono">{line_number}</td>'
        f"<td>{line.hits}</td><td>{_micros(line.total_time)}</td>"
        f"<td>{_micros(line.average_time)}</td><td>{share:.1f}%</td>"
        f'<td class="mono">{source}</td></tr>'
    )


def _micros(seconds: float) -> str:
    """Format a duration in microseconds, matching the terminal report's unit."""
    return f"{seconds * 1e6:.1f} µs"
