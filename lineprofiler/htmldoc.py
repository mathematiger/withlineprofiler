"""The shared skeleton for the self-contained HTML reports.

Both reports are one file that opens offline: no CDN, no webfont, no separate stylesheet.
That is a deliberate constraint rather than a minimalist affectation — a profiling artifact
gets attached to a ticket, copied to a laptop, and opened six months later on a machine with
no network, and one that renders blank under those conditions was not worth generating.

Nothing here knows about phases or source lines; it holds only the parts both reports share.
"""
from __future__ import annotations

import json
from typing import Union

# What ``json.dumps`` accepts, spelled out rather than left as ``Any`` so a caller passing
# something unserialisable is caught here instead of at render time. Spelled with ``Union``
# because this is evaluated at runtime, where PEP 604's ``|`` needs 3.10+ and a string
# forward reference inside it would not resolve on the versions this package supports.
JsonValue = Union[  # noqa: UP007
    None, bool, int, float, str, list["JsonValue"], dict[str, "JsonValue"],
]

# Deliberately monochrome plus one accent. The reports encode their meaning in position and
# length, never in hue alone, so they survive being printed or read with colour-vision
# deficiency; the accent marks the one thing worth the eye jumping to.
_STYLE = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #6a6a6a; --rule: #e3e3e3;
  --panel: #fafafa; --accent: #b23c17; --cool: #2f6f9f;
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16181a; --fg: #e8e6e3; --muted: #9a9a9a; --rule: #2c2f33;
    --panel: #1d2023; --accent: #ff7a4d; --cool: #6fb3e0;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.5rem 4rem; background: var(--bg); color: var(--fg);
  font: 14px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Helvetica, sans-serif;
}
main { max-width: 78rem; margin: 0 auto; }
h1 { font-size: 1.4rem; font-weight: 600; margin: 0 0 .2rem; letter-spacing: -.01em; }
h2 {
  font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: .09em;
  color: var(--muted); margin: 2.4rem 0 .7rem; padding-bottom: .35rem;
  border-bottom: 1px solid var(--rule);
}
.sub { color: var(--muted); margin: 0 0 .4rem; }
.note { color: var(--muted); font-size: .85rem; margin: .5rem 0 0; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: .3rem .5rem; border-bottom: 1px solid var(--rule); }
th:first-child, td:first-child { text-align: left; }
th {
  font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted);
  font-weight: 600;
}
code, .mono, pre {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.tiles { display: flex; flex-wrap: wrap; gap: .6rem; margin: .2rem 0 0; }
.tile {
  background: var(--panel); border: 1px solid var(--rule); border-radius: 6px;
  padding: .55rem .8rem; min-width: 8.5rem;
}
.tile .k {
  font-size: .68rem; text-transform: uppercase; letter-spacing: .07em; color: var(--muted);
}
.tile .v { font-size: 1.05rem; font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; }
svg { display: block; max-width: 100%; height: auto; }
"""


def document(
    title: str,
    body: str,
    data: JsonValue = None,
    style: str = "",
    script: str = "",
) -> str:
    """Wrap ``body`` in a complete HTML page, optionally embedding ``data`` as JSON.

    The JSON block is the machine-readable record the page was drawn from, carried along so
    a reader can extract the exact numbers behind any figure without re-running anything.

    ``style`` and ``script`` are appended *inline*, never linked. The report and source pages
    pass neither and stay script-free; the trace timeline passes a script because panning and
    zooming a hundred thousand spans cannot be done with static SVG. Inline is what preserves
    the property that actually matters — the file opens offline, six months later, with no
    network — which a CDN reference would break just as thoroughly as a stylesheet link.
    """
    payload = ""
    if data is not None:
        payload = (
            '\n<script type="application/json" id="lineprofiler-data">'
            f"{embed_json(data)}</script>"
        )
    extra_style = f"\n<style>{style}</style>" if style else ""
    behaviour = f"\n<script>{script}</script>" if script else ""
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(title)}</title>\n"
        f"<style>{_STYLE}</style>{extra_style}\n</head>\n<body>\n<main>\n"
        f"{body}\n</main>{payload}{behaviour}\n</body>\n</html>\n"
    )


def embed_json(data: JsonValue) -> str:
    """Serialise ``data`` for a ``<script type="application/json">`` block.

    ``</`` is escaped because the HTML parser looks for the closing tag *textually*, without
    understanding JSON: a phase named ``</script>`` would otherwise end the block early and
    spill the rest of the document into the page. Phase names come from user code, so that
    is reachable input rather than a hypothetical.
    """
    return json.dumps(data, indent=2, default=str).replace("</", "<\\/")


def escape(text: str) -> str:
    """Escape text for use in element content or a quoted attribute.

    Control characters are replaced as well as the four markup characters, because they are a
    different failure with the same cause. ``&<>"`` change the markup; a newline or a NUL does
    not — both are legal in an HTML document — and both still wreck the table they land in. A
    phase name is user data and can hold either, so a name of ``a\\nb\\rc\\x00d`` reached a
    ``<td>`` intact and split one cell across lines. Replacing rather than dropping keeps two
    names that differ only by a control character distinguishable on the page.
    """
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return "".join(
        "\ufffd" if (ch < " " or "\x7f" <= ch <= "\x9f" or ch in "\u2028\u2029") else ch
        for ch in escaped
    )


MAX_LABEL_DRAWN = 90
"""How much of a phase name any page draws before it is cut.

Wide enough that no ordinary path reaches it, so the common row is untouched. A name built
from data — a file path, a URL, a serialised config — has no bound at all, and an unbounded
label stretches its table until every column beside it is off-screen, once per table.
"""


def clip_label(text: str) -> str:
    """Bound a user-supplied label to what a page can draw, marking the cut.

    The tail is kept for the same reason ``report.format_label`` keeps it — the leaf is what a
    reader is looking for — and the ellipsis is what makes that safe, since an unmarked
    truncation prints a name that does not exist. Returns plain text, not markup: callers
    escape it like any other label, and the complete name remains in the embedded JSON, which
    is where a reader goes for exact values anyway.
    """
    if len(text) <= MAX_LABEL_DRAWN:
        return text
    return "…" + text[-(MAX_LABEL_DRAWN - 1):]


def tile(key: str, value: str) -> str:
    """One labelled statistic, for the rows of summary figures both reports open with."""
    return f'<div class="tile"><div class="k">{escape(key)}</div>' \
           f'<div class="v">{escape(value)}</div></div>'
