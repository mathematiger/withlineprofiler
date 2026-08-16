# HTML reports

Both tools can write a self-contained HTML page: one file, inline styles, no CDN, no webfont
and no script. It opens offline, months later, on a machine that has never heard of this
package — which is what you want from something you attach to a ticket.

## The accounting report

```
lineprofiler report profile/ --format html -o report.html
```

The page carries:

- **An icicle chart of the phase tree.** Width is wall time. Colour is the share of that time
  the phase spent *waiting* rather than running, because in a queue-driven pipeline the
  blocked phase is usually the answer and it is invisible in a chart coloured by name.
- **The pipeline breakdown** per role, the same table the text report prints.
- **I/O, GPU and memory** blocks, with the read and write rate series drawn as sparklines.
- **Caveats**, when a run lost a worker or merged a superseded attempt. A run that is not
  complete must not look complete here either.

Children in the chart are scaled to fit inside their parent. A phase entered recursively, or
from two threads, honestly reports more child wall time than parent wall time; drawing that
as an overflow would read as a measurement error rather than the aggregation it is.

### The embedded data block

Every page carries the `--format json` document in a
`<script type="application/json" id="lineprofiler-data">` block. It is embedded *alongside*
the figures rather than used to draw them, and there is a test asserting the two match. So
you can pull the exact numbers behind any bar out of the page itself:

```python
import json, re
html = open("report.html").read()
block = re.search(
    r'<script type="application/json" id="lineprofiler-data">(.*?)</script>',
    html, re.DOTALL,
).group(1)
data = json.loads(block.replace("<\\/", "</"))
print(data["roles"][0]["phases"])
```

(`</` is escaped on the way in, because the HTML parser looks for the closing tag textually
and a phase named `</script>` would otherwise end the block early.)

## The line-profiler report

```python
from lineprofiler import LineProfiler

profiler = LineProfiler()
with profiler:
    your_function()
profiler.to_html("profile.html")
```

The page shows a **hotspots table** — the slowest individual lines across every profiled
function — followed by each function's full annotated source.

Two choices worth knowing:

- **Heat is scaled per function, not globally.** A function's own slowest line is its
  darkest, so a cheap function stays readable instead of washing out next to an expensive
  one. The cross-function ranking is the hotspots table; the two views answer different
  questions and neither compromises for the other.
- **Lines that never ran are shown greyed, not omitted.** A zero-hit line inside a profiled
  function means a branch was not taken, which is worth seeing — and dropping it would break
  the flow of the code you are reading.

## Formats

| Command | Output |
|---|---|
| `lineprofiler report <dir>` | text table, to stdout |
| `lineprofiler report <dir> --format json` | the machine-readable document, for CI gates and diffs |
| `lineprofiler report <dir> --format html -o out.html` | the page above |
| `lineprofiler compare <a> <b> [--format json]` | what changed between two runs |

`--output`/`-o` writes to a path instead of stdout. It does not create parent directories: on
a command line a path that does not exist is usually a typo, and failing loudly beats
scattering directories. The library call `write_html()` does create them, since a caller
writing to `reports/run-17.html` from code means it.
