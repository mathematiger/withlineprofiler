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
| `lineprofiler trace <dir> -o trace.html` | the timeline page |
| `lineprofiler trace <dir> --max-spans N` | cap what is drawn, keeping the longest spans |
| `lineprofiler trace <dir> -q` | suppress the stderr progress lines |
| `lineprofiler trace <dir> --fail-over N` | exit non-zero if any finding costs more than N% of the run |

`--output`/`-o` writes to a path instead of stdout. It does not create parent directories: on
a command line a path that does not exist is usually a typo, and failing loudly beats
scattering directories. The library call `write_html()` does create them, since a caller
writing to `reports/run-17.html` from code means it.

Every command also works as `python -m lineprofiler ...`, for when the console script is not on `PATH`.

### From code

The same two documents can be written by the script that produced the run, without shelling out to the CLI:

```python
from lineprofiler.accounting import write_report, write_trace

write_report("profile", "reports/run-17.html", format="html")   # text | json | html
write_trace("profile", "reports/run-17-trace.html")             # html | json
```

The formats are exactly the CLI's, and an unrecognised one raises rather than quietly falling back. Both create parent directories, as `write_html()` does. Both read the trace sidecars: the findings, the occupancy and the request-lifecycle blocks are all derived from spans, so a run recorded with `trace=True` renders with everything it was instrumented for. `write_trace(..., format="json")` emits the same document `lineprofiler trace --format json` prints — one derivation, so a gate and the file beside it cannot disagree.

## The trace timeline

```
lineprofiler trace profile/ -o trace.html
```

The report answers *where did the time go?* This answers the question a set of totals
structurally cannot: *why was this worker idle at that moment, and who was it waiting for?*
A phase tree can say `queue_get` was 80% wait; only a timeline can show that the wait began
the instant the learner finished and ended when actor 3 finally published its batch.

The page is ordered as conclusions first, evidence after — you should know what is wrong
before you are asked to read a chart:

- **Findings**, ranked by how much of the run each one cost. This is the page's answer:
  "`iteration/queue_get` spent 100% of its time blocked, costing 54% of the run — released by
  `actor` on a recorded signal/wait_on pair, so this is a queue, not a hang." Each finding
  that names something on the chart carries a **show on timeline** button that zooms to it, so
  the claim and the picture stay connected.

  The queue-versus-stall verdict prefers *recorded* evidence to inference: a matched
  `signal`/`wait_on` pair names the producer outright, and only in its absence does the page
  fall back to asking how busy everyone else was during the wait. That fallback is hedged in
  the wording, because a producer working in short bursts across a long wait scores low
  against it while genuinely producing throughout.

  A parent phase that does nothing but call a blocking child is *not* reported separately: it
  inherits all of its child's wait, and saying it twice pushes the real second-place finding
  off the list. A run with nothing wrong says so rather than inventing a finding.

- **A phase summary** — Vampir's Function Summary, in this package's vocabulary. Every phase
  by total wall time across all lanes, with a `self` column excluding time inside nested
  phases, so a wrapper never out-ranks the callee that actually spent the time. The bar is
  drawn in the same blend as the chart, so scanning the column separates "expensive because it
  works" from "expensive because it waits" without reading a number. This is the fastest route
  to *what should I fix*: a phase costing 40% of the run scattered over ten thousand short
  calls is invisible on a timeline and top of this table.

- **One lane per worker thread, on a shared clock.** Idle time is drawn as absence, so a
  starved worker reads as a row full of gaps.
- **Nesting, as rows within a lane.** A phase entered inside another is drawn beneath its
  caller, one row per level, so a lane reads as a call structure and not a flat list: you can
  see that `iteration` spent its time in `queue_get` and then `train_step`, in that order.
  Deeper than eight levels folds onto the last row, and the page says how many did.
- **A "Call order" table**, restating each lane's calls in the order they ran, indented by
  depth. The chart shows order by position, which is the right way to see it and the wrong way
  to quote it — a busy lane collapses into a stripe, and this does not.
- **Wait shading inside each span**, using the same blend as the icicle chart: reddish is
  working, blue is blocked. There is a test asserting the two agree.
- **Arrows from a producer to the consumer it released**, drawn from `signal`/`wait_on`.
- **A critical path**: the chain of spans that actually set the run's length, walked backwards
  through those arrows. This is the part that turns "everything looks slow" into an ordered
  list of what waited on what.
- **A lane table separating "phase open" from "on CPU".** The gap between those two columns
  *is* the waiting.
- **GPU utilisation lanes** from the 1 Hz sampler, so an empty CPU lane can be checked against
  a busy or idle device.

### Reading the timeline

A legend sits above the chart rather than in a footnote under it, because a reader who has to
scroll past the chart to learn what its colours mean will read the chart wrong first. It names
all five conventions: on-CPU red, blocked blue, the blend between them, hatched grey for spans
whose CPU time was never measured, and the outline for the critical path.

What the controls do is written on the controls. What a **click** does is stated before you
click: it pins the span and fills a panel underneath with who released it (from the arrows),
how much of it was blocked, and what every other lane was doing while it ran — the
stall-versus-queue question answered for one span without dragging a range. Clicking it again
unpins.

| gesture | what it does |
|---|---|
| drag | pan |
| scroll | zoom about the cursor |
| hover | exact figures for one span |
| click a span | pin it; everything not causally upstream dims, and the panel names who released it |
| click a lane label | fold that lane to one row — its spans stay drawn, they stop claiming a row each |
| measure a range | what every lane was doing over a window you drag |

With the chart focused, it also drives from the keyboard — a canvas has no keyboard
affordance at all unless one is given to it:

| key | what it does |
|---|---|
| `←` `→` | pan |
| `+` `−` | zoom |
| `n` `p` | step forward/back along the critical path, re-centring and pinning each span |
| `0` | reset the zoom |
| `Esc` | unpin |

Stepping the critical path with `n` is the fastest way to read it even with a mouse: the
chain is the ordered answer to *what set this run's length*, and hunting for outlined bars
across lanes that may be far apart vertically is not.

Folding matters once a run has more than a handful of workers. With sixteen actors the chart
is taller than any screen, so the learner and an actor cannot be seen at once — which is
usually the exact comparison the page was opened to make. A folded lane keeps its slot and
its spans; it never hides activity.

### This page ships JavaScript

It is the only one that does. The constraint it relaxes is narrow: still one file, still no
CDN, no webfont and no network — a timeline over a hundred thousand spans needs pan and zoom,
and static SVG cannot provide them. The report and source pages remain script-free and their
tests still assert it.

Text is written to the page with `textContent`, never `innerHTML`. Phase names come from your
code, and a profiling artifact gets mailed around and opened by other people.

### Recording a trace

Tracing is **off by default**, because the phase tree is bounded and a timeline is not. Four
ways to turn it on, in increasing order of what they ask of your code:

| | change to your code | what you get |
|---|---|---|
| `LINEPROFILER_TRACE=auto` | **nothing** | lanes and nesting derived from function calls |
| `LINEPROFILER_TRACE=1` | **nothing** | lanes from the phases you already name |
| `Profiler(..., trace=True)` | **one word** | the same, set in code |
| `signal()` / `wait_on()` | **two lines per queue** | arrows and a cross-process critical path |

```python
profiler = Profiler(run_dir="profile", role="actor", trace=True)
```

Only the last tier needs new calls, and they go at the queue boundaries you already know are
interesting:

```python
queue.put(batch)
profiler.signal("batch", batch.id)        # in the producer

with profiler.phase("queue_get"):
    batch = queue.get()
profiler.wait_on("batch", batch.id)       # in the consumer
```

An unmatched `wait_on` is reported on the page as unmatched — never raised. Half a pipeline
being instrumented is the normal state of an incremental rollout, and it must cost you arrows
rather than a crash.

### What it cannot tell you

- **The buffer is bounded** (`trace_capacity`, 200k spans by default). A wrapped ring keeps
  the *newest* spans and the page says how many it dropped; a truncated trace never renders
  as a complete one.
- **`trace="auto"` cannot measure CPU time.** `thread_time_ns()` is a real syscall at ~590 ns,
  which is not affordable per function call. Auto spans are drawn hatched and their wait is
  reported as *unknown*, not as zero. Use it to find where the phases belong, then name them.
- **Cross-host alignment is only as good as NTP.** Within one host the shared axis is exact.
  Across hosts, treat sub-millisecond gaps as noise — the page says which case it is in.
- **`trace="auto"` needs Python 3.12+** (`sys.monitoring`) and cannot run alongside
  coverage.py, pdb or `backend="viztracer"`. It fails loudly rather than recording nothing.
