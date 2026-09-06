# The line profiler

Line-by-line tracing for a bounded region: wrap code in a `with` block and it records
per-line hit counts and timing. Only code under your project folder is traced — the folder
is auto-detected by walking up to the nearest `.git` — so the output is your code, not the
stdlib and site-packages.

The timing is `line_profiler`'s: its C callback does the measuring, and this package decides
what it watches, so nothing has to be decorated. Measured at ~240 ns per line event, the same
as calling `line_profiler` yourself. That is still ~12x the cost of running the code
uninstrumented, so it is for a region you already suspect, not a whole training run. For
that, use the accounting layer.

- **Zero configuration** – just wrap code in a `with` block, or use `start_profiling()`/
  `stop_profiling()` (see [Configuration](configuration.md#adopting-either-tool-in-two-lines))
- **Nothing to decorate** – closures, methods, scripts and modules imported inside the block
  are all found on their first call
- **Line-level timing** – see exactly which lines are slow
- **Auto-filtering** – only profiles code in your project (auto-detects git repo root), further
  narrowed by an optional `[tool.lineprofiler]` table
- **Flexible output** – sort by time, hits, or line number; filter by threshold

```python
from lineprofiler import LineProfiler
profiler = LineProfiler(project_folder="path/to/your/project")
profiler.clear()
with profiler:
  your_function()
profiler.print_global_top_stats(min_time_us=0.01, top_n=40)
```

| Method | Description |
|--------|-------------|
| `print_stats(min_time_us, top_n_lines, sort_by, stream)` | Print per-function statistics, in source order by default |
| `print_global_top_stats(top_n, min_time_us, sort_by, stream)` | Print top N lines across all functions |
| `get_stats()` | Get raw `FunctionStats` dictionary |
| `to_html(path, title)` | Write the annotated, heat-coloured source view |
| `dump_stats(path)` | Write a `.lprof` file — readable by `python -m line_profiler` and `LineStats.from_files()` |
| `region(name)` | Name the block that follows, so its lines are reported separately |
| `print_regions(top_n, min_time_us, stream)` | Print each region's slowest lines |
| `region_stats()` / `region_entries()` | Per-region `FunctionStats`, and how often each was entered |
| `clear()` / `reset()` | Clear all collected data |

| Function | Description |
|--------|-------------|
| `start_profiling(project_folder=None)` | Start ambient profiling — the two-line alternative to `with profiler:` |
| `stop_profiling(print_stats=True)` | Stop it, optionally printing the top-lines report, returning the profiler |

## Regions: which phase did the line belong to?

A line profile tells you line 52 is slow. It cannot tell you that line 52 is slow *during
selection* and fine during backpropagation, because the same line is one row however many
phases run through it. Name the phases and it can:

```python
profiler = LineProfiler()
with profiler:
    for _ in range(iterations):
        with profiler.region("select"):
            node, state = select(root, root_state)
        with profiler.region("rollout"):
            reward = rollout(state)

profiler.print_regions()
```

```
==================================================================================================================================
Region: select  —  2086.0 µs, 62.8% of profiled time, 200 entries, 10.4 µs each
==================================================================================================================================
File::Function                                     Line   Hits       Time (µs)     Per Hit (µs)   % Region  Line Content
----------------------------------------------------------------------------------------------------------------------------------
sample.py::score_children                          8      1600       585.0         0.4            22.2          explore = c * math.sqrt(math....
sample.py::score_children                          6      1800       503.0         0.3            19.1      for child in children:
```

The same name may be entered any number of times and accumulates, and `region_entries()` says
how often — which is what turns a total into a cost per iteration.

**Regions nest, and the reading is inclusive.** A line inside `rollout` is billed to `rollout`
and to every region open around it, the same way a phase's wall time in the accounting layer
includes its children.

**The shares are approximate, and that is stated on the report.** Three separate reasons stop
them summing to 100%: regions may nest, so their totals overlap; they need not cover the whole
run, so there may be a gap; and under the C engine each region is timed by *its own*
`line_profiler` instance, which reads the clock a few tens of nanoseconds after — or before —
the session's on every line event, depending on the order the engine's instance set happens to
iterate. Measured at roughly **46 ns per line event**, which is invisible on real work and
around a fifth of a very short synthetic loop, and it can push a single region slightly past
100%. The µs-per-entry column beside the share is the figure that survives a rerun, and the
hit counts are exact under both engines.

**A region is a window, not a call stack.** Every line executed while it is open is billed to
it — including lines in the frame that opened it, and including lines on other threads.
Opening regions concurrently on several threads therefore does not mean anything useful.

**Entering a region while the profiler is not active records nothing** and costs one boolean
test, so the calls are safe to leave in code that is usually not profiled.

### What a region costs

| Engine | Per line event, region open | Per boundary crossed |
|---|---|---|
| `line_profiler` | ~400 ns (from ~240) | ~7.6 µs |
| `builtin`, `sys.monitoring` | ~1,435 ns (from ~900) | ~430 ns |
| `builtin`, `sys.settrace` | ~1,420 ns | ~1,760 ns |

The engines trade places, and which one wins depends on the shape of your instrumentation.
The C engine keeps a second profiler per region and pays to switch it on and off, so its
boundary is expensive; the pure-Python engine only appends to a list at the boundary and pays
on every line instead. **Many small regions favour the builtin engine; a few regions around
substantial work favour the C engine** — and that is the usual case, so it is the default.

The placement rule is the accounting layer's: put a region where the entry count is bounded by
your loop, not by your data. Two hundred entries costing 7.6 µs each is 1.5 ms, which is noise
next to a profiled run; two hundred thousand is not.

The C engine's region total also *excludes* the cost of opening and closing it, which the
session's total absorbs into the surrounding line. That is the cleaner number, and it is why a
run's regions can sum to slightly less than the session.

## Engines

`LineProfiler(engine=...)` selects who does the timing.

| `engine` | What runs | Cost per line event |
|---|---|---|
| `"line_profiler"` (default where installed) | `line_profiler`'s C callback, fed by a `sys.monitoring` discovery hook | ~240 ns |
| `"builtin"` | this package's pure-Python callback on `sys.monitoring` (3.12+) or `sys.settrace` | ~895 ns / ~915 ns |

Both bill a line that calls a function for the whole call, both keep per-thread state, and both
feed the same reports — the engine changes what the measurement costs, not what it means.
Passing `backend="monitoring"` or `backend="settrace"` selects the builtin engine.

Below Python 3.12 the C engine has no discovery hook, so it registers the functions of every
in-project module already imported and misses anything imported later; on 3.12+ everything is
found on its first call, closures and `runpy` scripts included.

## Running a script without editing it

    lineprofiler run script.py [args...] [--top N] [--functions] [--html report.html]

The `kernprof` equivalent, with no decorators to add: the script runs normally, everything
under its project folder is profiled, its exit status is preserved, and the summary prints even
if it raises.

## What it does not do

- Two functions with identical bytecode whose line numbers overlap cannot be told apart by the
  C engine's index, so the second one found is left unprofiled rather than reported as a blend
  of the two. Reach for `engine="builtin"`, which keys on the code object instead.
- `sys.settrace` is global and single-tracer, so that backend cannot run alongside another
  tracing profiler (including `accounting`'s `backend=` window). The `sys.monitoring` ones can.
- No memory, native-time or GPU attribution. That is [Scalene](https://github.com/plasma-umass/scalene)'s
  job and it does it well; see [the comparison](comparison.md).
