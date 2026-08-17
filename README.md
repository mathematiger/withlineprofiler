# with-line-profiler

[![PyPI](https://img.shields.io/pypi/v/with-line-profiler.svg)](https://pypi.org/project/with-line-profiler/)
[![CI](https://github.com/mathematiger/withlineprofiler/actions/workflows/ci.yml/badge.svg)](https://github.com/mathematiger/withlineprofiler/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-informational.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/with-line-profiler.svg)](https://pypi.org/project/with-line-profiler/)

Two independent profiling tools in one distribution: line-by-line tracing for a region you
suspect, and low-overhead phase accounting for a run too long to trace. Zero dependencies on
Python 3.11+, MIT licensed.

```
pip install with-line-profiler
```

```python
# example.py
from lineprofiler import LineProfiler

def slow_function():
    total = 0
    for i in range(1_000_000):
        total += i * i
    return total

profiler = LineProfiler()
with profiler:
    slow_function()
profiler.print_stats()
```

```
====================================================================================================
File: /path/to/example.py
Function: slow_function at line 3
Total time: 710676.0 µs
====================================================================================================
Line #   Hits       Time (µs)       Per Hit (µs)    % Time     Line Content
----------------------------------------------------------------------------------------------------
6        1000000    364848.9        0.4             51.3       total += i * i
5        1000001    345825.8        0.3             48.7       for i in range(1_000_000):
7        1          0.7             0.7             0.0        return total
4        1          0.6             0.6             0.0        total = 0
```

(numbers vary by machine; `File:` prints the absolute path)

No decorators to add, no separate `kernprof` run, no build step. Only code under your project
folder is traced — the folder is auto-detected by walking up to the nearest `.git` — so the
output is your code, not the standard library.

Want a picture instead? `profiler.to_html("profile.html")` writes an annotated,
heat-coloured source view as a single self-contained file.

## Which tool

| Tool | What it does | Use it when |
|---|---|---|
| **`lineprofiler.LineProfiler`** | Line-by-line tracing for a bounded region, scoped to your project folder. | You have narrowed the problem down and want per-line timings inside it. |
| **`lineprofiler.accounting`** | Semantic accounting for regions *you* name. Aggregates only — counts, sums, a fixed-bucket histogram — at ~2 µs per phase, across every process in a pipeline. | You are profiling a long, multi-process training run and need to know which phase, which role and which node the time went to. |

They share nothing but the distribution: `accounting` never imports `LineProfiler`. If you
arrived here for a training run, you want the accounting layer — it is the one built to stay
enabled for twelve hours.

```python
from lineprofiler.accounting import start, stop

start(role="actor")     # or Profiler(...) / with profiler.phase(...)
...
stop()
```

```
lineprofiler report profile/                        # the run, as a table
lineprofiler report profile/ --format html -o r.html # ...or as a page you can share
```

### Why was that worker idle?

A report says `queue_get` was 80% wait. It cannot say *when*, or *what for* — a total has no
position on a clock. Turn on the timeline and the answer is a picture:

```python
Profiler(run_dir="profile", role="actor", trace=True)
```

```
lineprofiler trace profile/ -o trace.html
```

One lane per worker on a shared clock, idle time drawn as absence, and — where you mark a
queue with `signal()` / `wait_on()` — arrows from a producer to the consumer it unblocked,
plus the critical path that actually set the run's length.

Nothing to instrument first: `LINEPROFILER_TRACE=auto` derives the lanes from function calls
in your project, with no change to your code at all.

## Documentation

- [The line profiler](https://github.com/mathematiger/withlineprofiler/blob/main/docs/line-profiler.md) — the `with` block, `start_profiling()`, and what it does not do
- [The accounting layer](https://github.com/mathematiger/withlineprofiler/blob/main/docs/accounting.md) — phases, counters, and instrumenting without threading an argument
- [Accounting recipes](https://github.com/mathematiger/withlineprofiler/blob/main/docs/accounting-recipes.md) — reading the report, I/O and GPU bottlenecks, overhead budgets, exporting to W&B
- [Multiple processes and nodes](https://github.com/mathematiger/withlineprofiler/blob/main/docs/multiprocess.md) — Slurm, forking, preemption, heavy backends
- [HTML reports](https://github.com/mathematiger/withlineprofiler/blob/main/docs/html-reports.md) — the icicle chart, the trace timeline, the annotated source view, and the embedded data block
- [Configuration](https://github.com/mathematiger/withlineprofiler/blob/main/docs/configuration.md) — environment variables, `[tool.lineprofiler]`, optional dependencies
- [Comparison with other profilers](https://github.com/mathematiger/withlineprofiler/blob/main/docs/comparison.md) — `line_profiler`, py-spy, Scalene, VizTracer, and when to use those instead

## Python support

3.10 and newer. On 3.12+ the line profiler uses `sys.monitoring`, so it can run alongside
coverage.py, pdb and other tracing tools; below that it falls back to `sys.settrace`, which
is a single global hook and cannot. `tomli` is required only on 3.10, where `tomllib` is not
yet in the standard library.

## Licence

MIT

The claude.md is partially created from https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md
