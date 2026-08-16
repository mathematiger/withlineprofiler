# The line profiler

Line-by-line tracing for a bounded region: wrap code in a `with` block and it records
per-line hit counts and timing. Only code under your project folder is traced — the folder
is auto-detected by walking up to the nearest `.git` — so the output is your code, not the
stdlib and site-packages.

This is the expensive one. `sys.settrace` fires on every line of every in-project frame, so
it is for a region you already suspect, not for a whole training run. For that, use the
accounting layer above.

- **Zero configuration** – just wrap code in a `with` block, or use `start_profiling()`/
  `stop_profiling()` (see [Configuration](configuration.md#adopting-either-tool-in-two-lines))
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
| `print_stats(min_time_us, top_n_lines, sort_by)` | Print per-function statistics |
| `print_global_top_stats(top_n, min_time_us, sort_by)` | Print top N lines across all functions |
| `get_stats()` | Get raw `FunctionStats` dictionary |
| `clear()` / `reset()` | Clear all collected data |

| Function | Description |
|--------|-------------|
| `start_profiling(project_folder=None)` | Start ambient profiling — the two-line alternative to `with profiler:` |
| `stop_profiling(print_stats=True)` | Stop it, optionally printing the top-lines report, returning the profiler |

`sys.settrace` is global and single-tracer, so this profiler is not thread-safe and cannot
run alongside another tracing profiler (including `accounting`'s `backend=` window).
