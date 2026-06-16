# lineprofiler
Statistical profiler to find lines that take a long time to compute. One can specify a folder, wherein the profiler traces lines.
The profiler can be bound using `with`.

## Features
- **Zero configuration** – just wrap code in a `with` block
- **Line-level timing** – see exactly which lines are slow
- **Auto-filtering** – only profiles code in your project (auto-detects git repo root)
- **Flexible output** – sort by time, hits, or line number; filter by threshold

## Installation
`pip install with-line-profiler`

## Workflow
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

## Licence
MIT

The claude.md is partially created from https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md
