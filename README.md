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

## Accounting layer (for long RL runs)

`lineprofiler.accounting` is a separate, always-on layer for multi-hour, multi-process
training runs, where line-level tracing is far too expensive. You name the regions; it
records aggregates only — counts, sums and a fixed-bucket histogram per region — so memory
per phase is constant no matter how long the run lasts.

Every process in the pipeline constructs one, tagged with the *role* it plays. Roles are
free-form strings: whatever your architecture calls its processes.

```python
from lineprofiler.accounting import Profiler

# in each actor process
profiler = Profiler(run_dir="profile", role="actor")   # or LINEPROFILER_ROLE=actor
with profiler:
    for _ in range(steps):
        with profiler.phase("iteration"), profiler.phase("self_play"):
            with profiler.phase("mcts"):
                profiler.count("mcts_simulations", 64)

# in the learner process
profiler = Profiler(run_dir="profile", role="learner")
with profiler.phase("checkpoint", io=True):            # exact byte attribution
    save(model)
```

```
lineprofiler report profile/
lineprofiler compare profile_a/ profile_b/ [--json]
```

The report is grouped by role, because sixteen actors always dominate a single global
percentage whether or not they are the bottleneck:

```
Runtime 4h 12m   Processes 17   Roles actor x16, learner x1

ACTOR  (16 processes, imbalance 1.22)
──────────────────────────────────────────────────────────────
mcts                           82.9%       2h 31m
env_step                       17.1%          27m

DOMINANT PHASES                     self    wait       p50       p99
self_play/mcts                    2h 31m     18%     9.4ms    18.5ms
    + mcts_simulations           4,800     6,377.0/s   156.8us/ea

ITERATIONS  (75 entries)
  mean     12.1ms   p50     11.4ms   p95     19.0ms   p99     20.6ms

I/O BY PHASE (measured exactly)
──────────────────────────────────────────────────────────────
  iteration/checkpoint      r        0 B   w    40.0 MB    785.7 MB/s

I/O
──────────────────────────────────────────────────────────────
Write                              40.0 MB      108.2 MB/s
  write █     █      █     █      █                 peak 1.0 GB/s

GPU
──────────────────────────────────────────────────────────────
Utilisation (sampled)              71.0%
VRAM reserved (peak)               8.4 GB
```

`wait%` is the share of wall time the thread was not running on a CPU — blocked on a queue,
a lock, the GIL or a syscall. In a queue-driven pipeline it is usually the number that
explains the run.

### Finding I/O bottlenecks

Two blocks, with different guarantees:

- **`I/O BY PHASE (measured exactly)`** — phases opened with `phase(name, io=True)`. The
  byte counters are read at that phase's own entry and exit, so there is no attribution
  ambiguity. This is the block to trust.
- **`I/O`** — derived from the 1 Hz sampler. Totals are exact, but attribution to a phase
  has a resolution of one sample interval, so a 10 ms checkpoint inside a 1 s interval lands
  on whichever phase was open when the interval began. The sparkline shows *when* the bytes
  moved, which a total alone hides.

Per-operation attribution needs eBPF and is out of scope. Per-process I/O *time* is not
exposed by any OS counter — use a phase's `wait%` as the blocked-time proxy instead.

### Overhead

Measured on Python 3.12 with `benchmarks/bench_accounting.py`, per phase enter+exit:

| | ns/call |
|---|---|
| `phase()`, `enabled=False` | 320 |
| `phase()`, `enabled=True`, `measure_cpu=False` | 1840 |
| `phase()`, `enabled=True`, `measure_cpu=True` | 3390 |
| `phase(io=True)` | 40300 |
| `count()` | 345 |

`measure_cpu` (on by default) is what produces `wait%`, and it doubles the cost:
`time.thread_time_ns()` reads `CLOCK_THREAD_CPUTIME_ID`, which is not in the vDSO, so each
call is a real syscall at roughly 590 ns. `io=True` costs two `/proc` reads — negligible on
a 10 ms checkpoint, ruinous on an inner loop.

At around a microsecond per phase, put `phase()` around searches, env steps and train
steps — not inside an inner simulation loop. Use `count()` there instead; it is five times
cheaper and gives you the rate anyway.

### Multiple processes

Every process writes its own `w_<pid>_<uuid8>.json`; `report` merges them. The uuid matters
because a restarted worker reuses its rank but not its pid.

`spawn`, `fork` and `forkserver` are all supported and tested at 1, 4 and 16 workers. A
worker that raises still contributes everything it recorded before dying; a worker
`SIGKILL`ed before its first flush leaves nothing, and the report says so rather than
under-reporting silently.

Enabling a profiler sets `LINEPROFILER_PROFILE=1` and `LINEPROFILER_RUN_DIR` in the
environment, so `Profiler(role="actor")` in a spawned worker joins the parent's run with no
configuration threaded through. **`forkserver` is the exception**: its daemon is forked once
and its children inherit the daemon's environment as it was when the daemon started, so
export `LINEPROFILER_PROFILE=1` in the shell before training, or pass `enabled` and
`run_dir` to each worker explicitly.

Forking is handled: a forked child gets its own file, an empty tree and a clean phase stack,
and the profiler's own threads are stopped for the duration of the fork so that enabling it
never adds fork-deadlock risk to a codebase that forks.

### Heavy profilers, for a bounded window

```python
Profiler(run_dir="profile", backend="torch", backend_window=(100, 110),
         window_phase="iteration")
```

Starts `torch.profiler` on the 100th entry into `iteration` and stops it at the 110th,
writing a Chrome trace into `profile/backend/`. `backend` is a single enum value, so two
heavy profilers cannot be active at once — they contend for the same interpreter hooks.

`Profiler(..., annotate=True)` additionally wraps every phase in an NVTX range and a
`torch.profiler.record_function`, so an externally started `nsys profile` or Kineto capture
shows your phase names. This package never launches nsys itself.

### What it deliberately does not do

Function-level tracing, CUDA kernel timing, GPU compute-versus-wait attribution and
per-line memory are left to `torch.profiler`, VizTracer, memray and nsys. The GPU block
reports NVML's whole-device *busy* percentage, which is a utilisation number, not a
breakdown. `backend="torch"` gets you the breakdown for a window.

### Optional dependencies

`psutil` (memory and I/O), `torch` (CUDA memory), `nvidia-ml-py` (GPU utilisation),
`viztracer`. Each is imported lazily behind a capability check; a missing one disables one
block of the report and never raises.

## Licence
MIT

The claude.md is partially created from https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md
