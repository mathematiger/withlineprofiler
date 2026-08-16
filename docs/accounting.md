# The accounting layer

Semantic phase accounting for long, multi-process runs. Aggregates only — counts, sums and a
fixed-bucket histogram per region — so memory per phase is constant no matter how long the
run lasts, and cheap enough to leave enabled for twelve hours.

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
lineprofiler report profile/ --no-samples     # phases only; skips the resource blocks
lineprofiler report profile/ --json           # the same run as data, for CI gates and diffs
lineprofiler compare profile_a/ profile_b/ [--json]
```

The report is grouped by role, because sixteen actors always dominate a single global
percentage whether or not they are the bottleneck:

```
Runtime 4h 12m   Processes 17   Roles actor x16, learner x1
Hosts node07, node08 (2 nodes)   Run 20260813T2241-471c94

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
  iteration/load_batch      r   408.7 MB   w        0 B    327.6 MB/s
                            + 359.3 MB read from page cache
  iteration/checkpoint      r        0 B   w    40.0 MB    785.7 MB/s

I/O
──────────────────────────────────────────────────────────────
Read (from disk)                  408.7 MB        1.1 GB/s
Read (from page cache)            359.3 MB
Write                              40.0 MB      108.2 MB/s
  write █     █      █     █      █                 peak 1.0 GB/s

GPU
──────────────────────────────────────────────────────────────
                               busy   this run     idle
GPU 0                         92.4%      64.1%     7.6%
GPU 1                          8.0%        n/a    92.0%
VRAM allocated (peak)               6.1 GB
VRAM reserved (peak)                8.4 GB
```

`wait%` is the share of wall time the thread was not running on a CPU — blocked on a queue,
a lock, the GIL or a syscall. In a queue-driven pipeline it is usually the number that
explains the run. It pairs with the phase's **wall** time, never with its self time: waiting
inside a child still counts, so `wait / self` exceeds 100% for any phase wrapping a blocking
call.

### Instrumenting without threading a profiler argument

`profiler.phase(...)` needs the object. Reaching a function five call levels down therefore
means adding a `profiler` parameter to every caller in between — the search, the episode loop,
the actor session, the inference server — for one phase.

Pass `install=True` and use the module-level functions instead:

```python
from lineprofiler import accounting
from lineprofiler.accounting import Profiler

# once, wherever you set the run up
profiler = Profiler(run_dir="profile", role="actor", install=True)

# anywhere at all, with no argument threaded to it
def uct_search(root):
    with accounting.phase("mcts"):
        accounting.count("simulations", 64)
```

**With no profiler installed these do nothing**, at ~300 ns per call — the same cost as
`enabled=False` — so library code can carry the calls permanently whether or not anything is
profiling it. Resolving the installed profiler costs ~38 ns on an enabled phase, about 1%,
which is why this is not a reason to keep passing the object around.

| Function | Equivalent |
|---|---|
| `accounting.phase(name, io=…, sync=…)` | `profiler.phase(...)` |
| `accounting.count(name, n)` | `profiler.count(...)` |
| `accounting.current()` | the deepest open phase, or `""` — useful on a log line |
| `accounting.installed_profiler()` | the installed instance, or `None` |

`close()` uninstalls, so a closed profiler is never resolvable and a second `install=True`
warns rather than silently taking over. A forked child resolves *its own* profiler, not the
parent's — the fork handlers re-point it along with the worker file. Explicit
`profiler.phase(...)` keeps working unchanged; `install=True` only adds a second way in.

### What it does not do

Function-level tracing, CUDA kernel timing, GPU compute-versus-wait attribution and
per-line memory are left to `torch.profiler`, VizTracer, memray and nsys. The GPU block
reports utilisation — per device, and split into your run's share and everyone else's — which
tells you *whether* the GPU is the constraint, never *which kernel* is. `backend="torch"` gets
you that breakdown for a window.
