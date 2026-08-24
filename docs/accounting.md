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

Naming a `run_dir` turns the profiler on: nobody passes one and means "write nothing there". Without one, the switch is `LINEPROFILER_PROFILE=1` in the environment, which is what lets library code carry `accounting.phase(...)` calls permanently at no cost. An explicit `enabled=` beats both, in either direction, so a launcher can turn a run off without editing the call — and a profiler that ends up disabled with phases entered on it says so on `close()` rather than leaving an empty directory to be discovered later.

```
lineprofiler report profile/
lineprofiler report profile/ --no-samples     # phases only; skips the resource blocks
lineprofiler report profile/ --json           # the same run as data, for CI gates and diffs
lineprofiler compare profile_a/ profile_b/ [--json]
```

Every command also works as `python -m lineprofiler report profile/`, which needs only the package to be importable — no console script on `PATH`. From inside the script that produced the run, `write_report("profile", "report.html", format="html")` writes the same report to a file directly; `write_trace` does the same for the timeline. Both take the CLI's formats.

The report is grouped by role, because sixteen actors always dominate a single global
percentage whether or not they are the bottleneck:

```
Runtime 4h 12m   Processes 17   Roles actor x16, learner x1
Hosts node07, node08 (2 nodes)   Run 20260813T2241-471c94

ACTOR  (16 processes, imbalance 1.22)
──────────────────────────────────────────────────────────────
mcts                           82.9%       2h 31m
env_step                       17.1%          27m

DOMINANT PHASES          entries        self    wait       p50       p99
self_play/mcts                75      2h 31m     18%     9.4ms    18.5ms
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
| `accounting.phase(name, io=…, sync=…, async_work=…)` | `profiler.phase(...)` |
| `accounting.count(name, n)` | `profiler.count(...)` — for work units only; entry counts are already the report's `entries` column |
| `accounting.trace_begin(channel, key)` | `profiler.trace_begin(...)` |
| `accounting.trace_mark(channel, key, name, sample=…)` | `profiler.trace_mark(...)` |
| `accounting.trace_end(channel, key)` | `profiler.trace_end(...)` |
| `accounting.current()` | the deepest open phase, or `""` — useful on a log line |
| `accounting.installed_profiler()` | the installed instance, or `None` |

`close()` uninstalls, so a closed profiler is never resolvable and a second `install=True`
warns rather than silently taking over. A forked child resolves *its own* profiler, not the
parent's — the fork handlers re-point it along with the worker file. Explicit
`profiler.phase(...)` keeps working unchanged; `install=True` only adds a second way in.

### What it ran on

Every report opens with a `RESOURCES` section: what the run consumed, what the machines had,
and both normalised per worker process.

```
RESOURCES
──────────────────────────────────────────────────────────────
                            used     available      per proc
CPU  peak             16.2 cores      60 cores          1.35
CPU  mean              3.8 cores   (6% of box)          0.32
RAM  peak RSS           985.9 MB        2.0 TB       82.2 MB
GPU  devices                                 1

  node2: 128 cores (60 available to this job), 2.0 TB RAM, 1x NVIDIA A100-SXM4-40GB
  per-proc figures are over 12 process(es) (actor x12)
  heaviest process held 84.2 MB RSS against a 82.2 MB mean
```

This is what makes two runs comparable. Profile the same workload at four workers and at
twelve, and the per-process column answers the scaling question directly — a flat figure means
linear scaling, a rising one means contention. The gap between the heaviest process and the
mean says whether a total is evenly spread or carried by one fat worker.

The `available` column is the affinity mask when a scheduler constrained the job and the
machine's core count otherwise. Under Slurm or in a container these differ, and both are
printed: the machine total alone overstates the headroom, the quota alone hides what the box
was.

**Two VRAM rows, because there are two instruments.** `VRAM peak alloc` is the torch caching allocator's view: your tensors, and nothing else. `VRAM peak held` is what the device reports for this run's pids — the column `nvidia-smi` shows — and it additionally counts the CUDA primary context every process with a device pays for, a few hundred MB each that the allocator never sees. On a run with several workers the second dominates: 304.8 MB of allocator against 2.7 GB actually held. Ask the allocator row how big your tensors are; ask the held row whether another worker fits on the card. The held row needs `nvidia-ml-py` and is omitted — not zeroed — where the driver will not attribute memory per process, which some MIG and vGPU configurations do not.

When a role holds VRAM with no allocator activity at all, the report names it: that is a CUDA context in a process doing no GPU work, and the usual cause is `phase(sync=True)` in a worker that holds no model. `Profiler(cuda_sync=False)` is the fix, and current versions avoid creating the context in the first place.

CPU sampling needs `psutil`; device models and VRAM totals need `nvidia-ml-py`. Without them
the corresponding rows are omitted and the report says so — a resource that was never measured
is never rendered as zero. Capacity is recorded per worker, so a run spanning a fat node and a
thin one prints one line per host rather than one merged figure that describes neither.

`report_as_dict` carries the same numbers under `machine`, as `used` and `capacity_by_host`.

### What it does not do

Function-level tracing, CUDA kernel timing, GPU compute-versus-wait attribution and
per-line memory are left to `torch.profiler`, VizTracer, memray and nsys. The GPU block
reports utilisation — per device, and split into your run's share and everyone else's — which
tells you *whether* the GPU is the constraint, never *which kernel* is. `backend="torch"` gets
you that breakdown for a window.

**It does not attribute per asyncio task.** Phase statistics are per *thread*, which is what lets the hot path take no locks — and asyncio tasks share a thread. So a phase held across an `await` while another task enters the same phase is recorded as nested inside itself: every level claims the full duration, the outermost row reports one entry for however many requests were served, and past 32 levels the phases fold and record nothing at all. The numbers are wrong rather than missing, so both the profiler and the report say so — a `RuntimeWarning` when it happens, and a line in the report header for whoever opens the file later.

This matters most for an inference server, which is the common asyncio shape in an RL pipeline. Two ways to get correct numbers: put the phase around a region that does not `await` (the encode, the forward, the scatter — each measured on its own), or run one task per thread. To measure the *waiting* itself, use `signal_ready()`/`wait_on()` or the request lifecycle (`trace_begin`/`trace_mark`/`trace_end`), which are built for exactly this and record timestamps rather than a stack.

## The trace timeline

Everything above records *aggregates* — how much, how many, how long on average. When the
question is *when*, and *who was waiting for whom*, record a timeline instead:

```python
profiler = Profiler(run_dir="profile", role="actor", trace=True)
```

```
lineprofiler trace profile/ -o trace.html
```

Off by default: the phase tree is bounded by design, and a timeline is not. When on, spans go
to a fixed-capacity ring (`trace_capacity`, 200k by default) that keeps the newest and reports
what it dropped.

For arrows between processes, mark the two ends of a dependency:

```python
queue.put(batch)
profiler.signal("batch", batch.id)

with profiler.phase("queue_get"):
    batch = queue.get()
profiler.wait_on("batch", batch.id)
```

Both are no-ops when tracing is off, so they are safe to leave in permanently.

With no `phase()` calls at all, `LINEPROFILER_TRACE=auto` derives spans from function calls in
your project (3.12+). It cannot measure CPU time — the page marks those spans *unknown* rather
than claiming they never waited — so treat it as a way to find where the phases belong.

See [html-reports.md](html-reports.md) for what the page shows and
[accounting-recipes.md](accounting-recipes.md) for reading it.
