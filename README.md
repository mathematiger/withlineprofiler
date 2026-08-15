# with-line-profiler

[![PyPI](https://img.shields.io/pypi/v/with-line-profiler.svg)](https://pypi.org/project/with-line-profiler/)
[![CI](https://github.com/mathematiger/withlineprofiler/actions/workflows/ci.yml/badge.svg)](https://github.com/mathematiger/withlineprofiler/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-informational.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/with-line-profiler.svg)](https://pypi.org/project/with-line-profiler/)

Two independent profiling tools in one distribution: line-by-line tracing for a region you
suspect, and low-overhead phase accounting for a run too long to trace. Zero required
dependencies, MIT licensed.

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

That's the whole API surface for a first try — no decorators to add to the function, no
separate `kernprof` invocation, no build step. `project_folder` auto-detects your nearest git
repo root, so only your own code is traced; the standard library and installed packages are
skipped automatically. See [Adopting either tool in two lines](#adopting-either-tool-in-two-lines)
for dropping this into a script you don't want to restructure around a `with` block, or
[vs. `line_profiler`](#vs-line_profiler) if you're comparing against the incumbent.

| Tool | What it does | Use it when |
|---|---|---|
| **`lineprofiler.accounting`** | Semantic accounting for regions *you* name. Aggregates only — counts, sums, a fixed-bucket histogram — at ~2 µs per phase, across every process in a pipeline. | You are profiling a long, multi-process training run and need to know which phase, which role and which node the time went to. |
| **`lineprofiler.LineProfiler`** | Line-by-line tracing inside a `with` block, scoped to your project folder. | You have narrowed the problem to one region and want per-line timings inside it. |

They share nothing but the distribution: `accounting` never imports `LineProfiler`. If you
arrived here for a training run, you want the accounting layer — it is the one built to stay
enabled for twelve hours.

## Installation

```
pip install with-line-profiler
```

See [Optional dependencies](#optional-dependencies) for the extras that enable the memory,
I/O and GPU blocks.

## vs. `line_profiler`

The most likely thing you've already tried is [`line_profiler`](https://github.com/pyutils/line_profiler)
(`kernprof`). Both give you per-line hit counts and timings; the difference is in what you have
to do to your code first:

| | `line_profiler` | `lineprofiler.LineProfiler` |
|---|---|---|
| Mark functions to profile | `@profile` decorator on each one (or `kernprof -l`, which injects it) | Nothing — every function under your project folder while the `with` block/`start_profiling()` is active |
| Run it | `kernprof -lv script.py`, a separate invocation | Two lines inside your existing script, run it normally |
| Mechanism | C-accelerated line tracer | Pure-Python `sys.settrace` |
| Overhead | Lower — built for leaving `@profile` on hot code | Higher — meant for a bounded region, not a whole run |
| Notebook/REPL use | Needs the `%lprun` IPython extension | `with profiler:` works as-is, no extension |

If you already know exactly which function is slow and want the lowest possible overhead,
`line_profiler` is the better tool. If you want to point at a *region* of code — including code
you didn't write and can't add a decorator to — without a separate build/run step, that's what
this package is for.

## Adopting either tool in two lines

Both tools also come as a module-level entry/exit pair — the alternative to a `with` block for
dropping either one into a function you do not want to restructure:

```python
from lineprofiler.accounting import start, stop

start(role="actor")        # line 1: top of the region
...                         # existing code, unmodified
stop()                      # line 2: bottom of the region
```

```python
from lineprofiler import start_profiling, stop_profiling

start_profiling()          # line 1
...                         # existing code, unmodified
stop_profiling()           # line 2
```

Both are **opt-in and off by default**: with nothing configured, `start_profiling()` /
`accounting.start()` cost a no-op check and nothing else, so the two lines are safe to leave
committed permanently rather than added and removed per debugging session. Turn profiling on
with:

```
LINEPROFILER_ENABLED=1   # master switch for start_profiling()/stop_profiling()
```

`accounting.start()`/`stop()` use `accounting`'s own existing `LINEPROFILER_PROFILE` switch
(see below) — the two tools' switches are independent, matching the fact that they share
nothing but the distribution.

To scope `start_profiling()` to part of the codebase without touching a single call site, add
an optional table to `pyproject.toml`:

```toml
[tool.lineprofiler]
include = ["src/mypkg/**"]        # only these paths are traced (glob, relative to the project root)
exclude = ["src/mypkg/generated/**"]  # exclude wins over include
functions = ["*.train_step", "*.forward"]  # only these function/method names (glob over __qualname__)
```

All three are optional and default to "everything under the project root" — the same behavior
as constructing `LineProfiler()` directly. This table is read once per process and cached; it
never runs on the hot per-line path.

`with profiler:` (below) remains the better choice inside a Jupyter notebook or any region
you're actively iterating on interactively — the two APIs are interchangeable and neither is
deprecated by the other.

## Accounting layer (`lineprofiler.accounting`)

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

### Finding I/O bottlenecks

Two blocks, with different guarantees:

- **`I/O BY PHASE (measured exactly)`** — phases opened with `phase(name, io=True)`. The
  byte counters are read at that phase's own entry and exit, so there is no attribution
  ambiguity. This is the block to trust.
- **`I/O`** — derived from the 1 Hz sampler. Totals are exact, but attribution to a phase
  has a resolution of one sample interval, so a 10 ms checkpoint inside a 1 s interval lands
  on whichever phase was open when the interval began. The sparkline shows *when* the bytes
  moved, which a total alone hides.

Both blocks report two layers, because on Linux they answer different questions:

| | counter | measures |
|---|---|---|
| `from disk` | `read_bytes`/`write_bytes` | traffic that reached the block device |
| `from page cache` | `read_chars` − `read_bytes` | bytes your program read that RAM served |

**A warm dataset moves no disk bytes at all.** If your shards fit in page cache, the disk
counter correctly reports zero while the loader is still copying gigabytes — so a run that
looks I/O-free by `read_bytes` alone may still be loader-bound. The cache line is what tells
you the reads happened; `wait%` on the same phase tells you whether they cost you anything.

Bytes moved while no phase was open are labelled `(no phase open)`, never billed to the
root, and the block prints what share of traffic landed there. A high share means the sample
interval was too coarse for this run, not that the root did the work — wrap those regions in
`io=True` and read the exact block instead.

The profiler excludes its own sample and snapshot writes from both layers. It measures them
rather than estimating: rewriting a 500-byte worker file costs whole blocks for data, inode
and journal, which was measured at eight times the bytes handed to `write()`.

Per-operation attribution needs eBPF and is out of scope. Per-process I/O *time* is not
exposed by any OS counter — use a phase's `wait%` as the blocked-time proxy instead. Note
also that `write_bytes` is writeback-dependent: bytes land on the phase that was open when
the kernel flushed them unless that phase calls `fsync`, whereas `write_chars` is charged to
the phase that called `write`.

### Finding GPU bottlenecks

The GPU block reports every device NVML can see, in two columns that answer different
questions:

- **`busy`** — `nvmlDeviceGetUtilizationRates`, the share of time *any* kernel from *any*
  process was resident on that device. On a shared node this includes other tenants.
- **`this run`** — `nvmlDeviceGetProcessUtilization` summed over this run's own pids. Sixteen
  actors on one device add up here, which is the point: it is the load *you* place on it.

A device at 92% busy of which your run owns 64% is contended, not saturated, and the fix is
scheduling rather than a faster kernel. `n/a` means NVML never attributed a sample to your
pids on that device — no work of yours ran there, as distinct from `0.0%`, which is measured
idleness. Windows in which NVML reports no kernels for a pid count as zero for that pid, so a
worker busy one second in ten reads 10%, not 100%.

Indices are NVML's, so they are the machine's physical devices; `CUDA_VISIBLE_DEVICES`
renumbers what torch sees but not what appears here.

#### Phase timings and asynchronous CUDA

CUDA launches are asynchronous, so by default a phase around a forward pass measures the time
to *enqueue* its kernels. Measured on an A100, the same matmul chain reports **0.40 ms**
unsynchronised against **687 ms** synchronised — a factor of 1,718. Their real cost surfaces later, as `wait%` on whichever phase
happens to synchronise — usually one that copies a result back and did nothing wrong. When you
want a phase's wall time to mean GPU time:

```python
with profiler.phase("forward", sync=True):
    logits = model(batch)
```

`sync=True` drains the queue at *both* ends of the phase. Entry matters as much as exit:
synchronising only on exit bills this phase for whatever an earlier one left queued. It is a
no-op when torch is absent or no CUDA device is visible.

The cost is the pipelining you give up — across that boundary the CPU can no longer run ahead
of the GPU — so put it on the phases you are actively measuring, not on every phase in the
loop, and take your headline timings from a run with it off once you know where the work is.
Note also that `wait%` on a synchronised phase depends on how the driver waits: a blocking
sync shows up as wait, a spinning one burns CPU and shows up as none.

Kernel-level attribution is still `backend="torch"`; this only fixes which phase the time
lands on.

### When a measurement is missing

The layer distinguishes "measured zero" from "could not measure", because conflating them
produces confident wrong numbers rather than obvious gaps:

- A sample interval whose OS counters could not be read contributes **no** bytes, and the I/O
  block says how many intervals were dropped. Differencing across such a gap used to bill a
  phase for the process's whole cumulative traffic — hundreds of gigabytes from one failed
  `/proc` read.
- A `phase(io=True)` whose boundary read failed records nothing rather than a fabricated
  delta.
- A worker whose snapshots were failing, or that stopped writing well before the run ended, is
  named under `CAVEATS`. Its file still parses, so staleness is derived at report time.
- A worker file that cannot be read costs that worker, not the run.

### Overhead

Measured on Python 3.12 with `benchmarks/bench_accounting.py`, per phase enter+exit:

| | ns/call |
|---|---|
| `phase()`, `enabled=False` | 359 |
| `accounting.phase()`, nothing installed | 350 |
| `accounting.phase()` vs `profiler.phase()`, enabled | +38 |
| `phase()`, `enabled=True`, `measure_cpu=False` | 2349 |
| `phase()`, `enabled=True`, `measure_cpu=True` | 3909 |
| `phase(io=True)` | 44322 |
| `phase(sample=0.01)`, skipped entry | 1156 |
| `count()` | 384 |

`sync=True` is absent from the table because its cost is not the profiler's: it is however
long the GPU still had to run. Phases that do *not* set it are unaffected — the check is one
branch, inside the noise of the numbers above.

`measure_cpu` (on by default) is what produces `wait%`, and it doubles the cost:
`time.thread_time_ns()` reads `CLOCK_THREAD_CPUTIME_ID`, which is not in the vDSO, so each
call is a real syscall at roughly 590 ns. `io=True` costs two `/proc` reads and two reads of
the overhead counter — negligible on a 10 ms checkpoint, ruinous on an inner loop.

**Budget it as a ratio, not as a rule about loops: keep phase overhead under ~1% of the region
you are measuring.** At ~2 µs per phase that means a phase is affordable around anything taking
more than ~200 µs, and the table above is there so you can decide per call site without
measuring.

Worked example, from an MCTS search: 250 simulations × 3 phases (select / expand / backup) ×
~2 µs is 1.5 ms against a 2.4 s search — 0.06%, comfortably worth it. And that select/expand/
backup split is exactly what says *where* a slow search went, which `count()` cannot tell you:
counters give rates, not attribution.

So the rule is the budget, not the nesting depth. Where the ratio does not clear — a loop body
of a few microseconds — use `count()` instead; it is five times cheaper and gives you the rate
anyway.

#### Sampled phases, when you want the split and cannot afford it

`phase(name, sample=0.01)` measures one entry in a hundred and scales the result, for a region
worth breaking down but too hot to instrument at full rate.

```python
for _ in range(250):
    with profiler.phase("select", sample=0.01):
        ...
```

**Read the measured saving before reaching for it — it is not the sampling rate:**

| | ns/call |
|---|---|
| `phase()`, every entry measured (default `measure_cpu=True`) | 3909 |
| `phase(sample=0.01)` | 1156 |
| `count()` | 384 |

About **3.4x**, not 100x. What a phase costs is mostly Python — the call, the scope object,
the context-manager protocol — and sampling can only avoid the measurement, not the call. So:

- If you want a **rate**, `count()` is still three times cheaper than a sampled phase. Use it.
- If you want **attribution** — *which* of select/expand/backup the search went into, which a
  counter cannot answer — a sampled phase buys you that for a third of the price.

Everything derived from a sampled phase is an **estimate**, and the report says so: the row is
prefixed `~` and a note names the rate.

```
DOMINANT PHASES                     self    wait       p50       p99
~uct_search                       2h 31m     18%     9.4ms    18.5ms

  ~ = estimated from a sample, not measured. Totals are scaled by the rate:
      uct_search              1 entry in 100
```

That labelling is the condition on which the option exists. Every other number here is
measured, and merging a sampled phase into a measured one marks the result as estimated too —
a partly-estimated total presented as measured is exactly the failure this layer is built
around.

Two things follow from how it works:

- **Sampling a phase samples its whole subtree.** A skipped entry records nothing for itself
  or anything beneath it. Counting children at full rate under a parent counted at one in `n`
  would leave two rates mixed in one tree — a plausible wrong number rather than an obvious
  one. Counters and `io=True` bytes inside the phase are scaled by the same factor.
- **Selection is a deterministic stride, not a random draw** — a draw costs about as much as
  the phase it is avoiding. The cost is aliasing: a workload whose period lines up with the
  stride keeps measuring the same point in it.

### Phase names must not be built from data

`phase(f"episode_{i}")` grows the phase tree for the life of the process — every node carries a
dense 512-bucket histogram that is also rewritten into every snapshot — until it folds at 4096
paths and the report stops being readable. `count()` raises on a float rather than truncating
it; a generated name is the more damaging mistake and used to have no equivalent protection.

One name in isolation says nothing: `conv2d` and `resnet50` are good names. What gives a
generated one away is repetition of a *shape*, so the profiler counts distinct names per shape
and warns once, well before the cap:

```
128 distinct phase names share the shape 'episode_#' (most recently 'episode_127').
Names built from data grow the phase tree until it folds at 4096 paths and the report
stops being readable — use a fixed name and count() for the varying part.
```

`Profiler(..., strict_names=True)` turns that into an error on the *second* name sharing a
shape, which makes "my phase vocabulary is fixed" a guarantee the profiler checks rather than
something to pin in a test by hand.

### Two threads in one process

`role` is per process. A learner taking gradient steps and a collector draining a queue into a
replay buffer are one process with two very different answers to "where did the time go?", and
both were reported as `learner`. `Profiler(..., thread_names=True)` nests each thread's phases
under its thread name:

```
LEARNER  (1 process, imbalance 1.00)
──────────────────────────────────────────────────────────────
learner                        71.0%       2h 58m
collector                      29.0%       1h 12m

DOMINANT PHASES                     self    wait       p50       p99
learner/train_step                2h 51m      4%    41.2ms    58.9ms
collector/drain_queue             1h 09m     94%     2.1ms   210.4ms
```

which is what makes the 94% answerable: it is the collector blocked on the queue, not the
learner. Off by default, because it changes the shape of the reported tree and most processes
have only one interesting thread. The prefixing happens at merge time, so it costs nothing per
phase — set `threading.current_thread().name` to something meaningful and it shows up.

### Exporting to W&B or TensorBoard during the run

`merged_tree()` is cumulative, so publishing a per-interval metric means keeping the last
reading and subtracting. `deltas()` does that for you, and `on_snapshot()` gives you somewhere
to put it:

```python
profiler = Profiler(run_dir="profile", role="learner", snapshot_interval_s=30.0)

@profiler.on_snapshot
def export(_tree):
    for path, stats in profiler.deltas().items():
        name = "/".join(path)
        wandb.log({
            f"profile/{name}/wall_s":  stats.wall_ns / 1e9,
            f"profile/{name}/wait":    stats.wait_ns / stats.wall_ns,   # wall, not self
            f"profile/{name}/p50_ms":  stats.hist.quantile(0.5) / 1e6,
            f"profile/{name}/calls":   stats.calls,
        })
```

Quantiles survive the subtraction — histograms are bucket counts, so the difference of two
cumulative histograms is the histogram of the interval between them, and a slow interval's p50
is not dragged down by the fast ones before it. A phase that did nothing in the interval is
**absent** rather than present at zero, so an exporter never publishes a flat line as activity.

Two things worth knowing:

- **`deltas()` has its own cursor**, independent of `on_snapshot`. Calling it inside the
  callback, as above, is the intended combination; calling it elsewhere as well will split the
  intervals between the two call sites.
- **Callbacks fire only on the periodic flush** — not from `close()`, and not from a snapshot
  taken in a signal handler, where running arbitrary user code risks deadlocking the process on
  its own final flush. You therefore lose the last partial interval; read the run directory
  afterwards for the complete picture. A callback that raises is counted and skipped, never
  propagated, so an exporter that loses its connection cannot stop the flush timer.

### Using it in tests

A merged run is a machine-readable record of *what actually executed* — which roles started,
which phases ran, how much work each did, which workers went quiet. That makes it an assertion
target, not just something to read:

```python
from lineprofiler.accounting import merge_run

run = merge_run("profile", with_samples=False)   # phases only: fast, and megabytes smaller

assert "evaluator" in run.roles                  # the process actually started
assert run.tree[("iteration", "mcts")].calls > 0 # the code path actually ran
assert run.unreadable == []                      # no worker died before its first flush
```

This catches a class of bug that unit tests structurally cannot, because no other artifact
records cross-process behaviour. Real examples:

| Symptom | Assertion that catches it |
|---|---|
| An evaluator process never spawned, while the supervisor logged "Evaluator ✓" (it was alive, just idle) | `"evaluator" in run.roles` |
| A restarted actor silently ran a different environment | `run.tree[(...)]` phase vocabulary |
| A 12-hour async run wedged with a dead collector, producing no output at all | `written_at` staleness, below |
| Diagnostics behind a `getattr` default produced nothing while every test stayed green | `run.tree[(...)].counters` |

Staleness and loss are derived at report time rather than trusted from the file, because a
worker whose flushes died leaves a file that parses perfectly and is simply hours out of date:

```python
latest = max(w.written_at for w in run.workers)
assert all(latest - w.written_at < 300 for w in run.workers), "a worker stopped writing"
```

For a CI gate outside Python, `lineprofiler report <dir> --json` gives the same document —
`run`, `roles[].phases[]`, `workers[]` and `caveats` — with the shares and quantiles already
derived. `caveats` is part of that document on purpose: a run that lost a worker must not read
as a complete result to a program either.

Set `enabled=True` explicitly in tests rather than relying on `LINEPROFILER_PROFILE`, and pass
`snapshot_interval_s=None, sample_interval_s=None` to keep the run deterministic and
thread-free; call `close()` to flush.

### Multiple processes, and multiple nodes

**`run_dir` is resolved to an absolute path at construction**, before it is exported to
children. A relative default like `"profile"` otherwise means a *different* directory in every
process that has its own working directory — which is exactly what a batch system hands each
rank — so one run scattered across the filesystem and merged as several short ones. A relative
path resolves against the working directory of the process that constructed the profiler,
which is what you meant by it; `$SLURM_SUBMIT_DIR` is deliberately *not* used, because portals
set it to their own installation directory (Open OnDemand reports
`/var/www/ood/apps/sys/dashboard`), which is somewhere you neither chose nor can usually write.
Passing an absolute `run_dir` remains the clearest thing to do.

Every process writes its own `workers/<host>/w_<run>_<pid>_<uuid8>.json`; `report` merges
them. The uuid matters because a restarted worker reuses its rank but not its pid, and the
per-host directory keeps a large run from concentrating two files per rank — plus a rename
per flush — into one directory, which is a metadata hot spot on Lustre.

Each worker records the node it ran on and its rank, read from whichever launcher is present
(`SLURM_PROCID`, `RANK`, `OMPI_COMM_WORLD_RANK`, `PMI_RANK`) along with the batch job id.
That is what makes *which node is slow?* answerable; the report names the nodes involved and
counts processes by worker file rather than by pid, which collides across nodes.

**Runs are identified.** A rerun into the same directory is a separate attempt: `report` shows
the newest and names the superseded ones rather than merging them, which used to inflate every
total for a requeued job. Children inherit the attempt through `LINEPROFILER_RUN_ID`.

**On preemption**, `SIGUSR1` and `SIGHUP` flush before exit alongside `SIGTERM` — Slurm's
`--signal=USR1@120` idiom terminates without running `atexit`, so the last interval used to be
lost exactly when you wanted it. `SIGKILL` remains unreachable; the periodic snapshot is what
survives it.

**`os._exit()` is the other unreachable exit, and it is not exotic** — it is how a
multiprocessing entrypoint normally tears a worker down, and how most "exit immediately without
running cleanup" paths are written. It skips `atexit` *and* never delivers a signal, so neither
hook above fires and everything since the last periodic flush is lost. If your teardown path
calls it, call `close()` yourself first:

```python
profiler.close()
os._exit(0)
```

The run still parses and still looks complete — it is simply missing its tail, which is the
failure mode this layer works hardest to avoid elsewhere. Lower `snapshot_interval_s` if you
cannot reach the exit path.

**An enabled profiler changes the process, and `close()` changes it back.** Constructing one
registers an `atexit` hook, chains the three signals above, and registers `os.register_at_fork`
callbacks — all process-global, none of it scoped to the object. `close()` removes the `atexit`
hook and puts the signal handlers back. The fork callbacks are the exception: CPython has no
`unregister_at_fork`, so they stay registered for the life of the interpreter and instead go
inert, dispatching over weak references and skipping any profiler that has closed.

This matters most inside a test suite, where profilers are constructed and discarded in the same
interpreter as everything else:

```python
def test_something(tmp_path):
    profiler = Profiler(run_dir=tmp_path, enabled=True)
    try:
        ...
    finally:
        profiler.close()      # not optional: it is what un-does the above
```

Closing order need not match construction order — a parent closed before its child is handled —
but a profiler that is *never* closed keeps its handlers for the rest of the process. If a host
installs its own handler on top of a live profiler, `close()` deliberately leaves the profiler's
handler in place rather than delete the host's; it is inert by then and still chains correctly.

Better still, assert against a subprocess run rather than embedding a profiler in the test
process at all — see [Using it in tests](#using-it-in-tests). That is the pattern this layer is
built for, and it sidesteps the question entirely.

**On a large run**, pass `--no-samples`. Resource samples dominate merge memory — a 12-hour
worker holds roughly 28 MB of them, about 1.8 GB across 64 workers, and the derived intervals
roughly double the peak. Phase trees for the same run are a few megabytes.

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

### What it does not do

Function-level tracing, CUDA kernel timing, GPU compute-versus-wait attribution and
per-line memory are left to `torch.profiler`, VizTracer, memray and nsys. The GPU block
reports utilisation — per device, and split into your run's share and everyone else's — which
tells you *whether* the GPU is the constraint, never *which kernel* is. `backend="torch"` gets
you that breakdown for a window.

### Optional dependencies

None of `psutil`, `torch`, `nvidia-ml-py` or `viztracer` are required. Each is imported
lazily behind a capability check (`capabilities.py`); whichever are missing just disables the
block of the report they feed, and construction never raises.

| Package | Extra | Powers |
|---|---|---|
| `psutil` | `resources` | RSS (memory block) and per-process I/O counters (`I/O` block, both `bytes` and `chars` layers) |
| `nvidia-ml-py` | `gpu` | GPU block: whole-device (`busy`) and per-pid (`this run`) utilisation, read by the 1 Hz sampler |
| `torch` | none — install separately | CUDA allocator stats (VRAM allocated/reserved), `phase(sync=True)`, NVTX ranges (`annotate=True`), the `backend="torch"` window |
| `viztracer` | `viztracer` | the `backend="viztracer"` window |

```
pip install with-line-profiler[resources]   # psutil  -> memory + I/O blocks
pip install with-line-profiler[gpu]         # nvidia-ml-py -> GPU utilisation block
pip install with-line-profiler[viztracer]   # viztracer backend
pip install with-line-profiler[all]         # psutil + nvidia-ml-py + viztracer
pip install torch                           # separately; CUDA memory, sync=True, annotate=True, backend="torch"
```

None of this is threaded through your code — construct `Profiler` the same way regardless,
and each block just appears once its package is importable and, for NVML, once a device is
visible:

```python
profiler = Profiler(run_dir="profile", role="learner")
```

**`psutil`** drives the sampler's memory and I/O rows: `Process.memory_info().rss` for the
memory block, and `Process.io_counters()` for both layers reported under `I/O` —
`read_bytes`/`write_bytes` (block device) and `read_chars`/`write_chars` (syscalls, cache hits
included). Without it the sampler still starts; those two blocks are absent, nothing else is.

**`torch`** is read for CUDA rather than declared as a dependency, so install it yourself if you want GPU features. It backs:
- `cuda_alloc`/`cuda_reserved` sampler rows → `VRAM allocated (peak)` / `VRAM reserved (peak)`
  in the GPU block, via `torch.cuda.memory_allocated()`/`memory_reserved()`
- `phase(name, sync=True)`, which calls `torch.cuda.synchronize()` at both ends of the phase
  so its wall time reflects GPU completion rather than kernel enqueue; a no-op when torch is
  absent or no CUDA device is visible
- `Profiler(..., annotate=True)`, which wraps every phase in `torch.cuda.nvtx.range_push/pop`
  (falling back to the standalone `nvtx` package) and `torch.profiler.record_function`, so an externally started `nsys profile` or Kineto capture shows your phase names
- `Backend.TORCH`, the `backend="torch"` window below

**`nvidia-ml-py`** (imported as `pynvml`) is initialised once on first use; if `nvmlInit()`
fails — no driver, no GPU — the capability degrades to `None` and the sampler skips GPU rows.
It supplies the two numbers the GPU block reports per device: `nvmlDeviceGetUtilizationRates`
(whole-device `busy`) and `nvmlDeviceGetProcessUtilization` (`this run`, this pid's share).

**`viztracer`** backs only `Backend.VIZTRACER`, the other heavy-profiler option below; it is
never imported for anything else.

#### Running a heavy backend for a window

```python
profiler = Profiler(
    run_dir="profile",
    backend="torch",              # or "viztracer"
    backend_window=(100, 110),    # start on the 100th entry into "iteration", stop on the 110th
    window_phase="iteration",
)
```

Starts the chosen backend on the 100th entry into `iteration` and stops it on the 110th,
writing its artifact under `profile/backend/` — a Chrome trace (`torch_trace.json`, open at
`chrome://tracing` or with Perfetto) for `backend="torch"`, a VizTracer capture
(`viztracer.json`, open with `vizviewer`) for `backend="viztracer"`. `backend` is a single
enum value: `line_profiler`, `cProfile`, VizTracer and `torch.profiler` all contend for the
interpreter's trace hook, so only one heavy profiler can run at a time. If the chosen
package isn't installed, the window degrades to a no-op and records `unavailable_reason` in
`metadata.json` instead of raising.

## Line profiler (`lineprofiler.LineProfiler`)

Line-by-line tracing for a bounded region: wrap code in a `with` block and it records
per-line hit counts and timing. Only code under your project folder is traced — the folder
is auto-detected by walking up to the nearest `.git` — so the output is your code, not the
stdlib and site-packages.

This is the expensive one. `sys.settrace` fires on every line of every in-project frame, so
it is for a region you already suspect, not for a whole training run. For that, use the
accounting layer above.

- **Zero configuration** – just wrap code in a `with` block, or use `start_profiling()`/
  `stop_profiling()` (see [Adopting either tool in two lines](#adopting-either-tool-in-two-lines))
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

## Licence
MIT

The claude.md is partially created from https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md
