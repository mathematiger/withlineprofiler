# Accounting recipes

Reading the report, budgeting the overhead, and the failure modes worth knowing about.

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

## Seeing *why* a worker was idle

The report tells you `queue_get` was 80% wait. It cannot tell you what the learner was
waiting *for*, because a total has no position on a clock. That is what the trace timeline
is for:

```
lineprofiler trace profile/ -o trace.html
```

### What it costs to adopt

Measured, not estimated — the numbers below are `git diff --stat` over a real
two-actor/one-learner pipeline, each tier applied as its own commit:

| tier | your diff | what it buys |
|---|---|---|
| `LINEPROFILER_TRACE=auto` | **0 lines** | lanes and nesting derived from function calls |
| `LINEPROFILER_TRACE=1` | **0 lines** | lanes from the phases you already name |
| `trace=True` | **1 line per `Profiler(...)`** | the same, set in code |
| `signal()` / `wait_on()` | **1 line per queue endpoint** | arrows, and a critical path that crosses processes |

Adding `phase()` to a codebase that has none is the *existing* cost of this package — about
28 lines on the pipeline above — and it buys the text report and icicle chart too. Nothing in
the timeline requires it: start at `auto`, find where the time goes, then name the four or
five phases it points at.

### The two-line version

```python
queue.put(batch)
profiler.signal("batch", batch.id)        # producer: it is ready

with profiler.phase("queue_get"):
    batch = queue.get()
profiler.wait_on("batch", batch.id)       # consumer: I needed it here
```

`key` only has to match on both sides — a step number, an id, a UUID. An unmatched `wait_on`
is reported on the page, never raised.

### Reading the result

Start with the **lane table**, which separates *phase open* from *on CPU*:

```
lane                role       phase open    on CPU   blocked
actor 263801#0      actor           96.4%     95.0%      1.3%
learner 263800#0    learner         96.9%     38.5%     58.4%
```

The learner has a phase open almost the whole run and spends 38% of it on a CPU. That 58%
gap is the answer, and the **critical path** below the chart names who caused it: alternating
`queue_get` at 100% wait and `train_step` at 1%, with an arrow back to whichever actor was
still generating.

The fix that follows is an architecture change — more actors, a deeper queue, cheaper env
steps — and it is the sort of change you want evidence for before making.

### When *not* to reach for it

- **Not for a twelve-hour run at `auto`.** Auto-tracing costs per *function call*. Use it as a
  discovery tool over a few iterations, then switch to named phases for the long run.
- **Not to measure CPU time on auto spans.** They cannot; the page says *unknown* rather than
  guessing. Named phases measure it.
- **Not as a substitute for Perfetto or nsys** when you need kernel-level or per-line detail.
  `backend="torch"` still exists for exactly that, for a bounded window.
