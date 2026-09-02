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

`sync=True` drains the queue at *both* ends of the phase. Entry matters as much as exit: synchronising only on exit bills this phase for whatever an earlier one left queued. It is a no-op when torch is absent or no CUDA device is visible.

It is also a no-op in a process that has not initialised CUDA, and that one is load-bearing rather than an optimisation. `torch.cuda.synchronize()` is what creates a process's CUDA primary context — measured at **414 MiB of VRAM on an A100**, where neither `import torch` nor `torch.cuda.is_available()` costs anything — so on a GPU box, one `sync=True` phase in a worker that holds no model used to buy it a context it never used. In a pipeline whose actors talk to an inference server over a queue, that is 414 MiB per actor: 1.7 GB at four actors, ~13 GB of a 40 GB card at thirty-two. Since a process with no context has submitted no device work, the drain is skipped there and nothing is lost. The check is a flag read per synchronised phase, and a process that initialises CUDA later starts synchronising from that point.

Two escape hatches, both on the constructor. `Profiler(cuda_sync=False)` switches synchronisation off outright for a role you already know is CPU-only — it composes with the per-role construction you are probably already doing:

```python
profiler = Profiler(role="actor", cuda_sync=False)   # this worker holds no model
```

`Profiler(cuda_sync=True)` restores the unconditional drain, context and all, for a caller who wants to synchronise a device this process has not yet touched.

The cost is the pipelining you give up — across that boundary the CPU can no longer run ahead
of the GPU — so put it on the phases you are actively measuring, not on every phase in the
loop, and take your headline timings from a run with it off once you know where the work is.
Note also that `wait%` on a synchronised phase depends on how the driver waits: a blocking
sync shows up as wait, a spinning one burns CPU and shows up as none.

Kernel-level attribution is still `backend="torch"`; this only fixes which phase the time
lands on.

#### When you cannot afford to synchronise

Synchronising costs the pipelining, so the common case is a run left unsynchronised — and the
common case is therefore a phase table whose numbers answer a different question than they
appear to. Nothing in the output used to record which way `sync` was set, so two runs
producing identical-looking tables could mean completely different things.

Declare it instead:

```python
with profiler.phase("forward", async_work=True):
    out = model.recurrent_inference(hidden, action)   # kernels still in flight at exit
```

The phase is measured exactly as before; what changes is that the report says so:

```
DOMINANT PHASES          entries        self    wait       p50       p99
†forward                  13,349      1m 31s      1%     6.3ms    11.5ms

  † = wall time excludes un-awaited device work (async_work=True). This is
      submission time, not device compute. Re-run that phase with sync=True to
      attribute the device time to it:
      forward                 13,349 entries
      If a phase's cost does not scale with its batch counter, it is launch-bound:
      the time is per-kernel launch overhead, not per-element work. Compare its wall
      time at batch 1 against a large batch; flat means fewer, bigger launches (a
      captured CUDA graph) is the fix, not a faster kernel.
```

That last paragraph is the question the mark leaves open. Knowing 6.3 ms is submission time does not say *why* submitting takes 6.3 ms, and the measurement that settles it is one the profiler cannot run for you: time the same phase at batch 1 and at a large batch. Flat wall time across a 512x range means the cost is per-launch, not per-element — one reported case measured 2.13 ms at both ends, at 145 kernels per call, and replacing the sequence with a captured CUDA graph took it to 0.39 ms for a 2.89x end-to-end win. The counter rows the report already prints are what you compare against; the note only points at the comparison.

`async_work` is ignored when `sync=True` — that phase *did* wait for its work — so flipping
one to the other is how you turn a submission time into a device time, and the mark
disappears when you do. It costs one bool test per phase, so it is safe on an inner loop, and
it applies to anything that submits without awaiting: device queues, `io_uring`, non-blocking
sockets, background executors.

The corroborating evidence is the `GPU BY PHASE (sampled)` block, which buckets the 1 Hz
device samples by the phase open when each was taken:

```
GPU BY PHASE (sampled)          p50    p95        VRAM   samples
train_step/fwd                  81%    91%      3.2 GB        20
forward                          8%     9%      1.1 GB        20
```

A phase holding most of a process's time at 8% device utilisation is the contradiction that
says the wall time is submission time — previously a manual join between the phase table and
the timeline's GPU series, which are on different timebases.

#### Decomposing a queue wait

`wait%` says a phase was blocked; it cannot say on *what*. A single `queue_wait` total fuses
intervals whose remedies point in opposite directions — the request sitting unclaimed (batch
harder), the server assembling a batch around it (shrink the window), the server computing it
(cheaper model), the reply travelling back (fewer hops). `signal()`/`wait_on()` cannot split
them either: the producer signals at *response* time, so an arrow spans only the last of the
four.

Mark the checkpoints instead, in whichever process owns each transition:

```python
# client, before the put
profiling.trace_begin("inference", request_id)
# server, on dequeue into a batch
profiling.trace_mark("inference", request_id, "admitted")
profiling.trace_mark("inference", request_id, "computed")
# client, after the get
profiling.trace_end("inference", request_id)
```

```
REQUEST LIFECYCLE
inference                        422.8ms  (6 req)
    ├─ begin → admitted             301.0ms    71%    50.2ms/ea
    ├─ admitted → computed          120.8ms    29%    20.1ms/ea
    └─ computed → end               931.3us     0%   155.2us/ea
```

71% before admission is a batching problem, not a model problem. Marks reuse the link ring
and its drop policy, so the cost is the same order as `signal()`. At high request rates pass
`sample=0.01`: selection is by key hash rather than a counter, so every checkpoint of a given
request is kept or dropped together — across processes, without any shared state.

Incomplete lifecycles contribute nothing rather than a partial segment, and checkpoints that
arrive out of order (cross-host clock skew) are dropped rather than counted as negative time.

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
| `phase(async_work=True)` | 3911 — free next to `measure_cpu=True` |
| `trace_mark()`, `trace=True` | 1277 |

`sync=True` is absent from the table because its cost is not the profiler's: it is however long the GPU still had to run. Phases that do *not* set it are unaffected — the check is one branch, inside the noise of the numbers above. A phase that *does* set it pays one extra `torch.cuda.is_initialized()` flag read per drain, which is what keeps a CPU-only worker from opening a CUDA context; against a `cudaDeviceSynchronize` measured in microseconds to milliseconds it does not register.

`measure_cpu` (on by default) is what produces `wait%`, and it doubles the cost:
`time.thread_time_ns()` reads `CLOCK_THREAD_CPUTIME_ID`, which is not in the vDSO, so each
call is a real syscall at roughly 590 ns. `io=True` costs two `/proc` reads and two reads of
the overhead counter — negligible on a 10 ms checkpoint, ruinous on an inner loop.

**Budget it as a ratio, not as a rule about loops: keep phase overhead under ~1% of the region
you are measuring.** At the default settings that is ~3.9 µs per phase, so a phase is affordable
around anything taking more than ~400 µs. Budget against the row you will actually run: the
figure above is `measure_cpu=True`, which is the default because `wait%` is derived from it, and
turning it off roughly halves the cost at the price of that column. The table above is there so
you can decide per call site without measuring.

#### If a phase is too expensive, in the order worth trying

Every row of the table above is dominated by Python's own dispatch — allocating the scope, entering and leaving the `with`, and the dictionary work behind the phase tree. There is no configuration that makes those cheaper, so the levers are all about doing less, and they are worth trying in this order:

1. **Turn off `measure_cpu` for that profiler.** The single largest one: two `thread_time_ns()` syscalls at ~590 ns each, measured at **~1.6 µs of a ~4.2 µs phase — 38% of it**. The cost is `wait%`, which is usually the most valuable column on the page, so this is a real trade and not a free win. Take it when you already know a region is CPU-bound, or when you are measuring throughput rather than diagnosing a stall.
2. **Sample the phase** with `sample=0.01`. Saves ~3.4x rather than 100x — the cost is Python call overhead, not measurement — and everything derived from it becomes a labelled estimate.
3. **Move the phase outward.** A phase around the loop instead of inside it pays once per loop rather than once per iteration, and `count()` inside the loop still gives you the rate at about a fifth of the price.
4. **Leave `trace` off unless you need the timeline.** It is off by default; the untraced path costs one identity test.

Not levers, despite looking like them: `sync=True` costs whatever the GPU still had to do rather than any profiler time, and `async_work=True` is one bool test. Neither is worth removing for speed.

#### The part that lands inside the number, not beside it

The table above is what a phase costs *your program*. A different and smaller quantity is what a phase adds to *its own reported wall time*, and it is the one that affects accuracy rather than speed.

`__enter__` reads its clock last and `__exit__` reads its clock first, so almost all of the cost above falls outside the measured interval. What remains inside is the interpreter's own dispatch between those two reads — measured at **~210 ns** with `measure_cpu=False`, **~240 ns** with `measure_cpu=True`, and **~250 ns** with `trace=True`, on the same box as the table above.

That floor is close to irreducible in Python: a bare context manager that does nothing but read `perf_counter_ns()` at each end reports ~185 ns for an empty body, of which ~95 ns is the cost of reading the clock twice. The profiler sits roughly 25 ns above a context manager that measures nothing at all.

**What it means for a reading:** every phase's wall time is inflated by about a quarter of a microsecond, once per entry, and the inflation does not scale with the phase's duration.

| phase duration | inflation from the floor |
|---|---|
| 1 ms and up | under 0.03% — ignore it |
| 100 µs | ~0.25% |
| 10 µs | ~2.5% |
| 1 µs | ~25% — the number is mostly measurement |

This is the same ratio the ~1% budget above already enforces, seen from the other side: a phase that clears the budget for speed has also cleared it for accuracy. It matters when you compare a *sum* of very many short phases against a wall clock — a phase entered a million times carries roughly 0.25 s of floor in its total — and when you difference two nested phases whose durations are close.

Neither `self_ns` nor `wait_ns` corrects for it, and the report does not subtract it: the correction would be an estimate, and substituting an estimate for a measurement is exactly what this layer refuses to do elsewhere. It is stated here instead so the arithmetic is yours to do.

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
DOMINANT PHASES          entries        self    wait       p50       p99
~uct_search                6,400      2h 31m     18%     9.4ms    18.5ms

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

DOMINANT PHASES          entries        self    wait       p50       p99
learner/train_step         4,200      2h 51m      4%    41.2ms    58.9ms
collector/drain_queue     33,600      1h 09m     94%     2.1ms   210.4ms
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

The first question about any long wait is binary: **is this a hang, or is it queueing behind
real work?** The two look identical in a phase table and have nothing in common as problems.
The report answers it per role, from the trace:

```
  while actor waited, concurrently active: server 88%, learner 3%
```

Work happening elsewhere means queueing; silence everywhere means a stall, and the report
says that outright instead of leaving a blank. On the timeline page, **select a range** and
brush across a wait to get the same breakdown for that moment.

Scriptable, without parsing the page:

```python
from lineprofiler.accounting import overlap_ns

overlap_ns([(wait_start, wait_end)], [(busy_start, busy_end)])   # nanoseconds in both
```

Large runs render with `--max-spans N`, which keeps the longest spans and states how many it
dropped rather than failing after the profiled run has already succeeded. Progress goes to
stderr, so a slow render is distinguishable from a stuck one; `-q` silences it.

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
