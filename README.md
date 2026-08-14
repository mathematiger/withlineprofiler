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
lineprofiler report profile/ --no-samples     # phases only; skips the resource blocks
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
| `phase()`, `enabled=False` | 320 |
| `phase()`, `enabled=True`, `measure_cpu=False` | 2280 |
| `phase()`, `enabled=True`, `measure_cpu=True` | 3840 |
| `phase(io=True)` | 44900 |
| `count()` | 355 |

`sync=True` is absent from the table because its cost is not the profiler's: it is however
long the GPU still had to run. Phases that do *not* set it are unaffected — the check is one
branch, inside the noise of the numbers above.

`measure_cpu` (on by default) is what produces `wait%`, and it doubles the cost:
`time.thread_time_ns()` reads `CLOCK_THREAD_CPUTIME_ID`, which is not in the vDSO, so each
call is a real syscall at roughly 590 ns. `io=True` costs two `/proc` reads and two reads of
the overhead counter — negligible on a 10 ms checkpoint, ruinous on an inner loop.

At around a microsecond per phase, put `phase()` around searches, env steps and train
steps — not inside an inner simulation loop. Use `count()` there instead; it is five times
cheaper and gives you the rate anyway.

### Multiple processes, and multiple nodes

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

### What it deliberately does not do

Function-level tracing, CUDA kernel timing, GPU compute-versus-wait attribution and
per-line memory are left to `torch.profiler`, VizTracer, memray and nsys. The GPU block
reports utilisation — per device, and split into your run's share and everyone else's — which
tells you *whether* the GPU is the constraint, never *which kernel* is. `backend="torch"` gets
you that breakdown for a window.

### Optional dependencies

`psutil` (memory and I/O), `torch` (CUDA memory), `nvidia-ml-py` (GPU utilisation),
`viztracer`. Each is imported lazily behind a capability check; a missing one disables one
block of the report and never raises.

## Licence
MIT

The claude.md is partially created from https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md
