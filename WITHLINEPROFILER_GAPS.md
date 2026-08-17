# Instrumentation gaps for `withlineprofiler`

Written against **lineprofiler 0.6.0** (`.venv/.../site-packages/lineprofiler/accounting`), from a real
investigation: "MuZero actors spend 96% of self-play blocked — on what?"

**How to read this.** Each gap states what happened, what the current library does, what is missing,
and a concrete proposal. Two gaps (#3, #6) turned out to be **partly our own misuse**, and I have
separated "library gap" from "caller bug" in those — please do not implement the caller bug.

Verified environment facts this document relies on:
- `phase(name, io=False, sync=False, sample=1.0)`; `count(name, n=1)`; `wait_on(channel, key)`;
  `signal_ready(channel, key)` (our shim re-exports it as `profiling.signal`).
- `PhaseStats` = `calls, wall_ns, cpu_ns, child_wall_ns, hist, counters, sample_stride`, with derived
  `self_ns = wall - child_wall` and `wait_ns = wall - cpu`.
- `Span` = `phase_id, thread_id, t0_ns, t1_ns, cpu_ns, flags`; `flags` is a bitfield with only
  `FLAG_AUTO = 1` and `FLAG_SAMPLED = 2` used.
- `Link` = `channel, key, kind, t_ns, thread_id`.
- `report.py` already renders per-phase counter rows (total, `/s`, `/ea`) via `_counter_rows`, and a
  GPU block with a footnote via `_gpu_footnote`.

The run referenced throughout: `profile_runs/case30_trace_small`, 2 actors + 1 inference server +
1 learner, `num_simulations=40`, `envs_per_actor=2`, `inference_mode=server`, `inference_ipc=queue`,
`profile_forward_sync=False`, `profile_ram=False`.

---

## Gap 1 — A phase wrapping unsynchronized GPU work reports a wall time that is not what it looks like

**Severity: high. This one produces confidently wrong conclusions.**

### What happened

The server's phase table read:

```
forward    1m 31s    wait 1%    p50 6.3ms    p99 11.5ms      (13,349 entries)
scatter     2.00s    wait 0%    p50 140.4us
```

The natural reading — "91.5 s of GPU forward at 6.3 ms each, the GPU is the bottleneck, shrink the
model" — is **wrong**. Independent evidence:

- GPU utilization over the steady-state window was **mean 6.6%, median 7.0%, max 9.0%**, with zero
  samples above 20%.
- Independent benchmarking of this network: dynamics forward is **0.69 ms at any batch from 1 to 512**.
- The net has `hidden_dim=64`; it cannot occupy a GPU.

The cause is that `profile_forward_sync=False` (the default), so the code is:

```python
with profiling.phase("forward", sync=self._forward_sync):   # sync=False
    out = model.recurrent_inference(hidden.to(dev), action.to(dev))
```

CUDA is asynchronous. The phase closes when the kernels are *enqueued*, not when they *finish*. So
`forward`'s 6.3 ms is Python dispatch + H2D/D2H copy setup, and the GPU work is billed to whatever
later operation happens to synchronize.

### Why this is a library gap, not just a caller mistake

The `sync` parameter exists and is documented. But **nothing in the output records which way it was
set.** Two runs producing byte-identical-looking phase tables can mean completely different things,
and the reader has no way to tell. Worse, the default is `sync=False`, so *the misleading case is the
common case*. The measurement is not wrong — it is unlabeled, which is worse, because it is
indistinguishable from the labeled one.

This generalizes past CUDA: any phase wrapping an async submission (device queues, `io_uring`,
non-blocking sockets, background executors) has the same problem.

### Proposal

1. **Add a `Span.flags` bit and a `PhaseStats` marker.** `flags` has 30 free bits. Add
   `FLAG_ASYNC_UNSYNCED = 4`, set on every span opened by a `phase()` whose enclosed work was *not*
   synchronized, and propagate a boolean (or a count of such entries) into `PhaseStats`.

2. **Make the report say so.** A phase whose entries carry that bit gets a marker and a footnote:

   ```
   DOMINANT PHASES                     self    wait       p50       p99
   forward †                         1m 31s      1%     6.3ms    11.5ms

   † wall time excludes un-awaited device work (sync=False). This is CPU submission
     time, not device compute. Re-run with sync=True to attribute device time here.
   ```

3. **How to know the work was async.** Options, cheapest first:
   - **Declarative (recommended):** invert the existing parameter. `phase(..., sync=False)` already
     tells the library the caller knows async work is inside. Treat an explicit `sync=False` as the
     signal, and set the flag. Zero new cost, zero new API.
   - **Detected:** when the `torch` backend is active, compare a cheap CUDA event at phase exit against
     the phase's own `t1_ns`; a nonzero gap proves unfinished work. Costs an event record per phase —
     make it opt-in.

4. **Related, and the reason I was able to catch this at all:** with `profile_ram=False` the text
   report contains **no GPU section whatsoever** (`_gpu_footnote` never renders), while the HTML still
   embeds `gpu.points`. The one piece of evidence that refutes the phase table was present in the
   artifact and absent from the summary. **Ask:** if any GPU sample exists, print the GPU block in the
   text report regardless of `profile_ram`; if none exists, print one line saying GPU data was not
   collected and how to enable it. Silence reads as "no GPU involved."

**Acceptance test:** a phase wrapping a deliberately slow CUDA matmul with `sync=False` must be
visibly marked in `profile_report.txt`, and the same code with `sync=True` must not be.

---

## Gap 2 — Queue wait is one undifferentiated number

**Severity: high. This is the gap that left the original question unanswerable.**

### What happened

After instrumenting the client, the actor tree read:

```
mcts/recurrent_inference/ipc/queue_wait   280.40s   wait 98.7%
mcts/initial_inference/ipc/queue_wait      10.37s   wait 99.5%
```

Correct, and still not actionable. That single bar fuses four intervals with **opposite** remedies:

| # | Interval | Correct fix if it dominates |
|---|---|---|
| (a) | request sat in the queue, unclaimed | more batching / more actors |
| (b) | server was assembling my batch (`inference_batch_window_ms`) | shrink or remove the window |
| (c) | server was computing the batch containing me | cheaper model / bigger batch |
| (d) | reply sat in the response queue + my wake latency | reduce IPC hops, shared memory |

We eventually established the answer — (a)+(b), because batching was capped and the GPU was idle —
but only via manual interval arithmetic over the raw span JSON.

### Why `signal_ready`/`wait_on` does not cover it

The mechanism exists and we use it. It is not sufficient: **all 23,646 arrows in that run sum to
6.8 s, against 187.6 s of measured wait — 3.6%.** A `Link` pair records `signal` → `wait`, and the
server signals *at response time*, so an arrow spans only interval (d), whose p50 is **0.196 ms**. The
95%+ of the wait that happens *before* the signal is structurally outside what a two-point link can
express.

### Proposal

Support a **request lifecycle** — one key, several named checkpoints, across processes — rather than
only a producer→consumer pair. `Link` already carries `(channel, key, kind, t_ns, thread_id)`; `kind`
is currently `signal`/`wait`. Generalize it:

```python
profiling.trace_begin("inference", key)     # client, before put
profiling.trace_mark("inference", key, "admitted")   # server, on dequeue into batch
profiling.trace_mark("inference", key, "compute_start")
profiling.trace_mark("inference", key, "compute_end")
profiling.trace_end("inference", key)       # client, after get
```

The consumer can then decompose any client block into named segments, because the marks are
timestamped in the processes that own them. Report form:

```
mcts/recurrent_inference/ipc/queue_wait   280.40s
    ├─ enqueued → admitted      198.2s  (71%)   <- batching-limited
    ├─ admitted → compute_start   9.1s  ( 3%)
    ├─ compute_start → end       61.3s  (22%)
    └─ compute_end → delivered   11.8s  ( 4%)
```

That table alone would have answered the original question in one read.

**Cost control:** marks are the same order of cost as today's links and can reuse the identical
buffer, drop policy (`dropped_links`), and clock alignment. Sampling matters here — at 13,349 requests
per 3 minutes, allow `trace_mark(..., sample=0.01)`; keep the *shape* of the breakdown, not every
request.

**Weaker fallback, if the above is too invasive:** let the server attach its own elapsed time to the
response, and let the client declare "of my block, this much was remote compute." Much less
informative — it cannot separate (a) from (b) — but far cheaper.

**Acceptance test:** with an artificial server that sleeps a known 50 ms before admitting and a known
20 ms computing, the report must attribute ≈50 ms and ≈20 ms to the right segments.

---

## Gap 3 — Batch occupancy: partly a caller bug (ours), partly a missing distribution

**Severity: medium for the library. The high-severity half is our bug — do not implement it.**

### The caller bug (ours, not yours)

The single most diagnostic number in the whole investigation — **1.9 requests per forward against a
cap of 2** — came from a hand-rolled `logging.info` in `_BatchFillStats`, not from the profiler. I
first wrote this up as "the profiler cannot express occupancy." That was wrong, and the correction is
worth recording:

`count()` attributes to *the phase currently open on this thread*. Our calls are placed **outside**
the `with` blocks:

```python
with profiling.phase("forward", sync=...):
    out = model.recurrent_inference(...)
with profiling.phase("scatter"):
    self._scatter(...)
profiling.count("recurrent_requests", len(recurrent))   # <- outside both phases
```

`_process_batch` is not itself wrapped in a phase, so these counts land on the server's **root** node
and never appear under `forward`. `report.py:_counter_rows` renders totals, `/s`, and `/ea` per phase
and would have printed `+ recurrent_requests  N  R/s  T/ea` — the occupancy figure — if the counts had
been inside the phase. **Our fix**, tracked separately, is to move the `count()` calls inside the
`forward` block.

### The genuine library gap

Even placed correctly, `counters` is a **running sum per phase node** (`add_count` does
`counters[name] += amount`). From a total you can recover the *mean* rows per entry
(`total / calls`) but **not the distribution**. For a batching server the mean is the least
interesting statistic: mean 1.9 is consistent with "always exactly 2" (a hard cap — what actually
happened) and with "usually 1, occasionally 8" (bursty arrival). Those demand different fixes, and the
mean cannot tell them apart.

Note the asymmetry: `PhaseStats` already keeps a `DurationHistogram` for *duration*, so `p50`/`p99`
exist for time but not for work units.

### Proposal

1. **Per-entry counter values, not just a sum.** Let a count be attached to the phase *instance*:

   ```python
   with profiling.phase("forward", counts={"rows": len(batch)}):
       ...
   ```

   Reuse the existing `DurationHistogram` machinery for these values and report the same quantiles
   already reported for time:

   ```
   forward    1m 31s   wait 1%   p50 6.3ms   p99 11.5ms
       + rows            25,336    p50 2   p95 2   max 2      <- a hard cap is obvious
   ```

   A `p50 == p95 == max` signature reads as "capped" at a glance; that is the whole finding.

2. **Show it on hover in the HTML**, next to the span's duration.

3. **Cheap variant if per-instance storage is unwelcome:** keep the sum but add
   `count_min`/`count_max` alongside it. Much of the diagnostic value for a fraction of the cost —
   `min == max` still exposes a hard cap.

**Acceptance test:** a phase entered 100 times with `rows` drawn from `{1, 8}` must report a
distribution distinguishable from 100 entries of constant `rows=4.5`-equivalent total.

---

## Gap 4 — A trace does not record the code revision it measured

**Severity: medium. It silently invalidates analysis, and it bit this investigation.**

### What happened

I derived a conclusion from `1.9 requests per forward against cap 2` — that the concurrent-request
supply was exhausted. Checking the source showed the cap had a bug: `_max_batch = num_actors`, which
truncates a lockstep actor's E-row request to 1/E of what is pending. The fix
(`num_actors * envs_per_actor`) **was already present in the working tree, uncommitted**. The profiled
run had executed the *committed* code.

So the trace measured a constraint that no longer exists in the tree being analyzed. Consequences: the
baseline understates achievable batching, and `1.9 ≈ cap 2` proves only *the cap was binding*, not
that the supply was exhausted — a materially different claim that changes which fix comes first.

Nothing in `profile_report.txt`, `trace.html`, or `fingerprint.json` records the source revision;
`fingerprint.json` captures host, CPU count, ulimits, and shm — the environment, not the code.

### Proposal

Record provenance in run metadata and print it in the report header:

```
Runtime 2m 56s   Processes 4   Roles actor x2, inference_server x1, learner x1
Host node0   Run 20260817T091934-9ddbb7
Source c49ce84 (+dirty: 26 files, diff sha 3f9a1c)      <- new
```

- `git rev-parse HEAD`, plus a dirty flag; ideally a hash of `git diff HEAD` so two dirty runs are
  distinguishable.
- Fall back silently when not a git repo, and allow the host program to supply the string (we may want
  a config hash too — the effective `TrainingConfig`, since a config change alters behavior as much as
  a code change).
- Cost is one subprocess call at init, or zero if the embedding program passes it in.

**Why it matters beyond hygiene:** the whole point of this layer is that "an estimate that cannot be
told apart from a measurement is precisely the failure this layer exists to avoid" (`sample_stride`
docstring). A measurement of superseded code is the same class of failure, one level up.

**Acceptance test:** report header shows the commit; a dirty tree is visibly marked.

---

## Gap 5 — No way to ask "what was everyone else doing while I was blocked?"

**Severity: medium. Pure ergonomics, but it was the single most laborious step.**

### What happened

The key question behind a 96%-wait phase is binary: **is this a hang, or is it queueing behind real
work?** Those look identical in the phase table and have nothing in common as problems.

Answering it required extracting the embedded JSON from an 11.7 MB HTML file, then computing, by hand,
the intersection of every actor `mcts` wait interval against every server `forward` interval. Result:
**165.4 s of the 187.6 s wait overlapped the server actively computing, so only 12% (22.2 s) was true
idle** — not a hang. Also derived that way: ~24 server forwards per `mcts` span, server duty cycle
58%, inter-forward gap p50 0.819 ms.

All the raw data was present. No affordance surfaced it.

### Proposal

1. **HTML:** clicking or brushing a wait region highlights the spans concurrent with it on every other
   lane, with a small summary — "during this 320 ms wait: `inference_server` busy 92% (`forward`
   ×43), `learner` busy 4%". The renderer already has all spans in one timebase.

2. **Text report:** for the top few wait-dominated phases, a "blocked while others were" attribution:

   ```
   mcts/recurrent_inference/ipc/queue_wait   280.40s wait
       concurrent: inference_server/forward 88%, inference_server idle 9%, learner 3%
   ```

3. **API:** expose the interval-intersection primitive
   (`overlap(phase_a, phase_b) -> ns`) so this is scriptable without hand-parsing HTML. This is the
   piece I actually needed and had to reimplement.

**Acceptance test:** a synthetic run where worker A blocks 100 ms, of which worker B is busy exactly
60 ms, reports ≈60% concurrent-busy.

---

## Gap 6 — The GPU counter is not joinable to phases

**Severity: medium, and it is the mechanical reason Gap 1 was hard to catch.**

### What happened

`Sample` already carries rich per-device GPU data: `gpu_util`, `gpu_utils`, `gpu_proc_utils`,
`cuda_alloc`, `cuda_reserved`, alongside `phase`. But in the rendered artifact, GPU data appears as
`gpu.points` — a flat `[[t, util], ...]` series on its own timebase with no linkage to spans. So
"what was GPU utilization *during* the `forward` phase?" is a manual join.

That join is exactly what refutes Gap 1, which makes this gap load-bearing rather than cosmetic.

Compounding it: with `profile_ram=False` the text report's GPU block never renders at all (see Gap 1,
item 4), so the summary is silent about the GPU while the HTML quietly holds the data.

### Proposal

1. **Per-phase device aggregates.** `Sample` already has a `phase` field, so samples can be bucketed
   by the phase open when they were taken — the same mechanism `analysis.py` uses to attribute I/O
   bytes to phases. Report:

   ```
   GPU BY PHASE (sampled)
     forward          util p50  7%   p95  9%    cuda_alloc 1.2 GB
     train_step/fwd   util p50 71%   p95 94%    cuda_alloc 3.4 GB
   ```

   With this table, Gap 1's contradiction — a phase named `forward` holding 97.9% of server time at 7%
   device utilization — is visible on the page instead of requiring a hand-rolled join.

2. **Attribution caveat, stated plainly.** Sampling attributes to *the phase open at sample time*, and
   an async-submitting phase may have its device work land under a *later* phase — the same caveat
   `_exact_io_block` already carries for bytes. Say so in the footnote; the existing
   `_gpu_footnote` text ("whole-device busy time from NVML, not a compute-vs-wait split") is the right
   spirit and needs one more sentence about async attribution.

3. **Memory alongside utilization**, per device — `cuda_alloc`/`cuda_reserved` are collected but not
   surfaced per phase, and OOM diagnosis needs peak-by-phase.

**Acceptance test:** a run with one GPU-heavy and one GPU-idle phase must show clearly different
per-phase utilization.

---

## Gap 7 — Render does not scale, and fails late

**Severity: medium. Wasted the expensive half of two runs.**

### What happened

`python -m lineprofiler.accounting.cli trace <dir> -o trace.html` on a 26-process profile **hit a
900 s timeout and produced nothing** — after the profiled run had already completed successfully. The
expensive part (running the workload under profiling) succeeded; the cheap part (viewing it) discarded
the result. Two verification runs were lost this way, and the working artifact
(`case30_trace_small`, 4 processes, 31,801 spans, 23,786 arrows) is 11.7 MB of HTML, so the growth
curve is steep in both time and size.

There is no progress output, so a long render is indistinguishable from a hang — the same ambiguity
Gap 5 describes, one level up.

### Proposal

1. **Progress to stderr**: workers loaded, spans aligned, HTML written. Even coarse output separates
   "slow" from "stuck".
2. **`--max-spans N`** with documented downsampling (drop shortest spans first, or per-lane
   stratified), so a large trace *degrades* instead of failing. Report what was dropped —
   `WorkerTrace` already tracks `dropped`/`dropped_links`, so the vocabulary exists.
3. **Write incrementally / stream**, so a timeout leaves a usable partial file.
4. **Print an estimate up front**: "4.2 M spans across 26 workers; rendering may take >10 min; consider
   `--max-spans`."
5. **Minor data question:** the embedded JSON reports `duration_us = 157,594,849` (157.6 s) for a run
   whose report header says `2m 56s` (176 s). Both are defensible (span extent vs. wall clock), but the
   ~19 s discrepancy is unexplained in the artifact and made me second-guess the timebase while doing
   the Gap 5 overlap arithmetic. Either reconcile them or label them distinctly.

**Acceptance test:** a synthetic 5 M-span profile renders under `--max-spans 200000`, and the plain
invocation prints progress.

---

## Gap 8 — Percentages have no stated denominator

**Severity: low. Cost minutes, not hours — but it cost them repeatedly.**

### What happened

The role block reads:

```
ACTOR  (2 processes, imbalance 1.03)
mcts                           66.0%        3m 16s
step                           24.0%        1m 11s
session_build                  10.1%        29.94s
```

Nothing states what the percentage is *of*. Candidates: share of wall clock, of busy time, of summed
`self_ns`, of the role's total across processes. They give materially different readings, and here it
matters: the same run's HTML lane metadata says the actor lanes were `busy 97% / working 35.2%`. A
reader trying to reconcile "`mcts` is 66%" with "the lane worked 35.2% of the time" cannot, without
reading `report.py`.

`3m 16s` for `mcts` also exceeds the 2m 56s runtime in the header — correct, since it sums two
processes, but unlabeled it reads as an error and briefly did.

The library is otherwise scrupulous about this: `PhaseStats.wait_ns` carries a long docstring on
pairing with `wall_ns` and never `self_ns`, and the report repeats it. That care stops at the role
summary.

### Proposal

1. **Name the denominator in the header**: `ACTOR (2 processes, imbalance 1.03) — % of role self time`.
2. **Mark summed-across-process columns**, e.g. `3m 16s (Σ2 proc)`, so totals exceeding wall clock
   read as intentional.
3. **Put `busy` / `working` per role in the text report.** They are currently HTML-only, and
   `working %` is the headline number for an actor — the finding "actors are busy 97% but working
   35.2%" *is* the bottleneck statement, and it is absent from the text summary.
4. **Define both terms once** in the report's legend: `busy` = not idle at the OS level; `working` =
   executing on CPU inside an instrumented phase.

**Acceptance test:** a reader can reconcile every percentage in the role block against a stated base
without opening the source.

---

## Summary — priority order

| # | Gap | Severity | Why |
|---|---|---|---|
| 1 | Async device work unmarked in wall time | **High** | Produces confidently wrong conclusions; misleading case is the default |
| 2 | Queue wait not decomposable | **High** | Left the original question unanswerable; links cover only 3.6% of the wait |
| 4 | No source revision in trace | Medium | Silently invalidates analysis; cheap to fix |
| 6 | GPU samples not joinable to phases | Medium | The mechanical reason #1 is hard to catch |
| 3 | Counter distribution (not just sum) | Medium | Mean cannot distinguish "capped" from "bursty" |
| 5 | No concurrency view for a wait | Medium | Hang vs. queueing is the first question about any wait |
| 7 | Render does not scale, fails late | Medium | Discards successful expensive runs |
| 8 | Unstated percentage denominators | Low | Small, repeated friction |

If only two are implemented, **#1 and #2** are the ones that change conclusions rather than
convenience. #1 is nearly free if `sync=False` is simply treated as the declaration it already is.
