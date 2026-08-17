# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[semantic](https://semver.org/spec/v2.0.0.html).

## [0.6.0]

The report could say *how much* time a phase spent waiting, but never *when*, or *for whom*.
This release adds the view that answers that, and three ways to reach it that ask progressively
more of your code — starting with none at all.

### Added — a trace timeline: lanes on one clock, with arrows

`lineprofiler trace <run_dir> -o trace.html` draws one lane per worker thread on a shared time
axis: wait shading inside each span, GPU utilisation lanes beneath, arrows from a producer to
the consumer it unblocked, and a **critical path** walked backwards through those arrows — the
chain of spans that actually determined how long the run took.

A phase tree is a set of totals, and a total has no position on a clock. So this records
individual entries in a **fixed-capacity ring buffer** (`trace=True`, `trace_capacity=200_000`),
which keeps memory flat over a twelve-hour run and reports how many spans it had to drop. A
truncated timeline never renders as a complete one.

Tracing is **off by default**. The phase tree is bounded; a timeline is not.

Each lane is drawn as a small **flame chart**: one row per nesting level, so a phase called
from inside another sits beneath its caller rather than on top of it, and a lane reads as a
call structure instead of a list. A **Call order** table restates the same thing exactly —
each lane's calls in the order they ran, indented by depth — because a dense lane collapses
into a stripe on any chart. Nesting past eight levels folds onto the last row, and the number
folded is stated.

### Added — `signal()` / `wait_on()`, for cross-process causality

Two calls at a queue boundary — `signal("batch", key)` where you publish, `wait_on("batch",
key)` where you block — let the merge match a producer to the consumer it released. That is
the only part of the timeline needing new calls, and it is one line per endpoint. An unmatched
`wait_on` is reported on the page as unmatched, never raised: half a pipeline being
instrumented is the normal state of an incremental rollout.

### Added — `trace="auto"`, which needs no instrumentation at all

Derives spans from function entry and exit via `sys.monitoring` (3.12+), scoped to your project
by the same `.git`-rooted detection the line profiler uses — so stdlib, site-packages and this
package itself stay out of the picture. A codebase with no `phase()` calls gets a timeline from
`LINEPROFILER_TRACE=auto` and no diff whatsoever.

It cannot measure CPU time: `thread_time_ns()` is a real syscall at ~590 ns, which is not
affordable per function call. Those spans are drawn hatched and their wait reported as
*unknown* — never as zero, which would read as "this never waited". Use it to find where the
phases belong, then name them.

### Changed — the trace page ships JavaScript; the other two still do not

Pan and zoom over a hundred thousand spans cannot be done with static SVG. The timeline page
inlines vanilla JS — still one file, no CDN, no webfont, no network — and asserts that
separately. `report.html` and the source page remain script-free, and their tests are
unchanged. Text reaches the page through `textContent`, never `innerHTML`: phase names come
from user code, and a profiling artifact gets mailed around and opened by other people.

### Notes

- Spans are written to an append-only `<worker>.trace` sidecar, not into the worker snapshot.
  The snapshot is complete state rewritten atomically; folding a span array into it would
  rewrite tens of megabytes on every flush. A torn final line costs that batch and nothing
  earlier.
- `merge_run(..., with_trace=True)` is opt-in, so the existing report pays nothing for files
  it does not draw.
- Untraced phases are unaffected: the gate is one identity test, bounded by `test_overhead.py`.
  Tracing adds roughly 1 µs per phase when on.

## [0.5.0]

Four changes aimed at the same problem: the package was hard to adopt, hard to discover, and
excluded a large part of the audience it was written for.

### Added — a `sys.monitoring` backend, so the line profiler stops fighting other tools

`sys.settrace` is a single global hook, so `LineProfiler` could not run alongside coverage.py,
pdb, or any other tracing profiler. Anyone who tried it under `pytest-cov` got confusing
numbers and no explanation. On 3.12+ the profiler now uses `sys.monitoring`, which gives each
tool its own slot; below 3.12 it falls back to `sys.settrace`. The backend is chosen
automatically and can be overridden with `LineProfiler(backend=...)`, which is what keeps the
fallback path exercised on a runner that defaults to the newer one.

Two differences between the backends are deliberate and pinned by tests:

- **`monitoring` profiles the body of the `with` block itself.** `sys.settrace` only affects
  frames created after it is installed, so the block's own lines were never traced — a
  limitation documented since the first release. The new backend has no such restriction.
- **Two `monitoring` profilers cannot nest.** They would contend for one tool slot, so the
  inner one is refused. `settrace` tracers chain instead. Nesting double-counts either way;
  the refusal says so rather than returning inflated numbers.

The subtle part is that `sys.monitoring`'s `DISABLE` — the per-code-object opt-out that makes
the `[tool.lineprofiler]` filter free rather than merely cached — **outlives the session that
returned it**. A function filtered out in one session stayed filtered for every later session
in the process, reporting a confident zero for code that ran. `_enable_monitoring` calls
`restart_events()` to clear that, and there is a regression test which was confirmed to fail
when the call is removed.

### Added — self-contained HTML reports for both tools

`lineprofiler report <dir> --format html` draws the phase tree as an icicle chart — width is
wall time, colour is the share spent waiting rather than running — alongside the I/O, GPU and
memory blocks. `LineProfiler.to_html()` writes annotated source with per-line heat.

Both are a single file with inline styles and hand-built SVG: no CDN, no webfont, no script,
and no new dependency. Each page also embeds the `--format json` document verbatim, so the
exact numbers behind any figure can be extracted from the page itself; a test asserts the two
agree.

The CLI gains `--format {text,json,html}` and `--output`/`-o`. `--json` still works as a
hidden alias, since it appears in released documentation.

### Added — Python 3.10 and 3.11 support

`requires-python` drops from `>=3.12` to `>=3.10`, which was excluding much of the ML and RL
audience this package was written for. `tomli` is required only on 3.10; the package remains
dependency-free on 3.11 and above. On 3.10, `co_qualname` does not exist, so a dotted
`functions = ["MyClass.step"]` pattern matches on the bare name there — documented rather
than emulated, since guessing a qualified name from a code object would silently match the
wrong function.

### Changed — documentation split out of the README

The README was ~800 lines, which put design philosophy and caveats between a newcomer and
their first result. It is now a short page — install, a runnable example, which-tool — with
the depth moved to `docs/`, including a new comparison against `line_profiler`, py-spy,
Scalene and VizTracer that says plainly when those are the better choice.

### Internal

`sibling_shares()` and `wait_share()` are extracted from the text report's formatters so the
HTML renderer consumes the same derivation rather than reimplementing it. The text report's
golden files are unchanged. `tests/test_overhead.py`'s guard, which checked only
`sys.gettrace()`, now also checks the `sys.monitoring` slots — otherwise the hot-path
assertions would run against a monitoring-instrumented interpreter.

## [0.4.1]

### Added — opt-in, two-line adoption for existing projects

- **`lineprofiler.start_profiling()` / `stop_profiling()`** — the entry/exit alternative to
  `with profiler:`. Wraps ambient line-by-line profiling in two calls instead of a `with`
  block, so dropping this into an existing function or script costs two lines rather than
  restructuring it around a context manager. Opt-in: both are near-free no-ops unless
  `LINEPROFILER_ENABLED` is set (or a `[tool.lineprofiler]` table exists in `pyproject.toml`),
  so they are safe to leave in place permanently. `with profiler:` is unchanged and remains the
  better fit for notebooks and scoped regions.
- **`lineprofiler.accounting.start()` / `stop()`** — the same two-line shape for the accounting
  layer, collapsing the documented `Profiler(..., install=True)` + `.close()` pattern into one
  call each.
- **`lineprofiler.config`** — a shared, opt-in configuration layer read once per process:
  `LINEPROFILER_ENABLED` is the master switch, and an optional `[tool.lineprofiler]` table in
  `pyproject.toml` (`include`/`exclude` path globs, `functions` name globs) narrows what
  `LineProfiler`/`start_profiling()` traces, without touching the profiled code itself. Uses
  the standard library's `tomllib`; no new dependency.

### Documentation — discoverability

- README now leads with a runnable hello-world example and status badges (PyPI, CI, license,
  supported Python versions) before the two-tool comparison table, and adds a `vs. line_profiler`
  section for anyone arriving already familiar with `kernprof`/`@profile`.
- Added `line-profiler`, `tracing` and `jupyter` to the PyPI keywords so the package surfaces
  for those searches.

## [0.4.0]

Driven by a review from integrating 0.3.0 into a MuZero/AlphaZero training pipeline — 16
actors, a learner, an evaluator, an inference server and a reanalyse worker under Slurm. The
integration replaced five hand-rolled timing accumulators and net *removed* code; what follows
is what it cost on the way in.

### Fixed — the report printed labels and shares that were wrong

- **A share no longer divides one I/O counter layer by the other.**
  `unattributed_read_share` took its numerator and denominator through independent `or`
  fallbacks, so unattributed traffic that was entirely page-cache (no block bytes) was divided
  by a total that stayed on the block layer. A run with 110.3 MB of cached reads over 816.0 KB
  of disk reads printed **`13845% of reads`**. Over 100% is at least visibly broken; the same
  expression returned a *plausible* wrong number whenever both layers were non-zero, which is
  the common case. Both operands now come from one layer — the syscall layer, which is what the
  process actually asked for and the only one a warm dataset populates — and the report names
  which layer the percentage is in.
- **Truncated phase labels no longer print names that do not exist.** Six sites across
  `report.py` and `compare.py` cut a label to a fixed width with no marker and no guaranteed
  column gap. `train_step/forward_backward` rendered as `rain_step/forward_backwardr` — a name
  the reader can neither find nor grep, run into the heading beside it — and the comparison
  table turned `iteration/checkpoint_to_object_store` into `/checkpoint_to_object_store`. One
  `format_label` now keeps the tail (the leaf is the informative end), marks the cut with an
  ellipsis, and reserves a column so a label can never abut its neighbour. `_label` was also
  cutting from the wrong end, discarding the leaf it exists to show.
- **A fast counter no longer collides with its own rate.** The rate field filled exactly at
  eight figures, so 64 entries at 19,161,676.6/s printed as `6419,161,676.6/s`. Columns are now
  separated by a literal space, and a number that overflows its field pushes the row right
  rather than being truncated — a truncated number is a wrong number.
- **A relative `run_dir` is resolved before it is propagated.** The default `"profile"` was
  exported to children verbatim through `LINEPROFILER_RUN_DIR`, so a worker with its own
  working directory — which is what a batch system hands each rank — wrote its file somewhere
  else entirely and one run merged as several short ones. Resolved against the constructing
  process's working directory, and deliberately **not** against `$SLURM_SUBMIT_DIR`: portals
  set that to their own installation directory (Open OnDemand reports
  `/var/www/ood/apps/sys/dashboard`), which is neither chosen by the user nor usually writable.
- **`close()` now un-does what an enabled profiler did to the process.** It wrote the final
  snapshot and stopped its threads, but left every process-global hook in place: the `atexit`
  registration, the chained `SIGTERM`/`SIGUSR1`/`SIGHUP` handlers, and the
  `os.register_at_fork` callbacks. A process that merely constructed and closed a profiler was
  permanently altered, and the damage surfaced nowhere near its cause — in the report that
  found this, two in-process profiler tests left the interpreter unable to terminate its own
  forked children, and the failures appeared in an unrelated file several hundred tests away,
  reproducing only in a full run. `close()` now unregisters the `atexit` hook and restores each
  signal disposition, splicing itself out of the chain when something else has chained on top,
  so closing a parent before the child it constructed no longer strands a handler. A handler
  the host installed above the profiler is never clobbered.
- **A `fork()` after `close()` no longer resurrects the profiler.** `os.register_at_fork`
  callbacks cannot be unregistered — CPython offers no API — and the child's callback set
  `_closed` back to `False` unconditionally. Every later fork therefore handed the child a
  *live* profiler: a new writer that re-created the run directory it had finished with, a
  sampler thread and a flush timer, writing files nothing had asked for and, since `close()`
  uninstalls, invisible to the module-level `phase()` that was supposed to resolve it. Closed
  is now terminal, and the callbacks go inert instead.
- **An enabled profiler is no longer immortal.** The three fork callbacks were bound methods,
  so `os.register_at_fork` held every profiler — and its phase trees, thread states and writer
  — for the life of the interpreter. One registration per process now dispatches over weak
  references, and a closed or dropped profiler can actually be collected. This suite alone
  constructs over a hundred.
- **`close()` no longer leaves `LINEPROFILER_RUN_ID` (and the other propagated variables)
  in the environment for whoever constructs the next profiler.** `_propagate_to_children`
  only fills in what was unset; `close()` now removes exactly those keys and nothing a real
  launcher or an outer profiler had already exported. A process that opens and closes several
  profilers in turn — a sweep script running one config after another, or this test suite —
  used to hand the second profiler the first one's run id, which is what let unrelated workers
  read as one attempt purely by accident. Fixing the leak exposed that nothing had ever
  actually propagated a shared run id to workers that don't overlap a still-open parent, so
  `Profiler(run_id=...)` is new: pass it explicitly to correlate several workers into one
  attempt, which is also the only way to do it under `forkserver`, whose daemon environment is
  frozen before any later export can reach it.

### Added — instrumenting without threading an argument

- **`Profiler(..., install=True)` and module-level `accounting.phase()`, `count()`, `current()`
  and `installed_profiler()`.** `phase()` being an instance method meant that instrumenting a
  function five levels down required adding a `profiler` parameter to every caller in between,
  and the reviewer wrote a 70-line process-global shim instead — as, they observed, every real
  integration would. With nothing installed the calls cost ~350 ns, the same as
  `enabled=False`, so library code can carry them permanently. Resolving the installed profiler
  costs ~38 ns on an enabled phase. `close()` uninstalls, a second install warns, and a forked
  child resolves its own profiler rather than the parent's dead one.

### Added — reading a run while it is still running, and afterwards

- **`Profiler.deltas()` and `Profiler.on_snapshot(callback)`.** `merged_tree()` is cumulative,
  so exporting a per-interval figure to W&B or TensorBoard meant every user keeping their own
  copy of the last reading and subtracting it. Quantiles survive the subtraction because
  histograms are bucket counts. A phase idle over the interval is absent rather than present at
  zero, so an exporter never publishes a flat line as activity. Callbacks fire only on the
  periodic flush — not from `close()` or a signal handler, where arbitrary user code could
  deadlock the process on its own final flush — and one that raises is counted and skipped
  rather than stopping the flush timer.
- **`lineprofiler report --json`.** `compare` had it and `report` did not, so the CLI could not
  gate CI or diff sweep arms without re-deriving every share and quantile in the caller.
  `caveats` is part of the document on purpose: a run that lost a worker must not read as
  complete to a program either.
- **A "Using it in tests" section in the README**, for machinery that already existed and was
  undocumented. `merge_run(run_dir, with_samples=False)` makes a run an assertion target for
  cross-process behaviour that unit tests structurally cannot see — a role that never started,
  a phase that never ran, a worker that went quiet. The reviewer found four real incidents that
  way.

### Added — saying which numbers are estimates, and which names are generated

- **`phase(name, sample=0.01)`**, for a region worth breaking down but too hot to measure at
  full rate. Everything derived from one is an **estimate** and is reported as one: the node
  carries its stride, the report prefixes the row with `~` and names the rate, and merging a
  sampled node into a measured one marks the result too. Sampling a phase samples its whole
  subtree — counting children at full rate beneath a parent counted at one in *n* would mix two
  rates in one tree, which is the plausible-wrong-number failure this layer exists to avoid.
  **The saving is about 3.4x, not the sampling rate** (3909 → 1156 ns/call): what a phase costs
  is mostly Python, and sampling can only avoid the measurement, not the call. `count()` at
  384 ns remains the right answer when a rate is all you need.
- **A warning when phase names look generated**, and `strict_names=True` to make it an error.
  `count()` raises on a float; a name built from data was the more damaging mistake and had no
  equivalent protection — it degraded the report rather than raising, and only announced itself
  at `MAX_PHASES`, by which point the run was unreadable. One name in isolation says nothing
  (`conv2d` is a good name), so the check counts distinct names per *shape* and fires at 128,
  well before the fold. It runs only when a path is first admitted, never on the hot path.
- **`thread_names=True`**, nesting each thread's phases under its thread name. `role` is per
  process, and a learner taking gradient steps beside a collector draining a queue is one
  process with two very different answers to "where did the time go?" — both were reported as
  `learner`. The prefixing happens at merge time, so it costs nothing per phase.

### Changed

- **`os._exit()` is documented beside the signal handling.** It skips `atexit` *and* never
  delivers a signal, so neither existing hook fires, and it is the ordinary way a
  multiprocessing entrypoint tears a worker down. The run still parses and looks complete; it
  is simply missing its tail.
- **`wait_ns` is documented as pairing with `wall_ns`, never `self_ns`.** Waiting inside a
  child counts towards the parent, so `wait / self` exceeds 100% for any phase wrapping a
  blocking call — a bug the reviewer hit on their first pass at an exporter.
- **The overhead guidance is a budget, not a rule about loops.** "Keep phase overhead under ~1%
  of the region you are measuring" replaces "not inside an inner simulation loop", which had
  told the reviewer not to do the most useful measurement in the integration: 250 simulations ×
  3 phases × ~2 µs is 1.5 ms against a 2.4 s search.
- **The README leads with the accounting layer**, and says which of the two tools a reader
  wants. The distribution is still `with-line-profiler` and the module still `lineprofiler`; a
  rename would break every existing import to fix what is a documentation problem.
- `enabled=False` no longer constructs a `psutil.Process`: `open_process()` ran unconditionally,
  one line above the check that skips everything else.
- `lineprofiler/accounting/phase.py` is now `phasetree.py`, freeing the name for the
  module-level `phase()` function. Importing a submodule and a function under one name is an
  import-order-dependent trap.

## [0.3.0]

Hardening pass for multi-node HPC use. Three of the fixes below correct cases where the
profiler reported numbers that were **wrong** rather than missing, each leaving behind a file
that parsed cleanly and a report that looked complete.

### Fixed — silently wrong numbers

- **A failed I/O counter read is no longer differenced as a real reading.**
  `read_io_snapshot` returned an all-zero `IoSnapshot` on any failure, and zero is a legal
  counter value. Because the counters are cumulative, the interval *after* a failed read
  computed `current − 0` — the process's entire lifetime of traffic — and billed it to
  whichever phase was open. A window with 2 KB of real reads reported 372.5 GB. `IoSnapshot`
  now carries `available`, unmeasured intervals contribute no bytes, and the report states
  how many intervals were dropped so the totals read as a floor rather than a measurement.
  The same fix applies at `phase(io=True)` boundaries, which record nothing at all when
  either boundary reading failed.
- **One failing snapshot no longer ends flushing for the rest of the run.** `_on_timer`
  re-armed the timer only on success, so a single exception — `ENOSPC` on shared scratch, or
  the live-dict merge race below — froze the worker file permanently. It stayed valid JSON,
  never appeared in the report's lost-worker list, and the run under-reported by however many
  hours remained. The timer now re-arms in a `finally`, `SnapshotWriter.write` returns a
  failure instead of raising, and the sampler thread survives a failed row.
- **`PhaseStats.merge` no longer iterates a live dictionary.** A snapshot taken while an
  owning thread introduced a new counter name raised
  `RuntimeError: dictionary changed size during iteration`, which killed the flush thread.
- **Re-running into the same directory no longer merges the previous attempt.** Runs now
  carry a run id, propagated to children through `LINEPROFILER_RUN_ID` and recorded in every
  worker file. `merge_run` reports the newest attempt and names the superseded ones instead
  of absorbing them, which used to inflate every total for a requeued Slurm job.
- **`LineProfiler` no longer leaks its trace function.** Re-entering an active instance saved
  the profiler's own callback as the tracer to restore, leaving a global trace function
  dispatched on every Python call for the lifetime of the process. It now raises.
- **`LineProfiler` autodetection no longer collapses to a single file.** Without a `.git`
  directory, `_find_repo_root` returned the resolved *file* path, so only that one module was
  ever profiled. It now returns the containing directory.

### Added — multi-node

- **Host, rank and job id in every worker file**, resolved from Slurm, `torch.distributed`,
  Open MPI, MPICH, MVAPICH or PBS/LSF/Flux. The report names the nodes involved. Previously a
  worker recorded only a pid, and *which node is slow?* was unanswerable.
- **Worker files are sharded by host** (`workers/<host>/`), so a run does not concentrate two
  files per rank plus a rename per flush into one directory — a single-MDT hot spot on Lustre.
- **`SIGUSR1` and `SIGHUP` flush before exit**, alongside `SIGTERM`. Slurm sends `SIGUSR1`
  ahead of preemption (`--signal=USR1@120`) and its default disposition skips `atexit`, so
  everything since the last periodic flush was lost precisely when it mattered most.
- **`fsync` before rename**, on the file and its parent directory. `os.replace` is atomic on
  ext4, XFS, NFS and Lustre, but atomicity is not durability: without this a node that lost
  power could leave a zero-length worker file.
- **`metadata.json` is written atomically**, through a temporary and `os.replace`. The
  previous check-then-write raced across ranks starting simultaneously, and on a shared
  filesystem — where `exists()` is cached — left JSON that parsed as nothing.

### Added — bounded resources

- **A cap of 4096 distinct phase paths per thread**, past which phases fold into their parent
  and the profiler warns once, naming the path that overflowed. A phase name built from data
  (`phase(f"episode_{i}")`) was an unbounded leak whose only symptom was a process slowly
  getting fatter.
- **`lineprofiler report --no-samples`** and `merge_run(..., with_samples=False)`, reading
  phase trees without the resource samples. Samples dominate merge memory: a 12-hour worker
  holds about 28 MB of them, roughly 1.8 GB across 64 workers, and the derived intervals
  double the peak.
- **A structurally invalid worker file now costs one worker, not the run.** Valid JSON with
  the wrong shape used to raise straight out of `merge_run`.

### Changed

- **`compare` reports sample size and median alongside the mean.** It was ratios of arithmetic
  means with no `n`, presenting a phase measured twice with the same authority as one measured
  ten thousand times. Rows below 30 entries are marked thin; rows whose mean and median
  disagree are marked as tail changes, which is a different bug with a different fix. Phases
  present in only one run now sort first instead of below every improvement.
- **The report's process count comes from worker files, not distinct pids.** Pid namespaces
  are per-node, so the header undercounted every multi-node run.
- **`LOST` became `CAVEATS`**, and now also names superseded attempts, workers reporting
  failed writes, and workers that stopped writing well before the run ended.
- `selfio`'s lock is reentrant. A `SIGTERM` arriving while the main thread held it inside an
  `io=True` phase boundary deadlocked the process on its own final flush.

### Validated

The GPU paths had only ever been exercised against a `SimpleNamespace` stub of pynvml. They
are now verified on an A100-SXM4-40GB with real `torch`, `nvidia-ml-py` and `nvtx`:

- `nvmlDeviceGetProcessUtilization` does raise `NVMLError_NotFound` on an empty sampling
  window, which the stub assumed and nothing had checked.
- `phase(sync=True)` measured **687 ms** against **0.40 ms** for the same unsynchronised
  matmul chain — a factor of 1,718, confirming both that asynchronous launches make an
  unsynchronised phase meaningless for GPU work and that the option fixes it.
- The `torch` backend window writes a real Chrome trace.

New `tests/test_gpu_hardware.py` keeps these honest; it skips cleanly without the hardware.

### Infrastructure

- GitHub Actions running ruff, mypy and pytest on 3.12 and 3.13, plus coverage, an overhead
  job, and a build job that runs the test suite from the unpacked sdist.
- `tests/test_overhead.py` asserts loose upper bounds on the phase hot path, so the ns/call
  figures the README quotes cannot rot unnoticed.
- Tests and benchmarks ship in the sdist, so a downstream packager can validate a build.
- `[project.urls]` pointed at a repository that does not exist (`lineprofiler` rather than
  `withlineprofiler`).

## [0.2.0]

- Added `lineprofiler.accounting`: phase trees, mergeable histograms, cross-process snapshot
  merge, run comparison, text reports and the `lineprofiler` CLI.
- Corrected I/O attribution: both counter layers, self-I/O deduction, and unattributed bytes
  labelled rather than billed to the root.
- Per-device and per-process GPU utilisation; `phase(sync=True)`.
- **Breaking:** `Profiler.io_counters()` returns an `IoSnapshot`, not a 2-tuple.

## [0.1.1]

- Renamed the distribution to `with_line_profiler`.
- Reduced line-profiler overhead.

## [0.1.0]

- Initial `LineProfiler`.
