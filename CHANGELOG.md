# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[semantic](https://semver.org/spec/v2.0.0.html).

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
