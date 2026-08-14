# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[semantic](https://semver.org/spec/v2.0.0.html).

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
