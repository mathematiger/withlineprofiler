# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`with_line_profiler` (PyPI name; import as `lineprofiler`) ships two independent tools:

1. **`lineprofiler.LineProfiler`** — a context-manager-based line-by-line profiler. Wrap code in `with profiler:` and it records per-line hit counts and timing for code that lives inside a project folder.
2. **`lineprofiler.accounting`** — a low-overhead semantic accounting layer for long, multi-process RL training runs. You name the regions; it stores aggregates only (counts, sums, a fixed-bucket histogram), so memory per phase is constant regardless of run length. Roughly 3.4 µs per phase, designed to stay enabled for a twelve-hour run.

The two share nothing but the distribution. `accounting` never imports `LineProfiler`.

## Commands

The repo root is `withlineprofiler/` (the parent folder is not a git repo). Run everything from there.

- Install (dev): `poetry install --extras resources` (creates `.venv` with the `dev` + `test` groups; **install psutil or the I/O and memory paths silently degrade to their absent-capability branches and stop being tested**)
- Install for GPU validation: `poetry install --extras all` plus `torch`. The torch wheels need their CUDA libs on `LD_LIBRARY_PATH`; see the `nvidia/*/lib` trick in CI.
- Lint: `poetry run ruff check lineprofiler tests` (config in `pyproject.toml`: line-length 100, target py312, rules `E,F,W,I,N,UP,ANN,B,C4,SIM`, ignoring `ANN101/ANN102` for ruff `^0.5`)
- Type-check: `poetry run mypy lineprofiler tests` (configured `strict = true`, `python_version = 3.12`)
- Test: `poetry run pytest tests/`
- Note: `test_the_profilers_own_writes_are_not_attributed_to_a_phase` is timing-sensitive (0.3 s window, background flush threads racing the `selfio` deduction) and has been seen to fail once under heavy machine load. Re-run before investigating.
- Single test: `poetry run pytest tests/test_profiler.py::test_sleep_line_dominates_timing -q`
- Build: `poetry build` (build backend is `poetry-core`; metadata is PEP 621 `[project]`, the wheel packages only the `lineprofiler` dir)

- Benchmark: `poetry run python benchmarks/bench_accounting.py` (the numbers quoted in the README come from here; re-run and update them if the hot path changes)
- Regenerate report golden files: `LINEPROFILER_UPDATE_GOLDEN=1 poetry run pytest tests/test_accounting_report_golden.py`
- Coverage: `poetry run pytest tests/ --cov=lineprofiler --cov-report=term-missing`
- CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs ruff, mypy, pytest on 3.12/3.13, coverage, the overhead job, and a build job that runs the suite from the unpacked sdist.

Tests (269 total, ~17 s; plus 8 GPU-only in `test_gpu_hardware.py`):

- [tests/test_profiler.py](tests/test_profiler.py) — the line profiler (27). Several tests reach into private state (`_is_in_project_folder`, `_project_cache`, `_source_cache`, `_function_stats`), so renaming those attributes breaks tests.
- [tests/test_accounting.py](tests/test_accounting.py) — histogram, phase tree, counters, snapshots, merge.
- [tests/test_accounting_resources.py](tests/test_accounting_resources.py) — roles, sampler, analysis, compare, backends, annotation, per-device GPU utilisation, synchronised phases, and a parametrised settings matrix. The GPU tests drive a `SimpleNamespace` stub of pynvml, so they run without a device.
- [tests/test_accounting_multiprocess.py](tests/test_accounting_multiprocess.py) — `{spawn, fork, forkserver}` × `{1, 4, 16}` workers, fork safety, failure modes.
- [tests/test_accounting_report_golden.py](tests/test_accounting_report_golden.py) — byte-for-byte report/compare output, built from fixed numbers so it is deterministic.
- [tests/test_accounting_resilience.py](tests/test_accounting_resilience.py) — the failures that used to produce *wrong* numbers rather than missing ones: I/O counter gaps, flush-thread death, attempt merging, invalid worker files, multi-node identity, scheduler signals, phase cardinality.
- [tests/conftest.py](tests/conftest.py) — two suite-wide hygiene fixtures: one closes any enabled profiler a test left open (sixteen do), the other fails the *session* if `SIGTERM`/`SIGUSR1`/`SIGHUP` are not back where they started. The leak they guard against is cumulative and surfaces far from its cause, so a per-test check would not catch it.
- [tests/test_overhead.py](tests/test_overhead.py) — loose upper bounds on the hot path so the README's ns/call table cannot rot. Bounds are ~4x measured; they catch order-of-magnitude regressions, not drift.
- [tests/test_gpu_hardware.py](tests/test_gpu_hardware.py) — needs a real GPU, driver and torch. Skips cleanly without them. This is what validates the pynvml stub's assumptions.

The version number lives in **two** places that must be bumped together: `[project].version` in `pyproject.toml` and `__version__` in [lineprofiler/__init__.py](lineprofiler/__init__.py). Both are currently 0.4.0. [CHANGELOG.md](CHANGELOG.md) records what changed and why.

Everything targets Python 3.12 — `requires-python`, mypy, ruff and the checked-in `.venv`.

## Architecture

Everything lives in [lineprofiler/profiler.py](lineprofiler/profiler.py) (~465 lines); [lineprofiler/__init__.py](lineprofiler/__init__.py) only re-exports the public API (`LineProfiler`, `FunctionStats`, `LineStats`). Pieces:

- `LineStats` / `FunctionStats` — dataclasses holding per-line and per-function accumulated `hits` and `total_time`. Functions are keyed by the tuple `(filename, function_name, first_line)`.
- `LineProfiler` — the context manager. Core mechanism:
  - `__enter__` registers `self._trace_callback` via `sys.settrace` (saving the previous tracer); `__exit__` restores it.
  - `_trace_callback` handles `call` / `line` / `return` events. **Timing model:** the delta between two events is attributed to the *previous* line (`self._last_key`, `self._last_line`). The reference timestamp for the next line is taken at the *end* of the callback (a second `perf_counter()` call), so the profiler's own bookkeeping is excluded from the reported per-line times. The line being attributed to is identified from the current `frame` (`_ensure_function`), so the caller's lines are still timed correctly after a nested in-project call returns.
  - **Project filtering is central.** Only frames whose filename is under `self._project_folder` are traced (`_is_in_project_folder`). If `project_folder` is not passed to the constructor, it auto-detects by walking up from the *caller's* file to the nearest `.git` directory (`_find_repo_root`). This is why profiling stays scoped to the user's own code instead of stdlib/site-packages.
  - **Caching keeps overhead/memory down.** `_is_in_project_folder` caches its verdict per filename in `_project_cache` (so `Path.resolve()` runs once per file, not per `call` event). Source lines are read once per file into `_source_cache`, and every `FunctionStats.source_lines` for a given file is the *same* dict object — so a file's source is held in memory only once regardless of how many of its functions are profiled.
  - Reporting: `print_stats` (per-function tables), `print_global_top_stats` (top-N lines across all functions), `get_stats` (raw dict), and `clear`/`reset` (reset state and caches; `reset` is an alias). Both printers re-check `_is_in_project_folder` before emitting a function, and truncate the source column (50 chars per-function, 40 global) — so output width is fixed at 100 / 130 columns.
- `_GlobalLine` — private dataclass at the bottom of the module; the flattened row type produced by `_collect_global_lines` and consumed by `_print_global_row`. It is the only structure that carries a display filename (relative to the project folder, else basename).

## Architecture — `lineprofiler/accounting/`

Layered bottom-up; each module imports only from the ones above it in this list.

- `histogram.py` — `DurationHistogram`: 512 log-spaced buckets, 8 per octave, indexed with `int.bit_length()` (no float maths, no `math.log`) because this runs on every phase exit. Mergeable by summing buckets, which is what makes quantiles survive both the snapshot and the cross-worker merge.
- `phasetree.py` — `PhaseStats` (calls / wall / cpu / child_wall / histogram / counters / sample_stride) and the `PhaseTree`, a `dict` keyed by the full phase path. `self_ns = wall - child_wall`; `wait_ns = wall - cpu`, the blocked-on-something estimate. **`merge_trees` copies nodes on insert** — sharing them aliased worker statistics and corrupted per-worker totals.
- `capabilities.py` — every optional dependency (`psutil`, `torch`, `pynvml`, `nvtx`), resolved once and degraded to `None`. Nothing else imports them directly.
- `selfio.py` — the bytes the profiler's own sampler and snapshot writes cost, so they can be deducted from your phases. Two totals: `chars` is exact from the buffer length, `block_bytes` is **measured** by bracketing each of its own writes, because filesystem journal and inode churn amplify a 14 KB run of bookkeeping into ~116 KB of block traffic. Deducting the char figure from `write_bytes` would leave most of the overhead in place.
- `sampler.py` — the 1 Hz daemon thread: RSS, `io_counters`, CUDA allocator, NVML utilisation, each row tagged with the deepest open phase. Utilisation is recorded **per device and at two levels**: `gpu_utils` is whole-device busy (every process's kernels) and `gpu_proc_utils` is what NVML attributes to this pid. Handles are resolved once in `_open_devices`; `gpu_util` survives as the mean over devices so files written by either version parse under both. Brackets the run with a baseline and a final row, because the OS counters are cumulative and only useful as differences. `IoSnapshot` carries **both** counter layers: `*_bytes` (block device) and `*_chars` (syscall). Only the pair together makes a page-cached read visible — a warm dataset moves zero disk bytes.
- `snapshot.py` — one complete JSON document per worker at `workers/w_<pid>_<uuid8>.json`, replaced atomically via `os.replace`. That *is* the crash-resilience mechanism: a torn write leaves the previous file intact. The uuid matters because a restarted worker reuses its rank but not its pid.
- `analysis.py` — pure derivation from samples. **`analyse_processes` differences each process separately**; pooling samples across processes inflates totals and misattributes them. GPU utilisation combines the two levels differently on purpose: whole-device readings are *averaged* across processes (they all observed the same device) while per-process readings are *summed* (they are disjoint slices). Bytes with no phase open go to `NO_PHASE` (`"(no phase open)"`), never to the root, and `unattributed_*_share` reports how much landed there — a high share means the sample rate was too coarse, not that the root did the work.
- `backend.py` — `Backend` enum plus `BackendWindow`, which starts one heavy profiler across a range of entries into a phase you name. A single enum value, so two backends are not representable.
- `profiler.py` — the `Profiler` itself. Statistics are **per thread**, merged only at snapshot time, so the hot path takes no locks. `_PhaseScope.__exit__` inlines `record()` and `observe()`, and the annotation guards, because at ~3 µs per phase a Python-level call is measurable.
- `report.py` / `compare.py` / `cli.py` — rendering. The report groups by role and descends past any single-phase level to find where the work actually branches.

### The wrong-numbers class of bug

This layer's worst failure mode is not crashing — it is continuing, writing a file that parses,
and reporting a result that looks complete. Four such defects have been fixed and each has a
regression test; the *pattern* is what to watch for when changing this code:

- **Never let "unmeasured" be represented by a valid value.** `IoSnapshot.available` exists
  because an all-zero snapshot was indistinguishable from a real reading, and the counters are
  cumulative, so one failed read fabricated the process's entire lifetime of traffic on one
  phase. Anything differenced needs the same treatment.
- **Never let a background thread die silently.** `_on_timer` re-arms in a `finally`;
  `SnapshotWriter.write` returns `False` rather than raising; the sampler survives a bad row.
  A frozen worker file is valid JSON and is invisible without the staleness check in
  `_degraded_rows`.
- **Never merge two attempts.** Worker files carry a `run_id`; `_split_by_attempt` keeps the
  newest and reports the rest.
- **Never let one bad file cost the run.** `_read_worker` guards everything after the JSON
  parse, not just the parse.

### Accounting gotchas

- The hot path is the product. Before changing `_PhaseScope`, run the benchmark; a method call costs ~80 ns there.
- `measure_cpu=True` doubles phase cost: `time.thread_time_ns()` is a real syscall (`CLOCK_THREAD_CPUTIME_ID` is not in the vDSO, ~590 ns).
- The profiler stops its own threads around a `fork` (`os.register_at_fork`) so enabling it never adds fork-deadlock risk, and re-initialises fully in the child so it does not inherit the parent's file or tree.
- **`close()` un-does the process-global state construction created**, and this is load-bearing: the `atexit` hook is unregistered and each chained signal restored. Restoring is a *splice*, not a pop — closing order need not match construction order, so a profiler that is no longer top of the chain hands its predecessor to whoever holds it as theirs. A handler the host installed above us is never clobbered; we stay installed and inert.
- **Fork callbacks are registered once per interpreter, not once per profiler**, and dispatch over `_fork_registry` weak references. `os.register_at_fork` has no unregister, so bound methods there made every profiler immortal. A closed profiler is skipped, and `_reinitialise_after_fork` must **never** reset `_closed` — that used to hand any later fork a live writer, sampler and flush timer for a profiler the process had finished with.
- Environment propagation (`LINEPROFILER_PROFILE`, `LINEPROFILER_RUN_DIR`, `LINEPROFILER_RUN_ID`) reaches `spawn` children but **not** `forkserver` children, whose daemon froze its environment at start. There is a test asserting this limitation.
- **`_propagate_to_children` only fills in what was unset, and `close()` only removes what it filled in.** `os.environ` outlives the `Profiler` instance, so a long-lived process that opens and closes several profilers in turn — this test suite, or a sweep script running one training run per config — must not have the second profiler inherit the first one's `run_id` from a variable the first left behind: same-directory reruns would stop looking like separate attempts to `_split_by_attempt` and get silently merged instead of one superseding the other. Anything already in the environment when a profiler is constructed (a real launcher's export, or a still-open profiler higher in the call stack) is left alone, both on the way in and on the way out.
- `io_*` counters hold bytes, not work units, so `_counter_rows` skips them; they are rendered by `_exact_io_block`.
- Worker files live at `workers/<host>/w_<run_id>_<pid>_<uuid8>.json`. `merge_run` uses `rglob`; anything reaching into that layout (several tests do) must too.
- `MAX_PHASES` (4096) bounds distinct paths per thread; past it phases fold into the parent and `_note_phase_overflow` warns once. `MAX_DEPTH` (32) bounds nesting. **Both fold silently in the sense that folded phases record nothing at all** — not their time into the ancestor.
- `identity.py` is the only place that reads scheduler environment variables. It imports no scheduler library and never will.
- **`run_dir` is resolved to absolute at construction**, against the *cwd* — never against `$SLURM_SUBMIT_DIR`, which portals point at their own install directory (Open OnDemand: `/var/www/ood/apps/sys/dashboard`). The relative default used to be exported to children verbatim and scattered one run across per-rank working directories.
- **The ambient profiler is process-global mutable state** (`_installed` in `profiler.py`). Two ways it goes wrong silently: a fork that does not re-point it (handled — the module global is copied and `_reinitialise_after_fork` fixes the object it names), and `close()` that does not uninstall (it does). Module-level `phase()`/`count()`/`current()` no-op at ~350 ns when nothing is installed.
- **`sample=` produces estimates, and they must stay labelled.** `PhaseStats.sample_stride` is 0 for measured and *n* for sampled; `merge` takes the max so one sampled contributor taints a total; the report prefixes `~` and names the rate. A skipped entry suppresses its **whole subtree** via `_ThreadState.suppressed` — children at full rate under a sampled parent would mix two rates in one tree. The saving is ~3.4x, not the sampling rate: the cost is Python call overhead, not measurement.
- **Phase-name shape checking runs in `_admit` only** — once per distinct path, never on the hot path — which is what makes a regex there free.
- `selfio._lock` is an `RLock` on purpose — a signal handler runs on the main thread, which may already hold it inside an `io=True` phase boundary.
- `phase(name, sync=True)` calls `torch.cuda.synchronize` at **both** ends. Exit-only would bill the phase for kernels an earlier phase queued. The callable is resolved once into `Profiler._cuda_sync` and bound at scope construction, so a phase that does not synchronise costs one `is not None` test; `torch.cuda.is_available()` initialises the driver, so it must never be called per phase.
- A missing NVML/CUDA capability is expressed as `None`, never as a no-op lambda — the hot path skips on identity rather than paying for a call that does nothing.
- `nvmlDeviceGetProcessUtilization` **raises** when its window held no samples; that is the ordinary idle case, not an error worth logging. The timestamp cursor (`_proc_util_since`) advances so each NVML sample is counted once.
- An `io=True` phase records **four** counters: `io_read_bytes`/`io_write_bytes` (block layer) and `io_read_chars`/`io_write_chars` (syscall). `write_bytes` is writeback-dependent and only lands on the writing phase if it `fsync`s; `write_chars` is always charged correctly. Prefer chars when asking who wrote, bytes when asking what the device saw.
- `Profiler.io_counters()` returns an `IoSnapshot`, not a 2-tuple. Test availability with `.is_empty()`.
- `selfio.reset()` runs in `_reinitialise_after_fork`: the child's OS byte counters restart at zero, so an inherited overhead total would over-deduct for the child's whole life.

## Gotchas

- `sys.settrace` is global and single-tracer; this profiler is not thread-safe. Recursive/nested calls share one `FunctionStats` per function key (no per-call-depth breakdown), but their per-line timing is correct because the line is identified from the live `frame` on every event rather than from a single remembered key.
- A few user-facing strings contain typos (e.g. `"filename not in folde"` in `print_stats`); preserve or fix deliberately, don't assume they're bugs to silently rewrite.
- `get_stats()` returns the live internal dict, not a copy — callers can mutate profiler state through it, and `clear()` empties the same object.
- `__init__` snapshots `sys.gettrace()`, but `__enter__` re-reads it; only the `__enter__` value is restored on exit. Constructing a profiler long before entering it is therefore safe.
- `print(...)` calls carry `# noqa: T201` (and `PLR2004`, `ARG002`) for rule sets not currently in `select`; leave them, they document intent if the rule set is widened.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
 

 ## 5. Test your code

 Check afterwards first for mypy errors and fix them; and then for ruff errors (also fix).
 Do not delete TODOs in this process.
 This code should be TLR7-ready at the end of this project.

 ## 6. Clean Code — Single Purpose Functions

**Every function does one thing. Name it after what it does.**

- A function should have one reason to exist. If you need "and" to describe it, split it.
- Keep functions short (< 30 lines as guideline). If it scrolls, it's too long.
- No side effects hidden behind innocent names. `get_X()` must not mutate state.
- Extract repeated logic into named helpers — but only when used ≥ 2 times.
- Parameters: fewer is better. More than 4? Consider a dataclass or restructuring.

Ask yourself: "Can I understand this function without reading its body?" If not, rename or restructure.

## 7. Readability — Structure Over Comments

**Code should read top-down like a narrative. Util files are for shared plumbing.**

- Public functions at the top, private helpers below. Reader sees intent before implementation.
- Group related logic into clearly named functions — prefer readable call chains over inline blocks.
- Extract pure utility logic (math helpers, string formatting, generic transforms) into `*_utils.py` files alongside the module that uses them.
- Don't create a god-object `utils.py` — scope utils to their domain (e.g. `mcts_utils.py`, `network_utils.py`).
- Naming: variables and functions should make comments unnecessary. `filtered_actions` > `fa`. `compute_td_target` > `calc`.
- Blank lines separate logical blocks within a function — treat them like paragraph breaks.

The test: A new team member should understand the module's flow by reading function names alone.