# Plan: `lineprofiler` as a front end over `line_profiler`

Status as of 2026-09-06, against `line_profiler` 5.0.2 and Scalene 2.3.0. **Every phase is shipped**: 0 to 2 and 4 in 0.9.0, phase 3 in 0.10.0. Every number below is measured on one machine by `benchmarks/bench_lineprofiler.py`, which is a command rather than a session, so any claim here can be rechecked.

The goal this serves: profiling that is **usable** (a `with` block, or two lines, and nothing to decorate), **fast** (the C engine's cost, not a Python callback's) and **built on a trusted library** rather than on a second implementation of one.

## 1. Where it stands now

600,000 line events, best of three runs.

| | Runtime | Per line event | vs. no profiler |
|---|---|---|---|
| no profiler | 14 ms | 24 ns | 1.0x |
| **`with profiler:` — the default engine** | **142 ms** | **237 ns** | **10x** |
| `line_profiler`, called directly | 129 ms | 215 ns | 9x |
| `engine="builtin"`, `sys.monitoring` | 485 ms | 808 ns | 34x |
| `engine="builtin"`, `sys.settrace` | 500 ms | 833 ns | 35x |
| Scalene 2.3.0, `--cpu-only` / with memory | sampled | sampled | 2.0x / 2.7x |

The `with` block costs what `line_profiler` costs; the difference between 237 and 215 ns is inside the noise of this benchmark. The pure-Python engine is 3.4x more expensive and is now only the fallback.

## 2. What shipped in 0.9.0

### Phase 0 — the engine tells the truth (done)

Three defects that produced confident wrong numbers, all with regression tests:

- **A call line was billed almost nothing.** `b = inner()` reported 2.2 µs for a 9.5 ms call. The clock reset when the callee started and the caller never got its time back. It was also inconsistent — a call into the standard library *was* inclusive, because those frames are not traced. Both engines now keep a stack of open frames and let the caller's clock run across the call, which is what every other line profiler means by that column. **This changes reported numbers**, and the old ones were wrong rather than differently scoped.
- **Threads lost hits or were never recorded.** Four threads recorded 370,794 of 400,000 hits under `monitoring` (one shared "current line", mutated by every thread) and *nothing* from worker threads under `settrace` (`threading.settrace()` was never installed). The frame stack is now thread-local, `threading.settrace()` is installed alongside `sys.settrace()`, and the per-function record is created with `setdefault`. Four threads now record exactly 400,000.
- **A generator's `yield` line absorbed its consumer's time.** `PY_YIELD` was not subscribed, so a suspended generator looked like a running one. Now handled like a return, with `PY_RESUME`/`PY_THROW` reopening the frame.

Also: `start_profiling(enabled=True)` profiles without the environment variable, and the `filename not in folde` typo is gone.

### Phase 1 — `line_profiler` as the engine (done, and it is the default)

`LineProfiler(engine="auto" | "builtin" | "line_profiler")`, defaulting to the C engine wherever it imports. `line_profiler` is now a real dependency rather than an extra: it ships wheels for every supported interpreter, and making it optional would have meant the documented performance was the one most users did not get. The pure-Python engine remains, selected automatically where the import fails and by `engine="builtin"` or any `backend=`.

Discovery is what makes the `with` block work without decorators: a `sys.monitoring` hook on tool slot 4 registers each admitted code object the first time it runs, then opts out of that code, so it costs one callback per function per session. Modules already imported and the frames already on the stack are registered up front. Closures, methods, `runpy` scripts and modules imported inside the block are all covered on 3.12+.

Two design decisions are load-bearing, and both trade a rare wrong number for a common missing one. They are documented in the module and in CLAUDE.md:

- **Nothing the caller wrote is modified.** `line_profiler` disambiguates identical bytecode by padding a function's code object. A function found at its own first call is mid-flight when that happens, and the running frame keeps the old code — so the whole call goes unrecorded. Functions are registered through a holder with padding suppressed.
- **A line-overlap check replaces what padding bought.** `line_profiler` indexes a line by the hash of its bytecode and its line number. Two functions with identical bytecode and overlapping lines would share buckets, so the second is left unprofiled and named in `skipped` rather than reported as a blend. This is also why the engine instance is per session: a process-global one let a script's function collide with an unrelated one from a session that had already finished.

### Phase 2 — parity where it was cheap (done)

- **`dump_stats(path)`** writes `line_profiler`'s `.lprof` pickle from either engine. `python -m line_profiler run.lprof` displays it and `LineStats.from_files()` merges several — verified end to end by a test.
- **`lineprofiler run script.py [args...]`** — `kernprof` without the decorators. `--top`, `--functions`, `--html`; the script's exit status is preserved and the summary prints even when it raises.
- **`print_stats` defaults to source order** and both printers take `stream=`, so a table can go into a log or a test. The cross-function ranking still defaults to time, because ranking is what that table is for.

### Phase 4 — say what was measured (done)

`benchmarks/bench_lineprofiler.py` produces the table in section 1. `docs/comparison.md` carries the measured Scalene figures and no longer claims an overhead advantage this package does not have.

## 3. Phase 3 — line profile by named region (shipped in 0.10.0)

`with profiler.region("select"):` partitions the per-line statistics by a named block, and `print_regions()` prints one ranked table per region with its share, entry count and cost per entry. Both engines support it and agree on hit counts.

### The approaches, and why the plan's first choice lost

The obstacle is that the C engine reports totals, not when a line ran, so regions cannot be recovered after the fact. Four mechanisms were considered and the first three were measured or reasoned out before any was built.

**1. Snapshot and difference at each boundary** — the plan's own first choice, and the one the plan said to measure before committing. `get_stats()` on entry and exit; the difference is the region's share. Correct, needs no new timing, no new state. **Rejected on measurement: a full walk costs ~1.5 ms** on a 600-function registry, because `line_profiler` indexes one hash per bytecode offset and the walk visits all of them. Two walks per entry means four regions around a 200-iteration loop spend ~2.4 s inside the profiler doing bookkeeping. There is also no cheaper snapshot through the public API — `c_code_map` converts the whole C map to a Python dict on every access.

**2. One `line_profiler.LineProfiler` per region, enabled only while that region is open** — **shipped.** The region's share is measured rather than differenced. Prototyped before committing: a region around a third of a workload recorded 50,001 of 150,003 hits, exactly its third. **~7.6 µs per boundary, about 400x cheaper than differencing**, and nesting is inclusive for free because every open region's profiler is enabled at once. The costs are that each region carries its own copy of the index, and that every function registered with the session must be registered with each region too — including functions discovered later, or a region reports a confident zero for code first seen elsewhere.

**3. Tag the bill at billing time** — what the pure-Python engine does, since it writes the record itself and can simply append the open region names. Free of any second profiler, but only available to that engine. Shipped as the builtin engine's implementation, which is why the two have opposite cost profiles rather than one being a port of the other.

**4. Make regions builtin-engine-only.** Rejected without measurement: it would have meant the feature forced a 3.6x slower engine, which is the opposite of what 0.9.0 was for. A feature that only works on the slow path is a feature that quietly punishes people for using it.

### What the measurement changed

Two things were caught by measuring rather than reasoning, and both are in the benchmark now so they cannot rot:

- **The boundary was 12 µs before inlining.** Most of it was this package's own Python lines at the boundary being traced by the engine it had just armed. Caching `_RegionScope` per name and giving it the engine's enable/disable pair directly took it to ~7.6 µs. It is the same discipline the accounting layer applies to `_PhaseScope`.
- **The first version of the benchmark timed its own loop**, and reported a *negative* boundary cost for the builtin engine once a control loop was subtracted — the control's context manager was itself in-project and therefore traced, so it cost more than the thing being measured. The per-boundary figure now profiles an empty directory, so only the switch is timed.

### The resulting trade, which is worth knowing before instrumenting

| Engine | Per line event, region open | Per boundary |
|---|---|---|
| `line_profiler` | ~400 ns (from ~240) | ~7.6 µs |
| `builtin`, `sys.monitoring` | ~1,435 ns (from ~900) | ~430 ns |
| `builtin`, `sys.settrace` | ~1,420 ns | ~1,760 ns |

Many small regions favour the builtin engine; a few regions around substantial work favour the C engine, which is the common case and the default. The placement rule is the accounting layer's: put a region where the entry count is bounded by your loop, not by your data.

## 4. What is left

Nothing on this plan. The remaining open questions are ordinary product ones rather than planned work:

- **Regions in the HTML report.** `to_html()` still renders the session only. A per-region view would need a design pass on the page before it is worth building.
- **Regions across processes.** A region is a window in one process. Answering the same question across a pipeline is what the accounting layer is for, and the two should stay separate tools rather than grow into each other.

### Deliberately not planned

A sampling mode, memory attribution, native-versus-Python time. That is Scalene's territory, it is well served there, and `docs/comparison.md` points at it. Adding a worse version of it would make this package harder to describe and no better to use.

## 5. What stayed fixed

- `with profiler:` and `start_profiling()` … `stop_profiling()` on separate lines are the API. The engine is a keyword argument, never a different call site.
- Project-folder scoping and the `[tool.lineprofiler]` include / exclude / functions globs mean the same thing under both engines.
- The accounting layer is untouched and still depends on nothing. It never imports the line profiler or anything the line profiler needs.
