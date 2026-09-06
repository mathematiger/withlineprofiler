# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[semantic](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.10.0] - 2026-09-06

### Added — `with profiler.region("select"):`, per-line statistics by named phase

A line profile says line 52 is slow. It cannot say that line 52 is slow *during selection* and fine during backpropagation, because the same line is one row however many phases run through it. Naming the phases makes the split a measurement:

```python
with profiler:
    for _ in range(iterations):
        with profiler.region("select"):
            node = select(root)
        with profiler.region("rollout"):
            reward = rollout(node)

profiler.print_regions()
```

`print_regions()` prints each region's slowest lines with its share of the profiled total, its entry count and its cost per entry. `region_stats()` returns the same data in `get_stats()`'s shape, and `region_entries()` the counts. Both engines support it and agree on the hit counts.

Regions nest and the reading is **inclusive**: a line inside `rollout` is billed to `rollout` and to every region open around it, the same way a phase's wall time in the accounting layer includes its children. Shares are taken against the profiled total, so they do not sum to 100% — regions may nest and need not cover the whole run — and the report says so rather than normalising to a tidier wrong number.

A region is a **window, not a call stack**: every line executed while it is open is billed to it, including lines in the frame that opened it and lines on other threads. Opening regions concurrently on several threads does not mean anything useful, and is documented as such rather than silently producing a number. Entering a region while the profiler is not active records nothing and costs one boolean test, so the calls are safe to leave in code that is usually not profiled.

### How it is implemented, and what was rejected

The obvious design is to snapshot the statistics at each region boundary and difference them. It was measured first and rejected: a full `get_stats()` walk costs **~1.5 ms** on a 600-function registry, because `line_profiler` indexes one hash per bytecode offset and the walk visits all of them. At two walks per entry, four regions around a 200-iteration loop would spend about 2.4 seconds inside the profiler doing nothing but bookkeeping.

What shipped instead gives each region its own `line_profiler.LineProfiler`, enabled only while that region is open. The region's share is then *measured* rather than differenced, at **~7.6 µs** per boundary against the 3 ms the snapshot approach would have cost — about 400x cheaper. Every function registered with the session is registered with each region's profiler as well, including functions discovered later, or a region would report a confident zero for code that ran inside it but was first seen elsewhere.

The pure-Python engine needs none of that: it does its own billing, so it appends the open region names to the record it is already writing. That gives it the opposite cost profile, and the benchmark reports both:

| Engine | Per line event, region open | Per boundary |
|---|---|---|
| `line_profiler` | ~400 ns (from ~240) | ~7.6 µs |
| `builtin`, `sys.monitoring` | ~1,435 ns (from ~900) | ~430 ns |
| `builtin`, `sys.settrace` | ~1,420 ns | ~1,760 ns |

Many small regions favour the builtin engine; a few regions around substantial work favour the C engine, which is the usual case and the default. The placement rule is the accounting layer's: put a region where the entry count is bounded by your loop, not by your data.

`_RegionScope` is cached per name and does its own bookkeeping rather than calling back into the profiler, for the same reason the accounting layer inlines `_PhaseScope.__exit__`: every Python line executed at a boundary is a line the profiler is itself timing. Inlining took the C engine's boundary from ~12 µs to ~7.6 µs.

The C engine's region total *excludes* the cost of opening and closing the region, which the session's total absorbs into the surrounding line — so a run's regions can sum to slightly less than the session. That is the cleaner number of the two.

### Known limit — region shares are approximate, and the report says so

Under the C engine a region is timed by its own `line_profiler` instance, and `line_profiler` re-reads the clock inside its per-instance loop: the second instance to be visited stamps its "last line" a few tens of nanoseconds after the first. Which one that is depends on the iteration order of a `set`, so a region's total sits either side of the session's, by roughly **46 ns per line event**. On real work that is invisible; on a three-millisecond synthetic loop it is about a fifth, and it can print a single region at slightly over 100% of the profiled total.

Rather than clamp the number or normalise it away, `print_regions()` states the three reasons its shares do not sum to 100% — nesting, gaps, and independent timing — and points at the µs-per-entry column, which is stable across runs. Hit counts are exact under both engines, and the two engines agree on them.

### Added — the benchmark covers regions

`benchmarks/bench_lineprofiler.py` now reports both region columns. The per-boundary figure profiles an empty directory on purpose, so neither the timing loop nor the region machinery is traced and what is left is the switch itself; the cost of the `with` line you write is one line event, already priced in the other column. An earlier version of this measurement timed its own loop and reported a *negative* cost for the builtin engine, which is how the artefact was caught.

## [0.9.0] - 2026-09-06

### Changed (breaking) — the line profiler is a front end over `line_profiler`

The per-line timing is now done by [`line_profiler`](https://github.com/pyutils/line_profiler)'s C callback, which becomes this package's one runtime dependency. `with profiler:` is unchanged and still names no function: a `sys.monitoring` discovery hook registers each admitted code object the first time it runs, so closures, methods, `runpy` scripts and modules imported inside the block are all found without a decorator, a `kernprof` run or a build step.

Measured over 600,000 line events on one machine (`benchmarks/bench_lineprofiler.py`, new): 239 ns per event through this `with` block against 240 ns calling `line_profiler` directly — the wrapper is free — and 895 ns for the pure-Python engine it replaces. That engine remains as `engine="builtin"` and is selected automatically where `line_profiler` cannot be imported, so nothing loses the ability to profile; `backend=` also selects it. Both engines report the same shape of answer.

Below Python 3.12 there is no discovery hook, so the C engine registers the functions of every in-project module already imported and misses anything imported later. The builtin engine has no such limit.

### Fixed — a line that called a function was billed almost nothing

`b = inner()` reported 2.2 µs for a call that took 9.5 ms. The engine reset its clock when the callee started and never gave the caller its time back on return, so the call line was billed only the interpreter's dispatch. It was also inconsistent: a call into the standard library *was* billed inclusively, because those frames are not traced, so the same syntax meant two different things depending on where the callee lived.

Both engines now bill a call line for the whole call, which is what `line_profiler`, `cProfile` and every other line profiler mean by that column. The rewrite keeps a stack of open frames per thread and closes each one on return, so a nested call's time lands on the line that made it rather than disappearing.

**This changes reported numbers.** A line calling in-project code will now show a larger figure, and the percentages around it shift accordingly. The old numbers were wrong, not merely differently scoped.

### Fixed — threads lost hits, or were not recorded at all

Four threads running the same function recorded 370,794 of 400,000 hits under the `monitoring` backend: one shared "current line" was mutated by every thread. Under `settrace` the same test recorded *nothing* from the worker threads, because `sys.settrace` only affects the thread that installs it and `threading.settrace()` was never called.

The open-frame stack is now thread-local, `threading.settrace()` is installed alongside `sys.settrace()`, and the per-function record is created with `setdefault` so two threads reaching a function at the same moment cannot each build one and lose the loser's hits. Four threads now record exactly 400,000, and there is a test.

### Fixed — a generator's `yield` line absorbed its consumer's time

`PY_YIELD` was not subscribed, so a suspended generator looked like a running one: everything the consumer did between two `next()` calls was billed to the `yield`. A generator yielding three times into a loop that slept 10 ms per item reported 30 ms on the `yield` line. The event is now handled like a return, and `PY_RESUME`/`PY_THROW` reopen the frame.

### Added — `lineprofiler run script.py`

    lineprofiler run train.py --epochs 3 --top 20 --html profile.html

`kernprof` without the decorators: the script runs normally under `runpy`, everything under its project folder is profiled, its own exit status is preserved, and the summary prints even when it raises — which is usually the moment the profile is wanted.

### Added — `.lprof` export, so the run leaves this package

`profiler.dump_stats("run.lprof")` writes `line_profiler`'s own pickle format from either engine. `python -m line_profiler run.lprof` displays it, `LineStats.from_files()` merges several, and anything that reads `kernprof` output reads this.

### Added — `start_profiling(enabled=True)`

The two-line pair only ever profiled when `LINEPROFILER_ENABLED` was truthy, which is right for a call committed to a repository and wrong for a session where the call site *is* the switch — and it contradicted the `with` block a few lines above it in the README, which profiles immediately. `enabled=True` profiles there and then, `enabled=False` never does, and the default still asks the environment.

### Changed — `print_stats` defaults to source order and takes a stream

Every other line profiler prints a function's lines in the order they appear, because the table is read against the code. This one sorted by time, which scrambled the function. `sort_by="time"` still does that on request, and the cross-function ranking is unchanged — ranking is what *that* table is for. Both printers now accept `stream=`, so a report can go into a log or a test instead of stdout.

Also fixed: the `filename not in folde` typo in `print_stats`.

### Known limits

Two functions with identical bytecode whose line numbers overlap cannot be told apart by the C engine's index, which keys on the hash of the bytecode and the line number. The second one found is left unprofiled rather than reported as a blend of the two; `engine="builtin"` keys on the code object and has no such limit. The engine never rewrites the caller's bytecode to work around this, because doing so loses the call that is in flight at that moment — which, for a function discovered at its first call, is every call it would otherwise record.

## [0.8.5] - 2026-09-06

### Documented — a version history in the README

One table, 0.1.0 to here, one line per release. The changelog is 950 lines and answers "what
exactly changed in 0.8.3"; nothing answered "what has this thing become since 0.1.0" without
reading all of it.

## [0.8.4] - 2026-09-05

### Changed (breaking) — `Profiler(run_dir=...)` now records

Passing a run directory turns the profiler on. Before this, `accounting.start(run_dir="profile")` without `LINEPROFILER_PROFILE=1` in the environment ran clean, exited zero and wrote nothing — not the worker file, not even the directory — so every phase the caller had wrapped their training loop in disappeared, and the first sign of it was an empty path much later. Nobody names a run directory and means "write nothing there".

The switch now resolves in this order: an explicit `enabled=`, then an explicit `run_dir=`, then `LINEPROFILER_PROFILE`, then off. `enabled=False` still wins over a run directory, so a launcher can turn a run off without editing the call site. Library code that carries `accounting.phase(...)` calls permanently is unaffected: it passes neither argument, so it still costs a no-op check until the environment says otherwise.

The implication is keyed on the constructor *argument*, never on the resolved directory. An enabled profiler exports `LINEPROFILER_RUN_DIR` to its children, so reading the resolved path would let a `spawn` child switch itself on from an inherited variable — against an explicit `enabled=False` of its own.

This is a behaviour change on a 0.x line, shipped as a patch release because the public surface is unchanged: no name was added, removed or renamed, only the default answer to "is this profiler on". A caller who wants the old shape passes `enabled=False` explicitly.

### Added — a disabled profiler that was used says so when it closes

The other half of the same silence. A profiler can still end up disabled with phases entered on it — `enabled=False` passed deliberately, or an environment variable that did not reach a `forkserver` worker — and the measurement is gone either way. `close()` now warns once, naming the directory that was never written and both remedies (`enabled=True` at the call, or `LINEPROFILER_PROFILE=1` at the launcher), because which one applies depends on whether the caller owns the code or the launcher.

A disabled profiler nobody used closes silently: that is exactly what the no-op path is for. The count is one integer increment on the disabled path, which measures at 450 ns/call against the 1600 ns budget the overhead suite holds it to.

### Added — an empty or missing run directory says which it is

`lineprofiler report` on a directory that did not exist printed `Runtime 0ns   Processes 0   Roles none` — a well-formed report that reads as a successful measurement of a very fast program. `rglob` over a missing directory yields nothing and raises nothing, so nothing downstream could tell the two apart.

`MergedRun` now carries an `empty_reason`, and every renderer states it: the text report and the HTML page say either *"No run directory — nothing was profiled at this path"* or *"No worker files — the directory exists but no profiler wrote to it. The profiler was disabled, or `close()` was never reached."*, and the JSON document grows a `caveats.empty` key. A populated run is untouched, byte for byte — the key appears only when it applies, so a caller comparing `caveats` against a fixed dict keeps working.

`report` also exits **2** for a missing directory, which is what argparse already uses for a usage error, and **0** for one that exists and holds nothing: a run nobody profiled is a legitimate answer and must not fail a pipeline that reports on every run it finds. `1` stays reserved for `trace --fail-over`. The report is written either way. A directory of corrupt worker files is neither case — something was written, and `caveats.unreadable` already says so precisely.

### Added — `write_report()` and `write_trace()`

The path from a run directory to a file someone can open used to run through the command line only, so a training script that had just finished a run could not save its own report without shelling out to itself.

```python
from lineprofiler.accounting import write_report, write_trace

write_report("profile", "reports/run-17.html", format="html")   # text | json | html
write_trace("profile", "reports/run-17-trace.html")             # html | json
```

The formats are exactly the CLI's, and an unknown one raises rather than falling back to text — a typo that silently wrote the wrong format is discovered when someone opens the file. Both create parent directories, matching `write_html()`, and deliberately unlike the CLI, where a path that does not exist is usually a typo worth failing on. Both read the trace sidecars, so a run recorded with `trace=True` renders with the findings, occupancy and lifecycle blocks it was instrumented for.

The trace JSON document moved out of `cli.py` into `findings.py` as `trace_as_dict`, beside the two derivations it serialises, so the CLI and the library emit one document rather than two hand-maintained copies that drift. A test pins them equal.

### Added — `python -m lineprofiler` and `import with_line_profiler`

Two names that did not work and had no reason not to.

`python -m lineprofiler report profile/` now runs the CLI. The console script is not always reachable — an unactivated virtualenv, a `pip install --user` whose scripts directory is off `PATH`, a batch job invoking `python` by absolute path — while the module form works wherever the package is importable, which is where the profiler that wrote the run was. (`python -m lineprofiler.accounting.cli` already worked; only the package form did not.)

`import with_line_profiler` now works too, re-exporting the same five public names plus `__version__` and `accounting`. The distribution installs as `with-line-profiler` and imports as `lineprofiler`; guessing the pip name gave an `ImportError` that named neither the cause nor the fix. `lineprofiler` remains the documented name and nothing new will live in the shim.

### Fixed — a single-lane run's wait is no longer called a stall

With one lane there is no concurrency to have observed, so "no other lane had a phase open during that wait" is not evidence of anything — whatever released the wait lives in a process the run never profiled. The finding said *"nothing was being produced while this blocked — a stall rather than a queue"*, which sent a reader hunting for a hang inside code that was correctly waiting its turn.

A single-lane run now says that it recorded one lane, that nothing here can say what the wait was for, and points at the process on the other side of it. Where two or more lanes were recorded and none was busy, the old wording stands — the trace watched every lane it has. The precedence above this is untouched: a recorded `signal`/`wait_on` arrow still settles the question before either branch is reached, so a single-lane run with an arrow still correctly reports as a queue.

## [0.8.3] - 2026-09-02

### Fixed — the `CPU peak` figure was quantisation noise, not a measurement

`RESOURCES` reported `CPU peak 8.7 cores` on a run whose true concurrent peak was about 1.5, and the identical benchmark reported `1.4`, `8.7` and `19.5` cores across three runs on the same machine with the same workload. It is the figure a reader sizes a job with, it was wrong by 5.8x, and it was non-deterministic between identical runs.

The cause is the baseline row. `ResourceSampler._run` writes it "before any interval elapses", and `_add_process_metrics` reads `psutil.Process.cpu_percent()` on it — which differences against the priming call inside `_detect_capabilities`, a thread start and a file open earlier, on the order of a millisecond. `psutil` then divides a CPU-time delta quantised to the kernel tick by that sub-millisecond wall interval. Below one tick the quotient is not a noisy measurement but a coin flip between `0` and `tick / interval`: measured on a process spinning at exactly one core, with `SC_CLK_TCK` at 100 Hz, a 1.4 ms interval returned `0.0` fifty-one times in sixty and exceeded 200% nine times, peaking at 703%. Those outliers are maxima, so they land squarely on a *peak*.

A reading is now kept only when its interval spans at least four kernel ticks, and is recorded as the existing `-1.0` "not measured" sentinel otherwise. Four ticks bounds the quantisation error at 25%, and from two ticks upward the same measurement was already exact; the floor is derived from `SC_CLK_TCK` rather than hard-coded, so it is 40 ms at 100 Hz and 4 ms on a 1000 Hz kernel. `-1.0` rather than `0.0` because the report already distinguishes them and an unmeasured row recorded as an idle one would drag every mean down instead of leaving a gap.

The reading is still *taken* on the rows whose value is discarded. `cpu_percent()` differences against its own previous call, so skipping the call would fold the sub-tick gap into the following sample and move the corruption one row down rather than remove it.

`RESOURCES` now also prints the CPU skew beside that peak: `heaviest process peaked at 1.06 cores against a 1.21 mean`. `CpuUsage.max_process` was already computed and already carried into `report_as_dict` as `cpu_cores_max_process`, and its own docstring already said that against `peak / processes` it is the skew — only `_resource_notes` never rendered it, while the RAM row beside it has had that line all along. It belongs with the fix above rather than after it: `CPU peak` sums every process at its own peak, so it describes an instant that need never have existed, and the skew is the sentence that says whether one process drove the total or the load was spread. Printing the alarming figure without it is what makes a reader over-size a job.

### Fixed — `sync=True` no longer opens a CUDA context in a process that never uses the GPU

`Profiler` resolved `torch.cuda.synchronize` for every enabled profiler on a box where a device was visible, and `phase(sync=True)` then called it. That call is what creates a process's CUDA primary context: measured on an A100, `import torch` holds no VRAM, `torch.cuda.is_available()` returning `True` holds no VRAM, and the first `synchronize()` holds **414 MiB**. So a CPU-only worker — an actor that holds no model and talks to an inference server over a queue — bought a context it never used, purely by being profiled. On a four-actor debug run that was 1.7 GB of a 40 GB card; the same pipeline's recommended 32-actor profile would be ~13 GB. This is profiling changing the thing it measures, and it scales with worker count.

The obvious caller-side workaround does not work and fails silently: torch caches the visible-device count on the first `is_available()`, so `CUDA_VISIBLE_DEVICES=""` set inside a spawned worker is already too late, and the worker has no way to detect that it failed. A fix that works has to bracket every `Process.start()` in the parent.

`cuda_synchronize()` now returns a callable that drains only while `torch.cuda.is_initialized()`. A process that has not initialised CUDA by the time it opens a `sync=True` phase has submitted no device work, so the drain is a no-op *by definition* — skipping it is free and correct, and it fixes every existing caller with no API change. The check is a flag read, not a driver call, and it is deliberately not cached: a process that initialises CUDA later starts synchronising from that point, and a forked child that inherited the callable without the parent's context gets the right answer. `Profiler(cuda_sync=False)` switches synchronisation off outright for a role known to be CPU-only, and `cuda_sync=True` restores the unconditional drain.

The comment that justified resolving the callable eagerly claimed `torch.cuda.is_available()` initialises the driver. Measured, it does not — the cost sits on the other side of that line, and the comment now says what `is_available()` actually costs.

### Fixed — GPU compute is no longer reported as "blocked", and no longer called a queue

`wait_ns` is `max(0, wall - cpu)`, documented as wall time during which the thread was not executing on a CPU. That is the right definition for a queue `get()` and the wrong one for a phase waiting on the GPU: a thread inside `.backward()` or `cudaStreamSynchronize` has released the GIL and is off-CPU, so legitimate device compute was counted as blocked. The findings block then went further and attributed it: *"`train_step/forward_backward` spent 34% of its time blocked … released by actor on a recorded signal/wait_on pair, so this is a queue, not a hang."* The 34% was the backward pass. The finding's own closing sentence — 100% of the wait was after the producer had already signalled — was the evidence against its conclusion, printed as a supporting detail.

Two defects, fixed separately. A span now records `FLAG_DEVICE_SYNC` when its phase actually drained a CUDA queue, and `_explain_wait` returns a device explanation before it considers any peer: a `sync=True` phase waited on the device by construction, so nothing about that wait is attributable to another process. The headline says "waiting on the device" rather than "blocked" for those phases. The flag is set only when a drain really happened, so a `sync=True` phase in a process with no CUDA context keeps its ordinary wait explanation. It is derived in `_record_span`, which runs only when tracing is on, so the untraced hot path is unchanged.

The report made the same claim in two further places, and both are fixed with it. A role's `while learner waited, no other lane was active — this is a stall, not a queue` line now excludes synchronised phases and the phases that enclose them, and says the off-CPU time was device work instead. The timeline's phase summary marks such a row `‡` with a footnote, because that table is what a reader ranks by before reading the findings above it, and an unmarked `blocked 100%` there reads as a process waiting on a peer.

Separately, `_releasing_role` matched every arrow addressed to the waiting *worker*, whenever it arrived — so one instrumented queue boundary anywhere in a process explained every blocked phase in it, naming a producer that had nothing to do with the wait. Only arrows that landed inside the phase's own spans count now, tested by bisect over a running maximum of the span end times so a phase with six figures of spans stays affordable.

### Added — the VRAM figure that includes what actually scales

The `RESOURCES` block reported `VRAM peak alloc` from `torch.cuda.memory_allocated()` — the caching allocator's view — against `nvmlDeviceGetMemoryInfo().total` as its denominator: an allocator number under a device-total column. On the run above it read `304.8 MB / 40.0 GB` while `nvidia-smi` showed **2,702 MiB** held by the same eight processes, an 8.9x gap that is entirely per-process CUDA context. The one figure a reader uses to answer "can I raise `num_actors`?" was the one that omits the term growing with `num_actors`, and it hid the context leak above completely.

The sampler now also records `cuda_proc_used` per pid via `nvmlDeviceGetComputeRunningProcesses` — the column `nvidia-smi` prints — and the report shows it as a second row, `VRAM peak held`, beside the allocator row rather than instead of it. The allocator figure is genuinely the right answer to "how big are my tensors", and replacing it would silently change what every archived report is being compared against. The HTML report gains the matching tile, and both pages carry one line saying which instrument each figure came from. `report_as_dict` carries it as `vram_held_peak`, `None` rather than `0` when the driver would not say.

`-1` is the unmeasured sentinel and `0` is a real reading — a process holding no VRAM is the finding, not a gap. Two cases are left unmeasured because zero would be a lie: a driver that will not attribute memory per process (some MIG and vGPU configurations report `usedGpuMemory` as `None`), and a pid-namespace mismatch in a container, where NVML reports host pids and `os.getpid()` is namespaced — detected by holding allocator memory while NVML lists no row for us.

Finally the derived hint, which is what turns the two rows into an answer: when a role holds VRAM with no allocator activity at all, the report names it — *"2 actor process(es) hold 414.0 MB of VRAM each with no allocator activity — a CUDA context in a process doing no GPU work"* — and points at `cuda_sync=False`. That is the signature of the first entry above, and it collapses a two-hour investigation into a glance.

### Added — the `async_work` footnote names the measurement that explains a slow submission

The `†` note says a phase's wall time is submission time, not device compute, which is what points a reader at the real bottleneck. It did not say why submitting might be slow. It now names the decisive comparison — a phase whose cost does not scale with its batch counter is launch-bound, so compare its wall time at batch 1 against a large batch — because that measurement is one the profiler cannot run for you but can point at, using counter rows it already prints. The reported case measured flat at 2.13 ms across a 512x batch range at 145 kernels per call, and a captured CUDA graph took it to 0.39 ms for 2.89x end to end.

### Fixed — a wall clock that steps mid-run no longer destroys the whole timeline

`perf_counter_ns` is monotonic; `time.time_ns` is not. An NTP step, a resumed VM or a container clock correction moves the wall clock between two of a worker's clock anchors, and `to_common_epoch` fitted a straight line through the pair — so every span in that bracket was mapped onto an axis dilated by four or five orders of magnitude, and *reversed* when the step went backwards.

Measured on a real 0.25-second run with one backward hour step: a 40 ms interval placed at −1,440 s, every span after the step drawn as `0ns`, a 2 ms span rendered as `1m 11s`, the page's headline reading `traced span 57m 48s`, the lane table reporting `phase open 0.0% / on CPU 0.0%`, and the top-ranked finding claiming the lane *"had no phase open for 100% of the run"* — which `lineprofiler trace --fail-over` would have failed a build on. The run was single-host, so the caveat block asserted the exact opposite of the truth: *"the shared time axis is exact: every process read the same clocks."*

`usable_anchors` now rejects any anchor whose wall-clock elapsed disagrees with the monotonic elapsed by more than a factor of two, measured against the run's first anchor. That band is far wider than the drift the anchors exist to correct — slew is parts per million, and even chrony's aggressive default caps near 8% — and far narrower than any step worth catching, so it rejects steps without rejecting corrections. Verified against a real twelve-flush run: thirteen anchors written, thirteen kept.

The repair is invisible by nature, so it is disclosed rather than assumed. `AlignedTrace.clock_steps` names each affected worker; the timeline states it beside the headline figures, not only in the trailing caveats, because this page has already learned once that a disclosure eighty lines under a confident conclusion does not reach whoever acts on it. `lineprofiler trace --format json` carries `clock_steps` for the same reason — the JSON is the output a machine acts on. Durations are unaffected either way: they are `perf_counter` deltas, and only absolute placement after the step depends on the anchor that was thrown away. The one-host accuracy note stops claiming the axis is "exact" when a clock stepped, since printing that one line under "the wall clock stepped mid-run" reads as an arithmetic bug in the tool.

### Fixed — a finding that compares two hosts now carries its clock caveat

Several findings are claims about the *relative* timing of processes: `only one of 2 lanes was active for 52% of the run`, a lane idle while another worked, a phase blocked across lanes. Within a host that relation is exact — every process reads the same two clocks. Across hosts it is only as good as NTP, and `alignment_accuracy_note` has always said so.

It said so in `Caveats`, at the foot of the page, about 7,800 characters below the ranked findings that rest on it. The page is deliberately ordered conclusions-first so a reader can stop after the findings; doing that meant never seeing the one sentence that qualifies them. This is the same defect as the superseded-worker disclosure that used to sit below a header claiming `Processes 1` — the caveat was present, correct, and in the wrong place.

The findings block now carries one sentence on a run spanning more than one host, pointing at `Caveats` for the full statement. Single-host runs — most runs — are unchanged, so the note does not become furniture readers learn to skip.

### Fixed — concurrent asyncio tasks are no longer silently recorded as nesting

Phase statistics are per *thread*, which is what lets the hot path take no locks. Asyncio tasks share a thread, so a phase held across an `await` while another task enters the same phase goes onto the stack twice and is stored as its own child. Eight concurrent requests produced `handle_request/handle_request/…` eight levels deep, every level claiming the full duration, and the outermost row reporting `entries 1` for eight requests served. At sixty-four they reached `MAX_DEPTH` and folded, which records nothing at all — sixty-three entries survived for sixty-four requests.

That is the failure mode this layer exists to prevent: not a crash, but a complete-looking report that answers the throughput question wrong. It matters because an async inference server is the standard shape for batched policy inference in an RL pipeline.

Attributing per task needs a per-task stack and a `contextvars` lookup on every phase entry, which a ~3 µs hot path cannot afford. So this reports rather than repairs. The profiler raises a `RuntimeWarning` naming the phase, the cause and the two workarounds — put the phase around a region that does not `await`, or run one task per thread — and the report header carries the same caveat for whoever opens the file later, since a warning printed to a terminal is gone by the time a profile is read. Detection sits in `_admit`, which runs once per distinct phase path and never on the hot path, and is gated on a phase repeating in its own path *and* a running event loop: sequential `async` code that nests different phases is measured correctly and stays quiet, as does ordinary threaded nesting.

### Fixed — the advertised phase overhead now describes the default configuration

The README's headline quoted **~2 µs per phase**. The default is `measure_cpu=True` — it has to be, since `wait%` is derived from it — and that measures **4.16 µs**, so the advertised figure was about half the cost a first user actually pays. The README contradicted itself twenty lines later (5.4 µs for the same call), and `docs/accounting-recipes.md` derived its "a phase is affordable above ~200 µs" rule from the non-default row while its own table listed 3909 ns for the default.

Overhead figures are load-bearing in this package — the pitch is that it is cheap enough to leave enabled for twelve hours, and a reader deciding where to put a `with` block does that arithmetic with the headline number. Both places now quote the default, name it, and give the lever (`measure_cpu=False` roughly halves it at the price of the `wait%` column). The affordability threshold moves from ~200 µs to ~400 µs.

### Fixed — one worker file with a broken clock no longer costs the entire report

`_read_worker` guards everything after the JSON parse precisely so that an unusable file is a lost worker rather than a lost run. `float("inf")` and `float("nan")` defeated that guard by satisfying it: they *are* floats, so every cast and every `except (TypeError, ValueError)` passed them through. The value then reached `format_ns`, whose `int()` raised `ValueError` from the first line of the header — so a report over sixty-four workers was lost entirely because one of them wrote a bad timestamp.

This is reachable rather than hypothetical. `json.dumps` writes `Infinity` and `NaN` by default and `json.loads` reads them back, so the profiler's own snapshot writer round-trips a non-finite clock reading silently. The same path delivers a *negative* runtime — a wall clock stepped backwards mid-run by an NTP correction on a long job — which printed as `Runtime -1000000000000ns`, a duration that cannot exist.

`format_ns` now returns `n/a` for any non-finite or negative input. It is the single chokepoint every duration on every page passes through, so the text report, the HTML report, the comparison table and the timeline are all covered by the one change. The sentinel is deliberately not `?`, which the header already uses for an unknown host and `compare` uses for a thin sample: `Runtime ?` directly above `Host ?` reads as one kind of gap in two places.

Because `n/a` says a figure is missing without saying why — and the reader needs that to know how much of the rest to trust — the header now carries a line naming the cause and stating what is unaffected: phase totals are `perf_counter` deltas and never come from the wall clock, so every number below the header is still a measurement. It is silent on a healthy run.

### Fixed — a phase named from data no longer stretches every page it appears on

A phase name is user data, and one built from a value — a file path, a URL, a serialised config — has no length bound. The text report's pipeline breakdown was the single label not passed through `format_label`: a bare `{name:<28}` pads a short name but does not bound a long one, so a 10,000-character name printed a 10,022-character line and pushed its own percentage and wall time into a column nobody scans. The two HTML pages had no bound at all and repeated the name once per table cell and per SVG tooltip.

Labels are now bounded at two chokepoints — `format_label` for text, the new `htmldoc.clip_label` (90 columns) for HTML — both keeping the tail, because the leaf is what a reader greps for, and both marking the cut, because an unmarked truncation prints a name that does not exist. The complete name stays in the embedded JSON block the page is drawn from. On a run carrying one such name the HTML report fell from 39,587 to 19,724 bytes and the timeline from 84,384 to 54,899.

A measured *count* is still never truncated, however wide: an overflowing field pushes the row right rather than printing a wrong number. Only the derived per-second rate is compacted — `format_rate` switches to exponent form above 1e15, so a counter of 2^70 on a microsecond phase no longer claims `61,202,261,312,462,999,586,340,864.0/s`, thirty digits quoted to the tenth from a single entry.

### Fixed — a control character in a phase name no longer breaks a row apart

A name of `a\nb\rc\x00d` rendered as four lines in the text report, three of which carried no numbers and so read as three phases that do not exist, and reached a `<td>` in the timeline page intact, carrying a NUL into a file people open in a browser. This is not an escaping hole and escaping does not address it: these characters are legal in HTML and in a terminal, and destroy the layout anyway.

C0 and C1 control characters plus U+2028/U+2029 are now replaced with U+FFFD in `report._printable` and in `htmldoc.escape` — the two functions every label already passes through. Replacing rather than dropping keeps two names differing only by a control character distinguishable on the page.

### Fixed — a phase blocked on several lanes no longer costs more than 100% of the run

A finding read `mcts/ipc/queue_wait spent 100% of its time blocked, costing 155% of the run`, and the timeline's phase summary showed `share of run 183.6%` beside it. Both summed a figure across every lane the phase ran on and then divided by the run's wall clock, which is one lane's worth of time: two actors each blocked for a whole run genuinely wait two lane-seconds per wall second. The arithmetic was right and the denominator was not.

A percentage over 100% reads as a bug rather than as a sum, and it appeared in the *first sentence of the first finding* — the one place on the page where losing the reader's trust costs the most, because every correct number below it then looks equally suspect.

Both now divide by the lane time the phase could have occupied (`traced span × lanes`), and both say so: the finding appends `across 2 lanes` once more than one is involved, and the summary's note names the denominator and explains why the share column does not fall strictly (rows are ordered by summed wall time, so a phase on two lanes outranks a longer one on a single lane). `Finding.cost_pct` is what the ranking sorts on, so findings that span different numbers of workers are now comparable — with `wait_for_batch` on one lane correctly outranking `queue_wait` on two, which the old denominator had inverted.

### Fixed — losing most of the workers is said beside the process count, not only at the foot

Four workers that each construct their own `Profiler` without inheriting `LINEPROFILER_RUN_ID` get one attempt id each, so `_split_by_attempt` reads a healthy four-way job as four competing attempts, keeps one and discards three. `CAVEATS` declared this correctly — but it prints below the findings, the role blocks and every total, all of which were computed from the surviving quarter under a header reading `Processes 1`. A reader who stops at the conclusions never learns they are conclusions about one worker.

The header now carries a `WARNING` whenever the excluded files outnumber the kept ones, naming the likely cause and the one-line fix (pass the same `run_id=`, or let children inherit it from the parent). An ordinary rerun into the same directory — one superseded attempt against one kept worker — is what the mechanism exists for and stays silent, so the line does not become noise.

### Added — the device strip says what it is and what it cannot attribute

The GPU utilisation strip was drawn under the lanes with no label, legend or note of any kind, directly beneath rows of span-resolution bars. Nothing said it is a 1 Hz whole-device reading covering every process on the device, and nothing said that no reading is tied to the call that launched the kernel — so the natural reading of a busy strip beside a busy lane is that one caused the other, which is the same wrong conclusion as the async-submission trap, arrived at from the other direction.

A note now sits with the strip stating the source, the resolution, that it covers every process, and that it must not be used to attribute device time to a phase. It is driven by the same payload the script draws from, so it cannot describe a strip that is not painted, and it is absent entirely on a run with no device.

### Documented — the measurement floor, and the levers when a phase costs too much

Two additions to `docs/accounting-recipes.md`, both measured rather than estimated:

- **What a phase bills *into* its own reported wall time**, as distinct from the ~2.6 µs it takes from the surrounding program that the overhead table already covers. `__enter__` reads its clock last and `__exit__` reads its first, so what remains inside is the interpreter's dispatch between them: ~210 ns at `measure_cpu=False`, ~240 ns with it on, ~250 ns tracing. That is within ~25 ns of a bare context manager that measures nothing, so it is close to irreducible in Python — but it is a per-entry inflation that does not shrink with the phase, which matters when summing very many short phases or differencing two nested ones. The report does not subtract it, because the correction would be an estimate standing in for a measurement. `benchmarks/bench_accounting.py` now prints it, so the documented figures cannot rot.
- **The levers, in the order worth trying**, since every row of the overhead table is dominated by Python's own dispatch and no setting makes that cheaper. `measure_cpu=False` is the largest at ~1.6 µs of a ~4.2 µs phase (38%), against the cost of losing `wait%`; then sampling (~3.5x, not the sampling rate); then moving the phase out of the loop; then leaving `trace` off. `sync=True` and `async_work=True` are named as *not* levers — the first costs GPU time rather than profiler time, the second is one bool test.

## [0.8.2] - 2026-08-20

### Fixed — `nvml_module()` no longer reports a GPU that isn't there

A runner where the NVML driver library initialises cleanly but enumerates zero devices — the
state on plain CPU CI hosts — used to make `nvml_module()` return the module anyway, since
`_initialise_nvml` only checked whether `pynvml.nvmlInit()` raised. That flipped
`SamplerCapabilities.gpu_util` to `True` with no device behind it, and `test_gpu_hardware.py`'s
`requires_nvml` skip (gated on `nvml_module() is None`) stopped skipping, so its three tests ran
against zero devices instead of being excluded: a bare `nvmlDeviceGetHandleByIndex(0)` failing,
a `ZeroDivisionError` averaging an empty device list, and a missing `GPU 0` block in the
rendered report. `_initialise_nvml` now also checks `nvmlDeviceGetCount() > 0` and degrades to
`None` when it isn't, matching every other capability in this module.

### Added — the report shows how many times each phase ran

`DOMINANT PHASES` gained an **`entries`** column. The number was always recorded — every phase
entry increments `PhaseStats.calls` on the way out — but the text report printed it for exactly
one phase per role, in the `ITERATIONS (n entries)` block. Everywhere else a reader who wanted
it had to add `count("nodes_created")` and pay 384 ns per entry to be told something the layer
already knew.

That makes a whole class of counter deletable: the ones whose amount is always 1. The rule is
now **write a `count()` only when the amount varies** — `count("children_scored",
len(node.children))` still earns its place, because no inspection of the code can supply
`len()`, and its `567 ns/ea` and `always 8` rows are what a bare entry count cannot give you.

Two details on the way:

- **The `entries` column comes out of the label field, which is now 23 columns wide** (the row
  grew from 68 to 70). Twenty-two characters is what an ordinary `parent/leaf` pair costs, so
  `iteration/train_step` and `select/score_children` print in full where a tighter field cut
  their heads off — and a truncated label is the one column a reader cannot reconstruct from
  the others.
- **A `~`/`†` mark no longer widens its row.** The label field is reduced by the mark width,
  but `_label` truncated to a fixed width regardless, so a row carrying both marks printed one
  column wider than its neighbours. Only visible on a label long enough to be truncated, which
  is why it survived this long.

### Added — auto-traced spans say which file, function and line they came from

`trace="auto"` already produced a timeline from uninstrumented code, but every span was a bare
name: the page could say `simulator/step` cost 71% of the run and not say where `step` lives.
The reader had to go and find it, which for a qualname appearing in several modules is a
search, not a lookup.

Spans derived from function calls now carry an `Origin` — file, qualified name and the
function's first line — surfaced in four places, ordered by how a reader meets them: the
**phase summary** and **critical path** tables annotate each row with `envs/simulator.py:9`,
the **hover tooltip** adds the function and location, and the **pinned detail panel** carries
the full absolute path, which is what gets pasted into an editor.

Three properties are load-bearing:

- **Origins are interned beside the phase path, not stored per span.** A location is a property
  of the code object, so a function called a million times records it once. The hot path got
  *cheaper*, not more expensive: `_on_start` now looks the phase id up by code object instead
  of rebuilding a path tuple per call.
- **A named phase has no origin, and the absence is preserved.** There is no code object behind
  a name, so those spans carry `None` and render without a location rather than with an
  invented one. Where one phase path spans several definitions, the summary row says nothing
  instead of naming whichever file it saw first.
- **The page states that a location is the `def` line, not the line that blocked.** A span
  covers a whole call; the line that actually spent the time is not recorded, and the caveat
  block says so rather than letting a definition line be read as a measurement.

### Added — the trace timeline says what is wrong, instead of only showing it

The timeline page opened with a lane table and a canvas: all evidence, no conclusion. It
required the reader to already know what a healthy run looks like, and it never said what
clicking anything would do — clicking a span drew an outline and nothing else, while the
module docstring claimed it would "trace what it was waiting for".

Three changes, in the order a reader meets them:

- **A `Findings` section leads the page**, ranking bottlenecks by the share of the run each one
  cost and stating each in a sentence: which phase blocked, what it cost, and whether it was a
  queue or a stall. Findings that name something drawable carry a *show on timeline* button
  that zooms the chart to it.
- **A `Phase summary` table** — Vampir's Function Summary — ranks every phase by total wall
  time across all lanes, with a `self` column excluding nested phases so a wrapper cannot
  out-rank its callee. A phase spread over ten thousand short calls is invisible on a timeline
  and top of this table.
- **The chart explains itself**: a legend above it names all five drawing conventions, the
  controls say what they do, and clicking a span now pins it and reports who released it, how
  much of it was blocked, and what every other lane was doing meanwhile.

New module `lineprofiler/accounting/findings.py` holds the derivations, so the text report,
the page and a CI gate can reach the same conclusions from the same numbers, and a finding can
be tested without parsing HTML.

Two judgements worth recording, both of which produced a *wrong* conclusion before they were
fixed:

- **A matched `signal`/`wait_on` pair outranks concurrency when classifying a wait.** The
  learner's `queue_get` was released by an actor's signal on every iteration and was still
  reported as a stall, because the actors' short bursts covered only a quarter of the wait's
  union. Recorded evidence names the producer; concurrency only infers one, and the fallback
  now says so in its own wording.
- **A parent that only waits inside a blocking child is not reported separately.** It inherits
  all of the child's wait, so `iteration` at 55% blocked and `iteration/queue_get` at 54% were
  one fact wearing two hats — and the duplicate pushed the genuine second finding off the list.

`_self_times` replaced a pairwise containment search that was O(spans²); at the 120k-span
drawing cap that column alone was billions of comparisons.

### Added — findings reach the terminal, CI, and the keyboard

The findings above are a derivation, not a rendering, so they now surface everywhere the run
does:

- **The text report leads with them.** `lineprofiler report <dir>` prints the same ranked
  block above `RESOURCES`. Silent on a run with no trace — findings come from spans, and a
  phase tree alone cannot say who waited for whom.
- **Both JSON documents carry them.** `report --format json` gains a `findings` key;
  `trace --format json` gains `findings` and a full `phases` summary.
- **`trace --fail-over PCT` exits non-zero** when a finding costs more than `PCT`% of the
  traced span, turning the timeline into a regression gate. Inert until a threshold is set, and
  the page is still written on the run it failed.

The timeline itself gained the three things that made it hard to use on a real run:

- **Keyboard navigation.** `← →` pan, `+ −` zoom, `n p` step along the critical path,
  `0` reset, `Esc` unpin. The canvas is given a `tabIndex`, without which a key handler is
  attached to an element that can never receive a key event — a feature that fails silently.
- **Foldable lanes.** Clicking a lane's label folds it to one row. With sixteen actors the
  chart is taller than any screen, so the learner and an actor cannot be seen at once — usually
  the exact comparison the page was opened to make. A folded lane keeps its spans; they stop
  claiming a row each, and nothing is hidden.
- **Causal-chain highlighting.** Pinning a span now dims everything not causally upstream of
  it and walks the arrows back through the producers that released it — what the module
  docstring had promised all along while the click only drew an outline. The walk is bounded by
  a visited set and a guard, because two workers each waiting on the other produce a genuine
  cycle, and hanging the browser is worse than any wrong picture.

## [0.8.0]

### Added — every report opens with what the run used and what the machine had

A `RESOURCES` section is now the first thing in the text, HTML and JSON reports. It states CPU,
RAM and VRAM consumption as a run total, as a per-process figure, and against the capacity of
the machines that ran it.

The gap it closes is comparison. A report said `Runtime 1m 30s   Processes 16` and nothing about
the hardware underneath, so two profiles of the same workload — one on a 128-core node, one on a
laptop, or the same node at two worker counts — could not be told apart by reading them. Every
timing difference was equally attributable to the code, the machine or the dataset, and the
reader had to supply the missing half from memory.

Three things are new underneath:

- **CPU is now sampled.** `Sample.cpu_percent` joins the 1 Hz row, converted to core-equivalents
  at analysis time. It follows the `gpu_util` convention exactly: `-1.0` means *not measured* and
  `0.0` means *measured, idle*, because a run that reported no CPU and a run that used none must
  not render identically.
- **`hardware.py` records what the box has** — physical cores, SMT siblings, the affinity mask,
  total RAM, and each device's model and VRAM. Modelled on `provenance.py`: resolved once,
  degraded to `{}` on any failure, and never able to break a run. It is cached per process, so a
  `spawn` run pays roughly 60 ms once rather than per worker, and a forked child inherits the
  dict rather than re-entering NVML.
- **Capacity is stored per worker, not per run.** `metadata.json` is written by whichever rank
  wins the race, so recording it there would describe one node of a heterogeneous job and
  silently attribute its capacity to every other. Each worker carries its own machine's
  inventory, and the report prints one line per host.

The affinity mask is reported next to the core count whenever the two differ, which is the
ordinary case under Slurm or in a container: a node with 128 cores that gave this job 60 has
neither figure alone as its honest denominator.

Per-process figures come with the heaviest single process beside the mean. That gap is the whole
scaling signal — flat per-process RSS across two worker counts says memory scales linearly; a max
well above the mean says one worker will hit the ceiling first.

`report_as_dict` gains a `machine` key (`used` plus `capacity_by_host`). The existing `resources`
key is untouched — it holds the io/memory/gpu breakdown and is already published.

Absent stays absent throughout. A resource with no reading omits its row, a resource with no
capacity leaves that cell blank, a utilisation percentage is suppressed rather than divided by a
missing field, and a run recorded before this release renders without a capacity column and says
why.

## [0.7.0]

Eight gaps found by a real investigation — "MuZero actors spend 96% of self-play blocked, on
what?" — where this layer held the data but not the affordance, and twice where it produced a
confidently wrong reading. The through-line is the failure the `sample_stride` docstring names:
*an estimate that cannot be told apart from a measurement*. Most of what follows is that same
failure one level up — a measurement that is correct but unlabelled, so the reader cannot tell
which question it answers.

### Added — a phase can say its device work was never awaited

`phase(name, async_work=True)` declares that a phase submits work it does not wait for: CUDA
kernels, a device queue, `io_uring`, a background executor. The phase is measured exactly as
before; what changes is that the report marks the row `†` and names the remedy.

The reading this prevents is expensive. CUDA launches are asynchronous, so a phase around an
unsynchronised forward pass reports the time to *enqueue* — which looks exactly like the cost
of running, and differs by orders of magnitude. In the run this came from, `forward` held 1m 31s
at a plausible 6.3 ms p50 while the GPU sat at 7% utilisation. Every number was right; the
implication was not.

`sync=True` wins over `async_work=True` — that phase did wait for its work — so flipping one to
the other turns a submission time into a device time and the mark disappears. Costs one bool
test per phase, so it belongs on the phase in the inner loop. Deliberately **not** inferred from
`sync=False`: that is the default, so inferring it would mark every phase of every run,
including every phase of a CPU-only one, and the mark would distinguish nothing.

### Added — `GPU BY PHASE`, and the GPU block no longer goes silent

Device samples are bucketed by the phase open when each was taken, with utilisation quantiles
and per-phase VRAM peaks. `Sample` already carried the phase; the join simply had nowhere to
land, so "what was the GPU doing *during* `forward`?" meant intersecting the phase table with
the timeline's `gpu.points` series by hand, on two different timebases. That join is what
refutes the paragraph above, which makes it load-bearing rather than cosmetic.

The text and HTML reports gated their GPU blocks differently, so a run with CUDA memory but no
NVML showed a block in one and nothing in the other. Both now share `SampleAnalysis.has_gpu`,
and a run with samples but no GPU data says so in one line instead of rendering nothing —
silence reads as "no GPU involved", which is precisely how an idle device stays invisible.

### Added — request lifecycles: `trace_begin` / `trace_mark` / `trace_end`

One key, several named checkpoints, stamped in the processes that own them, decomposed offline
into named segments:

```
REQUEST LIFECYCLE
inference                        422.8ms  (6 req)
    ├─ begin → admitted             301.0ms    71%    50.2ms/ea
    ├─ admitted → computed          120.8ms    29%    20.1ms/ea
    └─ computed → end               931.3us     0%   155.2us/ea
```

`signal()`/`wait_on()` could not do this and never could: the producer signals at *response*
time, so an arrow spans only the last of the four intervals a queue wait actually contains.
Measured on the run this came from, every arrow together covered **3.6%** of the wait. The four
intervals have opposite remedies — batch harder, shrink the batching window, cheaper model,
fewer hops — so the fused total is correct and unactionable.

Marks reuse the link ring, its drop policy and its clock alignment. `sample=` selects by key
hash rather than a counter, so every checkpoint of one request is kept or dropped together
across processes without shared state; a counter would keep a request's `admitted` mark on the
server while dropping its `begin` on the client, yielding segments no request experienced.
Incomplete lifecycles contribute nothing, and checkpoints that arrive out of order — ordinary
cross-host skew — are dropped rather than counted as negative time.

### Added — source provenance in the run metadata and every report header

```
Source c49ce841 (+dirty: 26 files, diff sha 3f9a1c)
```

A trace recorded the environment thoroughly and said nothing about the code. That silently
invalidates analysis: the investigation this came from drew a conclusion from a profile of the
*committed* code while the working tree had already fixed the constraint the profile found — a
claim about a program that no longer existed.

One `git` call at startup, on the one rank that writes the metadata, after the dedupe. Empty on
any failure: not a repository, no `git`, a timeout. Provenance is a courtesy the report prints
when it can, never a reason a run fails or stalls. `Profiler(source={...})` overrides it, which
is also where a config hash belongs.

### Added — counter spread, so a cap is distinguishable from a burst

`counters` was a running sum, which yields a mean and nothing else. For a batching server the
mean is the least interesting statistic: "1.9 rows per forward against a cap of 2" is equally
consistent with *always exactly 2* (the cap binds — raise it) and *usually 1, occasionally 8*
(the supply is not there — batch harder). `counter_min`/`counter_max` render as `always 2` or
`1..9`, and the two readings stop looking alike.

The cheap half of the proposal deliberately: per-instance histograms would add 512 buckets per
counter per phase against the existing `MAX_PHASES` budget, and `min == max` is the whole
finding.

### Added — "is this a hang, or is it queueing?"

`overlap_ns(a, b)` intersects two interval lists — the primitive that question needs, and the
one that had to be reimplemented by hand against JSON extracted from an 11.7 MB page. Exported
publicly. `concurrent_activity` builds on it, and the report answers per role:

```
  while actor waited, concurrently active: server 88%, learner 3%
```

Work elsewhere means queueing; silence everywhere means a stall, and the report says which.
The timeline page gains a **select a range** mode: brush a wait and every other lane's busy
share for that window is summarised beneath the chart.

### Added — `busy` / `working` in the text report, and named denominators

`busy 97% / working 35%` *is* the bottleneck statement for an actor, and it was HTML-only.
Both terms are now defined in the report where they appear.

The role block stated percentages without stating what they were *of*. Four denominators were
plausible and they read materially differently, so the base is named outright, and columns that
sum across processes are marked `(Σ2 proc)` — a total exceeding the run's own runtime is correct
for a multi-process role and reads as an error until the summing is stated.

### Changed — the trace render degrades instead of failing

A 26-process render hit a 900 s timeout and produced nothing, *after* the profiled run had
succeeded: the expensive half of the work discarded to save the cheap half. `lineprofiler trace`
now takes `--max-spans N`, prints coarse progress to stderr (`-q` silences it), and estimates
up front when a run exceeds the cap.

The span cap already existed and already counted what it dropped — but the count reached
neither the page nor the JS, despite a docstring promising "never hidden". It now appears in the
caveats. The timeline's `duration_us` is relabelled **traced span**, with a note: first span to
last is legitimately shorter than the run's wall clock, and the unexplained difference cost a
reader time.

### Fixed — link overflow was O(n) from the front

`record_link` evicted with `list.pop(0)`, shifting every surviving link. Tolerable at a handful
of links per iteration, not at several lifecycle marks per request. Now a `deque` with `maxlen`:
O(1), keeping the newest, exactly as the span ring does.

### Fixed — sampled memory with no phase open was billed to `(root)`

The byte path called this `(no phase open)` and explained why: it is an admission that the
sample rate was too coarse, not a finding about the root. The memory path used `(root)` and
therefore read as the latter. Both now agree.

### Notes

`PhaseStats` gains `async_entries`, `counter_min` and `counter_max`; `SampleAnalysis` gains
`gpu_by_phase`. All read through defaults, so a 0.6.0 worker file still parses — a file written
before these existed declared nothing async and recorded no extremes, which is what the defaults
say. The hot path is unchanged: `phase(async_work=True)` measures 4227 ns/call against 4224 for
a plain `measure_cpu=True` phase, and `test_overhead.py` bounds it as a ratio.

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
