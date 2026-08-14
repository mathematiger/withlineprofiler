# Integration feedback — `with-line-profiler` 0.3.0

From integrating the accounting layer into a MuZero/AlphaZero training pipeline for power-grid
topology control: 16 actor processes + a learner thread + an evaluator process + an inference
server + a reanalyse worker, `spawn` start method, Slurm, runs measured in hours. The integration
replaced five hand-rolled timing accumulators and net *removed* code.

It went well. Everything below is what I hit on the way, ordered by how much it cost me.

---

## Two report bugs — real-world repros, both already fixed in the working tree

Both of these bit me on the **released 0.3.0 wheel**, which is what `pip`/`poetry` resolves and
therefore what my project consumes. Both already have a fix in this repo's uncommitted working
tree (`_same_layer_share` in `analysis.py`, the `"…" + text[-(width - 1):]` elide in `report.py`).
So this section is not a bug report so much as independent confirmation, with concrete output —
and a request to **cut a release**, because the wheel is what downstream sees.

### 1. `unattributed_read_share` divides page-cache chars by disk bytes

`analysis.py:113-125`:

```python
return _share(
    loose.read_bytes or loose.read_chars,
    self.totals.read_bytes or self.totals.read_chars,
)
```

The two `or` fallbacks resolve independently. When the unattributed traffic is entirely page-cache
(`read_bytes == 0`, so the numerator falls through to `read_chars`) but the run's totals *do* have
disk reads (`read_bytes` truthy, so the denominator stays at the byte layer), the ratio mixes the
two layers. My run printed:

```
  13845% of reads and 0% of writes moved while no phase was
   open — too coarse to attribute. Wrap those regions in io=True.
```

110.3 MB of cache reads over 816.0 KB of disk reads. Both numbers are correct and both are printed
correctly three lines above; only the share is wrong. A percentage over 100 is at least obviously
broken — but the same expression yields a *plausible* wrong number whenever both layers are
non-zero, which is the common case. Pick one layer for both operands (chars is the honest one,
since that is what the process actually asked for) and the note text should name which.

`unattributed_write_share` has the same shape. (Working tree: fixed — both operands now come from
the same layer via `_same_layer_share`. Worth keeping the note text naming *which* layer it
reports, since the two answer different questions.)

### 2. Phase labels are left-truncated into names that do not exist

`report.py:242` and `:298` use `phase[-26:]`, so a 27-character path loses its first character
and prints as a name the reader can neither find nor grep:

```
  rain_step/forward_backwardr        0 B   w     1.3 MB  cache 4.4 KB
```

That is `train_step/forward_backward`. Two problems: the truncation is unmarked, so it reads as a
real phase called `rain_step`; and at exactly 26+ characters the label consumes the column gap and
runs into the `r` of the read column. Keeping the tail of a path is the right call — the leaf is
the informative end — but it needs an ellipsis and a guaranteed separator: `…ep/forward_backward `.
(Working tree: fixed, `"…" + text[-(width - 1):]`. Check the column gap too — in my output the
overflowing label also swallowed the space before the `r` of the read column, which the ellipsis
alone does not restore.)

---

### 3. `close()` leaves its signal and fork hooks installed — this one cost me real time

Not cosmetic, and I do not think it is fixed in the working tree. `_install_exit_hooks` installs a
`SIGTERM`/`SIGUSR1`/`SIGHUP` handler and `os.register_at_fork(...)` callbacks. `close()` writes the
final snapshot, stops the threads and drops the run-dir registration — but it removes neither. So a
process that constructs and closes a profiler is permanently altered:

```python
>>> check("before profiler")
before profiler:      exitcode=-15 alive=False SIGTERM handler=0
>>> prof = Profiler(..., enabled=True); prof.close()
>>> check("after profiler close")
after profiler close: exitcode=-15 alive=False SIGTERM handler=<function Profiler._chain_signal.<locals>.handler>
```

The handler chains correctly, so the minimal case still dies with the right status — but the hooks
now belong to the interpreter forever, and the `register_at_fork` callbacks *cannot* be
unregistered at all (CPython offers no API), so every subsequent `fork()` in that process runs
`_reinitialise_after_fork` against a closed profiler.

What that cost me: two unit tests that constructed an enabled profiler in-process (to assert on
`merged_tree()`) left the pytest interpreter unable to terminate its own forked children. Two
unrelated tests in a different file started failing — `proc.terminate(); proc.join(); assert not
alive` — and running those two files together **hung outright**. The failures pointed at
`multiprocessing`, several hundred tests away from the profiler, and reproduced only in the full
suite. I lost a while to it.

Concretely:
- `close()` should restore the previous signal disposition it saved in `_chain_signal`, and
  `atexit.unregister(self.close)`.
- The fork callbacks should no-op when `self._closed` — they cannot be removed, so they must
  become inert.
- Worth documenting either way: **an enabled `Profiler` changes process-global state that outlives
  it**, so embedding one inside a test (or any host process that forks) needs care. My fix was to
  stop constructing one in-process entirely and assert against a subprocess run instead — which is
  the better test anyway, but I would rather have learned it from a docstring than from a hang.

---

## Integration friction

**4. No ambient profiler — this was by far the biggest cost.** `phase()` is an instance method, so
instrumenting `uct_search` means threading a `profiler` argument through every caller between it
and wherever the object was constructed: the search, the episode loop, the actor session, the
inference server. I wrote a 70-line process-global shim instead, and I think every real
integration will write the same one. Consider shipping it: `Profiler(..., install=True)` plus
module-level `lineprofiler.accounting.phase()/count()/current()` that resolve the installed
instance and no-op when there is none. That single change would have made my diff about a third
smaller and removed the only piece of this integration that is mine to maintain.

**4. No live-export hook.** Every user in your stated audience — multi-hour RL training — already
has W&B or TensorBoard, and wants the breakdown *during* the run, not only after it. `merged_tree()`
is cumulative, so each of us re-implements the same delta cache against the previous read. Ship
`Profiler.deltas()` or `on_snapshot(callback)`. (My version is ~25 lines and had a bug on the first
pass: I divided `wait_ns` by `self_ns`, which reports >100% for any parent that waits inside a
child, because `wait_ns` spans the whole phase. Worth documenting that pairing explicitly — `wait_ns`
goes with `wall_ns`, never with `self_ns`.)

**5. `close()` under `os._exit()` is the failure mode you should document next to the signals.**
The README covers `atexit` and `SIGTERM`/`SIGUSR1`/`SIGHUP`. It does not mention `os._exit`, which
skips `atexit` entirely and is the *normal* way a multiprocessing entrypoint tears down — this repo
uses it in two places, including the main training entrypoint. I only got a final snapshot because
I called `close()` explicitly in the orchestrator's cleanup path. One sentence in the signals
paragraph would save the next person a silently truncated run.

**6. The testing story is the best thing here and it is undocumented.** `merge_run(run_dir,
with_samples=False)` turns a run into a machine-readable record of what actually executed — which
roles started, which phases ran, how much work each did — which makes it an *assertion target*,
not just a report. I wrote nine tests against `run.roles`, `run.tree`, `run.workers[].written_at`
and `run.unreadable`; they are one line each, and four of them correspond to real incidents in this
repo that the existing suite could not see, because no other artifact records cross-process
behaviour:

- an evaluator process that never spawned, while the supervisor logged "Evaluator ✓" (it was alive)
- a restarted actor that silently ran a different environment
- a 12-hour async run wedged with a dead collector, producing no output at all
- diagnostics behind a `getattr` default that produced nothing while every test stayed green

The last two fall straight out of `written_at` staleness and `unreadable`. A "Using it in tests"
section would sell the package better than anything in the current README, and a couple of helpers
(`assert_roles`, `assert_no_stale_workers`) or a pytest fixture would make it a two-line adoption.

The one thing that made it awkward: **`report` has no `--json`** (`compare` does). I went to the
Python API instead, which was fine and arguably better, but it means the CLI cannot be used to gate
CI or diff sweep arms without re-implementing the derivations.

**7. `role` is per-process, but my most interesting split is per-thread.** My learner and my data
collector are two threads in one process doing completely unrelated work — gradient steps versus
draining a queue into a replay buffer — and the report can only call both "learner". Since phase
*stats* are already thread-local, a per-thread role, or just the thread name on top-level phases,
would come nearly free and would answer "which of my two threads is the 35% wait?".

**8. The default `run_dir="profile"` is relative to CWD.** On a batch system with per-worker CWD
that scatters one run across directories. This repo has been burned badly by exactly this class of
bug — 81 GB of cwd-relative output written into the source tree before anyone noticed. Prefer
requiring an explicit `run_dir`, or defaulting to `$SLURM_SUBMIT_DIR` when it is set.

**9. Naming.** The pip package is `with-line-profiler`, the module is `lineprofiler`, and it holds
two unrelated tools: `LineProfiler` (tracing) and `accounting.Profiler`. This repo imports both,
and `from lineprofiler import LineProfiler` gives a reader no signal about which layer is meant.
Align the package and module names, and lead the README with the accounting layer — it is the one
described as production-grade, and the one people will keep.

**10. "Do not put `phase()` in an inner loop" collides with the breakdown an MCTS user wants.**
Select/expand/backup per simulation is *precisely* the split that says where a slow search went,
and `count()` cannot answer it — counters give rates, not attribution. At 250 simulations per
search and ~2 µs per phase, three phases per simulation is 1.5 ms against a 2.4 s search, so it
was fine here; but the README's advice told me not to do the thing that turned out to be the most
useful measurement in the integration. A sampled phase (`phase(..., sample=0.01)`) would resolve
the tension properly, and in the meantime the guidance could be stated as a ratio ("keep phase
overhead under ~1% of the region you are measuring") rather than as a rule about loops.

**11. `count()` rejects a float, but `phase()` accepts a name built from data** until it silently
folds at 4096 paths. The name is the more damaging mistake of the two — it degrades a report
rather than raising — so it deserves at least the same protection. Warn on the first name that
looks data-derived, or offer `strict_names=True`. I ended up pinning the phase vocabulary in a
test to get this guarantee myself.

**12. `Profiler(enabled=False)` still constructs the object** and calls `open_process()`. Every
user will need the opt-in-behind-a-config-flag pattern; a documented null object
(`Profiler.disabled()`) would make it obvious, and would let `enabled=False` mean "allocate
nothing" rather than "allocate and then check a flag".

---

## What already works, and should not be traded away

The deliberate separation of **"measured zero" from "could not measure"** is the reason I trust the
I/O numbers at all. The 0.3.0 changelog entry about a failed counter read being differenced into
372.5 GB describes exactly the failure that makes people stop believing a profiler, and the `io_ok`
flag plus the dropped-interval count is the right fix. Same for naming superseded run attempts
instead of merging them — a requeued Slurm job is the common case here, and silently doubled totals
would have been worse than no totals.

Smaller things that made this easier and are worth keeping: `with_samples=False` on `merge_run`
(my test fixture would otherwise pay for megabytes it never reads), the per-worker `written_at`
that makes staleness derivable at report time rather than trusted from the file, `sync=True`
draining at *both* ends of a phase (my hand-rolled version synced only on exit and was therefore
billing each forward for whatever the previous one left queued — the package's version is simply
more correct than what it replaced), and `count()` raising on a float instead of truncating.

The overhead table in the README is also doing real work: it is what let me decide, per call site
and without measuring, which regions could take a phase and which should take a counter.
