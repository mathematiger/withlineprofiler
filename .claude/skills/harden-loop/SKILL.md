---
name: harden-loop
description: Run one iteration of the reliability loop for this profiler — pick an unhardened surface, break it with a real adversarial run, fix what broke, add the regression test, verify. Use when asked to "harden", "find breaking points", "make this reliable/production-ready", "find edge cases", or to continue/iterate the hardening work. Not for ordinary feature work or a specific known bug.
---

# The hardening loop

One invocation = **one surface, taken from red to green**. Resist doing four surfaces shallowly; a surface half-hardened is a surface that reads as done and is not.

The output of an iteration is always the same four things, in this order: **a failing test, a fix, a passing test, a line in the ledger.** If you cannot write the failing test first, you have not actually found a defect — you have found a suspicion. Go get the evidence.

## Why this project needs a loop rather than a checklist

This package's stated worst failure mode is not the crash. It is in `CLAUDE.md` under *The wrong-numbers class of bug*: continuing, writing a file that parses, and reporting a result that looks complete. A crash is self-reporting; a confident wrong number is not, and it is acted on. Four such defects have already been found and fixed here, each with a regression test.

That is why the loop's bias is **"what would this print if the measurement failed?"** rather than "what raises?". A surface that raises loudly is usually fine. A surface that silently substitutes a plausible value is the bug.

## Step 1 — Choose the surface

Read `HARDENING.md` (the ledger) first. It records what has been probed, what was found, and what was deliberately judged not-a-bug. **Do not re-probe a surface listed as done unless the code under it has changed** — that is the main way this loop wastes an iteration.

Pick the highest-value unprobed surface. Ranked by where this codebase's damage actually lives:

1. **Anything differenced or accumulated** — byte counters, CPU times, GPU samples, anything cumulative. This is where "unmeasured" gets silently represented as a valid value. The existing `IoSnapshot.available` and `Sample.cpu_percent == -1.0` sentinels exist because of exactly this; ask whether a new field has the same protection.
2. **Anything merged across workers, attempts, hosts or threads** — merge is where one worker's total becomes everyone's, and where two attempts become one run that never happened.
3. **Anything rendered** — a number reaching a page without its denominator, its units, or the caveat that qualifies it. See the `report-readability` skill; a page that misleads is a defect of the same severity as a wrong number, because it produces the same wrong action.
4. **Anything a user names** — phase names, counter names, roles, paths. See the `adversarial-inputs` skill.
5. **Anything with a background thread, a signal, a fork, or an `atexit`** — the lifecycle surfaces, where the failure is a leak that surfaces far from its cause.
6. **Any concurrency model the design did not assume.** Phase stacks are per *thread*. Ask what happens under the models a user's code actually uses — asyncio tasks (found: concurrency recorded as nesting), thread pools, greenlets, subinterpreters. The bug is not that the tool lacks a feature; it is that it produces a confident tree describing a call structure that never existed.
7. **The numbers the docs state about the tool itself.** An overhead figure is load-bearing here — the whole pitch is "cheap enough to leave on for twelve hours", and a reader budgets instrumentation with it. Re-measure the headline claims against `benchmarks/bench_accounting.py` and check they describe the *default* configuration; found once already at ~2× optimistic.

State which surface you picked and why, in one line, before you start.

## Step 2 — Break it with a real run

**The evidence must come from running the thing, not from reading it.** A defect argued from source is a hypothesis; reviewers of this project have been wrong about the source before (see the `WITHLINEPROFILER_GAPS.md` gaps #3 and #6, where two "library gaps" turned out to be caller bugs). A defect demonstrated by an artifact on disk is a fact.

Write a probe script into the scratchpad — never into `tests/` yet, and never into the repo. Run a real `Profiler` with `enabled=True`, render the real artifact, then assert against what landed on disk.

Two traps that will cost you an iteration each:

- **`enabled=True` is required.** Profiling is opt-in; without it (or `LINEPROFILER_PROFILE`) every method is a no-op, `close()` succeeds, and **no files are written at all**. A probe that forgets this reports a spectacular-looking "silently writes nothing" bug that is just the documented default.
- **`run_dir` resolves against the cwd** at construction. Run the probe from the repo root with an absolute `run_dir`, or the artifacts land somewhere you are not looking and you will report a phantom.

Also: do not pipe a render through `head`. `cli trace` streams progress and then writes; `head` closes the pipe and the write never happens, which looks exactly like "the renderer produced nothing".

## Step 3 — Judge it before you fix it

Not everything you find is a bug, and **filing a non-bug as a bug is worse than missing one** — it adds a test that pins wrong behaviour in place. Sort each finding:

- **Wrong number** — the output states something untrue. Always fix.
- **Silent loss** — the output omits something and does not say it omitted it. Almost always fix; the fix is usually a caveat, not a computation.
- **Unreadable but correct** — the number is right and the reader cannot use it. Fix, but as a rendering change; do not touch the measurement.
- **Working as designed** — write it in the ledger as judged-not-a-bug, **with the reasoning**, so the next iteration does not re-litigate it.

Check the finding against `CLAUDE.md`'s gotcha list before fixing. Many surprising behaviours there are load-bearing and have a paragraph explaining what broke when it was done the other way — `sample_stride`, `counter_min`/`counter_max` not being differenceable, `UNMEASURED = -1`, the fork registry's weak references. **Changing one of those without reading its paragraph reintroduces a fixed bug.**

## Step 4 — Fix at the right layer

Match the existing structure; this codebase is layered strictly (`histogram` → `phasetree` → ... → `report`) and each module imports only from the ones above it.

- A **measurement** defect is fixed where it is measured, never in the renderer. A renderer that patches a bad number hides the defect from every other consumer.
- A **presentation** defect is fixed in the renderer, never by changing what is recorded.
- Never let the hot path pay for a fix. `_PhaseScope.__exit__` runs at ~3 µs; a method call there costs ~80 ns. If the fix touches it, run `benchmarks/bench_accounting.py` before and after and put both numbers in the commit message.
- A new field written on the hot path must be written in **both** `_PhaseScope.__exit__` *and* `PhaseStats.record()`, or it stays zero in real runs while unit tests calling `record()` pass. This has bitten before.

## Step 5 — The regression test

Write the test **so that it fails before the fix**. Then apply the fix. Then watch it pass. In that order — a test written after a fix routinely passes for reasons unrelated to the fix, and this project has ~9,700 lines of tests whose value depends entirely on each one having been seen to fail once.

Place it in the file that owns the concern (`tests/test_accounting_resilience.py` for wrong-numbers defects, `test_html_trace.py` for the timeline page, and so on — `CLAUDE.md` lists what each file owns). Name it as the claim it defends, in a sentence: `test_a_phase_name_longer_than_the_column_is_truncated_in_the_report`, not `test_long_names`.

**Build the input by hand where the real thing cannot promise the shape you need.** `tests/test_accounting_findings.py` does this deliberately — a finding about "the lane idle 80% of the run" needs a run idle exactly 80%, which no real workload will give you. Follow that precedent rather than profiling something and hoping.

Determinism is not optional: no wall-clock thresholds, no sleeps used as ordering, nothing that depends on machine load. There is already one timing-sensitive test in this suite and `CLAUDE.md` has to carry a paragraph apologising for it. Do not add the second.

## Step 6 — Verify, then record

Run, in this order, and paste the real output — never assert a check passed without having run it:

```
poetry run pytest tests/ -q          # whole suite, not just your file
poetry run mypy lineprofiler tests   # strict
poetry run ruff check lineprofiler tests
```

If the change can reach a report, source or trace page, **render a real page and read it** — `CLAUDE.md` §5b makes this binding, and the `report-readability` skill is the checklist for that read. A passing suite says well-formed; it does not say readable.

Then append to `HARDENING.md`. One row per iteration:

| Surface | Probe | Finding | Severity | Fix | Test | Status |

Keep judged-not-a-bug rows in the table. They are the most valuable rows in it — they are what stops the loop from cycling.

## Closing an iteration

Report to the user: the surface, what actually broke (with the evidence), what you changed, the test that now defends it, and the suite result. Then name the surface you would probe next and stop. **Do not roll straight into the next iteration** unless asked — the user decides how much of this to buy.

If a surface turns out clean, that is a complete and successful iteration. Record it as clean with what you probed, and say so plainly. A loop that can only report defects will start inventing them.
