---
name: adversarial-inputs
description: Probe this profiler with hostile user-supplied values — phase and counter names, extreme magnitudes, degenerate runs, corrupt worker files — and check what reaches the rendered report, HTML and timeline. Use when hunting edge cases, testing input handling or escaping, or when a change touches phase naming, counters, merging, or any renderer.
---

# Probing with hostile inputs

Everything a user names reaches a rendered page: phase names, counter names, roles, file paths. This package mails those pages to other people. So the question at every surface is not "does it crash" but **"what does it print, and is that true?"**

## Running a probe correctly

Four traps, each of which produces a convincing phantom bug. Every one of these has already burned an iteration:

1. **`enabled=True`, always.** Profiling is opt-in. Without it, every method is a no-op, `close()` returns cleanly, and **not one file is written**. A probe missing this reports "the profiler silently produces nothing" — which is the documented default, not a defect.
2. **Absolute `run_dir`, run from the repo root.** It resolves against the cwd at construction. Get this wrong and the artifacts exist somewhere you never look.
3. **Never pipe a render through `head`.** `cli trace` prints progress, then writes the file. `head` closes the pipe first and the write is lost — indistinguishable from a renderer that produced nothing.
4. **Define helpers before the table that calls them.** A list of `lambda`s built above its own helper defs raises `NameError` for every case and reports eleven false failures.

Skeleton — scratchpad only, never `tests/`:

```python
from pathlib import Path
from lineprofiler.accounting import Profiler
from lineprofiler import accounting

RUN = Path("/abs/path/to/scratchpad/run").resolve()

def _phase(n):
    with accounting.phase(n):
        pass

p = Profiler(run_dir=str(RUN), role="probe", install=True, enabled=True, trace=True)
for label, fn in CASES:          # CASES defined after the helpers
    try:
        fn(); print("OK   ", label)
    except Exception as e:
        print("RAISE", label, type(e).__name__, e)
p.close()
```

Then render **all three** outputs and assert against the bytes on disk. A probe that stops at "it didn't raise" has tested nothing that matters:

```
poetry run python -m lineprofiler.accounting.cli report RUN -o r.txt
poetry run python -m lineprofiler.accounting.cli report RUN --format html -o r.html
poetry run python -m lineprofiler.accounting.cli trace  RUN -o t.html
```

Inspect with Python, not `grep`. Counting occurrences of a multi-line or control-character needle with shell tools gives silently wrong answers here.

## The input catalogue

**Names** (phase, counter, role): empty string; a single space; `a/b` (the path separator — does it forge a fake nesting level?); 10,000 characters; `</script><img src=x onerror=alert(1)>`; a NUL and other C0 control characters; newline and carriage return; RTL override `U+202E`; astral-plane emoji; a name differing from another only by trailing whitespace; two names that are distinct strings but identical after truncation.

**Magnitudes**: `count(n, 0)`; a negative count; `2**70` (beyond int64 — does it survive the array-backed buffers and JSON?); a count that overflows when summed across 10,000 workers. Floats are already rejected by `count()` with a clear `TypeError`; that is correct, keep it.

**Degenerate runs**: zero phases; one phase entered once and lasting 0 ns; a phase entered but never exited (crash mid-phase); nesting past `MAX_DEPTH` (32); more than `MAX_PHASES` (4096) distinct paths; a run whose wall time is zero (does anything divide by it?).

**Corrupt artifacts** — these are the highest-value ones, because they are what a crashed HPC job actually leaves behind: a truncated final line in a `.trace` sidecar; a `w_*.json` that is valid JSON but missing a key; one holding a string where a number belongs; a zero-byte worker file; two workers with the same pid and different uuids; workers from two attempts in one directory; a worker whose clock anchor disagrees with the others by an hour.

**Values of the right type that cannot be true** — the sharpest sub-case of the above, and the one that finds bugs after the structural cases are all defended. `_read_worker` casts with `float()`/`int()` and catches `TypeError`/`ValueError`, which is complete against a *string where a number belongs* and blind to `float("inf")`, which genuinely is a float. Try: non-finite and negative timestamps; `written_at` before `started_at` (an NTP step mid-run, not hypothetical); negative `wall_ns` or `calls`; `cpu_ns` greater than `wall_ns`; zero `calls` with non-zero `wall_ns`. Note that `json.dumps` writes `Infinity`/`NaN` by default and `json.loads` reads them back, so the profiler's own writer round-trips a non-finite value silently — these arrive from the library, not only from a hand-written file.

## What is already defended — do not re-report these

Verified by probing at version 0.8.2. Re-check only if the relevant code changes.

- **`</script>` cannot break out of the embedded data block.** `htmldoc.embed_json` escapes `</` as `<\/`, so a phase name containing `</script><img src=x onerror=...>` lands inside the JSON as an inert string in both `report --format html` and `trace`. A raw substring match for `<img src=x onerror` **will hit** in both files and is *not* a finding — read its context before reporting it. The XSS defense holds.
- **The timeline script writes with `textContent`, never `innerHTML`**, and there is a test pinning that.
- **`count()` rejects non-int values** with `TypeError: count() takes an int, got float`. NaN and infinity are refused at the door, which is the right layer.
- **Phase-name *shape* checking already exists** and warns once per distinct shape when names are built from data (`episode_0`, `episode_1`, …), naming the culprit. It runs in `_admit` only — once per distinct path, never on the hot path.

- **Label width and control characters are bounded at the chokepoints** (fixed in iteration 1). Text labels pass through `report.format_label`, which calls `_printable`; HTML labels pass through `htmldoc.escape` and `htmldoc.clip_label` (90 columns, marked cut). C0/C1 and U+2028/U+2029 become U+FFFD. **A new renderer that formats a user-supplied name without going through one of those reopens all of it** — that is the thing to check when reviewing a new page or column.
- **A measured count is never truncated**, however large. That is deliberate and documented in `_counter_rows`: a truncated number is a wrong number, so an overflowing field pushes the row right instead. Only the *derived* rate is compacted (`format_rate`, exponent form above 1e15). Do not "fix" the long count.

## Where a finding is likely to still be

The input catalogue above is only partly worked through. Nothing has yet probed **corrupt artifacts** — the highest-value group, because it is what a crashed HPC job actually leaves behind — or degenerate runs, merge across attempts, or the lifecycle surfaces. See `HARDENING.md` for the current list and what each iteration has already settled.

## Turning a probe into a test

Probes stay in the scratchpad. What lands in `tests/` is one deterministic case per defect, built by hand, named as the claim it defends, and **seen to fail before the fix**. Put escaping and rendering cases in `tests/test_html_trace.py` / `tests/test_html.py`, and wrong-number and corrupt-file cases in `tests/test_accounting_resilience.py`, which already owns "failures that used to produce wrong numbers rather than missing ones".
