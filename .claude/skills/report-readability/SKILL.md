---
name: report-readability
description: Render a real report, HTML report or trace timeline from a real run and read it as a first-time reader would — checking that every figure states its denominator, units and caveat, that nothing overflows or truncates, and that the page answers the question it was opened with. Use after any change that can reach a rendered page, and when improving explainability of the produced files.
---

# Reading the page as a stranger

`CLAUDE.md` §5b makes this binding: any change that can reach a report, source or trace page ends by generating that page and looking at it. **A passing test suite says the page is well-formed. It does not say the page is readable**, and this package's entire output is something a person opens and has to understand.

The failure this catches is specific and it is not cosmetic. A number without its denominator gets read against the wrong base, and the reader takes the wrong action — the same damage as a wrong number, arrived at differently. This project has shipped that defect before: a role block printing `mcts 66.0%` with nothing stating what the percentage was *of*, next to HTML lane metadata saying the same lanes worked 35.2% of the time. Both were correct. Together they were unreconcilable without reading `report.py`.

## Render from a real run

Not a fixture. Fixtures are built to have the shape the test wants and will not show you the overflow, the collision or the missing caveat.

```
poetry run python -m lineprofiler.accounting.cli report RUN -o r.txt
poetry run python -m lineprofiler.accounting.cli report RUN --format html -o r.html
poetry run python -m lineprofiler.accounting.cli trace  RUN -o t.html
```

Read all three. They fail differently: the text report fails on column width, the HTML report on unstated denominators, the timeline on things discoverable only by hovering.

## The four questions

Ask each one against the actual bytes, in order.

### 1. Does the page answer the question it was opened with?

Nobody opens a profile out of curiosity. They open it holding a question — *why is this slow*, *is the GPU the constraint*, *is this worker hung or queueing*. If answering still requires arithmetic somewhere else, the page has not delivered.

The sharpest version of this test: **the tables are read before the chart, and most readers never hover.** Anything available only on hover is, for most of your audience, absent. The timeline is ordered conclusions-first for exactly this reason — `Findings`, then `Phase summary`, then the chart, then supporting tables — because a reader dropped straight into a canvas has to already know what a healthy run looks like.

### 2. Is every figure labelled with what it means and what it excludes?

**An unlabelled estimate is read as a measurement.** This is the project's governing rule about output and it has the most history behind it.

- **Every percentage names its denominator.** Share of wall clock, of role self time, of summed lane-seconds — these give materially different readings from the same run.
- **Anything summed across lanes is divided by lanes, never by `trace.duration_ns`.** Two actors blocked for a whole run wait two lane-seconds per wall second; a wall-clock denominator once produced *"costing 155% of the run"* at the top of the findings. A share above 100% reads as a bug and discredits every correct number beneath it.
- **A column summed across processes says so** — `3m 16s (Σ2 proc)` — or a total legitimately exceeding the runtime in the header reads as an error.
- **Sampled figures stay marked.** `sample_stride` is 0 for measured and *n* for sampled; the report prefixes `~` and names the rate. Merging takes the max, so one sampled contributor taints a total. Never let a `~` get dropped in a renderer.
- **Estimates are never presented as measurements.** `wait_ns = wall − cpu` is *blocked-on-something*, not proof of a queue. A span's location names where a function is **defined**, never the line that blocked — a span covers a whole call, and nothing may present `co_firstlineno` as the line that spent the time.
- **The GPU strip is 1 Hz, whole-device, every process, and joins to nothing.** It carries a note saying so. Never let the layout imply a reading belongs to the span drawn above it.
- **Absent is not zero.** A resource never measured is omitted with a reason, never rendered as `0`. `busy 0.0%` (a real reading: the device was idle) and `this run n/a` (NVML attributed no sample to these pids) are different statements and the page keeps them apart.
- **A sentinel must not already mean something else on the same page.** When you add one, grep for it first. `"?"` was the obvious marker for an uncomputable runtime and was wrong: the header already prints `Host ?` for an unknown host and `compare` uses `?` for a thin sample, so `Runtime ?` sitting directly above `Host ?` read as one kind of gap in two places. Reusing a symbol that means something else is the presentation-layer form of letting "unmeasured" be represented by a valid value. `UNUSABLE` is `n/a`, which is what the wait column already prints.

### 3. Does anything overflow, run together or truncate?

Long paths, deep phase names and wide tables are where this breaks. Check mechanically rather than by eye:

```python
print(max(len(l) for l in open("r.txt")))   # the text report is a fixed-width contract
```

Labels are bounded at two chokepoints — `report.format_label` for text, `htmldoc.clip_label` for HTML — and both mark the cut with `…` and keep the tail, because the leaf is what a reader greps for. **A new column or page that formats a name without going through one of those reopens the defect**, which is the thing to check when reviewing a new renderer.

A measured *number*, by contrast, is never truncated: an overflowing field pushes the row right rather than printing a wrong value. Only derived figures are compacted (`format_rate`). Keep that distinction — truncating a label loses nothing recoverable, truncating a measurement invents one.

### 4. Does the page state its limits where the reader hits them?

**A caveat must sit with the number it qualifies, not only in a trailing block.** This one has a scar: `CAVEATS` correctly listed three superseded workers while the header said `Processes 1` and the findings drew confident conclusions from the surviving quarter — 80 lines above the disclosure. The fix put the warning in the header, where the reader is when the number misleads them.

So: for each caveat in the trailing block, find the number it qualifies and ask whether a reader who stops there is misled. If yes, the caveat is in the wrong place.

**This is the most-repeated defect in this codebase — three instances so far**, and every one had the caveat present, correct, and too far down: the superseded-worker disclosure below a header claiming `Processes 1`; `Runtime n/a` not saying the clock was why or that phase totals were unaffected; the cross-host NTP bound 7,800 characters below the findings that rest on it. Treat it as the default suspicion rather than a rare case.

The structural cause is worth naming, because it will recur: **a conclusions-first page moves the reader's stopping point, and every caveat has to move with it.** Putting a findings block at the top invalidated the placement of every disclosure below it, none of which had changed. So whenever you add a summary section, reorder a page, or make anything render earlier, re-run this question against *every* existing caveat — not only the text you just wrote.

Corollary — **do not invent a conclusion to fill a space.** `_findings_block` is deliberately silent without a trace, because findings come from spans, and a weaker finding derived from phase totals would put a claim at the top that nothing below it supports. Silence beats an unsupported headline.

## Then fix, re-render, and read again

Iterate until a first-time reader would reach the right conclusion without help. That is the stopping condition — not "the tests pass".

When a readability fix lands, pin it with a test asserting the *rendered* property: the maximum line width, the presence of the denominator string in the header, that a `~` survives a merge, that a caveat appears in the header and not only the footer. `tests/test_html_trace.py` already works this way — it asserts every `data-anchor` names something the payload can actually draw, because a jump button pointing at a phase the chart never carried does nothing, silently.
