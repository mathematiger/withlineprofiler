# How this compares to the other Python profilers

Python has several good profilers and they do genuinely different things. This page is here
to help you pick the right one, including when that is not this package.

## At a glance

| | `lineprofiler` (this) | [`line_profiler`](https://github.com/pyutils/line_profiler) | [py-spy](https://github.com/benfred/py-spy) | [Scalene](https://github.com/plasma-umass/scalene) | [VizTracer](https://github.com/gaogaotiantian/viztracer) |
|---|---|---|---|---|---|
| **Granularity** | per line | per line | per line, sampled | per line, and per function | per function call, on a timeline |
| **Marking what to profile** | nothing — every function under your project folder while active | a `@profile` decorator on each function (`kernprof` injects it) | nothing | nothing | nothing |
| **How you run it** | two lines inside your existing script, run it normally | `kernprof -lv script.py`, a separate invocation | `py-spy record -- python script.py`, or attach to a live pid | `scalene script.py` | `viztracer script.py` |
| **Mechanism** | `sys.monitoring` (3.12+) or `sys.settrace` | `sys.monitoring` (3.12+), callback in C++ | sampling, from a separate process | sampling with interposition | C-extension tracer |
| **Overhead** | high — meant for a region you already suspect | lower; the 4.0 C++ rewrite cut 0.3–1 µs per line hit, up to ~4x faster | very low | low to moderate | roughly 2–4x |
| **Attach to an already-running process** | no | no | **yes** | no | no |
| **Memory / GPU** | no (but see the accounting layer) | no | no | **yes, both** | no |
| **Multi-process aggregation** | **yes, via `lineprofiler.accounting`** | no | per process | limited | per-process traces |
| **Output** | terminal tables, self-contained HTML | terminal table | flamegraph SVG, speedscope, live TUI | terminal and a web GUI | Perfetto timeline |

## When to use which

**Use `line_profiler`** if you already know which function is slow and want the lowest
possible per-line overhead. Its tracing callback is implemented in C++ and it is the faster
tool for a hot function you can decorate. If you are willing to add `@profile` and run
`kernprof`, it is the better choice for that job.

**Use py-spy** if you cannot modify or restart the process — a production service, a job
already running, something wedged that you need a stack trace out of. Nothing else here can
attach to a live pid, and its sampling overhead is low enough to point at production.

**Use Scalene** if the question involves memory or GPU as well as CPU, or if you do not yet
know which of the three is the problem. It separates Python time from native time and tracks
allocations, which is a different and often more useful question than "which line is slow".

**Use VizTracer** if you need to see *ordering* — what ran when, on which thread, in what
sequence. A timeline answers concurrency questions that no aggregate can.

**Use this package** when you want to point at a *region* of code and see per-line timings
without adding decorators, without a separate build or run step, and including code you did
not write and cannot annotate. And use its accounting layer when the question spans processes
and nodes — sixteen actors, a learner and an inference server under a batch scheduler — which
is the case none of the others cover.

The two are complementary in practice: use the accounting layer to find which phase and which
role the time went to across the run, then the line profiler on the region it points at.

## Honest notes

- **`sys.monitoring` is not a differentiator.** `line_profiler` 5.0 adopted it too. It is
  what lets either tool coexist with coverage.py rather than fighting for the trace hook, and
  both benefit.
- **This package's line profiler is the slower one.** It is pure Python where
  `line_profiler`'s hot path is C++. That is the right trade for a bounded region you are
  investigating, and the wrong one for leaving enabled on hot code.
- **The accounting layer is the genuinely uncommon piece.** Aggregate phase timing that
  merges across processes and nodes, with per-role attribution and a wait/compute split, is
  not something the others set out to do.

---

Verified against the `line_profiler` 5.0 changelog, the py-spy README, the Scalene README and
the VizTracer documentation. Last checked 2026-08. A comparison table that silently rots is
the documentation equivalent of a wrong number — if you find a claim here that is out of
date, please open an issue.

## Trace timeline vs VizTracer / Perfetto / nsys

`lineprofiler trace` overlaps their territory, so it is worth being explicit about where it
stops.

**Use the trace timeline when** the question is *which worker was idle, and who was it waiting
for?* It is built for a multi-process pipeline: lanes on a shared clock across processes and
hosts, wait-vs-work shading from measured CPU time, and arrows you declare at your own queue
boundaries. It records aggregates by default and spans only when asked, so it can be left on.

**Use VizTracer or Perfetto when** you need every call in one process at full fidelity, a
flame graph over call stacks, or a UI with real search. VizTracer traces everything by
default; that is its strength and the reason it is not something you leave running for twelve
hours.

**Use nsys or `torch.profiler` when** the answer is on the GPU — kernel timings, memory
transfers, stream overlap. This package deliberately does not reimplement those:
`backend="torch"` starts a Kineto capture for a bounded window, and `annotate=True` puts your
phase names into an externally started nsys capture.

What the trace timeline does **not** do:

- No C extensions or kernels — a `torch` call is one opaque span.
- No per-line detail; that is `LineProfiler`'s job.
- `trace="auto"` measures wall time only, and says so on the page rather than implying zero
  wait.
- It is bounded by a ring buffer, so a long run keeps the most recent spans, not all of them.
