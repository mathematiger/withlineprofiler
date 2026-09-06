# How this compares to the other Python profilers

Python has several good profilers and they do genuinely different things. This page is here
to help you pick the right one, including when that is not this package.

## At a glance

| | `lineprofiler` (this) | [`line_profiler`](https://github.com/pyutils/line_profiler) | [py-spy](https://github.com/benfred/py-spy) | [Scalene](https://github.com/plasma-umass/scalene) | [VizTracer](https://github.com/gaogaotiantian/viztracer) |
|---|---|---|---|---|---|
| **Granularity** | per line | per line | per line, sampled | per line, and per function | per function call, on a timeline |
| **Marking what to profile** | nothing — every function under your project folder while active | a `@profile` decorator on each function (`kernprof` injects it) | nothing | nothing | nothing |
| **How you run it** | two lines inside your existing script, run it normally | `kernprof -lv script.py`, a separate invocation | `py-spy record -- python script.py`, or attach to a live pid | `scalene script.py` | `viztracer script.py` |
| **Mechanism** | `line_profiler`'s C callback, told what to watch by a `sys.monitoring` discovery hook | `sys.monitoring` (3.12+), callback in C++ | sampling, from a separate process | sampling with interposition | C-extension tracer |
| **Overhead** | ~240 ns per line event — the same engine, so the same cost | ~240 ns per line event | very low | low to moderate | roughly 2–4x |
| **Attach to an already-running process** | no | no | **yes** | no | no |
| **Memory / GPU** | no (but see the accounting layer) | no | no | **yes, both** | no |
| **Multi-process aggregation** | **yes, via `lineprofiler.accounting`** | no | per process | limited | per-process traces |
| **Output** | terminal tables, self-contained HTML | terminal table | flamegraph SVG, speedscope, live TUI | terminal and a web GUI | Perfetto timeline |

## When to use which

**Use `line_profiler`** directly if `kernprof` is already in your workflow, if you want to
profile exactly one decorated function rather than a region, or if you need the parts of its
surface this package does not wrap — `--prof-mod` auto-profiling, `%lprun`, rich output, the
scoping policies. This package *uses* `line_profiler` for the timing, so the numbers are the
same either way; what differs is how you say what to profile.

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

- **The timing is `line_profiler`'s.** Since 0.9.0 this package's line profiler is a front end:
  it decides *what* to watch and renders the result, and `line_profiler`'s C callback does the
  measuring. Measured on one machine over 600,000 line events, `with profiler:` costs 239 ns
  per event and `line_profiler` called directly costs 240 ns — the wrapper is free. What this
  package adds is that you never name a function.
- **The pure-Python engine is the fallback, and it is 3.7x slower.** ~895 ns per line event
  against ~240. It runs when `line_profiler` is not importable, or when you ask for it with
  `engine="builtin"` or a `backend=`. Its numbers agree with the C engine's; only the cost of
  taking them differs.
- **`sys.monitoring` is not a differentiator.** `line_profiler` 5.0 adopted it too, and this
  package is built on it. Both coexist with coverage.py rather than fighting for the trace hook.
- **The accounting layer is the genuinely uncommon piece.** Aggregate phase timing that
  merges across processes and nodes, with per-role attribution and a wait/compute split, is
  not something the others set out to do.

---

Verified against `line_profiler` 5.0.2 and Scalene 2.3.0 as installed, the py-spy README and
the VizTracer documentation. Last checked 2026-09. The overhead figures come from
`benchmarks/bench_lineprofiler.py`, which is one command — `poetry run python
benchmarks/bench_lineprofiler.py` — so a claim here can be rechecked rather than trusted. A
comparison table that silently rots is the documentation equivalent of a wrong number; if you
find one that is out of date, please open an issue.

### Scalene, measured

Scalene samples rather than traces, so it answers a different question and costs far less:
the same loop runs 2.0x slower under `scalene run --cli --cpu-only` and 2.7x with memory
profiling on, against 12x for either line-tracing engine here. Reach for it when the question
is memory, native-versus-Python time, or GPU, or when you do not yet know which of the three
it is. What it cannot give you is a hit count: a sampler cannot say a line ran 49,200 times,
and that number is what makes phase boundaries visible in the worked example in the README.
It also has no exact answer for a region a few milliseconds long, where a handful of samples
is all it gets.

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
