# Multiple processes, and multiple nodes

**`run_dir` is resolved to an absolute path at construction**, before it is exported to
children. A relative default like `"profile"` otherwise means a *different* directory in every
process that has its own working directory — which is exactly what a batch system hands each
rank — so one run scattered across the filesystem and merged as several short ones. A relative
path resolves against the working directory of the process that constructed the profiler,
which is what you meant by it; `$SLURM_SUBMIT_DIR` is deliberately *not* used, because portals
set it to their own installation directory (Open OnDemand reports
`/var/www/ood/apps/sys/dashboard`), which is somewhere you neither chose nor can usually write.
Passing an absolute `run_dir` remains the clearest thing to do.

Every process writes its own `workers/<host>/w_<run>_<pid>_<uuid8>.json`; `report` merges
them. The uuid matters because a restarted worker reuses its rank but not its pid, and the
per-host directory keeps a large run from concentrating two files per rank — plus a rename
per flush — into one directory, which is a metadata hot spot on Lustre.

Each worker records the node it ran on and its rank, read from whichever launcher is present
(`SLURM_PROCID`, `RANK`, `OMPI_COMM_WORLD_RANK`, `PMI_RANK`) along with the batch job id.
That is what makes *which node is slow?* answerable; the report names the nodes involved and
counts processes by worker file rather than by pid, which collides across nodes.

**Runs are identified.** A rerun into the same directory is a separate attempt: `report` shows
the newest and names the superseded ones rather than merging them, which used to inflate every
total for a requeued job. Children inherit the attempt through `LINEPROFILER_RUN_ID`.

**Runs record the code they measured.** The report header names the git revision, and marks a
dirty tree with a file count and a hash of the diff:

```
Runtime 2m 56s   Processes 4   Roles actor x2, inference_server x1, learner x1
Host node0   Run 20260817T091934-9ddbb7
Source c49ce841 (+dirty: 26 files, diff sha 3f9a1c)
```

This closes a gap that silently invalidates analysis: a profile of the *committed* code, read
against a working tree that has since fixed the constraint the profile found, is a claim about
a program that no longer exists. One `git` call at startup, on the one rank that writes the run
metadata, and silence — not an error — when there is no repository or no `git`. Pass
`Profiler(source={...})` to supply your own instead; a config hash belongs here too, since a
config change alters behaviour as much as a code change.

**On preemption**, `SIGUSR1` and `SIGHUP` flush before exit alongside `SIGTERM` — Slurm's
`--signal=USR1@120` idiom terminates without running `atexit`, so the last interval used to be
lost exactly when you wanted it. `SIGKILL` remains unreachable; the periodic snapshot is what
survives it.

**`os._exit()` is the other unreachable exit, and it is not exotic** — it is how a
multiprocessing entrypoint normally tears a worker down, and how most "exit immediately without
running cleanup" paths are written. It skips `atexit` *and* never delivers a signal, so neither
hook above fires and everything since the last periodic flush is lost. If your teardown path
calls it, call `close()` yourself first:

```python
profiler.close()
os._exit(0)
```

The run still parses and still looks complete — it is simply missing its tail, which is the
failure mode this layer works hardest to avoid elsewhere. Lower `snapshot_interval_s` if you
cannot reach the exit path.

**An enabled profiler changes the process, and `close()` changes it back.** Constructing one
registers an `atexit` hook, chains the three signals above, and registers `os.register_at_fork`
callbacks — all process-global, none of it scoped to the object. `close()` removes the `atexit`
hook and puts the signal handlers back. The fork callbacks are the exception: CPython has no
`unregister_at_fork`, so they stay registered for the life of the interpreter and instead go
inert, dispatching over weak references and skipping any profiler that has closed.

This matters most inside a test suite, where profilers are constructed and discarded in the same
interpreter as everything else:

```python
def test_something(tmp_path):
    profiler = Profiler(run_dir=tmp_path, enabled=True)
    try:
        ...
    finally:
        profiler.close()      # not optional: it is what un-does the above
```

Closing order need not match construction order — a parent closed before its child is handled —
but a profiler that is *never* closed keeps its handlers for the rest of the process. If a host
installs its own handler on top of a live profiler, `close()` deliberately leaves the profiler's
handler in place rather than delete the host's; it is inert by then and still chains correctly.

Better still, assert against a subprocess run rather than embedding a profiler in the test
process at all — see [Using it in tests](accounting-recipes.md#using-it-in-tests). That is the pattern this layer is
built for, and it sidesteps the question entirely.

**On a large run**, pass `--no-samples`. Resource samples dominate merge memory — a 12-hour
worker holds roughly 28 MB of them, about 1.8 GB across 64 workers, and the derived intervals
roughly double the peak. Phase trees for the same run are a few megabytes.

`spawn`, `fork` and `forkserver` are all supported and tested at 1, 4 and 16 workers. A
worker that raises still contributes everything it recorded before dying; a worker
`SIGKILL`ed before its first flush leaves nothing, and the report says so rather than
under-reporting silently.

Enabling a profiler sets `LINEPROFILER_PROFILE=1` and `LINEPROFILER_RUN_DIR` in the
environment, so `Profiler(role="actor")` in a spawned worker joins the parent's run with no
configuration threaded through. **`forkserver` is the exception**: its daemon is forked once
and its children inherit the daemon's environment as it was when the daemon started, so
export `LINEPROFILER_PROFILE=1` in the shell before training, or pass `enabled` and
`run_dir` to each worker explicitly.

Forking is handled: a forked child gets its own file, an empty tree and a clean phase stack,
and the profiler's own threads are stopped for the duration of the fork so that enabling it
never adds fork-deadlock risk to a codebase that forks.

### Heavy profilers, for a bounded window

```python
Profiler(run_dir="profile", backend="torch", backend_window=(100, 110),
         window_phase="iteration")
```

Starts `torch.profiler` on the 100th entry into `iteration` and stops it at the 110th,
writing a Chrome trace into `profile/backend/`. `backend` is a single enum value, so two
heavy profilers cannot be active at once — they contend for the same interpreter hooks.

`Profiler(..., annotate=True)` additionally wraps every phase in an NVTX range and a
`torch.profiler.record_function`, so an externally started `nsys profile` or Kineto capture
shows your phase names. This package never launches nsys itself.

#### Running a heavy backend for a window

```python
profiler = Profiler(
    run_dir="profile",
    backend="torch",              # or "viztracer"
    backend_window=(100, 110),    # start on the 100th entry into "iteration", stop on the 110th
    window_phase="iteration",
)
```

Starts the chosen backend on the 100th entry into `iteration` and stops it on the 110th,
writing its artifact under `profile/backend/` — a Chrome trace (`torch_trace.json`, open at
`chrome://tracing` or with Perfetto) for `backend="torch"`, a VizTracer capture
(`viztracer.json`, open with `vizviewer`) for `backend="viztracer"`. `backend` is a single
enum value: `line_profiler`, `cProfile`, VizTracer and `torch.profiler` all contend for the
interpreter's trace hook, so only one heavy profiler can run at a time. If the chosen
package isn't installed, the window degrades to a no-op and records `unavailable_reason` in
`metadata.json` instead of raising.
