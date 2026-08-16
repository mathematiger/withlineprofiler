# Configuration

Everything that changes what gets profiled or what the package needs installed.

## Adopting either tool in two lines

Both tools also come as a module-level entry/exit pair — the alternative to a `with` block for
dropping either one into a function you do not want to restructure:

```python
from lineprofiler.accounting import start, stop

start(role="actor")        # line 1: top of the region
...                         # existing code, unmodified
stop()                      # line 2: bottom of the region
```

```python
from lineprofiler import start_profiling, stop_profiling

start_profiling()          # line 1
...                         # existing code, unmodified
stop_profiling()           # line 2
```

Both are **opt-in and off by default**: with nothing configured, `start_profiling()` /
`accounting.start()` cost a no-op check and nothing else, so the two lines are safe to leave
committed permanently rather than added and removed per debugging session. Turn profiling on
with:

```
LINEPROFILER_ENABLED=1   # master switch for start_profiling()/stop_profiling()
```

`accounting.start()`/`stop()` use `accounting`'s own existing `LINEPROFILER_PROFILE` switch
(see below) — the two tools' switches are independent, matching the fact that they share
nothing but the distribution.

To scope `start_profiling()` to part of the codebase without touching a single call site, add
an optional table to `pyproject.toml`:

```toml
[tool.lineprofiler]
include = ["src/mypkg/**"]        # only these paths are traced (glob, relative to the project root)
exclude = ["src/mypkg/generated/**"]  # exclude wins over include
functions = ["*.train_step", "*.forward"]  # only these function/method names (glob over __qualname__)
```

All three are optional and default to "everything under the project root" — the same behavior
as constructing `LineProfiler()` directly. This table is read once per process and cached; it
never runs on the hot per-line path.

`with profiler:` (below) remains the better choice inside a Jupyter notebook or any region
you're actively iterating on interactively — the two APIs are interchangeable and neither is
deprecated by the other.

### Optional dependencies

None of `psutil`, `torch`, `nvidia-ml-py` or `viztracer` are required. Each is imported
lazily behind a capability check (`capabilities.py`); whichever are missing just disables the
block of the report they feed, and construction never raises.

| Package | Extra | Powers |
|---|---|---|
| `psutil` | `resources` | RSS (memory block) and per-process I/O counters (`I/O` block, both `bytes` and `chars` layers) |
| `nvidia-ml-py` | `gpu` | GPU block: whole-device (`busy`) and per-pid (`this run`) utilisation, read by the 1 Hz sampler |
| `torch` | none — install separately | CUDA allocator stats (VRAM allocated/reserved), `phase(sync=True)`, NVTX ranges (`annotate=True`), the `backend="torch"` window |
| `viztracer` | `viztracer` | the `backend="viztracer"` window |

```
pip install with-line-profiler[resources]   # psutil  -> memory + I/O blocks
pip install with-line-profiler[gpu]         # nvidia-ml-py -> GPU utilisation block
pip install with-line-profiler[viztracer]   # viztracer backend
pip install with-line-profiler[all]         # psutil + nvidia-ml-py + viztracer
pip install torch                           # separately; CUDA memory, sync=True, annotate=True, backend="torch"
```

None of this is threaded through your code — construct `Profiler` the same way regardless,
and each block just appears once its package is importable and, for NVML, once a device is
visible:

```python
profiler = Profiler(run_dir="profile", role="learner")
```

**`psutil`** drives the sampler's memory and I/O rows: `Process.memory_info().rss` for the
memory block, and `Process.io_counters()` for both layers reported under `I/O` —
`read_bytes`/`write_bytes` (block device) and `read_chars`/`write_chars` (syscalls, cache hits
included). Without it the sampler still starts; those two blocks are absent, nothing else is.

**`torch`** is read for CUDA rather than declared as a dependency, so install it yourself if you want GPU features. It backs:
- `cuda_alloc`/`cuda_reserved` sampler rows → `VRAM allocated (peak)` / `VRAM reserved (peak)`
  in the GPU block, via `torch.cuda.memory_allocated()`/`memory_reserved()`
- `phase(name, sync=True)`, which calls `torch.cuda.synchronize()` at both ends of the phase
  so its wall time reflects GPU completion rather than kernel enqueue; a no-op when torch is
  absent or no CUDA device is visible
- `Profiler(..., annotate=True)`, which wraps every phase in `torch.cuda.nvtx.range_push/pop`
  (falling back to the standalone `nvtx` package) and `torch.profiler.record_function`, so an externally started `nsys profile` or Kineto capture shows your phase names
- `Backend.TORCH`, the `backend="torch"` window below

**`nvidia-ml-py`** (imported as `pynvml`) is initialised once on first use; if `nvmlInit()`
fails — no driver, no GPU — the capability degrades to `None` and the sampler skips GPU rows.
It supplies the two numbers the GPU block reports per device: `nvmlDeviceGetUtilizationRates`
(whole-device `busy`) and `nvmlDeviceGetProcessUtilization` (`this run`, this pid's share).

**`viztracer`** backs only `Backend.VIZTRACER`, the other heavy-profiler option below; it is
never imported for anything else.
