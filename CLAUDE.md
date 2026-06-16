# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`with_line_profiler` (PyPI name; import as `lineprofiler`) is a context-manager-based line-by-line profiler. Wrap code in `with profiler:` and it records per-line hit counts and timing for code that lives inside a project folder.

## Commands

- Install (dev): `poetry install` (creates a venv with the `dev` + `test` dependency groups)
- Lint: `poetry run ruff check lineprofiler` (config in `pyproject.toml`: line-length 100, target py312, rules `E,F,W,I,N,UP,ANN,B,C4,SIM`, ignoring `ANN101/ANN102` for ruff `^0.5`)
- Type-check: `poetry run mypy lineprofiler` (configured `strict = true`, `python_version = 3.12`)
- Test: `poetry run pytest tests/`
- Build: `poetry build` (build backend is `poetry-core`; metadata is PEP 621 `[project]`, the wheel packages only the `lineprofiler` dir)

Tests live in [tests/test_profiler.py](tests/test_profiler.py): sample functions defined inside that file are profiled with `project_folder` pointing at the tests directory.

## Architecture

Everything lives in [lineprofiler/profiler.py](lineprofiler/profiler.py); [lineprofiler/__init__.py](lineprofiler/__init__.py) only re-exports the public API. Three pieces:

- `LineStats` / `FunctionStats` — dataclasses holding per-line and per-function accumulated `hits` and `total_time`. Functions are keyed by the tuple `(filename, function_name, first_line)`.
- `LineProfiler` — the context manager. Core mechanism:
  - `__enter__` registers `self._trace_callback` via `sys.settrace` (saving the previous tracer); `__exit__` restores it.
  - `_trace_callback` handles `call` / `line` / `return` events. **Timing model:** the delta between two events is attributed to the *previous* line (`self._last_key`, `self._last_line`). The reference timestamp for the next line is taken at the *end* of the callback (a second `perf_counter()` call), so the profiler's own bookkeeping is excluded from the reported per-line times. The line being attributed to is identified from the current `frame` (`_ensure_function`), so the caller's lines are still timed correctly after a nested in-project call returns.
  - **Project filtering is central.** Only frames whose filename is under `self._project_folder` are traced (`_is_in_project_folder`). If `project_folder` is not passed to the constructor, it auto-detects by walking up from the *caller's* file to the nearest `.git` directory (`_find_repo_root`). This is why profiling stays scoped to the user's own code instead of stdlib/site-packages.
  - **Caching keeps overhead/memory down.** `_is_in_project_folder` caches its verdict per filename in `_project_cache` (so `Path.resolve()` runs once per file, not per `call` event). Source lines are read once per file into `_source_cache`, and every `FunctionStats.source_lines` for a given file is the *same* dict object — so a file's source is held in memory only once regardless of how many of its functions are profiled.
  - Reporting: `print_stats` (per-function tables), `print_global_top_stats` (top-N lines across all functions), `get_stats` (raw dict), and `clear`/`reset` (reset state and caches; `reset` is an alias).

## Gotchas

- `sys.settrace` is global and single-tracer; this profiler is not thread-safe. Recursive/nested calls share one `FunctionStats` per function key (no per-call-depth breakdown), but their per-line timing is correct because the line is identified from the live `frame` on every event rather than from a single remembered key.
- Some inline comments and a couple of print strings are in German / contain typos (e.g. `"filename not in folde"`); preserve or fix deliberately, don't assume they're bugs to silently rewrite.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
 

 ## 5. Test your code

 Check afterwards first for mypy errors and fix them; and then for ruff errors (also fix).
 Do not delete TODOs in this process.
 This code should be TLR7-ready at the end of this project.

 ## 6. Clean Code — Single Purpose Functions

**Every function does one thing. Name it after what it does.**

- A function should have one reason to exist. If you need "and" to describe it, split it.
- Keep functions short (< 30 lines as guideline). If it scrolls, it's too long.
- No side effects hidden behind innocent names. `get_X()` must not mutate state.
- Extract repeated logic into named helpers — but only when used ≥ 2 times.
- Parameters: fewer is better. More than 4? Consider a dataclass or restructuring.

Ask yourself: "Can I understand this function without reading its body?" If not, rename or restructure.

## 7. Readability — Structure Over Comments

**Code should read top-down like a narrative. Util files are for shared plumbing.**

- Public functions at the top, private helpers below. Reader sees intent before implementation.
- Group related logic into clearly named functions — prefer readable call chains over inline blocks.
- Extract pure utility logic (math helpers, string formatting, generic transforms) into `*_utils.py` files alongside the module that uses them.
- Don't create a god-object `utils.py` — scope utils to their domain (e.g. `mcts_utils.py`, `network_utils.py`).
- Naming: variables and functions should make comments unnecessary. `filtered_actions` > `fa`. `compute_td_target` > `calc`.
- Blank lines separate logical blocks within a function — treat them like paragraph breaks.

The test: A new team member should understand the module's flow by reading function names alone.