# with-line-profiler

[![PyPI](https://img.shields.io/pypi/v/with-line-profiler.svg)](https://pypi.org/project/with-line-profiler/)
[![CI](https://github.com/mathematiger/withlineprofiler/actions/workflows/ci.yml/badge.svg)](https://github.com/mathematiger/withlineprofiler/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-informational.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/with-line-profiler.svg)](https://pypi.org/project/with-line-profiler/)

Two independent profiling tools in one distribution: line-by-line tracing for a region you suspect, and low-overhead phase accounting for a run too long to trace. Zero dependencies on Python 3.10+, MIT licensed.

```
pip install with-line-profiler
```

```python
# example.py
from lineprofiler import LineProfiler

def slow_function():
    total = 0
    for i in range(1_000_000):
        total += i * i
    return total

profiler = LineProfiler()
with profiler:
    slow_function()
profiler.print_stats()
```

```
====================================================================================================
File: /path/to/example.py
Function: slow_function at line 3
Total time: 710676.0 µs
====================================================================================================
Line #   Hits       Time (µs)       Per Hit (µs)    % Time     Line Content
----------------------------------------------------------------------------------------------------
6        1000000    364848.9        0.4             51.3       total += i * i
5        1000001    345825.8        0.3             48.7       for i in range(1_000_000):
7        1          0.7             0.7             0.0        return total
4        1          0.6             0.6             0.0        total = 0
```

(numbers vary by machine; `File:` prints the absolute path)

No decorators to add, no separate `kernprof` run, no build step. Only code under your project folder is traced — the folder is auto-detected by walking up to the nearest `.git` — so the output is your code, not the standard library.

Want a picture instead? `profiler.to_html("profile.html")` writes an annotated, heat-coloured source view as a single self-contained file.

## Which tool

| Tool | What it does | Use it when |
|---|---|---|
| **`lineprofiler.LineProfiler`** | Line-by-line tracing for a bounded region, scoped to your project folder. | You have narrowed the problem down and want per-line timings inside it. |
| **`lineprofiler.accounting`** | Semantic accounting for regions *you* name. Aggregates only — counts, sums, a fixed-bucket histogram — at ~2 µs per phase, across every process in a pipeline. | You are profiling a long, multi-process training run and need to know which phase, which role and which node the time went to. |

They share nothing but the distribution: `accounting` never imports `LineProfiler`. If you arrived here for a training run, you want the accounting layer — it is the one built to stay enabled for twelve hours.

```python
from lineprofiler.accounting import start, stop

start(role="actor")     # or Profiler(...) / with profiler.phase(...)
...
stop()
```

```
lineprofiler report profile/                        # the run, as a table
lineprofiler report profile/ --format html -o r.html # ...or as a page you can share
```

### Why was that worker idle?

A report says `queue_get` was 80% wait. It cannot say *when*, or *what for* — a total has no position on a clock. Turn on the timeline and the answer is a picture:

```python
Profiler(run_dir="profile", role="actor", trace=True)
```

```
lineprofiler trace profile/ -o trace.html
```

One lane per worker on a shared clock, idle time drawn as absence, and — where you mark a queue with `signal()` / `wait_on()` — arrows from a producer to the consumer it unblocked, plus the critical path that actually set the run's length.

Nothing to instrument first: `LINEPROFILER_TRACE=auto` derives the lanes from function calls in your project, with no change to your code at all.

## Worked example: finding the four phases of an MCTS

Textbooks describe Monte-Carlo Tree Search as four phases — **selection**, **expansion**, **simulation**, **backpropagation** — but the code that actually gets written is usually one long `uct_search` function with four `while` loops in it. The profile is what tells you where the seams really are, and what each seam is worth.

Here is the loop, unedited, as it is usually first written:

```python
def uct_search(root_state, iterations=2000, c=1.4):
    root = Node(root_state)
    for _ in range(iterations):
        node = root
        state = root_state
        while not node.untried and node.children:          # descend the tree
            best, best_score = None, -1e30
            for child in node.children:
                exploit = child.value / child.visits
                explore = c * math.sqrt(math.log(node.visits) / child.visits)
                score = exploit + explore
                if score > best_score:
                    best, best_score = child, score
            node = best
            state = state.play(node.state.board[-1])
        if node.untried:                                    # grow a leaf
            move = node.untried.pop(random.randrange(len(node.untried)))
            state = state.play(move)
            child = Node(state, node)
            node.children.append(child)
            node = child
        while not state.is_terminal():                      # play it out
            state = state.play(random.choice(state.moves()))
        reward = state.reward()
        while node is not None:                             # push the result back up
            node.visits += 1
            node.value += reward
            node = node.parent
    return max(root.children, key=lambda n: n.visits)
```

Wrap it and look:

```python
from lineprofiler import LineProfiler

profiler = LineProfiler()
with profiler:
    uct_search(Game(), iterations=2000)
profiler.print_stats()
```

```
Function: uct_search at line 43
Total time: 189418.1 µs
Line #   Hits       Time (µs)       Per Hit (µs)    % Time     Line Content
----------------------------------------------------------------------------------------------------
52       49200      36221.5         0.7             19.1           explore = c * math.sqrt(math.log(...))
51       49200      25941.7         0.5             13.7           exploit = child.value / child.visits
50       55350      25591.9         0.5             13.5       for child in node.children:
53       49200      24236.1         0.5             12.8           score = exploit + explore
54       49200      22382.1         0.5             11.8           if score > best_score:
55       15066      6989.3          0.5             3.7                best, best_score = child, score
67       12150      5399.5          0.4             2.9       while node is not None:
69       10150      4927.3          0.5             2.6           node.value += reward
68       10150      4918.2          0.5             2.6           node.visits += 1
70       10150      4561.0          0.4             2.4           node = node.parent
48       8150       3854.4          0.5             2.0       while not node.untried and node.children:
57       6150       3663.5          0.6             1.9           state = state.play(node.state.board[-1])
64       5850       2648.7          0.5             1.4       while not state.is_terminal():
59       2000       1805.6          0.9             1.0           move = node.untried.pop(...)
65       3850       1742.1          0.5             0.9           state = state.play(random.choice(...))
...
```

The report is sorted by time, but the **line numbers sort themselves into four contiguous bands** — and those bands are the four phases. Group the rows by band and the structure the textbook claims falls straight out of the measurement:

| Lines | Phase | Share | Hits on the hot line |
|---|---|---|---|
| 48–57 | **Selection** — the UCT descent | **82.0%** | 49,200 |
| 58–63 | **Expansion** — pop an untried move, attach a node | 3.5% | 2,000 |
| 64–66 | **Simulation** — the random playout | 2.7% | 3,850 |
| 67–70 | **Backpropagation** — walk parents to the root | 10.3% | 10,150 |

The hit counts are what make the boundaries unambiguous. Expansion runs **once per iteration** (2,000 hits — one new node per simulation, by definition). Backpropagation runs once per node on the path (10,150 — the average path is ~5 deep). Selection's inner scoring lines run **49,200 times**, twenty-five times per iteration, because it is a loop over children nested inside a loop down the tree. Anything sharing a hit count belongs to the same phase; a change in hit count is a phase boundary.

That also settles which phase deserves attention first. The intuition for MCTS is that the rollout is the expensive part — it is the bit that plays a whole game. Here it is 2.7%, and the UCT formula on line 52 alone outweighs the entire simulation phase by a factor of seven, because it is evaluated once per child per level, and `math.log(node.visits)` is recomputed for every sibling when it is constant across them.

### Cutting the code along the seams

The bands are the refactoring plan: one function per phase, named for what the measurement showed it to be, and each call wrapped in a `with` block that declares which phase it is.

```python
from lineprofiler import accounting
from lineprofiler.accounting import Profiler

Profiler(run_dir="profile", role="actor", install=True)   # once, at start-up

def select(root, root_state, c=1.4):          # was lines 48–57
    ...
def expand(node, state):                      # was lines 58–63
    ...
def rollout(state):                           # was lines 64–66
    ...
def backpropagate(node, reward):              # was lines 67–70
    ...

def uct_search(root_state, iterations=2000):
    root = Node(root_state)
    for _ in range(iterations):
        with accounting.phase("iteration"):
            with accounting.phase("select"):
                node, state = select(root, root_state)
            with accounting.phase("expand"):
                node, state = expand(node, state)
            with accounting.phase("rollout"):
                reward = rollout(state)
            with accounting.phase("backpropagate"):
                backpropagate(node, reward)
    return max(root.children, key=lambda n: n.visits)
```

There is no `count("simulations")` here on purpose. One simulation is one entry into `iteration`, and the report counts phase entries by itself — see [Which counters you do not have to write](#which-counters-you-do-not-have-to-write) below.

The `with` blocks are doing two jobs at once. As *code* they are a table of contents: the body of `uct_search` is now four lines that name the algorithm's four phases in order, and the four `while` loops that used to be inlined have become four functions you can read, test and optimise one at a time. As *instrumentation* they survive the refactor — a line profile is attached to line numbers and is invalidated by the first edit, whereas `phase("select")` still means selection after you rewrite the loop inside it.

### The phases nest, and the nesting is the extra information

A `with` block can go inside another `with` block, and the report follows the nesting. Put one around the whole iteration and two more inside `select`, where the line profile said the time actually was:

```python
def select(root, root_state, c=1.4):
    node, state = root, root_state
    depth = 0
    while not node.untried and node.children:
        with accounting.phase("score_children"):
            best, best_score = None, -1e30
            for child in node.children:
                score = child.value / child.visits + c * math.sqrt(
                    math.log(node.visits) / child.visits
                )
                if score > best_score:
                    best, best_score = child, score
            accounting.count("children_scored", len(node.children))
        with accounting.phase("descend"):
            node = best
            state = state.play(node.state.board[-1])
        depth += 1
    accounting.count("tree_depth", depth)
    return node, state
```

The two `count()` calls are not decoration, and they are not there to save you a `print`. A phase measures *how long something took*; a counter measures *how much work it did*, and neither number means much without the other. `accounting.count(name, n)` adds `n` to a counter on whichever phase is innermost at that moment — so `children_scored` lands on `score_children` because it is inside that `with` block, and `tree_depth` lands on `select` because the descent loop has ended by the time it runs. Placement is attribution: move a `count()` one line up or down across a `with` boundary and it will be divided by a different phase's wall time.

That division is the whole point. The report has 28.7 ms of `score_children` and 50,600 children scored, and prints the quotient — **567 ns each** — because that is the figure that survives a change of workload. Double the iteration count and the 28.7 ms doubles; the 567 ns does not. Without the counter the same phase reads only "28.7 ms, 6,325 calls, p50 4.5 µs", which conflates *the pass got slower* with *the pass had more children to score*. Those two call for opposite fixes — optimise the inner arithmetic, or cut the branching factor — and a timing-only report cannot tell you which one happened.

`tree_depth` is doing a second job that timings cannot do at all: asserting the algorithm's shape. Its row reads `6,325 … 0..4`, and both halves are claims about MCTS rather than about speed. The total is the summed depth over 2,000 descents — 3.16 plies on average — and the `0..4` is the per-call range, so the search never got deeper than four. A tree that shallow after 2,000 iterations means the UCT constant is exploring too widely, and no amount of making `score_children` faster will fix that. Likewise `children_scored always 8` says the branching factor never varied, so an unexpected `1..8` after a refactor would be a bug in move generation, showing up in the profile before it ever shows up in the win rate.

Drop the two lines and the report still renders — `count()` is optional, and the phase rows are unchanged. What you lose is every derived figure attached to them:

| With the counters | Without them |
|---|---|
| `children_scored 50,600 … 567 ns/ea  always 8` | nothing — the row disappears |
| Cost per unit of work, comparable across runs | wall time per phase, comparable only against itself |
| Branching factor and tree depth as measured invariants | invariants you assert in a test, or not at all |
| "the pass got slower" vs "the pass got bigger" | one number that mixes both |

The cost of keeping them is 384 ns per call — roughly a fifteenth of a phase entry, which is why the placement rule for counters is looser than the one for `with` blocks. A `count()` in an inner loop is usually affordable where a `phase()` in the same place would not be; when you have already decided to open a phase, counting the work it did is nearly free by comparison.

### Which counters you do not have to write

One kind of counter is already redundant: the one that counts *entries*. Every phase records how many times it was entered — that is the `entries` column, and it is there whether or not you ask, at no cost, because a phase has to increment something on the way out regardless. So `accounting.count("nodes_created")` inside `expand` writes a number the report was going to print anyway. Delete it and read `expand`'s `entries` instead; the `always 1` finding was never telling you anything the entry count was not.

What cannot be automatic is the *amount*. `children_scored` is `len(node.children)` and `tree_depth` is however deep that descent went — numbers that live in your data, not in the shape of your code, and nothing the library can inspect will produce them. That is the whole division:

| Question | Who answers it |
|---|---|
| How many times did this phase run? | the `entries` column, automatically |
| How long did each entry take? | `self`, `p50`, `p99`, automatically |
| How much work happened inside an entry? | `count()`, and only `count()` |
| Was that amount steady or bursty? | `count()`, via `always 8` / `1..5` |

The rule that falls out: **write a `count()` only when the number varies.** A counter whose amount is always 1 is an entry count you paid 384 ns for. A counter whose amount comes from the data is the only way that number will ever reach the report.

```
ACTOR  (1 process, imbalance 1.00)
  % of phase wall time at the first branching level
──────────────────────────────────────────────────────────────
select                         60.1%       111.2ms
rollout                         6.4%        11.8ms
expand                          5.0%         9.3ms
backpropagate                   2.6%         4.8ms
Other                          25.8%        47.7ms

DOMINANT PHASES          entries        self    wait       p50       p99
iteration/select           2,000      76.8ms      0%    52.7us    80.1us
    + tree_depth                6,325    56,882.3/s   17.6us/ea  0..4
iteration                  2,000      47.7ms      0%    90.1us   114.5us
select/score_children      6,325      28.7ms      0%     4.5us     5.6us
    + children_scored          50,600 1,763,122.8/s    567ns/ea  always 8
iteration/rollout          2,000      11.8ms      0%     5.8us    11.1us
    + rollout_plies             3,675   311,062.0/s    3.2us/ea  1..5
iteration/expand           2,000       9.3ms      0%     4.4us    10.4us
select/descend             6,325       5.7ms      0%     853ns     3.4us

ITERATIONS  (2000 entries)
  mean     92.4us   p50     90.1us   p95    108.7us   p99    114.5us
```

Six things are on this page that the line profile could not express at all:

**1. Wall versus self time splits a phase from its parts.** `select` has 111.2 ms of wall time but 76.8 ms of *self* time; its two sub-phases account for 28.7 + 5.7 = 34.4 ms, and 111.2 − 34.4 = 76.8 exactly. The line profile gives one number per line and leaves you to add them up; the tree does the subtraction and tells you how much of a phase is *not* in anything you named.

**2. `Other` at 25.8% is a measurement, not a gap.** That is `iteration` wall time minus its four children — real work inside the loop that no `with` block claims. A line profile shows unattributed time only as rows you forgot to read; here it is a labelled line item that says *go name this*.

**3. Counters give per-phase distributions, not just totals.** `tree_depth 0..4` says the search never went deeper than four plies. `children_scored always 8` says every scoring pass saw exactly eight children — the branching factor never varied. `rollout_plies 1..5` says a playout is one to five moves. A `Hits` column can tell you a line ran 49,200 times; only a counter can tell you that it ran 25.3 times per iteration and that the number was *always eight per pass*.

**4. Entry counts prove the algorithm's shape, and cost nothing to get.** The `entries` column is the number of times each phase was entered, which the layer records for every phase whether or not you ask. `score_children` ran 6,325 times for 2,000 iterations — 3.16 descents per iteration, which is the mean tree depth, independently derived and never counted by hand. `expand` ran exactly 2,000 times: expansion adds one node per iteration, as the algorithm requires, and a refactor that broke that invariant would show up here as a number that stopped matching `iteration`.

**5. Cost per unit of work, not per line.** `children_scored` at **567 ns each** is the number to optimise against, and it is stable across runs and problem sizes in a way that "line 52 took 36 ms" is not. Hoisting `math.log(node.visits)` out of the sibling loop moves this figure directly.

**6. Percentiles show the spread.** `iteration` is 90.1 µs at p50 and 114.5 µs at p99 — a tight distribution, so the mean is honest here. A phase whose p99 is twenty times its p50 has a tail that an average would have hidden completely, which is the usual reason a queue backs up.

### What the nesting costs

Phases are cheap but not free, and the bill scales with *entries*, not with lines of code:

| Version | Phase entries | Runtime |
|---|---|---|
| No instrumentation | 0 | 38 ms |
| Four phases per iteration | 8,000 | 88 ms |
| Nested, with counters | 22,650 | 196 ms |

Both enabled rows work out to ~6.3 µs per phase entry on this machine (the layer's own benchmark reports 5.4 µs for `phase()` with `measure_cpu=True`). The nested version is slower not because it is more deeply nested but because `score_children` sits *inside* the descent loop and fires 6,325 times instead of 2,000.

That is the rule for placing a `with` block: **put phases where the iteration count is bounded by your loop, not by your data.** Wrap the four phases permanently — 8,000 entries against a 2,000-iteration search is noise at production scale. Push sub-phases into the inner loop while you are actively investigating it, read the distribution, then take them out again. With no profiler installed the same calls cost ~285 ns and record nothing, so they are safe to leave in library code either way.

### The two tools disagree, and the disagreement is the lesson

Selection is 82% under line tracing and 53% under accounting; the same workload runs 190 ms traced and 35 ms untraced. Line tracing bills per *line executed*, so it inflates whatever is line-dense — and selection's six-line inner loop over 49,200 children is the most line-dense code in the function. Rollout, which spends its time inside `state.play()` and `random.choice()`, is barely touched by the tracer and so looks smaller than it is.

Use each tool for the question it answers. The line profile told you *where the phase boundaries are* and *which line inside a phase to fix* — line 52, unambiguously. The accounting report tells you *what the phases actually cost* once nobody is watching every line, which is the number to take into a scaling decision. Getting the boundaries from the cheap tool is not possible; getting the true shares from the expensive one is not either.

## Worked example: finding an I/O bottleneck

Phases carry an `io=True` flag that reads the process byte counters at the region's own entry and exit, so its bytes are attributed **exactly** rather than inferred from a 1 Hz sampler. That turns a vague "the copy step is slow" into a number you can act on.

Here is the case every data pipeline meets: move 4,000 records of 4 KiB each. The obvious version copies one file per record; the batched version concatenates them into shards of 250 and writes each shard once. **The bytes are identical — only the number of syscalls differs.**

```python
import shutil
from lineprofiler import accounting

def write_per_item(src_dir, dst_dir):
    """The obvious version: copy each record as its own file."""
    for path in sorted(src_dir.iterdir()):
        with accounting.phase("copy_one", io=True):
            shutil.copyfile(path, dst_dir / path.name)
        accounting.count("records", 1)

def write_batched(src_dir, dst_dir, batch=250):
    """The batched version: concatenate into shards, one write per shard."""
    paths = sorted(src_dir.iterdir())
    for shard_no in range(0, len(paths), batch):
        chunk = paths[shard_no:shard_no + batch]
        with accounting.phase("read_batch", io=True):
            blob = b"".join(p.read_bytes() for p in chunk)
        with accounting.phase("write_batch", io=True):
            (dst_dir / f"shard_{shard_no:05d}.bin").write_bytes(blob)
        accounting.count("records", len(chunk))
```

The table below is in the exact format `lineprofiler report profile/` prints — the section headers, column layout and nesting all come straight from the renderer, though the specific timings are illustrative rather than captured from one pinned run. To produce your own: install a `Profiler` with a `run_dir` and a `role` (the report's `COPIER` header is `role.upper()`), wrap each variant in its own top-level phase — `accounting.phase("copy_one", ...)` inside `write_per_item` then nests under it as `per_item/copy_one`, which is why the row below has that path — close the profiler, then point the `report` subcommand at the run directory:

```python
from lineprofiler.accounting import Profiler

profiler = Profiler(run_dir="profile", role="copier", install=True)
with accounting.phase("per_item"):
    write_per_item(src_dir, dst_dir)
with accounting.phase("batched"):
    write_batched(src_dir, dst_dir)
profiler.close()
```

```
lineprofiler report profile/
```

```
COPIER  (1 process, imbalance 1.00)
  % of phase wall time at the first branching level
──────────────────────────────────────────────────────────────
per_item                       89.4%       712.0ms
batched                        10.6%        84.1ms

DOMINANT PHASES          entries        self    wait       p50       p99
per_item/copy_one          4,000     436.2ms      0%   108.5us   143.2us
per_item                       1     275.7ms      1%   704.6ms   737.5ms
    + records                   4,000     5,618.1/s  178.0us/ea  always 1
batched/read_batch            16      54.7ms      1%     3.3ms     4.6ms
batched                        1      19.1ms      1%    88.1ms    92.2ms
    + records                   4,000    47,588.8/s   21.0us/ea  always 250
batched/write_batch           16      10.3ms      1%   627.3us   715.7us

I/O BY PHASE (measured exactly)
──────────────────────────────────────────────────────────────
  per_item/copy_one         r        0 B   w    15.6 MB     35.8 MB/s
                            + 16.1 MB read from page cache
  batched/read_batch        r        0 B   w        0 B         0 B/s
                            + 15.6 MB read from page cache
  batched/write_batch       r        0 B   w    15.6 MB      1.5 GB/s
```

**Batching is 8.5× faster to move exactly the same data** — 712.0 ms against 84.1 ms. The report shows *why*, in four numbers that a wall-clock timer alone would not give you:

**1. The bytes are identical, so the bytes are not the problem.** Both paths write 15.6 MB (16,384,000 bytes, exactly). When the payload is constant and the time is not, the cost is in the *per-operation* overhead — open, close, allocate an inode, update a directory — not in the data. This is the single most useful thing `io=True` tells you, because it eliminates the explanation everyone reaches for first.

**2. Write throughput differs by 42×: 35.8 MB/s versus 1.5 GB/s.** Same disk, same bytes, same second. A rate this far below the device's capability is the signature of a workload that is syscall-bound rather than bandwidth-bound. The batched figure is what the hardware can actually do; the per-item figure is what you get after paying 4,000 file creations for it.

**3. The counter converts it to a unit price.** `records` reads **178.0 µs each** per-item against **21.0 µs each** batched. That is the number to carry into a capacity estimate: at 178 µs a record, a 10-million-record shard takes half an hour; at 21 µs it takes three and a half minutes. Totals do not scale in your head — unit costs do.

**4. `self` time exposes the overhead that is not I/O at all.** `per_item` has 712.0 ms wall but only 436.2 ms in `copy_one`; the remaining **275.7 ms** is the loop itself — `iterdir`, sorting, building 4,000 destination paths. Even if the writes were free, that part would remain. The batched loop's equivalent remainder is 19.1 ms, because it runs 16 times instead of 4,000.

Note also what the `r` column says: **0 B read from disk** in every phase, with the reads appearing on the `read from page cache` line instead. The source files were written moments earlier and were still in RAM. A run that looks I/O-free by disk bytes alone can still be loader-bound — the cache line is what tells you the reads really happened, and it is why the report prints both layers rather than one.

### The rule this measures

**Batch until the per-operation cost is small next to the payload.** A 4 KiB write costs roughly the same in syscall overhead as a 1 MiB write, so the fixed cost is 100% of a small operation's time and 0.4% of a large one's. The same effect is why MPI-IO codes prefer large contiguous collective writes over many small independent ones, and why parallel filesystems punish small-file workloads hardest: the metadata server, not the bandwidth, is the queue you are standing in.

Wrap the candidate regions in `io=True`, read the `MB/s` column, and compare it to what the device should do. A large gap means the fix is a bigger batch, not a faster disk.

> `io=True` needs `psutil` (`pip install with-line-profiler[resources]`); without it the phase
> still times normally and the byte columns are omitted rather than reported as zero. It reads
> `/proc` at each end — tens of microseconds — so it belongs on coarse regions that really
> touch the disk, never on an inner loop.

## What it is built on, and what it plugs into

The core of this package depends on **nothing**. Both tools are built directly on interpreter and OS primitives, which is why `pip install with-line-profiler` pulls in no third-party packages at all (except `tomli`, and only on Python 3.10):

| Built on | Used for |
|---|---|
| `sys.monitoring` (3.12+), `sys.settrace` below | the line profiler's per-line events |
| `time.perf_counter_ns` | wall-clock timing everywhere |
| `time.thread_time_ns` | CPU time, so `wait%` = wall − CPU |
| `/proc` (Linux) | per-process I/O byte counters |

Everything below is **optional**. Each is imported lazily behind a capability check, so a missing one silently disables one block of the report instead of raising — install only what answers a question you actually have.

### The one-line version

```
pip install with-line-profiler[all]     # psutil + nvidia-ml-py + viztracer
pip install torch                       # separately — see below
```

### What each one buys you

| Package | Install | What it adds | Reach for it when |
|---|---|---|---|
| **[psutil](https://github.com/giampaolo/psutil)** | `with-line-profiler[resources]` | The `RESOURCES` and `I/O` blocks: RSS, CPU-cores-used, and per-process read/write bytes split into disk vs page cache. | Almost always. This is the one to install first — without it the report cannot tell you what the run *cost*, only where the time went. |
| **[nvidia-ml-py](https://pypi.org/project/nvidia-ml-py/)** | `with-line-profiler[gpu]` | The `GPU` block: per-device utilisation, split into your run's share (`this run`) and the whole device (`busy`), sampled at 1 Hz. | You are on a GPU box and need to know whether the GPU is the constraint — especially on a *shared* node, where "busy" includes other tenants. |
| **[torch](https://pytorch.org/)** | `pip install torch` (never a dependency of this package) | VRAM allocated/reserved; `phase(sync=True)`; `annotate=True`; the `backend="torch"` window. | You are training. The `sync=True` and `annotate=True` features are the ones people miss — see the two sections below. |
| **[VizTracer](https://github.com/gaogaotiantian/viztracer)** | `with-line-profiler[viztracer]` | The `backend="viztracer"` window: a full per-call timeline for a bounded slice of the run. | A phase is slow and you do not know which *function* inside it is responsible. |

### The two heavy backends

These are the SOTA tracers this package deliberately does **not** reimplement. Instead it starts one of them for a **bounded window**, expressed in entries of a phase you name — so you get full-fidelity detail for ten iterations out of a twelve-hour run, rather than a trace file that would be terabytes:

```python
Profiler(run_dir="profile", role="actor",
         backend="torch",            # or "viztracer"
         backend_window=(100, 110),  # entries 100–110 of...
         window_phase="iteration")   # ...the phase named "iteration"
```

| Backend | Produces | Open it with | Good for |
|---|---|---|---|
| `backend="torch"` | `backend/torch_trace.json` (Kineto/Chrome trace) | `chrome://tracing`, [Perfetto](https://ui.perfetto.dev) | CUDA kernel timings, GPU-vs-CPU attribution, operator-level cost — the things the 1 Hz GPU block *cannot* tell you. |
| `backend="viztracer"` | `backend/viztracer.json` | `vizviewer`, Perfetto | Every Python call in the window on one timeline, at full fidelity. |

**Only one backend can be active at a time.** VizTracer, cProfile and `line_profiler` all compete for the interpreter's single trace hook, and `torch.profiler` distorts timing enough that combining it with another sampler makes both meaningless — so `backend` is one enum value, not a set of flags.

### Feeding your phase names to external tools

`annotate=True` is the integration worth knowing about. It wraps every phase in `torch.cuda.nvtx.range_push/pop` (falling back to the standalone [`nvtx`](https://pypi.org/project/nvtx/) package) **and** `torch.profiler.record_function`:

```python
Profiler(run_dir="profile", role="actor", annotate=True)
```

Now an externally started [`nsys profile`](https://developer.nvidia.com/nsight-systems) or Kineto capture shows **your** phase names — `select`, `rollout`, `backpropagate` — as labelled ranges next to the CUDA kernels, instead of anonymous Python frames. This package never launches `nsys` itself; it just makes sure your semantics survive into whatever profiler you point at the process.

### How to tell what is missing

A resource that was never measured is never rendered as zero — the report omits the block and says why. Compare the two worked examples above: the MCTS run prints an empty `RESOURCES` section because `psutil` was not installed for it, while the I/O run — which needs `psutil` for `io=True` — prints CPU, RAM and the exact per-phase byte columns. If a block you expected is absent, install the matching extra above and re-run.

The GPU block in the I/O example shows the same principle from the other side: it reports `busy 0.0%` (a real measurement — the device was idle) but `this run n/a` (NVML never attributed a sample to these pids, because no work of ours ran there). Those are different statements, and the report keeps them apart.

## Documentation

- [The line profiler](https://github.com/mathematiger/withlineprofiler/blob/main/docs/line-profiler.md) — the `with` block, `start_profiling()`, and what it does not do
- [The accounting layer](https://github.com/mathematiger/withlineprofiler/blob/main/docs/accounting.md) — phases, counters, and instrumenting without threading an argument
- [Accounting recipes](https://github.com/mathematiger/withlineprofiler/blob/main/docs/accounting-recipes.md) — reading the report, I/O and GPU bottlenecks, overhead budgets, exporting to W&B
- [Multiple processes and nodes](https://github.com/mathematiger/withlineprofiler/blob/main/docs/multiprocess.md) — Slurm, forking, preemption, heavy backends
- [HTML reports](https://github.com/mathematiger/withlineprofiler/blob/main/docs/html-reports.md) — the icicle chart, the trace timeline, the annotated source view, and the embedded data block
- [Configuration](https://github.com/mathematiger/withlineprofiler/blob/main/docs/configuration.md) — environment variables, `[tool.lineprofiler]`, optional dependencies
- [Comparison with other profilers](https://github.com/mathematiger/withlineprofiler/blob/main/docs/comparison.md) — `line_profiler`, py-spy, Scalene, VizTracer, and when to use those instead

## Python support

3.10 and newer. On 3.12+ the line profiler uses `sys.monitoring`, so it can run alongside coverage.py, pdb and other tracing tools; below that it falls back to `sys.settrace`, which is a single global hook and cannot. `tomli` is required only on 3.10, where `tomllib` is not yet in the standard library.

## Licence

MIT

The claude.md is partially created from https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md
