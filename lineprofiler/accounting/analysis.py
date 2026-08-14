"""Derived quantities: per-phase I/O, memory trends, GPU utilisation, activity sparklines.

Nothing here reads the machine. It works only on samples already collected, so it is pure
and cheap to test.

Byte counters from the OS are cumulative, so per-phase I/O is the difference between
consecutive samples attributed to the phase that was open at the *start* of the interval.
That is an approximation with a resolution of one sample interval: a burst of reads shorter
than the interval lands on whichever phase happened to be open. It is honest for finding
which phase does the bulk of the I/O, and wrong for attributing individual operations —
that needs eBPF, which is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lineprofiler.accounting.sampler import Sample

_SPARK_CHARS = " ▁▂▃▄▅▆▇█"


NO_PHASE = "(no phase open)"
"""Label for bytes moved while no phase was on the stack — an admission, not a finding."""


@dataclass(slots=True)
class IoTotals:
    """Bytes read and written, and the wall time over which that happened.

    ``*_bytes`` is block-layer traffic and ``*_chars`` is syscall-level, so the difference
    between them is what the page cache served without touching the device.
    """

    read_bytes: int = 0
    write_bytes: int = 0
    read_chars: int = 0
    write_chars: int = 0
    seconds: float = 0.0

    @property
    def read_rate(self) -> float:
        return self.read_bytes / self.seconds if self.seconds else 0.0

    @property
    def write_rate(self) -> float:
        return self.write_bytes / self.seconds if self.seconds else 0.0

    @property
    def cached_read_bytes(self) -> int:
        """Bytes the program read that never came off the disk."""
        return max(0, self.read_chars - self.read_bytes)


@dataclass(slots=True)
class MemoryTrend:
    """Resident-memory behaviour over the run.

    ``slope_bytes_per_s`` comes from a least-squares fit. A sustained positive slope is the
    leak indicator; a flat or noisy slope is not.
    """

    first_rss: int = 0
    last_rss: int = 0
    peak_rss: int = 0
    slope_bytes_per_s: float = 0.0

    @property
    def growth_bytes(self) -> int:
        return self.last_rss - self.first_rss


@dataclass(slots=True)
class GpuDevice:
    """One device's utilisation over the run, split by who caused it.

    ``busy_mean`` is NVML's whole-device percentage and therefore includes kernels from
    processes outside this run. ``ours_mean`` sums the per-process figure across this run's
    workers, so on an exclusive node the two converge and on a shared one the gap is other
    tenants. ``ours_mean`` can exceed ``busy_mean`` slightly: the two come from different
    NVML sampling windows.
    """

    index: int
    busy_mean: float = -1.0
    ours_mean: float = -1.0


@dataclass(slots=True)
class SampleAnalysis:
    """Everything the report derives from one run's resource samples."""

    totals: IoTotals = field(default_factory=IoTotals)
    io_by_phase: dict[str, IoTotals] = field(default_factory=dict)
    memory: MemoryTrend = field(default_factory=MemoryTrend)
    memory_by_phase: dict[str, MemoryTrend] = field(default_factory=dict)
    peak_cuda_alloc: int = 0
    peak_cuda_reserved: int = 0
    gpu_util_mean: float = -1.0
    gpu_devices: list[GpuDevice] = field(default_factory=list)
    read_series: list[float] = field(default_factory=list)
    write_series: list[float] = field(default_factory=list)
    io_gap_intervals: int = 0
    """Intervals whose bytes were discarded because a counter read failed at one end. Any
    non-zero value means the I/O totals below are a floor, not a measurement."""
    io_intervals: int = 0

    @property
    def has_samples(self) -> bool:
        return self.memory.last_rss > 0 or self.totals.read_bytes > 0

    @property
    def unattributed_read_share(self) -> float:
        """Share of read traffic the sampler could not pin to any phase, in ``0.0..1.0``.

        High here means the sample interval was too coarse for this run, not that the root
        did the reading. Compare against the exactly-measured ``io=True`` phases instead.
        """
        loose = self.io_by_phase.get(NO_PHASE)
        if loose is None:
            return 0.0
        return _share(
            loose.read_bytes or loose.read_chars,
            self.totals.read_bytes or self.totals.read_chars,
        )

    @property
    def unattributed_write_share(self) -> float:
        """Share of write traffic the sampler could not pin to any phase, in ``0.0..1.0``."""
        loose = self.io_by_phase.get(NO_PHASE)
        if loose is None:
            return 0.0
        return _share(
            loose.write_bytes or loose.write_chars,
            self.totals.write_bytes or self.totals.write_chars,
        )


@dataclass(slots=True)
class _Interval:
    """The change in each counter between two consecutive samples of one process."""

    start: float
    seconds: float
    phase: str
    read: int
    write: int
    read_chars: int
    write_chars: int
    measured: bool = True


def analyse(samples: list[Sample], series_width: int = 48) -> SampleAnalysis:
    """Derive I/O, memory and GPU summaries from **one process's** samples.

    Test specifically:
        - a workload writing a known number of bytes reports within a band, not exactly
        - bytes are attributed to the phase that was open during the interval
        - a growing list produces a positive RSS slope; a steady workload does not
        - fewer than two samples produces empty output rather than a division by zero
    """
    return analyse_processes([samples], series_width)


def analyse_processes(
    per_process: list[list[Sample]],
    series_width: int = 48,
) -> SampleAnalysis:
    """Derive the same summaries across several processes, differencing each separately.

    The OS byte counters are per-process cumulative totals, so differencing samples that
    came from *different* processes is meaningless — it produces both inflated totals and
    attribution to whichever process happened to be sampled next. Each process is therefore
    reduced to intervals on its own, and only the intervals are pooled.

    Test specifically:
        - two processes each writing N bytes report 2N in total, not more
        - bytes land on the phase of the process that wrote them, not an interleaved one
        - one process with samples and one without still produces the first one's totals
        - two processes on one device sum their own GPU shares but average the device's busy
          percentage, which they both observed
    """
    analysis = SampleAnalysis()
    intervals: list[_Interval] = []
    trends: list[MemoryTrend] = []
    ordered_per_process: list[list[Sample]] = []

    for samples in per_process:
        ordered = sorted(samples, key=lambda s: s.t)
        ordered_per_process.append(ordered)
        if len(ordered) < 2:
            continue
        intervals.extend(_intervals_of(ordered))
        _accumulate_memory(analysis, ordered, trends)

    if not intervals and not trends:
        return analysis

    _fill_io_from_intervals(analysis, intervals)
    analysis.memory = _combine_trends(trends)
    _fill_gpu(analysis, ordered_per_process)
    analysis.read_series = _rate_series(intervals, "read", series_width)
    analysis.write_series = _rate_series(intervals, "write", series_width)
    return analysis


def sparkline(values: list[float]) -> str:
    """Render a series as one line of block characters, scaled to its own maximum.

    Makes a burst of I/O visible as a spike in time, which a single total never shows.
    """
    if not values:
        return ""
    peak = max(values)
    if peak <= 0:
        return _SPARK_CHARS[0] * len(values)
    step = len(_SPARK_CHARS) - 1
    return "".join(_SPARK_CHARS[min(step, int(value / peak * step + 0.5))] for value in values)


def format_bytes(value: float) -> str:
    """Render a byte count with a binary unit."""
    magnitude = float(abs(value))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if magnitude < 1024 or unit == "TB":
            sign = "-" if value < 0 else ""
            precision = 0 if unit == "B" else 1
            return f"{sign}{magnitude:.{precision}f} {unit}"
        magnitude /= 1024
    return f"{value:.0f} B"


def _intervals_of(ordered: list[Sample]) -> list[_Interval]:
    """Reduce one process's samples to the deltas between consecutive rows.

    Writes the profiler itself performed between the two rows are deducted, so its own
    sample and snapshot files never appear as the workload's I/O.

    An interval touching a row whose counters could not be read carries no bytes. The
    counters are cumulative, so differencing against a failed read would charge the process's
    entire lifetime traffic to one interval — hundreds of gigabytes materialising on whatever
    phase happened to be open. Losing one interval's bytes is the honest failure; the count
    of intervals dropped this way is reported so the loss is visible rather than assumed.
    """
    intervals = []
    for previous, current in zip(ordered, ordered[1:], strict=False):
        measured = previous.io_ok and current.io_ok
        blocks = max(0, current.self_write_bytes - previous.self_write_bytes)
        chars = max(0, current.self_write_chars - previous.self_write_chars)
        intervals.append(
            _Interval(
                start=previous.t,
                seconds=max(0.0, current.t - previous.t),
                phase=previous.phase or NO_PHASE,
                read=max(0, current.read_bytes - previous.read_bytes) if measured else 0,
                write=(
                    max(0, current.write_bytes - previous.write_bytes - blocks)
                    if measured
                    else 0
                ),
                read_chars=max(0, current.read_chars - previous.read_chars) if measured else 0,
                write_chars=(
                    max(0, current.write_chars - previous.write_chars - chars)
                    if measured
                    else 0
                ),
                measured=measured,
            ),
        )
    return intervals


def _fill_io_from_intervals(analysis: SampleAnalysis, intervals: list[_Interval]) -> None:
    """Sum intervals into run totals and per-phase totals."""
    analysis.io_intervals = len(intervals)
    analysis.io_gap_intervals = sum(1 for interval in intervals if not interval.measured)
    for interval in intervals:
        analysis.totals.read_bytes += interval.read
        analysis.totals.write_bytes += interval.write
        analysis.totals.read_chars += interval.read_chars
        analysis.totals.write_chars += interval.write_chars

        bucket = analysis.io_by_phase.setdefault(interval.phase, IoTotals())
        bucket.read_bytes += interval.read
        bucket.write_bytes += interval.write
        bucket.read_chars += interval.read_chars
        bucket.write_chars += interval.write_chars
        bucket.seconds += interval.seconds
    analysis.totals.seconds = _wall_span(intervals)


def _share(part: int, whole: int) -> float:
    """``part / whole``, or zero when there was nothing to divide."""
    return part / whole if whole > 0 else 0.0


def _wall_span(intervals: list[_Interval]) -> float:
    """Wall-clock span the intervals cover, so run-wide rates are not summed per process."""
    if not intervals:
        return 0.0
    start = min(interval.start for interval in intervals)
    end = max(interval.start + interval.seconds for interval in intervals)
    return max(0.0, end - start)


def _accumulate_memory(
    analysis: SampleAnalysis,
    ordered: list[Sample],
    trends: list[MemoryTrend],
) -> None:
    """Record one process's RSS trend, overall and per phase."""
    with_rss = [sample for sample in ordered if sample.rss]
    if not with_rss:
        return
    trends.append(_trend(with_rss))

    by_phase: dict[str, list[Sample]] = {}
    for sample in with_rss:
        by_phase.setdefault(sample.phase or "(root)", []).append(sample)
    for phase, group in by_phase.items():
        if len(group) < 2:
            continue
        existing = analysis.memory_by_phase.get(phase)
        analysis.memory_by_phase[phase] = (
            _trend(group) if existing is None else _combine_trends([existing, _trend(group)])
        )


def _combine_trends(trends: list[MemoryTrend]) -> MemoryTrend:
    """Sum per-process trends into a whole-run footprint.

    Summing is the right operation for "how much memory does this run hold": four workers
    at 1 GB each occupy 4 GB, and four workers each leaking 1 MB/s leak 4 MB/s together.
    """
    if not trends:
        return MemoryTrend()
    return MemoryTrend(
        first_rss=sum(trend.first_rss for trend in trends),
        last_rss=sum(trend.last_rss for trend in trends),
        peak_rss=sum(trend.peak_rss for trend in trends),
        slope_bytes_per_s=sum(trend.slope_bytes_per_s for trend in trends),
    )


def _trend(samples: list[Sample]) -> MemoryTrend:
    """Least-squares slope of RSS against time, plus first, last and peak."""
    trend = MemoryTrend(
        first_rss=samples[0].rss,
        last_rss=samples[-1].rss,
        peak_rss=max(sample.rss for sample in samples),
    )
    if len(samples) < 2:
        return trend

    origin = samples[0].t
    times = [sample.t - origin for sample in samples]
    mean_t = sum(times) / len(times)
    mean_rss = sum(sample.rss for sample in samples) / len(samples)
    variance = sum((t - mean_t) ** 2 for t in times)
    if variance <= 0:
        return trend
    covariance = sum(
        (t - mean_t) * (sample.rss - mean_rss) for t, sample in zip(times, samples, strict=False)
    )
    trend.slope_bytes_per_s = covariance / variance
    return trend


def _fill_gpu(analysis: SampleAnalysis, per_process: list[list[Sample]]) -> None:
    pooled = [sample for samples in per_process for sample in samples]
    analysis.peak_cuda_alloc = max((s.cuda_alloc for s in pooled), default=0)
    analysis.peak_cuda_reserved = max((s.cuda_reserved for s in pooled), default=0)
    utilisations = [s.gpu_util for s in pooled if s.gpu_util >= 0]
    if utilisations:
        analysis.gpu_util_mean = sum(utilisations) / len(utilisations)
    analysis.gpu_devices = _gpu_devices(per_process)


def _gpu_devices(per_process: list[list[Sample]]) -> list[GpuDevice]:
    """Per-device utilisation: the device's own busy share, and this run's share of it.

    The two are combined differently on purpose. Every process observes the same device, so
    pooling their whole-device readings averages the same quantity and stays a percentage.
    The per-process readings are disjoint — each is one worker's slice — so they are summed,
    which is what makes sixteen actors on one device add up to the load they actually place
    on it.
    """
    busy: dict[int, list[float]] = {}
    shares: dict[int, list[float]] = {}
    for samples in per_process:
        for index, value in _process_shares(samples).items():
            shares.setdefault(index, []).append(value)
        for sample in samples:
            for index, value in sample.gpu_utils.items():
                busy.setdefault(index, []).append(value)

    devices = []
    for index in sorted(set(busy) | set(shares)):
        devices.append(
            GpuDevice(
                index=index,
                busy_mean=_mean(busy.get(index, [])),
                ours_mean=sum(shares[index]) if index in shares else -1.0,
            ),
        )
    return devices


def _process_shares(samples: list[Sample]) -> dict[int, float]:
    """One process's mean SM utilisation per device, over the samples where it was readable.

    A window in which NVML attributed no kernels to this pid means the process did no GPU
    work then, so it is averaged in as a zero rather than skipped — dividing only by the
    windows that reported would turn a worker busy one second in ten into a busy worker. A
    device the process never got a single reading for is left out entirely: that is missing
    data, not idleness.
    """
    totals: dict[int, float] = {}
    readable: dict[int, int] = {}
    for sample in samples:
        for index in sample.gpu_utils:
            readable[index] = readable.get(index, 0) + 1
        for index, value in sample.gpu_proc_utils.items():
            totals[index] = totals.get(index, 0.0) + value
    return {index: total / max(1, readable.get(index, 0)) for index, total in totals.items()}


def _mean(values: list[float]) -> float:
    """Arithmetic mean, or ``-1.0`` for "never measured" — the convention for utilisation."""
    return sum(values) / len(values) if values else -1.0


def _rate_series(intervals: list[_Interval], attribute: str, width: int) -> list[float]:
    """Bucket the run's wall-clock span into ``width`` slots, returning bytes/second in each.

    Intervals from every process land in the same wall-clock buckets, so a burst shows up
    at the moment it happened regardless of which worker caused it.
    """
    if not intervals:
        return []
    origin = min(interval.start for interval in intervals)
    span = _wall_span(intervals)
    if span <= 0:
        return []

    totals = [0.0] * width
    for interval in intervals:
        index = min(width - 1, int((interval.start - origin) / span * width))
        totals[index] += getattr(interval, attribute)
    slot_seconds = span / width
    return [total / slot_seconds for total in totals]
