"""Fixed-size, mergeable duration histogram.

The accounting layer never stores raw durations: a 12-hour reinforcement-learning run
produces billions of phase entries, so per-event storage is not an option. Instead every
duration lands in one of a fixed number of log-spaced buckets. Memory per histogram is
therefore constant regardless of run length, and two histograms merge by summing buckets
element-wise — which is what makes quantiles survive both the periodic snapshot and the
cross-worker merge.
"""

from __future__ import annotations

BUCKET_COUNT = 512
"""Number of buckets. Covers 1 ns to roughly 2**63 ns at 8 buckets per octave."""

_SUB_BUCKET_BITS = 3
"""log2 of the number of sub-buckets per power-of-two octave (8 sub-buckets)."""

_LINEAR_LIMIT = 16
"""Durations below this are their own bucket, so tiny values stay exact."""

_FIRST_LOG_INDEX = 40
"""``bucket_index(_LINEAR_LIMIT)``. Indices between the linear range and this are unreachable."""


def bucket_index(duration_ns: int) -> int:
    """Return the bucket that ``duration_ns`` falls into.

    Pure integer arithmetic — no float conversion, no ``math.log`` — because this runs on
    every phase exit. Durations of zero or less collapse into bucket 0.

    Test specifically:
        - the mapping is monotone: ``a <= b`` implies ``bucket_index(a) <= bucket_index(b)``
        - values below ``_LINEAR_LIMIT`` map to themselves
        - no input produces an index outside ``range(BUCKET_COUNT)``
    """
    if duration_ns < _LINEAR_LIMIT:
        return duration_ns if duration_ns > 0 else 0
    octave = duration_ns.bit_length()
    return (octave << _SUB_BUCKET_BITS) | ((duration_ns >> (octave - 4)) & 7)


def bucket_lower_ns(index: int) -> int:
    """Return the inclusive lower bound of bucket ``index``, in nanoseconds.

    Indices 16..39 are unreachable — ``bucket_index`` jumps from 15 straight to 40 — so the
    value returned for them is meaningless but harmless.

    Test specifically:
        - ``bucket_lower_ns(bucket_index(d)) <= d`` for a wide range of ``d``
        - bounds are contiguous: the upper bound of one bucket is the lower bound of the next
    """
    if index < _FIRST_LOG_INDEX:
        return index
    octave = index >> _SUB_BUCKET_BITS
    return (8 | (index & 7)) << (octave - 4)


def bucket_upper_ns(index: int) -> int:
    """Return the exclusive upper bound of bucket ``index``, in nanoseconds."""
    if index < _LINEAR_LIMIT:
        return index + 1
    return bucket_lower_ns(index + 1)


class DurationHistogram:
    """Log-spaced bucket counts for a set of durations.

    Test specifically:
        - ``merge`` is associative and commutative over three or more instances
        - merging an empty histogram is the identity
        - quantiles are within one bucket width of the exact quantile on a known
          distribution (e.g. 10k samples from a lognormal)
        - ``to_sparse``/``from_sparse`` round-trips exactly
    """

    __slots__ = ("buckets", "count")

    def __init__(self) -> None:
        self.buckets: list[int] = [0] * BUCKET_COUNT
        self.count: int = 0

    def observe(self, duration_ns: int) -> None:
        """Record one duration. This is the hot path — keep it to two operations."""
        self.buckets[bucket_index(duration_ns)] += 1
        self.count += 1

    def merge(self, other: DurationHistogram) -> None:
        """Add ``other``'s counts into this histogram, in place."""
        if other.count == 0:
            return
        buckets = self.buckets
        for index, value in enumerate(other.buckets):
            if value:
                buckets[index] += value
        self.count += other.count

    def difference(self, baseline: DurationHistogram) -> DurationHistogram:
        """Return the counts accumulated since ``baseline`` was taken.

        Bucket-wise subtraction, which is exactly what makes quantiles derivable for an
        interval rather than only for the run so far: these are counts, so the difference of
        two cumulative histograms is the histogram of what happened in between.

        Clamped at zero per bucket. A baseline can legitimately exceed the current value when
        it was taken from a merge that included a thread which has since been folded, and a
        negative count would corrupt every quantile derived from it.
        """
        result = DurationHistogram()
        total = 0
        for index, value in enumerate(self.buckets):
            delta = value - baseline.buckets[index]
            if delta > 0:
                result.buckets[index] = delta
                total += delta
        result.count = total
        return result

    def quantile(self, q: float) -> float:
        """Return the estimated ``q``-quantile in nanoseconds (``q`` in [0, 1]).

        Interpolates linearly inside the containing bucket, so the error is bounded by the
        bucket width (about 9% at 8 sub-buckets per octave) rather than by the octave.
        """
        if self.count == 0:
            return 0.0
        target = q * self.count
        cumulative = 0
        for index, value in enumerate(self.buckets):
            if value == 0:
                continue
            cumulative += value
            if cumulative >= target:
                lower = bucket_lower_ns(index)
                upper = bucket_upper_ns(index)
                position = (target - (cumulative - value)) / value
                return lower + position * (upper - lower)
        return float(bucket_upper_ns(len(self.buckets) - 1))

    def to_sparse(self) -> dict[str, int]:
        """Return only the non-empty buckets, keyed by index, for serialisation.

        Almost every bucket is empty in practice, so this keeps a snapshot line small.
        """
        return {str(index): value for index, value in enumerate(self.buckets) if value}

    @classmethod
    def from_sparse(cls, sparse: dict[str, int]) -> DurationHistogram:
        """Rebuild a histogram from :meth:`to_sparse` output."""
        histogram = cls()
        for key, value in sparse.items():
            histogram.buckets[int(key)] = value
        histogram.count = sum(sparse.values())
        return histogram
