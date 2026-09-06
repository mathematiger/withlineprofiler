"""The benchmark's workload, in its own module so an engine can discover it by import."""
from __future__ import annotations


def work(n: int) -> int:
    total = 0
    for i in range(n):
        total += i * i
    return total
