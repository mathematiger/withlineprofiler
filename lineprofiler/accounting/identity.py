"""Which machine and which rank a worker was, resolved from the batch system.

A worker file used to record only a pid. That is enough on one node and useless across
several: pid namespaces are per-node, so ranks on different nodes routinely share a pid, and
the first question anyone asks about an imbalanced multi-node job — *which node is slow?* —
could not be answered at all.

Nothing here imports a scheduler library. Every scheduler already publishes what we need in
the environment, and reading it costs nothing on a machine that has none.
"""

from __future__ import annotations

import os
import socket

RANK_VARIABLES = ("SLURM_PROCID", "RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "MV2_COMM_WORLD_RANK")
"""Global rank, in preference order: Slurm, torch.distributed, Open MPI, MPICH, MVAPICH."""

LOCAL_RANK_VARIABLES = ("SLURM_LOCALID", "LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK")
"""Rank within a node — which GPU this worker most likely owns."""

WORLD_SIZE_VARIABLES = ("SLURM_NTASKS", "WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE")
"""How many workers the launcher intended, which is how a missing one becomes visible."""

JOB_VARIABLES = ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "FLUX_JOB_ID")
"""Batch job identifier, so a run directory can be correlated with its scheduler record."""


def describe() -> dict[str, object]:
    """Return this process's placement: host, ranks, world size and job id.

    Absent values are omitted rather than recorded as ``None``, so a single-machine run
    carries only a hostname and the report stays quiet about ranks nobody assigned.

    Test specifically:
        - a bare environment yields only ``host``
        - Slurm variables win over torch's when both are set
        - a non-numeric rank is ignored rather than raising
    """
    placement: dict[str, object] = {"host": hostname()}
    for key, names in (
        ("rank", RANK_VARIABLES),
        ("local_rank", LOCAL_RANK_VARIABLES),
        ("world_size", WORLD_SIZE_VARIABLES),
    ):
        value = _first_int(names)
        if value is not None:
            placement[key] = value
    job = _first_str(JOB_VARIABLES)
    if job is not None:
        placement["job_id"] = job
    return placement


def hostname() -> str:
    """The node's name, short form — the fully qualified one makes report columns unreadable."""
    return socket.gethostname().split(".")[0]


def _first_int(names: tuple[str, ...]) -> int | None:
    """First variable that is set and parses as an integer."""
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return None


def _first_str(names: tuple[str, ...]) -> str | None:
    for name in names:
        raw = os.environ.get(name)
        if raw:
            return raw
    return None
