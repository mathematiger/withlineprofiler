"""Which code a run measured, resolved from git.

A trace records the environment thoroughly — host, ranks, job id, python version — and says
nothing about the source it ran. That gap silently invalidates analysis: a conclusion drawn
from a profile of the *committed* code, read against a working tree that has since fixed the
very constraint the profile found, is a claim about a program that no longer exists. The
failure is the same one ``sample_stride`` exists to prevent, one level up — a measurement that
cannot be told apart from a measurement of something else.

Nothing here imports a git library. One subprocess at startup, on the process that writes the
run metadata, and silence on any machine where it does not apply.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

_TIMEOUT_S = 2.0
"""Ceiling on each git call. A slow or wedged filesystem must not delay a training run's
startup; provenance is worth having and never worth blocking on."""

_DIFF_SHA_CHARS = 6
"""Prefix of the working-tree diff hash. Enough to tell two dirty runs apart, which is all it
is for — it identifies a diff, it does not reconstruct one."""


def describe_source(cwd: str | Path | None = None) -> dict[str, object]:
    """Return the git revision this process is running, or ``{}`` when that is unknowable.

    Empty on any failure — not a repository, no ``git`` binary, a timeout, a broken index.
    Provenance is a courtesy the report prints when it can; a run must never fail, stall or
    warn because it could not be established.

    Test specifically:
        - a directory that is not a repository yields ``{}``
        - a clean checkout reports a commit and no dirty marker
        - a modified working tree is marked dirty, with a file count and a diff hash
        - two different working-tree diffs produce different hashes
        - a missing ``git`` binary yields ``{}`` rather than raising
    """
    root = Path(cwd) if cwd is not None else Path.cwd()
    commit = _git(root, "rev-parse", "HEAD")
    if commit is None:
        return {}

    source: dict[str, object] = {"commit": commit}
    dirty_files = _dirty_file_count(root)
    if dirty_files:
        source["dirty_files"] = dirty_files
        diff_sha = _diff_sha(root)
        if diff_sha is not None:
            source["diff_sha"] = diff_sha
    return source


def format_source(source: dict[str, object]) -> str:
    """Render :func:`describe_source` output as one line, or ``""`` when there is nothing.

    Shared by every renderer so the three places that print run identity cannot describe the
    same run differently.
    """
    commit = source.get("commit")
    if not commit:
        return ""
    short = str(commit)[:8]
    dirty_files = source.get("dirty_files")
    if not dirty_files:
        return f"Source {short}"
    diff_sha = source.get("diff_sha")
    suffix = f", diff sha {diff_sha}" if diff_sha else ""
    return f"Source {short} (+dirty: {dirty_files} files{suffix})"


def source_of(metadata: dict[str, object]) -> str:
    """Render the source line straight from a run's metadata, or ``""`` when absent.

    The three renderers that print run identity all reach for it the same way, and the shape
    check belongs with the writer rather than repeated in each of them.
    """
    source = metadata.get("source")
    if not isinstance(source, dict):
        return ""
    return format_source(source)


def _dirty_file_count(root: Path) -> int:
    """How many tracked files differ from ``HEAD``, or ``0`` when that cannot be determined.

    Counts the porcelain status rather than trusting a boolean, because "how dirty" is what
    tells a reader whether the profiled code was a near-match for the tree they are reading
    or a different program.
    """
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    if not status:
        return 0
    return len(status.splitlines())


def _diff_sha(root: Path) -> str | None:
    """A short hash of the working-tree diff, so two dirty runs are distinguishable.

    Without it every dirty run of the same commit looks identical in the report, which is the
    case where confusing two runs is most likely and most costly.
    """
    diff = _git(root, "diff", "HEAD")
    if diff is None:
        return None
    digest = hashlib.sha256(diff.encode("utf-8", "replace")).hexdigest()
    return digest[:_DIFF_SHA_CHARS]


def _git(root: Path, *arguments: str) -> str | None:
    """Run one git command in ``root``, returning its stdout or ``None`` on any failure.

    Every foreseeable failure is ordinary here — no repository, no binary, a timeout — so all
    of them collapse to the same "unknowable" answer rather than being distinguished. The
    broad catch is deliberate: this must not be able to break a run.
    """
    try:
        finished = subprocess.run(  # noqa: S603
            ["git", *arguments],  # noqa: S607
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if finished.returncode != 0:
        return None
    return finished.stdout.strip()
