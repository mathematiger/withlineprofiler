"""Opt-in configuration, read once from an env var and an optional ``pyproject.toml`` table.

Both are off/empty by default: profiling only starts when ``LINEPROFILER_ENABLED`` is truthy,
and every file is in scope until a ``[tool.lineprofiler]`` table narrows it. Resolved once per
project root and cached, so this never runs on a hot path.
"""
from __future__ import annotations

import fnmatch
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10, where tomllib is not in the stdlib
    import tomli as tomllib

ENV_ENABLED = "LINEPROFILER_ENABLED"


@dataclass(frozen=True)
class ProfilerConfig:
    """Resolved opt-in settings: whether to profile, and which files/functions to include."""

    enabled: bool
    include: tuple[str, ...] = field(default_factory=tuple)
    exclude: tuple[str, ...] = field(default_factory=tuple)
    functions: tuple[str, ...] = field(default_factory=tuple)

    def allows_function(self, qualified_name: str) -> bool:
        """Whether ``qualified_name`` (e.g. a function's ``__qualname__``) may be profiled.

        With no ``functions`` patterns configured, every function is allowed.
        """
        if not self.functions:
            return True
        return any(fnmatch.fnmatch(qualified_name, pattern) for pattern in self.functions)

    def allows_path(self, path: str) -> bool:
        """Whether ``path`` may be profiled, given the ``include``/``exclude`` glob lists.

        ``exclude`` wins over ``include``. With no ``include`` patterns configured, every path
        is included (subject to ``exclude``).
        """
        if any(fnmatch.fnmatch(path, pattern) for pattern in self.exclude):
            return False
        if not self.include:
            return True
        return any(fnmatch.fnmatch(path, pattern) for pattern in self.include)


_UNSET: object = object()
_cache: dict[Path, ProfilerConfig | object] = {}


def get_config(start_path: str | Path | None = None) -> ProfilerConfig:
    """Resolve and cache the opt-in config for the project containing ``start_path``.

    ``start_path`` defaults to the current working directory. The project root is the nearest
    ancestor directory containing ``.git`` (falling back to ``start_path`` itself), matching
    how ``LineProfiler`` auto-detects its own ``project_folder``.
    """
    root = find_project_root(start_path if start_path is not None else Path.cwd())
    cached = _cache.get(root, _UNSET)
    if cached is not _UNSET:
        return cached  # type: ignore[return-value]

    config = _load_config(root)
    _cache[root] = config
    return config


def find_project_root(start_path: str | Path) -> Path:
    """Return the nearest ancestor of ``start_path`` containing ``.git``, else its directory.

    ``start_path`` may be a file (e.g. ``__file__`` of the calling module) or a directory.
    """
    p = Path(start_path).resolve()

    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent

    return p.parent if p.is_file() else p


def _load_config(root: Path) -> ProfilerConfig:
    enabled = _truthy(os.environ.get(ENV_ENABLED, ""))
    table = _read_tool_table(root / "pyproject.toml")
    return ProfilerConfig(
        enabled=enabled,
        include=_string_tuple(table.get("include", ())),
        exclude=_string_tuple(table.get("exclude", ())),
        functions=_string_tuple(table.get("functions", ())),
    )


def _string_tuple(value: object) -> tuple[str, ...]:
    """Coerce a TOML array-of-strings value to a tuple, ignoring a malformed entry."""
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _read_tool_table(pyproject_path: Path) -> dict[str, object]:
    """Return the ``[tool.lineprofiler]`` table, or ``{}`` if missing or unreadable."""
    try:
        with pyproject_path.open("rb") as f:
            document = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}

    tool_table = document.get("tool", {})
    if not isinstance(tool_table, dict):
        return {}
    lineprofiler_table = tool_table.get("lineprofiler", {})
    return lineprofiler_table if isinstance(lineprofiler_table, dict) else {}


def _truthy(value: str) -> bool:
    return bool(value.strip()) and value not in {"0", "false", "False"}
