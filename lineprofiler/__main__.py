"""``python -m lineprofiler`` — the CLI, without needing the console script on ``PATH``.

The installed ``lineprofiler`` entry point is not always reachable: a virtualenv that was not
activated, a ``pip install --user`` whose scripts directory is not on ``PATH``, a batch job
that runs ``python`` by absolute path. The module form works wherever the package is
importable, which is exactly where the profiler that wrote the run was.

Deliberately no ``if __name__ == "__main__"`` guard: a ``__main__.py`` only ever runs as
``__main__``, so the guard would be dead code.
"""

import sys

from lineprofiler.accounting.cli import main

sys.exit(main())
