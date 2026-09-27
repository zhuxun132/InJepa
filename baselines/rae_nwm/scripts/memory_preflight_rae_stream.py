#!/usr/bin/env python3
"""CLI wrapper for the bounded official RAE-stream memory preflight.

The implementation lives in :mod:`rae_stream.memory_preflight`; this file is
only a stable repository entry point so operators do not need to invoke a
package module by hand.  It never changes the upstream trainer.
"""

from __future__ import annotations

from pathlib import Path
import sys


def _bootstrap() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_bootstrap()

from rae_stream.memory_preflight import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
