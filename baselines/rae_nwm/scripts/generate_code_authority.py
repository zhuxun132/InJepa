#!/usr/bin/env python3
"""Generate the external byte authority for the RAE-stream checkout.

The JSON is intentionally emitted outside the checkout.  The launcher later
receives its SHA-256 as an operator-supplied argument and verifies every
listed source/config file before creating the unchanged official trainer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    root = args.repo_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from rae_stream.authority import _runtime_paths
    from rae_stream.config_guard import OFFICIAL_COMMIT, canonical_digest, sha256_file

    paths = sorted(_runtime_paths(root))
    files = {relative: sha256_file(root / relative) for relative in paths}
    payload: dict[str, object] = {
        "schema": "rae_stream_code_authority_v1",
        "official_commit": OFFICIAL_COMMIT,
        "files": files,
    }
    payload["payload_sha256"] = canonical_digest(payload)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"path": str(output), "sha256": sha256_file(output), "files": len(files)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
