"""Fail-closed source and design authority checks."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


class AuthorityError(RuntimeError):
    """Raised when an implementation authority check cannot be established."""


def git_head(repo: Path) -> str:
    """Return the repository HEAD, failing closed on git execution errors."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        raise AuthorityError(f"failed to resolve git HEAD for {repo}") from exc
    return result.stdout.strip()


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    """Return whether ``ancestor`` is an ancestor of ``descendant``."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            capture_output=True,
            text=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise AuthorityError(f"git ancestry check failed for {repo}") from exc

    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise AuthorityError(
        f"git ancestry check failed for {repo} with return code {result.returncode}"
    )


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Compute a file SHA-256 digest with bounded reads."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_authority(
    *,
    intact_repo: Path,
    expected_intact: str,
    expected_design: str,
    observed_design: str,
) -> dict[str, str]:
    """Verify the upstream ancestry and exact frozen design identity."""
    implementation_head = git_head(intact_repo)
    if not is_ancestor(intact_repo, expected_intact, implementation_head):
        raise AuthorityError(
            "INTACT base is not an ancestor of implementation: "
            f"{expected_intact}"
        )
    if observed_design != expected_design:
        raise AuthorityError(
            "design bundle mismatch: "
            f"expected {expected_design}, observed {observed_design}"
        )

    return {
        "intact_base_commit": expected_intact,
        "implementation_head": implementation_head,
        "design_bundle_sha256": observed_design,
    }
