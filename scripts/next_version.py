"""Resolve a release version from reachable Git tags without creating a tag."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from packaging.version import InvalidVersion, Version

BUMP_LEVELS = ("auto", "patch", "minor", "major")
STAGES = {"alpha": "a", "beta": "b", "rc": "rc", "stable": None}
PRE_ORDER = {"a": 0, "b": 1, "rc": 2}


def _public_version(tag: str) -> Version | None:
    try:
        version = Version(tag[1:])
    except InvalidVersion:
        return None
    return version if version.local is None else None


def _tags(repo: Path, *options: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), "tag", *options, "--list", "v*"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def _next_version(baseline: Version, bump: str, stage: str) -> Version:
    if (
        baseline.epoch
        or baseline.post is not None
        or baseline.dev is not None
        or len(baseline.release) != 3
    ):
        raise ValueError(
            "Automatic bumps need a three-part release or a/b/rc version; use --version"
        )

    release = list(baseline.release)
    if bump != "auto" or baseline.pre is None:
        index = {"auto": 2, "patch": 2, "minor": 1, "major": 0}[bump]
        release[index] += 1
        release[index + 1 :] = [0] * (2 - index)

    current_stage = baseline.pre[0] if baseline.pre else None
    target_stage = current_stage if stage == "keep-current" else STAGES[stage]
    number = 1
    if tuple(release) == baseline.release and baseline.pre and target_stage:
        if PRE_ORDER[target_stage] < PRE_ORDER[baseline.pre[0]]:
            raise ValueError("Stage would move backwards; bump patch/minor/major or use --version")
        if target_stage == current_stage:
            number = baseline.pre[1] + 1

    text = ".".join(map(str, release))
    if target_stage:
        text += f"{target_stage}{number}"
    version = Version(text)
    if version <= baseline:
        raise ValueError("Calculated version must advance the baseline; use --version")
    return version


def resolve_version(
    repo: Path,
    *,
    ref: str = "HEAD",
    bump: str = "auto",
    stage: str = "keep-current",
    override: str = "",
) -> dict[str, str | None]:
    """Select a version while preserving tag reservations across branches."""
    if override and (bump != "auto" or stage != "keep-current"):
        raise ValueError("Version override is exclusive; reset bump/stage to their defaults")
    versions = {
        tag: version for tag in _tags(repo) if (version := _public_version(tag)) is not None
    }
    reachable = _tags(repo, "--merged", ref)
    baseline = max((versions[tag] for tag in reachable if tag in versions), default=None)

    if override:
        version = Version(override)
        if version.local is not None or str(version) != override:
            raise ValueError("Use a canonical PEP 440 public version without leading v or +local")
    else:
        if baseline is None:
            raise ValueError("No reachable release tag; set the first release with --version")
        version = _next_version(baseline, bump, stage)
    reserved = next((tag for tag, used in versions.items() if used == version), None)
    if reserved is not None:
        route = (
            "retry release.yml on the existing tag"
            if override
            else "choose another bump or set --version"
        )
        raise ValueError(f"Version {version} is reserved by {reserved}; {route}")
    return {"baseline": str(baseline) if baseline is not None else None, "version": str(version)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--ref", default="HEAD", help="Commit whose reachable tags define the baseline"
    )
    parser.add_argument("--bump", choices=BUMP_LEVELS, default="auto")
    parser.add_argument("--stage", choices=("keep-current", *STAGES), default="keep-current")
    parser.add_argument(
        "--version", default="", help="Exact PEP 440 override; keep bump/stage defaults"
    )
    args = parser.parse_args(argv)
    try:
        result = resolve_version(
            args.repo, ref=args.ref, bump=args.bump, stage=args.stage, override=args.version
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
