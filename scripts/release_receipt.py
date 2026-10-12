"""Validate exact-candidate release evidence before publication."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_REQUIRED_SCOPE = {"strict-docs", "installed-wheel-readme-mean-analysis"}
_SHA = re.compile(r"^[0-9a-f]{40}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


def verify(receipt_path: Path, expected_sha: str) -> dict[str, Any]:
    if not _SHA.fullmatch(expected_sha):
        raise ValueError("expected candidate SHA must be a full lowercase 40-character commit SHA")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"release receipt is missing or invalid: {exc}") from exc
    if not isinstance(receipt, dict):
        raise ValueError("release receipt must be a JSON object")
    if receipt.get("candidate_sha") != expected_sha:
        raise ValueError("release receipt candidate SHA does not match the expected candidate")
    if not isinstance(receipt.get("tag"), str) or not re.fullmatch(
        r"v[0-9][A-Za-z0-9.!+-]*", receipt["tag"]
    ):
        raise ValueError("release receipt has no valid version tag")
    if receipt["tag"][1:] != receipt.get("version"):
        raise ValueError("release receipt tag and package version do not match")
    for field in ("wheel_sha256", "sdist_sha256"):
        if not isinstance(receipt.get(field), str) or not _HASH.fullmatch(receipt[field]):
            raise ValueError(f"release receipt has no valid {field}")
    if not isinstance(receipt.get("run_url"), str) or not receipt["run_url"].startswith("https://"):
        raise ValueError("release receipt has no exact workflow run URL")
    scope = receipt.get("scope")
    if not isinstance(scope, list) or not _REQUIRED_SCOPE.issubset(scope):
        raise ValueError(
            "release receipt does not cover strict docs and installed-wheel README smoke"
        )
    for name in ("docs", "readme_smoke"):
        evidence = receipt.get(name)
        if not isinstance(evidence, dict):
            raise ValueError(f"release receipt is missing {name} evidence")
        if evidence.get("status") != "success":
            raise ValueError(
                f"{name} evidence is not successful (status={evidence.get('status')!r})"
            )
        if evidence.get("sha") != expected_sha:
            raise ValueError(f"{name} evidence SHA does not match the exact candidate")
        if not isinstance(evidence.get("run_url"), str) or not evidence["run_url"].startswith(
            "https://"
        ):
            raise ValueError(f"{name} evidence has no exact run URL")
    for field in ("unresolved", "deviations"):
        if not isinstance(receipt.get(field), list):
            raise ValueError(f"release receipt {field} must be a list")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser(
        "verify", help="fail closed unless exact-candidate evidence passes"
    )
    verify_parser.add_argument("--receipt", type=Path, required=True)
    verify_parser.add_argument("--sha", required=True)
    args = parser.parse_args()
    try:
        receipt = verify(args.receipt, args.sha)
    except ValueError as exc:
        parser.exit(1, f"release receipt rejected: {exc}\n")
    print(f"release receipt accepted: {receipt['tag']} candidate={receipt['candidate_sha']}")


if __name__ == "__main__":
    main()
