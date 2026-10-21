from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_receipt.py"
SHA = "a" * 40


def receipt() -> dict[str, object]:
    return {
        "candidate_sha": SHA,
        "tag": "v1.2.3",
        "version": "1.2.3",
        "wheel_sha256": hashlib.sha256(b"wheel").hexdigest(),
        "sdist_sha256": hashlib.sha256(b"sdist").hexdigest(),
        "run_url": "https://github.com/example/repo/actions/runs/123",
        "scope": ["strict-docs", "installed-wheel-readme-mean-analysis"],
        "docs": {"status": "success", "sha": SHA, "run_url": "https://example/docs"},
        "readme_smoke": {
            "status": "success",
            "sha": SHA,
            "run_url": "https://example/smoke",
        },
        "unresolved": [],
        "deviations": [],
    }


def run_receipt(tmp_path: Path, data: dict[str, object] | None) -> subprocess.CompletedProcess[str]:
    path = tmp_path / "receipt.json"
    if data is not None:
        path.write_text(json.dumps(data), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT), "verify", "--receipt", str(path), "--sha", SHA],
        text=True,
        capture_output=True,
        check=False,
    )


def test_matching_receipt_is_accepted(tmp_path: Path) -> None:
    result = run_receipt(tmp_path, receipt())
    assert result.returncode == 0, result.stderr


def test_absent_receipt_is_rejected(tmp_path: Path) -> None:
    result = run_receipt(tmp_path, None)
    assert result.returncode != 0


def test_failed_cancelled_and_stale_sha_evidence_are_rejected(tmp_path: Path) -> None:
    for status in ("failure", "cancelled"):
        evidence = receipt()
        evidence["docs"] = {"status": status, "sha": SHA, "run_url": "https://example/docs"}
        assert run_receipt(tmp_path, evidence).returncode != 0

    evidence = receipt()
    evidence["readme_smoke"] = {
        "status": "success",
        "sha": "b" * 40,
        "run_url": "https://example/smoke",
    }
    assert run_receipt(tmp_path, evidence).returncode != 0
