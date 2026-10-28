"""Measure exact source-content digest agreement across parity-case ingresses.

Run with ``uv run --extra demo --extra tables --extra dashboard python
scripts/measure_source_content_digest_parity.py``. This executes the public
constructors and readouts in every parity case and classifies the first exact
canonical-payload difference for each disagreeing constructor pair.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from hashlib import sha256
from itertools import combinations
from numbers import Real
from typing import Any, cast

from increment import _canonical
from increment.estimation.readout_types import StreamingDigest
from tests.parity_harness.cases import PARITY_CASES
from tests.parity_harness.runner import run_case


def _first_difference(left: Any, right: Any, path: str = "$") -> tuple[str, bool]:
    if type(left) is not type(right):
        return path, False
    differences: list[tuple[str, bool]] = []
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return path, False
        for key in sorted(left):
            field, float_only = _first_difference(left[key], right[key], f"{path}.{key}")
            if field:
                differences.append((field, float_only))
    elif isinstance(left, list):
        if len(left) != len(right):
            return path, False
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            field, float_only = _first_difference(left_item, right_item, f"{path}[{index}]")
            if field:
                differences.append((field, float_only))
    elif isinstance(left, Real) and not isinstance(left, bool):
        if left == right:
            return "", True
        if isinstance(left, float) and isinstance(right, float):
            scale = max(math.ulp(left), math.ulp(right))
            return path, math.isfinite(left) and math.isfinite(right) and abs(left - right) <= scale
        return path, False
    elif left == right:
        return "", True
    else:
        return path, False
    if not differences:
        return "", True
    return differences[0][0], all(float_only for _, float_only in differences)


def _capture_payloads(payload_by_digest: dict[str, dict[str, Any]]):
    original = _canonical.canonical_json_bytes

    def capture_payload(value: object) -> bytes:
        encoded = original(value)
        if isinstance(value, dict) and value.get("kind") == "increment.source_content":
            digest = "sha256:" + sha256(encoded).hexdigest()
            payload_by_digest[digest] = cast(dict[str, Any], value)
        return encoded

    return original, capture_payload


def _capture_streaming_rows(rows_by_digest: dict[str, list[Any]]):
    original_update = StreamingDigest.update
    original_hexdigest = StreamingDigest.hexdigest
    pending: dict[int, list[Any]] = {}

    def capture_update(self: StreamingDigest, row: object) -> None:
        pending.setdefault(id(self), []).append(json.loads(_canonical.canonical_json_bytes(row)))
        original_update(self, row)

    def capture_hexdigest(self: StreamingDigest) -> str:
        digest = original_hexdigest(self)
        rows_by_digest[digest] = pending.pop(id(self), [])
        return digest

    return original_update, original_hexdigest, capture_update, capture_hexdigest


def _expand_row_digests(value: Any, rows_by_digest: dict[str, list[Any]]) -> Any:
    if isinstance(value, dict):
        expanded = {
            key: _expand_row_digests(item, rows_by_digest)
            for key, item in value.items()
            if key != "digest" or item not in rows_by_digest
        }
        digest = value.get("digest")
        if isinstance(digest, str) and digest in rows_by_digest:
            expanded["rows"] = sorted(
                rows_by_digest[digest],
                key=_canonical.canonical_json_bytes,
            )
        return expanded
    if isinstance(value, list):
        return [_expand_row_digests(item, rows_by_digest) for item in value]
    return value


def main() -> None:
    matched_scenarios = 0
    agreeing_scenarios = 0
    matched_pairs = 0
    agreeing_pairs = 0
    classifications = {"float-only": 0, "representation": 0}
    examples: dict[str, tuple[str, str, str]] = {}
    for case in PARITY_CASES:
        payload_by_digest: dict[str, dict[str, Any]] = {}
        rows_by_digest: dict[str, list[Any]] = {}
        original_canonical_json_bytes, capture_payload = _capture_payloads(payload_by_digest)
        (
            original_update,
            original_hexdigest,
            capture_update,
            capture_hexdigest,
        ) = _capture_streaming_rows(rows_by_digest)
        _canonical.canonical_json_bytes = capture_payload
        StreamingDigest.update = capture_update
        StreamingDigest.hexdigest = capture_hexdigest
        try:
            measured_case = replace(
                case, readout_probe=None, source_probe=None, sequential_probe=None
            )
            result = run_case(measured_case)
        finally:
            _canonical.canonical_json_bytes = original_canonical_json_bytes
            StreamingDigest.update = original_update
            StreamingDigest.hexdigest = original_hexdigest
        digest_by_ingress = {
            name: digests[0]
            for name, digests in result.source_content_digests.items()
            if len(digests) == 1
        }
        payload_by_ingress = {
            name: _expand_row_digests(payload_by_digest[digest], rows_by_digest)
            for name, digest in digest_by_ingress.items()
            if digest in payload_by_digest
        }
        comparable = {
            name: digest for name, digest in digest_by_ingress.items() if name in payload_by_ingress
        }
        pairs = tuple(combinations(sorted(comparable), 2))
        if not pairs:
            continue
        matched_scenarios += 1
        scenario_agrees = True
        for left, right in pairs:
            matched_pairs += 1
            if comparable[left] == comparable[right]:
                agreeing_pairs += 1
                continue
            scenario_agrees = False
            field, float_only = _first_difference(
                payload_by_ingress[left], payload_by_ingress[right]
            )
            cause = "float-only" if float_only else "representation"
            classifications[cause] += 1
            examples.setdefault(cause, (case.id, f"{left} vs {right}", field))
        agreeing_scenarios += scenario_agrees

    print(
        f"Scenario agreement: {agreeing_scenarios}/{matched_scenarios} "
        f"({agreeing_scenarios / matched_scenarios:.1%}); "
        f"pair agreement: {agreeing_pairs}/{matched_pairs} "
        f"({agreeing_pairs / matched_pairs:.1%})"
    )
    disagreement_count = matched_pairs - agreeing_pairs
    print(
        f"Disagreements: {disagreement_count}; "
        f"float-only={classifications['float-only']}; "
        f"representation={classifications['representation']}"
    )
    for cause, (case_id, pair, field) in examples.items():
        print(f"EXAMPLE {cause}: {case_id} ({pair}), first difference at {field}")


if __name__ == "__main__":
    main()
