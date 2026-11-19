"""Check the published assignment vectors without importing either SDK."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

_VECTORS = Path(__file__).resolve().parents[1] / "docs/reference/assignment-v1-vectors.json"
_CUTOFF = 4_294_960_000
_STAGES = {"holdout", "layer", "enroll", "arm"}


def _same(actual: object, expected: object) -> bool:
    return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


def _frame(stage: str, key: str, unit: str, attempt: int) -> bytes:
    frame = bytearray(b"increment.alloc.v1\x00")
    for field in (stage, key, unit):
        encoded = field.encode("utf-8")
        frame.extend(struct.pack(">I", len(encoded)))
        frame.extend(encoded)
    frame.extend(struct.pack(">I", attempt))
    return bytes(frame)


class _Witnesses:
    def __init__(self, rows):
        self._buckets = {}
        self._unused = set()
        for row in rows:
            stage, key, unit, attempt = (row[k] for k in ("stage", "key", "unit_id", "attempt"))
            address = (stage, key, unit, attempt)
            if stage not in _STAGES or type(attempt) is not int or address in self._buckets:
                raise ValueError(f"Invalid or repeated hash witness: {address!r}")
            framed = _frame(*address)
            digest = hashlib.sha256(framed).digest()
            word = struct.unpack_from(">I", digest)[0]
            accepted = word < _CUTOFF
            bucket = word % 10_000 if accepted else None
            for field, actual in (
                ("input_hex", framed.hex()),
                ("digest_hex", digest.hex()),
                ("accepted", accepted),
                ("bucket", bucket),
            ):
                if not _same(actual, row[field]):
                    raise ValueError(f"Hash witness {address!r}: incorrect {field}")
            self._buckets[address] = bucket
            self._unused.add(address)

    def draw(self, stage, key, unit):
        attempt = 0
        while True:
            address = (stage, key, unit, attempt)
            if address not in self._buckets:
                raise ValueError(f"Missing hash witness: {address!r}")
            self._unused.discard(address)
            bucket = self._buckets[address]
            if bucket is not None:
                return bucket
            attempt += 1

    def check_used(self):
        if self._unused:
            raise ValueError(f"Unused hash witnesses: {sorted(self._unused)!r}")


def _membership(config, holdout, unit, witnesses):
    rate = int(holdout["held_out_bps"])
    key = holdout["key"]
    held = witnesses.draw("holdout", key, unit) < rate
    return {
        "protocol": 1,
        "revision": config["revision"],
        "holdout": key,
        "group": "held_out" if held else "eligible",
        "group_bps": rate if held else 10_000 - rate,
    }


def _result(config, case, witnesses):
    target, unit = case["target"], case["unit_id"]
    holdouts = config.get("holdouts", [])
    if case["op"] == "membership":
        holdout = next(row for row in holdouts if row["key"] == target)
        return _membership(config, holdout, unit, witnesses)
    if case["op"] != "decide":
        raise ValueError(f"Unexpected successful operation: {case['op']!r}")

    holdout = next((row for row in holdouts if target in row["targets"]), None)
    membership = None if holdout is None else _membership(config, holdout, unit, witnesses)
    decision = dict.fromkeys(("experiment", "arm", "layer_bps", "enrollment_bps", "arm_bps"))
    decision.update(protocol=1, revision=config["revision"], target=target, membership=membership)
    if membership is not None and membership["group"] == "held_out":
        return dict(decision, status="held_out")

    layer = next((row for row in config["layers"] if row["key"] == target), None)
    experiment, width = target, 10_000
    if layer is not None:
        point = witnesses.draw("layer", target, unit)
        selected = next(
            (row for row in layer["ranges"] if row["start_bps"] <= point < row["end_bps"]),
            None,
        )
        if selected is None:
            return dict(decision, status="unallocated")
        experiment = selected["experiment"]
        width = int(selected["end_bps"] - selected["start_bps"])

    row = next(row for row in config["experiments"] if row["key"] == experiment)
    rate = int(row["enrollment_bps"])
    decision.update(experiment=experiment, layer_bps=width, enrollment_bps=rate)
    if witnesses.draw("enroll", experiment, unit) >= rate:
        return dict(decision, status="not_enrolled")

    position = witnesses.draw("arm", experiment, unit)
    for arm in sorted(row["arms"], key=lambda arm: arm["key"].encode("utf-8")):
        weight = int(arm["weight_bps"])
        if position < weight:
            return dict(decision, status="assigned", arm=arm["key"], arm_bps=weight)
        position -= weight
    raise ValueError(f"Arms do not cover bucket in {experiment!r}")


def _check_cases(corpus, witnesses):
    seen = set()
    failures = 0
    for case in corpus["cases"]:
        name = case["id"]
        try:
            if name in seen or ("expected" in case) == ("error" in case):
                raise ValueError("Repeated ID or missing/ambiguous outcome")
            seen.add(name)
            config = corpus["configs"][case["config"]]
            if "error" in case:
                # Refusals are tested against the SDKs; there is no SDK validator here.
                op = case["op"]
                allowed = {
                    "compile": {"allocation.invalid_config"},
                    "decide": {"allocation.invalid_target", "allocation.invalid_unit"},
                    "membership": {"allocation.invalid_target", "allocation.invalid_unit"},
                }
                if case["error"] not in allowed[op]:
                    raise ValueError("Refusal code does not match the operation")
                failures += 1
            else:
                actual = _result(config, case, witnesses)
                if not _same(actual, case["expected"]):
                    raise ValueError(f"Expected {case['expected']!r}, derived {actual!r}")
        except (ValueError, KeyError, StopIteration, TypeError) as exc:
            raise ValueError(f"{name}: {exc}") from exc
    return failures


def main() -> None:
    """Check framing, digests and successful records, and report corpus counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, default=_VECTORS)
    args = parser.parse_args()
    try:
        corpus = json.loads(args.vectors.read_text(encoding="utf-8"))
        if not _same(corpus["protocol"], 1):
            raise ValueError("Expected protocol 1")
        witnesses = _Witnesses(corpus["hashes"])
        refusals = _check_cases(corpus, witnesses)
        witnesses.check_used()
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        parser.exit(1, f"{args.vectors.name}: {exc}\n")
    successes = len(corpus["cases"]) - refusals
    print(
        f"Assignment v1: {len(corpus['hashes'])} hash witnesses, {successes} successful records, "
        f"{refusals} refusal cases"
    )


if __name__ == "__main__":
    main()
