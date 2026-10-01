"""Tamper-evident draw accounting for calibration campaigns.

A campaign makes tens of millions of draws. Storing one record per draw costs
more in filesystem blocks than the evidence is worth, and independent files
cannot detect truncation or reordering. With a frozen seed and RNG law any draw
is reproducible, so the journal retains counts plus a hash chain and keeps whole
records only for draws that are diagnostically interesting.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Any, Self

JOURNAL_NAME = "journal.jsonl"
EXCEPTIONS_NAME = "exceptions.jsonl"


class JournalError(RuntimeError):
    """A journal is unreadable, broken, or inconsistent with its accounting."""


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _link(prev: str, payload: dict[str, Any]) -> str:
    return hashlib.sha256(prev.encode() + b"\x00" + _canonical(payload)).hexdigest()


def genesis(case_id: str) -> str:
    """Chain root bound to the case, so blocks cannot be spliced between cases."""
    return hashlib.sha256(b"calibration/journal/v1\x00" + case_id.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class JournalTotals:
    """Verified accounting for one case's journal."""

    case_id: str
    blocks: int
    started: int
    completed: int
    exceptional: int
    exceptional_dropped: int
    counters: Mapping[str, int]
    digest: str


class DrawJournal:
    """Append-only, hash-chained accounting with capped exception retention.

    Each sealed block records the draw range it covers, its cumulative counters,
    and a digest over the previous digest and its own payload, so a removed,
    edited, or reordered block breaks verification.
    """

    def __init__(
        self,
        directory: Path,
        *,
        case_id: str,
        block_size: int = 1000,
        exception_limit: int = 4096,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if exception_limit < 0:
            raise ValueError("exception_limit must be nonnegative")
        if not case_id:
            raise ValueError("case_id must be a nonempty string")
        directory.mkdir(parents=True, exist_ok=True)
        self._case_id = case_id
        self._block_size = block_size
        self._exception_limit = exception_limit
        self._digest = genesis(case_id)
        self._block = 0
        self._first: int | None = None
        self._last: int | None = None
        self._active: list[dict[str, Any]] = []
        self._started = 0
        self._completed = 0
        self._exceptional = 0
        self._dropped = 0
        self._retained = 0
        self._counters: dict[str, int] = {}
        # These are the totals represented by the durable chain.  Live totals
        # may include a partial block until it is sealed.
        self._recorded_started = 0
        self._recorded_completed = 0
        self._recorded_exceptional = 0
        self._recorded_dropped = 0
        self._recorded_counters: dict[str, int] = {}
        self._closed = False
        journal_path = directory / JOURNAL_NAME
        exceptions_path = directory / EXCEPTIONS_NAME
        if journal_path.exists() and journal_path.stat().st_size:
            # Verify committed chain metadata before repairing either file.
            # A tampered chain must remain untouched for forensic inspection.
            totals = verify(directory, case_id=case_id, _check_sidecar=False)
            self._repair_tails(
                journal_path,
                exceptions_path,
                expected_retained=totals.exceptional - totals.exceptional_dropped,
            )
            totals = verify(directory, case_id=case_id)
            self._digest = totals.digest
            self._block = totals.blocks
            self._started = totals.started
            self._completed = totals.completed
            self._exceptional = totals.exceptional
            self._dropped = totals.exceptional_dropped
            self._counters = dict(totals.counters)
            self._recorded_started = totals.started
            self._recorded_completed = totals.completed
            self._recorded_exceptional = totals.exceptional
            self._recorded_dropped = totals.exceptional_dropped
            self._recorded_counters = dict(totals.counters)
            if exceptions_path.exists():
                self._retained = len(exceptions_path.read_text(encoding="utf-8").splitlines())
        elif exceptions_path.exists() and exceptions_path.stat().st_size:
            # No sealed chain can vouch for sidecar rows.  An orphan sidecar
            # is a crash remnant and must be discarded coherently.
            exceptions_path.write_bytes(b"")
        self._journal = journal_path.open("a", encoding="utf-8")
        self._exceptions = exceptions_path.open("a", encoding="utf-8")

    @staticmethod
    def _repair_tails(
        journal_path: Path,
        exceptions_path: Path,
        *,
        expected_retained: int,
    ) -> None:
        """Discard only torn writes/orphan sidecar rows after chain verification."""
        raw = journal_path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raw = raw[: raw.rfind(b"\n") + 1]
            journal_path.write_bytes(raw)
        sidecar_before = exceptions_path.read_bytes() if exceptions_path.exists() else b""
        sidecar = sidecar_before
        if sidecar and not sidecar.endswith(b"\n"):
            sidecar = sidecar[: sidecar.rfind(b"\n") + 1]
        sidecar_rows = sidecar.splitlines()
        if len(sidecar_rows) > expected_retained:
            sidecar = b"\n".join(sidecar_rows[:expected_retained]) + (
                b"\n" if expected_retained else b""
            )
        if sidecar != sidecar_before:
            exceptions_path.write_bytes(sidecar)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def started(self, index: int) -> None:
        """Record that draw *index* began."""
        if self._closed:
            raise JournalError("journal is closed")
        if not self._active or self._active[-1]["started"] == self._block_size:
            self._active.append(
                {
                    "first": index,
                    "last": index,
                    "started": 0,
                    "completed": 0,
                    "exceptional": 0,
                    "dropped": 0,
                    "counters": {},
                    "pending": set(),
                    "records": [],
                }
            )
        block = self._active[-1]
        block["last"] = index
        block["started"] += 1
        block["pending"].add(index)
        self._started += 1

    def completed(
        self,
        index: int,
        record: Any = None,
        *,
        tally: Sequence[str] = (),
        exceptional: bool = False,
    ) -> None:
        """Record that draw *index* finished, retaining *record* only when exceptional."""
        if self._closed:
            raise JournalError("journal is closed")
        block = next(
            (candidate for candidate in reversed(self._active) if index in candidate["pending"]),
            None,
        )
        if block is None:
            raise JournalError(f"draw {index} was not started or was already completed")
        block["pending"].remove(index)
        block["completed"] += 1
        self._completed += 1
        for name in tally:
            block["counters"][name] = block["counters"].get(name, 0) + 1
            self._counters[name] = self._counters.get(name, 0) + 1
        if exceptional:
            block["exceptional"] += 1
            self._exceptional += 1
            if self._retained >= self._exception_limit:
                block["dropped"] += 1
                self._dropped += 1
            else:
                self._retained += 1
                block["records"].append({"case_id": self._case_id, "draw": index, "record": record})
        while (
            self._active
            and self._active[0]["started"] == self._block_size
            and not self._active[0]["pending"]
        ):
            finished = self._active.pop(0)
            self._first, self._last = finished["first"], finished["last"]
            self._seal(finished)

    def _payload(self, block: dict[str, Any]) -> dict[str, Any]:
        counters = dict(self._recorded_counters)
        for name, count in block["counters"].items():
            counters[name] = counters.get(name, 0) + count
        return {
            "case_id": self._case_id,
            "block": self._block,
            "first_draw": self._first,
            "last_draw": self._last,
            "started": self._recorded_started + block["started"],
            "completed": self._recorded_completed + block["completed"],
            "exceptional": self._recorded_exceptional + block["exceptional"],
            "exceptional_dropped": self._recorded_dropped + block["dropped"],
            "counters": dict(sorted(counters.items())),
        }

    def _seal(self, block: dict[str, Any]) -> None:
        if self._first is None or self._last is None:
            return
        # Publish retained records in block-seal order, not completion order.
        for record in block["records"]:
            self._exceptions.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        # Sidecar data must reach stable storage before the chain advertises
        # the corresponding exceptional count.
        self._exceptions.flush()
        os.fsync(self._exceptions.fileno())
        payload = self._payload(block)
        self._digest = _link(self._digest, payload)
        self._journal.write(
            json.dumps(
                {**payload, "digest": self._digest},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        self._journal.flush()
        os.fsync(self._journal.fileno())
        self._recorded_started = payload["started"]
        self._recorded_completed = payload["completed"]
        self._recorded_exceptional = payload["exceptional"]
        self._recorded_dropped = payload["exceptional_dropped"]
        self._recorded_counters = dict(payload["counters"])
        self._block += 1
        self._first = None
        self._last = None

    def totals(self) -> JournalTotals:
        """Current accounting; identical to what ``verify`` recomputes once closed."""
        return JournalTotals(
            case_id=self._case_id,
            blocks=self._block,
            started=self._started,
            completed=self._completed,
            exceptional=self._exceptional,
            exceptional_dropped=self._dropped,
            counters=MappingProxyType(dict(self._counters)),
            digest=self._digest,
        )

    def close(self) -> JournalTotals:
        """Seal any partial blocks and finalize both logs."""
        if not self._closed:
            while self._active:
                finished = self._active.pop(0)
                self._first, self._last = finished["first"], finished["last"]
                self._seal(finished)
            self._exceptions.flush()
            os.fsync(self._exceptions.fileno())
            self._exceptions.close()
            self._journal.close()
            self._closed = True
        return self.totals()


def verify(
    directory: Path,
    *,
    case_id: str,
    _check_sidecar: bool = True,
) -> JournalTotals:
    """Recompute the chain and return verified totals, or raise ``JournalError``."""
    path = directory / JOURNAL_NAME
    if not path.exists():
        raise JournalError(f"missing journal for {case_id!r}")
    digest = genesis(case_id)
    blocks = 0
    started = completed = exceptional = dropped = 0
    counters: dict[str, int] = {}
    previous_last: int | None = None
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raw = raw[: raw.rfind(b"\n") + 1]
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise JournalError("journal is not valid UTF-8") from error
    for number, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError as error:
            raise JournalError(f"unreadable journal line {number}") from error
        if not isinstance(row, dict) or "digest" not in row:
            raise JournalError(f"malformed journal line {number}")
        recorded = row.pop("digest")
        if row.get("case_id") != case_id:
            raise JournalError(f"journal line {number} belongs to another case")
        if row.get("block") != blocks:
            raise JournalError(f"journal block out of order at line {number}")
        digest = _link(digest, row)
        if digest != recorded:
            raise JournalError(f"journal chain broken at block {blocks}")
        first, last = row["first_draw"], row["last_draw"]
        if first > last or (previous_last is not None and first <= previous_last):
            raise JournalError(f"journal draw range overlaps or inverts at block {blocks}")
        previous_last = last
        started, completed = row["started"], row["completed"]
        exceptional, dropped = row["exceptional"], row["exceptional_dropped"]
        counters = dict(row["counters"])
        blocks += 1
    if completed > started:
        raise JournalError("journal completed more draws than it started")
    expected_retained = exceptional - dropped
    if _check_sidecar:
        exceptions_path = directory / EXCEPTIONS_NAME
        exception_lines = (
            exceptions_path.read_bytes().splitlines(keepends=True)
            if exceptions_path.exists()
            else []
        )
        if expected_retained < 0 or len(exception_lines) < expected_retained:
            raise JournalError("exception sidecar is missing committed records")
        for number, line in enumerate(exception_lines[:expected_retained]):
            if not line.endswith(b"\n"):
                raise JournalError(f"incomplete committed exception line {number}")
            try:
                exception = json.loads(line)
            except ValueError as error:
                raise JournalError(f"unreadable exception line {number}") from error
            if not isinstance(exception, dict) or exception.get("case_id") != case_id:
                raise JournalError(f"malformed exception line {number}")
    return JournalTotals(
        case_id=case_id,
        blocks=blocks,
        started=started,
        completed=completed,
        exceptional=exceptional,
        exceptional_dropped=dropped,
        counters=MappingProxyType(counters),
        digest=digest,
    )
