"""Two registered sharp nulls, outside the built-in decision family. See custom-evidence-sources.md."""

import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import cache
from math import comb
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import mannwhitneyu

from increment.errors import CodedError, CodedModel
from increment.sequential_state import sequential_refuse

# Registered before observing assignments or outcomes. Both procedures use this same roster.
METRICS = ("saved", "shared")
UNIT_ORDER = tuple(range(1, 14))
ALPHA = Fraction(1, 20)
PER_NULL = ALPHA / len(METRICS)
BIT = Annotated[int, Field(strict=True, ge=0, le=1)]
COMMAND = "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python docs/development/prototypes/custom_evidence.py"


class Reveal(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    unit: int
    assignment: BIT
    outcomes: tuple[tuple[str, BIT], ...]
    finalized: bool


@cache
def evidence(n, agreements):
    """Exact upper binomial tail and current capital: each match pays 3/2, each miss 1/2."""
    denominator = 1 << n
    p = Fraction(sum(comb(n, r) for r in range(agreements, n + 1)), denominator)
    e = Fraction(3**agreements, denominator)
    return p, e


@dataclass(frozen=True)
class Look:
    n: int
    p: tuple[float, ...]
    e: tuple[float, ...]
    fixed_reject: tuple[bool, ...]
    seq_reject: tuple[bool, ...]


class Monitor:
    def __init__(self):
        self._rows = ()
        self._looks = []
        self._agreements = [0] * len(METRICS)
        self._crossed = [False] * len(METRICS)

    def capture(self, rows):
        """Recapture a full prefix; identical history is not another bet."""
        rows = tuple(rows)
        consumed = len(self._rows)
        if rows[:consumed] != self._rows:
            sequential_refuse(
                "continuation.rewrite", "Consumed joint records changed or disappeared"
            )
        new = rows[consumed:]
        for i, row in enumerate(new, start=consumed):
            if not isinstance(row, Reveal):
                sequential_refuse(
                    "source.invalid", "Only raw, jointly finalized unit reveals are evidence"
                )
            if i >= len(UNIT_ORDER) or row.unit != UNIT_ORDER[i]:
                sequential_refuse(
                    "source.invalid", "Expected the next distinct registered unit", unit=row.unit
                )
            if not row.finalized:
                sequential_refuse("source.invalid", "Both outcomes must be final", unit=row.unit)
            if tuple(name for name, _ in row.outcomes) != METRICS:
                sequential_refuse(
                    "source.invalid",
                    "Expected both registered outcomes in roster order",
                    unit=row.unit,
                )

        for n, row in enumerate(new, start=consumed + 1):
            p_values, capitals, fixed = [], [], []
            for j, (_, y) in enumerate(row.outcomes):
                self._agreements[j] += row.assignment == y
                p, e = evidence(n, self._agreements[j])
                self._crossed[j] |= e >= 1 / PER_NULL
                p_values.append(float(p))
                capitals.append(float(e))
                fixed.append(n == len(UNIT_ORDER) and p <= PER_NULL)
            self._looks.append(
                Look(n, tuple(p_values), tuple(capitals), tuple(fixed), tuple(self._crossed))
            )
        self._rows = rows
        return tuple(self._looks)


def enumerate_null(outcomes):
    """Hold both outcome schedules fixed and visit every equally likely assignment path."""
    horizon = len(outcomes)
    masks = [sum(row[j] << i for i, row in enumerate(outcomes)) for j in range(len(METRICS))]
    counts = dict.fromkeys(
        (
            "fixed_unadjusted",
            "fixed_bonferroni",
            "repeated_unadjusted",
            "repeated_bonferroni",
            "anytime_bonferroni",
        ),
        0,
    )
    for assignment in range(1 << horizon):
        matches = [~(assignment ^ mask) for mask in masks]
        ever_raw = ever_fixed = ever_seq = False
        for n in range(1, horizon + 1):
            prefix = (1 << n) - 1
            # Both rules are monotone in agreements: the maximum tests 'at least one'.
            k = max((mask & prefix).bit_count() for mask in matches)
            p, e = evidence(n, k)
            raw, fixed = p <= ALPHA, p <= PER_NULL
            ever_raw |= raw
            ever_fixed |= fixed
            ever_seq |= e >= 1 / PER_NULL
        counts["fixed_unadjusted"] += raw
        counts["fixed_bonferroni"] += fixed
        counts["repeated_unadjusted"] += ever_raw
        counts["repeated_bonferroni"] += ever_fixed
        counts["anytime_bonferroni"] += ever_seq

    assert tuple(counts.values()) == (736, 184, 1260, 580, 56)
    paths = 1 << horizon
    return {
        "null": "both sharp nulls; outcomes fixed, assignments random",
        "fixed_horizon": horizon,
        "paths": paths,
        "event": "at least one rejection; repeated/anytime inspect every prefix",
        "crossings": {
            name: {"paths": count, "probability": count / paths} for name, count in counts.items()
        },
    }


def refusal_examples(rows, split):
    first, second = rows[:2]
    report = mannwhitneyu(
        *([row.outcomes[0][1] for row in rows[:split] if row.assignment == arm] for arm in (1, 0)),
        alternative="two-sided",
        method="asymptotic",
    )
    # Exercise the same entry point as a successful continuation, not a p-to-e conversion.
    mw = {"hypothesis": METRICS[0], "method": "Mann–Whitney U", "p": float(report.pvalue)}
    attempts = {
        "duplicate unit": ((first, first), "sequential.source.invalid"),
        "unfinalized unit": (
            (first, second.model_copy(update={"finalized": False})),
            "sequential.source.invalid",
        ),
        "changed consumed assignment": (
            (first.model_copy(update={"assignment": 1 - first.assignment}),),
            "sequential.continuation.rewrite",
        ),
        "changed consumed outcome": (
            (
                first.model_copy(
                    update={"outcomes": ((METRICS[0], 1 - first.outcomes[0][1]), first.outcomes[1])}
                ),
            ),
            "sequential.continuation.rewrite",
        ),
        "unregistered hypothesis": (
            (first, second.model_copy(update={"outcomes": (second.outcomes[0], ("clicked", 1))})),
            "sequential.source.invalid",
        ),
        "missing hypothesis": (
            (first, second.model_copy(update={"outcomes": (second.outcomes[0],)})),
            "sequential.source.invalid",
        ),
        "report-only MW p": ((first, mw), "sequential.source.invalid"),
    }
    monitor = Monitor()
    monitor.capture((first,))
    refused = {}
    for label, (records, expected) in attempts.items():
        try:
            monitor.capture(records)
        except CodedError as exc:
            assert exc.code == expected
            refused[label] = {"code": exc.code, "reason": str(exc)}
        else:
            raise AssertionError(f"unexpectedly admitted: {label}")
    assert monitor.capture(rows) == Monitor().capture(rows)
    return refused, mw


def main():
    outcomes = [(i % 2, (i // 2) % 2) for i in range(len(UNIT_ORDER))]
    split = 10
    # Illustrative path chosen for visibility, not an estimate of power or a sample from an alternative.
    rows = tuple(
        Reveal(
            unit=unit,
            assignment=y[0] if i < split else 1 - y[0],
            outcomes=tuple(zip(METRICS, y, strict=True)),
            finalized=True,
        )
        for i, (unit, y) in enumerate(zip(UNIT_ORDER, outcomes, strict=True))
    )
    monitor = Monitor()
    prefix = monitor.capture(rows[:split])
    assert monitor.capture(rows[:split]) == prefix
    looks = monitor.capture(rows)
    assert looks[:split] == prefix and looks == Monitor().capture(rows)
    assert prefix[-1].seq_reject[0] and looks[-1].seq_reject[0] and looks[-1].e[0] < 1 / PER_NULL
    assert evidence(0, 0) == (1, 1)
    refusals, mw = refusal_examples(rows, split)
    return {
        "hypotheses": METRICS,
        "alpha": float(ALPHA),
        "command": COMMAND,
        "family": "Separate custom family, outside all built-in decisions. Choose one monitoring rule in advance; do not pool rejections.",
        "assumptions": [
            "Each metric tests its own sharp no-effect null, not equality of means. Direction: excess A=Y agreement.",
            "No interference; independent fair Bernoulli assignments. Within-unit metrics may depend on one another.",
            "Unit roster/order, both binary outcomes, fixed horizon and half-stake committed before data.",
            "No assignment or outcome leakage: reveal both final outcomes with their assignment in registered order; no missing units.",
        ],
        "rules": {
            "fixed": "p is the exact upper binomial tail. Use p <= alpha/2 only at the registered horizon.",
            "sequential": "e is current capital from 1, times 3/2 on agreement or 1/2 otherwise. seq_reject latches any crossing of 2/alpha.",
            "risk": "Ville plus the union bound gives strong anytime FWER <= alpha. Enumeration measures global-null risk for this schedule only.",
        },
        "illustrative_path": {
            "assignment": [r.assignment for r in rows],
            "outcomes_in_hypothesis_order": outcomes,
        },
        "looks": [asdict(look) for look in looks],
        "enumeration": enumerate_null(outcomes),
        "replay": "Identical recapture, prefix extension and one-shot replay agree.",
        "report_only_at_n": split,
        "report_only": mw,
        "refusals": refusals,
        "integration": "Not a registered Increment process: EValueEvidence requires a certified raw_likelihood_v1 checkpoint. See custom-evidence-sources.md.",
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, allow_nan=False))
