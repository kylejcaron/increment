"""Experiment integrity gates as pure functions over (con, Table, Experiment).

Callers own caching and warn-once state - these functions never hold any.
Arm-free surfaces (the frame/moments/panel `MomentSource`s) never import
this module: mixed-assignment, cluster-label, and trigger-coverage gates
all presume a raw exposure event stream, which only the warehouse pipeline
retains.
"""

from __future__ import annotations

from typing import Literal, cast

from ibis import Table
from ibis.backends.sql import SQLBackend

from increment.errors import (
    IncrementWarning,
    InvalidRequestError,
    RefusalSpec,
    WarningSpec,
    refuse,
    warn,
)
from increment.query.builders import mixed_assignment_units
from increment.semantics.models import Experiment

_CLUSTER_CONFLICT = RefusalSpec(
    "query.integrity.cluster_conflict",
    InvalidRequestError,
    template="experiment {experiment!r}: cluster {cluster!r} has multiple non-null labels for enrolled unit(s) {units!r}",
)
# Both carry both counts: one exposure table can hold both hazards at once.
_MIXED_ASSIGNMENTS = RefusalSpec(
    "query.integrity.mixed_assignment_units",
    InvalidRequestError,
    template="{finding}; {route}",
    keys=frozenset({"mixed_count", "unassigned_count"}),
)
_UNASSIGNED_ASSIGNMENTS = RefusalSpec(
    "query.integrity.unassigned_assignment_units",
    InvalidRequestError,
    template="{finding}; {route}",
    keys=frozenset({"mixed_count", "unassigned_count"}),
)
_CLUSTER_LABELS = RefusalSpec(
    "query.integrity.cluster_labels",
    InvalidRequestError,
    template=(
        "experiment {experiment!r}: cluster field {cluster!r} has invalid {kind}: "
        "value/count {value!r}; constraint {constraint}"
    ),
)
_TRIGGER_ARM_MISSING = RefusalSpec(
    "query.integrity.trigger_arm_missing",
    InvalidRequestError,
    template="trigger {trigger!r} never fires in arm(s) {silent!r} (observed rates: {rates!r}); {route}",
)
_MIXED_ASSIGNMENTS_EXCLUDED_WARNING = WarningSpec(
    "query.integrity.mixed_assignments_excluded",
    IncrementWarning,
    lambda *, finding, suffix: f"Analysis: {finding}{suffix}; check assignment integrity.",
)


def validate_cluster_uniqueness(
    exposures: Table, cluster: str, experiment_name: str, *, con: SQLBackend | None = None
) -> None:
    """Refuse raw exposure rows carrying two non-null cluster labels for
    one unit -- a malformed cluster source, caught before dedup (e.g.
    `first_exposures`) can silently pick one and hide the conflict.
    """
    observed = exposures.filter(exposures[cluster].notnull())
    labels = observed.group_by("unit_id", "experiment_id").agg(n_labels=observed[cluster].nunique())
    conflicts = labels.filter(labels.n_labels > 1)
    query = conflicts.order_by("experiment_id", "unit_id").limit(5)
    sample = con.execute(query) if con is not None else query.execute()
    if not sample.empty:
        refuse(
            _CLUSTER_CONFLICT,
            cluster=cluster,
            experiment=experiment_name,
            units=tuple(str(unit) for unit in sample["unit_id"]),
        )


def arm_counts(con: SQLBackend, population: Table) -> dict[str, int]:
    """Unit counts per arm off an already-narrowed population table."""
    counts_tbl = population.group_by("group_id").agg(n=population.unit_id.count())
    rows = con.to_pyarrow(counts_tbl).to_pylist()
    return {str(r["group_id"]): int(r["n"]) for r in rows}


def mixed_assignment_snapshot(
    con: SQLBackend, exposure_events: Table, experiment: Experiment
) -> tuple[int, int, int]:
    """Count mixed and NULL assignments with one bounded aggregate."""
    integrity_tbl = mixed_assignment_units(exposure_events, experiment)
    rows = con.to_pyarrow(integrity_tbl).to_pylist()
    row = rows[0]
    return (
        int(row["mixed_count"]),
        int(row["unassigned_count"]),
        int(row["fingerprint"]),
    )


def mixed_assignment_count(con: SQLBackend, exposure_events: Table, experiment: Experiment) -> int:
    """Count units observed in more than one real arm."""
    return mixed_assignment_snapshot(con, exposure_events, experiment)[0]


def enforce_mixed_assignments(
    count: int,
    policy: Literal["error", "warn", "exclude"],
    *,
    unassigned_count: int = 0,
    already_warned: bool,
) -> bool:
    """Raise/warn per policy for mixed and NULL assignment units.

    One ``on_mixed_assignment`` policy governs both invalid assignment
    conditions: units seen in more than one arm and units with at least one NULL
    assignment label.
    """
    findings: list[str] = []
    if count:
        noun = "unit" if count == 1 else "units"
        findings.append(f"{count} {noun} appeared in more than one arm")
    if unassigned_count:
        noun = "unit" if unassigned_count == 1 else "units"
        findings.append(f"{unassigned_count} {noun} had no assignment label")
    if not findings:
        return already_warned
    finding = "; ".join(findings)
    if policy == "error":
        refuse(
            _MIXED_ASSIGNMENTS if count else _UNASSIGNED_ASSIGNMENTS,
            finding=finding,
            mixed_count=count,
            unassigned_count=unassigned_count,
            route="pass on_mixed_assignment='warn' or 'exclude', or repair assignment upstream",
        )
    if policy == "warn" and not already_warned:
        # Preserve the established wording for mixed-only warnings while
        # naming NULL assignments distinctly.
        suffix = (
            " and was excluded" if len(findings) == 1 else "; invalid assignments were excluded"
        )
        warn(
            _MIXED_ASSIGNMENTS_EXCLUDED_WARNING,
            context={"finding": finding, "suffix": suffix},
            stacklevel=4,
        )
        return True
    return already_warned


def validate_cluster_labels(
    exposures: Table,
    cluster: str,
    experiment_name: str,
    *,
    enforce_purity: bool = True,
    con: SQLBackend | None = None,
) -> None:
    """Reject missing labels and, for cluster randomization, labels spanning arms."""
    missing = exposures.filter(exposures[cluster].isnull()).count()
    n_null = int(cast("int", con.execute(missing) if con is not None else missing.execute()))
    if n_null:
        refuse(
            _CLUSTER_LABELS,
            experiment=experiment_name,
            cluster=cluster,
            kind="null labels",
            value=n_null,
            constraint="every enrolled unit must have a cluster label",
        )
    if not enforce_purity:
        return
    span = exposures.group_by(cluster).agg(n_groups=exposures.group_id.nunique())
    spanning = span.filter(span.n_groups > 1).count()
    n_span = int(cast("int", con.execute(spanning) if con is not None else spanning.execute()))
    if n_span:
        refuse(
            _CLUSTER_LABELS,
            experiment=experiment_name,
            cluster=cluster,
            kind="cross-arm labels",
            value=n_span,
            constraint="each randomization-grain cluster belongs to exactly one arm",
        )


def validate_trigger_fires_in_every_arm(rates: dict[str, float], trigger: str | None) -> None:
    """Refuse a declared trigger that never fires in a declared arm.

    A trigger that fires in one arm only is the treatment, not an
    eligibility rule - narrowing on it conditions the population on
    assignment and the comparison is no longer randomized.
    """
    silent = sorted(arm for arm, rate in rates.items() if rate == 0.0)
    if silent:
        refuse(
            _TRIGGER_ARM_MISSING,
            trigger=trigger,
            silent=tuple(silent),
            rates={a: round(r, 4) for a, r in rates.items()},
            route="analyze the assigned population, or declare an eligibility fact observable in every arm",
        )
