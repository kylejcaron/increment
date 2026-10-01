"""Immutable source-context state and the `MomentSource` protocol.

Leaf module under `increment.decision`: only `TYPE_CHECKING` imports
point upward at `decision`, `sources`, `semantics`, or `_analysis_config`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import date, datetime
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Protocol, cast, runtime_checkable

from increment._moment_plan import COMPLIANCE_ARM_FROM_CLUSTER_ROW
from increment.errors import (
    CapabilityError,
    CodedError,
    InvalidRequestError,
    RefusalSpec,
    raiser,
    refusals,
)

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

    from increment._analysis_config import ResolvedMetricConfig
    from increment.decision import CompiledDecisionPlan
    from increment.semantics.design import Encouragement, Observational, Randomized
    from increment.semantics.models import Metric
    from increment.sources import SourceOperation
Grain = Literal["total", "daily", "asof"]


def raise_legacy_compliance_state(*, study_id: str, reason: str) -> NoReturn:
    """Refuse identified legacy state that cannot recover cohort uptake."""
    _raise("source.compliance_summary.legacy_uptake_state", study_id=study_id, reason=reason)


def validate_compliance_completion(design: Encouragement, *, as_of: object | None) -> None:
    """Reject completion policies that cannot identify a finalized cohort."""
    if as_of is None or design.uptake.window_days is None:
        _raise("source.compliance_summary.completed_window_required")


def validate_compliance_design_match(summary: ComplianceSummary, design: Encouragement) -> None:
    """Refuse a reloaded/reused ``ComplianceSummary`` whose own design/cohort/
    time identity disagrees with the *design* the caller is requesting it
    for -- the "design/window mismatch" every producer/consumer validates
    before a readout uses the summary."""
    if not summary.matches_design(design):
        _raise(
            "source.compliance_summary.design_mismatch",
            study_id=summary.study_id,
            cohort=summary.cohort,
            window_days=summary.window_days,
            one_sided=summary.one_sided,
            design_fact=design.uptake.fact,
            design_window_days=design.uptake.window_days,
            design_one_sided=design.one_sided,
            control_group=summary.control_group,
            design_control_group=str(design.control_group),
        )


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "source.compliance_arm.n_units_positive": "ComplianceArm for group {group_id!r} needs n_units >= 1, got {n_units}",
        "source.compliance_arm.uptake_total_bounds": "ComplianceArm for group {group_id!r}: uptake_total={uptake_total!r} must be finite and within [0, n_units={n_units}]",
        "source.compliance_arm.partial_cluster_family": "ComplianceArm for group {group_id!r} carries a partial cluster-uptake bivariate family -- ref_uptake/cluster_uptake1/cluster_uptake2/ref_size/cluster_size1/cluster_size2/cluster_cross must all be present together or all be None",
        "source.compliance_arm.cluster_family_declaration_mismatch": "ComplianceArm for group {group_id!r}: n_clusters={n_clusters!r} but cluster-uptake bivariate family present={has_family} -- both must agree",
        "source.compliance_arm.n_clusters_positive": "ComplianceArm for group {group_id!r} needs n_clusters >= 1, got {n_clusters}",
        "source.compliance_arm.cluster_family_finite": "ComplianceArm for group {group_id!r}: {name}={value!r} is not finite",
        "source.compliance_summary.duplicate_arms": "ComplianceSummary for study {study_id!r} carries duplicate arm(s) {duplicates!r} -- exactly one canonical arm per group_id is required",
        "source.compliance_summary.cluster_family_mismatch": "ComplianceSummary for study {study_id!r}: declared cluster={declared_cluster!r} but arm {group_id!r} carries cluster-uptake family present={has_family} -- every arm must match the summary's own declared cluster grain",
        "source.compliance_summary.legacy_uptake_state": RefusalSpec(
            "source.compliance_summary.legacy_uptake_state",
            CapabilityError,
            template="ComplianceSummary for study {study_id!r} is unavailable from this reloaded source: {reason} -- re-export moments/artifacts built after compliance_summary support was added",
        ),
        "source.compliance_summary.design_mismatch": "ComplianceSummary for study {study_id!r} was built for cohort={cohort!r} window_days={window_days!r} one_sided={one_sided!r}, but the requested design declares uptake.fact={design_fact!r} window_days={design_window_days!r} one_sided={design_one_sided!r}; control={control_group!r} versus {design_control_group!r} -- these must agree before a caller can trust it",
        "source.compliance_summary.completed_window_required": "completed compliance requires an as-of day and a bounded uptake window",
        "source.compliance_summary.invalid_state": "Invalid compliance sufficient state: {reason}",
    },
)
_raise = raiser(_REFUSALS)


def invalid_compliance_state(reason: str) -> NoReturn:
    """Refuse malformed current state, separately from absent legacy state."""
    _raise("source.compliance_summary.invalid_state", reason=reason)


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_compliance_source_match(
    summary: ComplianceSummary, context: SourceContext, *, as_of: object | None = None
) -> None:
    """Keep source identity and the requested time bound attached to the data."""
    if (summary.study_id, summary.cluster, summary.as_of) != (
        context.study_id,
        context.cluster,
        as_of,
    ):
        invalid_compliance_state("compliance source, cluster or as-of identity mismatch")


#: ComplianceArm's cluster-grain wire fields, in the compliance plan's order
#: (COMPLIANCE_ARM_FROM_CLUSTER_ROW also maps each to its group_summary slot).
_CLUSTER_UPTAKE_FIELDS = tuple(
    name for name in COMPLIANCE_ARM_FROM_CLUSTER_ROW if name != "n_clusters"
)


@dataclass(frozen=True, slots=True)
class ComplianceArm:
    """One arm's design-level uptake sufficient state.

    Unit count and uptake total, computed on the design's own declared
    enrollment/uptake cohort -- never any outcome metric's moments, so
    metric declaration order, count, and per-metric missingness cannot
    change it. When a randomization/dependence cluster is declared, also
    carries the **complete centered bivariate moments** of cluster uptake
    totals ``U_g`` and cluster sizes ``M_g``: each family's reference mean,
    first residual sum, second residual sum, and their cross moment --
    the same sufficient state
    :func:`~increment.estimation.variance.cluster_uptake_moments` reads off
    an ``ArmStats`` row with ``x_role="uptake_total"``. These fields are
    never derived by averaging per-cluster uptake *rates*: a member-weighted
    ratio-of-totals needs the totals themselves.

    ``n_clusters`` is the independent cluster count ``K``; ``n_units`` is
    the true unit count. The two are tracked separately because, at cluster
    grain, the moments above are computed over the ``K`` clusters, not the
    ``n_units`` members -- collapsing them onto a single field the way a
    cluster-grain ``ArmStats.n`` does would discard one of the two counts.
    """

    group_id: str
    n_units: int
    uptake_total: float
    n_clusters: int | None = None
    ref_uptake: float | None = None
    cluster_uptake1: float | None = None
    cluster_uptake2: float | None = None
    ref_size: float | None = None
    cluster_size1: float | None = None
    cluster_size2: float | None = None
    cluster_cross: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id:
            invalid_compliance_state("group_id must be a nonempty string")
        if not _integer(self.n_units) or self.n_units < 1:
            _raise(
                "source.compliance_arm.n_units_positive",
                group_id=self.group_id,
                n_units=self.n_units,
            )
        if not _number(self.uptake_total) or not (0.0 <= self.uptake_total <= self.n_units):
            _raise(
                "source.compliance_arm.uptake_total_bounds",
                group_id=self.group_id,
                uptake_total=self.uptake_total,
                n_units=self.n_units,
            )
        cluster_values = tuple(getattr(self, name) for name in _CLUSTER_UPTAKE_FIELDS)
        present = [v for v in cluster_values if v is not None]
        if present and len(present) != len(cluster_values):
            _raise("source.compliance_arm.partial_cluster_family", group_id=self.group_id)
        has_family = bool(present)
        if (self.n_clusters is not None) != has_family:
            _raise(
                "source.compliance_arm.cluster_family_declaration_mismatch",
                group_id=self.group_id,
                n_clusters=self.n_clusters,
                has_family=has_family,
            )
        if self.n_clusters is not None and (
            not _integer(self.n_clusters) or not 1 <= self.n_clusters <= self.n_units
        ):
            _raise(
                "source.compliance_arm.n_clusters_positive",
                group_id=self.group_id,
                n_clusters=self.n_clusters,
            )
        for name, value in zip(_CLUSTER_UPTAKE_FIELDS, cluster_values, strict=True):
            if value is not None and not _number(value):
                _raise(
                    "source.compliance_arm.cluster_family_finite",
                    group_id=self.group_id,
                    name=name,
                    value=value,
                )

        if self.n_clusters is not None:
            try:
                self._validate_bivariate()
            except OverflowError:
                invalid_compliance_state("cluster state cannot represent the declared totals")

    def _validate_bivariate(self) -> None:
        k = self.n_clusters
        assert k is not None
        values = [getattr(self, name) for name in _CLUSTER_UPTAKE_FIELDS]
        ru, u1, u2, rm, m1, m2, um = map(Fraction, values)
        for total, reconstructed in ((self.uptake_total, k * ru + u1), (self.n_units, k * rm + m1)):
            # Accumulation and reference recovery each round at their own scale.
            slack = 64 * math.ulp(float(max(abs(total), abs(reconstructed), 1)))
            if abs(reconstructed - Fraction(total)) > slack:
                invalid_compliance_state("cluster references/residuals disagree with totals")
        vu, vm, cov = u2 - u1 * u1 / k, m2 - m1 * m1 / k, um - u1 * m1 / k
        slack = Fraction(64 * math.ulp(float(max(abs(u2), abs(m2), abs(um), 1))))
        if u2 < 0 or m2 < 0 or vu < -slack or vm < -slack:
            invalid_compliance_state("cluster second moments are inconsistent")
        if cov * cov > max(vu, 0) * max(vm, 0) + slack * (abs(vu) + abs(vm) + slack):
            invalid_compliance_state("cluster cross moment violates covariance bounds")

    def to_wire(self) -> dict[str, object]:
        """JSON-serializable payload for one arm."""
        return {
            "group_id": self.group_id,
            "n_units": self.n_units,
            "uptake_total": self.uptake_total,
            "n_clusters": self.n_clusters,
            "ref_uptake": self.ref_uptake,
            "cluster_uptake1": self.cluster_uptake1,
            "cluster_uptake2": self.cluster_uptake2,
            "ref_size": self.ref_size,
            "cluster_size1": self.cluster_size1,
            "cluster_size2": self.cluster_size2,
            "cluster_cross": self.cluster_cross,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, object]) -> ComplianceArm:
        """Parse complete typed arm state, refusing malformed current payloads."""
        payload = cast("Mapping[str, Any]", cast_mapping(payload))
        expected = {"group_id", "n_units", "uptake_total", "n_clusters", *_CLUSTER_UPTAKE_FIELDS}
        if set(payload) != expected:
            invalid_compliance_state("arm payload has missing or unknown fields")
        return cls(
            group_id=payload["group_id"],
            n_units=payload["n_units"],
            uptake_total=payload["uptake_total"],
            n_clusters=payload["n_clusters"],
            **{name: payload[name] for name in _CLUSTER_UPTAKE_FIELDS},
        )


@dataclass(frozen=True, slots=True)
class ComplianceSummary:
    """Design-level compliance sufficient state: one canonical arm per group.

    Built by ``MomentSource.compliance_summary``, independent of any
    outcome metric's declaration, order, count, or missingness -- the
    population is the design's own declared enrollment/uptake cohort
    (``design.uptake.fact``, honoring ``design.uptake.window_days`` and
    ``as_of``). ``cohort``/``window_days``/``one_sided``/``cluster``/
    ``as_of`` are this summary's design/cohort/time identity: a consumer
    combining or reloading summaries validates them before use rather than
    trusting a numerically-agreeing-by-accident pair.

    Frozen and duplicate-checked at construction: mutating a caller's own
    nested arms sequence after passing it in cannot alter this summary
    (``arms`` is copied into an immutable tuple), and reassigning any field
    raises.
    """

    study_id: str
    cohort: str
    window_days: int | None
    one_sided: bool
    cluster: str | None
    as_of: object | None
    arms: tuple[ComplianceArm, ...]
    control_group: str = "control"

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.study_id, self.cohort, self.control_group)
        ):
            invalid_compliance_state(
                "study, cohort and control identities must be nonempty strings"
            )
        if not isinstance(self.one_sided, bool):
            invalid_compliance_state("one_sided must be Boolean")
        if self.window_days is not None and (
            not _integer(self.window_days) or self.window_days < 1
        ):
            invalid_compliance_state("window_days must be a positive integer or null")
        if self.cluster is not None and (not isinstance(self.cluster, str) or not self.cluster):
            invalid_compliance_state("cluster must be a nonempty string or null")
        if self.as_of is not None and not (
            isinstance(self.as_of, (date, str)) or _number(self.as_of)
        ):
            invalid_compliance_state(
                "as_of must be a calendar, string or finite numeric day scalar"
            )
        if not isinstance(self.arms, Sequence) or not all(
            isinstance(arm, ComplianceArm) for arm in self.arms
        ):
            invalid_compliance_state("arms must contain typed ComplianceArm values")
        arms = tuple(sorted(self.arms, key=lambda arm: arm.group_id))
        object.__setattr__(self, "arms", arms)
        seen: set[str] = set()
        duplicates: set[str] = set()
        for arm in arms:
            if arm.group_id in seen:
                duplicates.add(arm.group_id)
            seen.add(arm.group_id)
        if duplicates:
            _raise(
                "source.compliance_summary.duplicate_arms",
                study_id=self.study_id,
                duplicates=tuple(sorted(duplicates)),
            )
        clustered = self.cluster is not None
        for arm in arms:
            has_family = arm.n_clusters is not None
            if has_family != clustered:
                _raise(
                    "source.compliance_summary.cluster_family_mismatch",
                    study_id=self.study_id,
                    group_id=arm.group_id,
                    declared_cluster=self.cluster,
                    has_family=has_family,
                )

    def arm(self, group_id: str) -> ComplianceArm | None:
        """The canonical arm for *group_id*, or ``None`` if absent."""
        return next((a for a in self.arms if a.group_id == group_id), None)

    def matches_design(self, design: Encouragement) -> bool:
        """Whether this summary's own identity agrees with *design*'s
        declared uptake cohort -- the "design/window mismatch" check a
        reloaded (wire) summary must pass before a caller trusts it."""
        return (
            self.control_group == str(design.control_group)
            and self.cohort == design.uptake.fact
            and self.window_days == design.uptake.window_days
            and self.one_sided == design.one_sided
        )

    def to_wire(self) -> dict[str, object]:
        """Version-1 payload, with sorted arms and explicit nullable time identity.

        Total cubes require as_of=null. Direct summary serialization retains
        ISO dates and numbers; strings and datetimes are tagged to retain their type.
        """
        return {
            "version": 1,
            "study_id": self.study_id,
            "control_group": self.control_group,
            "as_of": (
                self.as_of.isoformat()
                if type(self.as_of) is date
                else {"datetime": self.as_of.isoformat()}
                if isinstance(self.as_of, datetime)
                else {"label": self.as_of}
                if isinstance(self.as_of, str)
                else self.as_of
            ),
            "cohort": self.cohort,
            "window_days": self.window_days,
            "one_sided": self.one_sided,
            "cluster": self.cluster,
            "arms": [arm.to_wire() for arm in self.arms],
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, object], *, study_id: str) -> ComplianceSummary:
        """Parse and freeze current state; malformed state has a coded refusal."""
        payload = cast("Mapping[str, Any]", cast_mapping(payload))
        expected = {
            "version",
            "study_id",
            "control_group",
            "as_of",
            "cohort",
            "window_days",
            "one_sided",
            "cluster",
            "arms",
        }
        if (
            set(payload) != expected
            or type(payload["version"]) is not int
            or payload["version"] != 1
        ):
            invalid_compliance_state("unsupported version or incomplete compliance payload")
        if payload["study_id"] != study_id:
            invalid_compliance_state("compliance study identity differs from cube")
        arms_payload = payload["arms"]
        if not isinstance(arms_payload, (list, tuple)):
            invalid_compliance_state("arms must be a sequence")
        try:
            as_of = payload["as_of"]
            if isinstance(as_of, str):
                as_of = date.fromisoformat(as_of)
            elif isinstance(as_of, Mapping):
                if set(as_of) == {"label"} and isinstance(as_of["label"], str):
                    as_of = as_of["label"]
                elif set(as_of) == {"datetime"} and isinstance(as_of["datetime"], str):
                    as_of = datetime.fromisoformat(as_of["datetime"])
                else:
                    invalid_compliance_state("invalid tagged as-of scalar")
            return cls(
                study_id=study_id,
                control_group=payload["control_group"],
                cohort=payload["cohort"],
                window_days=payload["window_days"],
                one_sided=payload["one_sided"],
                cluster=payload["cluster"],
                as_of=as_of,
                arms=tuple(ComplianceArm.from_wire(cast_mapping(row)) for row in arms_payload),
            )
        except CodedError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            invalid_compliance_state(f"malformed compliance payload: {exc}")


def cast_mapping(value: object) -> Mapping[str, object]:
    """Reject a non-mapping arm row before ``ComplianceArm.from_wire``
    would otherwise raise a less specific error."""
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        invalid_compliance_state("expected an arm row mapping with string keys")
    return cast("Mapping[str, object]", value)


@dataclass(frozen=True, slots=True)
class SourceContext:
    """Immutable statistical construction state owned by a moment source."""

    study_id: str
    design: Randomized | Encouragement | Observational | None
    plan: CompiledDecisionPlan
    metrics: tuple[Metric, ...]
    configs: tuple[ResolvedMetricConfig, ...]
    cluster: str | None
    #: Declared whole-cluster policy-deployment grain: "cluster" only for an
    #: explicit cluster-level intervention, otherwise "unit". Neither `cluster`
    #: (a randomization/dependence grain) nor covariance clustering implies it.
    #: Unit-frame-serving sources pass the declared value; others keep "unit".
    intervention_grain: Literal["unit", "cluster"] = "unit"

    # SourceContext is construction state shared by every population view.
    # Triggered population selection belongs to the source-owned adapter.


@runtime_checkable
class MomentSource(Protocol):
    """A source of aggregated moments, decoupled from how they were computed."""

    capabilities: frozenset[Grain]
    operations: frozenset[SourceOperation]

    @property
    def context(self) -> SourceContext:
        """Immutable design, plan, metric, cluster, and study state."""
        ...

    @property
    def breakouts(self) -> Sequence[str]:
        """Unit-stable reporting identities this source can expose."""
        ...

    @property
    def shape(self) -> Literal["unit_summary", "unit_panel"] | None:
        """The source shape when it is owned by a dataframe adapter."""
        ...

    def moments(
        self,
        metric: Metric,
        *,
        grain: Grain = "total",
        by: Sequence[str] = (),
        completed_windows_only: bool = False,
        include_covariate: bool = False,
    ) -> Sequence[Mapping[str, Any]]:
        """Centered moments for `metric` at `grain`, optionally split by dimension.

        `completed_windows_only` applies only to "asof"-capable sources: it
        gates a retention series to closed (final) units, not the default
        open (provisional) one. Ignored elsewhere.
        """
        ...

    def unit_frame(self, metric: Metric, *, covariates: Sequence[str] = ()) -> IntoDataFrame:
        """One row per unit: `unit_id`, `group_id`, `y`, plus `covariates`.

        For a ratio metric, `y` is the numerator only; `y_den` (the
        per-unit denominator) is a separate column a caller combines with `y`.
        A declared `context.cluster` adds `cluster_id` metadata automatically,
        independently of `covariates`; callers must not request its source name.
        """
        ...

    def unit_counts(self) -> dict[str, int]: ...

    def cluster_counts(self) -> dict[str, int]:
        """Per-group distinct randomization-cluster counts.

        Only a source with a declared cluster column can answer this; others
        raise `cluster-grain capability refusal` instead of returning
        unit-grain counts under a cluster-grain name.
        """
        ...

    def compliance_dates(self) -> Sequence[object]:
        """Enrollment/uptake day labels, independent of outcome filters.

        Sources without as-of compliance support raise a coded capability refusal.
        """
        ...

    def compliance_summary(
        self,
        design: Encouragement,
        *,
        as_of: object | None = None,
        completed_windows_only: bool = False,
    ) -> ComplianceSummary:
        """Design-level uptake sufficient state, independent of any metric.

        Computed on `design`'s own declared enrollment/uptake cohort
        (`design.uptake.fact`, honoring `design.uptake.window_days`):
        declaring a different outcome metric, changing metric count or
        order, or an outcome metric's own missingness policy never changes
        the result. `as_of`, when given, matches this source's enrollment
        day-axis `ds` values returned by `compliance_dates()` and freezes
        cumulative uptake state through that day; `None` reads the declared window
        in full (the "ever" / whole-window cohort). ``completed_windows_only``
        requires an as-of day and a bounded uptake window, and admits only
        units whose uptake window has closed by that day.
        """
        ...

    def sql(self, *, grain: Grain = "total") -> dict[str, str]: ...

    def close(self) -> None: ...


SourceRoute = Literal["artifact", "native", "moments", "panel"]


def classify_source(src: MomentSource) -> SourceRoute:
    """Route a source for the readout families that branch on it: artifact sources
    declare the `artifact_experiment` operation; others are told apart by operations and shape."""
    operations = getattr(src, "operations", frozenset())
    if "artifact_experiment" in operations:
        return "artifact"
    if "moments_source" not in operations:
        shape = getattr(src, "shape", None)
        return "panel" if shape == "unit_panel" else "moments"
    return "native"


def compliance_summary_series(
    src: MomentSource,
    design: Encouragement,
    *,
    completed_windows_only: bool,
) -> dict[Any, ComplianceSummary]:
    """Read compliance on its own enrollment/uptake axis, never an outcome axis."""
    provider = getattr(src, "compliance_dates", None)
    if not callable(provider):
        raise_legacy_compliance_state(
            study_id=src.context.study_id,
            reason="source must implement the enrollment/uptake compliance_dates() contract",
        )
    # One series is one readout: a warehouse source checks assignment
    # integrity once here rather than on every per-date read.
    validated = getattr(src, "_validated_assignments", nullcontext)
    summaries = {}
    with validated():
        dates = provider()
        for ds in dict.fromkeys(dates):
            summary = src.compliance_summary(
                design, as_of=ds, completed_windows_only=completed_windows_only
            )
            validate_compliance_source_match(summary, src.context, as_of=ds)
            if summary.arms:
                summaries[ds] = summary
    return summaries


class RawOutcomeSource(Protocol):
    """A source that can expose exact pre-transform unit outcomes."""

    def unit_frame(
        self,
        metric: Metric,
        *,
        covariates: Sequence[str] = (),
        outcome_stage: Literal["transformed", "raw"] = "transformed",
    ) -> IntoDataFrame:
        """Return transformed or exact raw unit outcomes."""
        ...
