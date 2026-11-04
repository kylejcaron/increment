"""Small, source-free dashboard values for presentation contracts."""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import functools
import io
import lzma
import math
import pickle
from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel

from increment import Experiment, Scope, SourceReference
from increment._source_operations import DashboardExploreCapture, DashboardGroupData
from increment.dashboard import DashboardConfig, DashboardSnapshot
from increment.dashboard._data import DashboardOverview, _binding_fingerprint, enriched_rows
from increment.errors import CodedError
from increment.estimation.readout_types import (
    CellKey,
    CellRecord,
    IntegrityResult,
    PopulationRoster,
    ReadoutCapture,
    ReadoutMetadata,
    ReadoutScope,
    SourceReadoutScope,
    cell_order,
)
from increment.kernel.contracts import Registration
from increment.results import (
    BreakoutEstimate,
    DailyLiftEstimate,
    DailyLiftEstimates,
    DailyMetricValue,
    DailyMetricValues,
    Estimate,
    ExperimentHealth,
    HealthCheck,
    LiftEstimate,
    LiftEstimates,
    SRMResult,
)
from increment.semantics.models import ConversionMetric, MeanMetric
from tests.parity_harness.comparison import nested_close

COUNTRY = ("event_log", "country")
ADDED = "purchases_per_user"


def experiment_for(value: DashboardSnapshot) -> Experiment:
    """Describe a snapshot without binding a source or resolver."""
    return Experiment(
        experiment_id=value.experiment_name,
        design={},
        metrics={model.name: model for model in (*value.metrics, *value.offered_metrics)},
        registration=Registration(registration_hash="presentation"),
        source_refs=(SourceReference(name="events", uri="memory://presentation"),),
        scopes=(Scope(view="run", grain="total"),),
        analysis_context={"experiment": {"name": value.experiment_name}},
    )


def _group(metric, arm, count, value) -> DashboardGroupData:
    return DashboardGroupData(
        metric=metric.name,
        group_id=arm,
        assigned_units=count,
        eligible_units=count,
        observed_value=value,
        analysis_input_value=value,
        sum_value=value * count,
        event_count=count,
        retained_units=round(value * count) if metric.type == "conversion" else None,
        numerator=None,
        denominator=None,
        excluded_not_mature=0,
        excluded_no_observed_day=0,
        excluded_other=0,
        observation_end=dt.date(2025, 2, 14),
        window_start_days=0,
        window_end_days=14,
        source_kind="pinned_warehouse",
        prefix_id=None,
        unavailable={
            "retained_units": "not a binary outcome",
            "numerator": "not a ratio metric",
            "denominator": "not a ratio metric",
            "prefix_id": "no retained checkpoint",
        },
    )


def temporal_rows(value: DashboardSnapshot, view: str, *, segmented: bool = False):
    """Two dates distinguish metrics, arms, scopes, and unavailable points."""
    places = ("US", "CA") if segmented else (None,)
    if view == "cumulative_lift":
        return DailyLiftEstimates(
            DailyLiftEstimate(
                **estimate.model_dump(exclude={"ds"}),
                ds=dt.date(2025, 1, 15) + dt.timedelta(days=day),
                dimension="country" if segmented else None,
                dimension_value=place,
                source="event_log" if segmented else None,
            )
            for estimate in value.estimates
            for place in places
            for day in (0, 1)
        )
    return DailyMetricValues(
        DailyMetricValue(
            metric=model.name,
            group_id=arm,
            ds=dt.date(2025, 1, 15) + dt.timedelta(days=day),
            ds_basis="calendar",
            value=None if day == 0 else Estimate(value=0.3, lb=0.2, ub=0.4, level=0.95),
            unavailable="nonpositive_mean" if day == 0 else None,
            n=10,
            dimension="country" if segmented else None,
            dimension_value=place,
            source="event_log" if segmented else None,
        )
        for model in value.metrics
        for arm in (value.control_group, value.treatment_group)
        for place in places
        for day in (0, 1)
    )


def snapshot(**overrides: Any) -> DashboardSnapshot:
    """Construct a complete dashboard reading; overrides are snapshot fields."""
    metrics = (
        ConversionMetric(
            name="checkout_conversion",
            entity="user_id",
            fact="purchase",
            description="Did the user check out within 14 days of exposure?",
            preferred_direction="increase",
            window_days=14,
        ),
        MeanMetric(
            name="revenue_per_user",
            entity="user_id",
            fact="purchase",
            aggregation="sum",
            preferred_direction="increase",
            window_days=14,
        ),
        MeanMetric(
            name="session_seconds",
            entity="user_id",
            fact="session_end",
            aggregation="sum",
            preferred_direction="decrease",
            window_days=14,
        ),
    )
    estimates = tuple(
        LiftEstimate(
            metric=model.name,
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            role=role,
            multiplicity_status="declared_plan",
            source_snapshot_id="presentation",
            decision_scope_complete=True,
            sampling_available=True,
            alternative="less" if role == "guardrail" else "two-sided",
            preferred_direction=model.preferred_direction,
            lift=Estimate(value=effect, lb=effect - 0.03, ub=effect + 0.03, level=level),
        )
        for model, role, effect, level in zip(
            metrics,
            ("primary", "secondary", "guardrail"),
            (0.1, 0.2, -0.05123456789),
            (0.975, 0.975, 0.95),
            strict=True,
        )
    )
    collection = _collection(estimates, ("control", "treatment"))
    allocation = SRMResult(
        inference="always_valid",
        chi2_stat=0,
        fixed_p_value=1,
        log_e_value=-3,
        df=1,
        is_srm=False,
        alpha=0.001,
        observed={"control": 600, "treatment": 600},
        expected={"control": 0.5, "treatment": 0.5},
        min_expected_count=600,
    )
    health = ExperimentHealth(
        checks=(
            HealthCheck(
                check="assigned_allocation",
                analysis_population="assigned",
                availability="available",
                conclusion="not_rejected",
                context={},
                srm=allocation,
            ),
            *(
                HealthCheck(
                    check=kind,
                    analysis_population="triggered",
                    availability="not_applicable",
                    context={"reason": "This experiment does not declare a trigger."},
                )
                for kind in (
                    "triggered_balance",
                    "trigger_rates",
                    "trigger_maturity",
                    "feed_completeness",
                    "trigger_assumptions",
                )
            ),
        )
    )
    values: dict[str, Any] = {
        "experiment_name": "storefront_refresh",
        "binding_fingerprint": "presentation",
        "title": "storefront_refresh",
        "description": "A refreshed storefront",
        "start": dt.datetime(2025, 1, 15, tzinfo=dt.UTC),
        "end": dt.datetime(2025, 2, 14, tzinfo=dt.UTC),
        "control_group": "control",
        "treatment_group": "treatment",
        "primary_metric": metrics[0].name,
        "metrics": metrics,
        "breakouts": (COUNTRY,),
        "estimates": estimates,
        "readout_rows": enriched_rows(collection),
        "readout_collection": collection,
        "allocation": allocation,
        "allocation_refusal": None,
        "allocation_history": tuple(
            {
                "experiment_id": "storefront_refresh",
                "ds": dt.date(2025, 1, 15),
                "group_id": arm,
                "n_daily": 600,
                "n_cumulative": 600,
            }
            for arm in ("control", "treatment")
        ),
        "allocation_history_refusal": None,
        "config": DashboardConfig(expected_allocation={"control": 0.5, "treatment": 0.5}),
        "computed_at": dt.datetime(2025, 2, 15, tzinfo=dt.UTC),
        "group_data": tuple(
            _group(model, arm, 600, 0.2) for model in metrics for arm in ("control", "treatment")
        ),
        "health": health,
    }
    values.update(overrides)
    result = DashboardSnapshot(**values)
    captures = {}
    for breakout in (None, *result.breakouts):
        for view in ("cumulative_lift", "daily_values", "cumulative_values"):
            rows = temporal_rows(result, view, segmented=breakout is not None)
            for name in (None, *(model.name for model in result.metrics)):
                selected = type(rows)(row for row in rows if name is None or row.metric == name)
                for completed in (False, True) if view != "daily_values" else (False,):
                    key = ("assigned", view, name, completed, breakout)
                    captures[key] = DashboardExploreCapture.answered(
                        key, selected, collection=type(rows)
                    )
    result = replace(result, explore=captures if "explore" not in overrides else result.explore)
    if "binding_fingerprint" in overrides:
        return result
    experiment = experiment_for(result)
    return replace(
        result,
        binding_fingerprint=_binding_fingerprint(
            experiment.analysis_context["experiment"],
            experiment.metrics,
        ),
    )


def _collection(estimates, arms) -> LiftEstimates:
    source_id = "presentation"
    cells = tuple(sorted((CellKey.from_row(row) for row in estimates), key=cell_order))
    source = SourceReadoutScope(
        source_snapshot_id=source_id,
        cells=cells,
        decision_cells=cells,
        rosters=(
            PopulationRoster(
                analysis_population="assigned",
                arms=arms,
                source="experiment_counts",
                complete=True,
                control_group=arms[0],
            ),
        ),
        decision_complete_by_population={"assigned": True},
        integrity=(
            IntegrityResult(
                status="not_checked_missing_declaration",
                analysis_population="assigned",
                construction="none",
                alpha=None,
                observed=None,
                expected=None,
                randomization_grain=None,
                code="integrity.allocation_missing_or_nonconstant",
                context={"allocation": None},
            ),
        ),
    )
    metadata = ReadoutMetadata(
        scope=ReadoutScope(
            snapshot_id=source_id,
            cells=cells,
            decision_cells=cells,
            populations=("assigned",),
            by_source={source_id: source},
        ),
        cells=tuple(CellRecord(cell=cell, source_snapshot_id=source_id) for cell in cells),
    )
    return LiftEstimates(estimates, metadata=metadata)


def pricing_snapshot() -> DashboardSnapshot:
    model = ConversionMetric(
        name="purchase_rate",
        entity="user_id",
        fact="purchase",
        preferred_direction="increase",
        window_days=14,
    )
    estimate = LiftEstimate(
        metric=model.name,
        group_id="candidate",
        method="unadjusted",
        method_role="decision",
        role="primary",
        multiplicity_status="declared_plan",
        lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
        preferred_direction="increase",
    )
    estimate = estimate.model_copy(
        update={
            "source_snapshot_id": "presentation",
            "decision_scope_complete": True,
            "sampling_available": True,
        }
    )
    allocation = SRMResult(
        inference="always_valid",
        chi2_stat=0,
        fixed_p_value=1,
        log_e_value=-3,
        df=1,
        is_srm=False,
        alpha=0.001,
        observed={"baseline": 700, "candidate": 300},
        expected={"baseline": 0.7, "candidate": 0.3},
        min_expected_count=300,
    )
    base = snapshot()
    assert base.health is not None
    return snapshot(
        experiment_name="pricing_copy",
        title="Pricing copy rewrite",
        control_group="baseline",
        treatment_group="candidate",
        primary_metric=model.name,
        metrics=(model,),
        breakouts=(),
        estimates=(estimate,),
        readout_rows=enriched_rows(_collection((estimate,), ("baseline", "candidate"))),
        allocation=allocation,
        readout_collection=_collection((estimate,), ("baseline", "candidate")),
        group_data=tuple(
            _group(model, arm, count, 0.2) for arm, count in allocation.observed.items()
        ),
        config=DashboardConfig(
            expected_allocation=allocation.expected, provenance={"Fixture": "in-memory"}
        ),
        health=base.health.model_copy(
            update={
                "checks": (
                    base.health.checks[0].model_copy(update={"srm": allocation}),
                    *base.health.checks[1:],
                )
            }
        ),
    )


def overview_snapshot(*, selected: bool = True) -> DashboardSnapshot:
    model = MeanMetric(
        name=ADDED,
        description="Added purchase count per enrolled user.",
        entity="user_id",
        fact="purchase",
        aggregation="count",
        preferred_direction="increase",
        window_days=14,
    )
    common: dict[str, Any] = {
        "method": "mean",
        "method_role": "decision",
        "role": "exploratory",
        "lift": Estimate(value=0.2, lb=0.05, ub=0.35, level=0.95),
        "preferred_direction": "increase",
        "family_size": 2,
        "family_q": 0.05,
        "family_procedure": "bh",
        "family_nominal_alpha": 0.05,
        "family_guarantee": "finite_sample",
        "family_axes": ("metric", "breakout"),
        "discovery": True,
        "multiplicity_status": "exploratory_family",
    }
    whole = LiftEstimate(metric=ADDED, group_id="treatment", **common)
    segment = BreakoutEstimate(
        metric=ADDED,
        group_id="treatment",
        dimension="country",
        dimension_value="US",
        source="event_log",
        **common,
    )
    return snapshot(
        offered_metrics=(model,),
        exploratory_metrics=(model,) if selected else (),
        config=DashboardConfig(
            expected_allocation={"control": 0.5, "treatment": 0.5},
            exploratory_metrics=(ADDED,) if selected else (),
        ),
        overview=DashboardOverview(
            exploratory=(whole,), segments={COUNTRY: (segment,)}, family_size=2, family_q=0.05
        ),
    )


def binomial_region():
    from increment.estimation.binomial_rr import nuisance_beta
    from increment.results import BinomialConfidenceSet

    return BinomialConfidenceSet(
        lower=0.07,
        upper=0.13,
        alpha=0.025,
        level=0.975,
        decision_alpha=0.025,
        geometry="central",
        method="binomial_bb_difference_v3",
        numerical_qualification="scipy_special_function_error_model_conditional_v1",
        x_c=120,
        n_c=600,
        x_t=132,
        n_t=600,
        nuisance_beta=nuisance_beta(0.025),
    )


# Frozen preparations: ``prepare_dashboard`` output saved by
# ``uv run --extra dashboard python -m tests.test_dashboard_capture --freeze`` and pinned to a
# fresh capture of the same source by ``test_frozen_snapshot_matches_a_fresh_capture``.
FROZEN_DIR = Path(__file__).parent / "fixtures" / "dashboard"


@dataclasses.dataclass(frozen=True)
class FrozenDashboard:
    """One saved preparation with its experiment and independent engine reads.

    ``engine`` maps ``(view, metric, completed windows only, breakout)`` to what the public
    readout returned for that request when the snapshot was captured, or its refusal.
    """

    experiment: Experiment
    snapshot: DashboardSnapshot
    engine: Mapping[tuple[Any, ...], Any]


def _mapping_proxy(values: dict[Any, Any]) -> Mapping[Any, Any]:
    return MappingProxyType(values)


class _FrozenPickler(pickle.Pickler):
    def reducer_override(self, obj: Any) -> Any:
        if type(obj) is MappingProxyType:
            return _mapping_proxy, (dict(obj),)
        return NotImplemented


def write_frozen(
    name: str,
    *,
    experiment: Experiment,
    snapshot: DashboardSnapshot,
    engine: Mapping[tuple[Any, ...], Any] | None = None,
) -> Path:
    """Save one preparation exactly as captured."""
    buffer = io.BytesIO()
    _FrozenPickler(buffer, protocol=5).dump(
        {"experiment": experiment.to_json(), "snapshot": snapshot, "engine": dict(engine or {})}
    )
    path = FROZEN_DIR / f"{name}.pickle.xz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(lzma.compress(buffer.getvalue()))
    return path


@functools.cache
def frozen(name: str) -> FrozenDashboard:
    """The saved preparation ``name``; it holds no connection, resolver or source."""
    payload = pickle.loads(lzma.decompress((FROZEN_DIR / f"{name}.pickle.xz").read_bytes()))
    return FrozenDashboard(
        experiment=Experiment.from_json(payload["experiment"]),
        snapshot=payload["snapshot"],
        engine=MappingProxyType(payload["engine"]),
    )


def captured_form(
    value: Any, *, without: frozenset[str] = frozenset(), by_type: frozenset[str] = frozenset()
) -> Any:
    """A comparable tree of everything ``value`` holds; ``without`` names fields to leave out.

    Refusals keep their type, code and context; collections keep their rows and every
    attribute they carry; callables keep their qualified name. Fields named in ``by_type`` keep
    only their value's type. A pinned read's random capture identity is runtime-only and is
    left out. An unknown type is refused rather than skipped.
    """

    def entry(name: Any, value: Any) -> Any:
        return ("type", type(value).__qualname__) if name in by_type else walk(value)

    def walk(item: Any) -> Any:
        if isinstance(item, CodedError):
            return ("refusal", type(item).__qualname__, item.code, walk(item.context))
        if isinstance(item, BaseModel):
            names = [name for name in type(item).model_fields if name not in without]
            if isinstance(item, ReadoutCapture):
                names.remove("snapshot_id")
            return (
                type(item).__qualname__,
                {name: entry(name, getattr(item, name)) for name in names},
            )
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            fields = [field.name for field in dataclasses.fields(item)]
            return (
                type(item).__qualname__,
                {name: entry(name, getattr(item, name)) for name in fields if name not in without},
            )
        if isinstance(item, functools.partial):
            return ("partial", walk(item.func), walk(item.args), walk(item.keywords))
        if isinstance(item, type) or callable(item):
            return ("callable", item.__module__, item.__qualname__)
        if isinstance(item, enum.Enum):
            return (type(item).__qualname__, walk(item.value))
        if isinstance(item, Mapping):
            return {
                repr(walk(key)): entry(key, value)
                for key, value in item.items()
                if key not in without
            }
        if isinstance(item, list) and type(item) is not list:
            return (type(item).__qualname__, [walk(row) for row in item], walk(vars(item)))
        if isinstance(item, list | tuple):
            return [walk(entry) for entry in item]
        if isinstance(item, set | frozenset):
            return sorted((walk(entry) for entry in item), key=repr)
        if item is None or isinstance(item, bool | int | float | Fraction | str | bytes | dt.date):
            return item
        raise TypeError(f"no captured form for {type(item).__qualname__}")

    return walk(value)


def capture_differences(
    actual: Any,
    expected: Any,
    *,
    without: frozenset[str] = frozenset(),
    by_type: frozenset[str] = frozenset(),
) -> list[str]:
    """Every place two captured values differ.

    Floats agree to ``tests/parity_harness/comparison.py::TOLERANCE``, the tolerance for public
    readout values from different executions; every other leaf is compared exactly.
    """
    found: list[str] = []

    def compare(left: Any, right: Any, path: str) -> None:
        if isinstance(left, float) and isinstance(right, float):
            if (math.isnan(left) and math.isnan(right)) or nested_close(right, left):
                return
            found.append(f"{path}: {left!r} != {right!r}")
        elif type(left) is not type(right):
            found.append(f"{path}: {type(left).__name__} != {type(right).__name__}")
        elif isinstance(left, dict):
            for key in sorted(left.keys() | right.keys()):
                if key not in left or key not in right:
                    found.append(f"{path}[{key}]: present on one side only")
                else:
                    compare(left[key], right[key], f"{path}[{key}]")
        elif isinstance(left, list | tuple):
            if len(left) != len(right):
                found.append(f"{path}: {len(left)} entries != {len(right)}")
            for index, (one, other) in enumerate(zip(left, right, strict=False)):
                compare(one, other, f"{path}[{index}]")
        elif left != right:
            found.append(f"{path}: {left!r} != {right!r}")

    compare(
        captured_form(actual, without=without, by_type=by_type),
        captured_form(expected, without=without, by_type=by_type),
        "$",
    )
    return found


# Provenance digests of exact canonical float inputs; they promise no equality across
# aggregation orders (docs/guides/reading-results.md, "source_content_digest").
_EXACT_INPUT_DIGESTS = frozenset({"source_content_digest"})


def _group_rows(value: DashboardSnapshot) -> list[DashboardGroupData]:
    return sorted(
        value.group_data, key=lambda row: (row.analysis_population, row.metric, row.group_id)
    )


def preparation_differences(actual: DashboardSnapshot, expected: DashboardSnapshot) -> list[str]:
    """Every place two preparations of one source differ beyond the warehouse's reduction order.

    The wall clock is left out and exact-input digests are compared by type alone. Group rows
    have no promised order, so each population, metric and arm is compared with its own row.
    """
    return capture_differences(
        actual,
        expected,
        without=frozenset({"computed_at", "group_data"}),
        by_type=_EXACT_INPUT_DIGESTS,
    ) + capture_differences(_group_rows(actual), _group_rows(expected))
