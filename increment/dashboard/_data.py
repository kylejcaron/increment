"""Immutable dashboard state prepared from a caller-owned ``Analysis``.

Every number comes from the public readout API: this module builds no
warehouse query, owns no connection, and computes no statistics itself.
"""

from __future__ import annotations

import copy
import csv
import datetime as dt
import hashlib
import io
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from pydantic import BaseModel

from increment._canonical import canonical_json_bytes
from increment._source_operations import (
    BreakoutChoice,
    DashboardBreakoutReads,
    DashboardExploreCapture,
    DashboardGroupData,
    DashboardSnapshotPayload,
    ExploreKey,
)
from increment.breakout.estimates import (
    BreakoutEstimate,
    BreakoutEstimates,
    DailyLiftEstimates,
    DailyMetricValues,
    LiftEstimates,
)
from increment.dashboard._theme import MIDNIGHT, DashboardTheme, require_theme
from increment.errors import (
    CapabilityError,
    CodedError,
    InvalidRequestError,
    RefusalSpec,
    _freeze,
    refuse,
)
from increment.estimation.diagnostics import SRMResult
from increment.estimation.family import (
    exploratory_family_exclusion,
    select_exploratory_family,
)
from increment.tables import estimates_to_readout

if TYPE_CHECKING:
    import pyarrow as pa

    from increment.analysis import Analysis
    from increment.estimation.results import LiftEstimate
    from increment.semantics.models import Experiment, Metric

ExploreView = Literal["cumulative_lift", "daily_values", "cumulative_values", "segments"]

__all__ = [
    "DashboardConfig",
    "DashboardSnapshot",
    "ExploreView",
    "group_data_csv",
    "group_data_rows",
    "load_explore",
    "prepare_dashboard",
    "readout_csv",
]


_INVALID_CONFIG = RefusalSpec(
    "dashboard.invalid_config",
    InvalidRequestError,
    template="dashboard configuration is unusable: {reason}",
)

_INVALID_SNAPSHOT = RefusalSpec(
    "dashboard.invalid_snapshot",
    InvalidRequestError,
    template="dashboard snapshot is inconsistent: {reason}",
)

_INVALID_VIEW = RefusalSpec(
    "dashboard.invalid_view",
    InvalidRequestError,
    template="this dashboard view is unavailable: {reason}",
)

_UNSUPPORTED_EXPERIMENT = RefusalSpec(
    "dashboard.unsupported_experiment",
    CapabilityError,
    template="the dashboard supports ordinary two-arm randomized experiments: {reason}",
)

# An allocation check the source declines is not a passing check; the caller
# sees the original code and reason instead of a fabricated SRMResult.
_ALLOCATION_NOT_APPLICABLE = "dashboard.allocation_not_applicable"
# A source with no saved definitions offers no metrics to add.
_EXPLORATORY_SOURCE_LIMITED = "facade.analysis.exploratory_metrics_source_limited"


def _validated_allocation(raw: object) -> Mapping[str, float]:
    if not isinstance(raw, Mapping):
        refuse(
            _INVALID_CONFIG,
            reason="expected_allocation must be a mapping of arm label to target weight",
        )
    weights: dict[str, float] = {}
    for label, weight in raw.items():
        if not isinstance(label, str) or not label.strip():
            refuse(_INVALID_CONFIG, reason="every arm label must be a non-empty string")
        if isinstance(weight, bool) or not isinstance(weight, int | float):
            refuse(
                _INVALID_CONFIG,
                reason=f"arm {label!r} needs a numeric target weight",
                arm=label,
            )
        value = float(weight)
        if not math.isfinite(value) or value <= 0.0:
            refuse(
                _INVALID_CONFIG,
                reason=f"arm {label!r} needs a finite positive target weight",
                arm=label,
                weight=value,
            )
        weights[label] = value
    if len(weights) != 2:
        refuse(
            _INVALID_CONFIG,
            reason="expected_allocation must declare exactly two distinct arms",
            arms=tuple(weights),
        )
    return MappingProxyType(weights)


def _validated_provenance(raw: object) -> Mapping[str, str]:
    if not isinstance(raw, Mapping):
        refuse(_INVALID_CONFIG, reason="provenance must be a mapping of label to value")
    entries: dict[str, str] = {}
    for label, value in raw.items():
        if not isinstance(label, str) or not label.strip():
            refuse(_INVALID_CONFIG, reason="every provenance label must be a non-empty string")
        if not isinstance(value, str):
            refuse(
                _INVALID_CONFIG,
                reason=f"provenance value for {label!r} must be a string",
                label=label,
            )
        entries[label] = value
    return MappingProxyType(entries)


def _validated_units(raw: object) -> Mapping[str, str]:
    if not isinstance(raw, Mapping):
        refuse(_INVALID_CONFIG, reason="metric_units must be a mapping")
    values: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            refuse(_INVALID_CONFIG, reason="metric unit names must be non-empty strings")
        if not isinstance(value, str) or not value.strip():
            refuse(_INVALID_CONFIG, reason="metric units must be non-empty strings", metric=key)
        values[key] = value
    return MappingProxyType(values)


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    """Display configuration a caller supplies alongside their ``Analysis``.

    ``exploratory_metrics`` are the added metrics shown. The overview's exploratory family counts
    every saved metric the definitions offer whether shown or not, so choosing what to display,
    before or after seeing results, never changes the correction.
    """

    expected_allocation: Mapping[str, float]
    title: str | None = None
    source_label: str = ""
    provenance: Mapping[str, str] = field(default_factory=dict)
    metric_units: Mapping[str, str] = field(default_factory=dict)
    theme: DashboardTheme = MIDNIGHT
    exploratory_metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "expected_allocation", _validated_allocation(self.expected_allocation)
        )
        object.__setattr__(self, "provenance", _validated_provenance(self.provenance))
        object.__setattr__(self, "metric_units", _validated_units(self.metric_units))
        require_theme(self.theme)
        if self.title is not None and not self.title.strip():
            refuse(_INVALID_CONFIG, reason="title must be a non-empty string when supplied")
        names = self.exploratory_metrics
        if isinstance(names, str) or not all(isinstance(n, str) and n.strip() for n in names):
            refuse(_INVALID_CONFIG, reason="exploratory_metrics must be a sequence of metric names")
        if len(set(names)) != len(tuple(names)):
            refuse(_INVALID_CONFIG, reason="exploratory_metrics repeats a metric name")
        object.__setattr__(self, "exploratory_metrics", tuple(names))


@dataclass(frozen=True, slots=True)
class DashboardOverview:
    """The Explore overview's exploratory cells, captured with the headline.

    ``exploratory`` holds the added metrics' whole-window rows and ``segments`` each declared
    breakout's segment rows, all corrected as one BH family of
    ``family_size`` cells at ``family_q``. When the family itself is refused, ``family_refusal``
    holds that refusal and no exploratory cell is shown. ``refusals`` names each metric whose
    read was refused as ``(metric, place, refusal)``. ``exclusions`` names each cell left out
    of the family as ``(metric, place, reason)``; such a cell is shown uncorrected and never
    marked as a discovery. Every offered metric's cells are kept, shown or not.
    """

    exploratory: tuple[LiftEstimate, ...]
    segments: Mapping[BreakoutChoice, tuple[BreakoutEstimate, ...]]
    family_size: int
    family_q: float
    family_refusal: CodedError | None = None
    refusals: tuple[tuple[str, str, CodedError], ...] = ()
    exclusions: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "exploratory", tuple(self.exploratory))
        object.__setattr__(
            self,
            "segments",
            MappingProxyType({key: tuple(rows) for key, rows in self.segments.items()}),
        )


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    """One prepared reading of an experiment: metadata, allocation, results, Explore views.

    Read-only by construction. It holds no connection and is not a portable
    analysis format; reprepare from the source instead of persisting it.
    ``explore`` maps ``(view, metric or None, completed windows only, breakout)`` to the
    capture taken inside the same pinned read as the headline, where ``breakout`` is a
    declared ``(source, property)`` choice or ``None`` for the whole experiment;
    ``load_explore`` only looks these up. A hand-built snapshot with no captures refuses
    every Explore request. ``offered_metrics`` are every saved metric read and counted in the
    overview's family; ``exploratory_metrics`` are the ones shown.
    """

    experiment_name: str
    binding_fingerprint: str
    title: str
    description: str | None
    start: dt.datetime
    end: dt.datetime | None
    control_group: str
    treatment_group: str
    primary_metric: str
    metrics: tuple[Metric, ...]
    breakouts: tuple[tuple[str | None, str], ...]
    estimates: tuple[LiftEstimate, ...]
    readout_rows: tuple[Mapping[str, Any], ...]
    allocation: SRMResult | None
    allocation_refusal: tuple[str, str] | None
    allocation_history: tuple[Mapping[str, Any], ...]
    allocation_history_refusal: tuple[str, str] | None
    config: DashboardConfig
    computed_at: dt.datetime
    group_data: tuple[DashboardGroupData, ...]
    explore: Mapping[ExploreKey, DashboardExploreCapture] = field(default_factory=dict)
    overview: DashboardOverview | None = None
    exploratory_metrics: tuple[Metric, ...] = ()
    offered_metrics: tuple[Metric, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "exploratory_metrics", tuple(self.exploratory_metrics))
        object.__setattr__(self, "offered_metrics", tuple(self.offered_metrics))
        object.__setattr__(self, "estimates", tuple(self.estimates))
        object.__setattr__(self, "breakouts", tuple(self.breakouts))
        object.__setattr__(
            self,
            "readout_rows",
            tuple(_freeze(row) for row in self.readout_rows),
        )
        object.__setattr__(self, "group_data", tuple(self.group_data))
        object.__setattr__(self, "explore", MappingProxyType(dict(self.explore)))
        object.__setattr__(
            self,
            "allocation_history",
            tuple(_freeze(row) for row in self.allocation_history),
        )
        if self.allocation_history_refusal is not None:
            object.__setattr__(
                self, "allocation_history_refusal", tuple(self.allocation_history_refusal)
            )
            if self.allocation_history:
                refuse(_INVALID_SNAPSHOT, reason="allocation history cannot also be refused")
        if (self.allocation is None) == (self.allocation_refusal is None):
            refuse(
                _INVALID_SNAPSHOT,
                reason="exactly one of allocation or allocation_refusal must be present",
            )
        if len(self.readout_rows) != len(self.estimates):
            refuse(
                _INVALID_SNAPSHOT,
                reason="readout_rows must correspond one-to-one with estimates",
                rows=len(self.readout_rows),
                estimates=len(self.estimates),
            )
        expected = {
            (metric.name, arm)
            for metric in self.metrics
            for arm in (self.control_group, self.treatment_group)
        }
        keys = [(item.metric, item.group_id) for item in self.group_data]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            refuse(
                _INVALID_SNAPSHOT, reason="group evidence must cover each declared metric and arm"
            )
        for item in self.group_data:
            if item.source_kind not in ("pinned_warehouse", "retained_checkpoint"):
                refuse(_INVALID_SNAPSHOT, reason="group evidence has an unknown source kind")
            if any(
                not isinstance(reason, str) or not reason.strip()
                for reason in item.unavailable.values()
            ):
                refuse(
                    _INVALID_SNAPSHOT, reason="unavailable group evidence requires nonempty reasons"
                )
            counts = (
                item.eligible_units,
                item.assigned_units,
                item.event_count,
                item.retained_units,
                item.excluded_not_mature,
                item.excluded_no_observed_day,
                item.excluded_other,
            )
            if type(item.eligible_units) is not int or any(
                value is not None and (type(value) is not int or value < 0) for value in counts
            ):
                refuse(_INVALID_SNAPSHOT, reason="group evidence contains an invalid count")
            if item.retained_units is not None and item.retained_units > item.eligible_units:
                refuse(_INVALID_SNAPSHOT, reason="retained units exceed eligible units")
            if item.source_kind == "pinned_warehouse":
                excluded = (
                    item.excluded_not_mature,
                    item.excluded_no_observed_day,
                    item.excluded_other,
                )
                if item.assigned_units is None or any(value is None for value in excluded):
                    refuse(
                        _INVALID_SNAPSHOT,
                        reason="native group evidence requires complete accounting",
                    )
                if item.assigned_units != item.eligible_units + sum(
                    value for value in excluded if value is not None
                ):
                    refuse(
                        _INVALID_SNAPSHOT,
                        reason="group exclusions do not reconcile with assigned units",
                    )
            elif not item.prefix_id:
                refuse(
                    _INVALID_SNAPSHOT,
                    reason="retained group evidence requires its checkpoint prefix",
                )
            for name in (
                "observed_value",
                "analysis_input_value",
                "sum_value",
                "numerator",
                "denominator",
            ):
                value = getattr(item, name)
                if value is not None and not math.isfinite(value):
                    refuse(
                        _INVALID_SNAPSHOT,
                        reason=f"group evidence {name} must be finite or unavailable",
                    )
                if value is None and name not in item.unavailable:
                    refuse(
                        _INVALID_SNAPSHOT,
                        reason=f"unavailable group evidence {name} requires a reason",
                    )
            for name in (
                "assigned_units",
                "event_count",
                "retained_units",
                "excluded_not_mature",
                "excluded_no_observed_day",
                "excluded_other",
                "observation_end",
                "window_start_days",
                "window_end_days",
                "prefix_id",
            ):
                if getattr(item, name) is None and name not in item.unavailable:
                    refuse(
                        _INVALID_SNAPSHOT,
                        reason=f"unavailable group evidence {name} requires a reason",
                    )


def _csv_human_value(value: Any, *, name: str) -> Any:
    """Retain unrounded set metadata and neutralize spreadsheet formulas."""
    if name in ("confidence_set", "relative_confidence_set", "binomial_set") and isinstance(
        value, BaseModel
    ):
        value = value.model_dump_json(exclude={"raw", "reference"})
    if isinstance(value, str):
        for char in value:
            if not char.isprintable() or char in "=+-@":
                return "'" + value
            if not char.isspace():
                break
    return value


def readout_csv(snapshot: DashboardSnapshot) -> bytes:
    """Export every unrounded headline row; unavailable numbers stay empty.

    Formula markers after leading whitespace and control-prefixed strings
    receive an apostrophe for safe human-facing downloads.
    """
    fields = dict.fromkeys(key for row in snapshot.readout_rows for key in row)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields))
    writer.writeheader()
    writer.writerows(
        {key: _csv_human_value(value, name=key) for key, value in row.items()}
        for row in snapshot.readout_rows
    )
    return output.getvalue().encode("utf-8")


def _display_unit(snapshot: DashboardSnapshot, metric: str) -> str:
    model = metric_model(snapshot, metric)
    if model is not None and model.type in ("conversion", "retention"):
        return "%"
    return snapshot.config.metric_units.get(metric, "value")


def group_data_rows(
    snapshot: DashboardSnapshot, *, metric: str | None = None
) -> tuple[Mapping[str, Any], ...]:
    """Return deterministic, identifier-free observed group rows."""
    if metric is not None:
        require_metric(snapshot, metric)
    metric_order = {model.name: index for index, model in enumerate(snapshot.metrics)}
    arm_order = {snapshot.control_group: 0, snapshot.treatment_group: 1}
    selected = [item for item in snapshot.group_data if metric is None or item.metric == metric]
    selected.sort(
        key=lambda item: (
            metric_order.get(item.metric, len(metric_order)),
            arm_order.get(item.group_id, 2),
            item.group_id,
        )
    )
    rows: list[Mapping[str, Any]] = []
    for item in selected:
        row = {
            name: getattr(item, name)
            for name in DashboardGroupData.__dataclass_fields__
            if name != "unavailable"
        }
        row["unit"] = _display_unit(snapshot, item.metric)
        row["unavailable"] = dict(item.unavailable)
        row["experiment"] = snapshot.experiment_name
        row["computed_at"] = snapshot.computed_at.isoformat()
        row["binding_fingerprint"] = snapshot.binding_fingerprint
        rows.append(row)
    return tuple(cast("Mapping[str, Any]", _freeze(row)) for row in rows)


def group_data_csv(snapshot: DashboardSnapshot, *, metric: str | None = None) -> bytes:
    """Export exactly the rows returned by :func:`group_data_rows`."""
    rows = group_data_rows(snapshot, metric=metric)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(
        {
            key: _csv_human_value(
                json.dumps(dict(value), sort_keys=True) if key == "unavailable" else value,
                name=key,
            )
            for key, value in row.items()
        }
        for row in rows
    )
    return output.getvalue().encode("utf-8")


def _binding_fingerprint(experiment: Experiment, metrics: Sequence[Metric]) -> str:
    payload = {
        "experiment": experiment.model_dump(mode="json"),
        "metrics": [metric.model_dump(mode="json") for metric in metrics],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def prepare_dashboard(analysis: Analysis, *, config: DashboardConfig) -> DashboardSnapshot:
    """Run the headline analysis once and capture it as a read-only snapshot.

    This is the only place the dashboard computes results. Every Explore view a caller can
    later load is read inside the same pinned operation as the headline, allocation and group
    data, so ``load_explore`` never touches the warehouse and a later source change cannot
    make the views disagree with the headline; reexecuting the source cell is an intentional
    new snapshot. The caller keeps ownership of ``analysis`` and its connection: nothing here
    closes or replaces them.
    """
    experiment = _supported_experiment(analysis)
    primary_metric = _declared_primary(experiment)
    control_group, treatment_group = _declared_arms(experiment, config)
    metrics = _declared_metrics(analysis, experiment)
    offered = _offered_metrics(analysis, config)
    shown_models = _shown_metrics(offered, config.exploratory_metrics)
    added = tuple(model.name for model in offered)
    # Units may name any offered metric, shown or not.
    known = {metric.name for metric in metrics} | set(added)
    unknown_units = set(config.metric_units) - known
    if unknown_units:
        refuse(
            _INVALID_CONFIG,
            reason=f"metric_units names metrics the dashboard does not read: {sorted(unknown_units)!r}",
        )

    def _read(pinned: Analysis) -> DashboardSnapshotPayload:
        allocation, allocation_refusal = _allocation_check(pinned, config)
        allocation_history, allocation_history_refusal = _allocation_history(
            pinned, allocation_refusal
        )
        estimates = pinned.run()
        if not isinstance(estimates, LiftEstimates):
            refuse(
                _UNSUPPORTED_EXPERIMENT,
                reason="this experiment returns contrast results rather than lift estimates",
            )
        _require_treatment_arm(estimates, treatment=treatment_group)
        checkpoints = {
            estimate.metric: estimate.sequential_result.checkpoint
            for estimate in estimates
            if estimate.method_role == "decision" and estimate.sequential_result is not None
        }
        group_data = pinned.dashboard_group_data(metrics=metrics, checkpoints=checkpoints or None)
        explore = _capture_explore(
            pinned,
            names=tuple(metric.name for metric in metrics),
            added=added,
            scopes=tuple(
                ((b.source, b.property), pinned.dashboard_breakout_reads(b))
                for b in experiment.breakouts
            ),
        )
        return DashboardSnapshotPayload(
            allocation=allocation,
            allocation_refusal=allocation_refusal,
            allocation_history=allocation_history,
            allocation_history_refusal=allocation_history_refusal,
            estimates=tuple(estimates),
            group_data=group_data,
            explore=explore,
        )

    payload = analysis.dashboard_snapshot(_read, metrics=(*metrics, *offered))
    allocation = payload.allocation
    allocation_refusal = payload.allocation_refusal
    allocation_history = (
        () if payload.allocation_history is None else tuple(payload.allocation_history.to_pylist())
    )
    allocation_history_refusal = payload.allocation_history_refusal
    estimates = payload.estimates
    captures = {capture.key: capture for capture in payload.explore}
    overview_keys = [key for key in captures if key[0] in (_OVERVIEW_WHOLE, _OVERVIEW_SEGMENTS)]
    read = {key: captures.pop(key) for key in overview_keys}
    refusals = tuple(
        (
            str(key[1]),
            "whole experiment" if key[3] is None else str(key[3][1]),
            copy.copy(capture.refusal),
        )
        for key, capture in read.items()
        if key[1] is not None and capture.refusal is not None
    )
    overview = _overview(
        read.get((_OVERVIEW_WHOLE, None, False, None)),
        {
            key[3]: capture
            for key, capture in read.items()
            if key[0] == _OVERVIEW_SEGMENTS and key[1] is None and key[3] is not None
        },
        refusals=refusals,
        q=experiment.plan.q,
    )
    return DashboardSnapshot(
        experiment_name=experiment.name,
        binding_fingerprint=_binding_fingerprint(experiment, metrics),
        title=config.title or experiment.name,
        description=experiment.description,
        start=experiment.start,
        end=experiment.end,
        control_group=control_group,
        treatment_group=treatment_group,
        primary_metric=primary_metric,
        metrics=metrics,
        breakouts=tuple((b.source, b.property) for b in experiment.breakouts),
        estimates=estimates,
        readout_rows=enriched_rows(estimates),
        allocation=allocation,
        allocation_refusal=allocation_refusal,
        allocation_history=allocation_history,
        allocation_history_refusal=allocation_history_refusal,
        group_data=payload.group_data,
        explore=captures,
        overview=overview,
        exploratory_metrics=shown_models,
        offered_metrics=offered,
        config=config,
        computed_at=dt.datetime.now(dt.UTC),
    )


def _supported_experiment(analysis: Analysis) -> Experiment:
    try:
        experiment: Experiment | None = analysis.experiment
    except AttributeError:
        experiment = None
    if experiment is None:
        refuse(
            _UNSUPPORTED_EXPERIMENT,
            reason="the dashboard needs an Analysis with a declared experiment definition",
        )
    if experiment.design is not None:
        mechanism = experiment.design.mechanism
        refuse(
            _UNSUPPORTED_EXPERIMENT,
            reason=(
                f"the experiment declares a {mechanism} design; read its estimates "
                "with Analysis.run()"
            ),
            mechanism=mechanism,
        )
    return experiment


def _entry_name(entry: object) -> str:
    return entry if isinstance(entry, str) else str(getattr(entry, "metric", entry))


def _declared_primary(experiment: Experiment) -> str:
    primaries = [_entry_name(entry) for entry in experiment.plan.primaries]
    if len(primaries) != 1:
        refuse(
            _UNSUPPORTED_EXPERIMENT,
            reason="exactly one declared primary metric is required",
            primaries=tuple(primaries),
        )
    return primaries[0]


def _declared_arms(experiment: Experiment, config: DashboardConfig) -> tuple[str, str]:
    control = experiment.control_group
    labels = tuple(config.expected_allocation)
    if control not in labels:
        refuse(
            _INVALID_CONFIG,
            reason=(f"expected_allocation must include the experiment's control arm {control!r}"),
            control=control,
            arms=labels,
        )
    treatment = next(label for label in labels if label != control)
    return control, treatment


def _declared_metrics(analysis: Analysis, experiment: Experiment) -> tuple[Metric, ...]:
    """Declared metric models in plan order: primary, secondaries, guardrails."""
    by_name = {metric.name: metric for metric in analysis.metrics}
    ordered = [by_name.pop(name) for name in experiment.metric_names if name in by_name]
    return tuple(ordered) + tuple(by_name.values())


def _allocation_check(
    analysis: Analysis, config: DashboardConfig
) -> tuple[SRMResult | None, tuple[str, str] | None]:
    """The assigned-population allocation check, or its refusal code/reason.

    A source that cannot run the check yields a non-passing health item
    carrying the original code and reason. An invalid request, such as arm
    labels the data does not contain, is the caller's error and
    surfaces instead of becoming a badge. Only strings are retained: a caught
    exception would keep the caller's source referenced.
    """
    try:
        result = analysis.srm(expected=dict(config.expected_allocation), population="assigned")
    except CapabilityError as exc:
        return None, (exc.code, str(exc))
    if isinstance(result, SRMResult):
        return result, None
    return None, (_ALLOCATION_NOT_APPLICABLE, str(getattr(result, "reason", result)))


def _allocation_history(
    analysis: Analysis, allocation_refusal: tuple[str, str] | None
) -> tuple[pa.Table | None, tuple[str, str] | None]:
    """The assigned-population daily/cumulative allocation history, or its refusal.

    Skipped entirely when the allocation check itself was refused: a caller
    that could not run that check gets no history query either, and the
    snapshot inherits the same code/reason instead of a second, redundant
    refusal. A source that declines the history capability yields a
    non-passing health item, exactly like the allocation check itself; the
    exception is not retained, only its coded message.
    """
    if allocation_refusal is not None:
        return None, allocation_refusal
    try:
        table = analysis.allocation_history()
    except CapabilityError as exc:
        return None, (exc.code, str(exc))
    return table, None


def _require_treatment_arm(estimates: Sequence[LiftEstimate], *, treatment: str) -> None:
    arms = {estimate.group_id for estimate in estimates}
    if len(arms) > 1:
        refuse(
            _UNSUPPORTED_EXPERIMENT,
            reason="more than one treatment arm was estimated",
            arms=tuple(sorted(arms)),
        )
    if arms and arms != {treatment}:
        refuse(
            _INVALID_CONFIG,
            reason=f"expected_allocation names treatment arm {treatment!r}, results name another",
            configured=treatment,
            observed=tuple(sorted(arms)),
        )


def enriched_rows(
    estimates: Sequence[LiftEstimate | BreakoutEstimate],
) -> tuple[Mapping[str, Any], ...]:
    """Readout rows carrying each estimate's tested ``alternative``.

    The readout adapter drops the tested tail, so a one-sided guardrail would
    otherwise be indistinguishable from a two-sided test downstream.
    """
    rows = estimates_to_readout(list(estimates))
    if len(rows) != len(estimates):
        refuse(
            _INVALID_SNAPSHOT,
            reason="the readout adapter returned a different row count than estimates",
            rows=len(rows),
            estimates=len(estimates),
        )
    return tuple(
        MappingProxyType({**row, "alternative": estimate.alternative})
        for row, estimate in zip(rows, estimates, strict=True)
    )


# Shared read-only accessors for the rendering layer.


def decision_rows(snapshot: DashboardSnapshot) -> tuple[Mapping[str, Any], ...]:
    """Enriched rows for decision methods, in declared metric order."""
    return tuple(row for row in snapshot.readout_rows if row.get("method_role") == "decision")


def row_for_metric(snapshot: DashboardSnapshot, metric: str) -> Mapping[str, Any] | None:
    """The decision row for one metric, or ``None`` when it has none."""
    return next((row for row in decision_rows(snapshot) if row.get("metric") == metric), None)


def estimate_for_metric(snapshot: DashboardSnapshot, metric: str) -> LiftEstimate | None:
    """The original decision estimate for one metric, or ``None``."""
    return next(
        (
            estimate
            for estimate in snapshot.estimates
            if estimate.metric == metric and estimate.method_role == "decision"
        ),
        None,
    )


def metric_model(snapshot: DashboardSnapshot, metric: str) -> Metric | None:
    """The declared or added metric definition for one metric name, or ``None``."""
    return next((model for model in all_metrics(snapshot) if model.name == metric), None)


def all_metrics(snapshot: DashboardSnapshot) -> tuple[Metric, ...]:
    """Declared metrics in plan order, then the added exploratory metrics."""
    return (*snapshot.metrics, *snapshot.exploratory_metrics)


def metric_names(snapshot: DashboardSnapshot) -> tuple[str, ...]:
    """Declared metric names in plan order."""
    return tuple(model.name for model in snapshot.metrics)


def require_metric(snapshot: DashboardSnapshot, metric: str | None) -> Metric:
    """The declared metric definition, refusing a missing or undeclared name."""
    if metric is None:
        refuse(_INVALID_VIEW, reason="this view needs one explicit metric")
    model = metric_model(snapshot, metric)
    if model is None:
        refuse(
            _INVALID_VIEW,
            reason=f"metric {metric!r} is not declared on this experiment",
            metric=metric,
            declared=metric_names(snapshot),
        )
    return model


# Optional exploration: every view is captured with the headline, then only looked up.

# Each state the dashboard can ask of the source: (view, completed windows only). The completed
# gate is a cumulative maturity rule, so daily values and segments have only the open state.
_CAPTURED_STATES: tuple[tuple[str, bool], ...] = (
    ("cumulative_lift", False),
    ("cumulative_lift", True),
    ("cumulative_values", False),
    ("cumulative_values", True),
    ("daily_values", False),
    ("segments", False),
)
# Absolute per-arm values are computed per metric, with no cross-metric correction.
_INDEPENDENT_VIEWS = frozenset({"cumulative_values", "daily_values"})
# The overview's inputs, read for its exploratory family and never served by ``load_explore``:
# the added metrics' whole-window rows, and every metric's uncorrected segment rows for one
# declared breakout.
_OVERVIEW_WHOLE = "overview_whole"
_OVERVIEW_SEGMENTS = "overview_segments"


def _offered_metrics(analysis: Analysis, config: DashboardConfig) -> tuple[Metric, ...]:
    """Every saved metric the definitions offer; all are read and counted, shown or not.

    A source with no saved definitions offers none; asking it to show a metric is refused with
    its coded reason.
    """
    try:
        return tuple(analysis.available_metrics)
    except CodedError as exc:
        if config.exploratory_metrics or exc.code != _EXPLORATORY_SOURCE_LIMITED:
            raise
        return ()


def _shown_metrics(offered: Sequence[Metric], names: Sequence[str]) -> tuple[Metric, ...]:
    """The offered metrics named to show, refused unless each is offered."""
    available = {metric.name: metric for metric in offered}
    unknown = [name for name in names if name not in available]
    if unknown:
        refuse(
            _INVALID_CONFIG,
            reason=(
                f"exploratory metrics name metrics the definitions do not offer beyond the "
                f"experiment's own: {unknown!r}"
            ),
            unknown=tuple(unknown),
            available=tuple(available),
        )
    return tuple(available[name] for name in names)


def show_exploratory_metrics(
    snapshot: DashboardSnapshot, names: Sequence[str]
) -> DashboardSnapshot:
    """``snapshot`` showing ``names`` among its offered metrics.

    Every offered metric was read and corrected with the headline, so this reads nothing and
    recomputes nothing: the results, family and captures are the snapshot's own.
    """
    config = replace(snapshot.config, exploratory_metrics=tuple(names))
    shown = _shown_metrics(snapshot.offered_metrics, config.exploratory_metrics)
    return replace(snapshot, config=config, exploratory_metrics=shown)


def _decisions(capture: DashboardExploreCapture | None) -> list[Any]:
    if capture is None:
        return []
    return [row for row in capture.rows if row.method_role == "decision"]


def _overview(
    whole: DashboardExploreCapture | None,
    segments: Mapping[BreakoutChoice, DashboardExploreCapture],
    *,
    refusals: tuple[tuple[str, str, CodedError], ...],
    q: float,
) -> DashboardOverview:
    """Correct the added whole-window rows and every answered segment row as one BH family.

    A metric whose read was refused keeps its own refusal and adds no cells. A cell the family
    cannot take is left out with its reason rather than refusing every other cell.
    """
    exploratory = _decisions(whole)
    scoped = {breakout: _decisions(capture) for breakout, capture in segments.items()}
    cells = [*exploratory, *(row for rows in scoped.values() for row in rows)]
    reasons = [exploratory_family_exclusion(cell) for cell in cells]
    exclusions = tuple(
        (str(cell.metric), _cell_place(cell), reason)
        for cell, reason in zip(cells, reasons, strict=True)
        if reason is not None
    )
    admitted = [cell for cell, reason in zip(cells, reasons, strict=True) if reason is None]
    try:
        corrected = iter(select_exploratory_family(admitted, q=q))
    except CodedError as exc:
        return DashboardOverview(
            exploratory=(),
            segments={},
            family_size=len(admitted),
            family_q=q,
            family_refusal=copy.copy(exc),
            refusals=refusals,
            exclusions=exclusions,
        )
    results = [
        cell if reason is not None else next(corrected)
        for cell, reason in zip(cells, reasons, strict=True)
    ]
    sizes = {row.family_size for row in results if row.family_size is not None}
    rows = iter(results)
    added = tuple(cast("LiftEstimate", next(rows)) for _ in exploratory)
    captured = {
        breakout: tuple(cast("BreakoutEstimate", next(rows)) for _ in scoped[breakout])
        for breakout in segments
    }
    return DashboardOverview(
        exploratory=added,
        segments=captured,
        family_size=next(iter(sizes), 0),
        family_q=q,
        refusals=refusals,
        exclusions=exclusions,
    )


def _cell_place(cell: Any) -> str:
    dimension = getattr(cell, "dimension", None)
    if dimension is None:
        return "whole experiment"
    return f"{dimension} = {cell.dimension_value}"


def _capture_explore(
    pinned: Analysis,
    *,
    names: tuple[str, ...],
    added: tuple[str, ...] = (),
    scopes: tuple[tuple[BreakoutChoice, DashboardBreakoutReads], ...],
) -> tuple[DashboardExploreCapture, ...]:
    """Every Explore request ``load_explore`` can serve, answered from the pinned analysis.

    The lift family and segment estimates are read per metric and for every declared metric
    together, because each selection carries its own multiplicity. Absolute values are
    independent per metric, so one call over every metric serves them all; only when that call
    is refused is each metric asked alone, so one metric's refusal never hides another's series.
    Temporal views are read for the whole experiment and through each declared breakout's own
    reads, as are segments, so one breakout's refusal never hides another source of the same
    property. Source refusals are kept as the original coded errors. ``added`` metrics join no
    family, so one read per view and scope serves them all, never inside the declared metrics'
    joint family; a refused read, or a metric it returns nothing for, is asked alone. Each
    declared breakout also captures uncorrected segment rows for every declared and ``added``
    metric, the overview's exploratory family.
    """
    captures: dict[ExploreKey, DashboardExploreCapture] = {}

    def ask(
        reader: Analysis | DashboardBreakoutReads,
        view: str,
        metric: str | None,
        *,
        complete: bool,
        breakout: BreakoutChoice | None,
    ) -> DashboardExploreCapture:
        key = (view, metric, complete, breakout)
        selected = list(names) if metric is None else [metric]
        try:
            data = _read_view(reader, view, selected, complete=complete, added=added)
        except CodedError as exc:
            captures[key] = DashboardExploreCapture.refused(key, exc)
        else:
            captures[key] = DashboardExploreCapture.answered(key, data, collection=type(data))
        return captures[key]

    whole: tuple[tuple[BreakoutChoice | None, Analysis | DashboardBreakoutReads], ...] = (
        (None, pinned),
    )
    for view, complete in _CAPTURED_STATES:
        for breakout, reader in scopes if view == "segments" else (*whole, *scopes):
            joint = ask(reader, view, None, complete=complete, breakout=breakout)
            independent = view in _INDEPENDENT_VIEWS and joint.refusal is None
            for name in names:
                key = (view, name, complete, breakout)
                if len(names) == 1:
                    captures[key] = replace(joint, metric=name)
                elif independent:
                    assert joint.collection is not None
                    captures[key] = DashboardExploreCapture.answered(
                        key,
                        [row for row in joint.rows if row.metric == name],
                        collection=joint.collection,
                    )
                else:
                    ask(reader, view, name, complete=complete, breakout=breakout)
            if not added:
                continue
            # Added metrics join no family, so one read serves them all; a refused read, or a
            # metric it returns nothing for, is asked alone to keep that metric's own answer.
            try:
                batch = _read_view(reader, view, list(added), complete=complete, added=added)
            except CodedError:
                batch = None
            for name in added:
                rows = [] if batch is None else [row for row in batch if row.metric == name]
                if batch is None or not rows:
                    ask(reader, view, name, complete=complete, breakout=breakout)
                else:
                    key = (view, name, complete, breakout)
                    captures[key] = DashboardExploreCapture.answered(
                        key, rows, collection=type(batch)
                    )

    def overview_read(
        view: str,
        breakout: BreakoutChoice | None,
        declared: tuple[str, ...],
        read: Callable[[list[str], list[str]], Sequence[Any]],
        collection: Callable[[Iterable[Any]], Sequence[Any]],
    ) -> None:
        """One read of every metric; on a refusal, each metric alone, keeping its own refusal."""
        key = (view, None, False, breakout)
        try:
            captures[key] = DashboardExploreCapture.answered(
                key, read(list(declared), list(added)), collection=collection
            )
            return
        except CodedError:
            pass
        rows: list[Any] = []
        for name in (*declared, *added):
            one = (view, name, False, breakout)
            try:
                part = read([name] if name in declared else [], [name] if name in added else [])
            except CodedError as exc:
                captures[one] = DashboardExploreCapture.refused(one, exc)
            else:
                rows.extend(part)
        captures[key] = DashboardExploreCapture.answered(key, rows, collection=collection)

    def extra(added_names: list[str]) -> dict[str, Any]:
        return {"exploratory_metrics": added_names} if added_names else {}

    if added:
        overview_read(
            _OVERVIEW_WHOLE,
            None,
            (),
            lambda _, added_names: pinned.run(metrics=[], **extra(added_names)),
            LiftEstimates,
        )
    for breakout, reader in scopes:
        overview_read(
            _OVERVIEW_SEGMENTS,
            breakout,
            names,
            lambda declared, added_names, reader=reader: reader.uncorrected_segments(
                metrics=declared, **extra(added_names)
            ),
            BreakoutEstimates,
        )
    return tuple(captures.values())


def _read_view(
    reader: Analysis | DashboardBreakoutReads,
    view: str,
    selected: list[str],
    *,
    complete: bool,
    added: tuple[str, ...],
) -> Any:
    """One view for ``selected``; added metrics go through ``exploratory_metrics``."""
    declared = [name for name in selected if name not in added]
    extra = [name for name in selected if name in added]
    kwargs: dict[str, Any] = {"exploratory_metrics": extra} if extra else {}
    if view == "cumulative_lift":
        return reader.run_asof_lift(metrics=declared, completed_windows_only=complete, **kwargs)
    if view == "daily_values":
        return reader.run_daily(metrics=declared, **kwargs)
    if view == "cumulative_values":
        return reader.run_asof(metrics=declared, completed_windows_only=complete, **kwargs)
    return reader.run_breakout(metrics=declared, **kwargs)


def _captured(snapshot: DashboardSnapshot, key: ExploreKey, *, metric: str | None) -> Any:
    """A fresh collection of one captured view, or a fresh copy of its original refusal."""
    capture = snapshot.explore.get(key)
    if capture is None:
        view, _, complete, breakout = key
        refuse(
            _INVALID_VIEW,
            reason="this snapshot did not capture that view; prepare a new snapshot",
            view=view,
            metric=metric,
            completed_windows_only=complete,
            breakout=breakout,
        )
    return capture.load()


def load_explore(
    analysis: Analysis,
    *,
    snapshot: DashboardSnapshot,
    metric: str | None,
    view: ExploreView,
    completed_windows_only: bool = False,
    breakout: tuple[str | None, str] | None = None,
) -> DailyLiftEstimates | DailyMetricValues | BreakoutEstimates:
    """Load exactly the one advanced view a caller asked for.

    A lookup into the views captured with the headline by :func:`prepare_dashboard`: it
    reads no warehouse, never reruns the headline family, the allocation check, or
    materialization, and so cannot disagree with the snapshot or change a verdict. ``analysis``
    only proves the caller still holds the experiment the snapshot was prepared from. Each call
    returns a fresh collection; a state the source refused at capture raises its original coded
    error. ``metric=None`` selects every declared metric in snapshot order; a metric name
    selects just that metric.

    ``breakout`` names one declared ``(source, dimension)`` choice. Segment
    views require it when breakouts are declared; the temporal views break
    their series out by it. Each choice was read through that declared
    breakout alone, so another source of the same property, or its refusal,
    never appears in it. Without it a temporal view is the whole experiment.
    """
    if view not in get_args(ExploreView):
        refuse(_INVALID_VIEW, reason=f"unknown view {view!r}", view=view, metric=metric)
    if metric is not None:
        require_metric(snapshot, metric)
    _require_same_experiment(analysis, snapshot, metric=metric, view=view)
    if view in ("daily_values", "segments") and completed_windows_only:
        refuse(
            _INVALID_VIEW,
            reason=(
                f"completed windows only is a cumulative maturity gate; {view} cannot honour it"
            ),
            view=view,
            metric=metric,
        )
    if view == "segments":
        return _load_segments(snapshot=snapshot, metric=metric, breakout=breakout)
    if breakout is not None:
        breakout = _declared_breakout(snapshot, breakout, view=view, metric=metric)
    return _captured(snapshot, (view, metric, completed_windows_only, breakout), metric=metric)


def _require_same_experiment(
    analysis: Analysis, snapshot: DashboardSnapshot, *, metric: str | None, view: str
) -> None:
    """Refuse a source whose complete binding differs from the snapshot."""
    experiment = _supported_experiment(analysis)
    metrics = _declared_metrics(analysis, experiment)
    if (
        experiment.name != snapshot.experiment_name
        or _binding_fingerprint(experiment, metrics) != snapshot.binding_fingerprint
    ):
        refuse(
            _INVALID_VIEW,
            reason="the supplied analysis binding differs from the prepared snapshot",
            snapshot_experiment=snapshot.experiment_name,
            analysis_experiment=experiment.name,
            metric=metric,
            view=view,
        )


def _declared_breakout(
    snapshot: DashboardSnapshot,
    breakout: tuple[str | None, str],
    *,
    view: str,
    metric: str | None,
) -> BreakoutChoice:
    """The caller's choice, refused unless the experiment declares exactly it."""
    source, dimension = breakout
    requested = (source, dimension)
    if requested not in snapshot.breakouts:
        refuse(
            _INVALID_VIEW,
            reason=f"{requested!r} is not a declared breakout of this experiment",
            view=view,
            metric=metric,
            requested=requested,
            declared=snapshot.breakouts,
        )
    return requested


def _load_segments(
    *,
    snapshot: DashboardSnapshot,
    metric: str | None,
    breakout: tuple[str | None, str] | None,
) -> BreakoutEstimates:
    if not snapshot.breakouts:
        # A truthful empty state: no declared breakouts, so nothing was captured or needed.
        return BreakoutEstimates()
    if breakout is None:
        refuse(
            _INVALID_VIEW,
            reason="a segment view needs one declared (source, dimension) choice",
            view="segments",
            metric=metric,
            declared=snapshot.breakouts,
        )
    choice = _declared_breakout(snapshot, breakout, view="segments", metric=metric)
    source, dimension = choice
    estimates = _captured(snapshot, ("segments", metric, False, choice), metric=metric)
    # A registered sequential readout answers its registered dimension whatever the scope.
    return BreakoutEstimates(
        estimate
        for estimate in estimates
        if estimate.dimension == dimension
        and estimate.method_role == "decision"
        and (source is None or estimate.source == source)
    )
