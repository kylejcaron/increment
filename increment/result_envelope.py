"""Portable result cells retaining the complete inference row and readout scope."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType
from typing import Any, Literal, cast, get_args

import pyarrow as pa
from pydantic import BaseModel

from increment._interpretation import EffectEstimate, project_effect
from increment.errors import CodedError
from increment.estimation._fcr_reinterval import Construction, classify, reinterval_selected
from increment.estimation.readout_types import FamilyScope, ReadoutMetadata
from increment.estimation.results import LiftEstimate
from increment.kernel.contracts import CellIdentity

_SCHEMA_VERSION = 2
_METADATA_KEY = b"increment.readout_metadata"
_CONSTRUCTION_TAGS = frozenset((None, *get_args(Construction)))
_PROJECTION_FIELDS = (
    "null_lift",
    "null_abs",
    "role",
    "multiplicity_status",
    "discovery",
    "family_id",
    "family_axes",
    "family_q",
    "family_threshold",
    "family_procedure",
    "family_guarantee",
    "family_nominal_alpha",
    "family_size",
)


def _safe_projection_fields(row: LiftEstimate) -> dict[str, Any]:
    row_fields = row.model_dump(mode="json")
    return {name: row_fields[name] for name in _PROJECTION_FIELDS if name in row_fields}


def _report_summary(row: BaseModel, identity: CellIdentity) -> EffectEstimate:
    if hasattr(row, "lift"):
        effect = project_effect(row)
        reason = next(
            (
                str(getattr(row, name))
                for name in ("excluded", "unavailable", "unavailable_reason")
                if getattr(row, name, None) is not None
            ),
            None,
        )
        return replace(effect, reason=reason) if effect.reason is None and reason else effect
    rule = getattr(row, "rule", None)
    estimate = next(
        (
            getattr(row, name)
            for name in ("value", "estimate", "holdout_ate", "policy_value")
            if getattr(row, name, None) is not None
        ),
        None,
    )
    if estimate is None and rule is not None:
        estimate = getattr(rule, "policy_value", None)
    if hasattr(row, "observed_risk_ratio"):
        sensitivity = cast(Any, row)
        point = sensitivity.observed_risk_ratio - 1.0
        lower = (
            None
            if sensitivity.interval_lower_risk_ratio is None
            else sensitivity.interval_lower_risk_ratio - 1.0
        )
        upper = (
            None
            if sensitivity.interval_upper_risk_ratio is None
            else sensitivity.interval_upper_risk_ratio - 1.0
        )
        alpha = None if sensitivity.interval_level is None else 1.0 - sensitivity.interval_level
    else:
        point = getattr(estimate, "value", getattr(row, "ate", None))
        lower = getattr(estimate, "lb", getattr(row, "lb", None))
        upper = getattr(estimate, "ub", getattr(row, "ub", None))
        alpha = getattr(estimate, "alpha", None)
        if alpha is None:
            alpha = getattr(row, "alpha", None)
        if alpha is None:
            level = getattr(estimate, "level", None)
            if level is None:
                level = getattr(row, "interval_level", None)
            alpha = None if level is None else 1.0 - level
    reason_sources = (rule, row) if rule is not None else (row,)
    reason = next(
        (
            str(getattr(source, name))
            for source in reason_sources
            if source is not None
            for name in (
                "unavailable",
                "excluded",
                "unavailable_reason",
                "interval_unavailable_reason",
            )
            if getattr(source, name, None) is not None
        ),
        None if point is not None else "reported estimate is unavailable",
    )
    qualification = None
    policy = rule if rule is not None else row
    if identity.key.method in {"targeting", "targeting_selection"}:
        fraction = getattr(row, "selected_fraction", getattr(policy, "fraction", None))
        achieved = getattr(policy, "achieved_fraction", None)
        if fraction is not None:
            qualification = f"selected population: fraction={fraction:g}"
            if achieved is not None:
                qualification += f", achieved_fraction={achieved:g}"
    geometry = "interval" if lower is not None or upper is not None else None
    intervals = ((lower, upper),) if geometry is not None else ()
    return EffectEstimate(
        point=cast(float | None, point),
        lower=lower,
        upper=upper,
        alpha=alpha,
        value_scale=identity.key.value_scale or "relative",
        reason=reason,
        geometry=geometry,
        intervals=intervals,
        qualification=qualification,
    )


@dataclass(frozen=True, slots=True)
class _ProjectionEstimate:
    value: float
    lb: float | None
    ub: float | None
    alpha: float | None


@dataclass(frozen=True, slots=True)
class _ReportProjection:
    metric: str
    method: str | None
    method_role: Literal["decision", "sensitivity"] | None
    estimand: str | None
    analysis_population: Literal["assigned", "triggered"]
    value_scale: Literal["relative", "absolute"] | None
    group_id: str | None
    dimension: str | None
    dimension_value: str | None
    ds: Any
    ds_basis: Literal["calendar", "cohort"] | None
    source: str | None
    source_snapshot_id: str | None
    lift: _ProjectionEstimate | None
    unavailable: str | None
    relative_unavailable_reason: str | None
    note: str | None


def _json(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _restore_row(value: object) -> LiftEstimate:
    return LiftEstimate.model_validate_json(_json(_plain(value)))


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(v) for v in value)
    return value


class Operation(StrEnum):
    RELEVEL = "relevel"
    COMPARE_MARGIN = "compare_margin"
    COMBINE_FAMILY = "combine_family"


@dataclass(frozen=True, slots=True)
class ComponentStatus:
    state: Literal["available", "unavailable", "advisory"]
    code: str | None = None
    reason: str | None = None
    context: Mapping[str, Any] = field(default_factory=dict)
    advisories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", _freeze(self.context))
        object.__setattr__(self, "advisories", tuple(sorted(self.advisories)))


@dataclass(frozen=True, slots=True)
class RetainedState:
    construction: str | None
    reason: str
    kind: Literal["lift_sampling", "posterior", "asymptotic_sequential", "likelihood_sequential"]
    fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", _freeze(self.fields))


Component = tuple[ComponentStatus, RetainedState | None]


@dataclass(frozen=True, slots=True)
class Provenance:
    run_id: str | None
    execution_identity_hash: str | None
    package_version: str
    created_at: datetime

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise CodedError(
                "created_at must be timezone-aware",
                code="results.cell.provenance_timestamp_invalid",
                context={"field": "created_at"},
            )


@dataclass(frozen=True, slots=True)
class ResultCell:
    identity: CellIdentity
    label: str
    family: FamilyScope | None
    source_snapshot_id: str | None
    components: Mapping[Literal["sampling", "posterior", "sequential"], Component]
    provenance: Provenance
    summary: EffectEstimate | None
    projection_fields: Mapping[str, Any] = field(default_factory=dict)
    row_kind: str = "lift_estimate"
    row_payload: Mapping[str, Any] | None = None
    supports: Mapping[Literal["sampling", "posterior", "sequential"], frozenset[Operation]] = field(
        init=False
    )

    def __post_init__(self) -> None:
        components = MappingProxyType(dict(self.components))
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "projection_fields", _freeze(self.projection_fields))
        allowed_row_kinds = {
            "lift_estimate",
            "breakout",
            "daily_lift",
            "daily_metric",
            "asof_lift",
            "asof_metric",
            "cate",
            "switchback_variance",
            "cate_validation",
            "switchback_unit_t_approximation",
            "switchback_block_t",
            "switchback_unit_variance_envelope",
            "sensitivity",
            "targeting",
            "targeting_selection",
        }
        if self.row_kind not in allowed_row_kinds:
            raise CodedError(
                "Unsupported result-cell row kind",
                code="results.cell.schema_unsupported",
                context={"row_kind": self.row_kind},
            )
        payload = None if self.row_payload is None else _freeze(self.row_payload)
        if payload is not None and (
            not isinstance(payload, Mapping)
            or payload.get("payload_version") != 1
            or not isinstance(payload.get("data"), Mapping)
        ):
            raise CodedError(
                "Unsupported retained report payload",
                code="results.cell.schema_unsupported",
                context={"row_kind": self.row_kind},
            )
        object.__setattr__(self, "row_payload", payload)
        support: dict[str, frozenset[Operation]] = {}
        for name in ("sampling", "posterior", "sequential"):
            status, state = components[name]
            operations: set[Operation] = set()
            if (
                name == "sampling"
                and status.state in {"available", "advisory"}
                and state is not None
            ):
                row = _restore_row(state.fields)
                if _can_reconstruct_relevel(row) and _retained_relevel_alpha(row) is not None:
                    operations.add(Operation.RELEVEL)
            support[name] = frozenset(operations)
        object.__setattr__(self, "supports", MappingProxyType(support))

    def __reduce__(self):
        return _restore_result_cell, (self.to_payload(),)

    @classmethod
    def from_row(
        cls,
        row: LiftEstimate,
        metadata: ReadoutMetadata | None = None,
        *,
        created_at: datetime | None = None,
    ) -> ResultCell:
        construction, reason = classify(row)
        identity = CellIdentity.from_row(row)
        key = identity.key
        raw_outcomes_present = row.confidence_set is not None
        row_fields = row.model_dump(mode="json")
        sampling_state = (
            None
            if raw_outcomes_present
            else RetainedState(construction, reason, "lift_sampling", row_fields)
        )
        record = None
        if metadata is not None:
            record = next(
                (
                    item
                    for item in metadata.cells
                    if item.source_snapshot_id == row.source_snapshot_id and item.cell == key
                ),
                None,
            )
        failure = None if record is None else record.failure
        sampling = None if record is None else record.sampling
        advisory_codes = () if record is None else tuple(item.code for item in record.advisories)
        sampling_unavailable = (
            raw_outcomes_present
            or (sampling is not None and not sampling.available)
            or row.sampling_available is False
            or (row.sampling_available is None and row.prior_shrunk)
            or (failure is not None)
            or (
                row.failure_code is not None
                and row.lift is None
                and row.binomial_set is None
                and row.confidence_set is None
            )
        )
        if sampling_unavailable:
            sampling_status = ComponentStatus(
                "unavailable",
                code=(
                    failure.code
                    if failure is not None
                    else (
                        sampling.reason_code
                        if sampling is not None
                        else row.sampling_reason_code
                        or row.failure_code
                        or "results.component.unavailable"
                    )
                ),
                reason=reason,
                context=(
                    failure.context
                    if failure is not None
                    else (row.sampling_reason_context or row.failure_context or {"reason": reason})
                )
                or {},
                advisories=advisory_codes,
            )
        else:
            sampling_status = ComponentStatus(
                "advisory" if advisory_codes else "available",
                reason=reason if construction is None else None,
                advisories=advisory_codes,
            )
        posterior_available = (
            False
            if raw_outcomes_present
            else (
                (
                    record.posterior.available
                    if record is not None and record.posterior is not None
                    else None
                )
                if row.posterior_components is None and row.posterior_latent_mean is None
                else True
            )
        )
        posterior_available = bool(posterior_available)
        posterior_reason = (
            record.posterior.reason_code
            if record is not None and record.posterior is not None
            else row.posterior_reason_code
            or row.posterior_prob_favorable_reason_code
            or "posterior state is not retained"
        )
        posterior_reason_context = (
            record.posterior.reason_context or {}
            if record is not None and record.posterior is not None
            else row.posterior_reason_context or {}
        )
        posterior_status = (
            ComponentStatus("available")
            if posterior_available
            else ComponentStatus(
                "unavailable",
                code=posterior_reason or "results.component.unavailable",
                reason=posterior_reason,
                context=posterior_reason_context,
            )
        )
        posterior_state = (
            RetainedState(construction, reason, "posterior", row_fields)
            if posterior_available
            else None
        )
        from increment.estimation.sequential_result import (
            AsymptoticSequentialResult,
            SequentialInferenceResult,
        )

        sequential_result = row.sequential_result
        if isinstance(sequential_result, AsymptoticSequentialResult):
            sequential_kind = "asymptotic_sequential"
            sequential_reason = "alpha is registered; operations follow the registered procedure"
        elif isinstance(sequential_result, SequentialInferenceResult):
            sequential_kind = "likelihood_sequential"
            sequential_reason = (
                "likelihood sequential geometry is retained; no portable operation is registered"
            )
        else:
            sequential_kind = None
            sequential_reason = "no sequential result is retained"
        sequential_state = (
            RetainedState(construction, sequential_reason, sequential_kind, row_fields)
            if sequential_kind is not None
            else None
        )
        sequential_status = (
            ComponentStatus("available", reason=sequential_reason)
            if sequential_kind is not None
            else ComponentStatus(
                "unavailable",
                code="results.component.unavailable",
                reason=sequential_reason,
            )
        )
        family = None if metadata is None else metadata.family_of(row.source_snapshot_id, key)
        try:
            package_version = version("increment")
        except PackageNotFoundError:
            package_version = "0+unknown"
        stamp = created_at or datetime.now(UTC)
        summary = (
            _summary(row)
            if sampling_status.state != "unavailable" or sequential_status.state == "available"
            else None
        )
        return cls(
            identity=CellIdentity.from_row(row),
            label=row.method,
            family=family,
            source_snapshot_id=row.source_snapshot_id,
            components={
                "sampling": (sampling_status, sampling_state),
                "posterior": (posterior_status, posterior_state),
                "sequential": (sequential_status, sequential_state),
            },
            provenance=Provenance(None, None, package_version, stamp),
            summary=summary,
            projection_fields=_safe_projection_fields(row),
        )

    @classmethod
    def from_report_payload(
        cls,
        *,
        row_kind: str,
        identity: CellIdentity,
        label: str,
        row: BaseModel,
        source_snapshot_id: str | None = None,
        family: FamilyScope | None = None,
        created_at: datetime | None = None,
        identity_context: Mapping[str, Any] | None = None,
    ) -> ResultCell:
        """Retain an estimated report row without pretending it is a lift row."""
        if row_kind == "lift_estimate":
            raise CodedError(
                "Lift rows require the lift adapter",
                code="results.cell.schema_unsupported",
                context={"row_kind": row_kind},
            )
        try:
            package_version = version("increment")
        except PackageNotFoundError:
            package_version = "0+unknown"
        payload = row.model_dump(mode="json")
        reason = "report-only retained state has no registered portable operation"
        return cls(
            identity=identity,
            label=label,
            family=family,
            source_snapshot_id=source_snapshot_id or identity.source_snapshot_id,
            components={
                "sampling": (
                    ComponentStatus(
                        "unavailable",
                        code="results.operation.unsupported",
                        reason=reason,
                        context={"row_kind": row_kind},
                    ),
                    None,
                ),
                "posterior": (
                    ComponentStatus("unavailable", code="results.component.unavailable"),
                    None,
                ),
                "sequential": (
                    ComponentStatus("unavailable", code="results.component.unavailable"),
                    None,
                ),
            },
            provenance=Provenance(None, None, package_version, created_at or datetime.now(UTC)),
            summary=_report_summary(row, identity),
            row_kind=row_kind,
            row_payload={
                "payload_version": 1,
                "data": payload,
                **(
                    {"identity_context": _plain(identity_context)}
                    if identity_context is not None
                    else {}
                ),
            },
        )

    @classmethod
    def from_cate_report(
        cls,
        *,
        row: BaseModel,
        row_kind: Literal["cate", "cate_validation", "targeting", "targeting_selection"],
        label: str,
        identity_context: Mapping[str, Any],
        created_at: datetime | None = None,
    ) -> ResultCell:
        from importlib import import_module

        CellKey = import_module("increment.declarations").CellKey
        contracts = import_module("increment.kernel.contracts")

        source_identity = identity_context.get("source_identity")
        rule = getattr(row, "rule", None)
        validations = (
            getattr(row, "validation", None),
            None if rule is None else getattr(rule, "validation", None),
        )
        if row_kind == "targeting_selection":
            effective_population = None if rule is None else getattr(rule, "population", None)
        else:
            populations = [getattr(row, "population", None)]
            if rule is not None:
                populations.append(getattr(rule, "population", None))
            populations.extend(
                None if validation is None else getattr(validation, "population", None)
                for validation in validations
            )
            effective_population = next(
                (
                    population
                    for population in populations
                    if population is not None and "overlap" in str(population)
                ),
                None,
            )
        estimand = "overlap_subpopulation_ate" if effective_population is not None else "ate"
        policy = rule if rule is not None else row
        method_config: dict[str, Any] = {
            "control": identity_context["control"],
        }
        if row_kind == "cate":
            method_config.update(
                alpha=getattr(row, "alpha", None),
                ard=getattr(row, "beta_ard", None) is not None,
                interactions=getattr(row, "interaction_spec").model_dump(mode="json"),
                adjustments=getattr(row, "main_spec").model_dump(mode="json"),
            )
        elif row_kind == "cate_validation":
            method_config.update(
                alpha=getattr(row, "alpha", None),
                n_groups=len(getattr(row, "groups", ())),
                evaluation_population=_plain(getattr(row, "evaluation_population", None)),
            )
        if row_kind in {"targeting", "targeting_selection"}:
            method_config.update(
                fraction=getattr(row, "selected_fraction", getattr(policy, "fraction", None)),
                achieved_fraction=getattr(policy, "achieved_fraction", None),
                cost_per_treated=getattr(row, "cost_per_treated", 0.0),
                cluster_weight=getattr(policy, "cluster_weight", None),
                intervention_grain=getattr(policy, "intervention_grain", None),
                deploy_grain=getattr(policy, "deploy_grain", None),
                budget_rule=getattr(policy, "budget_rule", None),
                required_columns=getattr(policy, "required_columns", ()),
            )
            if row_kind == "targeting_selection":
                from increment.estimation.targeting import TargetingSelection

                selection = cast(TargetingSelection, row)
                method_config.update(
                    fractions=getattr(row, "fractions", ()),
                    n_folds=getattr(row, "n_folds", None),
                    seed=str(selection.seed),
                )
        method_config_hash = contracts.contract_hash("report_method_config", method_config)
        identity = contracts.CellIdentity(
            key=CellKey(
                kind="arm",
                metric=identity_context["metric"],
                method=row_kind,
                method_role="decision",
                estimand=estimand,
                analysis_population=identity_context["analysis_population"],
                value_scale="absolute",
                inference="fixed",
                alternative="two-sided",
                group_id=identity_context["treatment"],
            ),
            method_config_hash=method_config_hash,
            source_snapshot_id=identity_context["source_snapshot_id"],
            experiment_id=source_identity.get("study_id")
            if isinstance(source_identity, Mapping)
            else None,
            trigger_name=identity_context.get("trigger_name"),
            source_identity_hash=(
                hashlib.sha256(_json(_plain(source_identity)).encode("utf-8")).hexdigest()
                if source_identity is not None
                else None
            ),
            scope=contracts.Scope.model_validate(identity_context["scope"]),
        )
        return cls.from_report_payload(
            row_kind=row_kind,
            identity=identity,
            label=label,
            row=row,
            source_snapshot_id=identity_context["source_snapshot_id"],
            created_at=created_at,
            identity_context=identity_context,
        )

    def retained_row(self) -> LiftEstimate:
        state = self.components["sampling"][1]
        if state is None:
            raise CodedError(
                "No retained lift row",
                code="results.operation.unsupported",
                context={"reason": self.components["sampling"][0].reason},
            )
        return _restore_row(state.fields)

    def _projection_row(self) -> Any:
        state = (
            self.components["sampling"][1]
            or self.components["sequential"][1]
            or self.components["posterior"][1]
        )
        if state is not None:
            return _restore_row(state.fields)
        if self.row_payload is not None and self.row_kind in {
            "breakout",
            "daily_metric",
            "asof_metric",
            "daily_lift",
            "asof_lift",
            "switchback_variance",
            "switchback_unit_t_approximation",
            "switchback_block_t",
            "switchback_unit_variance_envelope",
        }:
            from importlib import import_module

            module_name = (
                "increment.breakout.estimates"
                if self.row_kind
                in {"breakout", "daily_metric", "asof_metric", "daily_lift", "asof_lift"}
                else "increment.estimation.contrast_results"
            )
            module = import_module(module_name)
            model_name = {
                "breakout": "BreakoutEstimate",
                "daily_metric": "DailyMetricValue",
                "asof_metric": "DailyMetricValue",
                "daily_lift": "DailyLiftEstimate",
                "asof_lift": "DailyLiftEstimate",
                "switchback_variance": "ContrastResult",
                "switchback_unit_t_approximation": "ContrastResult",
                "switchback_block_t": "ContrastResult",
                "switchback_unit_variance_envelope": "ContrastResult",
            }[self.row_kind]
            return getattr(module, model_name).model_validate_json(
                _json(_plain(self.row_payload["data"]))
            )
        if self.row_payload is not None and self.row_kind in {
            "cate",
            "cate_validation",
            "targeting",
            "targeting_selection",
            "sensitivity",
        }:
            key = self.identity.key
            summary = self.summary
            report_estimate = (
                None
                if summary is None or summary.point is None
                else _ProjectionEstimate(summary.point, summary.lower, summary.upper, summary.alpha)
            )
            unavailable = None if summary is None else summary.reason
            return _ReportProjection(
                metric=key.metric,
                method=key.method,
                method_role=key.method_role,
                estimand=key.estimand,
                analysis_population=key.analysis_population,
                value_scale=key.value_scale,
                group_id=key.group_id,
                dimension=key.dimension,
                dimension_value=key.dimension_value,
                ds=key.ds,
                ds_basis=key.ds_basis,
                source=key.source,
                source_snapshot_id=self.source_snapshot_id,
                lift=report_estimate,
                unavailable=unavailable if report_estimate is None else None,
                relative_unavailable_reason=unavailable,
                note=None if summary is None else summary.qualification,
            )
        status = self.components["sampling"][0]
        key = self.identity.key
        reason = status.reason or "sampling evidence is unavailable"
        context = dict(status.context) or {"reason": reason}
        code = status.code or "results.component.unavailable"
        return LiftEstimate(
            metric=key.metric,
            group_id=key.group_id or key.treatment_group or "unknown",
            method=key.method or self.label,
            method_role=key.method_role or "decision",
            inference=key.inference or "fixed",
            alternative=key.alternative or "two-sided",
            estimand=cast(
                Literal[
                    "itt", "compliance", "late", "ate", "plr_slope", "overlap_subpopulation_ate"
                ],
                key.estimand or "itt",
            ),
            analysis_population=key.analysis_population,
            value_scale=key.value_scale or "relative",
            ds=key.ds,
            source_snapshot_id=self.source_snapshot_id,
            sampling_available=False,
            sampling_reason_code=code,
            sampling_reason_context=context,
            failure_code=code,
            failure_context=context,
            **dict(self.projection_fields),
        )

    def to_payload(self) -> dict[str, Any]:
        sampling_state = self.components["sampling"][1]
        if sampling_state is None:
            row_payload = None
        else:
            row_payload = _plain(sampling_state.fields)
        comps = {}
        for name, (status, state) in self.components.items():
            comps[name] = {
                "status": {
                    "state": status.state,
                    "code": status.code,
                    "reason": status.reason,
                    "context": _plain(status.context),
                    "advisories": list(status.advisories),
                },
                "construction": None if state is None else state.construction,
                "reason": None if state is None else state.reason,
                "kind": None if state is None else state.kind,
                "supports": sorted(op.value for op in self.supports[name]),
            }
        return {
            "schema_version": _SCHEMA_VERSION,
            "row_kind": self.row_kind,
            "row_payload": None if self.row_payload is None else _plain(self.row_payload),
            "identity": self.identity.model_dump(mode="json"),
            "label": self.label,
            "family": None if self.family is None else self.family.model_dump(mode="json"),
            "source_snapshot_id": self.source_snapshot_id,
            "retained_row": row_payload,
            "projection_fields": _plain(self.projection_fields),
            "components": comps,
            "provenance": {
                "run_id": self.provenance.run_id,
                "execution_identity_hash": self.provenance.execution_identity_hash,
                "package_version": self.provenance.package_version,
                "created_at": self.provenance.created_at.isoformat(),
            },
            "summary": None
            if self.summary is None
            else {
                "point": self.summary.point,
                "lower": self.summary.lower,
                "upper": self.summary.upper,
                "alpha": self.summary.alpha,
                "value_scale": self.summary.value_scale,
                "reason": self.summary.reason,
                "geometry": self.summary.geometry,
                "intervals": self.summary.intervals,
                "qualification": self.summary.qualification,
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ResultCell:
        schema_version = payload.get("schema_version")
        if schema_version not in {1, _SCHEMA_VERSION}:
            raise CodedError(
                "Unsupported result-cell schema",
                code="results.cell.schema_unsupported",
                context={"schema_version": schema_version},
            )
        row_kind = payload.get("row_kind", "lift_estimate")
        row_payload = payload.get("row_payload")
        row_dump = payload.get("retained_row")
        if (row_kind == "lift_estimate") != (row_payload is None) or (
            row_kind != "lift_estimate" and row_dump is not None
        ):
            raise CodedError(
                "Result-cell row kind does not match retained payload",
                code="results.cell.schema_unsupported",
                context={"row_kind": row_kind},
            )
        row = None if row_dump is None else _restore_row(row_dump)
        comp_data = payload["components"]
        components: dict[Literal["sampling", "posterior", "sequential"], Component] = {}
        for name in ("sampling", "posterior", "sequential"):
            component = comp_data[name]
            raw_status = component["status"]
            if raw_status.get("state") not in {"available", "unavailable", "advisory"}:
                raise CodedError(
                    "Malformed result-cell component status",
                    code="results.cell.schema_unsupported",
                    context={"component": name},
                )
            construction = component.get("construction")
            if construction not in _CONSTRUCTION_TAGS:
                raise CodedError(
                    "Unsupported result-cell construction",
                    code="results.cell.schema_unsupported",
                    context={"construction": construction},
                )
            kind = component.get("kind")
            if kind not in {
                None,
                "lift_sampling",
                "posterior",
                "asymptotic_sequential",
                "likelihood_sequential",
            }:
                raise CodedError(
                    "Unsupported result-cell retained state",
                    code="results.cell.schema_unsupported",
                    context={"kind": kind},
                )
            if row_kind != "lift_estimate" and (
                construction is not None or kind is not None or component.get("supports")
            ):
                raise CodedError(
                    "Non-lift report row carries lift component state",
                    code="results.cell.schema_unsupported",
                    context={"row_kind": row_kind, "component": name},
                )
            status = ComponentStatus(
                raw_status["state"],
                raw_status.get("code"),
                raw_status.get("reason"),
                raw_status.get("context") or {},
                tuple(raw_status.get("advisories") or ()),
            )
            state = (
                None
                if row is None or kind is None
                else RetainedState(
                    construction, component.get("reason") or "", kind, row.model_dump(mode="json")
                )
            )
            components[name] = (status, state)
        prov = payload["provenance"]
        summary_data = payload.get("summary")
        if summary_data is not None:
            summary_data = dict(summary_data)
            summary_data["intervals"] = tuple(
                tuple(interval) for interval in summary_data.get("intervals", ())
            )
        return cls(
            identity=CellIdentity.model_validate(payload["identity"]),
            label=payload["label"],
            family=None
            if payload.get("family") is None
            else FamilyScope.model_validate(payload["family"]),
            source_snapshot_id=payload.get("source_snapshot_id"),
            components=components,
            provenance=Provenance(
                prov.get("run_id"),
                prov.get("execution_identity_hash"),
                prov["package_version"],
                datetime.fromisoformat(prov["created_at"]),
            ),
            summary=None if summary_data is None else EffectEstimate(**summary_data),
            projection_fields=payload.get("projection_fields") or {},
            row_kind=row_kind,
            row_payload=row_payload,
        )

    @staticmethod
    def to_arrow(cells: Sequence[ResultCell]) -> pa.Table:
        rows = []
        for cell in cells:
            payload = cell.to_payload()
            entry: dict[str, Any] = {
                "schema_version": _SCHEMA_VERSION,
                "row_kind": payload["row_kind"],
                "identity": _json(payload["identity"]),
                "label": cell.label,
                "family": None if cell.family is None else _json(cell.family),
                "source_snapshot_id": cell.source_snapshot_id,
                "provenance": _json(payload["provenance"]),
                "summary": None if cell.summary is None else _json(payload["summary"]),
                "retained_row": None
                if payload["retained_row"] is None
                else _json(payload["retained_row"]),
                "row_payload": None
                if payload["row_payload"] is None
                else _json(payload["row_payload"]),
                "projection_fields": _json(payload["projection_fields"]),
            }
            for name in ("sampling", "posterior", "sequential"):
                component = payload["components"][name]
                status = component["status"]
                entry[name] = {
                    "status": status["state"],
                    "code": status["code"],
                    "reason": status["reason"],
                    "context": _json(status["context"]),
                    "advisories": status["advisories"],
                    "construction": component["construction"],
                    "kind": component["kind"],
                    "state_reason": component["reason"],
                    "supports": component["supports"],
                }
            rows.append(entry)
        schema = _arrow_schema()
        table = pa.Table.from_pylist(rows, schema=schema)
        return table

    @staticmethod
    def from_arrow(table: pa.Table) -> list[ResultCell]:
        cells = []
        for entry in table.to_pylist():
            components = {}
            row_dump = None if entry["retained_row"] is None else json.loads(entry["retained_row"])
            row = None if row_dump is None else _restore_row(row_dump)
            construction, reason = (
                (None, "no retained sampling row") if row is None else classify(row)
            )
            for name in ("sampling", "posterior", "sequential"):
                comp = entry[name]
                stored_construction = comp["construction"]
                if stored_construction not in _CONSTRUCTION_TAGS:
                    raise CodedError(
                        "Unsupported result-cell construction",
                        code="results.cell.schema_unsupported",
                        context={"component": name, "construction": stored_construction},
                    )
                from increment.estimation.sequential_result import (
                    AsymptoticSequentialResult,
                    SequentialInferenceResult,
                )

                sequential_available = row is not None and isinstance(
                    row.sequential_result,
                    (AsymptoticSequentialResult, SequentialInferenceResult),
                )
                has_state = row is not None and (
                    name == "sampling"
                    or (name == "posterior" and comp["status"] in {"available", "advisory"})
                    or (name == "sequential" and sequential_available)
                )
                expected_construction = construction if has_state else None
                if stored_construction != expected_construction:
                    raise CodedError(
                        "Mismatched result-cell construction",
                        code="results.cell.schema_unsupported",
                        context={
                            "component": name,
                            "construction": stored_construction,
                            "expected": expected_construction,
                        },
                    )
                status = {
                    "state": comp["status"],
                    "code": comp["code"],
                    "reason": comp["reason"],
                    "context": json.loads(comp["context"] or "{}"),
                    "advisories": comp["advisories"] or [],
                }
                kind = (
                    (
                        "asymptotic_sequential"
                        if isinstance(row.sequential_result, AsymptoticSequentialResult)
                        else "likelihood_sequential"
                    )
                    if name == "sequential" and row is not None and has_state
                    else {
                        "sampling": "lift_sampling",
                        "posterior": "posterior",
                        "sequential": "asymptotic_sequential",
                    }[name]
                )
                state_reason = (
                    "alpha is registered; operations follow the registered procedure"
                    if name == "sequential"
                    else reason
                )
                components[name] = {
                    "status": status,
                    "construction": stored_construction,
                    "kind": comp.get("kind")
                    or (kind if has_state and row_dump is not None else None),
                    "reason": comp.get("state_reason") or state_reason,
                    "supports": comp["supports"] or [],
                }
            payload = {
                "schema_version": entry["schema_version"],
                "identity": json.loads(entry["identity"]),
                "label": entry["label"],
                "family": None if entry["family"] is None else json.loads(entry["family"]),
                "source_snapshot_id": entry["source_snapshot_id"],
                "row_kind": entry.get("row_kind", "lift_estimate"),
                "row_payload": None
                if entry.get("row_payload") is None
                else json.loads(entry["row_payload"]),
                "retained_row": row_dump,
                "projection_fields": json.loads(entry["projection_fields"])
                if entry.get("projection_fields")
                else {},
                "components": components,
                "provenance": json.loads(entry["provenance"]),
                "summary": None if entry["summary"] is None else json.loads(entry["summary"]),
            }
            cells.append(ResultCell.from_payload(payload))
        return cells


def _restore_result_cell(payload: Mapping[str, Any]) -> ResultCell:
    return ResultCell.from_payload(payload)


def _arrow_schema() -> pa.Schema:
    comp = pa.struct(
        [
            ("status", pa.string()),
            ("code", pa.string()),
            ("reason", pa.string()),
            ("context", pa.string()),
            ("advisories", pa.list_(pa.string())),
            ("construction", pa.string()),
            ("kind", pa.string()),
            ("state_reason", pa.string()),
            ("supports", pa.list_(pa.string())),
        ]
    )
    return pa.schema(
        [
            ("schema_version", pa.int32()),
            ("row_kind", pa.string()),
            ("identity", pa.string()),
            ("label", pa.string()),
            ("family", pa.string()),
            ("source_snapshot_id", pa.string()),
            ("provenance", pa.string()),
            ("summary", pa.string()),
            ("retained_row", pa.string()),
            ("row_payload", pa.string()),
            ("projection_fields", pa.string()),
            ("sampling", comp),
            ("posterior", comp),
            ("sequential", comp),
        ]
    )


def _can_reconstruct_relevel(row: LiftEstimate) -> bool:
    relative = (
        row.estimand == "itt"
        and row.value_scale == "relative"
        and row.scale == "log"
        and row.quantile_p_value is None
        and row.independent_mean_reference is None
        and row.relative_unavailable_reason is None
        and row.lift is not None
        and row.lift.log_mean is not None
        and row.lift.log_se is not None
        and row.lift.open_side is None
    )
    additive = (
        row.estimand == "itt"
        and row.abs_diff is not None
        and row.abs_se is not None
        and row.abs_reference_df is not None
        and row.lift is None
        and row.relative_unavailable_reason is not None
    )
    return row.reference_kind == "t" and row.reference_df is not None and (relative or additive)


def _retained_relevel_alpha(row: LiftEstimate) -> float | None:
    alphas = [
        alpha
        for alpha in (
            None if row.lift is None else row.lift.alpha,
            row.abs_alpha,
        )
        if alpha is not None
    ]
    if not alphas and row.confidence_set is not None:
        alphas.append(row.confidence_set.alpha)
    if not alphas and row.relative_confidence_set is not None:
        alphas.append(row.relative_confidence_set.alpha)
    if not alphas:
        return None
    retained = min(alphas)
    return retained if row.alternative == "two-sided" else retained / 2.0


def _conservative_relevel(row: LiftEstimate, alpha: float) -> LiftEstimate:
    retained_alpha = _retained_relevel_alpha(row)
    if retained_alpha is None or alpha > retained_alpha:
        raise CodedError(
            "Portable releveling cannot narrow beyond the retained interval level.",
            code="results.operation.alpha_above_allocation",
            context={"requested_alpha": alpha, "retained_alpha": retained_alpha},
        )
    return row.reintervalize(alpha)


def _summary(row: LiftEstimate) -> EffectEstimate:
    return project_effect(row)


def _cell_from_arrow_payload(payload: Mapping[str, object]) -> ResultCell:
    return ResultCell.from_payload(payload)


def relevel(cell: ResultCell, *, alpha: float) -> ResultCell:
    _require(cell, Operation.RELEVEL)
    updated = _conservative_relevel(cell.retained_row(), alpha)
    return _replace_row(cell, updated)


def _reinterval(cell: ResultCell, *, threshold: float) -> ResultCell:
    row = cell.retained_row()
    construction, reason = classify(row)
    if construction is None:
        raise CodedError(
            "Unsupported result operation",
            code="results.operation.unsupported",
            context={"operation": "reinterval", "reason": reason},
        )
    updated = reinterval_selected(row, construction, threshold)
    return _replace_row(cell, updated)


def _require(cell: ResultCell, operation: Operation) -> None:
    if operation not in cell.supports["sampling"]:
        status, state = cell.components["sampling"]
        reason = (
            state.reason if state is not None else status.reason or "no retained sampling state"
        )
        raise CodedError(
            "Unsupported result operation",
            code="results.operation.unsupported",
            context={"operation": operation.value, "reason": reason},
        )


def _replace_row(cell: ResultCell, row: LiftEstimate) -> ResultCell:
    metadata_components = dict(cell.components)
    construction, reason = classify(row)
    row_fields = row.model_dump(mode="json")
    old_status, old_state = metadata_components["sampling"]
    if old_state is not None:
        sampling_status = ComponentStatus(
            old_status.state,
            old_status.code,
            reason if construction is None else old_status.reason,
            old_status.context,
            old_status.advisories,
        )
        metadata_components["sampling"] = (
            sampling_status,
            RetainedState(construction, reason, "lift_sampling", row_fields),
        )
    for name in ("posterior", "sequential"):
        status, state = metadata_components[name]
        if state is not None:
            metadata_components[name] = (
                status,
                RetainedState(construction, reason, state.kind, row_fields),
            )
    summary = (
        _summary(row)
        if old_state is not None or metadata_components["sequential"][0].state == "available"
        else None
    )
    return ResultCell(
        cell.identity,
        cell.label,
        cell.family,
        row.source_snapshot_id,
        metadata_components,
        cell.provenance,
        summary,
        _safe_projection_fields(row),
    )


@dataclass(frozen=True, slots=True)
class AnalysisEnvelope:
    cells: tuple[ResultCell, ...]
    metadata: ReadoutMetadata | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "cells": [cell.to_payload() for cell in self.cells],
            "metadata": None if self.metadata is None else self.metadata.model_dump(mode="json"),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AnalysisEnvelope:
        schema_version = payload.get("schema_version")
        if schema_version not in {1, _SCHEMA_VERSION}:
            raise CodedError(
                "Unsupported result-cell schema",
                code="results.cell.schema_unsupported",
                context={"schema_version": schema_version},
            )
        metadata = payload.get("metadata")
        return cls(
            tuple(ResultCell.from_payload(cell) for cell in payload["cells"]),
            None if metadata is None else ReadoutMetadata.model_validate(metadata),
        )

    def to_arrow(self) -> pa.Table:
        table = ResultCell.to_arrow(self.cells)
        metadata = dict(table.schema.metadata or {})
        if self.metadata is not None:
            metadata[_METADATA_KEY] = _json(self.metadata).encode("utf-8")
        return table.replace_schema_metadata(metadata)

    @classmethod
    def from_arrow(cls, table: pa.Table) -> AnalysisEnvelope:
        encoded = (table.schema.metadata or {}).get(_METADATA_KEY)
        metadata = None if encoded is None else ReadoutMetadata.model_validate(json.loads(encoded))
        return cls(tuple(ResultCell.from_arrow(table)), metadata)


def _report_detail(cell: ResultCell) -> str | None:
    from importlib import import_module

    model_paths = {
        "cate_validation": ("increment.estimation.targeting", "CateValidation"),
        "targeting": ("increment.estimation.targeting", "TargetingRule"),
        "targeting_selection": ("increment.estimation.targeting", "TargetingSelection"),
        "sensitivity": ("increment.estimation.confounding", "ConfoundingSensitivity"),
    }
    if cell.row_payload is None:
        return None
    identity_context = cell.row_payload.get("identity_context")
    control = identity_context.get("control") if isinstance(identity_context, Mapping) else None
    if cell.row_kind == "cate":
        return f"CATE report; control={control}" if control is not None else "CATE report"
    if cell.row_kind not in model_paths:
        return None
    module_name, model_name = model_paths[cell.row_kind]
    model = getattr(import_module(module_name), model_name)
    report = model.model_validate_json(_json(_plain(cell.row_payload["data"])))
    return f"{report}; control={control}" if control is not None else str(report)


def render(envelope: AnalysisEnvelope, audience: Literal["brief", "full"] = "brief") -> str:
    if audience not in {"brief", "full"}:
        raise CodedError(
            "Unsupported result rendering audience",
            code="results.render.audience_invalid",
            context={"audience": audience},
        )
    from increment._interpretation import (
        interpret_readout,
        render_readout_brief,
        render_readout_text,
    )

    rows = [cell._projection_row() for cell in envelope.cells]
    interpretation = interpret_readout(rows, envelope.metadata)
    brief = render_readout_brief(envelope.cells, interpretation)
    if audience == "brief":
        return "\n".join(brief)
    report_details = tuple(
        f"{cell.label}: {detail}"
        for cell in envelope.cells
        if (detail := _report_detail(cell)) is not None
    )
    return "\n".join((render_readout_text(interpretation), *report_details, *brief))
