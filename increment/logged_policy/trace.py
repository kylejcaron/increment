"""Logged decision records and complete-horizon trace admission.

``PROPENSITY_FLOOR`` is a design-time floor on every positive candidate-action
probability, chosen or not. Zero support is allowed only where the evaluated
policies also assign zero mass. No trace is trimmed or clipped after the fact;
a sub-floor positive probability refuses admission as evidence the logger
broke its contract. Refusing a realized trace this way selects on
the outcomes that produced the low propensity, so the floor is a property to
enforce in the logger, not a filter to rely on at analysis time.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from numbers import Real
from typing import Any, Self, cast

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from increment._identity import canonical_id_strings
from increment.errors import CodedModel, _thaw
from increment.logged_policy._refusals import _raise
from increment.logged_policy.policy import PolicyRegistry, _freeze, _validate_distribution

PROPENSITY_FLOOR = 0.05
_PROPENSITY_MATCH_TOLERANCE = 1e-9
_REGISTRY_AUDIT_TOKEN = object()
_RECORDED_AUDIT_TOKEN = object()
_AUDITED_CONSTRUCTORS = ("from_records", "from_frame")


def _admission_route(
    registry: PolicyRegistry | None,
    recorded: object | None,
    *,
    recorded_parameter: str = "logging_distributions",
) -> str:
    if (registry is None) == (recorded is None):
        _raise(
            "logged_policy.trace.admission_route",
            supplied=tuple(
                name
                for name, value in (("registry", registry), (recorded_parameter, recorded))
                if value is not None
            ),
            alternatives=("registry", recorded_parameter),
        )
    return "registry" if registry is not None else "recorded"


def _recorded_distribution(
    record: DecisionRecord, reported: Mapping[str, object]
) -> tuple[float, ...]:
    if not isinstance(reported, Mapping):
        _raise(
            "logged_policy.trace.logging_distribution_keys",
            unit_id=record.unit_id,
            decision_index=record.decision_index,
            actions=type(reported).__name__,
            expected_actions=record.candidate_actions,
        )
    expected = record.candidate_actions
    if set(reported) != set(expected):
        _raise(
            "logged_policy.trace.logging_distribution_keys",
            unit_id=record.unit_id,
            decision_index=record.decision_index,
            actions=tuple(reported),
            expected_actions=expected,
        )
    probabilities = _reported_probabilities(reported)
    if probabilities is None:
        _raise(
            "logged_policy.support.policy_not_normalized",
            policy_id=record.logging_policy_id,
            version=record.logging_policy_version,
            context=dict(record.pre_decision_context),
            probabilities=dict(reported),
        )
    _validate_distribution(
        probabilities,
        policy_id=record.logging_policy_id,
        version=record.logging_policy_version,
        context=dict(record.pre_decision_context),
    )
    for action, probability in probabilities.items():
        if 0.0 < probability < PROPENSITY_FLOOR:
            _raise(
                "logged_policy.trace.propensity_below_floor",
                unit_id=record.unit_id,
                decision_index=record.decision_index,
                propensity=probability,
                floor=PROPENSITY_FLOOR,
                action=action,
                logging_policy_id=record.logging_policy_id,
                logging_policy_version=record.logging_policy_version,
                context=dict(record.pre_decision_context),
                route="record a complete logging distribution whose every positive candidate-action probability meets the declared propensity floor",
            )
    return tuple(probabilities[action] for action in expected)


RECORD_COLUMNS: tuple[str, ...] = (
    "decision_time",
    "unit_id",
    "decision_index",
    "candidate_actions",
    "chosen_action",
    "propensity",
    "logging_policy_id",
    "logging_policy_version",
    "update_batch",
    "reward_observation_boundary",
    "reward",
)


def _is_aware(when: datetime) -> bool:
    return when.tzinfo is not None and when.tzinfo.utcoffset(when) is not None


class DecisionRecord(CodedModel, BaseModel):
    """One logged decision and, once its boundary closes, its reward.

    ``propensity`` is the exact logging probability of ``chosen_action`` at
    decision time and must be at least ``PROPENSITY_FLOOR``, which the
    logging policy enforces by design. ``reward`` is ``None`` while the
    observation boundary is open; a trace admits a unit only when every
    reward is closed.

    Python-mode dumps preserve context types. Lossless JSON round trips require
    JSON-native context values; strings and arrays cannot restore arbitrary types.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_time: datetime
    unit_id: str = Field(min_length=1)
    decision_index: int = Field(ge=1)
    candidate_actions: tuple[str, ...] = Field(min_length=1)
    chosen_action: str
    propensity: float | None
    logging_policy_id: str = Field(min_length=1)
    logging_policy_version: str = Field(min_length=1)
    pre_decision_context: Mapping[str, Any] = Field(default_factory=dict)
    update_batch: str = Field(min_length=1)
    reward_observation_boundary: datetime
    reward: float | None = None

    @field_validator("pre_decision_context", mode="before")
    @classmethod
    def _raw_context_in_domain(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            _raise(
                "logged_policy.trace.context_value_unsupported",
                value_type=type(value).__name__,
            )
        return _freeze(value)

    @field_serializer("pre_decision_context")
    def _serialize_pre_decision_context(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return cast("dict[str, Any]", _thaw(value))

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        return type(self).model_validate, (self.model_dump(mode="python"),)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        copied = type(self).model_validate(self.model_dump(mode="python"))
        if memo is not None:
            memo[id(self)] = copied
        return copied

    @model_validator(mode="after")
    def _admissible(self) -> DecisionRecord:
        where = {"unit_id": self.unit_id, "decision_index": self.decision_index}
        if len(set(self.candidate_actions)) != len(self.candidate_actions) or (
            self.chosen_action not in self.candidate_actions
        ):
            _raise(
                "logged_policy.trace.candidate_set_mismatch",
                **where,
                candidate_actions=self.candidate_actions,
                expected_candidate_actions=tuple(dict.fromkeys(self.candidate_actions)),
                chosen_action=self.chosen_action,
            )
        if self.propensity is None:
            _raise(
                "logged_policy.trace.propensity_missing",
                **where,
                logging_policy_id=self.logging_policy_id,
                logging_policy_version=self.logging_policy_version,
            )
        if not math.isfinite(self.propensity):
            _raise("logged_policy.trace.propensity_nonfinite", **where, propensity=self.propensity)
        if not 0.0 < self.propensity < 1.0:
            _raise(
                "logged_policy.trace.propensity_out_of_range", **where, propensity=self.propensity
            )
        if self.propensity < PROPENSITY_FLOOR:
            _raise(
                "logged_policy.trace.propensity_below_floor",
                **where,
                propensity=self.propensity,
                floor=PROPENSITY_FLOOR,
                action=self.chosen_action,
                logging_policy_id=self.logging_policy_id,
                logging_policy_version=self.logging_policy_version,
                context=dict(self.pre_decision_context),
                route=(
                    "supply a logging distribution whose every positive candidate-action "
                    "probability meets the declared propensity floor"
                ),
            )
        if self.reward is not None and not math.isfinite(self.reward):
            _raise("logged_policy.trace.reward_nonfinite", **where, reward=self.reward)
        if _is_aware(self.decision_time) != _is_aware(self.reward_observation_boundary):
            _raise(
                "logged_policy.trace.boundary_awareness_mixed",
                **where,
                decision_time=self.decision_time,
                reward_observation_boundary=self.reward_observation_boundary,
            )
        if self.reward_observation_boundary < self.decision_time:
            _raise(
                "logged_policy.trace.boundary_before_decision",
                **where,
                decision_time=self.decision_time,
                reward_observation_boundary=self.reward_observation_boundary,
            )
        object.__setattr__(self, "pre_decision_context", _freeze(self.pre_decision_context))
        return self

    @property
    def _logging_policy_key(self) -> tuple[str, str]:
        """Structured logging identity used for admission and diagnostics."""
        return (self.logging_policy_id, self.logging_policy_version)

    @property
    def logging_policy(self) -> str:
        """Slash-joined logging identity reserved for display."""
        return f"{self.logging_policy_id}/{self.logging_policy_version}"


def _reported_probabilities(reported: Mapping[str, object]) -> dict[str, float] | None:
    """Float every Real policy output; ``None`` once one is not Real or overflows a float."""
    probabilities: dict[str, float] = {}
    for action, value in reported.items():
        if not isinstance(value, Real):
            return None
        try:
            probabilities[action] = float(value)
        except (OverflowError, ValueError):
            return None
    return probabilities


def _audit_logging_distribution(
    record: DecisionRecord, registry: PolicyRegistry
) -> tuple[float, ...]:
    policy = registry.get(record.logging_policy_id, record.logging_policy_version)
    if policy is None:
        _raise(
            "logged_policy.trace.logging_policy_unregistered",
            unit_id=record.unit_id,
            decision_index=record.decision_index,
            logging_policy_id=record.logging_policy_id,
            logging_policy_version=record.logging_policy_version,
            registered=registry.versions,
        )
    context = record.pre_decision_context
    reported = {action: policy.probability(action, context) for action in record.candidate_actions}
    probabilities = _reported_probabilities(reported)
    if probabilities is None:
        _raise(
            "logged_policy.support.policy_not_normalized",
            policy_id=record.logging_policy_id,
            version=record.logging_policy_version,
            context=dict(context),
            probabilities=reported,
        )
    _validate_distribution(
        probabilities,
        policy_id=record.logging_policy_id,
        version=record.logging_policy_version,
        context=dict(context),
    )
    for action, probability in probabilities.items():
        if 0.0 < probability < PROPENSITY_FLOOR:
            _raise(
                "logged_policy.trace.propensity_below_floor",
                unit_id=record.unit_id,
                decision_index=record.decision_index,
                propensity=probability,
                floor=PROPENSITY_FLOOR,
                action=action,
                logging_policy_id=record.logging_policy_id,
                logging_policy_version=record.logging_policy_version,
                context=dict(context),
                route=(
                    "register a logging policy whose every positive candidate-action "
                    "probability meets the declared propensity floor"
                ),
            )
    return tuple(probabilities.values())


def _ordered_admission(
    records: Iterable[DecisionRecord | Mapping[str, Any]],
    *,
    registry: PolicyRegistry | None,
    logging_distributions: Iterable[Mapping[str, object]] | None,
) -> tuple[tuple[DecisionRecord, ...], tuple[tuple[float, ...], ...], object]:
    route = _admission_route(registry, logging_distributions)
    typed = [r if isinstance(r, DecisionRecord) else DecisionRecord(**r) for r in records]
    if not typed:
        _raise("logged_policy.trace.empty", n_records=0, constructors=_AUDITED_CONSTRUCTORS)
    if route == "registry":
        assert registry is not None
        paired = [(record, _audit_logging_distribution(record, registry)) for record in typed]
    else:
        assert logging_distributions is not None
        supplied = list(logging_distributions)
        if len(supplied) != len(typed):
            _raise(
                "logged_policy.trace.logging_distribution_misaligned",
                n_records=len(typed),
                n_distributions=len(supplied),
            )
        paired = [
            (record, _recorded_distribution(record, distribution))
            for record, distribution in zip(typed, supplied, strict=True)
        ]
    paired.sort(key=lambda pair: (pair[0].unit_id, pair[0].decision_index))
    ordered, distributions = zip(*paired, strict=True)
    token = _REGISTRY_AUDIT_TOKEN if route == "registry" else _RECORDED_AUDIT_TOKEN
    return tuple(ordered), tuple(distributions), token


def _admit_record(
    record: DecisionRecord, distribution: tuple[float, ...], expected: tuple[str, ...]
) -> None:
    """Refuse one record that breaks the trace-wide admission rules."""
    where = {"unit_id": record.unit_id, "decision_index": record.decision_index}
    if record.candidate_actions != expected:
        _raise(
            "logged_policy.trace.candidate_set_mismatch",
            **where,
            candidate_actions=record.candidate_actions,
            expected_candidate_actions=expected,
            chosen_action=record.chosen_action,
        )
    if record.reward is None:
        _raise(
            "logged_policy.trace.open_reward",
            **where,
            reward_observation_boundary=record.reward_observation_boundary,
        )
    registered = distribution[expected.index(record.chosen_action)]
    assert record.propensity is not None  # DecisionRecord refuses a missing propensity
    if not math.isclose(
        registered, record.propensity, rel_tol=_PROPENSITY_MATCH_TOLERANCE, abs_tol=0.0
    ):
        _raise(
            "logged_policy.trace.propensity_stale",
            **where,
            logging_policy_id=record.logging_policy_id,
            logging_policy_version=record.logging_policy_version,
            chosen_action=record.chosen_action,
            logged=record.propensity,
            registered=registered,
        )


class LoggedTrace(CodedModel, BaseModel):
    """Admitted, ordered, complete-horizon decision records.

    Construct with :meth:`from_records` or :meth:`from_frame`, which admit
    either a registered logging policy or a complete recorded distribution.
    The model validator re-checks every structural admission rule, so a trace
    instance is always ordered by ``(unit_id, decision_index)``, has exactly
    ``horizon`` closed rewards per unit, one fixed candidate set, and one
    admitted logging distribution per record.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    records: tuple[DecisionRecord, ...] = Field(min_length=1)
    horizon: int = Field(ge=1)
    logging_distributions: tuple[tuple[float, ...], ...]

    @classmethod
    def from_records(
        cls,
        records: Iterable[DecisionRecord | Mapping[str, Any]],
        *,
        registry: PolicyRegistry | None = None,
        logging_distributions: Iterable[Mapping[str, object]] | None = None,
        horizon: int | None = None,
    ) -> LoggedTrace:
        ordered, distributions, token = _ordered_admission(
            records,
            registry=registry,
            logging_distributions=logging_distributions,
        )
        resolved = max(r.decision_index for r in ordered) if horizon is None else horizon
        return cls.model_validate(
            {"records": ordered, "horizon": resolved, "logging_distributions": distributions},
            context={"audit": token},
        )

    @classmethod
    def from_frame(
        cls,
        frame: IntoDataFrame,
        *,
        registry: PolicyRegistry | None = None,
        logging_distribution_column: str | None = None,
        context_columns: Sequence[str] = ("x",),
        horizon: int | None = None,
    ) -> LoggedTrace:
        route = _admission_route(
            registry,
            logging_distribution_column,
            recorded_parameter="logging_distribution_column",
        )
        nwf = nw.from_native(frame, eager_only=True)
        required = (*RECORD_COLUMNS, *context_columns)
        if route == "recorded":
            assert logging_distribution_column is not None
            required += (logging_distribution_column,)
        missing = [c for c in required if c not in nwf.columns]
        if missing:
            _raise("logged_policy.trace.missing_columns", missing=tuple(missing))
        unit_column = nwf.get_column("unit_id")
        null_unit_ids = unit_column.is_null().to_list()
        if unit_column.dtype.is_integer() and not any(null_unit_ids):
            native_unit_ids = unit_column.to_numpy()
        else:
            native_unit_ids = np.asarray(unit_column.to_list(), dtype=object)
            for row_index, unit_id in enumerate(native_unit_ids):
                if null_unit_ids[row_index] or (
                    isinstance(unit_id, Real) and not math.isfinite(unit_id)
                ):
                    _raise("logged_policy.trace.unit_id_missing", row=row_index, unit_id=unit_id)
        canonical_unit_ids = canonical_id_strings(native_unit_ids, what="trace frame unit_id")
        records = []
        distributions = []
        for row_index, row in enumerate(nwf.iter_rows(named=True)):
            candidates = row["candidate_actions"]
            if isinstance(candidates, str):
                candidates = tuple(part.strip() for part in candidates.split(","))
            records.append(
                DecisionRecord(
                    decision_time=row["decision_time"],
                    unit_id=canonical_unit_ids[row_index],
                    decision_index=row["decision_index"],
                    candidate_actions=tuple(candidates),
                    chosen_action=row["chosen_action"],
                    propensity=row["propensity"],
                    logging_policy_id=row["logging_policy_id"],
                    logging_policy_version=row["logging_policy_version"],
                    pre_decision_context={c: row[c] for c in context_columns},
                    update_batch=row["update_batch"],
                    reward_observation_boundary=row["reward_observation_boundary"],
                    reward=row["reward"],
                )
            )
            if route == "recorded":
                assert logging_distribution_column is not None
                distributions.append(row[logging_distribution_column])
        return cls.from_records(
            records,
            registry=registry,
            logging_distributions=distributions if route == "recorded" else None,
            horizon=horizon,
        )

    @model_validator(mode="after")
    def _admitted(self, info: ValidationInfo) -> LoggedTrace:
        records = self.records
        if len(self.logging_distributions) != len(records):
            _raise(
                "logged_policy.trace.logging_distribution_misaligned",
                n_records=len(records),
                n_distributions=len(self.logging_distributions),
            )
        audit = None if info.context is None else info.context.get("audit")
        if audit is not _REGISTRY_AUDIT_TOKEN and audit is not _RECORDED_AUDIT_TOKEN:
            _raise(
                "logged_policy.trace.registry_audit_required", constructors=_AUDITED_CONSTRUCTORS
            )
        keys = [(r.unit_id, r.decision_index) for r in records]
        if keys != sorted(keys):
            first = next(
                i for i, (a, b) in enumerate(zip(keys, sorted(keys), strict=True)) if a != b
            )
            _raise(
                "logged_policy.trace.unordered",
                unit_id=records[first].unit_id,
                decision_indices=tuple(k[1] for k in keys if k[0] == records[first].unit_id),
                decision_times=tuple(
                    r.decision_time for r in records if r.unit_id == records[first].unit_id
                ),
            )
        reference = records[0]
        aware = _is_aware(reference.decision_time)
        for record in records:
            if _is_aware(record.decision_time) != aware:
                _raise(
                    "logged_policy.trace.decision_time_awareness_mixed",
                    unit_id=record.unit_id,
                    decision_index=record.decision_index,
                    decision_time=record.decision_time,
                    reference_unit_id=reference.unit_id,
                    reference_decision_index=reference.decision_index,
                    reference_decision_time=reference.decision_time,
                )
        expected = records[0].candidate_actions
        by_unit: dict[str, list[DecisionRecord]] = {}
        for record, distribution in zip(records, self.logging_distributions, strict=True):
            _admit_record(record, distribution, expected)
            by_unit.setdefault(record.unit_id, []).append(record)
        for unit_id, unit_records in by_unit.items():
            indices = [r.decision_index for r in unit_records]
            times = [r.decision_time for r in unit_records]
            if indices != list(range(1, len(indices) + 1)) or any(
                later <= earlier for earlier, later in zip(times, times[1:], strict=False)
            ):
                _raise(
                    "logged_policy.trace.unordered",
                    unit_id=unit_id,
                    decision_indices=tuple(indices),
                    decision_times=tuple(times),
                )
            if len(indices) != self.horizon:
                _raise(
                    "logged_policy.trace.incomplete_horizon",
                    unit_id=unit_id,
                    observed_horizon=len(indices),
                    horizon=self.horizon,
                )
        return self

    @property
    def candidate_actions(self) -> tuple[str, ...]:
        return self.records[0].candidate_actions

    @property
    def unit_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(r.unit_id for r in self.records))

    @property
    def n_units(self) -> int:
        return len(self.unit_ids)

    @property
    def _logging_policy_keys(self) -> tuple[tuple[str, str], ...]:
        by_time = sorted(self.records, key=lambda r: (r.decision_time, r.unit_id, r.decision_index))
        return tuple(dict.fromkeys(r._logging_policy_key for r in by_time))

    @property
    def logging_policy_versions(self) -> tuple[str, ...]:
        return tuple(f"{policy_id}/{version}" for policy_id, version in self._logging_policy_keys)

    @property
    def update_batches(self) -> tuple[str, ...]:
        by_time = sorted(self.records, key=lambda r: (r.decision_time, r.unit_id, r.decision_index))
        return tuple(dict.fromkeys(r.update_batch for r in by_time))
