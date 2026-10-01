"""Prospective unit-cycle moments, guaranteed bounds, and actual-model planning.

Effect arguments are absolute retained-aggregate effects, in the same units as
``procedure.null_abs``. Effects are normalized to binary64 before calculation.
MDE searches follow the favorable side of that null;
two-sided procedures use preferred_direction (increase when neutral/unspecified).
All Monte Carlo probabilities are unconditional, including admission refusals.
Simultaneous bands use DKW or finite-domain Hoeffding bounds and cover every
inspected binary64 effect, including atoms/ties. They quantify Monte Carlo
uncertainty, not misspecification of the declared law.
"""

from __future__ import annotations

import math
import sqlite3
import sys
from collections import Counter
from collections.abc import Callable, Generator, Hashable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, localcontext
from fractions import Fraction
from functools import cached_property
from tempfile import TemporaryDirectory
from typing import Literal, NoReturn

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from increment._unit_cycle import (
    FINITE_MAX,
    NEAREST_OVERFLOW,
    UnitCycleEnvelopeCutoff,
    unit_cycle_admission_outcome,
    unit_cycle_admission_rule,
    unit_cycle_envelope_cutoff,
    unit_cycle_report,
    unit_cycle_reporting_bounds,
)
from increment.errors import (
    CodedModel,
    InvalidRequestError,
    RefusalSpec,
    _safe_error_value,
    refuse,
)
from increment.estimation.decision_types import ContrastDecisionProcedure
from increment.power._search import (
    bisect_first_true,
    float_from_ordinal,
    float_ordinal,
    require_reason_when_null,
)
from increment.semantics.unit_cycle import (
    UnitCycleCycleLaw,
    UnitCycleInnovation,
    UnitCycleJointLaw,
    UnitCycleVarianceEnvelope,
)

_REFUSALS = {
    f"power.unit_cycle.{name}": RefusalSpec(
        f"power.unit_cycle.{name}",
        InvalidRequestError,
        template=(
            "{detail} (field={field!r}, value={value!r}, "
            "constraint={constraint!r}; route={route!r})"
        ),
        keys=frozenset({"field", "value", "constraint", "route", "detail"}),
    )
    for name in ("law", "procedure", "input", "numerical", "result")
}


def _invalid(
    name: str,
    message: str,
    *,
    field: str,
    value: object,
    constraint: str,
    route: str,
) -> NoReturn:
    """Raise a refusal naming the rejected field(s), their bounded values, the
    violated constraint and a repair route.

    A ``/``-joined field names one entry of the tuple ``value`` per segment.
    """
    refuse(
        _REFUSALS[f"power.unit_cycle.{name}"],
        detail=message,
        field=field,
        value=(
            tuple(_safe_error_value(item) for item in value)
            if isinstance(value, (tuple, list))
            else _safe_error_value(value)
        ),
        constraint=constraint,
        route=route,
    )


def _invalid_model(name: str, argument: str, exc: ValidationError) -> NoReturn:
    """Refuse a forged model argument at its first failing location."""
    error = exc.errors(include_url=False, include_context=False)[0]
    field = ".".join((argument, *(str(part) for part in error["loc"])))
    _invalid(
        name,
        f"{argument} failed {exc.title} validation with {exc.error_count()} error(s)",
        field=field,
        value=error["input"],
        constraint=error["msg"],
        route=f"Correct {field}, then pass the validated {exc.title}.",
    )


def _mismatch(*pairs: tuple[str, object, str | None, object]) -> tuple[str, tuple[object, ...]]:
    """Aligned field names and values of every ``(field, actual, source,
    expected)`` pair that disagrees; an empty field means all agree.

    A ``None`` source marks a required constant, so only the actual field is named.
    """
    fields: list[str] = []
    values: list[object] = []
    for field, actual, source, expected in pairs:
        if actual != expected:
            fields.append(field)
            values.append(actual)
            if source is not None:
                fields.append(source)
                values.append(expected)
    return "/".join(fields), tuple(values)


def _first_repeat(values: Sequence[Hashable]) -> int | None:
    """Index of the first entry equal to an earlier entry."""
    seen: set[Hashable] = set()
    for index, value in enumerate(values):
        if value in seen:
            return index
        seen.add(value)
    return None


def _require_reason(result: BaseModel, value_field: str, reason_field: str, *, route: str) -> None:
    value, reason = getattr(result, value_field), getattr(result, reason_field)
    require_reason_when_null(
        value,
        reason,
        name=value_field,
        invalid=lambda message: _invalid(
            "result",
            message,
            field=f"{value_field}/{reason_field}",
            value=(value, reason),
            constraint=f"exactly one of {value_field} and {reason_field} is null",
            route=route,
        ),
    )


def _validate_sampling_protocol(rng: str, sampler: str) -> None:
    if (rng, sampler) not in (
        (
            "numpy.PCG64/SeedSequence(seed,replicate)/unit-prefix-v1",
            "finite_type_reuse_exact_micro_v1",
        ),
        (
            "numpy.PCG64/SeedSequence(seed,replicate)/unit-prefix-v2",
            "finite_type_reuse_exact_micro_v2",
        ),
    ):
        _invalid(
            "result",
            "RNG and sampler must use matching protocol versions",
            field="rng/sampler",
            value=(rng, sampler),
            constraint=(
                "rng and sampler are the unit-prefix-v1/micro_v1 or unit-prefix-v2/micro_v2 pair"
            ),
            route="Record rng and sampler from the same protocol version.",
        )


class _Frozen(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")


class _AvailabilityResult(_Frozen):
    evaluation_scope: Literal["affine_unit_sufficient_state"] = "affine_unit_sufficient_state"
    availability_method: Literal["exact_reporting_certified_descriptive_v1"] = (
        "exact_reporting_certified_descriptive_v1"
    )


class UnitCycleLawMoments(_Frozen):
    """Population moments; correlation is undefined if either SD is zero."""

    law: UnitCycleJointLaw
    reference_effect: float | None = Field(allow_inf_nan=False)
    sd_a: float | None = Field(ge=0, allow_inf_nan=False)
    sd_g: float | None = Field(ge=0, allow_inf_nan=False)
    rho: float | None = Field(ge=-1, le=1, allow_inf_nan=False)
    residual_variance_upper: float | None = Field(ge=0, allow_inf_nan=False)
    reference_effect_unavailable_reason: Literal["unrepresentable"] | None = None
    sd_a_unavailable_reason: Literal["unrepresentable"] | None = None
    sd_g_unavailable_reason: Literal["unrepresentable"] | None = None
    rho_unavailable_reason: Literal["unrepresentable", "zero_variance"] | None = None
    residual_variance_upper_unavailable_reason: Literal["unrepresentable"] | None = None

    @model_validator(mode="after")
    def _reasons(self) -> UnitCycleLawMoments:
        for name in ("reference_effect", "sd_a", "sd_g", "rho", "residual_variance_upper"):
            _require_reason(
                self,
                name,
                f"{name}_unavailable_reason",
                route=(
                    f"Set {name}_unavailable_reason exactly when {name} is null, "
                    "as unit_cycle_law_moments reports."
                ),
            )
        return self


class UnitCyclePowerLowerBoundResult(_AvailabilityResult):
    power_kind: Literal["certified_power_lower_bound"] = "certified_power_lower_bound"
    envelope: UnitCycleVarianceEnvelope
    procedure: ContrastDecisionProcedure
    n: int = Field(ge=1, strict=True)
    effect_delta: float = Field(allow_inf_nan=False)
    lower_bound: float = Field(ge=0, le=1, allow_inf_nan=False)
    cutoff: float = Field(ge=0, allow_inf_nan=False)
    effective_alpha: float = Field(gt=0, lt=1, allow_inf_nan=False)
    refusal_probability_upper: float = Field(ge=0, le=1, allow_inf_nan=False)
    reason: (
        Literal[
            "non_favorable_effect", "insufficient_separation", "runtime_availability_not_certified"
        ]
        | None
    ) = None


class UnitCycleFailureCount(_Frozen):
    reason: str = Field(min_length=1)
    count: int = Field(ge=1, strict=True)


class UnitCycleModelPowerResult(_AvailabilityResult):
    """Certified and possible successful rejections of the sufficient-state adapter.

    ``power`` is null if admitted numerical evaluation is incomplete. Split-refused
    draws are known non-rejections even if their innovations also failed numerically.
    ``rejected`` counts successfully evaluated, admitted rejections.
    """

    power_kind: Literal["model_mc_power"] = "model_mc_power"
    law: UnitCycleJointLaw
    procedure: ContrastDecisionProcedure
    n: int = Field(ge=1, strict=True)
    effect_delta: float = Field(allow_inf_nan=False)
    power: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    lower_bound: float = Field(ge=0, le=1, allow_inf_nan=False)
    upper_bound: float = Field(ge=0, le=1, allow_inf_nan=False)
    power_unavailable_reason: Literal["numerical_failures"] | None = None
    attempted: int = Field(ge=1, strict=True)
    admitted: int = Field(ge=0, strict=True)
    failed: int = Field(ge=0, strict=True)
    failed_refused: int = Field(ge=0, strict=True)
    rejected: int = Field(ge=0, strict=True)
    possible_rejected: int = Field(ge=0, strict=True)
    known_runtime_refused: int = Field(ge=0, strict=True)
    availability_uncertified: int = Field(ge=0, strict=True)
    sampling_failed: int = Field(ge=0, strict=True)
    failures: tuple[UnitCycleFailureCount, ...] = ()
    seed: int = Field(ge=0, strict=True)
    rng: Literal[
        "numpy.PCG64/SeedSequence(seed,replicate)/unit-prefix-v1",
        "numpy.PCG64/SeedSequence(seed,replicate)/unit-prefix-v2",
    ] = "numpy.PCG64/SeedSequence(seed,replicate)/unit-prefix-v1"
    numpy_version: str = np.__version__
    sampler: Literal["finite_type_reuse_exact_micro_v1", "finite_type_reuse_exact_micro_v2"] = (
        "finite_type_reuse_exact_micro_v1"
    )
    mc_method: Literal[
        "simultaneous_dkw_massart",
        "simultaneous_binary64_hoeffding",
        "fixed_query_bernoulli_hoeffding",
    ] = "simultaneous_dkw_massart"
    mc_error: float = Field(gt=0, lt=1, allow_inf_nan=False)
    cdf_bands: int = Field(ge=2, le=8, strict=True)
    cdf_terms_per_bound: int = Field(ge=1, le=4, strict=True)
    cdf_error_each: float = Field(gt=0, lt=1, allow_inf_nan=False)
    cdf_radius: float = Field(ge=0, le=1, allow_inf_nan=False)
    numerical_status: Literal["ok", "incomplete"]
    mc_status: Literal["bounded"] = "bounded"
    cutoff: float = Field(ge=0, allow_inf_nan=False)
    effective_alpha: float = Field(gt=0, lt=1, allow_inf_nan=False)
    refusal_probability_upper: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def _counts(self) -> UnitCycleModelPowerResult:
        _validate_sampling_protocol(self.rng, self.sampler)
        unknown = self.failed - self.failed_refused
        if not 0 <= unknown <= self.admitted:
            _invalid(
                "result",
                "admitted numerical failures must be a subset of admissions",
                field="failed/failed_refused/admitted",
                value=(self.failed, self.failed_refused, self.admitted),
                constraint="0 <= failed - failed_refused <= admitted",
                route="Count unrefused numerical failures only among admitted draws.",
            )
        if not 0 <= self.rejected <= self.admitted - unknown <= self.attempted:
            _invalid(
                "result",
                "inconsistent attempted/admitted/failed/rejected counts",
                field="rejected/admitted/failed/failed_refused/attempted",
                value=(
                    self.rejected,
                    self.admitted,
                    self.failed,
                    self.failed_refused,
                    self.attempted,
                ),
                constraint="0 <= rejected <= admitted - (failed - failed_refused) <= attempted",
                route=(
                    "Count rejections only among successfully evaluated admitted draws, "
                    "and admissions only among attempted draws."
                ),
            )
        if unknown != (
            self.known_runtime_refused + self.availability_uncertified + self.sampling_failed
        ):
            _invalid(
                "result",
                "admitted failure categories must partition numerical failures",
                field=(
                    "failed/failed_refused/known_runtime_refused/"
                    "availability_uncertified/sampling_failed"
                ),
                value=(
                    self.failed,
                    self.failed_refused,
                    self.known_runtime_refused,
                    self.availability_uncertified,
                    self.sampling_failed,
                ),
                constraint=(
                    "failed - failed_refused == "
                    "known_runtime_refused + availability_uncertified + sampling_failed"
                ),
                route="Assign every admitted numerical failure to exactly one failure category.",
            )
        if (
            not self.rejected
            <= self.possible_rejected
            <= self.admitted - self.known_runtime_refused
        ):
            _invalid(
                "result",
                "invalid certified/possible rejection counts",
                field="rejected/possible_rejected/admitted/known_runtime_refused",
                value=(
                    self.rejected,
                    self.possible_rejected,
                    self.admitted,
                    self.known_runtime_refused,
                ),
                constraint="rejected <= possible_rejected <= admitted - known_runtime_refused",
                route=(
                    "Keep possible_rejected between the certified rejections and the "
                    "admitted draws not known to be refused at runtime."
                ),
            )
        if (
            self.possible_rejected - self.rejected
            > self.availability_uncertified + self.sampling_failed
        ):
            _invalid(
                "result",
                "unresolved rejection counts require unresolved availability",
                field="possible_rejected/rejected/availability_uncertified/sampling_failed",
                value=(
                    self.possible_rejected,
                    self.rejected,
                    self.availability_uncertified,
                    self.sampling_failed,
                ),
                constraint=(
                    "possible_rejected - rejected <= availability_uncertified + sampling_failed"
                ),
                route=(
                    "Count only availability-uncertified or sampling-failed draws as "
                    "unresolved possible rejections."
                ),
            )
        fixed = self.mc_method == "fixed_query_bernoulli_hoeffding"
        finite_domain = self.mc_method == "simultaneous_binary64_hoeffding"
        terms = (
            1 if fixed or finite_domain else (4 if self.procedure.alternative == "two-sided" else 2)
        )
        if self.cdf_terms_per_bound != terms or self.cdf_bands != 2 * terms:
            _invalid(
                "result",
                "Monte Carlo processes must match the declared coverage method",
                field="mc_method/procedure.alternative/cdf_terms_per_bound/cdf_bands",
                value=(
                    self.mc_method,
                    self.procedure.alternative,
                    self.cdf_terms_per_bound,
                    self.cdf_bands,
                ),
                constraint=(
                    f"cdf_terms_per_bound == {terms} and cdf_bands == {2 * terms} "
                    "for this mc_method and alternative"
                ),
                route="Record the band counts that mc_method uses for this alternative.",
            )
        if Fraction(self.cdf_error_each) * self.cdf_bands > Fraction(self.mc_error):
            _invalid(
                "result",
                "CDF allocations exceed the Monte Carlo error budget",
                field="cdf_error_each/cdf_bands/mc_error",
                value=(self.cdf_error_each, self.cdf_bands, self.mc_error),
                constraint="cdf_error_each * cdf_bands <= mc_error in exact arithmetic",
                route="Allocate at most mc_error across all cdf_bands.",
            )
        if finite_domain:
            each, radius = _binary64_band(self.attempted, self.mc_error)
        elif fixed:
            each, radius = _fixed_band(self.attempted, self.mc_error)
        else:
            each, radius = _band(self.attempted, self.mc_error, 2 * terms)
        bounds = _bounds(self.rejected, self.possible_rejected, self.attempted, radius, terms)
        if (
            self.cdf_error_each != each
            or self.cdf_radius != radius
            or (self.lower_bound, self.upper_bound) != bounds
        ):
            _invalid(
                "result",
                "Power bounds must match their declared coverage method and budget",
                field="cdf_error_each/cdf_radius/lower_bound/upper_bound",
                value=(self.cdf_error_each, self.cdf_radius, self.lower_bound, self.upper_bound),
                constraint=(
                    f"cdf_error_each == {each!r}, cdf_radius == {radius!r} and "
                    f"(lower_bound, upper_bound) == {bounds!r} for {self.mc_method} "
                    f"with mc_error={self.mc_error!r} over the attempted draws"
                ),
                route="Record the allocation, radius and bounds recomputed from these counts.",
            )
        if not 0 <= self.failed_refused <= min(self.failed, self.attempted - self.admitted):
            _invalid(
                "result",
                "invalid refused numerical-failure count",
                field="failed_refused/failed/attempted/admitted",
                value=(self.failed_refused, self.failed, self.attempted, self.admitted),
                constraint="0 <= failed_refused <= min(failed, attempted - admitted)",
                route="Count as refused failures only failed draws that admission refused.",
            )
        if self.admitted > self.attempted or sum(x.count for x in self.failures) != self.failed:
            _invalid(
                "result",
                "failure reasons must account for every numerical failure",
                field="admitted/attempted/sum(failures.count)/failed",
                value=(
                    self.admitted,
                    self.attempted,
                    sum(x.count for x in self.failures),
                    self.failed,
                ),
                constraint="admitted <= attempted and sum(failures.count) == failed",
                route="Record one failure reason count for every failed draw.",
            )
        repeated = _first_repeat([x.reason for x in self.failures])
        if repeated is not None:
            _invalid(
                "result",
                "failure reasons must be unique",
                field=f"failures[{repeated}].reason",
                value=self.failures[repeated].reason,
                constraint="each failure reason appears in one failures entry",
                route="Merge the repeated failure reason into a single count.",
            )
        if (self.power is None) != (unknown > 0):
            _invalid(
                "result",
                "power is unavailable exactly when admitted numerical draws failed",
                field="power/failed/failed_refused",
                value=(self.power, self.failed, self.failed_refused),
                constraint="power is null exactly when failed - failed_refused > 0",
                route="Null power exactly when admitted draws failed numerically.",
            )
        if (self.power_unavailable_reason is not None) != (self.power is None):
            _invalid(
                "result",
                "a null power requires its reason",
                field="power/power_unavailable_reason",
                value=(self.power, self.power_unavailable_reason),
                constraint="power_unavailable_reason is set exactly when power is null",
                route=(
                    "Set power_unavailable_reason='numerical_failures' exactly when power is null."
                ),
            )
        if self.numerical_status != ("incomplete" if self.failed else "ok"):
            _invalid(
                "result",
                "numerical status must match failure counts",
                field="numerical_status/failed",
                value=(self.numerical_status, self.failed),
                constraint="numerical_status == 'incomplete' when failed > 0, else 'ok'",
                route="Derive numerical_status from the failed count.",
            )
        if self.lower_bound > self.upper_bound:
            _invalid(
                "result",
                "power bounds must be ordered",
                field="lower_bound/upper_bound",
                value=(self.lower_bound, self.upper_bound),
                constraint="lower_bound <= upper_bound",
                route="Record the outward bounds recomputed from these counts.",
            )
        if self.power is not None:
            if self.power != self.rejected / self.attempted:
                _invalid(
                    "result",
                    "available power must equal rejected divided by attempted draws",
                    field="power/rejected/attempted",
                    value=(self.power, self.rejected, self.attempted),
                    constraint="power == rejected / attempted",
                    route="Set power to rejected / attempted.",
                )
            if not self.lower_bound <= self.power <= self.upper_bound:
                _invalid(
                    "result",
                    "power must belong to its bounds",
                    field="power/lower_bound/upper_bound",
                    value=(self.power, self.lower_bound, self.upper_bound),
                    constraint="lower_bound <= power <= upper_bound",
                    route="Record power and its bounds from the same counts.",
                )
        return self


class UnitCycleModelMdeResult(_AvailabilityResult):
    """Selection-safe certification over an explicitly declared effect grid.

    The grid is normalized to immutable binary64 values and ordered from the
    procedure null in the favorable direction.  All candidates use one
    experiment stream; the per-effect error allocation protects selection.
    """

    power_kind: Literal["model_mc_power"] = "model_mc_power"
    law: UnitCycleJointLaw
    procedure: ContrastDecisionProcedure
    n: int = Field(ge=1, strict=True)
    target_power: float = Field(gt=0, lt=1, allow_inf_nan=False)
    plausible_effect: float | None = Field(allow_inf_nan=False)
    feasible_effect: float | None = Field(allow_inf_nan=False)
    effect_grid: tuple[float, ...] = Field(min_length=1)
    mc_error_per_effect: float = Field(gt=0, lt=1, allow_inf_nan=False)
    plausible_unavailable_reason: Literal["grid_exhausted"] | None = None
    feasible_unavailable_reason: Literal["grid_exhausted"] | None = None
    power_at_feasible: UnitCycleModelPowerResult | None
    attempted: int = Field(ge=1, strict=True)
    admitted: int = Field(ge=0, strict=True)
    failed: int = Field(ge=0, strict=True)
    failed_refused: int = Field(ge=0, strict=True)
    diagnostic_effect: float = Field(allow_inf_nan=False)
    known_runtime_refused: int = Field(ge=0, strict=True)
    availability_uncertified: int = Field(ge=0, strict=True)
    sampling_failed: int = Field(ge=0, strict=True)
    failures: tuple[UnitCycleFailureCount, ...] = ()
    seed: int = Field(ge=0, strict=True)
    rng: Literal[
        "numpy.PCG64/SeedSequence(seed,replicate)/unit-prefix-v1",
        "numpy.PCG64/SeedSequence(seed,replicate)/unit-prefix-v2",
    ] = "numpy.PCG64/SeedSequence(seed,replicate)/unit-prefix-v1"
    numpy_version: str = np.__version__
    sampler: Literal["finite_type_reuse_exact_micro_v1", "finite_type_reuse_exact_micro_v2"] = (
        "finite_type_reuse_exact_micro_v1"
    )
    mc_method: Literal["simultaneous_effect_grid_bernoulli_hoeffding"] = (
        "simultaneous_effect_grid_bernoulli_hoeffding"
    )
    mc_error: float = Field(gt=0, lt=1, allow_inf_nan=False)
    cdf_bands: int = Field(ge=2, strict=True)
    cdf_terms_per_bound: int = Field(ge=1, strict=True)
    cdf_error_each: float = Field(gt=0, lt=1, allow_inf_nan=False)
    cdf_radius: float = Field(ge=0, le=1, allow_inf_nan=False)
    search_status: Literal["certified_grid_effect", "grid_exhausted"]
    search_domain: Literal["declared_effect_grid"] = "declared_effect_grid"

    @model_validator(mode="after")
    def _bracket(self) -> UnitCycleModelMdeResult:
        _validate_sampling_protocol(self.rng, self.sampler)
        if self.diagnostic_effect != self.procedure.null_abs:
            _invalid(
                "result",
                "MDE top-level diagnostics must describe the procedure null",
                field="diagnostic_effect/procedure.null_abs",
                value=(self.diagnostic_effect, self.procedure.null_abs),
                constraint="diagnostic_effect == procedure.null_abs",
                route="Set diagnostic_effect to procedure.null_abs.",
            )
        repeated = _first_repeat(self.effect_grid)
        if repeated is not None:
            _invalid(
                "result",
                "effect grid must contain unique binary64 effects",
                field=f"effect_grid[{repeated}]",
                value=self.effect_grid[repeated],
                constraint="effect_grid entries are distinct binary64 values",
                route="Remove the repeated effect from effect_grid.",
            )
        sign = _sign(self.procedure)
        nonfinite = next((i for i, x in enumerate(self.effect_grid) if not math.isfinite(x)), None)
        if nonfinite is not None:
            _invalid(
                "result",
                "effect grid must contain finite effects",
                field=f"effect_grid[{nonfinite}]",
                value=self.effect_grid[nonfinite],
                constraint="every effect_grid entry is finite",
                route="Replace the non-finite effect with a finite binary64 effect.",
            )
        misordered = next(
            (
                i
                for i in range(len(self.effect_grid) - 1)
                if sign * self.effect_grid[i] >= sign * self.effect_grid[i + 1]
            ),
            None,
        )
        if misordered is not None:
            _invalid(
                "result",
                "effect grid must be strictly ordered from the null",
                field=f"effect_grid[{misordered}]/effect_grid[{misordered + 1}]",
                value=(self.effect_grid[misordered], self.effect_grid[misordered + 1]),
                constraint=(
                    "effect_grid moves strictly away from the null in the favorable "
                    f"direction ({'increasing' if sign > 0 else 'decreasing'})"
                ),
                route="Sort effect_grid strictly away from procedure.null_abs.",
            )
        unfavorable = next(
            (
                i
                for i, value in enumerate(self.effect_grid)
                if sign * value < sign * self.procedure.null_abs
            ),
            None,
        )
        if unfavorable is not None:
            _invalid(
                "result",
                "effect grid must stay in the favorable direction from the null",
                field=f"effect_grid[{unfavorable}]/procedure.null_abs",
                value=(self.effect_grid[unfavorable], self.procedure.null_abs),
                constraint=(
                    "every effect lies at or beyond procedure.null_abs in the favorable "
                    f"direction ({'increasing' if sign > 0 else 'decreasing'})"
                ),
                route="Remove effects on the unfavorable side of procedure.null_abs.",
            )
        if Fraction(self.mc_error_per_effect) * len(self.effect_grid) > Fraction(self.mc_error):
            _invalid(
                "result",
                "grid allocations exceed the Monte Carlo error budget",
                field="mc_error_per_effect/len(effect_grid)/mc_error",
                value=(self.mc_error_per_effect, len(self.effect_grid), self.mc_error),
                constraint="mc_error_per_effect * len(effect_grid) <= mc_error in exact arithmetic",
                route="Allocate at most mc_error / len(effect_grid) to each effect.",
            )
        each, radius = _fixed_band(self.attempted, self.mc_error_per_effect)
        if self.cdf_error_each != each:
            _invalid(
                "result",
                "grid error metadata must match fixed-query allocation",
                field="cdf_error_each/mc_error_per_effect",
                value=(self.cdf_error_each, self.mc_error_per_effect),
                constraint=(
                    f"cdf_error_each == {each!r}, half of mc_error_per_effect rounded down"
                ),
                route="Record half of mc_error_per_effect, rounded down, as cdf_error_each.",
            )
        if self.cdf_radius != radius or self.cdf_terms_per_bound != 1 or self.cdf_bands != 2:
            _invalid(
                "result",
                "grid bounds must match fixed-query Hoeffding metadata",
                field="cdf_radius/cdf_terms_per_bound/cdf_bands",
                value=(self.cdf_radius, self.cdf_terms_per_bound, self.cdf_bands),
                constraint=(
                    f"cdf_radius == {radius!r} for the attempted draws, "
                    "cdf_terms_per_bound == 1 and cdf_bands == 2"
                ),
                route="Record the fixed-query Hoeffding radius and band counts.",
            )
        if self.failed - self.failed_refused != (
            self.known_runtime_refused + self.availability_uncertified + self.sampling_failed
        ):
            _invalid(
                "result",
                "MDE failure categories must partition admitted failures",
                field=(
                    "failed/failed_refused/known_runtime_refused/"
                    "availability_uncertified/sampling_failed"
                ),
                value=(
                    self.failed,
                    self.failed_refused,
                    self.known_runtime_refused,
                    self.availability_uncertified,
                    self.sampling_failed,
                ),
                constraint=(
                    "failed - failed_refused == "
                    "known_runtime_refused + availability_uncertified + sampling_failed"
                ),
                route="Assign every admitted numerical failure to exactly one failure category.",
            )
        for name in ("plausible", "feasible"):
            _require_reason(
                self,
                f"{name}_effect",
                f"{name}_unavailable_reason",
                route=(
                    f"Set {name}_unavailable_reason='grid_exhausted' exactly when "
                    f"{name}_effect is null."
                ),
            )
            effect = getattr(self, f"{name}_effect")
            if effect is not None and effect not in self.effect_grid:
                _invalid(
                    "result",
                    f"{name} effect must belong to the declared grid",
                    field=f"{name}_effect",
                    value=effect,
                    constraint=f"{name}_effect is an entry of effect_grid",
                    route=f"Choose {name}_effect from effect_grid.",
                )
        if self.feasible_effect is not None and (
            self.plausible_effect is None
            or sign * self.plausible_effect > sign * self.feasible_effect
        ):
            _invalid(
                "result",
                "a certified effect requires an earlier or equal plausible effect",
                field="plausible_effect/feasible_effect",
                value=(self.plausible_effect, self.feasible_effect),
                constraint=(
                    "plausible_effect is set and no farther from the null than feasible_effect"
                ),
                route=(
                    "Report the first grid effect whose upper bound reaches target_power "
                    "as plausible_effect."
                ),
            )
        if (self.feasible_effect is None) != (self.power_at_feasible is None):
            _invalid(
                "result",
                "feasible effect requires its power certificate",
                field="feasible_effect/power_at_feasible",
                value=(self.feasible_effect, self.power_at_feasible),
                constraint="power_at_feasible is present exactly when feasible_effect is set",
                route="Attach the power certificate exactly when a feasible effect is reported.",
            )
        certificate = self.power_at_feasible
        if certificate is not None:
            mismatched, values = _mismatch(
                ("power_at_feasible.law", certificate.law, "law", self.law),
                ("power_at_feasible.procedure", certificate.procedure, "procedure", self.procedure),
                ("power_at_feasible.n", certificate.n, "n", self.n),
                (
                    "power_at_feasible.effect_delta",
                    certificate.effect_delta,
                    "feasible_effect",
                    self.feasible_effect,
                ),
                ("power_at_feasible.seed", certificate.seed, "seed", self.seed),
                ("power_at_feasible.rng", certificate.rng, "rng", self.rng),
                ("power_at_feasible.sampler", certificate.sampler, "sampler", self.sampler),
                (
                    "power_at_feasible.numpy_version",
                    certificate.numpy_version,
                    "numpy_version",
                    self.numpy_version,
                ),
                ("power_at_feasible.attempted", certificate.attempted, "attempted", self.attempted),
            )
            if mismatched:
                _invalid(
                    "result",
                    "grid certificate must retain its model and MC stream",
                    field=mismatched,
                    value=values,
                    constraint=(
                        "each power_at_feasible field equals the result field paired with it"
                    ),
                    route=(
                        "Attach the certificate computed from this result's law, procedure, n, "
                        "feasible_effect and Monte Carlo stream."
                    ),
                )
            if certificate.lower_bound < self.target_power:
                _invalid(
                    "result",
                    "feasible effect must meet target with its lower bound",
                    field="power_at_feasible.lower_bound/target_power",
                    value=(certificate.lower_bound, self.target_power),
                    constraint="power_at_feasible.lower_bound >= target_power",
                    route=(
                        "Report as feasible only an effect whose lower bound reaches target_power."
                    ),
                )
            mismatched, values = _mismatch(
                (
                    "power_at_feasible.mc_error",
                    certificate.mc_error,
                    "mc_error_per_effect",
                    self.mc_error_per_effect,
                ),
                (
                    "power_at_feasible.cdf_error_each",
                    certificate.cdf_error_each,
                    "cdf_error_each",
                    self.cdf_error_each,
                ),
                (
                    "power_at_feasible.cdf_radius",
                    certificate.cdf_radius,
                    "cdf_radius",
                    self.cdf_radius,
                ),
                (
                    "power_at_feasible.cdf_bands",
                    certificate.cdf_bands,
                    "cdf_bands",
                    self.cdf_bands,
                ),
                (
                    "power_at_feasible.cdf_terms_per_bound",
                    certificate.cdf_terms_per_bound,
                    "cdf_terms_per_bound",
                    self.cdf_terms_per_bound,
                ),
                (
                    "power_at_feasible.mc_method",
                    certificate.mc_method,
                    None,
                    "fixed_query_bernoulli_hoeffding",
                ),
            )
            if mismatched:
                _invalid(
                    "result",
                    "grid certificate must retain fixed-query bound metadata",
                    field=mismatched,
                    value=values,
                    constraint=(
                        "each power_at_feasible field equals the result field paired with it, "
                        "and power_at_feasible.mc_method == 'fixed_query_bernoulli_hoeffding'"
                    ),
                    route=("Attach the fixed-query certificate computed with mc_error_per_effect."),
                )
        if self.search_status != (
            "certified_grid_effect" if self.feasible_effect is not None else "grid_exhausted"
        ):
            _invalid(
                "result",
                "grid status must match its certificate",
                field="search_status/feasible_effect",
                value=(self.search_status, self.feasible_effect),
                constraint=(
                    "search_status == 'certified_grid_effect' when feasible_effect is set, "
                    "else 'grid_exhausted'"
                ),
                route="Derive search_status from whether a feasible effect was certified.",
            )
        if not 0 <= self.failed - self.failed_refused <= self.admitted <= self.attempted:
            _invalid(
                "result",
                "invalid MDE counts",
                field="failed/failed_refused/admitted/attempted",
                value=(self.failed, self.failed_refused, self.admitted, self.attempted),
                constraint="0 <= failed - failed_refused <= admitted <= attempted",
                route=(
                    "Count unrefused failures only among admitted draws, and admissions "
                    "only among attempted draws."
                ),
            )
        if not 0 <= self.failed_refused <= min(self.failed, self.attempted - self.admitted):
            _invalid(
                "result",
                "invalid refused numerical-failure count",
                field="failed_refused/failed/attempted/admitted",
                value=(self.failed_refused, self.failed, self.attempted, self.admitted),
                constraint="0 <= failed_refused <= min(failed, attempted - admitted)",
                route="Count as refused failures only failed draws that admission refused.",
            )
        if sum(x.count for x in self.failures) != self.failed:
            _invalid(
                "result",
                "MDE failures require complete reasons",
                field="sum(failures.count)/failed",
                value=(sum(x.count for x in self.failures), self.failed),
                constraint="sum(failures.count) == failed",
                route="Record one failure reason count for every failed draw.",
            )
        return self


class UnitCycleModelRequiredUnitsResult(_AvailabilityResult):
    """Every N from 1 through max_n is checked; no monotonicity is assumed."""

    power_kind: Literal["model_mc_power"] = "model_mc_power"
    law: UnitCycleJointLaw
    procedure: ContrastDecisionProcedure
    effect_delta: float = Field(allow_inf_nan=False)
    target_power: float = Field(gt=0, lt=1, allow_inf_nan=False)
    max_n: int = Field(ge=1, strict=True)
    plausible_n: int | None = Field(ge=1, strict=True)
    feasible_n: int | None = Field(ge=1, strict=True)
    plausible_unavailable_reason: Literal["search_limit"] | None = None
    feasible_unavailable_reason: Literal["search_limit"] | None = None
    power_at_feasible: UnitCycleModelPowerResult | None
    checked_n: int = Field(ge=1, strict=True)
    unavailable_n: int = Field(ge=0, strict=True)
    unavailable_reasons: tuple[UnitCycleFailureCount, ...] = ()
    repetitions: int = Field(ge=1, strict=True)
    seed: int = Field(ge=0, strict=True)
    rng: Literal[
        "numpy.PCG64/SeedSequence(seed,replicate)/unit-prefix-v1",
        "numpy.PCG64/SeedSequence(seed,replicate)/unit-prefix-v2",
    ] = "numpy.PCG64/SeedSequence(seed,replicate)/unit-prefix-v1"
    numpy_version: str = np.__version__
    sampler: Literal["finite_type_reuse_exact_micro_v1", "finite_type_reuse_exact_micro_v2"] = (
        "finite_type_reuse_exact_micro_v1"
    )
    mc_method: Literal["simultaneous_n_bernoulli_hoeffding"] = "simultaneous_n_bernoulli_hoeffding"
    mc_error: float = Field(gt=0, lt=1, allow_inf_nan=False)
    mc_error_per_n: float = Field(gt=0, lt=1, allow_inf_nan=False)
    cdf_bands: int = Field(ge=2, le=2, strict=True)
    cdf_terms_per_bound: int = Field(ge=1, le=1, strict=True)
    cdf_error_each: float = Field(gt=0, lt=1, allow_inf_nan=False)
    cdf_radius: float = Field(ge=0, le=1, allow_inf_nan=False)
    attempted_across_n: int = Field(ge=0, strict=True)
    admitted_across_n: int = Field(ge=0, strict=True)
    failed_across_n: int = Field(ge=0, strict=True)
    failed_refused_across_n: int = Field(ge=0, strict=True)
    known_runtime_refused_across_n: int = Field(ge=0, strict=True)
    availability_uncertified_across_n: int = Field(ge=0, strict=True)
    sampling_failed_across_n: int = Field(ge=0, strict=True)
    failures_across_n: tuple[UnitCycleFailureCount, ...] = ()
    search_status: Literal["bracketed", "search_limit", "numerical_incomplete"]

    @model_validator(mode="after")
    def _bracket(self) -> UnitCycleModelRequiredUnitsResult:
        _validate_sampling_protocol(self.rng, self.sampler)
        each, radius = _fixed_band(self.repetitions, self.mc_error_per_n)
        if self.cdf_error_each != each or self.cdf_radius != radius:
            _invalid(
                "result",
                "Required-N bounds must match their per-size error budget",
                field="cdf_error_each/cdf_radius",
                value=(self.cdf_error_each, self.cdf_radius),
                constraint=(
                    f"cdf_error_each == {each!r} and cdf_radius == {radius!r} for "
                    f"mc_error_per_n={self.mc_error_per_n!r} over each size's repetitions"
                ),
                route="Record the fixed-query allocation and radius of mc_error_per_n.",
            )
        if Fraction(self.mc_error_per_n) * self.max_n > Fraction(self.mc_error) or Fraction(
            self.cdf_error_each
        ) * self.cdf_bands > Fraction(self.mc_error_per_n):
            _invalid(
                "result",
                "required-N allocations exceed the Monte Carlo error budget",
                field="mc_error_per_n/max_n/mc_error/cdf_error_each/cdf_bands",
                value=(
                    self.mc_error_per_n,
                    self.max_n,
                    self.mc_error,
                    self.cdf_error_each,
                    self.cdf_bands,
                ),
                constraint=(
                    "mc_error_per_n * max_n <= mc_error and "
                    "cdf_error_each * cdf_bands <= mc_error_per_n in exact arithmetic"
                ),
                route=(
                    "Allocate at most mc_error / max_n to each N and at most mc_error_per_n "
                    "across its bands."
                ),
            )
        numerical = self.failed_across_n > 0 or any(
            item.reason != "unit_cycle.error_budget_exhausted" for item in self.unavailable_reasons
        )
        expected_status = (
            "numerical_incomplete"
            if numerical
            else "bracketed"
            if self.feasible_n is not None
            else "search_limit"
        )
        if self.search_status != expected_status:
            _invalid(
                "result",
                "required-N status must match its certificate and failures",
                field="search_status/feasible_n/failed_across_n/unavailable_reasons.reason",
                value=(
                    self.search_status,
                    self.feasible_n,
                    self.failed_across_n,
                    tuple(item.reason for item in self.unavailable_reasons),
                ),
                constraint=(
                    f"search_status == {expected_status!r}: 'numerical_incomplete' after "
                    "numerical failures or non-budget unavailable N, else 'bracketed' with a "
                    "feasible_n, else 'search_limit'"
                ),
                route="Derive search_status from the failures, unavailable reasons and feasible_n.",
            )
        if self.unavailable_n > self.checked_n:
            _invalid(
                "result",
                "unavailable N cannot exceed checked N",
                field="unavailable_n/checked_n",
                value=(self.unavailable_n, self.checked_n),
                constraint="unavailable_n <= checked_n",
                route="Count unavailable sizes only among the checked sizes.",
            )
        if self.checked_n != self.max_n:
            _invalid(
                "result",
                "required-N search must check every N from 1 through max_n",
                field="checked_n/max_n",
                value=(self.checked_n, self.max_n),
                constraint="checked_n == max_n",
                route="Check every N from 1 through max_n.",
            )
        for name in ("plausible", "feasible"):
            value = getattr(self, f"{name}_n")
            _require_reason(
                self,
                f"{name}_n",
                f"{name}_unavailable_reason",
                route=(
                    f"Set {name}_unavailable_reason='search_limit' exactly when {name}_n is null."
                ),
            )
            if value is not None and value > self.max_n:
                _invalid(
                    "result",
                    "N must lie within the search domain",
                    field=f"{name}_n/max_n",
                    value=(value, self.max_n),
                    constraint=f"{name}_n <= max_n",
                    route=f"Choose {name}_n from the searched sizes 1 through max_n.",
                )
        if self.feasible_n is not None and (
            self.plausible_n is None or self.plausible_n > self.feasible_n
        ):
            _invalid(
                "result",
                "plausible N cannot exceed feasible N",
                field="plausible_n/feasible_n",
                value=(self.plausible_n, self.feasible_n),
                constraint="plausible_n is set and plausible_n <= feasible_n",
                route="Report the first N whose upper bound reaches target_power as plausible_n.",
            )
        if (self.feasible_n is None) != (self.power_at_feasible is None):
            _invalid(
                "result",
                "feasible N requires its power certificate",
                field="feasible_n/power_at_feasible",
                value=(self.feasible_n, self.power_at_feasible),
                constraint="power_at_feasible is present exactly when feasible_n is set",
                route="Attach the power certificate exactly when a feasible N is reported.",
            )
        certificate = self.power_at_feasible
        if certificate is not None and (
            certificate.n != self.feasible_n or certificate.lower_bound < self.target_power
        ):
            _invalid(
                "result",
                "feasible N certificate must meet the target",
                field="power_at_feasible.n/feasible_n/power_at_feasible.lower_bound/target_power",
                value=(certificate.n, self.feasible_n, certificate.lower_bound, self.target_power),
                constraint=(
                    "power_at_feasible.n == feasible_n and "
                    "power_at_feasible.lower_bound >= target_power"
                ),
                route=(
                    "Attach the certificate of feasible_n, whose lower bound reaches target_power."
                ),
            )
        if certificate is not None:
            mismatched, values = _mismatch(
                ("power_at_feasible.law", certificate.law, "law", self.law),
                ("power_at_feasible.procedure", certificate.procedure, "procedure", self.procedure),
                (
                    "power_at_feasible.effect_delta",
                    certificate.effect_delta,
                    "effect_delta",
                    self.effect_delta,
                ),
                ("power_at_feasible.seed", certificate.seed, "seed", self.seed),
                ("power_at_feasible.rng", certificate.rng, "rng", self.rng),
                ("power_at_feasible.sampler", certificate.sampler, "sampler", self.sampler),
                (
                    "power_at_feasible.numpy_version",
                    certificate.numpy_version,
                    "numpy_version",
                    self.numpy_version,
                ),
                (
                    "power_at_feasible.attempted",
                    certificate.attempted,
                    "repetitions",
                    self.repetitions,
                ),
                (
                    "power_at_feasible.mc_error",
                    certificate.mc_error,
                    "mc_error_per_n",
                    self.mc_error_per_n,
                ),
                (
                    "power_at_feasible.mc_method",
                    certificate.mc_method,
                    None,
                    "fixed_query_bernoulli_hoeffding",
                ),
                ("power_at_feasible.cdf_bands", certificate.cdf_bands, "cdf_bands", self.cdf_bands),
                (
                    "power_at_feasible.cdf_terms_per_bound",
                    certificate.cdf_terms_per_bound,
                    "cdf_terms_per_bound",
                    self.cdf_terms_per_bound,
                ),
                (
                    "power_at_feasible.cdf_error_each",
                    certificate.cdf_error_each,
                    "cdf_error_each",
                    self.cdf_error_each,
                ),
                (
                    "power_at_feasible.cdf_radius",
                    certificate.cdf_radius,
                    "cdf_radius",
                    self.cdf_radius,
                ),
            )
            if mismatched:
                _invalid(
                    "result",
                    "required-N certificate must retain its model and MC stream",
                    field=mismatched,
                    value=values,
                    constraint=(
                        "each power_at_feasible field equals the result field paired with it, "
                        "and power_at_feasible.mc_method == 'fixed_query_bernoulli_hoeffding'"
                    ),
                    route=(
                        "Attach the fixed-query certificate computed from this result's law, "
                        "procedure, effect_delta, Monte Carlo stream and mc_error_per_n."
                    ),
                )
        if sum(x.count for x in self.unavailable_reasons) != self.unavailable_n:
            _invalid(
                "result",
                "unavailable N counts require complete reasons",
                field="sum(unavailable_reasons.count)/unavailable_n",
                value=(sum(x.count for x in self.unavailable_reasons), self.unavailable_n),
                constraint="sum(unavailable_reasons.count) == unavailable_n",
                route="Record one unavailable reason count for every unavailable N.",
            )
        if self.attempted_across_n != (self.checked_n - self.unavailable_n) * self.repetitions:
            _invalid(
                "result",
                "attempt counts must include every evaluated N",
                field="attempted_across_n/checked_n/unavailable_n/repetitions",
                value=(
                    self.attempted_across_n,
                    self.checked_n,
                    self.unavailable_n,
                    self.repetitions,
                ),
                constraint="attempted_across_n == (checked_n - unavailable_n) * repetitions",
                route="Count repetitions attempted draws for every evaluated N.",
            )
        if not 0 <= self.failed_across_n - self.failed_refused_across_n <= self.admitted_across_n:
            _invalid(
                "result",
                "invalid admitted failure count across N",
                field="failed_across_n/failed_refused_across_n/admitted_across_n",
                value=(
                    self.failed_across_n,
                    self.failed_refused_across_n,
                    self.admitted_across_n,
                ),
                constraint="0 <= failed_across_n - failed_refused_across_n <= admitted_across_n",
                route="Count unrefused numerical failures only among admitted draws.",
            )
        if (
            not 0
            <= self.failed_refused_across_n
            <= min(
                self.failed_across_n,
                self.attempted_across_n - self.admitted_across_n,
            )
        ):
            _invalid(
                "result",
                "invalid refused failure count across N",
                field=(
                    "failed_refused_across_n/failed_across_n/attempted_across_n/admitted_across_n"
                ),
                value=(
                    self.failed_refused_across_n,
                    self.failed_across_n,
                    self.attempted_across_n,
                    self.admitted_across_n,
                ),
                constraint=(
                    "0 <= failed_refused_across_n <= "
                    "min(failed_across_n, attempted_across_n - admitted_across_n)"
                ),
                route="Count as refused failures only failed draws that admission refused.",
            )
        if sum(x.count for x in self.failures_across_n) != self.failed_across_n:
            _invalid(
                "result",
                "failures across N require complete reasons",
                field="sum(failures_across_n.count)/failed_across_n",
                value=(sum(x.count for x in self.failures_across_n), self.failed_across_n),
                constraint="sum(failures_across_n.count) == failed_across_n",
                route="Record one failure reason count for every failed draw across N.",
            )
        if self.failed_across_n - self.failed_refused_across_n != (
            self.known_runtime_refused_across_n
            + self.availability_uncertified_across_n
            + self.sampling_failed_across_n
        ):
            _invalid(
                "result",
                "required-N failure categories must partition admitted failures",
                field=(
                    "failed_across_n/failed_refused_across_n/known_runtime_refused_across_n/"
                    "availability_uncertified_across_n/sampling_failed_across_n"
                ),
                value=(
                    self.failed_across_n,
                    self.failed_refused_across_n,
                    self.known_runtime_refused_across_n,
                    self.availability_uncertified_across_n,
                    self.sampling_failed_across_n,
                ),
                constraint=(
                    "failed_across_n - failed_refused_across_n == known_runtime_refused_across_n "
                    "+ availability_uncertified_across_n + sampling_failed_across_n"
                ),
                route="Assign every admitted numerical failure to exactly one failure category.",
            )
        return self


@dataclass(frozen=True)
class _Moments:
    mean: Fraction
    var_a: Fraction
    var_g: Fraction
    cov: Fraction
    residual: Fraction


def _law(law: UnitCycleJointLaw) -> UnitCycleJointLaw:
    if not isinstance(law, UnitCycleJointLaw):
        _invalid(
            "law",
            "law must be a UnitCycleJointLaw",
            field="law",
            value=law,
            constraint="instance of UnitCycleJointLaw",
            route="Pass a UnitCycleJointLaw instance.",
        )
    try:
        return UnitCycleJointLaw.model_validate(law)
    except InvalidRequestError:
        raise
    except ValidationError as exc:
        _invalid_model("law", "law", exc)


def _moments(law: UnitCycleJointLaw) -> _Moments:
    p = Fraction(law.assignment.sequence.probability_ct)
    q, cycles = 1 - p, len(law.types[0].cycles)
    wc, wt = 1 / (2 * p), 1 / (2 * q)
    total = sum((Fraction(t.weight) for t in law.types), Fraction())
    means = tuple(
        sum((Fraction(c.ct_mean) + Fraction(c.tc_mean) for c in t.cycles), Fraction())
        / (2 * cycles)
        for t in law.types
    )
    mean = sum(
        (Fraction(t.weight) * m / total for t, m in zip(law.types, means, strict=True)),
        Fraction(),
    )
    va = cov = residual = Fraction()
    reuse = Fraction(law.reuse_probability)
    for t, tm in zip(law.types, means, strict=True):
        order_a = order_r = order_cov = independent_noise = noise_order = noise_mean = Fraction()
        for c in t.cycles:
            ac, at = wc * Fraction(c.ct_mean), wt * Fraction(c.tc_mean)
            lc, lt = wc * Fraction(c.ct_noise_load), wt * Fraction(c.tc_noise_load)
            order_a += p * q * (ac - at) ** 2
            order_r += (
                p * q * (wc * (Fraction(c.ct_mean) - mean) - wt * (Fraction(c.tc_mean) - mean)) ** 2
            )
            order_cov += p * q * (ac - at) * (wc - wt)
            independent_noise += p * lc**2 / c.ct_innovation_count
            independent_noise += q * lt**2 / c.tc_innovation_count
            noise_order += p * q * (lc - lt) ** 2
            noise_mean += p * lc + q * lt
        noise = (
            (1 - reuse) * independent_noise + reuse * (noise_order + noise_mean**2)
        ) / cycles**2
        weight = Fraction(t.weight) / total
        va += weight * ((tm - mean) ** 2 + order_a / cycles**2 + noise)
        residual += weight * ((tm - mean) ** 2 + order_r / cycles**2 + noise)
        cov += weight * order_cov / cycles**2
    return _Moments(mean, va, p * q * (wc - wt) ** 2 / cycles, cov, residual)


def _float(value: Fraction, *, upward: bool = False) -> float | None:
    try:
        result = float(value)
    except OverflowError:
        return None
    if upward and Fraction(result) < value:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result) or (value != 0 and result == 0):
        return None
    return result


def _down(value: Fraction) -> float:
    result = float(value)
    if Fraction(result) > value:
        result = math.nextafter(result, -math.inf)
    return result


def _sqrt(value: Fraction) -> float | None:
    with localcontext() as ctx:
        ctx.prec = 100
        result = float((Decimal(value.numerator) / Decimal(value.denominator)).sqrt())
    return result if math.isfinite(result) and (result != 0 or value == 0) else None


def unit_cycle_law_moments(law: UnitCycleJointLaw) -> UnitCycleLawMoments:
    """Exact rational centered moments of every positive-weight type and cycle.

    Noise uses its known standardized variance, including the reuse cross terms.
    Only the final public floats are rounded; the residual bound rounds outward.
    """
    law = _law(law)
    m = _moments(law)
    mean, sa, sg = _float(m.mean), _sqrt(m.var_a), _sqrt(m.var_g)
    rho = None
    if m.var_a and m.var_g:
        absolute = _sqrt(m.cov**2 / (m.var_a * m.var_g))
        if absolute is not None:
            rho = -absolute if m.cov < 0 else absolute
    residual = _float(m.residual, upward=True)
    return UnitCycleLawMoments(
        law=law,
        reference_effect=mean,
        sd_a=sa,
        sd_g=sg,
        rho=rho,
        residual_variance_upper=residual,
        reference_effect_unavailable_reason="unrepresentable" if mean is None else None,
        sd_a_unavailable_reason="unrepresentable" if sa is None else None,
        sd_g_unavailable_reason="unrepresentable" if sg is None else None,
        rho_unavailable_reason=(
            "zero_variance" if not m.var_a or not m.var_g else "unrepresentable"
        )
        if rho is None
        else None,
        residual_variance_upper_unavailable_reason="unrepresentable" if residual is None else None,
    )


def unit_cycle_variance_envelope(law: UnitCycleJointLaw) -> UnitCycleVarianceEnvelope:
    """Derive an outward residual envelope from the *declared* complete law."""
    law = _law(law)
    residual = _moments(law).residual
    variance = _float(residual, upward=True)
    if variance is None:
        _invalid(
            "numerical",
            "residual variance has no finite representable upper bound",
            field="residual_variance_upper",
            value=f"{Decimal(residual.numerator) / residual.denominator:.17E}",
            constraint="the exact residual variance, rounded upward, is a finite binary64 value",
            route=(
                "Declare smaller means or noise loads, or a probability_ct closer to 0.5, so "
                "the residual variance does not exceed the largest finite binary64 value."
            ),
        )
    return UnitCycleVarianceEnvelope(
        assignment=law.assignment,
        metric=law.metric,
        control_group=law.control_group,
        treatment_group=law.treatment_group,
        response_meaning=law.response_meaning,
        cycles_per_unit=len(law.types[0].cycles),
        residual_variance_upper=variance,
        provenance=law.provenance,
    )


def _integer(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _invalid(
            "input",
            f"{name} must be an integer >= {minimum}",
            field=name,
            value=value,
            constraint=f"integer >= {minimum}, excluding bool",
            route=f"Pass an integer {name} >= {minimum}.",
        )


def _probability(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
        _invalid(
            "input",
            f"{name} must be finite and strictly between zero and one",
            field=name,
            value=value,
            constraint="finite value strictly between 0 and 1",
            route=f"Pass a finite {name} strictly between 0 and 1.",
        )


def _effect(value: float, *, field: str) -> float:
    if isinstance(value, bool):
        _invalid(
            "input",
            f"{field} must be finite",
            field=field,
            value=value,
            constraint="finite real number",
            route=f"Pass a finite real value for {field}.",
        )
    try:
        effect = float(value)
    except (TypeError, ValueError, OverflowError):
        _invalid(
            "input",
            f"{field} must be finite",
            field=field,
            value=value,
            constraint="finite real number",
            route=f"Pass a finite real value for {field}.",
        )
    if not math.isfinite(effect):
        _invalid(
            "input",
            f"{field} must be finite",
            field=field,
            value=value,
            constraint="finite real number",
            route=f"Pass a finite real value for {field}.",
        )
    return effect


def _procedure(procedure: ContrastDecisionProcedure) -> ContrastDecisionProcedure:
    if not isinstance(procedure, ContrastDecisionProcedure):
        _invalid(
            "procedure",
            "procedure must be a ContrastDecisionProcedure",
            field="procedure",
            value=procedure,
            constraint="instance of ContrastDecisionProcedure",
            route="Pass a ContrastDecisionProcedure instance.",
        )
    try:
        return ContrastDecisionProcedure.model_validate(procedure)
    except InvalidRequestError:
        raise
    except ValidationError as exc:
        _invalid_model("procedure", "procedure", exc)


def _match_procedure(
    envelope: UnitCycleVarianceEnvelope,
    procedure: ContrastDecisionProcedure,
) -> None:
    if procedure.metric != envelope.metric:
        _invalid(
            "procedure",
            "procedure must carry the matching variance envelope and metric",
            field="procedure.metric/envelope.metric",
            value=(procedure.metric, envelope.metric),
            constraint="procedure.metric == envelope.metric",
            route="Declare the procedure for the envelope's metric.",
        )
    reference = procedure.reference
    if reference != envelope:
        mismatched: str = ""
        values: tuple[object, ...] = ()
        if isinstance(reference, UnitCycleVarianceEnvelope):
            mismatched, values = _mismatch(
                *(
                    (
                        f"procedure.reference.{name}",
                        getattr(reference, name),
                        f"envelope.{name}",
                        getattr(envelope, name),
                    )
                    for name in UnitCycleVarianceEnvelope.model_fields
                )
            )
        _invalid(
            "procedure",
            "procedure must carry the matching variance envelope and metric",
            field=mismatched or "procedure.reference",
            value=values if mismatched else reference,
            constraint="procedure.reference equals the variance envelope being planned",
            route=(
                "Build the procedure with reference=unit_cycle_variance_envelope(law), or the "
                "envelope passed to unit_cycle_power_lower_bound."
            ),
        )


def _envelope(envelope: UnitCycleVarianceEnvelope) -> UnitCycleVarianceEnvelope:
    if not isinstance(envelope, UnitCycleVarianceEnvelope):
        _invalid(
            "procedure",
            "envelope must be a UnitCycleVarianceEnvelope",
            field="envelope",
            value=envelope,
            constraint="instance of UnitCycleVarianceEnvelope",
            route="Pass a UnitCycleVarianceEnvelope instance.",
        )
    try:
        return UnitCycleVarianceEnvelope.model_validate(envelope)
    except InvalidRequestError:
        raise
    except ValidationError as exc:
        _invalid_model("procedure", "envelope", exc)


def _sign(procedure: ContrastDecisionProcedure) -> int:
    if procedure.alternative == "less":
        return -1
    if procedure.alternative == "two-sided" and procedure.preferred_direction == "decrease":
        return -1
    return 1


def unit_cycle_power_lower_bound(
    envelope: UnitCycleVarianceEnvelope,
    procedure: ContrastDecisionProcedure,
    *,
    n: int,
    effect_delta: float,
) -> UnitCyclePowerLowerBoundResult:
    """Supplementary Cantelli bound; zero is not an unattainability claim."""
    _integer("n", n, 1)
    effect_delta = _effect(effect_delta, field="effect_delta")
    envelope = _envelope(envelope)
    procedure = _procedure(procedure)
    _match_procedure(envelope, procedure)
    cutoff = unit_cycle_envelope_cutoff(
        envelope,
        n=n,
        alpha=procedure.alpha,
        alternative=procedure.alternative,
    )
    p = Fraction(envelope.assignment.sequence.probability_ct)
    distance = _sign(procedure) * (Fraction(effect_delta) - Fraction(procedure.null_abs))
    rejection = _RejectionRule.from_design(
        envelope, procedure, n=n, refusal=cutoff.refusal_probability_upper
    )
    radius = (
        None
        if rejection.squared is None
        else _first_float(_Boundary(Fraction(), Fraction(1)), rejection.squared, strict=False)
    )
    separation = Fraction() if radius is None else distance / (2 * max(p, 1 - p)) - Fraction(radius)
    bound = Fraction()
    reason: (
        Literal[
            "non_favorable_effect", "insufficient_separation", "runtime_availability_not_certified"
        ]
        | None
    ) = None
    if distance <= 0:
        reason = "non_favorable_effect"
    elif separation <= 0:
        reason = "insufficient_separation"
    else:
        bound = max(
            Fraction(),
            separation**2 / (Fraction(envelope.residual_variance_upper) / n + separation**2)
            - Fraction(cutoff.refusal_probability_upper),
        )
        availability_loss = _availability_loss(envelope, procedure, n, effect_delta, cutoff.value)
        if bound > 0 and availability_loss >= bound:
            reason = "runtime_availability_not_certified"
        bound = max(Fraction(), bound - availability_loss)
    return UnitCyclePowerLowerBoundResult(
        envelope=envelope,
        procedure=procedure,
        n=n,
        effect_delta=effect_delta,
        lower_bound=_down(bound),
        cutoff=cutoff.value,
        effective_alpha=cutoff.effective_alpha,
        refusal_probability_upper=cutoff.refusal_probability_upper,
        reason=reason,
    )


def _mc_inputs(repetitions: int, seed: int, mc_error: float, batch_size: int) -> None:
    _integer("repetitions", repetitions, 1)
    _integer("seed", seed, 0)
    _probability("mc_error", mc_error)
    _integer("batch_size", batch_size, 1)


def _allocate(error: float, parts: int) -> float:
    result = _down(Fraction(error) / parts)
    if result == 0:
        _invalid(
            "numerical",
            "Monte Carlo allocation is below binary64 range",
            field="mc_error_allocation",
            value=(error, parts),
            constraint="error / parts, rounded down to binary64, is positive",
            route="Increase mc_error, or allocate it over fewer effects, sizes or bands.",
        )
    return result


def _band(repetitions: int, error: float, bands: int) -> tuple[float, float]:
    """Outward DKW radius for a simultaneous threshold/event sweep.

    ``bands`` includes both lower/upper processes and each endpoint process
    (two for one-sided, four for two-sided). Effect searches use these bands;
    finite size searches instead allocate fixed-query error across sizes.
    """
    each = _allocate(error, bands)
    with localcontext() as ctx:
        ctx.prec = 100
        ctx.rounding = ROUND_CEILING
        log = (Decimal(2) / Decimal.from_float(each)).ln().next_plus()
        radius = (log / (2 * repetitions)).sqrt().next_plus()
        exact_upper = Fraction(radius)
    upper = _float(min(Fraction(1), exact_upper), upward=True)
    if upper is None:
        _invalid(
            "numerical",
            "Monte Carlo radius is not representable",
            field="cdf_radius",
            value=f"{radius:.17E}",
            constraint="the outward DKW radius rounds upward to a finite binary64 value",
            route="Choose repetitions and mc_error whose radius is a finite binary64 value.",
        )
    return each, upper


def _fixed_band(repetitions: int, error: float) -> tuple[float, float]:
    """One-query Bernoulli envelope, with a one-sided Hoeffding margin.

    For either count process, Hoeffding gives
    ``P(p < k/R-r) <= exp(-2 R r**2)`` and the analogous upper tail.
    Allocating ``error/2`` to the lower and upper count processes and
    rounding outward gives a fixed-query bound, not a simultaneous curve.
    """
    each = _allocate(error, 2)
    with localcontext() as ctx:
        ctx.prec = 100
        ctx.rounding = ROUND_CEILING
        log = Decimal(1) / Decimal.from_float(each)
        log = log.ln().next_plus()
        radius = (log / (2 * repetitions)).sqrt().next_plus()
        exact_upper = Fraction(radius)
    upper = _float(min(Fraction(1), exact_upper), upward=True)
    if upper is None:
        _invalid(
            "numerical",
            "Monte Carlo radius is not representable",
            field="cdf_radius",
            value=f"{radius:.17E}",
            constraint="the outward Hoeffding radius rounds upward to a finite binary64 value",
            route="Choose repetitions and mc_error whose radius is a finite binary64 value.",
        )
    return each, upper


def _binary64_band(repetitions: int, error: float) -> tuple[float, float]:
    """Union both Bernoulli bounds over at most 2**64 effect encodings."""
    return _fixed_band(repetitions, _allocate(error, 1 << 64))


def _simultaneous_band(
    repetitions: int, error: float, alternative: str
) -> tuple[
    Literal["simultaneous_dkw_massart", "simultaneous_binary64_hoeffding"],
    int,
    float,
    float,
]:
    """Choose by deterministic error margins, never by observed outcomes."""
    terms = 4 if alternative == "two-sided" else 2
    each, radius = _band(repetitions, error, 2 * terms)
    if _down(Fraction(error) / (1 << 65)) > 0:
        finite_each, finite_radius = _binary64_band(repetitions, error)
        if Fraction(finite_radius) < terms * Fraction(radius):
            return "simultaneous_binary64_hoeffding", 1, finite_each, finite_radius
    return "simultaneous_dkw_massart", terms, each, radius


def _bounds(
    rejected: int, possible: int, total: int, radius: float, terms: int
) -> tuple[float, float]:
    margin = terms * Fraction(radius)
    lo = max(Fraction(), Fraction(rejected, total) - margin)
    hi = min(Fraction(1), Fraction(possible, total) + margin)
    upper = _float(hi, upward=True)
    assert upper is not None
    return _down(lo), upper


def _randbelow(rng: np.random.Generator, limit: int) -> int:
    """Exact discrete probabilities, including arbitrarily small positive weights."""
    bits = (limit - 1).bit_length()
    while True:
        result = 0
        for offset in range(0, bits, 64):
            result |= int(rng.bit_generator.random_raw()) << offset
        result &= (1 << bits) - 1
        if result < limit:
            return result


def _bernoulli(rng: np.random.Generator, probability: Fraction) -> bool:
    return _randbelow(rng, probability.denominator) < probability.numerator


class _NumericalDraw(Exception):
    pass


def _innovation(rng: np.random.Generator, innovation: UnitCycleInnovation) -> float:
    try:
        if innovation.kind == "normal":
            value = float(rng.standard_normal())
        elif innovation.kind == "centered_gamma":
            shape = innovation.shape
            draw = float(rng.gamma(shape))
            if draw == 0 or shape + math.sqrt(shape) == shape:
                raise _NumericalDraw("gamma_sample_resolution")
            value = (draw - shape) / math.sqrt(shape)
        else:
            shape = innovation.shape
            z = float(rng.standard_normal())
            square = shape * shape
            if not math.isfinite(square) or square == 0:
                raise _NumericalDraw("innovation_scale_unrepresentable")
            if square < 1:
                value = math.expm1(shape * z - square / 2) / math.sqrt(math.expm1(square))
            else:
                denominator = math.sqrt(-math.expm1(-square))
                positive = math.exp(shape * z - square)
                center = math.exp(-square / 2)
                if positive == 0 or center == 0:
                    raise _NumericalDraw("innovation_tail_underflow")
                value = (positive - center) / denominator
    except (OverflowError, FloatingPointError, ValueError) as exc:
        raise _NumericalDraw("innovation_unrepresentable") from exc
    if not math.isfinite(value):
        raise _NumericalDraw("innovation_unrepresentable")
    return value


@dataclass(frozen=True, slots=True)
class _Choice:
    """Exact constants of one cycle under one realised order."""

    order: bool
    weight: Fraction
    centered_mean: Fraction
    load: Fraction
    count: int
    weighted_load: Fraction


@dataclass(frozen=True)
class _Sampler:
    law: UnitCycleJointLaw
    mean: Fraction
    weights: tuple[int, ...]
    total_weight: int
    p: Fraction
    reuse: Fraction
    choices: tuple[tuple[tuple[_Choice, _Choice], ...], ...]


def _sampler(law: UnitCycleJointLaw) -> _Sampler:
    weights = tuple(Fraction(t.weight) for t in law.types)
    denominator = math.lcm(*(x.denominator for x in weights))
    integers = tuple(x.numerator * (denominator // x.denominator) for x in weights)
    mean = _moments(law).mean
    p = Fraction(law.assignment.sequence.probability_ct)
    order_weights = (Fraction(1, 2) / (1 - p), Fraction(1, 2) / p)

    def choice(cycle: UnitCycleCycleLaw, *, order: bool) -> _Choice:
        weight = order_weights[order]
        load = Fraction(cycle.ct_noise_load if order else cycle.tc_noise_load)
        centered = Fraction(cycle.ct_mean if order else cycle.tc_mean) - mean
        count = cycle.ct_innovation_count if order else cycle.tc_innovation_count
        return _Choice(order, weight, centered, load, count, weight * load)

    return _Sampler(
        law,
        mean,
        integers,
        sum(integers),
        p,
        Fraction(law.reuse_probability),
        tuple(
            tuple((choice(c, order=False), choice(c, order=True)) for c in t.cycles)
            for t in law.types
        ),
    )


def _unit(
    sampler: _Sampler, rng: np.random.Generator
) -> tuple[Fraction, Fraction, int, str | None]:
    draw = _randbelow(rng, sampler.total_weight)
    index = 0
    while draw >= sampler.weights[index]:
        draw -= sampler.weights[index]
        index += 1
    reuse = _bernoulli(rng, sampler.reuse)
    chosen = [cycle[_bernoulli(rng, sampler.p)] for cycle in sampler.choices[index]]
    effective_reuse_load = sum((choice.weighted_load for choice in chosen), Fraction())
    common, reason = 0.0, None
    if reuse and effective_reuse_load:
        try:
            common = _innovation(rng, sampler.law.innovation)
        except _NumericalDraw as exc:
            reason = str(exc)
    residual, slope = Fraction(), Fraction()
    ct = sum(choice.order for choice in chosen)
    for choice in chosen:
        noise = Fraction(common) if reuse else Fraction()
        if not reuse and choice.load:
            for _ in range(choice.count):
                try:
                    value = _innovation(rng, sampler.law.innovation)
                except _NumericalDraw as exc:
                    reason = reason or str(exc)
                else:
                    noise += Fraction(value) / choice.count
        residual += choice.weight * (choice.centered_mean + choice.load * noise)
        slope += choice.weight
    cycles = len(chosen)
    return residual / cycles, slope / cycles, ct, reason


@dataclass(frozen=True)
class _Prefix:
    """Exact centered affine unit state; retained object count is bounded."""

    n: int
    u: Fraction
    g: Fraction
    ct: int
    reason: str | None
    q_uu: Fraction = Fraction()
    q_ug: Fraction = Fraction()
    q_gg: Fraction = Fraction()


def _prefixes(sampler: _Sampler, *, maximum: int, replicate: int, seed: int) -> Iterator[_Prefix]:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, replicate])))
    u, g, ct, failure = Fraction(), Fraction(), 0, None
    q_uu = q_ug = q_gg = Fraction()
    for n in range(1, maximum + 1):
        a, slope, count, reason = _unit(sampler, rng)
        du, dg = a - u, slope - g
        multiplier = Fraction(n - 1, n)
        q_uu += multiplier * du**2
        q_ug += multiplier * du * dg
        q_gg += multiplier * dg**2
        u += du / n
        g += dg / n
        ct += count
        failure = failure or reason
        yield _Prefix(n, u, g, ct, failure, q_uu, q_ug, q_gg)


def _experiments(
    sampler: _Sampler, *, n: int, repetitions: int, seed: int, batch_size: int
) -> Generator[_Prefix, None, None]:
    for start in range(0, repetitions, batch_size):
        for replicate in range(start, min(start + batch_size, repetitions)):
            for state in _prefixes(
                sampler,
                maximum=n,
                replicate=replicate,
                seed=seed,
            ):
                if state.n == n:
                    yield state


@dataclass(frozen=True)
class _RejectionRule:
    """Squared deployed boundary: zero excludes the null atom; None never rejects.

    Upward rounding gives p < alpha exactly when the rational probability is
    at most alpha's binary64 predecessor. Positive-variance boundaries are closed.
    """

    squared: Fraction | None

    @classmethod
    def from_design(
        cls,
        envelope: UnitCycleVarianceEnvelope,
        procedure: ContrastDecisionProcedure,
        *,
        n: int,
        refusal: float,
    ) -> _RejectionRule:
        variance = Fraction(envelope.residual_variance_upper) / n
        # Admission validation has already established refusal < alpha.
        if not variance:
            return cls(Fraction())
        cap = Fraction(math.nextafter(procedure.alpha, 0.0)) - Fraction(refusal)
        if cap <= 0:
            return cls(None)
        squared = variance / cap
        if procedure.alternative != "two-sided":
            squared *= 1 - cap
        return cls(squared)


def _reject(
    u: Fraction,
    g: Fraction,
    effect: float,
    procedure: ContrastDecisionProcedure,
    rule: _RejectionRule,
) -> bool:
    residual = u + (Fraction(effect) - Fraction(procedure.null_abs)) * g
    if rule.squared is None or residual == 0:
        return False
    favorable = -residual if procedure.alternative == "less" else residual
    if procedure.alternative != "two-sided" and favorable <= 0:
        return False
    return residual**2 >= rule.squared


def _failures(reasons: Counter[str]) -> tuple[UnitCycleFailureCount, ...]:
    return tuple(
        UnitCycleFailureCount(reason=key, count=value)
        for key, value in sorted(reasons.items())
        if value
    )


def _power_result(
    law: UnitCycleJointLaw,
    procedure: ContrastDecisionProcedure,
    *,
    n: int,
    effect: float,
    repetitions: int,
    counts: _Counts,
    seed: int,
    error: float,
    cutoff: UnitCycleEnvelopeCutoff,
    simultaneous: bool = False,
) -> UnitCycleModelPowerResult:
    if simultaneous:
        method, terms, each, radius = _simultaneous_band(
            repetitions,
            error,
            procedure.alternative,
        )
    else:
        each, radius = _fixed_band(repetitions, error)
        terms = 1
        method = "fixed_query_bernoulli_hoeffding"
    bands = 2 * terms
    lo, hi = _bounds(counts.rejected, counts.possible_rejected, repetitions, radius, terms)
    failed = sum(counts.reasons.values())
    unknown = failed - counts.failed_refused
    return UnitCycleModelPowerResult(
        rng="numpy.PCG64/SeedSequence(seed,replicate)/unit-prefix-v2",
        sampler="finite_type_reuse_exact_micro_v2",
        law=law,
        procedure=procedure,
        n=n,
        effect_delta=effect,
        power=None if unknown else counts.rejected / repetitions,
        lower_bound=lo,
        upper_bound=hi,
        power_unavailable_reason="numerical_failures" if unknown else None,
        attempted=repetitions,
        admitted=counts.admitted,
        failed=failed,
        rejected=counts.rejected,
        possible_rejected=counts.possible_rejected,
        known_runtime_refused=counts.known_runtime_refused,
        availability_uncertified=counts.availability_uncertified,
        sampling_failed=counts.sampling_failed,
        failed_refused=counts.failed_refused,
        failures=_failures(counts.reasons),
        seed=seed,
        mc_error=error,
        mc_method=method,
        cdf_bands=bands,
        cdf_terms_per_bound=terms,
        cdf_error_each=each,
        cdf_radius=radius,
        numerical_status="incomplete" if failed else "ok",
        cutoff=cutoff.value,
        effective_alpha=cutoff.effective_alpha,
        refusal_probability_upper=cutoff.refusal_probability_upper,
    )


def unit_cycle_model_power(
    law: UnitCycleJointLaw,
    procedure: ContrastDecisionProcedure,
    *,
    n: int,
    effect_delta: float,
    repetitions: int,
    seed: int,
    mc_error: float,
    batch_size: int = 512,
    simultaneous_effects: bool = False,
) -> UnitCycleModelPowerResult:
    """Research-only complete-law Monte Carlo power oracle.

    Ordinary planning uses the pilot-fitted moment-t path in
    ``increment.power.switchback``.  This entry point remains for independent
    calibration and validation where a complete law is scientifically
    justified; it is not an ordinary-user prerequisite.

    By default the error budget covers one prospectively fixed effect.
    ``simultaneous_effects=True`` covers all binary64 effects at this N and
    law, allowing same-stream adaptive selection. Effects are normalized to
    binary64 before calculation and recorded at that precision.
    Queries across different N or laws require separate error allocation.
    """
    if not isinstance(simultaneous_effects, bool):
        _invalid(
            "input",
            "simultaneous_effects must be a boolean",
            field="simultaneous_effects",
            value=simultaneous_effects,
            constraint="bool",
            route="Pass True or False for simultaneous_effects.",
        )
    _integer("n", n, 1)
    effect_delta = _effect(effect_delta, field="effect_delta")
    _mc_inputs(repetitions, seed, mc_error, batch_size)
    law = _law(law)
    procedure = _procedure(procedure)
    envelope = unit_cycle_variance_envelope(law)
    _match_procedure(envelope, procedure)
    cutoff = unit_cycle_envelope_cutoff(
        envelope,
        n=n,
        alpha=procedure.alpha,
        alternative=procedure.alternative,
    )
    if simultaneous_effects:
        _simultaneous_band(
            repetitions,
            mc_error,
            procedure.alternative,
        )
    else:
        _fixed_band(repetitions, mc_error)
    rule = unit_cycle_admission_rule(
        n * envelope.cycles_per_unit, law.assignment.sequence.probability_ct
    )
    rejection = _RejectionRule.from_design(
        envelope, procedure, n=n, refusal=cutoff.refusal_probability_upper
    )
    counts = _Counts(0, 0, Counter())
    for state in _experiments(
        _sampler(law),
        n=n,
        repetitions=repetitions,
        seed=seed,
        batch_size=batch_size,
    ):
        _count_draw(
            counts,
            _Draw(state, law, procedure.null_abs),
            procedure,
            effect_delta,
            cutoff.value,
            rejection,
            admitted=rule.minimum_ct <= state.ct <= rule.maximum_ct,
        )
    return _power_result(
        law,
        procedure,
        n=n,
        effect=effect_delta,
        repetitions=repetitions,
        counts=counts,
        seed=seed,
        error=mc_error,
        cutoff=cutoff,
        simultaneous=simultaneous_effects,
    )


@dataclass(frozen=True)
class _Boundary:
    a: Fraction
    b: Fraction = Fraction()

    def compare(self, other: _Boundary, h: Fraction = Fraction()) -> int:
        a, b = self.a - other.a, self.b - other.b
        if not b or not h:
            return (a > 0) - (a < 0)
        if not a or (a > 0) == (b > 0):
            return 1 if b > 0 else -1
        difference = a * a - b * b * h
        return ((difference > 0) - (difference < 0)) * (1 if a > 0 else -1)


@dataclass(frozen=True)
class _Interval:
    low: _Boundary
    high: _Boundary
    low_closed: bool = True
    high_closed: bool = True

    def intersect(self, other: _Interval, h: Fraction = Fraction()) -> _Interval | None:
        left = self.low.compare(other.low, h)
        right = self.high.compare(other.high, h)
        low = self.low if left >= 0 else other.low
        high = self.high if right <= 0 else other.high
        lc = (
            self.low_closed and other.low_closed
            if left == 0
            else self.low_closed
            if left > 0
            else other.low_closed
        )
        hc = (
            self.high_closed and other.high_closed
            if right == 0
            else self.high_closed
            if right < 0
            else other.high_closed
        )
        order = low.compare(high, h)
        return _Interval(low, high, lc, hc) if order < 0 or (order == 0 and lc and hc) else None

    def contains(self, value: Fraction, h: Fraction = Fraction()) -> bool:
        point = _Boundary(value)
        left, right = self.low.compare(point, h), self.high.compare(point, h)
        return (left < 0 or (left == 0 and self.low_closed)) and (
            right > 0 or (right == 0 and self.high_closed)
        )


def _window_width(n: int) -> Fraction:
    """Dyadic width with ample room for rounded residual sums and squares."""
    width = Fraction(2) ** ((1023 - n.bit_length() - 8) // 2)
    while 256 * n * width**2 > FINITE_MAX or 16 * n * width > FINITE_MAX:
        width /= 2
    return width


def _first_float(value: _Boundary, h: Fraction, *, strict: bool) -> float | None:
    last = 0x7FEFFFFFFFFFFFFF

    def accepted(ordinal: int) -> bool:
        comparison = value.compare(_Boundary(Fraction(float_from_ordinal(ordinal))), h)
        return comparison < 0 if strict else comparison <= 0

    if not accepted(last):
        return None
    ordinal = bisect_first_true(-last, last, accepted)
    return float_from_ordinal(ordinal)


_FINITE_EFFECTS = _Interval(_Boundary(-FINITE_MAX), _Boundary(FINITE_MAX))


def _availability_loss(
    envelope: UnitCycleVarianceEnvelope,
    procedure: ContrastDecisionProcedure,
    n: int,
    effect: float,
    cutoff: float,
) -> Fraction:
    """Envelope-only upper bound on sufficient-state reporting/summary failure."""
    p = Fraction(envelope.assignment.sequence.probability_ct)
    slopes = (1 / (2 * max(p, 1 - p)), 1 / (2 * min(p, 1 - p)))
    variance = Fraction(envelope.residual_variance_upper)
    delta = Fraction(effect)
    if any(g >= NEAREST_OVERFLOW for g in slopes):
        return Fraction(1)
    if variance == 0:
        try:
            for g in slopes:
                unit_cycle_report(delta * g, g, cutoff, procedure.alternative)
        except InvalidRequestError:
            return Fraction(1)
        report_loss = Fraction()
    else:
        low = -NEAREST_OVERFLOW
        high = NEAREST_OVERFLOW
        for g in slopes:
            (a, _), (b, _) = unit_cycle_reporting_bounds(g, cutoff, procedure.alternative)
            low = max(low, a - delta * g)
            high = min(high, b - delta * g)
        distance = min(-low, high)
        report_loss = min(Fraction(1), variance / n / distance**2) if distance > 0 else Fraction(1)
    if n == 1 or (variance == 0 and p == Fraction(1, 2)):
        summary_loss = Fraction()
    else:
        room = _window_width(n) / 4 - abs(delta) * max(slopes)
        summary_loss = min(Fraction(1), n * variance / room**2) if room > 0 else Fraction(1)
    return min(Fraction(1), report_loss + summary_loss)


def _descriptive_certificate(state: _Prefix, null: float) -> _Interval | None:
    """One rational interval certifying finite scalar moments of rounded units.

    This proves availability, not equality of exact and rounded second moments.
    Outside the interval descriptive construction remains unresolved.
    """
    if state.n == 1 or (state.q_uu == 0 and state.q_gg == 0):
        return _FINITE_EFFECTS.intersect(
            _Interval(
                _Boundary((-NEAREST_OVERFLOW - state.u) / state.g),
                _Boundary((NEAREST_OVERFLOW - state.u) / state.g),
                False,
                False,
            )
        )
    pivot = -state.q_ug / state.q_gg if state.q_gg else Fraction(null)
    pivot = max(-FINITE_MAX, min(FINITE_MAX, pivot))
    mean = state.u + pivot * state.g
    z = Fraction(float(mean)) if -NEAREST_OVERFLOW < mean < NEAREST_OVERFLOW else Fraction()
    width = _window_width(state.n)
    left = max(-FINITE_MAX, z - width / 2)
    right = min(FINITE_MAX, z + width / 2)
    lower, upper = float(left), float(right)
    if Fraction(lower) < left:
        lower = math.nextafter(lower, math.inf)
    if Fraction(upper) > right:
        upper = math.nextafter(upper, -math.inf)
    a = (
        -NEAREST_OVERFLOW
        if lower == -sys.float_info.max
        else (Fraction(math.nextafter(lower, -math.inf)) + Fraction(lower)) / 2
    )
    b = (
        NEAREST_OVERFLOW
        if upper == sys.float_info.max
        else (Fraction(math.nextafter(upper, math.inf)) + Fraction(upper)) / 2
    )
    d = min(z - a, b - z)
    quadratic = 2 * (state.g**2 + state.q_gg)
    linear = 4 * ((state.u - z) * state.g + state.q_ug)
    constant = 2 * ((state.u - z) ** 2 + state.q_uu) - d**2
    vertex = -linear / (2 * quadratic)
    # Accepted floats x satisfy (x - vertex)^2 < radius_squared, one open interval
    # around the vertex, so each end is a monotone search from an approximate root.
    radius_squared = linear**2 / (4 * quadratic**2) - constant / quadratic
    if radius_squared <= 0:
        return None
    accepted = _within_radius(vertex, radius_squared)
    center = _clipped_float(vertex)
    center_rank = float_ordinal(center)
    last_rank = 0x7FEFFFFFFFFFFFFF
    candidate = next(
        (
            rank
            for rank in (center_rank, center_rank - 1, center_rank + 1)
            if -last_rank <= rank <= last_rank and accepted(rank)
        ),
        None,
    )
    if candidate is None:
        return None
    radius = math.sqrt(float(min(radius_squared, FINITE_MAX)))
    first = _extreme_accepted(
        accepted,
        inner=candidate,
        outer=-last_rank,
        guess=float_ordinal(_clipped_float(center - radius)),
    )
    last = _extreme_accepted(
        accepted,
        inner=candidate,
        outer=last_rank,
        guess=float_ordinal(_clipped_float(center + radius)),
    )
    return _Interval(
        _Boundary(Fraction(float_from_ordinal(first))),
        _Boundary(Fraction(float_from_ordinal(last))),
    )


def _clipped_float(value: Fraction | float) -> float:
    return float(max(-FINITE_MAX, min(FINITE_MAX, value)))


def _within_radius(center: Fraction, radius_squared: Fraction) -> Callable[[int], bool]:
    """Exact test of (x - center)^2 < radius_squared on the float of a rank.

    Cross-multiplying by the positive denominators keeps every step in integer
    arithmetic, so no rational normalisation runs per probe.
    """
    a, b = center.numerator, center.denominator
    c, d = radius_squared.numerator, radius_squared.denominator
    bound = c * b * b

    def accepted(rank: int) -> bool:
        xn, xd = float_from_ordinal(rank).as_integer_ratio()
        offset = xn * b - a * xd
        return offset * offset * d < bound * xd * xd

    return accepted


def _extreme_accepted(
    accepted: Callable[[int], bool], *, inner: int, outer: int, guess: int
) -> int:
    """Farthest accepted rank from ``inner`` toward ``outer``.

    ``inner`` is accepted and the accepted ranks form one interval, so the
    predicate is monotone along the way; the search brackets the boundary
    around ``guess`` and then bisects only that bracket.
    """
    direction = 1 if outer >= inner else -1
    limit = abs(outer - inner)

    def at(distance: int) -> int:
        return inner + direction * distance

    start = min(max(direction * (guess - inner), 0), limit)
    step = 1
    if accepted(at(start)):
        good, bad = start, limit + 1
        while good + step <= limit:
            if not accepted(at(good + step)):
                bad = good + step
                break
            good += step
            step *= 2
    else:
        good, bad = 0, start
        while bad - step > 0:
            if accepted(at(bad - step)):
                good = bad - step
                break
            bad -= step
            step *= 2
    while bad - good > 1:
        middle = (good + bad) // 2
        if accepted(at(middle)):
            good = middle
        else:
            bad = middle
    return at(good)


@dataclass
class _Counts:
    admitted: int
    rejected: int
    reasons: Counter[str]
    failed_refused: int = 0
    possible_rejected: int = 0
    known_runtime_refused: int = 0
    availability_uncertified: int = 0
    sampling_failed: int = 0


@dataclass
class _Draw:
    """One replicate's terminal state; checks that ignore the effect run once."""

    state: _Prefix
    law: UnitCycleJointLaw
    null: float

    @cached_property
    def admission_reason(self) -> str | None:
        try:
            unit_cycle_admission_outcome(
                self.state.n * len(self.law.types[0].cycles),
                self.law.assignment.sequence.probability_ct,
                self.state.ct,
            )
        except InvalidRequestError as exc:
            return str(exc.context["reason"])
        return None

    @cached_property
    def certificate(self) -> _Interval | None:
        return _descriptive_certificate(self.state, self.null)


def _count_draw(
    counts: _Counts,
    draw: _Draw,
    procedure: ContrastDecisionProcedure,
    effect: float,
    cutoff: float,
    rejection: _RejectionRule,
    *,
    admitted: bool,
) -> None:
    """Only proved unavailable outcomes leave the upper count."""
    state = draw.state
    if not admitted:
        reason = draw.admission_reason or state.reason
        if reason is not None:
            counts.reasons[reason] += 1
            counts.failed_refused += 1
        return
    counts.admitted += 1
    if draw.admission_reason is not None:
        counts.reasons[draw.admission_reason] += 1
        counts.known_runtime_refused += 1
        return
    if state.reason is not None:
        counts.reasons[state.reason] += 1
        counts.sampling_failed += 1
        counts.possible_rejected += 1
        return
    try:
        unit_cycle_report(
            state.u + Fraction(effect) * state.g, state.g, cutoff, procedure.alternative
        )
    except InvalidRequestError as exc:
        counts.reasons[str(exc.context["reason"])] += 1
        counts.known_runtime_refused += 1
        return
    rejects = int(_reject(state.u, state.g, effect, procedure, rejection))
    counts.possible_rejected += rejects
    certificate = draw.certificate
    if certificate is None or not certificate.contains(Fraction(effect)):
        counts.reasons["descriptive_availability_uncertified"] += 1
        counts.availability_uncertified += 1
    else:
        counts.rejected += rejects


def _normalize_effect_grid(
    effect_grid: Sequence[float], procedure: ContrastDecisionProcedure
) -> tuple[float, ...]:
    try:
        values = tuple(
            _effect(value, field=f"effect_grid[{index}]") for index, value in enumerate(effect_grid)
        )
    except TypeError:
        _invalid(
            "input",
            "effect_grid must be a nonempty finite sequence",
            field="effect_grid",
            value=effect_grid,
            constraint="nonempty finite sequence of real effects",
            route="Pass a finite nonempty effect_grid sequence.",
        )
    if not values:
        _invalid(
            "input",
            "effect_grid must be nonempty",
            field="effect_grid",
            value=values,
            constraint="at least one effect",
            route="Include at least one effect in effect_grid.",
        )
    if len(set(values)) != len(values):
        _invalid(
            "input",
            "effect_grid must contain unique binary64 effects",
            field="effect_grid",
            value=values,
            constraint="unique binary64 effects",
            route="Remove duplicate effects from effect_grid.",
        )
    sign = _sign(procedure)
    null = procedure.null_abs
    if any(sign * (value - null) < 0 for value in values):
        _invalid(
            "input",
            "effect_grid must be favorable effects at or beyond the null",
            field="effect_grid",
            value=values,
            constraint="all effects at or beyond the favorable null direction",
            route="Use effects at or beyond procedure.null_abs in its favorable direction.",
        )
    if any(sign * values[i] >= sign * values[i + 1] for i in range(len(values) - 1)):
        _invalid(
            "input",
            "effect_grid must be strictly ordered from the null",
            field="effect_grid",
            value=values,
            constraint="strict favorable ordering from the null",
            route="Sort effect_grid strictly from the null in the favorable direction.",
        )
    return values


def unit_cycle_model_mde(
    law: UnitCycleJointLaw,
    procedure: ContrastDecisionProcedure,
    *,
    n: int,
    effect_grid: Sequence[float],
    target_power: float,
    repetitions: int,
    seed: int,
    mc_error: float,
    batch_size: int = 512,
) -> UnitCycleModelMdeResult:
    """Research-only complete-law MDE certification over a declared effect grid.

    Ordinary MDE uses the pilot-fitted moment-t solver; this path remains for
    independent model-conditioned calibration and validation.
    """
    _integer("n", n, 1)
    _probability("target_power", target_power)
    _mc_inputs(repetitions, seed, mc_error, batch_size)
    law = _law(law)
    procedure = _procedure(procedure)
    envelope = unit_cycle_variance_envelope(law)
    _match_procedure(envelope, procedure)
    grid = _normalize_effect_grid(effect_grid, procedure)
    per_error = _allocate(mc_error, len(grid))
    each, radius = _fixed_band(repetitions, per_error)
    cutoff = unit_cycle_envelope_cutoff(
        envelope, n=n, alpha=procedure.alpha, alternative=procedure.alternative
    )
    rejection = _RejectionRule.from_design(
        envelope, procedure, n=n, refusal=cutoff.refusal_probability_upper
    )
    rule = unit_cycle_admission_rule(
        n * envelope.cycles_per_unit, law.assignment.sequence.probability_ct
    )
    candidate_counts = [_Counts(0, 0, Counter()) for _ in grid]
    null_counts = _Counts(0, 0, Counter())
    for state in _experiments(
        _sampler(law), n=n, repetitions=repetitions, seed=seed, batch_size=batch_size
    ):
        admitted = rule.minimum_ct <= state.ct <= rule.maximum_ct
        draw = _Draw(state, law, procedure.null_abs)
        _count_draw(
            null_counts,
            draw,
            procedure,
            procedure.null_abs,
            cutoff.value,
            rejection,
            admitted=admitted,
        )
        for candidate, effect in zip(candidate_counts, grid, strict=True):
            _count_draw(
                candidate,
                draw,
                procedure,
                effect,
                cutoff.value,
                rejection,
                admitted=admitted,
            )
    plausible = feasible = None
    for effect, counts in zip(grid, candidate_counts, strict=True):
        lo, hi = _bounds(counts.rejected, counts.possible_rejected, repetitions, radius, 1)
        if plausible is None and hi >= target_power:
            plausible = effect
        if feasible is None and lo >= target_power:
            feasible = effect
    certificate = None
    if feasible is not None:
        selected = candidate_counts[grid.index(feasible)]
        certificate = _power_result(
            law,
            procedure,
            n=n,
            effect=feasible,
            repetitions=repetitions,
            counts=selected,
            seed=seed,
            error=per_error,
            cutoff=cutoff,
        )
    failed = sum(null_counts.reasons.values())
    return UnitCycleModelMdeResult(
        rng="numpy.PCG64/SeedSequence(seed,replicate)/unit-prefix-v2",
        sampler="finite_type_reuse_exact_micro_v2",
        law=law,
        procedure=procedure,
        n=n,
        target_power=target_power,
        plausible_effect=plausible,
        feasible_effect=feasible,
        effect_grid=grid,
        mc_error_per_effect=per_error,
        plausible_unavailable_reason="grid_exhausted" if plausible is None else None,
        feasible_unavailable_reason="grid_exhausted" if feasible is None else None,
        power_at_feasible=certificate,
        attempted=repetitions,
        admitted=null_counts.admitted,
        failed=failed,
        failures=_failures(null_counts.reasons),
        seed=seed,
        failed_refused=null_counts.failed_refused,
        diagnostic_effect=procedure.null_abs,
        known_runtime_refused=null_counts.known_runtime_refused,
        availability_uncertified=null_counts.availability_uncertified,
        sampling_failed=null_counts.sampling_failed,
        mc_error=mc_error,
        mc_method="simultaneous_effect_grid_bernoulli_hoeffding",
        cdf_bands=2,
        cdf_terms_per_bound=1,
        cdf_error_each=each,
        cdf_radius=radius,
        search_status="certified_grid_effect" if feasible is not None else "grid_exhausted",
    )


@contextmanager
def _work_db() -> Iterator[sqlite3.Connection]:
    with TemporaryDirectory(prefix="increment-unit-cycle-") as directory:
        connection = sqlite3.connect(f"{directory}/planning.sqlite")
        try:
            connection.execute("PRAGMA cache_size = -2048")
            connection.execute("PRAGMA temp_store = FILE")
            yield connection
        finally:
            connection.close()


def _prepare_sizes(
    connection: sqlite3.Connection,
    envelope: UnitCycleVarianceEnvelope,
    procedure: ContrastDecisionProcedure,
    max_n: int,
) -> None:
    connection.execute(
        "CREATE TABLE sizes (n INTEGER PRIMARY KEY, cutoff REAL, alpha REAL, refusal REAL, "
        "squared TEXT, "
        "minimum_ct INTEGER, maximum_ct INTEGER, reason TEXT, admitted INTEGER DEFAULT 0, "
        "rejected INTEGER DEFAULT 0, failed INTEGER DEFAULT 0, failed_refused INTEGER DEFAULT 0, "
        "possible INTEGER DEFAULT 0, runtime_refused INTEGER DEFAULT 0, "
        "uncertified INTEGER DEFAULT 0, sampling_failed INTEGER DEFAULT 0)"
    )
    connection.execute(
        "CREATE TABLE failures (n INTEGER, reason TEXT, count INTEGER, PRIMARY KEY(n, reason))"
    )
    for n in range(1, max_n + 1):
        try:
            cutoff = unit_cycle_envelope_cutoff(
                envelope,
                n=n,
                alpha=procedure.alpha,
                alternative=procedure.alternative,
            )
            rule = unit_cycle_admission_rule(
                n * envelope.cycles_per_unit, envelope.assignment.sequence.probability_ct
            )
        except InvalidRequestError as exc:
            connection.execute("INSERT INTO sizes (n, reason) VALUES (?, ?)", (n, exc.code))
        else:
            rejection = _RejectionRule.from_design(
                envelope, procedure, n=n, refusal=cutoff.refusal_probability_upper
            )
            connection.execute(
                "INSERT INTO sizes (n, cutoff, alpha, refusal, squared, minimum_ct, maximum_ct) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    n,
                    cutoff.value,
                    cutoff.effective_alpha,
                    cutoff.refusal_probability_upper,
                    None if rejection.squared is None else str(rejection.squared),
                    rule.minimum_ct,
                    rule.maximum_ct,
                ),
            )
    connection.commit()


def _collect_sizes(
    connection: sqlite3.Connection,
    law: UnitCycleJointLaw,
    procedure: ContrastDecisionProcedure,
    *,
    max_n: int,
    effect: float,
    repetitions: int,
    seed: int,
    batch_size: int,
) -> None:
    sampler = _sampler(law)
    for start in range(0, repetitions, batch_size):
        for replicate in range(start, min(start + batch_size, repetitions)):
            for state in _prefixes(
                sampler,
                maximum=max_n,
                replicate=replicate,
                seed=seed,
            ):
                n = state.n
                cutoff, squared, minimum, maximum = connection.execute(
                    "SELECT cutoff, squared, minimum_ct, maximum_ct FROM sizes WHERE n = ?",
                    (n,),
                ).fetchone()
                if cutoff is None:
                    continue
                rejection = _RejectionRule(None if squared is None else Fraction(squared))
                counts = _Counts(0, 0, Counter())
                _count_draw(
                    counts,
                    _Draw(state, law, procedure.null_abs),
                    procedure,
                    effect,
                    cutoff,
                    rejection,
                    admitted=minimum <= state.ct <= maximum,
                )
                connection.execute(
                    "UPDATE sizes SET admitted=admitted+?, rejected=rejected+?, failed=failed+?, "
                    "failed_refused=failed_refused+?, possible=possible+?, "
                    "runtime_refused=runtime_refused+?, uncertified=uncertified+?, "
                    "sampling_failed=sampling_failed+? WHERE n=?",
                    (
                        counts.admitted,
                        counts.rejected,
                        sum(counts.reasons.values()),
                        counts.failed_refused,
                        counts.possible_rejected,
                        counts.known_runtime_refused,
                        counts.availability_uncertified,
                        counts.sampling_failed,
                        n,
                    ),
                )
                for reason in counts.reasons:
                    connection.execute(
                        "INSERT INTO failures VALUES (?, ?, 1) ON CONFLICT(n, reason) "
                        "DO UPDATE SET count=count+1",
                        (n, reason),
                    )
        connection.commit()


@dataclass(frozen=True)
class _SizeSelection:
    plausible: int | None
    feasible: int | None
    unavailable: Counter[str]
    numerical: bool
    attempted: int
    admitted: int
    failed: int
    failed_refused: int
    known_runtime_refused: int
    availability_uncertified: int
    sampling_failed: int


def _select_sizes(
    connection: sqlite3.Connection,
    *,
    repetitions: int,
    radius: float,
    bands: int,
    target: float,
) -> _SizeSelection:
    plausible = feasible = None
    unavailable: Counter[str] = Counter()
    numerical = False
    attempted = admitted_total = failed_total = failed_refused_total = 0
    runtime_total = uncertified_total = sampling_total = 0
    for row in connection.execute(
        "SELECT n, reason, admitted, rejected, failed, failed_refused, possible, "
        "runtime_refused, uncertified, sampling_failed FROM sizes ORDER BY n"
    ):
        (
            n,
            reason,
            admitted,
            rejected,
            failed,
            failed_refused,
            possible,
            runtime_refused,
            uncertified,
            sampling_failed,
        ) = row
        if reason is not None:
            unavailable[reason] += 1
            # Exhausted admission budget makes this N scientifically inadmissible.
            if reason != "unit_cycle.error_budget_exhausted":
                numerical = True
                if plausible is None:
                    plausible = n
            continue
        attempted += repetitions
        admitted_total += admitted
        failed_total += failed
        failed_refused_total += failed_refused
        runtime_total += runtime_refused
        uncertified_total += uncertified
        sampling_total += sampling_failed
        numerical |= failed > 0
        lo, hi = _bounds(rejected, possible, repetitions, radius, bands)
        if plausible is None and hi >= target:
            plausible = n
        if feasible is None and lo >= target:
            feasible = n
    return _SizeSelection(
        plausible,
        feasible,
        unavailable,
        numerical,
        attempted,
        admitted_total,
        failed_total,
        failed_refused_total,
        runtime_total,
        uncertified_total,
        sampling_total,
    )


def _size_certificate(
    connection: sqlite3.Connection,
    law: UnitCycleJointLaw,
    procedure: ContrastDecisionProcedure,
    *,
    n: int,
    effect: float,
    repetitions: int,
    seed: int,
    error: float,
) -> UnitCycleModelPowerResult:
    (
        cutoff,
        alpha,
        refusal,
        admitted,
        rejected,
        failed_refused,
        possible,
        runtime_refused,
        uncertified,
        sampling_failed,
    ) = connection.execute(
        "SELECT cutoff, alpha, refusal, admitted, rejected, failed_refused, possible, "
        "runtime_refused, uncertified, sampling_failed FROM sizes WHERE n=?",
        (n,),
    ).fetchone()
    reasons = Counter(
        dict(connection.execute("SELECT reason, count FROM failures WHERE n=?", (n,)))
    )
    return _power_result(
        law,
        procedure,
        n=n,
        effect=effect,
        repetitions=repetitions,
        counts=_Counts(
            admitted=admitted,
            rejected=rejected,
            reasons=reasons,
            failed_refused=failed_refused,
            possible_rejected=possible,
            known_runtime_refused=runtime_refused,
            availability_uncertified=uncertified,
            sampling_failed=sampling_failed,
        ),
        seed=seed,
        error=error,
        cutoff=UnitCycleEnvelopeCutoff(
            value=cutoff, effective_alpha=alpha, refusal_probability_upper=refusal
        ),
    )


def unit_cycle_model_required_units(
    law: UnitCycleJointLaw,
    procedure: ContrastDecisionProcedure,
    *,
    effect_delta: float,
    target_power: float,
    max_n: int,
    repetitions: int,
    seed: int,
    mc_error: float,
    batch_size: int = 512,
) -> UnitCycleModelRequiredUnitsResult:
    """Research-only complete-law required-unit search.

    Ordinary required-N planning uses the pilot-fitted moment-t solver; this
    exhaustive path remains for independent model-conditioned calibration.
    The MC error is divided over every searched integer before any draw.
    Each N gets a fixed-effect Bernoulli bound; their union covers the search.
    SQLite retains counters without arrays growing with max_n or repetitions.
    An exhausted admission budget excludes that N; numerical inability leaves it plausible.
    """
    _integer("max_n", max_n, 1)
    effect_delta = _effect(effect_delta, field="effect_delta")
    _probability("target_power", target_power)
    _mc_inputs(repetitions, seed, mc_error, batch_size)
    law = _law(law)
    procedure = _procedure(procedure)
    envelope = unit_cycle_variance_envelope(law)
    _match_procedure(envelope, procedure)
    error = _allocate(mc_error, max_n)
    terms = 1
    each, radius = _fixed_band(repetitions, error)
    with _work_db() as connection:
        _prepare_sizes(connection, envelope, procedure, max_n)
        _collect_sizes(
            connection,
            law,
            procedure,
            max_n=max_n,
            effect=effect_delta,
            repetitions=repetitions,
            seed=seed,
            batch_size=batch_size,
        )
        selected = _select_sizes(
            connection, repetitions=repetitions, radius=radius, bands=terms, target=target_power
        )
        certificate = (
            None
            if selected.feasible is None
            else _size_certificate(
                connection,
                law,
                procedure,
                n=selected.feasible,
                effect=effect_delta,
                repetitions=repetitions,
                seed=seed,
                error=error,
            )
        )
        reasons = Counter(
            dict(connection.execute("SELECT reason, SUM(count) FROM failures GROUP BY reason"))
        )
    return UnitCycleModelRequiredUnitsResult(
        rng="numpy.PCG64/SeedSequence(seed,replicate)/unit-prefix-v2",
        sampler="finite_type_reuse_exact_micro_v2",
        law=law,
        procedure=procedure,
        effect_delta=effect_delta,
        target_power=target_power,
        max_n=max_n,
        plausible_n=selected.plausible,
        feasible_n=selected.feasible,
        plausible_unavailable_reason="search_limit" if selected.plausible is None else None,
        feasible_unavailable_reason="search_limit" if selected.feasible is None else None,
        power_at_feasible=certificate,
        checked_n=max_n,
        unavailable_n=sum(selected.unavailable.values()),
        unavailable_reasons=_failures(selected.unavailable),
        repetitions=repetitions,
        seed=seed,
        mc_error=mc_error,
        mc_error_per_n=error,
        cdf_bands=2 * terms,
        cdf_terms_per_bound=terms,
        cdf_error_each=each,
        cdf_radius=radius,
        attempted_across_n=selected.attempted,
        admitted_across_n=selected.admitted,
        failed_across_n=selected.failed,
        failed_refused_across_n=selected.failed_refused,
        known_runtime_refused_across_n=selected.known_runtime_refused,
        availability_uncertified_across_n=selected.availability_uncertified,
        sampling_failed_across_n=selected.sampling_failed,
        failures_across_n=_failures(reasons),
        search_status=(
            "numerical_incomplete"
            if selected.numerical
            else "bracketed"
            if selected.feasible is not None
            else "search_limit"
        ),
    )


__all__ = [
    "UnitCycleFailureCount",
    "UnitCycleLawMoments",
    "UnitCycleModelMdeResult",
    "UnitCycleModelPowerResult",
    "UnitCycleModelRequiredUnitsResult",
    "UnitCyclePowerLowerBoundResult",
    "unit_cycle_law_moments",
    "unit_cycle_power_lower_bound",
    "unit_cycle_variance_envelope",
]
