"""Predeclared observation schedules for randomized longitudinal contrasts."""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Literal, NoReturn, cast

from pydantic import BaseModel, ConfigDict, model_validator

from increment.errors import CodedModel, InvalidRequestError, RefusalSpec, refuse

Time = int | dt.datetime
LONGITUDINAL_INPUT = RefusalSpec("longitudinal.input", InvalidRequestError, template="{message}")
LONGITUDINAL_OBSERVATION = RefusalSpec(
    "longitudinal.observation", InvalidRequestError, template="{message}"
)
LONGITUDINAL_ESTIMABILITY = RefusalSpec(
    "longitudinal.estimability", InvalidRequestError, template="{message}"
)
LONGITUDINAL_NUMERIC = RefusalSpec(
    "longitudinal.numeric", InvalidRequestError, template="{message}"
)


def invalid(message: str, **context: object) -> NoReturn:
    """Reject an invalid family request with immutable context."""
    refuse(LONGITUDINAL_INPUT, message=message, **context)


class LongitudinalCohort(CodedModel, BaseModel):
    """A roster cohort's treatment start and scheduled pre/post observation times.

    Times are integer indices or naive UTC datetimes, consistently throughout
    one design. Post weights default to equal weights and must sum to one.
    Pre-period outcomes always receive equal weights.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    name: str
    exposure: Time
    pre_periods: tuple[Time, ...]
    post_periods: tuple[Time, ...]
    post_weights: tuple[float, ...] = ()

    @model_validator(mode="after")
    def _validate_schedule(self) -> LongitudinalCohort:
        times = (self.exposure, *self.pre_periods, *self.post_periods)
        if not self.name or not self.pre_periods or not self.post_periods:
            invalid("Each named cohort requires pre and post observation times.")
        if any(type(t) is not type(self.exposure) for t in times):
            invalid("Use one time type: integer indices or naive UTC datetimes.")
        if isinstance(self.exposure, dt.datetime) and any(
            isinstance(t, dt.datetime) and t.tzinfo is not None for t in times
        ):
            invalid("Convert observation times to naive UTC datetimes before declaring the design.")
        if len(set(times[1:])) != len(times) - 1:
            invalid("Pre and post observation times must be unique and disjoint.")
        if any(cast(Any, t) >= self.exposure for t in self.pre_periods) or any(
            cast(Any, t) < self.exposure for t in self.post_periods
        ):
            invalid("Pre times must precede exposure; post times must be at or after exposure.")
        weights = self.post_weights or tuple(1 / len(self.post_periods) for _ in self.post_periods)
        if (
            len(weights) != len(self.post_periods)
            or any(not math.isfinite(w) or not 0 < w <= 1 for w in weights)
            or not math.isclose(math.fsum(weights), 1, rel_tol=0, abs_tol=1e-12)
        ):
            invalid("Post weights must be finite, positive, match post times, and sum to one.")
        total = math.fsum(weights)
        object.__setattr__(self, "post_weights", tuple(w / total for w in weights))
        return self


class LongitudinalDesign(CodedModel, BaseModel):
    """Fixed-horizon equal-unit additive effect in a predeclared assigned roster.

    Declaring this design asserts randomized, treatment-unaffected membership
    and observation schedules, no anticipation, and no between-grain spillovers.
    The constructor checks data consistency; it cannot establish these assumptions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    control_group: str
    treatment_group: str
    cohorts: tuple[LongitudinalCohort, ...]
    independence_grain: Literal["unit", "cluster"] = "unit"
    population: Literal["assigned_roster"] = "assigned_roster"
    observation_policy: Literal["complete_scheduled_roster"] = "complete_scheduled_roster"

    @model_validator(mode="after")
    def _validate_design(self) -> LongitudinalDesign:
        if (
            not self.control_group
            or not self.treatment_group
            or self.control_group == self.treatment_group
        ):
            invalid("Declare two distinct nonempty randomized arm names.")
        if not self.cohorts or len({c.name for c in self.cohorts}) != len(self.cohorts):
            invalid("Declare nonempty, uniquely named cohorts.")
        if len({type(c.exposure) for c in self.cohorts}) != 1:
            invalid("All cohorts must use the same observation time type.")
        return self
