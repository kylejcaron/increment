"""Declarations for two concurrently randomized binary experiments."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from increment._overlap.refusals import _raise
from increment.errors import CodedModel

JointLaw = Literal["independent", "dependent", "mutually_exclusive"]
TargetPopulation = Literal["jointly_eligible"]


class OverlapExperiment(CodedModel, BaseModel):
    """One binary experiment and its declared assignment probability.

    ``treatment_probability`` is the randomization probability among units
    eligible for this experiment. It defines the experiment's treatment mix,
    which weights the marginal effect of the other experiment.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    control: str = "control"
    treatment: str = "treatment"
    treatment_probability: float = 0.5

    @model_validator(mode="after")
    def _validate(self) -> OverlapExperiment:
        if self.control == self.treatment:
            _raise("overlap.design.arm_labels", name=self.name, label=self.control)
        p = self.treatment_probability
        if not (isinstance(p, float) and math.isfinite(p) and 0.0 < p < 1.0):
            _raise("overlap.design.treatment_probability", name=self.name, probability=p)
        return self


class OverlapDesign(CodedModel, BaseModel):
    """Two binary experiments, their joint assignment law, and the target population.

    Only independently randomized experiments are analyzed: under
    independence every launch combination is assigned with probability
    ``p_first`` or ``1 - p_first`` times ``p_second`` or ``1 - p_second``
    within the jointly eligible population. ``target`` is that population;
    units eligible for only one experiment are audited, never pooled.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    first: OverlapExperiment
    second: OverlapExperiment
    joint_law: JointLaw = "independent"
    target: TargetPopulation = "jointly_eligible"

    @model_validator(mode="after")
    def _validate(self) -> OverlapDesign:
        if self.first.name == self.second.name:
            _raise("overlap.design.duplicate_experiment", name=self.first.name)
        if self.joint_law != "independent":
            _raise("overlap.design.joint_law_unsupported", joint_law=self.joint_law)
        return self

    @property
    def experiments(self) -> tuple[OverlapExperiment, OverlapExperiment]:
        return (self.first, self.second)


class ExperimentColumns(CodedModel, BaseModel):
    """Per-unit columns recording one experiment's eligibility and assignment.

    ``eligible`` is a pre-treatment boolean. ``assignment`` holds the arm
    label for eligible units and is null for ineligible units.
    ``assigned_at`` is the assignment timestamp, null when unassigned.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    eligible: str = Field(min_length=1)
    assignment: str = Field(min_length=1)
    assigned_at: str = Field(min_length=1)

    @property
    def names(self) -> tuple[str, str, str]:
        return (self.eligible, self.assignment, self.assigned_at)
