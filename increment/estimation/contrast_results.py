"""Typed fixed-horizon switchback contrast results."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from operator import index
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, SupportsIndex, overload

import narwhals as nw
from narwhals.typing import IntoDataFrame
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from increment._literals import Alternative, PreferredDirection, Role
from increment.errors import CodedModel, InvalidRequestError, RefusalSpec, refuse
from increment.estimation.results import Estimate
from increment.semantics.assignment import paired_sequence
from increment.semantics.unit_cycle import (
    ProspectiveAssumptionProvenance,
    UnitCycleReference,
    UnitCycleTApproximation,
    UnitCycleVarianceEnvelope,
)

if TYPE_CHECKING:
    from increment.estimation.readout_types import ReadoutMetadata

Backend = Literal["pandas", "polars", "pyarrow"]

# Independent units and shared blocks have different replication grains.
RandomizationLaw = Literal[
    "independent_bernoulli_order",
    "shared_schedule",
    "independent_declared_sequence",
    "shared_declared_sequence",
]
IndependenceGrain = Literal["unit_cycle", "shared_block"]

# Paired CT/TC laws carry ``probability_ct``; declared multi-period laws carry
# ``sequence_probabilities`` instead. The grain, not the law, picks inference.
UNIT_CYCLE_LAWS = frozenset({"independent_bernoulli_order", "independent_declared_sequence"})
DECLARED_SEQUENCE_LAWS = frozenset({"independent_declared_sequence", "shared_declared_sequence"})


def law_grain(law: str) -> IndependenceGrain:
    """The independent replicate grain implied by a randomization law."""
    return "unit_cycle" if law in UNIT_CYCLE_LAWS else "shared_block"


def is_declared_sequence_law(law: str) -> bool:
    return law in DECLARED_SEQUENCE_LAWS


def declared_sequence_law(independence_unit: str) -> RandomizationLaw:
    """Result-level law name for a declared sequence draw at ``independence_unit``."""
    return (
        "independent_declared_sequence"
        if independence_unit == "unit_cycle"
        else "shared_declared_sequence"
    )


def canonical_sequence_probabilities(value: object) -> Mapping[str, float] | None:
    """Freeze a declared sequence mapping in key order, or pass through ``None``."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or not value:
        refuse(
            _REFUSALS["estimation.contrast.sequence_identity"],
            randomization_law="declared sequence",
        )
    copied: dict[str, float] = {}
    for key, probability in value.items():
        if (
            not isinstance(key, str)
            or not key
            or isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0.0 < probability <= 1.0
        ):
            refuse(
                _REFUSALS["estimation.contrast.sequence_identity"],
                randomization_law="declared sequence",
            )
        copied[key] = float(probability)
    return MappingProxyType(dict(sorted(copied.items())))


def drop_absent_sequence_law(data: dict[str, Any]) -> dict[str, Any]:
    """Omit ``sequence_probabilities`` when null so paired-law payloads are unchanged."""
    if data.get("sequence_probabilities", ...) is None:
        del data["sequence_probabilities"]
    return data


_REFUSALS = {
    "estimation.contrast.assignment_metadata": RefusalSpec(
        "estimation.contrast.assignment_metadata",
        InvalidRequestError,
        template="{randomization_law} requires {expected_grain}, got {independence_grain}",
    ),
    "estimation.contrast.replication_counts": RefusalSpec(
        "estimation.contrast.replication_counts",
        InvalidRequestError,
        template="contrast counts must match the declared independent replicate grain",
    ),
    "estimation.contrast.order_counts": RefusalSpec(
        "estimation.contrast.order_counts",
        InvalidRequestError,
        template="CT and TC counts must sum to the declared number of order draws",
    ),
    "estimation.contrast.reference_metadata": RefusalSpec(
        "estimation.contrast.reference_metadata",
        InvalidRequestError,
        template="contrast method, reference and degrees of freedom must match its replicates",
    ),
    "estimation.contrast.sequence_identity": RefusalSpec(
        "estimation.contrast.sequence_identity",
        InvalidRequestError,
        template=(
            "{randomization_law} requires probability_ct for paired CT/TC laws or "
            "sequence_probabilities for declared sequence laws, never both"
        ),
    ),
    "estimation.contrast.retained_window": RefusalSpec(
        "estimation.contrast.retained_window",
        InvalidRequestError,
        template="retained_steps={retained_steps} must equal observation_steps={observation_steps} minus carryover_order={carryover_order} and be positive",
    ),
}


def _validate_retained_window(
    observation_steps: int, retained_steps: int, carryover_order: int
) -> None:
    if retained_steps != observation_steps - carryover_order or retained_steps < 1:
        refuse(
            _REFUSALS["estimation.contrast.retained_window"],
            observation_steps=observation_steps,
            retained_steps=retained_steps,
            carryover_order=carryover_order,
        )


def _validate_assignment_metadata(law: RandomizationLaw, grain: IndependenceGrain) -> None:
    expected = law_grain(law)
    if grain != expected:
        refuse(
            _REFUSALS["estimation.contrast.assignment_metadata"],
            randomization_law=law,
            independence_grain=grain,
            expected_grain=expected,
        )


def _validate_replication_counts(
    law: RandomizationLaw, n_units: int, n_cycles: int, n_blocks: int | None
) -> int:
    if law in UNIT_CYCLE_LAWS:
        if n_blocks is None and n_cycles >= n_units:
            return n_units
    elif n_blocks is not None and n_cycles == n_units * n_blocks:
        return n_blocks
    refuse(
        _REFUSALS["estimation.contrast.replication_counts"],
        randomization_law=law,
        n_units=n_units,
        n_cycles=n_cycles,
        n_blocks=n_blocks,
    )


def _validate_order_counts(
    law: RandomizationLaw,
    n_cycles: int,
    n_blocks: int | None,
    ct_cycles: int | None,
    tc_cycles: int | None,
) -> None:
    if (law == "independent_bernoulli_order" or law in DECLARED_SEQUENCE_LAWS) and (
        ct_cycles is None and tc_cycles is None
    ):
        return
    draws = n_cycles if law == "independent_bernoulli_order" else n_blocks
    if (
        law in DECLARED_SEQUENCE_LAWS
        or ct_cycles is None
        or tc_cycles is None
        or ct_cycles + tc_cycles != draws
    ):
        refuse(
            _REFUSALS["estimation.contrast.order_counts"],
            randomization_law=law,
            n_cycles=n_cycles,
            n_blocks=n_blocks,
            ct_cycles=ct_cycles,
            tc_cycles=tc_cycles,
        )


def _validate_sequence_identity(
    law: RandomizationLaw,
    probability_ct: float | None,
    sequence_probabilities: Mapping[str, float] | None,
) -> None:
    declared = law in DECLARED_SEQUENCE_LAWS
    if (probability_ct is None) != declared or (sequence_probabilities is None) == declared:
        refuse(_REFUSALS["estimation.contrast.sequence_identity"], randomization_law=law)


def _validate_sequence_counts(
    law: RandomizationLaw,
    n_cycles: int,
    n_blocks: int | None,
    sequence_probabilities: Mapping[str, float] | None,
    sequence_counts: Mapping[str, int] | None,
    treatment_probabilities: tuple[float, ...] | None,
    realized_treated_share: float | None,
) -> None:
    """Realized sequence evidence is present exactly for declared laws and is coherent."""
    realized = (sequence_counts, treatment_probabilities, realized_treated_share)
    if sequence_probabilities is None:
        if any(value is not None for value in realized):
            refuse(_REFUSALS["estimation.contrast.sequence_identity"], randomization_law=law)
        return
    draws = n_cycles if law in UNIT_CYCLE_LAWS else n_blocks
    periods = len(next(iter(sequence_probabilities)))
    if (
        sequence_counts is None
        or treatment_probabilities is None
        or realized_treated_share is None
        or len(treatment_probabilities) != periods
        or not set(sequence_counts) <= set(sequence_probabilities)
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in sequence_counts.values()
        )
        or sum(sequence_counts.values()) != draws
    ):
        refuse(
            _REFUSALS["estimation.contrast.order_counts"],
            randomization_law=law,
            n_cycles=n_cycles,
            n_blocks=n_blocks,
            ct_cycles=None,
            tc_cycles=None,
        )


class ContrastResult(CodedModel, BaseModel):
    """One additive contrast with realized unit-cycle or shared-block order counts."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    metric: str = Field(min_length=1)
    control_group: str = Field(min_length=1)
    treatment_group: str = Field(min_length=1)
    method: Literal[
        "switchback_unit_t_approximation",
        "switchback_block_t",
        "switchback_unit_variance_envelope",
    ] = "switchback_unit_t_approximation"
    method_role: Literal["decision"] = "decision"
    role: Role = "unassigned"
    estimand: Literal[
        "retained_window_total_difference",
        "mean_unit_retained_window_difference",
        "retained_window_conversion_difference",
    ]
    aggregation: Literal["sum", "any"]
    # Paired CT/TC laws only; declared multi-period laws report ``sequence_probabilities``.
    probability_ct: float | None = Field(gt=0.0, lt=1.0)
    sequence_probabilities: Mapping[str, float] | None = None
    # Positive carryover orders discard additional post-washout steps.
    randomization_law: RandomizationLaw
    independence_grain: IndependenceGrain
    carryover_order: int = Field(ge=0, strict=True)
    observation_steps: int = Field(ge=1, strict=True)
    retained_steps: int = Field(ge=1, strict=True)
    identifying_assumption: Literal["no_residual_carryover_after_discarded_steps"] = (
        "no_residual_carryover_after_discarded_steps"
    )
    estimate: Estimate
    standard_error: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
        description="Descriptive sample SE over independent replicates; not the envelope reference.",
    )
    alternative: Alternative
    preferred_direction: PreferredDirection | None = None
    null_abs: float = Field(allow_inf_nan=False)
    alpha: float = Field(gt=0.0, lt=1.0)
    n_units: int = Field(ge=1, strict=True)
    n_cycles: int = Field(ge=1, strict=True)
    n_blocks: int | None = Field(default=None, ge=1, strict=True)
    dof: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    assignment: Literal["switchback"] = "switchback"
    inference: Literal["fixed"] = "fixed"
    reference: Literal[
        "unit_t_approximation",
        "block_t",
        "residual_chebyshev",
        "residual_cantelli",
    ] = "unit_t_approximation"
    dof_unavailable_reason: Literal["not_applicable"] | None = None
    standard_error_unavailable_reason: Literal["insufficient_replicates"] | None = None
    mean_slope: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    ct_cycles: int | None = Field(default=None, ge=0, strict=True)
    tc_cycles: int | None = Field(default=None, ge=0, strict=True)
    minimum_cycles_per_unit: int | None = Field(default=None, ge=1, strict=True)
    maximum_cycles_per_unit: int | None = Field(default=None, ge=1, strict=True)
    washout_steps: int = Field(default=0, ge=0, strict=True)
    effective_alpha: float | None = Field(default=None, gt=0, lt=1, allow_inf_nan=False)
    refusal_probability_upper: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    residual_cutoff: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    residual_p_value: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    reference_spec: UnitCycleReference | None = None
    provenance: ProspectiveAssumptionProvenance | None = None
    response_meaning: Literal["retained_total", "pre_normalized_retained_mean"] | None = None
    analysis_population: Literal["assigned"] = "assigned"
    source_snapshot_id: str | None = None
    decision_scope_complete: bool | None = None
    decision_scope_reason_code: str | None = None
    decision_scope_reason_context: Mapping[str, object] | None = None

    @model_validator(mode="after")
    def _freeze_scope_context(self):
        if self.decision_scope_reason_context is not None:
            from increment._canonical import canonical_json_bytes
            from increment._immutable import _FrozenMapping

            def freeze(value):
                if isinstance(value, Mapping):
                    frozen = _FrozenMapping({key: freeze(item) for key, item in value.items()})
                    canonical_json_bytes(dict(frozen))
                    return frozen
                if isinstance(value, (tuple, list)):
                    return tuple(freeze(item) for item in value)
                return value

            object.__setattr__(
                self, "decision_scope_reason_context", freeze(self.decision_scope_reason_context)
            )
        return self

    _sequence_probabilities = field_validator("sequence_probabilities", mode="before")(
        canonical_sequence_probabilities
    )

    @model_serializer(mode="wrap")
    def _omit_absent_sequence_law(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        if self.sequence_probabilities is not None:
            data["sequence_probabilities"] = dict(self.sequence_probabilities)
        return drop_absent_sequence_law(data)

    @model_validator(mode="after")
    def _consistent_design(self) -> ContrastResult:
        _validate_retained_window(self.observation_steps, self.retained_steps, self.carryover_order)
        _validate_assignment_metadata(self.randomization_law, self.independence_grain)
        _validate_sequence_identity(
            self.randomization_law, self.probability_ct, self.sequence_probabilities
        )
        replicates = _validate_replication_counts(
            self.randomization_law, self.n_units, self.n_cycles, self.n_blocks
        )
        _validate_order_counts(
            self.randomization_law, self.n_cycles, self.n_blocks, self.ct_cycles, self.tc_cycles
        )
        if self.method == "switchback_unit_variance_envelope":
            expected_reference = (
                "residual_chebyshev" if self.alternative == "two-sided" else "residual_cantelli"
            )
            envelope = self.reference_spec
            valid = (
                self.randomization_law == "independent_bernoulli_order"
                and isinstance(envelope, UnitCycleVarianceEnvelope)
                and self.reference == expected_reference
                and self.dof is None
                and self.dof_unavailable_reason == "not_applicable"
                and self.mean_slope is not None
                and self.effective_alpha is not None
                and self.refusal_probability_upper is not None
                and self.residual_cutoff is not None
                and self.residual_p_value is not None
                and self.ct_cycles is not None
                and self.tc_cycles is not None
                and self.ct_cycles + self.tc_cycles == self.n_cycles
                and self.minimum_cycles_per_unit == self.maximum_cycles_per_unit
                and self.minimum_cycles_per_unit == envelope.cycles_per_unit
                and self.metric == envelope.metric
                and self.control_group == envelope.control_group
                and self.treatment_group == envelope.treatment_group
                and self.aggregation == envelope.aggregation
                and self.estimand == envelope.estimand
                and self.n_cycles == self.n_units * envelope.cycles_per_unit
                and self.probability_ct == paired_sequence(envelope.assignment).probability_ct
                and self.washout_steps == envelope.assignment.window.washout_steps
                and self.observation_steps == envelope.assignment.window.observation_steps
                and self.carryover_order == envelope.assignment.window.carryover_order
                and self.estimate.alpha == self.alpha
                and (self.estimate.lb is not None or self.estimate.ub is not None)
                and self.provenance == envelope.provenance
                and self.response_meaning == envelope.response_meaning
                and self.estimate.open_side
                == (
                    "upper"
                    if self.alternative == "greater"
                    else "lower"
                    if self.alternative == "less"
                    else None
                )
                and (
                    (
                        self.standard_error is not None
                        and self.standard_error_unavailable_reason is None
                        and replicates >= 2
                    )
                    or (
                        self.standard_error is None
                        and self.standard_error_unavailable_reason == "insufficient_replicates"
                        and replicates == 1
                    )
                )
            )
            if valid:
                from increment._unit_cycle import unit_cycle_envelope_cutoff

                assert isinstance(envelope, UnitCycleVarianceEnvelope)
                cutoff = unit_cycle_envelope_cutoff(
                    envelope, n=self.n_units, alpha=self.alpha, alternative=self.alternative
                )
                # Outward-rounded display bounds need not exclude a significant null.
                valid = (
                    self.effective_alpha == cutoff.effective_alpha
                    and self.refusal_probability_upper == cutoff.refusal_probability_upper
                    and self.residual_cutoff == cutoff.value
                )
            if not valid:
                refuse(
                    _REFUSALS["estimation.contrast.reference_metadata"],
                    randomization_law=self.randomization_law,
                    method=self.method,
                    reference=self.reference,
                    dof=self.dof,
                    independent_replicates=replicates,
                )
            return self
        expected = (
            ("switchback_unit_t_approximation", "unit_t_approximation")
            if self.randomization_law in UNIT_CYCLE_LAWS
            else ("switchback_block_t", "block_t")
        )
        if (
            (self.method, self.reference) != expected
            or replicates < 2
            or self.dof != replicates - 1
            or self.standard_error is None
            or self.dof_unavailable_reason is not None
            or any(
                value is not None
                for value in (
                    self.standard_error_unavailable_reason,
                    self.effective_alpha,
                    self.refusal_probability_upper,
                    self.residual_cutoff,
                    self.residual_p_value,
                    self.provenance,
                    self.response_meaning,
                )
            )
            or (
                self.randomization_law in UNIT_CYCLE_LAWS
                and not isinstance(self.reference_spec, UnitCycleTApproximation)
            )
            or (self.randomization_law not in UNIT_CYCLE_LAWS and self.reference_spec is not None)
        ):
            refuse(
                _REFUSALS["estimation.contrast.reference_metadata"],
                randomization_law=self.randomization_law,
                method=self.method,
                reference=self.reference,
                dof=self.dof,
                independent_replicates=replicates,
            )
        return self

    def __repr__(self) -> str:
        """Show the additive contrast identity and its interval, not its full evidence model."""
        estimate = self.estimate
        if estimate.open_side == "lower" and estimate.ub is not None:
            interval = f"(-inf, {estimate.ub!r})"
        elif estimate.open_side == "upper" and estimate.lb is not None:
            interval = f"({estimate.lb!r}, +inf)"
        else:
            interval = f"({estimate.lb!r}, {estimate.ub!r})"
        return (
            f"ContrastResult(metric={self.metric!r}, control={self.control_group!r}, "
            f"treatment={self.treatment_group!r}, method={self.method!r}, "
            f"role={self.role!r}, estimand={self.estimand!r}, "
            f"population={self.analysis_population!r}, inference={self.inference!r}, "
            f"scale='absolute', estimate={estimate.value!r}, interval={interval}, "
            f"alternative={self.alternative!r}, null_abs={self.null_abs!r}, "
            f"randomization_law={self.randomization_law!r}, "
            f"independence_grain={self.independence_grain!r}, "
            f"carryover_order={self.carryover_order}, retained_steps={self.retained_steps}, "
            f"reference={self.reference!r}, decision_scope={self.decision_scope_complete!r})"
        )

    __str__ = __repr__

    def __repr_args__(self) -> list[tuple[str, object]]:
        """Give Rich a compact additive contrast summary."""
        return [
            ("metric", self.metric),
            ("control", self.control_group),
            ("treatment", self.treatment_group),
            ("population", self.analysis_population),
            ("scale", "absolute"),
            ("summary", self.__repr__()),
        ]


class ContrastResults(list[ContrastResult]):
    """Typed contrast results with immutable scope metadata and native frame output."""

    _model = ContrastResult

    def __init__(
        self,
        rows=(),
        *,
        metadata: ReadoutMetadata | None = None,
        source=None,
        sequential_snapshot=None,
    ):
        from increment.estimation.readout_types import validate_collection

        super().__init__(rows)
        self._metadata = metadata
        self.source = source
        self.sequential_snapshot = sequential_snapshot
        validate_collection(self, metadata)

    @property
    def metadata(self) -> ReadoutMetadata | None:
        return self._metadata

    @metadata.setter
    def metadata(self, value) -> None:
        self._mutation("metadata_set")

    @metadata.deleter
    def metadata(self) -> None:
        self._mutation("metadata_delete")

    def __reduce_ex__(self, protocol):
        from increment.estimation.readout_types import _restore_collection

        return _restore_collection, (
            type(self),
            tuple(self),
            self.metadata,
            self.source,
            self.sequential_snapshot,
        )

    def model_dump_json(self):
        from increment.estimation.readout_types import dump_collection

        return dump_collection(self)

    @overload
    def __getitem__(self, key: SupportsIndex) -> ContrastResult: ...

    @overload
    def __getitem__(self, key: slice) -> ContrastResults: ...

    def __getitem__(self, key: SupportsIndex | slice) -> ContrastResult | ContrastResults:
        if isinstance(key, slice):
            from increment.estimation.readout_types import partial_metadata

            rows = super().__getitem__(key)
            return type(self)(
                rows,
                metadata=partial_metadata(self.metadata, rows, "slice"),
                source=self.source,
                sequential_snapshot=self.sequential_snapshot,
            )
        return super().__getitem__(index(key))

    def filter(self, predicate):
        from increment.estimation.readout_types import partial_metadata

        rows = [row for row in self if predicate(row)]
        return type(self)(
            rows,
            metadata=partial_metadata(self.metadata, rows, "filter"),
            source=self.source,
            sequential_snapshot=self.sequential_snapshot,
        )

    def concat(self, other):
        from increment.estimation.readout_types import concat_collection

        return concat_collection(self, other)

    def __add__(self, other):
        return self.concat(other)

    def _mutation(self, operation):
        from increment.estimation.readout_types import refuse_readout

        refuse_readout(
            "readout.collection.mutation_unsupported",
            operation=operation,
            model=type(self).__name__,
        )

    def append(self, value):
        self._mutation("append")

    def extend(self, values):
        self._mutation("extend")

    def insert(self, index, value):
        self._mutation("insert")

    def pop(self, index=-1):
        self._mutation("pop")

    def remove(self, value):
        self._mutation("remove")

    def clear(self):
        self._mutation("clear")

    def reverse(self):
        self._mutation("reverse")

    def sort(self, *args, **kwargs):
        self._mutation("sort")

    def __setitem__(self, key, value):
        self._mutation("item_set")

    def __delitem__(self, key):
        self._mutation("item_delete")

    def __iadd__(self, value):
        self._mutation("iadd")

    def __imul__(self, value):
        self._mutation("imul")

    def to_frame(self, *, backend: Backend = "pandas") -> IntoDataFrame:
        """Return rows as a pandas, polars, or pyarrow frame.

        The nested :class:`Estimate` is expanded in the same shape as other
        increment result frames: ``estimate``, ``lb``, and ``ub`` columns.
        """
        columns = [
            "metric",
            "control_group",
            "treatment_group",
            "method",
            "method_role",
            "role",
            "estimand",
            "aggregation",
            "probability_ct",
            "sequence_probabilities",
            "randomization_law",
            "independence_grain",
            "carryover_order",
            "observation_steps",
            "retained_steps",
            "identifying_assumption",
            "estimate",
            "lb",
            "ub",
            "standard_error",
            "alternative",
            "preferred_direction",
            "null_abs",
            "alpha",
            "n_units",
            "n_cycles",
            "n_blocks",
            "ct_cycles",
            "tc_cycles",
            "dof",
            "assignment",
            "inference",
            "reference",
            "open_side",
            "dof_unavailable_reason",
            "standard_error_unavailable_reason",
            "mean_slope",
            "minimum_cycles_per_unit",
            "maximum_cycles_per_unit",
            "washout_steps",
            "effective_alpha",
            "refusal_probability_upper",
            "residual_cutoff",
            "residual_p_value",
            "reference_spec",
            "provenance",
            "response_meaning",
            "analysis_population",
            "source_snapshot_id",
            "decision_scope_complete",
            "decision_scope_reason_code",
            "decision_scope_reason_context",
        ]
        data: dict[str, list[object]] = {name: [] for name in columns}
        for result in self:
            for name in ContrastResult.model_fields:
                if name == "estimate":
                    estimate = result.estimate
                    data["estimate"].append(estimate.value)
                    data["lb"].append(estimate.lb)
                    data["ub"].append(estimate.ub)
                    data["open_side"].append(estimate.open_side)
                elif name in {"reference_spec", "provenance"}:
                    value = getattr(result, name)
                    data[name].append(
                        None
                        if value is None
                        else json.dumps(
                            value.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
                        )
                    )
                elif name == "sequence_probabilities":
                    value = result.sequence_probabilities
                    data[name].append(
                        None
                        if value is None
                        else json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
                    )
                elif name == "decision_scope_reason_context":
                    from increment.estimation.readout_types import thaw

                    value = getattr(result, name)
                    data[name].append(
                        None
                        if value is None
                        else json.dumps(thaw(value), sort_keys=True, separators=(",", ":"))
                    )
                else:
                    data[name].append(getattr(result, name))

        schema = {
            name: nw.Boolean()
            if name == "decision_scope_complete"
            else nw.Float64()
            if name
            in {
                "estimate",
                "lb",
                "ub",
                "standard_error",
                "probability_ct",
                "null_abs",
                "alpha",
                "dof",
                "mean_slope",
                "effective_alpha",
                "refusal_probability_upper",
                "residual_cutoff",
                "residual_p_value",
            }
            else nw.Int64()
            if name
            in {
                "n_units",
                "n_cycles",
                "n_blocks",
                "ct_cycles",
                "tc_cycles",
                "carryover_order",
                "observation_steps",
                "retained_steps",
                "washout_steps",
                "minimum_cycles_per_unit",
                "maximum_cycles_per_unit",
            }
            else nw.String()
            for name in columns
        }
        nullable_ints = {
            name: data[name]
            for name in (
                "n_blocks",
                "ct_cycles",
                "tc_cycles",
                "minimum_cycles_per_unit",
                "maximum_cycles_per_unit",
            )
        }
        if backend == "pandas":
            for name in nullable_ints:
                data.pop(name)
                schema.pop(name)
        frame = nw.from_dict(data, schema=schema, backend=backend).to_native()
        if backend == "pandas":
            import pandas as pd

            for name, values in nullable_ints.items():
                frame.insert(columns.index(name), name, pd.array(values, dtype="Int64"))
            frame["preferred_direction"] = pd.array(data["preferred_direction"], dtype="string")
        return (
            nw.from_native(frame, eager_only=True)
            .with_columns(
                nw.lit(None if self.metadata is None else self.metadata.partial).alias(
                    "view_partial"
                )
            )
            .to_native()
        )


__all__ = ["ContrastResult", "ContrastResults"]
