"""Signed conversion differences under personal nonoffer."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from math import isfinite
from typing import TYPE_CHECKING, ClassVar, Literal, Self, overload

from pydantic import BaseModel, ConfigDict, Field, model_validator

from increment.breakout.projection import to_frame, with_view_partial
from increment.errors import CodedModel
from increment.estimation.readout_types import (
    _merge_cell_identities,
    dump_collection,
    partial_metadata,
    refuse_readout,
    register_collection_types,
    validate_collection,
)
from increment.estimation.spillover import RangeBasis

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

    from increment._literals import Backend
    from increment.estimation.readout_types import ReadoutMetadata


class SpilloverEffect(CodedModel, BaseModel):
    """One registered policy minus its reference, targeting everyone under nonoffer."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    policy: str
    reference: str
    metric: str
    window_days: int = Field(gt=0)
    analysis_population: Literal["assigned"] = "assigned"
    own_assignment: Literal["unoffered"] = "unoffered"
    estimand: Literal["spillover"] = "spillover"
    value_scale: Literal["absolute"] = "absolute"
    inference: Literal["fixed"] = "fixed"
    method: Literal["two-stage-fixed-bet-v2"] = "two-stage-fixed-bet-v2"
    estimate: float = Field(allow_inf_nan=False)
    lower: float | None
    upper: float | None
    range_basis: RangeBasis
    alpha: float = Field(gt=0, lt=1)
    family_size: int = Field(gt=0)
    policy_count: int = Field(ge=2)
    direction: Literal["increase", "decrease", "inconclusive", "unavailable"] = "inconclusive"
    unavailable_reason: str | None = None
    registration_hash: str
    source_snapshot_id: str
    family_id: str

    @model_validator(mode="after")
    def _resolve_interval(self) -> Self:
        lower, upper = self.lower, self.upper
        reason = self.unavailable_reason
        if lower is None or upper is None or not (isfinite(lower) and isfinite(upper)):
            object.__setattr__(self, "lower", None)
            object.__setattr__(self, "upper", None)
            object.__setattr__(self, "range_basis", "unavailable")
            object.__setattr__(self, "unavailable_reason", reason or "nonfinite_interval")
            direction = "unavailable"
        elif lower > upper or reason is not None or self.range_basis == "unavailable":
            refuse_readout(
                "readout.serialization.identity_mismatch",
                source_snapshot_id=self.source_snapshot_id,
                cell_key=None,
                reason="range does not match its availability",
            )
        else:
            direction = "increase" if lower > 0 else "decrease" if upper < 0 else "inconclusive"
        object.__setattr__(self, "direction", direction)
        return self

    @property
    def sampling_available(self) -> bool:
        return self.lower is not None

    @property
    def sampling_reason_code(self) -> str | None:
        return self.unavailable_reason

    @property
    def sampling_reason_context(self) -> None:
        return None

    @property
    def multiplicity(self) -> str:
        if self.range_basis == "identical_offer_counts":
            return (
                "Both policies make the same number of offers in every group. If only the number "
                "of offers matters, their effect is zero by design."
            )
        if self.range_basis == "attainable_range":
            return (
                "The range covers every effect allowed by the offer counts; "
                "the outcomes do not establish a narrower range or direction."
            )
        if self.range_basis == "unavailable":
            return f"Range unavailable: {self.unavailable_reason}."
        return (
            f"The chance any of the {self.family_size} registered ranges misses its effect is at "
            f"most {self.alpha:g} in exact arithmetic. The endpoints approximate the rule that "
            f"splits that error budget across both tails of {self.policy_count} compared policies."
        )

    def _display(self, *, full: bool = False) -> str:
        bounds = (
            ""
            if self.lower is None or self.upper is None
            else (f" [{100 * self.lower:.4g}, {100 * self.upper:.4g}]")
        )
        note = (
            self.multiplicity
            if full or self.range_basis != "family_interval"
            else f"Family: {self.family_size} comparisons, alpha={self.alpha:g}."
        )
        return (
            f"{self.metric} ({self.window_days} days), {self.policy!r} minus {self.reference!r}: "
            f"{100 * self.estimate:+.4g}{bounds} percentage points; {self.direction}. "
            f"For all eligible people if they receive no offer themselves. "
            f"Estimated from people who received no offer. {note}"
        )


@dataclass(frozen=True, slots=True, repr=False)
class SpilloverEffects(Sequence[SpilloverEffect]):
    """Policy comparisons in registration order, sharing one capture and error budget."""

    _model: ClassVar[type[SpilloverEffect]] = SpilloverEffect
    _summary_columns: ClassVar[tuple[str, ...]] = (
        "metric",
        "window_days",
        "policy",
        "reference",
        "analysis_population",
        "own_assignment",
        "estimate",
        "lower",
        "upper",
        "direction",
        "range_basis",
        "unavailable_reason",
    )
    rows: tuple[SpilloverEffect, ...]
    metadata: ReadoutMetadata
    _default_backend: Backend | None = field(default=None, compare=False)
    source: None = field(default=None, kw_only=True)
    sequential_snapshot: None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        metadata = self.metadata
        if (
            metadata is None
            or metadata.run_id is None
            or metadata.registration_hash is None
            or len(metadata.scope.families) != 1
            or metadata.scope.families[0].view != "spillover"
            or self.source is not None
            or self.sequential_snapshot is not None
        ):
            refuse_readout(
                "readout.serialization.identity_mismatch",
                source_snapshot_id=None if metadata is None else metadata.scope.snapshot_id,
                cell_key=None,
                reason="policy comparisons require a single registered fixed-horizon family",
            )
        family = metadata.scope.families[0]
        policies = {policy for key in family.members for policy in (key.policy, key.reference)}
        terms = family.policy_interval
        if (
            not family.complete
            or family.source_snapshot_id != metadata.scope.snapshot_id
            or terms is None
            or any(
                row.registration_hash != metadata.registration_hash
                or row.family_size != len(family.members)
                or row.policy_count != len(policies)
                or row.metric != family.identity_metric
                or row.alpha != terms.alpha
                or row.window_days != terms.window_days
                for row in self.rows
            )
        ):
            refuse_readout(
                "readout.serialization.identity_mismatch",
                source_snapshot_id=metadata.scope.snapshot_id,
                cell_key=None,
                reason="policy comparisons differ from their registered family",
            )
        object.__setattr__(self, "rows", tuple(self.rows))
        validate_collection(self.rows, self.metadata)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[SpilloverEffect]:
        return iter(self.rows)

    @overload
    def __getitem__(self, index: int) -> SpilloverEffect: ...

    @overload
    def __getitem__(self, index: slice) -> SpilloverEffects: ...

    def __getitem__(self, index: int | slice) -> SpilloverEffect | SpilloverEffects:
        if isinstance(index, slice):
            rows = self.rows[index]
            return replace(self, rows=rows, metadata=partial_metadata(self.metadata, rows, "slice"))
        return self.rows[index]

    def filter(self, predicate) -> SpilloverEffects:
        rows = tuple(row for row in self.rows if predicate(row))
        return replace(self, rows=rows, metadata=partial_metadata(self.metadata, rows, "filter"))

    def concat(self, other: object) -> SpilloverEffects:
        """Reassemble views of one capture without changing its registered error budget."""
        if not isinstance(other, SpilloverEffects):
            refuse_readout(
                "readout.collection.concat_type_mismatch",
                left=type(self).__name__,
                right=type(other).__name__,
            )
        left, right = self.metadata, other.metadata
        ids = tuple(sorted({left.scope.snapshot_id, right.scope.snapshot_id}))
        mismatches = tuple(
            name
            for name in type(left).model_fields
            if name not in {"partial", "partial_reason"}
            and getattr(left, name) != getattr(right, name)
        )
        if mismatches:
            refuse_readout(
                "readout.collection.concat_metadata_mismatch", snapshot_ids=ids, fields=mismatches
            )
        _, indexed = _merge_cell_identities(self, other, ids)
        family = left.scope.families[0]
        rows = tuple(
            row
            for cell in family.members
            if (row := indexed.get((family.source_snapshot_id, cell))) is not None
        )
        return replace(
            self,
            rows=rows,
            metadata=partial_metadata(left, rows, "slice"),
            _default_backend=(
                self._default_backend if self._default_backend == other._default_backend else None
            ),
        )

    def model_dump_json(self) -> str:
        return dump_collection(self)

    def __repr__(self) -> str:
        lines = [row._display() for row in self.rows]
        if self.metadata.partial:
            size = len(self.metadata.scope.families[0].members)
            lines.insert(0, f"Partial view: {len(self)} of {size} registered comparisons.")
        return "\n".join(lines)

    def to_frame(
        self,
        *,
        backend: Backend | None = None,
        columns: Sequence[str] | Literal["summary"] | None = None,
    ) -> IntoDataFrame:
        """Return signed absolute differences and their registered policy coordinates."""
        frame = to_frame(
            self.rows,
            model=SpilloverEffect,
            backend=backend or self._default_backend,
            columns=self._summary_columns if columns == "summary" else columns,
        )
        return with_view_partial(frame, partial=self.metadata.partial)


register_collection_types(SpilloverEffects)
