"""Exact sequential source snapshots and verified parent-linked record prefixes.

Capture consumes finalized joint-unit records once. It retains record identities
and hashes, not warehouse-sized Python outcome tables or rounded moments.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from fractions import Fraction
from typing import TYPE_CHECKING, Literal, NoReturn

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from increment.errors import CapabilityError, CodedError, CodedModel, RefusalSpec, refuse
from increment.semantics.rational import PortableRational
from increment.semantics.sequential import (
    _LEGACY_CONTINUATION,
    ADJUSTED_LAWS,
    PUBLIC_LAWS,
    SequentialCell,
    SequentialRegistration,
    SequentialSamplingModel,
    refuse_legacy_asymptotic_family,
    retained_dimension,
)

if TYPE_CHECKING:
    from increment.estimation._sequential_likelihood import BernoulliState, GaussianState

_REFUSALS = {
    code: RefusalSpec("sequential." + code, CapabilityError, lambda *, reason: reason)
    for code in (
        "source.invalid",
        "continuation.rewrite",
        "route.unsupported",
        "transform.unpredictable",
        "freeze.dropped",
        "freeze.invalid",
    )
}
_REFUSALS["continuation.legacy"] = _LEGACY_CONTINUATION


def sequential_refuse(code: str, reason: str) -> NoReturn:
    refuse(_REFUSALS[code], reason=reason)


def validate_sequential_transform(metric) -> None:
    """Admit only outcome transforms that are predictable when each unit is revealed.

    A fixed winsorization threshold clips each observation as it arrives, so the
    clipped value is an ordinary bounded scalar under every sequential law. A
    percentile threshold is computed from accumulated data: whenever it moves it
    re-clips every earlier observation, re-weighting past increments with
    information unavailable when they were revealed.
    """
    if metric.type == "quantile":
        sequential_refuse(
            "route.unsupported",
            "quantiles need a matching sequential sampling proof; use fixed-horizon inference "
            "(valid for one planned analysis, not repeated looks)",
        )
    winsorization = getattr(metric, "winsorization", None)
    if winsorization is not None and winsorization.has_percentile:
        sequential_refuse(
            "transform.unpredictable",
            f"metric {metric.name!r}: a percentile winsorization threshold is computed from "
            "accumulated data, so each time it moves it re-clips every earlier observation "
            "and re-weights past increments with information unavailable when they were "
            "revealed. Fix the threshold from pre-period data (lower_value/upper_value), "
            "after which it is a fixed threshold applied per unit as it arrives, or use "
            "fixed-horizon inference (valid for one planned analysis, not repeated looks)",
        )


def require_public_laws(models: Iterable, surface: str) -> None:
    """Refuse a public *surface* unless every registered law is publicly admitted.

    The exact NIG and NIW laws stay private research diagnostics: their
    validity requires the data to literally be Gaussian, which no declaration
    can certify; the asymptotic laws need only the declared moment contract.
    """
    if any(model.law not in PUBLIC_LAWS for model in models):
        sequential_refuse(
            "route.unsupported",
            f"public {surface} admits Bernoulli or registered asymptotic mean, adjusted "
            "mean and ratio observations",
        )


def model_adjustment(model) -> Literal["predeclared", "retained"] | None:
    """How one registered model adjusts for a covariate.

    ``predeclared`` retains a scalar transformed by a pre-period coefficient;
    ``retained`` retains the joint per-unit vector and fits the coefficient
    from accumulated cross moments. Both require an asymptotic mean law.
    """
    if getattr(model, "adjustment", None) is not None:
        return "predeclared"
    return "retained" if model.law in ADJUSTED_LAWS else None


def adjustment_kind(registration, metric: str) -> Literal["predeclared", "retained"] | None:
    """``model_adjustment`` of the model registered for *metric*, if any."""
    for model in registration.models:
        if model.metric == metric:
            return model_adjustment(model)
    return None


def validate_sequential_methods(registration, metric: str, methods, *, prior=None) -> None:
    """Admit raw outcomes, or the CUPED construction the registration retains.

    A coefficient fitted from accumulated in-experiment outcomes is admissible
    only where the registration retains joint (Y, X) moments and the route is
    the asymptotic one (Lindon, Ham, Tingley and Bojinov 2022, arXiv:2210.08589).
    A predeclared coefficient is supported by the asymptotic scalar-mean law,
    not by the exact Bernoulli route.
    """
    if prior is not None:
        sequential_refuse(
            "route.unsupported",
            "registered runtime supports predictive priors, not posterior effect priors",
        )
    adjustment = adjustment_kind(registration, metric)
    model = next((m for m in registration.models if m.metric == metric), None)
    for method in methods:
        if method.name == "unadjusted" and method.variance_reduction == "none":
            if adjustment is not None:
                sequential_refuse(
                    "source.invalid",
                    f"metric {metric!r}: the registration retains a CUPED adjustment "
                    f"({adjustment} coefficient), so its decision method must be "
                    "Method(name='cuped', variance_reduction='cuped')",
                )
            continue
        if method.name == "cuped" and method.variance_reduction == "cuped":
            if adjustment is not None:
                continue
            if model is not None and model.law in ("scalar_mean", "ratio_mean"):
                sequential_refuse(
                    "transform.unpredictable",
                    f"metric {metric!r}: a CUPED coefficient fitted from accumulated "
                    "in-experiment outcomes needs retained joint (Y, X) moments, which the "
                    f"registered {model.law!r} state does not carry. Register the "
                    f"{'adjusted_ratio_mean' if model.law == 'ratio_mean' else 'adjusted_mean'!r} "
                    "law instead (automatic under InferenceSpec(kind='asymptotic_mean') when "
                    "a CUPED method is declared with MetricSpec(covariate=...) on a frame or "
                    "with n_pre_periods > 0 on a definitions experiment). "
                    "For scalar_mean only, a pre-period coefficient and centre can instead "
                    "be declared through InferenceSpec.adjustments or ScalarMeanModel.adjustment "
                    "(see fit_predeclared_adjustment)",
                )
            sequential_refuse(
                "transform.unpredictable",
                f"metric {metric!r}: the exact Bernoulli e-process route does not admit "
                "CUPED, including predeclared coefficients. Use raw Bernoulli outcomes "
                "without CUPED, or monitor under InferenceSpec(kind='asymptotic_mean'), "
                "whose adjusted laws retain the joint (Y, X) moments the robust asymptotic "
                "construction of Lindon, Ham, Tingley and Bojinov (2022, arXiv:2210.08589) "
                "reads. Pre-period coefficients are an option on its scalar_mean law",
            )
        sequential_refuse(
            "route.unsupported",
            f"metric {metric!r}: registered runtime supports raw outcomes or a registered "
            f"CUPED adjustment, not method {method.name!r}",
        )


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def canonical_id(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def registration_id(registration: SequentialRegistration) -> str:
    return canonical_id(registration.model_dump(mode="json"))


class _State(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")


class SequentialArmState(_State):
    """Portable exact state. No rounding of an aggregate is accepted as capture."""

    metric: str
    group_id: str
    segment: tuple[tuple[str, str], ...] = ()
    law: Literal[
        "bernoulli",
        "gaussian",
        "gaussian_ratio",
        "scalar_mean",
        "adjusted_mean",
        "ratio_mean",
        "adjusted_ratio_mean",
    ]
    n: int = Field(ge=0)
    successes: int | None = None
    mean: tuple[PortableRational, ...] = ()
    scatter: tuple[tuple[PortableRational, ...], ...] = ()

    @model_validator(mode="after")
    def _valid(self):
        from increment.estimation._sequential_likelihood import BernoulliState, GaussianState

        if self.segment != tuple(sorted(set(self.segment))):
            sequential_refuse("source.invalid", "arm segment keys must be unique and canonical")
        if self.law == "bernoulli":
            if self.successes is None or self.mean or self.scatter:
                sequential_refuse("source.invalid", "Bernoulli state has incompatible fields")
            BernoulliState(self.n, self.successes)
        else:
            if self.successes is not None or len(self.mean) != retained_dimension(self.law):
                sequential_refuse("source.invalid", "Gaussian state has incompatible fields")
            GaussianState(self.n, self.mean, self.scatter)
        return self

    def kernel(self) -> BernoulliState | GaussianState:
        from increment.estimation._sequential_likelihood import BernoulliState, GaussianState

        if self.law == "bernoulli":
            assert self.successes is not None
            return BernoulliState(self.n, self.successes)
        return GaussianState(self.n, self.mean, self.scatter)


class UnitRecordProof(_State):
    unit_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class SequentialCheckpoint(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    version: Literal[2] = 2

    @model_validator(mode="before")
    @classmethod
    def _current_version(cls, value):
        if isinstance(value, Mapping) and value.get("version", 2) != 2:
            sequential_refuse(
                "continuation.legacy",
                "legacy sequential checkpoints cannot replay without the current observation mapping and compliance policy",
            )
        return value

    registration_id: str
    prefix_id: str
    filtration_id: str
    cell: SequentialCell
    model: SequentialSamplingModel
    control: SequentialArmState
    treatment: SequentialArmState
    revealed_units: int = Field(ge=0)
    status: Literal["current", "frozen", "missing"] = "current"

    @model_validator(mode="after")
    def _compatible(self):
        if (
            self.cell.metric != self.model.metric
            or any(
                s.metric != self.cell.metric
                or s.segment != self.cell.segment
                or s.law != self.model.law
                for s in (self.control, self.treatment)
            )
            or self.treatment.group_id != self.cell.group_id
        ):
            sequential_refuse(
                "source.invalid", "checkpoint model, state or hypothesis identity mismatch"
            )
        if self.status == "missing" and self.control.n and self.treatment.n:
            sequential_refuse("source.invalid", "observed cell cannot be marked missing")
        if self.control.group_id == self.treatment.group_id:
            sequential_refuse("source.invalid", "a checkpoint cannot compare an arm with itself")
        if self.control.n + self.treatment.n > self.revealed_units:
            sequential_refuse("source.invalid", "arm counts exceed the joint revealed prefix")
        return self

    def verify_snapshot(self, snapshot: SequentialSnapshot) -> None:
        """Bind a current or frozen cell to an actual verified source prefix."""
        registration = snapshot.registration
        if (
            self.registration_id != snapshot.registration_id
            or self.filtration_id != registration.reveal.filtration_id
            or self.cell not in registration.roster
            or self.model not in registration.models
            or self.control.group_id != registration.control_group
        ):
            sequential_refuse("source.invalid", "checkpoint registration or hypothesis changed")
        if self.prefix_id == snapshot.prefix_id:
            states, count = snapshot.states, len(snapshot.records)
        else:
            ancestor = next((a for a in snapshot.ancestors if a.prefix_id == self.prefix_id), None)
            if self.status != "frozen" or ancestor is None:
                sequential_refuse(
                    "continuation.rewrite",
                    "checkpoint is not current or a frozen verified ancestor",
                )
            states, count = ancestor.states, ancestor.n_records
        if (
            self.revealed_units != count
            or self.control not in states
            or self.treatment not in states
        ):
            sequential_refuse("source.invalid", "checkpoint state differs from its verified prefix")

    @property
    def checkpoint_id(self) -> str:
        return canonical_id(self.model_dump(mode="json"))


class SequentialAncestor(_State):
    prefix_id: str
    n_records: int = Field(ge=0)
    states: tuple[SequentialArmState, ...]
    frozen: tuple[SequentialCheckpoint, ...] = ()


class _PrefixContent:
    """Content digests and assignment counts at chosen record-prefix boundaries.

    A snapshot re-derives its own content digest and the digest of every ancestor
    prefix, and every ancestor prefix is a prefix of the same record tuple.
    ``canonical_id`` writes object keys in sorted order, so the canonical bytes of
    a prefix payload are a byte prefix of the whole payload's bytes up to the close
    of the ``records`` array. Serializing the records once in reveal order and
    copying the digest state at each boundary therefore replaces one full
    re-serialization of the prefix per ancestor without moving a single digest
    byte: capture still derives ``prefix_id`` from ``canonical_id`` over the
    assembled payload, so a divergence between the two spellings refuses every
    capture instead of quietly rewriting a stored identity.

    Records are serialized one whole span at a time rather than one at a time,
    because a list's canonical bytes are its elements' bytes joined by the same
    separator. An ancestry-free prefix therefore costs exactly one serialization,
    as it did before boundaries existed.
    """

    __slots__ = (
        "_after_states_empty",
        "_after_states_frozen",
        "_before_states",
        "_counts",
        "_digests",
    )

    def __init__(
        self,
        version: int,
        registration_digest: str,
        records: tuple[UnitRecordProof, ...],
        boundaries: Iterable[int] = (),
    ) -> None:
        self._before_states = (
            b'],"registration_id":' + _canonical_bytes(registration_digest) + b',"states":['
        )
        # Frozen cells are an optional field in the digest, like elsewhere in this
        # wire format: present only when non-empty, so every snapshot captured
        # before freezing existed keeps its released identity unchanged.
        self._after_states_empty = b'],"version":' + _canonical_bytes(version) + b"}"
        self._after_states_frozen = b'],"version":' + _canonical_bytes(version) + b',"zzz_frozen":['
        total = len(records)
        hasher = hashlib.sha256(b'{"records":[')
        counts: dict[str, int] = {}
        self._digests = {}
        self._counts = {}
        start = 0
        for cut in sorted({0, total, *(n for n in boundaries if 0 <= n <= total)}):
            span = records[start:cut]
            if span:
                if start:
                    hasher.update(b",")
                # Strip the array brackets: what remains is exactly the element
                # bytes this span contributes to the whole "records" array.
                hasher.update(_canonical_bytes([r.model_dump(mode="json") for r in span])[1:-1])
                for record in span:
                    counts[record.group_id] = counts.get(record.group_id, 0) + 1
            self._digests[cut] = hasher.copy()
            self._counts[cut] = dict(counts)
            start = cut

    def digest(
        self,
        n_records: int,
        states: tuple[SequentialArmState, ...],
        frozen: Iterable[SequentialCheckpoint] = (),
    ) -> str:
        """Content digest of the first ``n_records`` records, closed over states and frozen.

        ``frozen`` is folded in only when non-empty, so an unfrozen snapshot's
        digest matches the released value from before freezing existed.
        """
        hasher = self._digests[n_records].copy()
        hasher.update(self._before_states)
        hasher.update(_canonical_bytes([s.model_dump(mode="json") for s in states])[1:-1])
        ordered = sorted(
            frozen, key=lambda c: (c.cell.metric, c.cell.group_id, c.cell.estimand, c.cell.segment)
        )
        if not ordered:
            hasher.update(self._after_states_empty)
        else:
            hasher.update(self._after_states_frozen)
            hasher.update(_canonical_bytes([c.checkpoint_id for c in ordered])[1:-1])
            hasher.update(b"]}")
        return hasher.hexdigest()

    def counts(self, n_records: int) -> dict[str, int]:
        """Units revealed per assignment within the first ``n_records`` records."""
        return self._counts[n_records]


class SequentialSnapshot(_State):
    version: Literal[2] = 2

    @model_validator(mode="before")
    @classmethod
    def _current_version(cls, value):
        if isinstance(value, Mapping):
            if value.get("version", 2) != 2:
                sequential_refuse(
                    "continuation.legacy",
                    "legacy sequential snapshots cannot continue without the current observation mapping and compliance policy",
                )
            refuse_legacy_asymptotic_family(value.get("registration"))
        return value

    registration: SequentialRegistration
    registration_id: str
    parent_id: str | None = None
    ancestors: tuple[SequentialAncestor, ...] = ()
    prefix_id: str
    records: tuple[UnitRecordProof, ...]
    states: tuple[SequentialArmState, ...]
    finalized: Literal[True]
    reveal_cursor: date | datetime | str | int | float | None = None
    frozen: tuple[SequentialCheckpoint, ...] = ()

    @field_serializer("reveal_cursor", when_used="json")
    def _serialize_cursor(self, value):
        if isinstance(value, datetime):
            return {"datetime": value.isoformat()}
        if isinstance(value, str):
            return {"label": value}
        return value

    @field_validator("reveal_cursor", mode="before")
    @classmethod
    def _deserialize_cursor(cls, value, info: ValidationInfo):
        if isinstance(value, Mapping):
            if set(value) == {"label"} and isinstance(value["label"], str):
                return value["label"]
            if set(value) == {"datetime"} and isinstance(value["datetime"], str):
                return datetime.fromisoformat(value["datetime"])
        if info.mode == "json" and isinstance(value, str):
            return date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def _identities(self):
        if (self.ancestors[-1].prefix_id if self.ancestors else None) != self.parent_id:
            sequential_refuse("source.invalid", "parent-linked ancestry is inconsistent")
        if self.registration_id != registration_id(self.registration):
            sequential_refuse("source.invalid", "registration digest mismatch")
        if len({r.unit_id for r in self.records}) != len(self.records):
            sequential_refuse("source.invalid", "duplicate unit identity in prefix")
        keys = [(s.metric, s.group_id, s.segment) for s in self.states]
        if keys != sorted(set(keys)):
            sequential_refuse("source.invalid", "states must have unique canonical keys")
        content = _PrefixContent(
            self.version,
            self.registration_id,
            self.records,
            (ancestor.n_records for ancestor in self.ancestors),
        )
        if self.prefix_id != content.digest(len(self.records), self.states, self.frozen):
            sequential_refuse("source.invalid", "snapshot content digest mismatch")
        chain_n = [a.n_records for a in self.ancestors] + [len(self.records)]
        chain_frozen = [a.frozen for a in self.ancestors] + [self.frozen]
        chain_states = [a.states for a in self.ancestors] + [self.states]
        chain_prefix = [a.prefix_id for a in self.ancestors] + [self.prefix_id]
        if chain_frozen[0]:
            sequential_refuse(
                "source.invalid", "the earliest chain entry cannot already have a frozen cell"
            )
        for i in range(1, len(chain_n)):
            prev_n, cur_n = chain_n[i - 1], chain_n[i]
            prev_frozen, cur_frozen = chain_frozen[i - 1], chain_frozen[i]
            if cur_n == prev_n:
                if chain_states[i - 1] != chain_states[i]:
                    sequential_refuse(
                        "source.invalid", "a same-count chain entry must retain identical states"
                    )
                if len({c.cell for c in cur_frozen}) != len(cur_frozen):
                    sequential_refuse(
                        "source.invalid", "a cell cannot be frozen twice in one snapshot"
                    )
                prev_by_cell = {c.cell: c for c in prev_frozen}
                cur_by_cell = {c.cell: c for c in cur_frozen}
                if set(prev_by_cell) - set(cur_by_cell) or any(
                    cur_by_cell[cell] != checkpoint for cell, checkpoint in prev_by_cell.items()
                ):
                    sequential_refuse(
                        "source.invalid",
                        "a same-count chain entry must retain every prior frozen cell unchanged",
                    )
                introduced = [
                    checkpoint
                    for cell, checkpoint in cur_by_cell.items()
                    if cell not in prev_by_cell
                ]
                if not introduced:
                    sequential_refuse(
                        "source.invalid",
                        "a same-count chain entry must strictly extend the prior freeze",
                    )
                if any(
                    c.prefix_id != chain_prefix[i - 1] or c.revealed_units != cur_n
                    for c in introduced
                ):
                    sequential_refuse(
                        "source.invalid",
                        "a newly frozen cell must be declared at its own look's own prefix",
                    )
            elif prev_n < cur_n:
                if cur_frozen != prev_frozen:
                    sequential_refuse(
                        "source.invalid",
                        "a frozen cell can only be added at a same-count declaration step",
                    )
            else:
                sequential_refuse("source.invalid", "ancestor prefixes must increase")
            if i <= len(self.ancestors):
                ancestor = self.ancestors[i - 1]
                if (
                    content.digest(ancestor.n_records, ancestor.states, ancestor.frozen)
                    != ancestor.prefix_id
                ):
                    sequential_refuse(
                        "source.invalid", "ancestor does not bind an unchanged record prefix"
                    )
        if len({c.cell for c in self.frozen}) != len(self.frozen):
            sequential_refuse("source.invalid", "a cell cannot be frozen twice in one snapshot")
        for checkpoint in self.frozen:
            checkpoint.verify_snapshot(self)
        expectation = _retained_expectation(self.registration)
        _validate_retained_states(expectation, self.states, content.counts(len(self.records)))
        for ancestor in self.ancestors:
            _validate_retained_states(
                expectation, ancestor.states, content.counts(ancestor.n_records)
            )
        return self

    def content_id(self) -> str:
        return _PrefixContent(self.version, self.registration_id, self.records).digest(
            len(self.records), self.states
        )

    def arm(self, metric: str, group_id: str, segment=()) -> SequentialArmState:
        for state in self.states:
            if (state.metric, state.group_id, state.segment) == (metric, group_id, tuple(segment)):
                return state
        sequential_refuse("source.invalid", "requested arm is outside the registered roster")

    def verify_parent(self, previous: SequentialSnapshot) -> None:
        """Prove this snapshot continues ``previous``: an append of new units or a
        freeze declared at the same look, through any recorded intermediate looks."""
        if self.registration_id != previous.registration_id:
            sequential_refuse(
                "continuation.rewrite", "source, model, prior, roster or definitions changed"
            )
        if self.prefix_id == previous.prefix_id:
            return
        if (
            len(self.records) < len(previous.records)
            or self.records[: len(previous.records)] != previous.records
        ):
            sequential_refuse(
                "continuation.rewrite", "append must preserve every prior finalized record"
            )
        link = (
            *previous.ancestors,
            SequentialAncestor(
                prefix_id=previous.prefix_id,
                n_records=len(previous.records),
                states=previous.states,
                frozen=previous.frozen,
            ),
        )
        if self.ancestors[: len(link)] != link:
            sequential_refuse("continuation.rewrite", "parent state or ancestry changed")
        prev_by_cell = {c.cell: c for c in previous.frozen}
        cur_by_cell = {c.cell: c for c in self.frozen}
        if set(prev_by_cell) - set(cur_by_cell) or any(
            cur_by_cell[cell] != checkpoint for cell, checkpoint in prev_by_cell.items()
        ):
            sequential_refuse(
                "freeze.dropped",
                "a previously frozen cell must stay frozen at every later look",
            )
        if len(self.records) == len(previous.records) and (
            self.states != previous.states or len(cur_by_cell) == len(prev_by_cell)
        ):
            sequential_refuse(
                "continuation.rewrite", "changed prefix contains no new units or freeze"
            )


def _retained_expectation(registration):
    """Roster keys and declared laws every retained state tuple must still match."""
    return _state_keys(registration), {m.metric: m.law for m in registration.models}


def _validate_retained_states(expectation, states, counts):
    """Check one prefix's retained roster, sampling laws and assignment counts.

    ``counts`` holds the units revealed per assignment in that prefix; the caller
    accumulates it in reveal order rather than recounting the prefix per ancestor.
    """
    roster_keys, laws = expectation
    keys = [(s.metric, s.group_id, s.segment) for s in states]
    if keys != roster_keys:
        sequential_refuse("source.invalid", "snapshot state roster differs from registration")
    if any(s.law != laws[s.metric] for s in states):
        sequential_refuse("source.invalid", "snapshot sampling model changed")
    for state in states:
        count = counts.get(state.group_id, 0)
        if state.n > count or (not state.segment and state.n != count):
            sequential_refuse("source.invalid", "state count differs from retained assignments")


def _state_keys(registration):
    return sorted(
        {
            (c.metric, arm, c.segment)
            for c in registration.roster
            for arm in (registration.control_group, c.group_id)
        }
    )


def _observation(value: object) -> Fraction:
    if isinstance(value, bool):
        return Fraction(int(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            sequential_refuse("source.invalid", "finalized observations must be finite")
        return Fraction(*value.as_integer_ratio())
    if isinstance(value, (int, Fraction)):
        return Fraction(value)
    sequential_refuse(
        "source.invalid", "observations must be exact integers, rationals or binary64"
    )


def _capture_registration(registration, source_id, definitions_id, finalized, previous, append):
    if not isinstance(registration, SequentialRegistration):
        sequential_refuse("source.invalid", "a pre-data registration is required")
    if not finalized or (source_id, definitions_id) != (
        registration.source_id,
        registration.definitions_id,
    ):
        sequential_refuse(
            "source.invalid", "source definitions or explicit finalization do not match"
        )
    rid = registration_id(registration)
    if previous is not None and previous.registration_id != rid:
        sequential_refuse("continuation.rewrite", "registration changed before source access")
    if append and previous is None:
        sequential_refuse(
            "source.invalid", "append capture requires an immutable parent checkpoint"
        )
    return rid


def _capture_sequential_diagnostic_snapshot(  # noqa: PLR0915
    registration: SequentialRegistration,
    records: Iterable[Mapping[str, object]],
    *,
    source_id: str,
    definitions_id: str,
    finalized: bool,
    previous: SequentialSnapshot | None = None,
    reveal_cursor: date | datetime | str | int | float | None = None,
    append: bool = False,
) -> SequentialSnapshot:
    """Capture a full prefix in committed reveal order, checking unchanged records.

    Each record contains canonical string unit_id/group_id, values mapping metric
    names to scalar or paired observations, and optional segment labels. Callers
    certify joint finalization; freshness/data_as_of is never substituted for it.
    Registration and continuation identity are checked before iterating records.
    With append=True, records contain only new units and previous is required;
    their proofs extend the immutable parent without rereading its observations.
    """
    from increment.estimation._sequential_likelihood import BernoulliState, GaussianState

    rid = _capture_registration(
        registration, source_id, definitions_id, finalized, previous, append
    )
    models = {m.metric: m for m in registration.models}
    states = (
        {
            key: BernoulliState(0, 0)
            if models[key[0]].law == "bernoulli"
            else GaussianState.empty(retained_dimension(models[key[0]].law))
            for key in _state_keys(registration)
        }
        if previous is None
        else {(s.metric, s.group_id, s.segment): s.kernel() for s in previous.states}
    )
    proofs = list(previous.records) if append and previous is not None else []
    seen = {p.unit_id for p in proofs}
    registered_groups = {
        registration.control_group,
        *(cell.group_id for cell in registration.roster),
    }
    required_dimensions = {k for cell in registration.roster for k, _ in cell.segment}
    for record in records:
        unit, group = record.get("unit_id"), record.get("group_id")
        if not isinstance(unit, str) or not unit or not isinstance(group, str) or not group:
            sequential_refuse(
                "source.invalid", "unit and assignment identities must be canonical strings"
            )
        if group not in registered_groups:
            sequential_refuse(
                "source.invalid",
                "record assignment is outside the complete registered arm roster",
            )
        if unit in seen:
            sequential_refuse("source.invalid", "a unit appears more than once in a joint prefix")
        seen.add(unit)
        raw_values = record.get("values")
        if not isinstance(raw_values, Mapping):
            sequential_refuse("source.invalid", "every relevant metric must be revealed together")
        values: dict[object, object] = dict(raw_values.items())
        if set(values) != set(models):
            sequential_refuse("source.invalid", "every relevant metric must be revealed together")
        segments = record.get("segments", {})
        if not isinstance(segments, Mapping) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in segments.items()
        ):
            sequential_refuse("source.invalid", "segment identities must be canonical strings")
        if not required_dimensions.issubset(segments):
            sequential_refuse("source.invalid", "record is missing a registered segment dimension")
        exact = {}
        for metric, model in models.items():
            raw = values[metric]
            vector = tuple(raw) if isinstance(raw, (tuple, list)) else (raw,)
            vector = tuple(_observation(v) for v in vector)
            if len(vector) != retained_dimension(model.law):
                sequential_refuse(
                    "source.invalid",
                    f"metric {metric!r}: {model.law} reveals a "
                    f"{retained_dimension(model.law)}-coordinate joint observation per unit",
                )
            if model.law == "bernoulli" and vector[0] not in (0, 1):
                sequential_refuse("source.invalid", "Bernoulli observation is not binary")
            exact[metric] = vector
        proof = UnitRecordProof(
            unit_id=unit,
            group_id=group,
            digest=canonical_id(
                {
                    "unit": unit,
                    "assignment": group,
                    "segments": sorted(segments.items()),
                    "values": [(m, [str(v) for v in exact[m]]) for m in sorted(exact)],
                    "definitions": definitions_id,
                    "source_identity": record.get("source_identity", {}),
                }
            ),
        )
        position = len(proofs)
        if (
            previous is not None
            and position < len(previous.records)
            and proof != previous.records[position]
        ):
            sequential_refuse(
                "continuation.rewrite",
                "prior unit identity, value, assignment or reveal order changed",
            )
        proofs.append(proof)
        if previous is not None and position < len(previous.records):
            continue
        for (metric, arm, segment), state in states.items():
            if arm != group or any(segments.get(k) != v for k, v in segment):
                continue
            if isinstance(state, BernoulliState):
                states[metric, arm, segment] = BernoulliState(
                    state.n + 1, state.successes + int(exact[metric][0])
                )
            else:
                states[metric, arm, segment] = state.merge(
                    GaussianState.from_rows((exact[metric],), dimension=len(state.mean))
                )
    if previous is not None and len(proofs) < len(previous.records):
        sequential_refuse("continuation.rewrite", "prefix removed finalized units")
    portable = tuple(
        SequentialArmState(
            metric=metric,
            group_id=group,
            segment=segment,
            law=models[metric].law,
            n=state.n,
            successes=state.successes if isinstance(state, BernoulliState) else None,
            mean=state.mean if isinstance(state, GaussianState) else (),
            scatter=state.scatter if isinstance(state, GaussianState) else (),
        )
        for (metric, group, segment), state in sorted(states.items())
    )
    frozen = previous.frozen if previous is not None else ()
    content = {
        "version": 2,
        "registration_id": rid,
        "records": [p.model_dump(mode="json") for p in proofs],
        "states": [s.model_dump(mode="json") for s in portable],
    }
    if frozen:
        content["zzz_frozen"] = [
            c.checkpoint_id
            for c in sorted(
                frozen,
                key=lambda c: (c.cell.metric, c.cell.group_id, c.cell.estimand, c.cell.segment),
            )
        ]
    prefix = canonical_id(content)
    if previous is not None and prefix == previous.prefix_id:
        if reveal_cursor is None:
            return previous
        return previous.model_copy(update={"reveal_cursor": reveal_cursor})
    snapshot = SequentialSnapshot(
        registration=registration,
        registration_id=rid,
        prefix_id=prefix,
        parent_id=previous.prefix_id if previous is not None else None,
        ancestors=(
            *previous.ancestors,
            SequentialAncestor(
                prefix_id=previous.prefix_id,
                n_records=len(previous.records),
                states=previous.states,
                frozen=previous.frozen,
            ),
        )
        if previous is not None
        else (),
        records=tuple(proofs),
        states=portable,
        finalized=True,
        reveal_cursor=reveal_cursor,
        frozen=frozen,
    )
    if previous is not None:
        snapshot.verify_parent(previous)
    return snapshot


def capture_sequential_snapshot(
    registration: SequentialRegistration,
    records: Iterable[Mapping[str, object]],
    *,
    source_id: str,
    definitions_id: str,
    finalized: bool,
    previous: SequentialSnapshot | None = None,
    reveal_cursor: date | datetime | str | int | float | None = None,
    append: bool = False,
) -> SequentialSnapshot:
    if isinstance(registration, SequentialRegistration):
        registration = SequentialRegistration.model_validate(registration)
        require_public_laws(registration.models, "anytime validity")
    return _capture_sequential_diagnostic_snapshot(
        registration,
        records,
        source_id=source_id,
        definitions_id=definitions_id,
        finalized=finalized,
        previous=previous,
        reveal_cursor=reveal_cursor,
        append=append,
    )


def declare_sequential_freeze_cells(
    snapshot: SequentialSnapshot, cells: Sequence[SequentialCell]
) -> SequentialSnapshot:
    """Freeze exact roster cells at exactly this snapshot's current look.

    Adds a real ancestor at the pre-freeze prefix (the snapshot's own current
    prefix_id before this call), so SequentialCheckpoint.verify_snapshot's
    existing "current or frozen ancestor" rule accepts every frozen entry at
    every later look. There is no external checkpoint argument, so a caller
    cannot supply an earlier look's evidence -- only THIS snapshot's own
    current state can ever be frozen. Refuses a cell outside the roster, one
    already frozen, or one with no observations on either arm yet.
    """
    if not cells:
        return snapshot
    registration = snapshot.registration
    roster = set(registration.roster)
    unknown = [cell for cell in cells if cell not in roster]
    if unknown:
        sequential_refuse(
            "source.invalid", f"freeze names cells outside the registered roster: {unknown}"
        )
    already = {c.cell for c in snapshot.frozen}
    duplicate = [cell for cell in cells if cell in already]
    if duplicate:
        sequential_refuse("source.invalid", f"already frozen: {duplicate}")
    models = {m.metric: m for m in registration.models}
    new_checkpoints = []
    for cell in cells:
        control = snapshot.arm(cell.metric, registration.control_group, cell.segment)
        treatment = snapshot.arm(cell.metric, cell.group_id, cell.segment)
        if not control.n or not treatment.n:
            sequential_refuse(
                "source.invalid",
                f"cell {cell!r} has no observations on both arms yet; freezing an undecided "
                "cell keeps no decision",
            )
        new_checkpoints.append(
            SequentialCheckpoint(
                registration_id=snapshot.registration_id,
                prefix_id=snapshot.prefix_id,
                filtration_id=registration.reveal.filtration_id,
                cell=cell,
                model=models[cell.metric],
                control=control,
                treatment=treatment,
                revealed_units=len(snapshot.records),
                status="frozen",
            )
        )
    frozen = tuple(
        sorted(
            (*snapshot.frozen, *new_checkpoints),
            key=lambda c: (c.cell.metric, c.cell.group_id, c.cell.estimand, c.cell.segment),
        )
    )
    content = _PrefixContent(
        snapshot.version,
        snapshot.registration_id,
        snapshot.records,
        (a.n_records for a in snapshot.ancestors),
    )
    return snapshot.model_copy(
        update={
            "frozen": frozen,
            "parent_id": snapshot.prefix_id,
            "ancestors": (
                *snapshot.ancestors,
                SequentialAncestor(
                    prefix_id=snapshot.prefix_id,
                    n_records=len(snapshot.records),
                    states=snapshot.states,
                    frozen=snapshot.frozen,
                ),
            ),
            "prefix_id": content.digest(len(snapshot.records), snapshot.states, frozen),
        }
    )


def declare_sequential_freeze(
    snapshot: SequentialSnapshot, metrics: Sequence[str]
) -> SequentialSnapshot:
    """Stop monitoring named metrics at exactly this snapshot's current look.

    Every registered cell of a named metric -- each treatment arm and each
    segment -- that has observations on both arms keeps this look's evidence
    at every later look. A cell with no observations yet has no evidence to
    keep and stays monitored; naming the metric at a later capture freezes it
    then. Only this snapshot's own current state can be frozen, never an
    earlier look's. A frozen secondary keeps its evidence, while its family's
    e-BH selection is redone at every look over frozen and current evidence.
    """
    registration = snapshot.registration
    by_metric: dict[str, list[SequentialCell]] = {}
    for cell in registration.roster:
        by_metric.setdefault(cell.metric, []).append(cell)
    unknown = sorted(set(metrics) - set(by_metric))
    if unknown:
        sequential_refuse(
            "freeze.invalid",
            f"freeze names metrics outside the registration: {unknown}; "
            f"the registered metrics are {sorted(by_metric)}",
        )
    already = {c.cell for c in snapshot.frozen}
    control = registration.control_group
    cells, idle = [], []
    for name in dict.fromkeys(metrics):
        ready = [
            cell
            for cell in by_metric[name]
            if cell not in already
            and snapshot.arm(cell.metric, control, cell.segment).n
            and snapshot.arm(cell.metric, cell.group_id, cell.segment).n
        ]
        if not ready:
            idle.append(name)
        cells.extend(ready)
    if idle:
        sequential_refuse(
            "freeze.invalid",
            f"nothing left to freeze for {idle}: every registered cell is already frozen or "
            "has no observations on both arms yet; freeze at a later capture once it has data",
        )
    return declare_sequential_freeze_cells(snapshot, cells)


def snapshot_from_json(payload: str) -> SequentialSnapshot:
    """Reject duplicate JSON keys and malformed text before validating a current checkpoint.

    Every rational in the payload is admitted by ``PortableRational`` before
    conversion. JSON syntax errors and integer literals beyond the reader's
    digit limit refuse with ``sequential.source.invalid``.
    """

    def pairs(entries):
        result = {}
        for key, value in entries:
            if key in result:
                sequential_refuse("source.invalid", "duplicate JSON key")
            result[key] = value
        return result

    try:
        data = json.loads(payload, object_pairs_hook=pairs)
    except CodedError:
        raise
    except ValueError:
        sequential_refuse("source.invalid", "checkpoint payload is not well-formed JSON")
    if not isinstance(data, dict) or data.get("version") != 2:
        sequential_refuse(
            "continuation.legacy", "legacy presentation/moments payload cannot resume a process"
        )
    return SequentialSnapshot.model_validate_json(payload)
