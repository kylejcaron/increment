"""Private field evaluation for the fraction-harmed study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Literal, TextIO, cast

import numpy as np
import pydantic
import scipy
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, field_validator

from calibration.harm_bounds import (
    Band,
    Cell,
    Counts4,
    Decision,
    DiagnosticContext,
    Interval,
    LeafMap,
    StudyError,
    decide,
    learn,
    summarize,
)

_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = _ROOT / "calibration/protocols/harm_study_v1.md"
_PROTOCOL_BASE = "77ab667b227b2005e18bcc3d0f56c23b6aaec651"
_SEED = 20261119083
_HILLSTROM_N = 42_613
_TOL = 1e-12
_RISK = 0.05
_THRESHOLDS = (0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.32, 0.50)


type _Source = Literal["hillstrom", "owner_trial"]
type _Status = Literal["unavailable", "research_only", "completed", "failed"]
type _Row = tuple[int, int, int, int]  # assignment, factual result, S, B
type _ActionKind = Literal["clear", "rule_out", "abstain"]
type _ActionKey = Literal[
    "action_if_clear", "action_if_rule_out", "action_if_inconclusive", "current_action"
]
type _CostBearer = Literal["sponsor", "participants", "sponsor_and_participants", "third_parties"]
type _Direction = Literal["one_is_beneficial"]
type _Probability = Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]
type _Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
type _Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
type _PositiveMoney = Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]
type _NonnegativeMoney = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
type _Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
type _Sha = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
type _Regime = Literal["below", "equal", "above"]

_FIXED_MAPS: tuple[LeafMap, LeafMap, LeafMap] = ((0, 0, 0, 0), (0, 0, 1, 1), (0, 1, 2, 3))
_PREREQUISITES = {
    "U0": ("assignment", "binary_outcome"),
    "F": ("assignment", "binary_outcome", "baseline_S"),
    "Q": ("assignment", "binary_outcome", "baseline_S", "baseline_B"),
    "H": ("assignment", "binary_outcome", "baseline_S", "baseline_B", "honest_split"),
    "M": ("assignment", "binary_outcome", "baseline_S", "honest_split"),
}


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class _CaseInput(_Strict):
    csv_path: str
    expected_sha256: str
    permission_ref: str
    design_ref: str
    followup_ref: str
    unit_ref: str
    sampling_target: str
    analysis_permitted: bool
    aggregates_permitted: bool


class _TrialDesign(_Strict):
    arms: tuple[_Text, _Text]
    outcome: _Text
    horizon: _Text
    positive_direction: _Direction
    assignment: Literal["bernoulli", "complete"]
    treatment_probability: _Probability
    assumptions_ref: _Text


class _Action(_Strict):
    kind: _ActionKind
    description: _Text
    cost_bearer: _CostBearer


class _ActionReceipt(_Strict):
    key: _ActionKey
    kind: _ActionKind
    cost_bearer: _CostBearer


class _ExactCosts(_Strict):
    cost_false_clear: _PositiveMoney
    cost_false_rule_out: _PositiveMoney
    cost_abstain: _NonnegativeMoney
    incremental_workflow_cost: _NonnegativeMoney
    minimum_gain: _NonnegativeMoney

    @field_validator(
        "cost_false_clear",
        "cost_false_rule_out",
        "cost_abstain",
        "incremental_workflow_cost",
        "minimum_gain",
    )
    @classmethod
    def _reportable_money(cls, value: Decimal) -> Decimal:
        rounded = float(value)
        if not math.isfinite(rounded) or (value != 0 and rounded == 0.0):
            raise ValueError("cost is outside the numeric report range")
        return value


class DecisionCard(_ExactCosts):
    """Prospective decision, matched to independently documented trial semantics."""

    owner: _Text
    decision_id: _Text
    signed_at: AwareDatetime
    unblind_at: AwareDatetime
    target: _Text
    arms: tuple[_Text, _Text]
    outcome: _Text
    horizon: _Text
    positive_direction: _Direction
    tau: _Probability
    max_false_action_risk: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
    currency_or_loss_unit: _Text
    action_if_clear: _Action
    action_if_rule_out: _Action
    action_if_inconclusive: _Action
    current_action: _Action
    representativeness_evidence: _Text
    permission_ref: _Text

    @field_validator("signed_at", "unblind_at", mode="before")
    @classmethod
    def _explicit_datetime(cls, value: object) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and "T" in value:
            return datetime.fromisoformat(value)
        raise ValueError("an ISO 8601 timestamp with a time zone is required")


class _Diagnostic(_Strict):
    reason: str
    context: DiagnosticContext


class _CardCheck(_Strict):
    eligible: DecisionCard | None = None
    sha256: _Sha | None = None
    sensitivity_tau: _Probability | None = None
    issue: _Diagnostic | None = None


class _InputReceipt(_Strict):
    schema_version: Literal[2] = 2
    source: _Source
    manifest_sha256: _Sha | None = None
    design_sha256: _Sha | None = None
    card_sha256: _Sha | None = None
    csv_sha256: _Sha | None = None
    expected_csv_sha256: _Sha | None = None
    permission_verified: bool = False
    design_verified: bool = False
    digest_verified: bool = False
    assignment: Literal["bernoulli", "complete", "reported_equal_allocation"] | None = None
    treatment_probability: _Probability | None = None


class _Observed(_Strict):
    n0: int
    y0: int
    n1: int
    y1: int
    mean_difference: float | None


class _Split(_Strict):
    training: int
    estimation: int


class _Method(_Strict):
    plugin: Interval | None
    L_ci: Interval
    U_ci: Interval
    outer: Interval
    reason: str | None
    leaf_map: LeafMap
    leaf_counts: tuple[Cell, ...]
    weight_mode: Literal["estimated"]
    outer_width: float
    inner_span: Interval | None
    prerequisites: tuple[str, ...]


class _Methods(_Strict):
    U0: _Method
    F: _Method
    Q: _Method
    H: _Method
    M: _Method


class _Pipeline(_Strict):
    methods: tuple[str, ...]
    outer: Interval | None
    plugin: Interval | None
    plugin_reason: str | None
    reason: Literal["discordant_bands"] | None
    outer_width: float | None
    prerequisites: tuple[str, ...]


class _Pipelines[T](_Strict):
    P0: T
    P1: T
    P2: T
    P3: T


class _Bounds(_Strict):
    observed: _Observed
    split: _Split
    methods: _Methods
    pipelines: _Pipelines[_Pipeline]
    historical_benchmark: bool
    limitations: tuple[str, ...]


class _BoundsPacket(_Strict):
    schema_version: Literal[2] = 2
    protocol_sha256: _Sha
    status: _Status
    reason: str | None
    bounds: _Bounds | None


class _Guessing(_Strict):
    clear: float
    rule_out: float


class _PipelineDecision(_Strict):
    decision: Decision
    action: _ActionReceipt
    changes_action: bool
    g_min: float
    g_max: float
    gain_exceeds_minimum: bool
    best_gain_exceeds_minimum: bool
    ambiguity: Literal["irreducible_at_features", "sampling_or_unresolved_identification"] | None
    guess_vs_wait_worst_excess: _Guessing | None


class _OwnerDecision(_Strict):
    tau: _Probability
    minimum_gain: _Nonnegative
    pipelines: _Pipelines[_PipelineDecision]
    smallest_useful_simple_pipeline: Literal["P0", "P1", "P2"] | None
    h_adds_decision: bool
    irreducible_at_features: bool
    current_action: _ActionReceipt
    incremental_workflow_cost: _Nonnegative
    cost_false_clear: _Positive
    cost_false_rule_out: _Positive
    cost_abstain: _Nonnegative
    exact_costs: _ExactCosts
    max_false_action_risk: float
    alternatives: tuple[_ActionReceipt, _ActionReceipt, _ActionReceipt]
    confirmation: None = None


class _Receipts(_Strict):
    inputs: _Sha = Field(serialization_alias="input_manifest.json")
    bounds: _Sha = Field(serialization_alias="bounds.json")
    thresholds: _Sha = Field(serialization_alias="thresholds.csv")


class _Reproduction(_Strict):
    source_revision: str
    protocol_base_revision: str
    harm_case_sha256: _Sha
    harm_bounds_sha256: _Sha
    python: str
    numpy: str
    scipy: str
    pydantic: str
    argv: tuple[str, ...]
    exact_arguments_sha256: _Sha
    rng: Literal["PCG64"] = "PCG64"
    seed: Literal[20261119083] = _SEED
    started_at: AwareDatetime
    ended_at: AwareDatetime


class _FieldDecision(_Strict):
    schema_version: Literal[2] = 2
    source: _Source
    status: _Status
    reason: str | None
    protocol_sha256: _Sha
    manifest_sha256: _Sha | None
    card_sha256: _Sha | None
    permission_verified: bool
    design_verified: bool
    bounds: _Pipelines[Interval | None] | None
    owner_decision: _OwnerDecision | None
    diagnostics: tuple[_Diagnostic, ...]
    receipts: _Receipts
    reproduction: _Reproduction


def _issue(reason: str, stage: str, coordinate: str, **details: str | int) -> StudyError:
    return StudyError(reason, (("stage", stage), ("coordinate", coordinate), *details.items()))


def _diagnostic(error: StudyError) -> _Diagnostic:
    return _Diagnostic(reason=error.reason, context=error.context)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_members(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON member")
        result[key] = value
    return result


def _validate[T: _Strict](model: type[T], raw: bytes, reason: str, stage: str) -> T:
    try:
        if model is DecisionCard:
            document = json.loads(
                raw, object_pairs_hook=_unique_members, parse_float=Decimal, parse_int=Decimal
            )
            if isinstance(document, dict) and isinstance(document.get("arms"), list):
                document["arms"] = tuple(document["arms"])
            return cast(T, DecisionCard.model_validate(document))
        json.loads(raw, object_pairs_hook=_unique_members)
        return model.model_validate_json(raw)
    except ValidationError as exc:
        field = exc.errors(include_input=False)[0]["loc"]
        allowed = set(model.model_fields)
        coordinate = str(field[0]) if field and field[0] in allowed else "document"
        raise _issue(reason, stage, coordinate) from None
    except (ValueError, RecursionError):
        raise _issue(reason, stage, "document") from None


def _local_path(value: str, reason: str, stage: str, coordinate: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or "\0" in value:
        raise _issue(reason, stage, coordinate)
    return path


def _read(path: Path, missing: str, failure: str, stage: str, coordinate: str) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        raise _issue(missing, stage, coordinate) from None
    except OSError:
        raise _issue(failure, stage, coordinate) from None


def _check_permission(manifest: _CaseInput) -> None:
    for key in ("permission_ref", "analysis_permitted", "aggregates_permitted"):
        value = getattr(manifest, key)
        if not (value.strip() if isinstance(value, str) else value):
            raise _issue("use_permission_unconfirmed", "preflight", key)


_UNBLIND_PENDING = "card_unblinding_not_reached"


def _check_card(
    card: DecisionCard, manifest: _CaseInput, design: _TrialDesign, started: datetime
) -> None:
    if card.signed_at >= card.unblind_at:
        raise _issue("nonprospective_decision_card", "card", "signed_at")
    if card.signed_at > started:
        raise _issue("future_card_signature", "card", "signed_at")
    if card.max_false_action_risk < _RISK:
        raise _issue("insufficient_risk_ceiling", "card", "max_false_action_risk")
    for key in ("arms", "outcome", "horizon", "positive_direction"):
        if getattr(card, key) != getattr(design, key):
            raise _issue("decision_card_mismatch", "card", key)
    for key, expected in (
        ("target", manifest.sampling_target),
        ("permission_ref", manifest.permission_ref),
    ):
        if getattr(card, key) != expected:
            raise _issue("decision_card_mismatch", "card", key)
    for key, kind in (
        ("action_if_clear", "clear"),
        ("action_if_rule_out", "rule_out"),
        ("action_if_inconclusive", "abstain"),
    ):
        if getattr(card, key).kind != kind:
            raise _issue("invalid_decision_card", "card", key)
    if card.unblind_at > started:
        raise _issue(_UNBLIND_PENDING, "card", "unblind_at")


def _eligible_card(
    path: Path | None, manifest: _CaseInput, design: _TrialDesign | None, started: datetime
) -> _CardCheck:
    digest = sensitivity_tau = None
    try:
        if path is None:
            raise _issue("missing_decision_card", "card", "document")
        raw = _read(path, "missing_decision_card", "card_read_error", "card", "document")
        digest = _sha(raw)
        card = _validate(DecisionCard, raw, "invalid_decision_card", "card")
        sensitivity_tau = card.tau
        if design is None:
            raise _issue("historical_card_inapplicable", "card", "target")
        _check_card(card, manifest, design, started)
    except StudyError as exc:
        return _CardCheck(sha256=digest, sensitivity_tau=sensitivity_tau, issue=_diagnostic(exc))
    return _CardCheck(eligible=card, sha256=digest, sensitivity_tau=sensitivity_tau)


def _card_diagnostics(card: _CardCheck) -> tuple[_Diagnostic, ...]:
    if card.issue is None:
        return ()
    if card.issue.reason == _UNBLIND_PENDING:
        raise StudyError(card.issue.reason, card.issue.context)
    return (card.issue,)


def _records(
    text: TextIO, fields: tuple[str, ...], *, exact: bool
) -> Iterable[tuple[int, dict[str, str]]]:
    reader = csv.reader(text, strict=True)
    header = next(reader, [])
    if (
        len(set(header)) != len(header)
        or not set(fields).issubset(header)
        or (exact and len(header) != len(fields))
    ):
        raise _issue("invalid_case_schema", "adapter", "header")
    columns = tuple(header.index(field) for field in fields)
    for ordinal, row in enumerate(reader, start=1):
        if len(row) != len(header):
            raise _issue("invalid_case_schema", "adapter", "record", ordinal=ordinal)
        yield ordinal, {field: row[index] for field, index in zip(fields, columns, strict=True)}


def _binary(value: str, reason: str, field: str, ordinal: int) -> int:
    if value == "0":
        return 0
    if value == "1":
        return 1
    raise _issue(reason, "adapter", field, ordinal=ordinal)


def _hillstrom_rows(text: TextIO) -> list[_Row]:
    result = []
    recency = {str(month): int(month <= 6) for month in range(1, 13)}
    arms = {"No E-Mail": 0, "Mens E-Mail": 1}
    for ordinal, row in _records(text, ("segment", "conversion", "newbie", "recency"), exact=False):
        segment = row["segment"]
        if segment == "Womens E-Mail":
            continue
        if segment not in arms:
            raise _issue("invalid_assignment", "adapter", "segment", ordinal=ordinal)
        y = _binary(row["conversion"], "invalid_binary_outcome", "conversion", ordinal)
        s = _binary(row["newbie"], "invalid_baseline_flag", "newbie", ordinal)
        if row["recency"] not in recency:
            raise _issue("invalid_baseline_flag", "adapter", "recency", ordinal=ordinal)
        result.append((arms[segment], y, s, recency[row["recency"]]))
    return result


def _owner_rows(text: TextIO) -> list[_Row]:
    result: dict[str, _Row] = {}
    for ordinal, row in _records(text, ("unit_id", "arm", "y", "s", "b"), exact=True):
        unit_id = row["unit_id"]
        if not unit_id or unit_id != unit_id.strip():
            raise _issue("invalid_unit_id", "adapter", "unit_id", ordinal=ordinal)
        if unit_id in result:
            raise _issue("duplicate_unit", "adapter", "unit_id", ordinal=ordinal)
        arm = _binary(row["arm"], "invalid_assignment", "arm", ordinal)
        y = _binary(row["y"], "invalid_binary_outcome", "y", ordinal)
        s = _binary(row["s"], "invalid_baseline_flag", "s", ordinal)
        b = _binary(row["b"], "invalid_baseline_flag", "b", ordinal)
        result[unit_id] = arm, y, s, b
    return [result[unit_id] for unit_id in sorted(result)]


def _read_rows(raw: bytes, source: _Source, design: _TrialDesign | None) -> list[_Row]:
    try:
        with io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8-sig", newline="") as stream:
            rows = _hillstrom_rows(stream) if source == "hillstrom" else _owner_rows(stream)
    except (UnicodeError, csv.Error):
        raise _issue("invalid_case_schema", "adapter", "csv_encoding_or_syntax") from None
    if not rows:
        raise _issue("empty_study", "adapter", "retained_sample")
    if source == "hillstrom" and len(rows) != _HILLSTROM_N:
        raise _issue(
            "hillstrom_count_mismatch", "adapter", "retained_sample", expected=_HILLSTROM_N
        )
    if design is not None and design.assignment == "complete":
        first_arm = rows[0][0]
        if not any(row[0] != first_arm for row in rows):
            raise _issue("assignment_design_mismatch", "adapter", "arm")
    return rows


def _counts(rows: Iterable[_Row]) -> Counts4:
    totals = [[0, 0, 0, 0] for _ in range(4)]
    for arm, y, s, b in rows:
        totals[2 * s + b][2 * arm] += 1
        totals[2 * s + b][2 * arm + 1] += y
    a, b, c, d = (Cell(*row) for row in totals)
    return a, b, c, d


def _sum_cells(cells: Iterable[Cell]) -> Cell:
    n0 = y0 = n1 = y1 = 0
    for cell in cells:
        n0 += cell.n0
        y0 += cell.y0
        n1 += cell.n1
        y1 += cell.y1
    return Cell(n0, y0, n1, y1)


def _inner(band: Band) -> Interval | None:
    lower, upper = band.L_ci[1], band.U_ci[0]
    return (lower, upper) if lower < upper else None


def _method(name: str, band: Band) -> _Method:
    return _Method(
        plugin=band.plugin,
        L_ci=band.L_ci,
        U_ci=band.U_ci,
        outer=band.outer,
        reason=band.reason,
        leaf_map=band.leaf_map,
        leaf_counts=band.leaf_counts,
        weight_mode="estimated",
        outer_width=band.width,
        inner_span=_inner(band),
        prerequisites=_PREREQUISITES[name],
    )


def _pipelines(methods: _Methods) -> _Pipelines[_Pipeline]:
    pipeline: list[_Pipeline] = []
    outer: Interval | None = methods.U0.outer
    used: list[str] = []
    requirements: dict[str, None] = {}
    for name in ("U0", "F", "Q", "H"):
        method: _Method = getattr(methods, name)
        used.append(name)
        requirements.update(dict.fromkeys(method.prerequisites))
        if outer is not None:
            lo, hi = max(outer[0], method.outer[0]), min(outer[1], method.outer[1])
            outer = (lo, hi) if lo <= hi else None
        pipeline.append(
            _Pipeline(
                methods=tuple(used),
                outer=outer,
                plugin=method.plugin if name == "U0" else None,
                plugin_reason=method.reason if name == "U0" else "outer_intersection_only",
                reason="discordant_bands" if outer is None else None,
                outer_width=None if outer is None else outer[1] - outer[0],
                prerequisites=tuple(requirements),
            )
        )
    p0, p1, p2, p3 = pipeline
    return _Pipelines(P0=p0, P1=p1, P2=p2, P3=p3)


def _estimate(rows: list[_Row], source: _Source) -> _Bounds:
    n_train = len(rows) // 3
    order = np.random.Generator(np.random.PCG64(_SEED)).permutation(len(rows))
    train = _counts(rows[i] for i in order[:n_train])
    evaluation = _counts(rows[i] for i in order[n_train:])
    a, b, c, d = (_sum_cells(pair) for pair in zip(train, evaluation, strict=True))
    full = (a, b, c, d)
    cell = _sum_cells(full)
    difference = cell.y1 / cell.n1 - cell.y0 / cell.n0 if cell.n0 and cell.n1 else None
    u0, f, q = (summarize(full, leaf_map, atom_weights=None) for leaf_map in _FIXED_MAPS)
    h = summarize(evaluation, learn(train, omit_b=False), atom_weights=None)
    m = summarize(evaluation, learn(train, omit_b=True), atom_weights=None)
    methods = _Methods(
        U0=_method("U0", u0),
        F=_method("F", f),
        Q=_method("Q", q),
        H=_method("H", h),
        M=_method("M", m),
    )
    historical = source == "hillstrom"
    assumptions = (
        "Coverage assumes independent units from the same named population, comparable randomized arms, "
        "no spillover and complete attribution and follow-up. The file cannot verify those assumptions.",
        "Harm is loss of the specified beneficial outcome, not general safety. No individual harm or "
        "field false-action rate is observed.",
    )
    if historical:
        assumptions += (
            "Historical benchmark only. Source row ordinal does not establish customer uniqueness. "
            "Assignment and equal conditional allocation are documentary assumptions.",
        )
    return _Bounds(
        observed=_Observed(
            n0=cell.n0, y0=cell.y0, n1=cell.n1, y1=cell.y1, mean_difference=difference
        ),
        split=_Split(training=n_train, estimation=len(rows) - n_train),
        methods=methods,
        pipelines=_pipelines(methods),
        historical_benchmark=historical,
        limitations=assumptions,
    )


def _regimes(band: Interval, tau: float) -> tuple[_Regime, ...]:
    lo, hi = band
    states: list[_Regime] = []
    if lo < tau:
        states.append("below")
    if lo <= tau <= hi:
        states.append("equal")
    if hi > tau:
        states.append("above")
    return tuple(states)


def _loss(action: _ActionKind, regime: _Regime, card: DecisionCard) -> Fraction:
    if action == "abstain":
        return Fraction(card.cost_abstain)
    if action == "clear":
        return Fraction(0 if regime == "below" else card.cost_false_clear)
    return Fraction(0 if regime == "above" else card.cost_false_rule_out)


def _usefulness(
    band: Interval, proposed: _ActionKind, card: DecisionCard
) -> tuple[Fraction, Fraction]:
    workflow_cost = Fraction(card.incremental_workflow_cost)
    gains = tuple(
        _loss(card.current_action.kind, state, card) - _loss(proposed, state, card) - workflow_cost
        for state in _regimes(band, card.tau)
    )
    return min(gains), max(gains)


def _action_receipt(key: _ActionKey, action: _Action) -> _ActionReceipt:
    return _ActionReceipt(key=key, kind=action.kind, cost_bearer=action.cost_bearer)


def _wire_gain(value: Fraction) -> float:
    try:
        return float(value)
    except OverflowError:
        raise _issue("nonfinite_usefulness", "decision", "gain") from None


def _pipeline_decision(
    band: Interval, card: DecisionCard, *, irreducible: bool
) -> _PipelineDecision:
    outcome = decide(band, card.tau)
    action = card.action_if_inconclusive
    key: _ActionKey = "action_if_inconclusive"
    ambiguity = guessing = None
    if outcome == "clear":
        action = card.action_if_clear
        key = "action_if_clear"
    elif outcome == "rule_out":
        action = card.action_if_rule_out
        key = "action_if_rule_out"
    else:
        ambiguity = (
            "irreducible_at_features" if irreducible else "sampling_or_unresolved_identification"
        )
        states = _regimes(band, card.tau)
        wait_cost = Fraction(card.cost_abstain)
        guessing = _Guessing(
            clear=float(max(_loss("clear", state, card) for state in states) - wait_cost),
            rule_out=float(max(_loss("rule_out", state, card) for state in states) - wait_cost),
        )
    g_min, g_max = _usefulness(band, action.kind, card)
    minimum = Fraction(card.minimum_gain)
    return _PipelineDecision(
        decision=outcome,
        action=_action_receipt(key, action),
        changes_action=action != card.current_action,
        g_min=_wire_gain(g_min),
        g_max=_wire_gain(g_max),
        gain_exceeds_minimum=g_min > minimum,
        best_gain_exceeds_minimum=g_max > minimum,
        ambiguity=ambiguity,
        guess_vs_wait_worst_excess=guessing,
    )


def _worthwhile(result: _PipelineDecision) -> bool:
    return (
        result.decision != "inconclusive" and result.changes_action and result.gain_exceeds_minimum
    )


def _owner_decision(bounds: _Bounds, card: DecisionCard) -> _OwnerDecision:
    inner = bounds.methods.Q.inner_span
    irreducible = inner is not None and inner[0] + _TOL < card.tau < inner[1] - _TOL
    decisions = []
    for name, pipe in bounds.pipelines:
        if pipe.outer is None:
            raise _issue("discordant_bands", "decision", name)
        decisions.append(_pipeline_decision(pipe.outer, card, irreducible=irreducible))
    p0, p1, p2, p3 = decisions
    simplest = None
    if _worthwhile(p0):
        simplest = "P0"
    elif _worthwhile(p1):
        simplest = "P1"
    elif _worthwhile(p2):
        simplest = "P2"
    return _OwnerDecision(
        tau=card.tau,
        minimum_gain=float(card.minimum_gain),
        pipelines=_Pipelines(P0=p0, P1=p1, P2=p2, P3=p3),
        smallest_useful_simple_pipeline=simplest,
        h_adds_decision=p2.decision == "inconclusive" and _worthwhile(p3),
        irreducible_at_features=irreducible,
        current_action=_action_receipt("current_action", card.current_action),
        incremental_workflow_cost=float(card.incremental_workflow_cost),
        cost_false_clear=float(card.cost_false_clear),
        cost_false_rule_out=float(card.cost_false_rule_out),
        cost_abstain=float(card.cost_abstain),
        exact_costs=_ExactCosts(
            cost_false_clear=card.cost_false_clear,
            cost_false_rule_out=card.cost_false_rule_out,
            cost_abstain=card.cost_abstain,
            incremental_workflow_cost=card.incremental_workflow_cost,
            minimum_gain=card.minimum_gain,
        ),
        max_false_action_risk=card.max_false_action_risk,
        alternatives=(
            _action_receipt("action_if_clear", card.action_if_clear),
            _action_receipt("action_if_rule_out", card.action_if_rule_out),
            _action_receipt("action_if_inconclusive", card.action_if_inconclusive),
        ),
    )


_UNAVAILABLE = frozenset(
    {
        "missing_case_manifest",
        "use_permission_unconfirmed",
        "unverified_trial_design",
        "invalid_expected_digest",
        "missing_case_data",
        "digest_mismatch",
        _UNBLIND_PENDING,
    }
)
_EXIT_CODES = {"completed": 0, "failed": 1, "research_only": 2, "unavailable": 2}
_THRESHOLD_HEADER = (
    "scope",
    "name",
    "tau",
    "outer_lower",
    "outer_upper",
    "decision",
    "reason",
    "threshold_origin",
)


class _Request(_Strict):
    source: _Source
    manifest: Path
    card: Path | None
    out: Path


def _json_bytes(record: _Strict) -> bytes:
    text = json.dumps(record.model_dump(mode="json", by_alias=True), allow_nan=False, indent=2)
    return (text + "\n").encode("utf-8")


def _artifact(out: Path, name: str, raw: bytes) -> str:
    with (out / name).open("xb") as stream:
        stream.write(raw)
    return _sha(raw)


def _threshold_rows(
    bounds: _Bounds | None, card: _CardCheck
) -> Iterable[tuple[str | float | None, ...]]:
    if bounds is None:
        return
    tau = card.sensitivity_tau
    card_origin = "eligible_card" if card.eligible is not None else "ineligible_card_sensitivity"
    thresholds = sorted(set(_THRESHOLDS) | ({tau} if tau is not None else set()))
    for scope, items in (("method", bounds.methods), ("pipeline", bounds.pipelines)):
        for name, band in items:
            for t in thresholds:
                origin = "protocol_grid" if t in _THRESHOLDS else card_origin
                if band.outer is None:
                    yield scope, name, t, None, None, "inconclusive", "discordant_bands", origin
                else:
                    lo, hi = band.outer
                    yield scope, name, t, lo, hi, decide(band.outer, t), None, origin


def _threshold_bytes(bounds: _Bounds | None, card: _CardCheck) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(_THRESHOLD_HEADER)
    writer.writerows(_threshold_rows(bounds, card))
    return text.getvalue().encode("utf-8")


def _reproduction(request: _Request, args: tuple[str, ...], started: datetime) -> _Reproduction:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    command = (
        "python",
        "-m",
        "calibration.harm_case",
        "--source",
        request.source,
        "--case-manifest",
        "<manifest>",
        "--out",
        "<out>",
    )
    if request.card is not None:
        command += ("--card", "<card>")
    return _Reproduction(
        source_revision=revision,
        protocol_base_revision=_PROTOCOL_BASE,
        harm_case_sha256=_sha(Path(__file__).read_bytes()),
        harm_bounds_sha256=_sha((_ROOT / "calibration/harm_bounds.py").read_bytes()),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        pydantic=pydantic.__version__,
        argv=command,
        exact_arguments_sha256=_sha(json.dumps(args, ensure_ascii=True).encode("ascii")),
        started_at=started,
        ended_at=datetime.now(UTC),
    )


def _check_design_refs(manifest: _CaseInput) -> None:
    for key in ("design_ref", "followup_ref", "unit_ref", "sampling_target"):
        if not getattr(manifest, key).strip():
            raise _issue("unverified_trial_design", "preflight", key)


def _check_digest_syntax(expected: str) -> None:
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise _issue("invalid_expected_digest", "preflight", "expected_sha256")


def _read_design(path: Path) -> bytes:
    return _read(
        path, "unverified_trial_design", "unverified_trial_design", "preflight", "design_ref"
    )


def _check_distinct_input(data: Path, resource: Path, coordinate: str) -> None:
    try:
        aliases = data == resource or data.samefile(resource)
    except FileNotFoundError:
        return
    except OSError:
        raise _issue("case_resource_check_error", "preflight", coordinate) from None
    if aliases:
        raise _issue("case_input_alias", "preflight", coordinate)


def _checked_paths(request: _Request, manifest: _CaseInput) -> tuple[Path, Path | None]:
    data = _local_path(manifest.csv_path, "invalid_case_path", "preflight", "csv_path")
    design = None
    if request.source == "owner_trial":
        design = _local_path(
            manifest.design_ref, "unverified_trial_design", "preflight", "design_ref"
        )
        _check_distinct_input(data, design, "design_ref")
    if request.card is not None:
        _check_distinct_input(data, request.card, "card")
    return data, design


def _assignment(design: _TrialDesign | None) -> dict[str, str | float | bool]:
    return {
        "design_verified": True,
        "assignment": design.assignment if design is not None else "reported_equal_allocation",
        "treatment_probability": design.treatment_probability if design is not None else 0.5,
    }


def _write_packets(
    out: Path,
    packet: _BoundsPacket,
    receipt: _InputReceipt,
    input_artifact_sha: str | None,
    card: _CardCheck,
) -> _Receipts:
    if input_artifact_sha is None:
        input_artifact_sha = _artifact(out, "input_manifest.json", _json_bytes(receipt))
    return _Receipts(
        inputs=input_artifact_sha,
        bounds=_artifact(out, "bounds.json", _json_bytes(packet)),
        thresholds=_artifact(out, "thresholds.csv", _threshold_bytes(packet.bounds, card)),
    )


def _outer_pipelines(bounds: _Bounds | None) -> _Pipelines[Interval | None] | None:
    if bounds is None:
        return None
    return _Pipelines(
        P0=bounds.pipelines.P0.outer,
        P1=bounds.pipelines.P1.outer,
        P2=bounds.pipelines.P2.outer,
        P3=bounds.pipelines.P3.outer,
    )


def _assess(
    bounds: _Bounds, card: DecisionCard | None, diagnostics: list[_Diagnostic]
) -> tuple[_Status, str | None, _OwnerDecision | None]:
    discordant = tuple(name for name, band in bounds.pipelines if band.outer is None)
    if discordant:
        diagnostics.extend(
            _diagnostic(_issue("discordant_bands", "decision", name)) for name in discordant
        )
        return "research_only", "discordant_bands", None
    if card is None:
        return "research_only", "missing_decision_card", None
    return "completed", None, _owner_decision(bounds, card)


def _run(request: _Request, args: tuple[str, ...]) -> int:
    started = datetime.now(UTC)
    protocol_digest = _sha(_PROTOCOL.read_bytes())
    reproduction = _reproduction(request, args, started)
    request.out.mkdir(parents=True)
    receipt = _InputReceipt(source=request.source)
    input_artifact_sha = None
    status: _Status = "completed"
    reason = None
    diagnostics: list[_Diagnostic] = []
    card = _CardCheck()
    bounds = owner = None
    try:
        raw_manifest = _read(
            request.manifest,
            "missing_case_manifest",
            "case_manifest_read_error",
            "preflight",
            "manifest",
        )
        receipt = receipt.model_copy(update={"manifest_sha256": _sha(raw_manifest)})
        manifest = _validate(_CaseInput, raw_manifest, "invalid_case_manifest", "preflight")
        _check_permission(manifest)
        receipt = receipt.model_copy(update={"permission_verified": True})
        _check_design_refs(manifest)
        _check_digest_syntax(manifest.expected_sha256)
        receipt = receipt.model_copy(update={"expected_csv_sha256": manifest.expected_sha256})
        path, design_path = _checked_paths(request, manifest)
        design = None
        if design_path is not None:
            raw_design = _read_design(design_path)
            receipt = receipt.model_copy(update={"design_sha256": _sha(raw_design)})
            design = _validate(_TrialDesign, raw_design, "unverified_trial_design", "preflight")
            if design.arms[0] == design.arms[1]:
                raise _issue("unverified_trial_design", "preflight", "arms")
        receipt = receipt.model_copy(update=_assignment(design))
        card = _eligible_card(request.card, manifest, design, started)
        receipt = receipt.model_copy(update={"card_sha256": card.sha256})
        diagnostics.extend(_card_diagnostics(card))
        raw = _read(path, "missing_case_data", "case_read_error", "preflight", "csv")
        csv_digest = _sha(raw)
        receipt = receipt.model_copy(update={"csv_sha256": csv_digest})
        if csv_digest != manifest.expected_sha256:
            raise _issue("digest_mismatch", "preflight", "expected_sha256")
        receipt = receipt.model_copy(update={"digest_verified": True})
        input_artifact_sha = _artifact(request.out, "input_manifest.json", _json_bytes(receipt))
        bounds = _estimate(_read_rows(raw, request.source, design), request.source)
        status, reason, owner = _assess(bounds, card.eligible, diagnostics)
    except StudyError as issue:
        reason = issue.reason
        diagnostics.append(_diagnostic(issue))
        status = "unavailable" if reason in _UNAVAILABLE else "failed"
    except (ArithmeticError, ValueError, RuntimeError):
        reason = "case_computation_error"
        diagnostics.append(_diagnostic(_issue(reason, "estimate", "field")))
        status = "failed"
    bounds_packet = _BoundsPacket(
        protocol_sha256=protocol_digest, status=status, reason=reason, bounds=bounds
    )
    receipts = _write_packets(request.out, bounds_packet, receipt, input_artifact_sha, card)
    result = _FieldDecision(
        source=request.source,
        status=status,
        reason=reason,
        protocol_sha256=protocol_digest,
        manifest_sha256=receipt.manifest_sha256,
        card_sha256=receipt.card_sha256,
        permission_verified=receipt.permission_verified,
        design_verified=receipt.design_verified,
        bounds=_outer_pipelines(bounds),
        owner_decision=owner,
        diagnostics=tuple(diagnostics),
        receipts=receipts,
        reproduction=reproduction.model_copy(update={"ended_at": datetime.now(UTC)}),
    )
    _artifact(request.out, "decision.json", _json_bytes(result))
    return _EXIT_CODES[status]


def main(argv: list[str] | None = None) -> int:
    """Write an aggregate-only field packet."""
    args = tuple(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("hillstrom", "owner_trial"), required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--card", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    ns = parser.parse_args(args)
    request = _Request(source=ns.source, manifest=ns.case_manifest, card=ns.card, out=ns.out)
    try:
        return _run(request, args)
    except (OSError, subprocess.CalledProcessError):
        print("Could not read the study files or write a fresh field packet.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
