"""Drives one ParityCase through every constructor it lists and compares
emitted rows, point estimates, interval bounds, retained sequential state,
and every multiplicity-bearing field a role-based plan produces.

Row identity is (metric, arm, estimand, value_scale, segment,
analysis_population) -- `analysis_population` is part of IDENTITY, not
payload, because an assigned-population row and a triggered-population row
for the same metric/arm are two DIFFERENT rows that must never collide
under one key (a probe of a triggered case confirmed a collision without
it: the triggered row silently overwrote the assigned row in `_normalize`'s
dict, and the harness could not have caught a triggered-only regression).
`method` moved out of identity into payload: a decision-method estimate
and its sensitivity-method sibling for the SAME (metric, arm, estimand)
are compared as two entries under the same identity's `by_method` mapping,
so a path that silently drops the sensitivity row is still caught by
dict-key comparison, not lost inside a tuple identity a caller has to know
to split on.

The compared payload is the full additive-and-relative surface: value, lb,
ub, level, role, discovery, family_q, family_threshold, family_guarantee,
family_nominal_alpha, family_axes, alternative, reference_kind, note,
inference, null_lift, null_abs, abs_diff, abs_se, abs_lb, abs_ub,
abs_reference_kind, abs_reference_df, reference_df, dof, quantile_p_value,
relative_unavailable_reason, relative_confidence_set (structurally, via
`model_dump()`), BreakoutEstimate's own `excluded`, and a sequential row's
own evidence (`sequential_result.log_e`/`decision_alpha` and its
checkpoint's `status`) -- every one of these plus emitted row sets, values,
bounds, failure/refusal codes, and retained sequential state, matching the
parity contract this harness exists to enforce.
Failure/refusal codes are covered by strict
row-SET equality (a per-cell failure that silently drops a row on one path
but not another IS a row-set mismatch) plus `waived_refusal_codes`'
code-exact assertion for a whole-constructor refusal; there is no separate
per-cell failure-code channel to inspect because `Analysis.run()` does not
expose one (a per-cell `DecisionFailure` surfaces only as an omitted row
and a warning, never a structured return value) -- row-set equality is the
correct, and only available, proxy.

Retained sequential state is compared as a PATH-INDEPENDENT fingerprint
(`sequential_fingerprint`, below), never `SequentialSnapshot.prefix_id`
directly. `prefix_id` is `canonical_id({registration_id, records,
states})` (`increment/sequential_state.py`), and `registration_id` is
derived from `definitions_id`, which hashes the OBSERVATION MAPPING --
`native_observation_mapping` (a dump of the definitions/experiment) on
warehouse paths, `frame_observation_mapping` (unit/group column names) on
dataframe paths (`increment/sequential_source.py`). Those two mappings are
never equal, so `prefix_id` differs between `from_definitions` and
`from_unit_summary` even over byte-identical data -- confirmed live: the
same 120-unit dataset produced `definitions_id` `1e97b6bc...` /
`prefix_id` `28ecf2bb...` on `from_definitions` and `definitions_id`
`cf1bf616...` / `prefix_id` `429fc17d...` on `from_unit_summary`. The
fingerprint strips every registration-derived identifier and compares only
the arm-level exact statistics (`SequentialArmState`: law, n, successes,
mean, scatter) and the SORTED `(unit_id, group_id)` set (not reveal
order, which differs across paths even for identical content -- see
`sequential_fingerprint`'s own docstring for the confirmed live probe),
which ARE path-independent. A raw `prefix_id`-equality claim was also
tried for the two constructor pairs this harness always builds from one
shared registration (`from_unit_day_artifact` publishes from the SAME
`from_definitions` analysis; `from_moments` exports from the SAME
`from_unit_summary` analysis) and dropped: confirmed live, even within
such a pair -- same `definitions_id`, same 120 records -- the raw
`prefix_id` still differs, because its digest is order-of-accumulation
sensitive and publish/reopen does not preserve that order. Only the
fingerprint is a safe cross-path (or cross-reconstruction) invariant.

Every one of the six constructor names in `CONSTRUCTORS` must appear in
`case.build` or `case.waive` (a name in neither is a defect this module's
`assert_parity` raises on, not a silent gap the caller has to notice).
"""

from __future__ import annotations

import math
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import pytest

from increment._analysis_config import UNSET
from increment.errors import CodedError, IncrementWarning
from tests.warning_codes import warning_codes

from .cases import CONSTRUCTORS, ParityCase, _close_parity_analysis

_TOLERANCE = 1e-9
_NUMERIC_FIELDS = (
    "value",
    "lb",
    "ub",
    "level",
    "abs_diff",
    "abs_se",
    "abs_lb",
    "abs_ub",
    "abs_reference_df",
    "reference_df",
    "dof",
    "quantile_p_value",
    "null_lift",
    "null_abs",
    "posterior_prob_favorable",
    # `family_nominal_alpha` is the family's own nominal significance level
    # (the other arm of `fcr_alpha = min(family_threshold, family_nominal_alpha)`,
    # `increment/estimation/results.py`) -- as float-valued and as
    # parity-relevant as `family_threshold`/`family_q` beside it.
    "family_nominal_alpha",
    # `sequential_result.log_e`/`decision_alpha` carry the e-BH/e-process
    # evidence and its reinverted allocation -- a path that landed on the
    # same bounds from different evidence, or reinverted at a different
    # alpha, would otherwise pass unnoticed.
    "sequential_log_e",
    "sequential_decision_alpha",
)
_EXACT_FIELDS = (
    "role",
    "method_role",
    "discovery",
    "family_q",
    "family_threshold",
    "alternative",
    "reference_kind",
    "note",
    "family_guarantee",
    "inference",
    "abs_reference_kind",
    "relative_unavailable_reason",
    "prior_shrunk",
    "prior_spec",
    # `BreakoutEstimate`-only; `getattr(..., None)` on a `LiftEstimate` row
    # is a correct absence, not a skipped comparison -- same rule as
    # `_row_identity`'s `analysis_population`.
    "excluded",
    # The axes a family correction ran over (e.g. `("metric", "arm")` vs
    # `("metric", "arm", "segment")`) -- a path mislabeling which axes it
    # corrected over would otherwise pass with equal bounds.
    "family_axes",
    # The retained checkpoint's freeze state ("current"/"frozen"/"missing")
    # behind `sequential_result` -- the freeze the numeric e-value fields
    # above don't carry.
    "sequential_checkpoint_status",
)
# `relative_confidence_set` nests an `Estimate`-bearing model: compared via
# `model_dump()` with the same numeric tolerance as every other bound,
# because two paths reaching the same Fieller set can differ in the last
# ULP of a bound the way `lift.value`/`lb`/`ub` already do.
_NESTED_FIELDS = ("relative_confidence_set",)


def _row_identity(row: Any) -> tuple:
    # `BreakoutEstimate` has no `analysis_population`; getattr yields None for
    # every breakout row, so breakout rows still compare consistently.
    return (
        row.metric,
        row.group_id,
        row.estimand,
        row.value_scale,
        getattr(row, "dimension_value", None),
        getattr(row, "analysis_population", None),
    )


def _row_payload(row: Any) -> dict[str, Any]:
    lift = row.lift
    relative_confidence_set = getattr(row, "relative_confidence_set", None)
    # `sequential_result` (log_e, decision_alpha, checkpoint.status) is
    # None on a non-sequential row -- both sides then compare as None,
    # the same absence rule `_row_identity`'s `analysis_population` uses.
    sequential_result = getattr(row, "sequential_result", None)
    prior_shrunk = getattr(row, "prior_shrunk", False)
    prior_spec = getattr(row, "prior_spec", None)
    return {
        "value": lift.value if lift is not None else None,
        "lb": lift.lb if lift is not None else None,
        "ub": lift.ub if lift is not None else None,
        "level": lift.level if lift is not None else None,
        # `role`/`discovery` are `LiftEstimate`-only; `BreakoutEstimate` rows
        # compare as None here and carry family_q/family_threshold instead.
        "role": getattr(row, "role", None),
        "method_role": row.method_role,
        "discovery": getattr(row, "discovery", None),
        "family_q": row.family_q,
        "family_threshold": row.family_threshold,
        "alternative": row.alternative,
        "reference_kind": row.reference_kind,
        "note": row.note,
        # `family_guarantee` lands in a follow-up; `getattr` keeps this
        # harness runnable against a LiftEstimate that does not carry the
        # field yet -- both sides compare as None until then, which is a
        # correct absence, not a silently-skipped comparison.
        "family_guarantee": getattr(row, "family_guarantee", None),
        "family_nominal_alpha": row.family_nominal_alpha,
        "inference": row.inference,
        "null_lift": row.null_lift,
        "null_abs": row.null_abs,
        "prior_shrunk": prior_shrunk,
        "prior_spec": prior_spec.model_dump(mode="json") if prior_spec is not None else None,
        "posterior_prob_favorable": row.prob_favorable() if prior_shrunk else None,
        "abs_diff": row.abs_diff,
        "abs_se": row.abs_se,
        "abs_lb": row.abs_lb,
        "abs_ub": row.abs_ub,
        "abs_reference_kind": row.abs_reference_kind,
        "abs_reference_df": row.abs_reference_df,
        "reference_df": row.reference_df,
        "dof": row.dof,
        # `quantile_p_value` is `LiftEstimate`-only; `BreakoutEstimate` has
        # no such field at all -- same absence rule as above.
        "quantile_p_value": getattr(row, "quantile_p_value", None),
        "relative_unavailable_reason": row.relative_unavailable_reason,
        "relative_confidence_set": (
            relative_confidence_set.model_dump() if relative_confidence_set is not None else None
        ),
        # `BreakoutEstimate`-only; absent on `LiftEstimate` rows.
        "excluded": getattr(row, "excluded", None),
        "family_axes": row.family_axes,
        "sequential_log_e": (
            float(sequential_result.log_e) if sequential_result is not None else None
        ),
        "sequential_decision_alpha": (
            float(sequential_result.decision_alpha) if sequential_result is not None else None
        ),
        "sequential_checkpoint_status": (
            sequential_result.checkpoint.status if sequential_result is not None else None
        ),
    }


def _normalize(estimates: Any) -> dict[tuple, dict[str, dict[str, Any]]]:
    """`{identity: {method: payload}}` -- `method` (decision vs. every
    sensitivity method) nests under identity rather than joining it, so a
    missing sensitivity row is a missing dict key, caught the same way a
    missing top-level row is."""
    from increment.estimation.contrast_results import ContrastResult

    out: dict[tuple, dict[str, dict[str, Any]]] = {}
    for row in estimates:
        if isinstance(row, ContrastResult):
            key = (row.metric, row.treatment_group, row.estimand, "absolute", None, None)
            out.setdefault(key, {})[row.method] = {"contrast": row.model_dump(mode="json")}
            continue
        out.setdefault(_row_identity(row), {})[row.method] = _row_payload(row)
    return out


def sequential_fingerprint(snapshot: Any) -> tuple:
    """Path-independent identity of retained sequential state: sorted
    per-arm `SequentialArmState` content (metric, group, segment, law, n,
    successes, mean, scatter) plus the SORTED `(unit_id, group_id)` set --
    deliberately never `prefix_id`/`registration_id`/`definitions_id`,
    which hash path-specific observation-mapping metadata into the
    registration identity and so differ across paths reading identical
    data by construction (see this module's own docstring for the
    confirmed live values). Records are sorted, not reveal-ordered: a
    probe over the same dataset confirmed the arm states match exactly
    between `from_definitions` and `from_unit_summary` while the raw
    `snapshot.records` sequence does not (`('c0','control'), ('c1',...`
    on the warehouse path vs. `('c28','control'), ('c33',...` on the frame
    path -- the two paths reveal the same 120 units in a different order,
    not a different set), so comparing the sorted set is the correct,
    order-independent content check; the sorted-tuple form matched exactly
    once records were sorted, confirmed live on the same dataset. Exported
    for reuse by any `ParityCase` comparing sequential state, including
    one capturing state after a freeze."""
    states = tuple(
        sorted(
            (s.metric, s.group_id, s.segment, s.law, s.n, s.successes, s.mean, s.scatter)
            for s in snapshot.states
        )
    )
    records = tuple(sorted((r.unit_id, r.group_id) for r in snapshot.records))
    return (states, records)


@dataclass(frozen=True)
class SequentialCapture:
    prefix_id: str
    fingerprint: tuple


@dataclass(frozen=True)
class CaseResult:
    rows: dict[str, dict[tuple, dict[str, dict[str, Any]]]]
    refusals: dict[str, str]
    sequential_state: dict[str, SequentialCapture] = field(default_factory=dict)


def run_case(case: ParityCase) -> CaseResult:
    """Attempt every constructor `case.build` lists; return normalized rows
    for the ones that succeeded, the `.code` for the ones that raised, and
    (when `case.sequential`) each surviving constructor's `SequentialCapture`
    (a path-independent fingerprint plus the raw `prefix_id`, both derived
    from the same `SequentialSnapshot`). Raises (not returns) any
    `CodedError` from a constructor that is neither waived nor expected to
    refuse -- a genuine parity break.

    A registered sequential source refuses `.run()` until a checkpoint
    has been captured -- confirmed live: calling `.run()` before
    `capture_sequential` on a bound native source raised
    `increment.errors.CapabilityError: source has no exact finalized
    joint-unit checkpoint; capture or adopt a current sequential
    snapshot` (`sequential.continuation.legacy`) the first time this
    harness ran a native-path sequential case end to end. `capture_sequential`
    therefore runs FIRST for a sequential case, and `.run()` reads off
    that one adopted checkpoint -- not run-then-capture-then-run-again.
    """
    rows: dict[str, dict[tuple, dict[str, dict[str, Any]]]] = {}
    refusals: dict[str, str] = {}
    sequential_state: dict[str, SequentialCapture] = {}
    for name, build in case.build.items():
        analysis = None
        try:
            analysis = build()
            estimand_kwargs = {"estimands": case.estimands} if case.estimands else {}
            prior_kwargs = {"prior": case.prior} if case.prior is not UNSET else {}
            metric_kwargs = {"metrics": list(case.metrics)} if case.metrics is not None else {}
            if case.sequential:
                as_of = getattr(analysis, "_sequential_as_of", None)
                snapshot_kwargs = {"as_of": as_of} if as_of is not None else {}
                snapshot = analysis.capture_sequential(finalized=True, **snapshot_kwargs)
                sequential_state[name] = SequentialCapture(
                    prefix_id=snapshot.prefix_id, fingerprint=sequential_fingerprint(snapshot)
                )
                from increment import snapshot_from_json
                from increment.sequential_state import declare_sequential_freeze_cells

                empty = declare_sequential_freeze_cells(snapshot, ())
                replayed = snapshot_from_json(empty.model_dump_json())
                assert replayed == snapshot, (
                    f"{case.id}: empty freeze changed retained state/rows for {name}"
                )
                continued = analysis.capture_sequential(
                    finalized=True, previous=replayed, **snapshot_kwargs
                )
                assert sequential_fingerprint(continued) == sequential_fingerprint(snapshot), (
                    f"{case.id}: replayed empty-freeze snapshot could not continue on {name}"
                )
            expected_warnings = case.expected_warning_codes.get(name, ())
            warning_context = (
                warnings.catch_warnings(record=True) if expected_warnings else nullcontext()
            )
            with warning_context as caught:
                if expected_warnings:
                    warnings.simplefilter("always", IncrementWarning)
                results = (
                    analysis.run_breakout(**estimand_kwargs, **prior_kwargs, **metric_kwargs)
                    if case.breakout_dimension
                    else analysis.run(**estimand_kwargs, **prior_kwargs, **metric_kwargs)
                )
                if case.readout_probe is not None:
                    case.readout_probe(results)
                if case.source_probe is not None:
                    case.source_probe(name, analysis)
            if expected_warnings:
                assert caught is not None
                assert set(warning_codes(caught)) == set(expected_warnings)
            rows[name] = _normalize(results)
        except CodedError as exc:
            expected = case.waived_refusal_codes.get(name)
            if expected is None:
                raise AssertionError(
                    f"{case.id}: {name} raised an unwaived refusal {exc.code!r} -- "
                    "either this constructor should match the others, or it needs "
                    "a `waive`/`waived_refusal_codes` entry naming why"
                ) from exc
            if exc.code != expected:
                raise AssertionError(
                    f"{case.id}: {name} was waived for {expected!r} but raised {exc.code!r}"
                ) from exc
            refusals[name] = exc.code
        finally:
            if analysis is not None:
                _close_parity_analysis(analysis)
    if case.sequential_probe is not None:
        case.sequential_probe()
    return CaseResult(rows=rows, refusals=refusals, sequential_state=sequential_state)


def _nested_close(expected: Any, actual: Any) -> bool:
    """Tolerant structural equality for a `model_dump()`-ed nested field:
    same recursive shape, floats compared with `_TOLERANCE` like every
    other numeric bound this harness checks."""
    if expected is None or actual is None:
        return expected is None and actual is None
    if isinstance(expected, float) and isinstance(actual, float):
        return abs(expected - actual) <= _TOLERANCE * max(1.0, abs(expected)) or (
            math.isinf(expected) and expected == actual
        )
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            _nested_close(expected[k], actual[k]) for k in expected
        )
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        return len(expected) == len(actual) and all(
            _nested_close(e, a) for e, a in zip(expected, actual, strict=True)
        )
    return expected == actual


def _assert_payload_equal(
    case_id: str, name: str, oracle_name: str, key: tuple, method: str, expected: dict, actual: dict
) -> None:
    if "contrast" in expected:
        assert actual.keys() == expected.keys()
        assert _nested_close(expected["contrast"], actual["contrast"]), (
            f"{case_id}: {name}/{oracle_name} disagree on contrast {key}/{method}"
        )
        return
    for field_name in _NUMERIC_FIELDS:
        e, a = expected[field_name], actual[field_name]
        if e is None or a is None:
            assert e is None and a is None, (
                f"{case_id}: {name}/{oracle_name} disagree on availability of "
                f"{field_name} for {key}/{method}: {a!r} vs {e!r}"
            )
            continue
        assert abs(e - a) <= _TOLERANCE * max(1.0, abs(e)) or (math.isinf(e) and e == a), (
            f"{case_id}: {name} vs {oracle_name} disagree on {field_name} for {key}/{method}: {a} != {e}"
        )
    for field_name in _EXACT_FIELDS:
        e, a = expected[field_name], actual[field_name]
        assert e == a, (
            f"{case_id}: {name} vs {oracle_name} disagree on {field_name} for {key}/{method}: {a!r} != {e!r}"
        )
    for field_name in _NESTED_FIELDS:
        e, a = expected[field_name], actual[field_name]
        assert _nested_close(e, a), (
            f"{case_id}: {name} vs {oracle_name} disagree on {field_name} for {key}/{method}: {a!r} != {e!r}"
        )


def _assert_no_silent_skip(case: ParityCase) -> None:
    """Every constructor is built, or waived with a reason, or both (a
    coded waiver): never neither. A name in `waive` without a matching
    `waived_refusal_codes` entry must be ABSENT from `build` (not
    attempted, reason recorded); a name WITH a coded entry must be PRESENT
    in `build` (attempted, expected to raise that code)."""
    covered = set(case.build) | set(case.waive)
    missing = set(CONSTRUCTORS) - covered
    assert not missing, (
        f"{case.id}: {sorted(missing)} neither built nor waived -- every "
        "constructor must be attempted or recorded as not attempted with a reason"
    )
    for name in case.waive:
        if name in case.waived_refusal_codes:
            assert name in case.build, (
                f"{case.id}: {name} has a waived_refusal_codes entry but is not in "
                "build -- a coded waiver requires actually attempting the constructor"
            )
        else:
            assert name not in case.build, (
                f"{case.id}: {name} is in build but only reason-waived (no "
                "waived_refusal_codes entry) -- either attempt it for real "
                "comparison or add the code it is expected to raise"
            )


def assert_parity(case: ParityCase, result: CaseResult) -> None:
    """Every non-waived constructor that produced rows must agree, row for
    row and field for field, with every other non-waived constructor --
    every waived constructor must actually have refused (not silently
    produced rows) -- every constructor is accounted for (built or waived,
    never neither) -- and (when `case.sequential`) every non-waived
    constructor's retained sequential state must share the same
    path-independent fingerprint."""
    _assert_no_silent_skip(case)
    for name in case.waived_refusal_codes:
        assert name in result.refusals, (
            f"{case.id}: {name} was waived for "
            f"{case.waived_refusal_codes[name]!r} but produced rows instead "
            "of refusing -- the waiver is stale, the capability now works"
        )
    live = {name: r for name, r in result.rows.items() if name not in case.waived_refusal_codes}
    if not live:
        pytest.fail(f"{case.id}: every constructor was waived -- nothing to compare")
    oracle_name, oracle_rows = next(iter(live.items()))
    if case.require_selection:
        discoveries = [
            payload.get("discovery")
            for methods in oracle_rows.values()
            for payload in methods.values()
        ]
        assert True in discoveries and False in discoveries, (
            f"{case.id}: expected at least one selected and one unselected row on "
            f"{oracle_name}, got discovery values {discoveries}"
        )
    for name, other_rows in live.items():
        assert other_rows.keys() == oracle_rows.keys(), (
            f"{case.id}: {name} emitted a different row set than {oracle_name} -- "
            f"only in {name}: {other_rows.keys() - oracle_rows.keys()}, "
            f"only in {oracle_name}: {oracle_rows.keys() - other_rows.keys()}"
        )
        for key, oracle_methods in oracle_rows.items():
            other_methods = other_rows[key]
            assert other_methods.keys() == oracle_methods.keys(), (
                f"{case.id}: {name} emitted different methods than {oracle_name} for row {key} -- "
                f"only in {name}: {other_methods.keys() - oracle_methods.keys()}, "
                f"only in {oracle_name}: {oracle_methods.keys() - other_methods.keys()}"
            )
            for method, expected_payload in oracle_methods.items():
                _assert_payload_equal(
                    case.id, name, oracle_name, key, method, expected_payload, other_methods[method]
                )
    if case.sequential:
        live_states = {
            name: state for name, state in result.sequential_state.items() if name in live
        }
        assert live_states, (
            f"{case.id}: declared sequential=True but no constructor captured a snapshot"
        )
        oracle_state_name, oracle_state = next(iter(live_states.items()))
        for name, state in live_states.items():
            assert state.fingerprint == oracle_state.fingerprint, (
                f"{case.id}: {name}'s retained sequential state disagrees with "
                f"{oracle_state_name}'s (path-independent fingerprint mismatch)"
            )
        # Raw `prefix_id` is not compared, even within shared-registration pairs:
        # its digest depends on unit-proof accumulation order, which
        # publish/reopen does not preserve. The fingerprint above is the invariant.
