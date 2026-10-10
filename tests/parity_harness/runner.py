"""Drives one ParityCase through every constructor it lists and compares
emitted rows, point estimates, interval bounds, retained sequential state,
and every multiplicity-bearing field a role-based plan produces.

Row identity is (metric, arm, estimand, value_scale, segment, analysis_population, ds,
ds_basis) -- `analysis_population` is part of IDENTITY, not
payload, because an assigned-population row and a triggered-population row
for the same metric/arm are two DIFFERENT rows that must never collide
under one key (a probe of a triggered case confirmed a collision without
it: the triggered row silently overwrote the assigned row in `_normalize`'s
dict, and the harness could not have caught a triggered-only regression).
`ds`/`ds_basis` name a day-axis row's day. A second row under one identity and method is
a defect (a join fan-out), refused by `_normalize` rather than overwritten.
`method` moved out of identity into payload: a decision-method estimate
and its sensitivity-method sibling for the SAME (metric, arm, estimand)
are compared as two entries under the same identity's `by_method` mapping,
so a path that silently drops the sensitivity row is still caught by
dict-key comparison, not lost inside a tuple identity a caller has to know
to split on.

The compared payload is every public field of the emitted row (its `model_dump()`), except
ingress-specific identity hashes that are validated against retained scope metadata before
normalization: `source_snapshot_id` and `family_id`, resolved fact `source`, the three identifiers a sequential
checkpoint derives from its registration (`registration_id`, `filtration_id`, `prefix_id`; they
hash the path's observation mapping, see below) and, inside a winsor confidence set's raw pool,
`study_id` (the source's identity) and `missingness` (a frame names its declared policy, a
warehouse its inclusion rule); the pool's outcome multisets, quantile, support and inference are
compared. All other fields compare automatically, including interval and its `alpha`/`log_mean`/
`log_se`, winsorization thresholds, counts and `confidence_set`, `n_clusters`, Fieller and
binomial sets, null reasons, family and policy metadata, and a sequential row's own evidence.
`family_id` is omitted because A0 ruling 12 defines it as a source-snapshot-specific physical
identity; the retained family scope is compared by its name, procedure and cell membership.
The runners construct each source against a fixed fixture and read it through each ingress, so
snapshot identity and warehouse fact-source identity are path-specific.
Floats agree within `TOLERANCE` relative with the same absolute floor; all other values
(including exact rationals) agree exactly.
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

import inspect
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import ValidationError

from increment import Analysis
from increment._analysis_config import UNSET
from increment.errors import CodedError, IncrementWarning
from tests.warning_codes import warning_codes

from .cases import CONSTRUCTORS, Absence, ParityCase, _close_parity_analysis
from .comparison import nested_close

# Dropped from the compared payload only after source identity is checked against the retained
# scope metadata; the other exclusions name the path rather than the result (see module docstring):
# the fact `source`, sequential checkpoint registration identifiers, and raw-pool path labels.
_POOL_LABELS = {"raw": {"study_id": True, "missingness": True}}
_PATH_SPECIFIC: dict[str, Any] = {
    "source": True,
    # source_snapshot_id is validated against retained scope before it is treated as path-specific.
    "source_snapshot_id": True,
    "sequential_result": {
        "checkpoint": {"registration_id": True, "filtration_id": True, "prefix_id": True}
    },
    "confidence_set": {**_POOL_LABELS, "reference": _POOL_LABELS},
}


def _row_identity(row: Any) -> tuple:
    # `analysis_population` distinguishes assigned and triggered breakout rows.
    # `ds` and `ds_basis` name a day-axis row's day; they are None otherwise.
    return (
        row.metric,
        row.group_id,
        row.estimand,
        row.value_scale,
        getattr(row, "dimension_value", None),
        getattr(row, "analysis_population", None),
        getattr(row, "ds", None),
        getattr(row, "ds_basis", None),
    )


def _p_value(row: Any) -> float | None:
    """The row's presentation sampling p-value where its reference permits one."""
    p_value = getattr(row, "p_value", None)
    if (
        p_value is None
        or row.inference != "fixed"
        or row.null_abs is not None
        or row.lift is None
        and row.reference_kind != "binomial"
    ):
        return None
    try:
        value = p_value()
        return None if value is None else float(value)
    except CodedError:
        return None


def _row_payload(row: Any) -> dict[str, Any]:
    """Every public field of the row, plus the derived values the dump does not carry."""
    payload = row.model_dump(mode="python", exclude=_PATH_SPECIFIC)
    payload["p_value"] = _p_value(row)
    payload["posterior_prob_favorable"] = (
        row.prob_favorable() if getattr(row, "posterior_available", None) is True else None
    )
    # `log_e` is a property of the result, not a dumped field. `sequential_result` is None on
    # a non-sequential row: both sides then compare as None.
    sequential_result = getattr(row, "sequential_result", None)
    payload["sequential_log_e"] = (
        float(sequential_result.log_e) if sequential_result is not None else None
    )
    return payload


def _validate_source_snapshot_ids(case_id: str, name: str, results: Any) -> dict[tuple, Any]:
    """Validate row sources and return their source-qualified family scope."""
    rows = list(results)
    if not rows:
        return {}
    metadata = getattr(results, "metadata", None)
    scope = getattr(metadata, "scope", None)
    by_source = getattr(scope, "by_source", None)
    if scope is None or not by_source:
        raise AssertionError(
            f"{case_id}: {name} emitted rows without retained source-scope metadata"
        )
    from increment.estimation.readout_types import CellKey

    family_by_cell = {}
    for family in scope.families:
        family_source = by_source.get(family.source_snapshot_id)
        if family_source is None or not set(family.members) <= set(family_source.cells):
            raise AssertionError(
                f"{case_id}: {name} family membership is outside its retained source scope"
            )
        for cell in family.members:
            key = family.source_snapshot_id, cell
            if key in family_by_cell:
                raise AssertionError(f"{case_id}: {name} scope assigns a cell to multiple families")
            family_by_cell[key] = family
    family_for_row = {}
    for row in rows:
        source_id = getattr(row, "source_snapshot_id", None)
        if source_id is None:
            raise AssertionError(f"{case_id}: {name} emitted a row without source_snapshot_id")
        source = by_source.get(source_id)
        if source is None or source.source_snapshot_id != source_id:
            raise AssertionError(
                f"{case_id}: {name} row source_snapshot_id does not match retained scope metadata"
            )
        cell = CellKey.from_row(row)
        family = family_by_cell.get((source_id, cell))
        row_family_id = getattr(row, "family_id", None)
        if row_family_id != (None if family is None else family.family_id):
            raise AssertionError(
                f"{case_id}: {name} row family_id does not match retained family scope"
            )
        if family is not None and family.family is not None:
            row_procedure = (
                getattr(row, "family_axes", None),
                getattr(row, "family_q", None),
                getattr(row, "family_guarantee", None),
            )
            scope_procedure = (
                family.family.axes,
                family.family.q,
                family.family.validity_regime,
            )
            if row_procedure != scope_procedure:
                raise AssertionError(
                    f"{case_id}: {name} row family procedure {row_procedure!r} "
                    f"does not match retained scope {scope_procedure!r}"
                )
        family_for_row[(source_id, cell)] = family
    return family_for_row


def _validate_unscoped_output(case_id: str, name: str, results: Any, reason: str) -> None:
    """Require explicitly classified test outputs to remain genuinely unscoped."""
    if not reason.strip():
        raise AssertionError(f"{case_id}: {name} unscoped output needs a reason")
    metadata = getattr(results, "metadata", None)
    if getattr(metadata, "scope", None) is not None:
        raise AssertionError(f"{case_id}: {name} was classified unscoped but carries a scope")
    if any(getattr(row, "source_snapshot_id", None) is not None for row in results):
        raise AssertionError(f"{case_id}: {name} was classified unscoped but has source identities")


def _family_map_for_output(case: ParityCase, name: str, results: Any) -> dict[tuple, Any]:
    reason = case.unscoped_outputs.get(name)
    if reason is None:
        return _validate_source_snapshot_ids(case.id, name, results)
    _validate_unscoped_output(case.id, name, results, reason)
    return {}


def _family_signature(family: Any) -> dict[str, Any] | None:
    """Return path-independent family identity, procedure, and member cells."""
    if family is None:
        return None
    return {
        "name": family.name,
        "procedure": None if family.family is None else family.family.model_dump(mode="json"),
        "members": tuple(
            sorted(
                (cell.model_dump(mode="json", exclude={"source"}) for cell in family.members),
                key=lambda value: str(sorted(value.items())),
            )
        ),
    }


def _record(
    out: dict[tuple, dict[str, dict[str, Any]]], key: tuple, method: str, payload: dict[str, Any]
) -> None:
    methods = out.setdefault(key, {})
    if method in methods:
        raise AssertionError(
            f"a path emitted more than one row for {key}/{method}: a duplicate (for example a "
            "join fan-out) must not compare equal to a single row"
        )
    methods[method] = payload


def _normalize(estimates: Any, *, family_for_row: dict[tuple, Any] | None = None):
    """Normalize path identities while retaining the family contract in each row payload."""
    from increment.breakout.estimates import DailyMetricValue
    from increment.estimation.contrast_results import ContrastResult
    from increment.estimation.readout_types import CellKey

    family_for_row = {} if family_for_row is None else family_for_row
    out: dict[tuple, dict[str, dict[str, Any]]] = {}
    for row in estimates:
        source_id = getattr(row, "source_snapshot_id", None)
        family = family_for_row.get((source_id, CellKey.from_row(row)))
        family_signature = _family_signature(family)
        if isinstance(row, ContrastResult):
            key = (
                row.metric,
                row.treatment_group,
                row.estimand,
                "absolute",
                None,
                None,
                None,
                None,
            )
            payload = row.model_dump(mode="json", exclude={"source_snapshot_id", "family_id"})
            payload["_family_scope"] = family_signature
            _record(out, key, row.method, {"contrast": payload})
            continue
        if isinstance(row, DailyMetricValue):
            # A per-day absolute value has no method, estimand or lift; `source`
            # names the resolved fact source, which only warehouse paths know.
            key = (
                row.metric,
                row.group_id,
                "value",
                "absolute",
                row.dimension_value,
                None,
                row.ds,
                row.ds_basis,
            )
            payload = row.model_dump(exclude={"source", "source_snapshot_id"})
            payload["_family_scope"] = family_signature
            _record(out, key, "value", {"daily_value": payload})
            continue
        payload = _row_payload(row)
        payload.pop("source_snapshot_id", None)
        payload.pop("family_id", None)
        payload["_family_scope"] = family_signature
        _record(out, _row_identity(row), row.method, payload)
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
    triggered = getattr(snapshot, "triggered", None)
    if triggered is None:
        return (states, records)
    # A trigger-declared capture retains a second chain; parity holds for both.
    return (states, records, sequential_fingerprint(triggered))


@dataclass(frozen=True)
class SequentialCapture:
    prefix_id: str
    fingerprint: tuple


@dataclass(frozen=True)
class CaseResult:
    rows: dict[str, dict[tuple, dict[str, dict[str, Any]]]]
    refusals: dict[str, str]
    sequential_state: dict[str, SequentialCapture] = field(default_factory=dict)
    # Ingresses whose constructor signature or schema cannot express the request: name ->
    # the `Absence` the attempt raised (a `TypeError` for an unsupported keyword). They
    # have no refusal code and are never row-compared.
    absences: dict[str, Absence] = field(default_factory=dict)


def _read(analysis: Any, case: ParityCase) -> Any:
    """Read the one method `case` names: `run`/`run_breakout`, or a single day-axis method
    (`run_daily`, `run_daily_lift`, `run_asof`, `run_asof_lift`). A day-axis value series and
    its lift are separate cases, so one refusing never hides the other's rows."""
    estimand_kwargs = {"estimands": case.estimands} if case.estimands else {}
    prior_kwargs = {"prior": case.prior} if case.prior is not UNSET else {}
    metric_kwargs = {"metrics": list(case.metrics)} if case.metrics is not None else {}
    if case.view is None:
        if case.breakout_dimension:
            return analysis.run_breakout(**estimand_kwargs, **prior_kwargs, **metric_kwargs)
        return analysis.run(**estimand_kwargs, **prior_kwargs, **metric_kwargs)
    dimension = {"dimension": case.breakout_dimension} if case.breakout_dimension else {}
    population = {"population": case.population}
    match case.view:
        case "daily":
            return analysis.run_daily(**metric_kwargs, **dimension, **population)
        case "daily_lift":
            return analysis.run_daily_lift(
                **prior_kwargs, **metric_kwargs, **dimension, **population
            )
        case "asof":
            return analysis.run_asof(**metric_kwargs, **dimension, **population)
        case _:
            return analysis.run_asof_lift(
                **estimand_kwargs, **prior_kwargs, **metric_kwargs, **dimension, **population
            )


def _assert_names_absent_field(case_id: str, name: str, exc: Exception, absent: Absence) -> None:
    """The constructor failed because of the declared field, not for another reason.

    A schema absence must carry a validation error that locates the field and rejects it as
    an unknown input. A keyword absence must name a keyword the constructor's signature does
    not accept. A case declaring several unsupported inputs is held only to the one it names,
    so each is checked by the cell that declares it alone.
    """
    if isinstance(exc, ValidationError):
        located = [
            error
            for error in exc.errors()
            if absent.field in error["loc"] and error["type"] == "extra_forbidden"
        ]
        assert located, (
            f"{case_id}: {name} raised a validation error, but none rejects {absent.field!r} "
            f"as an unknown input: {[(e['loc'], e['type']) for e in exc.errors()]}"
        )
        return
    parameters = inspect.signature(getattr(Analysis, name)).parameters.values()
    assert not any(
        p.name == absent.field or p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters
    ), f"{case_id}: {name} accepts {absent.field!r}, so its absence is not structural"
    assert absent.field in str(exc), (
        f"{case_id}: {name} raised {type(exc).__name__} without naming {absent.field!r}: {exc}"
    )


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
    absences: dict[str, Absence] = {}
    sequential_state: dict[str, SequentialCapture] = {}
    for name, build in case.build.items():
        analysis = None
        try:
            try:
                analysis = build()
            except Exception as exc:
                # Absence is the constructor refusing to express the request. An error of
                # the same type raised later, once it accepted it, is a defect and propagates.
                absent = case.expected_absence.get(name)
                if absent is None:
                    raise
                if type(exc) is not absent.error:
                    raise AssertionError(
                        f"{case.id}: {name} was declared absent via {absent.error.__name__} "
                        f"but raised {type(exc).__name__}: {exc}"
                    ) from exc
                _assert_names_absent_field(case.id, name, exc, absent)
                absences[name] = absent
                continue
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
                results = _read(analysis, case)
                if case.readout_probe is not None:
                    case.readout_probe(results)
                if case.source_probe is not None:
                    case.source_probe(name, analysis)
            if expected_warnings:
                assert caught is not None
                assert set(warning_codes(caught)) == set(expected_warnings)
            family_for_row = _family_map_for_output(case, name, results)
            rows[name] = _normalize(results, family_for_row=family_for_row)
        except Exception as exc:
            if not isinstance(exc, CodedError):
                raise
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
    return CaseResult(
        rows=rows, refusals=refusals, sequential_state=sequential_state, absences=absences
    )


def _assert_payload_equal(
    case_id: str, name: str, oracle_name: str, key: tuple, method: str, expected: dict, actual: dict
) -> None:
    assert actual.keys() == expected.keys(), (
        f"{case_id}: {name} vs {oracle_name} compare different fields for {key}/{method}"
    )
    for field_name, e in expected.items():
        a = actual[field_name]
        assert nested_close(e, a), (
            f"{case_id}: {name} vs {oracle_name} disagree on {field_name} "
            f"for {key}/{method}: {a!r} != {e!r}"
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
        elif name in case.expected_absence:
            assert name in case.build, (
                f"{case.id}: {name} has an expected_absence entry but is not in "
                "build -- a declared absence requires actually attempting the constructor"
            )
        else:
            assert name not in case.build, (
                f"{case.id}: {name} is in build but only reason-waived (no "
                "waived_refusal_codes entry) -- either attempt it for real "
                "comparison or add the code it is expected to raise"
            )
    for name in case.expected_absence:
        assert name in case.waive, f"{case.id}: {name} is declared absent but has no waive reason"


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
    for name, absent in case.expected_absence.items():
        assert result.absences.get(name) == absent, (
            f"{case.id}: {name} was declared absent via {absent.error.__name__} on "
            f"{absent.field!r} but the attempt did not raise it -- the signature now "
            "expresses the request"
        )
    if case.refusal_only:
        # Every attempted ingress must refuse or be absent: an unwaived constructor that
        # produced rows would otherwise self-compare as `live` and pass.
        assert case.build, f"{case.id}: refusal_only but no ingress was attempted"
        assert not result.rows, (
            f"{case.id}: refusal_only but {sorted(result.rows)} produced rows instead of refusing"
        )
        assert set(case.build) == set(result.refusals) | set(result.absences), (
            f"{case.id}: every attempted ingress must refuse or be absent, but "
            f"{sorted(set(case.build) - set(result.refusals) - set(result.absences))} did neither"
        )
        return
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
