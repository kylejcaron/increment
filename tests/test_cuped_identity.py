from __future__ import annotations

import json
import math
import pickle
from collections.abc import Iterable
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pyarrow as pa
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from increment import AnalysisPlan, CupedCovariate, Method, MetricSpec, from_definitions, from_frame
from increment.errors import CodedError
from increment.estimation.decision_types import PValueEvidence
from increment.estimation.readout_types import ReadoutResults
from increment.estimation.results import LiftEstimate
from tests._override_warnings import allow_declared_overrides
from tests.analysis_factory import view_readout
from tests.covariate_cases import allow_tenure_imbalance


def _method(kind: str, *, name: str, covariate: str) -> Method:
    if kind == "naive":
        return Method.unadjusted(name=name)
    declaration = CupedCovariate(name=covariate, kind="continuous")
    factory = Method.cuped if kind == "scalar" else Method.cuped_multicovariate
    return factory(name=name, covariates=declaration)


def _frame_readout(
    decision: str,
    companion: str | None,
    *,
    panel: bool,
    companion_scale: float = 1.0,
    decision_scale: float = 1.0,
    role: Literal["primary", "secondary"] = "primary",
    encouragement: bool = False,
):
    rows = [
        {
            "unit": f"{arm}-{i}",
            "arm": arm,
            "day": 1,
            "exposure": 1,
            "revenue": 20.0 + (i % 11) + 0.3 * (i % 3) + 2.0 * arm_index,
            "y": decision_scale * float(i % 11),
            "__covariate_y": companion_scale * float((3 * i) % 13),
            "uptake": int(arm_index > 0 and i % 4 != 0),
        }
        for arm_index, arm in enumerate(("control", "treatment", "other"))
        for i in range(64)
    ]
    method = _method(decision, name="decision", covariate="y")
    sensitivities = (
        ()
        if companion is None
        else (_method(companion, name="sensitivity", covariate="__covariate_y"),)
    )
    kwargs: dict[str, Any] = {
        "unit": "unit",
        "group": "arm",
        "control": "control",
        "metrics": [
            MetricSpec(name="revenue", decision_method=method, sensitivity_methods=sensitivities)
        ],
        "plan": AnalysisPlan(primary="revenue")
        if role == "primary"
        else AnalysisPlan(secondaries=["revenue"]),
    }
    if panel:
        kwargs["date"] = "day"
    if encouragement:
        from increment import Encouragement, UptakeSpec

        kwargs.pop("control")
        kwargs["design"] = Encouragement(control_group="control", uptake=UptakeSpec(fact="uptake"))
        kwargs["uptake"] = "uptake"
    table = pa.Table.from_pylist(rows)
    experiment, resolver = (
        from_frame(table, shape="panel", **kwargs)
        if panel
        else from_frame(table, shape="summary", **kwargs)
    )
    return view_readout(experiment, resolver, estimands=("itt",))


@pytest.mark.grid(
    "Decision/companion evidence types are crossed with panel and role because identity preservation is independently required for each output family."
)
@pytest.mark.parametrize("panel", [False, True])
@pytest.mark.parametrize("role", ["primary", "secondary"])
@pytest.mark.parametrize(
    ("decision", "companion"),
    [
        ("scalar", "scalar"),
        ("scalar", "mixed"),
        ("mixed", "scalar"),
        ("mixed", "mixed"),
        ("naive", "scalar"),
        ("naive", "mixed"),
    ],
)
def test_decision_family_ignores_companion_evidence(panel, role, decision, companion):
    first = _frame_readout(decision, companion, panel=panel, role=role)
    changed = _frame_readout(decision, companion, panel=panel, role=role, companion_scale=2.0)
    alone = _frame_readout(decision, None, panel=panel, role=role)
    decision_rows = [
        {
            row.group_id: row
            for row in result
            if isinstance(row, LiftEstimate) and row.method_role == "decision"
        }
        for result in (first, changed, alone)
    ]
    assert (
        first.metadata is not None and changed.metadata is not None and alone.metadata is not None
    )
    assert first.metadata.scope.snapshot_id != changed.metadata.scope.snapshot_id
    assert first.metadata.scope.snapshot_id != alone.metadata.scope.snapshot_id
    for group, original in decision_rows[0].items():
        assert original.family_id is not None
        for other in decision_rows[1:]:
            assert original.require_lift().value == pytest.approx(
                other[group].require_lift().value, rel=1e-10
            )
            assert original.require_lift().log_se == pytest.approx(
                other[group].require_lift().log_se, rel=1e-10
            )
            assert original.family_id == other[group].family_id
    evidence = []
    for result in (first, changed, alone):
        assert result.metadata is not None
        evidence.append(
            {
                record.cell.group_id: record.sampling.evidence
                for record in result.metadata.cells
                if record.cell.method_role == "decision" and record.sampling is not None
            }
        )
    for group, original in evidence[0].items():
        assert isinstance(original, PValueEvidence)
        for other in evidence[1:]:
            assert isinstance(other[group], PValueEvidence)
            assert original.hypothesis == other[group].hypothesis
            assert original.reference == other[group].reference
            assert original.p_value == pytest.approx(other[group].p_value, rel=1e-10)
    decision_snapshots = []
    for result in (first, changed, alone):
        assert result.metadata is not None
        assert all(
            family.source_snapshot_id == result.metadata.scope.snapshot_id
            for family in result.metadata.scope.families
        )
        decision_snapshots.append(
            {
                family.source_snapshot_id
                if family.decision_snapshot_id is None
                else family.decision_snapshot_id
                for family in result.metadata.scope.families
            }
        )
        assert (
            ReadoutResults.model_validate_json(result.model_dump_json()).metadata == result.metadata
        )
    assert decision_snapshots[0] == decision_snapshots[1] == decision_snapshots[2]
    if decision != "naive":
        changed_decision = _frame_readout(
            decision, companion, panel=panel, role=role, decision_scale=2.0
        )
        assert next(
            row.family_id for row in changed_decision if row.method_role == "decision"
        ) != next(row.family_id for row in first if row.method_role == "decision")


def test_serialized_decision_identity_cannot_be_replaced():
    result = _frame_readout("scalar", "scalar", panel=False)
    payload = json.loads(result.model_dump_json())
    payload["metadata"]["scope"]["families"][0]["decision_snapshot_id"] = "sha256:" + "0" * 64
    with pytest.raises(CodedError) as raised:
        ReadoutResults.model_validate_json(json.dumps(payload))
    assert raised.value.code == "readout.serialization.identity_mismatch"


@pytest.mark.parametrize("panel", [False, True])
@pytest.mark.parametrize("role", ["primary", "secondary"])
@pytest.mark.parametrize("decision", ["scalar", "naive"])
def test_encouragement_decision_identity_ignores_companion_evidence(panel, role, decision):
    first = _frame_readout(decision, "scalar", panel=panel, role=role, encouragement=True)
    changed = _frame_readout(
        decision,
        "scalar",
        panel=panel,
        role=role,
        encouragement=True,
        companion_scale=2.0,
    )
    alone = _frame_readout(decision, None, panel=panel, role=role, encouragement=True)
    assert first.metadata is not None and changed.metadata is not None
    assert first.metadata.scope.snapshot_id != changed.metadata.scope.snapshot_id
    originals = {
        row.group_id: row
        for row in first
        if isinstance(row, LiftEstimate) and row.method_role == "decision"
    }
    for result in (changed, alone):
        for row in result:
            if isinstance(row, LiftEstimate) and row.method_role == "decision":
                assert originals[row.group_id].family_id == row.family_id
                assert originals[row.group_id].require_lift().value == pytest.approx(
                    row.require_lift().value, rel=1e-10
                )
                assert originals[row.group_id].require_lift().log_se == pytest.approx(
                    row.require_lift().log_se, rel=1e-10
                )
        assert (
            ReadoutResults.model_validate_json(result.model_dump_json()).metadata == result.metadata
        )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("route", ["native", "artifact"])
@pytest.mark.parametrize("decision_kind", ["scalar", "mixed", "naive"])
@allow_declared_overrides
@allow_tenure_imbalance
def test_definitions_and_artifact_keep_decision_family_when_companion_changes(
    tmp_path, route, decision_kind
):
    from tests.test_artifact_covariates import _publish_and_reopen
    from tests.test_cuped_covariate_sources import _scalar_native_case

    path, con, methods, *_ = _scalar_native_case(tmp_path)
    if decision_kind == "scalar":
        decision = methods[0]
    elif decision_kind == "mixed":
        decision = Method.cuped_multicovariate(
            name="decision",
            covariates=tuple(
                covariate.model_copy(update={"missing": "refuse"})
                for covariate in methods[0].covariates
            ),
        )
    else:
        decision = Method.unadjusted(name="decision")
    if route == "artifact":
        (experiment, resolver), _manifest = _publish_and_reopen(
            path, con, request_covariate=False, full_catalog=True
        )
    else:
        experiment, resolver = from_definitions(path, con, experiment_name="cov_test")
    try:
        with cast(Any, resolver):
            first = view_readout(
                experiment, resolver, decision_method=decision, sensitivity_methods=(methods[1],)
            )
            if route == "native":
                con.raw_sql("UPDATE cov_events SET history_b = 2 * history_b")
                companion = methods[1]
            else:
                companion = Method.cuped_multicovariate(
                    name="alternate", covariates=methods[1].covariates
                )
            changed = view_readout(
                experiment, resolver, decision_method=decision, sensitivity_methods=(companion,)
            )
            alone = view_readout(
                experiment, resolver, decision_method=decision, sensitivity_methods=()
            )
        assert first.metadata is not None and changed.metadata is not None
        assert first.metadata.scope.snapshot_id != changed.metadata.scope.snapshot_id
        original = next(
            row for row in first if isinstance(row, LiftEstimate) and row.method_role == "decision"
        )
        assert original.family_id is not None
        for result in (changed, alone):
            (row,) = [
                row
                for row in result
                if isinstance(row, LiftEstimate) and row.method_role == "decision"
            ]
            assert original.family_id == row.family_id
            assert original.require_lift().value == pytest.approx(
                row.require_lift().value, rel=1e-10
            )
            assert original.require_lift().log_se == pytest.approx(
                row.require_lift().log_se, rel=1e-10
            )
    finally:
        con.disconnect()


@pytest.mark.parametrize("panel", [False, True])
@pytest.mark.parametrize(
    ("role", "companion", "snapshot", "family"),
    [
        (
            "primary",
            None,
            "f00bd3ac8ecda3864a732fc8b076a65615499ede90a5cc9f5e41b769ab33456d",
            "fb168e77d6fef186f2c31e65fc5fa4c163bac4494665d91a96454d32bd6c1f5c",
        ),
        (
            "secondary",
            None,
            "2d16882bf9f3947e9123ad9bd3d89b8b5b96494e79e347b803d09bc7b8011900",
            "166c482c55d171d548166428b5c236db21818b365ee243f35350297c551f0a21",
        ),
        (
            "primary",
            "naive",
            "45aadbdf8e1acea8c960aa829d18a30b2f1e9b871cb3b22ac2132db4fc61b675",
            "94c32075ee60f7dbca390ab23537f4c02b96ac02b9bdee955127c51435edc1dd",
        ),
        (
            "secondary",
            "naive",
            "c3124309dee28b7e6b1e4d8d4550f7f0004d2ed828009700619521b0c0afa519",
            "0b667cc4d1b115189739f949024cc0fd6e7945dacca0c30e9784c4c4e66a479d",
        ),
    ],
)
def test_no_covariate_snapshot_and_family_identities_stay_pinned(
    panel, role, companion, snapshot, family
):
    result = _frame_readout("naive", companion, panel=panel, role=role)
    assert result.metadata is not None
    assert result.metadata.scope.snapshot_id == f"sha256:{snapshot}"
    assert {item.family_id for item in result.metadata.scope.families} == {f"sha256:{family}"}


def test_prechange_schema_two_saved_result_loads_without_identity_changes():
    text = (
        Path(__file__).parent / "fixtures" / "readout_schema2_e8949261_prechange.json"
    ).read_text()
    payload = json.loads(text)
    assert payload["schema_version"] == 2
    assert all(
        "decision_snapshot_id" not in family for family in payload["metadata"]["scope"]["families"]
    )
    result = ReadoutResults.model_validate_json(text)
    assert json.loads(result.model_dump_json()) == payload
    assert result.metadata is not None
    assert result.metadata.scope.snapshot_id == (
        "sha256:835b937b88aedfe9f17f8b132b63f5962c195eb478a9140a5467148c6384434f"
    )
    family = result.metadata.scope.families[0]
    assert family.decision_snapshot_id is None
    assert family.family_id == (
        "sha256:91c648d63721c68775efd37ff934f16468c9b09b9e7aebb49f1f87c7ca516374"
    )
    assert family.complete


@pytest.mark.parametrize("copy_kind", ["pickle", "deepcopy"])
def test_family_scope_copy_preserves_nonempty_decision_snapshot(copy_kind):
    result = _frame_readout("scalar", "scalar", panel=False, role="primary")
    assert result.metadata is not None
    family = result.metadata.scope.families[0]
    assert family.decision_snapshot_id is not None
    copied = pickle.loads(pickle.dumps(family)) if copy_kind == "pickle" else deepcopy(family)
    assert copied == family
    assert copied.family_id == family.family_id
    assert copied.decision_snapshot_id == family.decision_snapshot_id
    assert copied.model_dump_json() == family.model_dump_json()


# Offsets that cancel within an arm (summary) or within a unit's days (panel): a summation
# order that adds the small values between them keeps their sum, another loses it.
_CANCELLING = (1e16, -1e16)
_COVARIATES = (
    CupedCovariate(name="x1", kind="continuous"),
    CupedCovariate(name="x2", kind="continuous"),
)
_IDENTITY_METHODS = {
    "unadjusted": Method.unadjusted(),
    "scalar": Method.cuped(covariates=_COVARIATES[0]),
    "mixed": Method.cuped_multicovariate(name="linear", covariates=_COVARIATES),
}


def _identity_rows(*, days: int = 1, n_units: int = 2000) -> list[dict[str, Any]]:
    """Non-integer outcomes and covariates: integer data sums exactly in any order."""
    rng = np.random.default_rng(11)
    rows = []
    for unit in range(n_units):
        treated = unit % 2
        x1, x2 = rng.normal(50.0, 10.0), rng.normal(3.0, 1.0)
        total = 100.0 + 0.8 * x1 + 2.0 * x2 + 1.5 * treated + rng.normal(0.0, 5.0)
        shares = rng.dirichlet(np.ones(days)) if days > 1 else (1.0,)
        for day, share in enumerate(shares):
            value = total * share
            if days > 1 and day < len(_CANCELLING):
                value += _CANCELLING[day]
            elif days == 1 and unit < 2 * len(_CANCELLING):
                value += _CANCELLING[unit // 2]
            rows.append(
                {
                    "unit": f"u{unit}",
                    "arm": "treatment" if treated else "control",
                    "day": day + 1,
                    "y": value,
                    "x1": x1,
                    "x2": x2,
                }
            )
    return rows


def _identity(frame: Any, method: str, *, panel: bool) -> tuple[str, str | None]:
    """The snapshot id and source content digest of one frame readout."""
    import warnings

    from increment.errors import IncrementWarning

    kwargs: dict[str, Any] = {
        "unit": "unit",
        "group": "arm",
        "control": "control",
        "metrics": [MetricSpec(name="y", decision_method=_IDENTITY_METHODS[method])],
    }
    experiment, resolver = (
        from_frame(frame, shape="panel", date="day", **kwargs)
        if panel
        else from_frame(frame, shape="summary", **kwargs)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IncrementWarning)
        result = view_readout(experiment, resolver)
    assert result.metadata is not None
    return result.metadata.scope.snapshot_id, result.metadata.source_content_digest


def _polars(rows: list[dict[str, Any]], *, chunk: int | None = None) -> Any:
    import polars as pl

    if chunk is None:
        return pl.DataFrame(rows)
    parts = [pl.DataFrame(rows[start : start + chunk]) for start in range(0, len(rows), chunk)]
    return pl.concat(parts, rechunk=False)


def _record_batches(rows: list[dict[str, Any]], *, chunk: int = 16) -> pa.Table:
    """A pyarrow table of many record batches, as data read from files arrives."""
    return pa.Table.from_batches(
        [
            pa.RecordBatch.from_pylist(rows[start : start + chunk])
            for start in range(0, len(rows), chunk)
        ]
    )


@pytest.fixture
def arrow_threads():
    """Arrow's multithreaded execution, which the test session limits to one thread."""
    previous = pa.cpu_count()
    pa.set_cpu_count(4)
    yield
    pa.set_cpu_count(previous)


@pytest.mark.parametrize("panel", [False, True], ids=["summary", "panel"])
@pytest.mark.parametrize("decision", ["scalar", "mixed"])
def test_frame_cuped_snapshot_id_is_a_function_of_the_rows(decision, panel):
    rows = _identity_rows()
    shuffled = [rows[index] for index in np.random.default_rng(3).permutation(len(rows))]
    frames = (_polars(rows), _polars(rows), _polars(shuffled), _polars(rows, chunk=64))
    identities = {_identity(frame, decision, panel=panel) for frame in frames}
    assert len(identities) == 1, identities


@pytest.mark.parametrize("library", ["polars", "pandas", "pyarrow"])
def test_panel_snapshot_id_survives_reordering_days_within_units(library, arrow_threads):
    import pandas as pd

    rows = _identity_rows(days=3, n_units=600)
    reordered = [
        row for start in range(0, len(rows), 3) for row in reversed(rows[start : start + 3])
    ]
    rotated = [
        row
        for start in range(0, len(rows), 3)
        for row in (*rows[start + 1 : start + 3], rows[start])
    ]
    build = {
        "polars": _polars,
        "pandas": pd.DataFrame,
        "pyarrow": _record_batches,
    }[library]
    identities = {
        method: {
            _identity(build(order), method, panel=True) for order in (rows, reordered, rotated)
        }
        for method in ("unadjusted", "mixed")
    }
    assert all(len(found) == 1 for found in identities.values()), identities


@pytest.mark.parametrize("method", ["unadjusted", "scalar", "mixed"])
def test_frame_snapshot_id_is_the_same_across_dataframe_libraries(method):
    import pandas as pd

    rows = _identity_rows()
    frames = (_polars(rows), pd.DataFrame(rows), pa.Table.from_pylist(rows))
    identities = {_identity(frame, method, panel=False) for frame in frames}
    assert len(identities) == 1, identities


def _rounded_once(values: Iterable[float]) -> float:
    """The exact sum of *values* rounded once, to ±inf beyond the largest float."""
    total = Fraction(0)
    for value in values:
        total += Fraction(value)
    try:
        return float(total)
    except OverflowError:
        return math.inf if total > 0 else -math.inf


_FINITE = st.floats(allow_nan=False, allow_infinity=False)


@settings(max_examples=300, deadline=None)
@given(
    values=st.lists(
        st.one_of(
            _FINITE,
            st.sampled_from([1e16, -1e16, 1.0, 1e308, -1e308, 1.7976931348623157e308, 5e-324]),
        ),
        max_size=40,
    ),
    data=st.data(),
)
@example(values=[1e16, 1.0, -1e16], data=None)
@example(values=[1e308, 1e308, -1e308], data=None)
@example(values=[1.7976931348623157e308, 1.7976931348623157e308], data=None)
@example(values=[5e-324, -5e-324, 2.2250738585072014e-308], data=None)
@example(values=[1e300, 3.0, 4.0, 1e-300, 5.0, 6.0, -1e300], data=None)
def test_exact_sum_rounds_the_exact_total_once_in_any_order(values, data):
    from increment._frame_moments import _exact_group_sums, exact_sum

    groups = [index % 3 for index in range(len(values))]
    if data is not None:
        groups = data.draw(st.lists(st.integers(0, 2), min_size=len(values), max_size=len(values)))
        order = data.draw(st.permutations(range(len(values))))
    else:
        order = list(reversed(range(len(values))))
    assert exact_sum(values) == _rounded_once(values)
    assert exact_sum([values[index] for index in order]) == _rounded_once(values)
    codes = np.asarray([groups[index] for index in order], dtype=np.int64)
    permuted = np.asarray([values[index] for index in order], dtype=np.float64)
    # Pairs of 1.0 and -1.0 leave each group's total unchanged and outnumber the 3 x 2048
    # group-exponent bins, so the padded sums take the binned route.
    padding_codes = np.repeat(np.arange(3, dtype=np.int64), 2 * 2048)
    padding = np.tile([1.0, -1.0], 3 * 2048)
    # Values more than 2**96 apart in one group (1e300 beside 1e-300) take the exact_sum route.
    routes = (
        _exact_group_sums(codes, 3, permuted),
        _exact_group_sums(
            np.concatenate([codes, padding_codes]), 3, np.concatenate([permuted, padding])
        ),
    )
    for group in range(3):
        expected = _rounded_once(
            value for value, label in zip(values, groups, strict=True) if label == group
        )
        assert [route[group] for route in routes] == [expected] * len(routes)
