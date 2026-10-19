from __future__ import annotations

import json
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from increment import Analysis, Method, MetricSpec
from increment._canonical import canonical_json_bytes
from increment.errors import CodedError, WireFormatError
from increment.semantics.models import AnalysisPlan, CupedCovariate

pytestmark = pytest.mark.integration

_SNAPSHOT_FIELDS = {"source_snapshot_id", "family_id"}


def _records(rows_per_cell=(41, 29, 37, 33), offset=0):
    records = []
    for a, arm in enumerate(("control", "treatment")):
        for cell, n in enumerate(rows_per_cell):
            for i in range(n):
                noise = math.sin(0.7 * (i + offset) + a)
                records.append(
                    {
                        "unit": f"{arm}-{cell}-{i + offset}",
                        "arm": arm,
                        "outcome": 10 + 0.5 * a + 0.2 * cell + noise,
                        "den": 3 + 0.1 * cell + 0.1 * noise,
                        "history_a": float((i * 3) % 11) + noise * 0.1,
                        "history_b": float((i * 7) % 13) + noise * 0.2,
                        "segment": "A" if cell < 2 else "B",
                        "device": "mobile" if cell % 2 else "desktop",
                    }
                )
    return records


def _methods(family):
    scalars = tuple(
        Method.cuped(name=label, covariates=CupedCovariate(name=column, kind="continuous"))
        for label, column in (("primary_history", "history_a"), ("other_history", "history_b"))
    )
    mixed = Method.cuped_multicovariate(
        name="crossed_categories",
        covariates=(
            CupedCovariate(name="segment", kind="categorical"),
            CupedCovariate(name="device", kind="categorical"),
        ),
    )
    return {"scalar": scalars, "categorical": (mixed,), "combined": (*scalars, mixed)}[family]


def _spec(ratio, methods):
    return MetricSpec(
        name="outcome",
        type="ratio" if ratio else "mean",
        numerator="outcome" if ratio else None,
        denominator="den" if ratio else None,
        decision_method=methods[0],
        sensitivity_methods=methods[1:],
    )


def _case(*, ratio=False, family="scalar", records=None, alpha=0.05):
    spec = _spec(ratio, _methods(family))
    analysis = Analysis.from_unit_summary(
        pa.Table.from_pylist(_records() if records is None else records),
        unit="unit",
        group="arm",
        control="control",
        experiment_id="compact",
        metrics=[spec],
        plan=AnalysisPlan(alpha=alpha),
    )
    return analysis, spec


def _export_rows(analysis, tmp_path, name="compact.parquet"):
    destination = tmp_path / name
    analysis.export(destination)
    table = pq.read_table(destination)
    return table, table.to_pylist()


def _same_result(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _same_result(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            _same_result(a, b)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-10, abs=1e-12)
    else:
        assert actual == expected


def _dumps(rows, *, exclude=frozenset()):
    return [row.model_dump(exclude=set(exclude)) for row in rows]


@pytest.mark.parametrize("ratio", [False, True])
@pytest.mark.parametrize("family", ["scalar", "categorical", "combined"])
def test_format13_method_scoped_export_reloads_exact_inference(tmp_path, ratio, family):
    analysis, spec = _case(ratio=ratio, family=family)
    try:
        expected = analysis.run()
        table, rows = _export_rows(analysis, tmp_path)
        assert {row["moments_format"] for row in rows} == {13}
        assert table.schema.metadata[b"increment.moments_format"] == b"13"
        ordinary = [row for row in rows if row.get("record_kind") != "cuped_evidence"]
        assert len({(row["metric"], row["group_id"]) for row in ordinary}) == len(ordinary) == 2
        envelopes = [row for row in rows if row.get("record_kind") == "cuped_evidence"]
        assert len(envelopes) == {"scalar": 2, "categorical": 1, "combined": 3}[family]
        for row in envelopes:
            payload = json.loads(row["cuped_evidence"])
            assert payload["version"] == 1
            assert row["cuped_evidence"] == canonical_json_bytes(payload).decode()
        aggregate = family != "scalar"
        exclude = _SNAPSHOT_FIELDS if aggregate else frozenset()
        restored = Analysis.from_moments(
            rows, metrics=[spec], control="control", experiment_id="compact"
        )
        again = Analysis.from_moments(
            rows, metrics=[spec], control="control", experiment_id="compact"
        )
        try:
            actual = restored.run()
            _same_result(_dumps(actual, exclude=exclude), _dumps(expected, exclude=exclude))
            assert actual.metadata is not None and expected.metadata is not None
            replayed = again.run()
            assert replayed.metadata is not None
            assert replayed.metadata.scope.snapshot_id == actual.metadata.scope.snapshot_id
            if aggregate:
                assert actual.metadata.scope.snapshot_id != expected.metadata.scope.snapshot_id
            else:
                assert actual.metadata.scope.snapshot_id == expected.metadata.scope.snapshot_id
            for alpha in (0.01, 0.2):
                changed, _ = _case(ratio=ratio, family=family, alpha=alpha)
                replay = Analysis.from_moments(
                    rows,
                    metrics=[spec],
                    control="control",
                    experiment_id="compact",
                    plan=AnalysisPlan(alpha=alpha),
                )
                try:
                    _same_result(
                        _dumps(replay.run(), exclude=exclude),
                        _dumps(changed.run(), exclude=exclude),
                    )
                finally:
                    replay.close()
                    changed.close()
        finally:
            restored.close()
            again.close()
    finally:
        analysis.close()


def test_three_arm_envelopes_are_row_order_invariant(tmp_path):
    records = [
        {
            "unit": f"{arm}-{index}",
            "arm": arm,
            "outcome": 10 + arm_index * 2 + index,
            "den": 3 + index % 3,
            "history_a": float(index),
            "history_b": float(index * 2),
            "segment": "A" if index % 2 else "B",
            "device": "mobile" if index % 3 else "desktop",
        }
        for arm_index, arm in enumerate(("control", "treatment_a", "treatment_b"))
        for index in range(24)
    ]
    spec = _spec(False, _methods("categorical"))
    exports = []
    snapshots = []
    for index, source_rows in enumerate((records, list(reversed(records)))):
        analysis = Analysis.from_unit_summary(
            pa.Table.from_pylist(source_rows),
            unit="unit",
            group="arm",
            control="control",
            experiment_id="three-arm",
            metrics=[spec],
            plan=AnalysisPlan(),
        )
        try:
            _, rows = _export_rows(analysis, tmp_path, f"three-arm-{index}.parquet")
            exports.append(
                sorted(
                    row["cuped_evidence"]
                    for row in rows
                    if row.get("record_kind") == "cuped_evidence"
                )
            )
            restored = Analysis.from_moments(
                rows, metrics=[spec], control="control", experiment_id="three-arm"
            )
            try:
                result = restored.run()
                assert result.metadata is not None
                snapshots.append(result.metadata.scope.snapshot_id)
            finally:
                restored.close()
        finally:
            analysis.close()
    assert exports[0] == exports[1]
    assert json.loads(exports[0][0])["evidence"]["arms"] == [
        "control",
        "treatment_a",
        "treatment_b",
    ]
    assert snapshots[0] == snapshots[1]


def test_single_scalar_family_export_remains_format_twelve(tmp_path):
    analysis, _ = _case(family="scalar", records=None)
    single = _spec(False, _methods("scalar")[:1])
    analysis.close()
    analysis = Analysis.from_unit_summary(
        pa.Table.from_pylist(_records()),
        unit="unit",
        group="arm",
        control="control",
        experiment_id="compact",
        metrics=[single],
    )
    try:
        table, rows = _export_rows(analysis, tmp_path)
        assert {row["moments_format"] for row in rows} == {12}
        assert "record_kind" not in table.column_names
        assert "cuped_evidence" not in table.column_names
    finally:
        analysis.close()


def _categorical_rows(tmp_path):
    analysis, spec = _case(family="combined")
    try:
        return _export_rows(analysis, tmp_path)[1], spec
    finally:
        analysis.close()


def _rewrite(row, change):
    payload = json.loads(row["cuped_evidence"])
    change(payload)
    return {**row, "cuped_evidence": canonical_json_bytes(payload).decode()}


def _first_envelope(rows, kind):
    return next(
        index
        for index, row in enumerate(rows)
        if row.get("record_kind") == "cuped_evidence"
        and json.loads(row["cuped_evidence"])["kind"] == kind
    )


_MALFORMED = {
    "future version": lambda rows: _replace_payload(rows, lambda p: p.update(version=2)),
    "reserved kind": lambda rows: _replace_payload(
        rows, lambda p: p.update(kind="fitted_multicovariate")
    ),
    "unknown kind": lambda rows: _replace_payload(rows, lambda p: p.update(kind="matrix")),
    "unknown field": lambda rows: _replace_payload(rows, lambda p: p.update(extra=1)),
    "missing field": lambda rows: _replace_payload(rows, lambda p: p.pop("population")),
    "wrong source": lambda rows: _replace_payload(rows, lambda p: p.update(source_sha256="0" * 64)),
    "unknown metric": lambda rows: _replace_payload(rows, lambda p: p.update(metric="other")),
    "unknown population": lambda rows: _replace_payload(rows, lambda p: p.update(population="all")),
    "breakout coordinate": lambda rows: _replace_payload(
        rows, lambda p: p.update(breakout="device")
    ),
    "noncanonical bytes": lambda rows: [
        {**row, "cuped_evidence": row["cuped_evidence"].replace(",", ", ", 1)}
        if index == _first_envelope(rows, "categorical_cells")
        else row
        for index, row in enumerate(rows)
    ],
    "duplicate envelope": lambda rows: [*rows, rows[_first_envelope(rows, "categorical_cells")]],
    "conflicting envelope": lambda rows: [
        *rows,
        _rewrite(
            rows[_first_envelope(rows, "categorical_cells")],
            lambda p: p["evidence"]["counts"][0].__setitem__(0, p["evidence"]["counts"][0][0] + 1),
        ),
    ],
    "overloaded ordinary column": lambda rows: [
        {**row, "cy2": 1.0} if index == _first_envelope(rows, "categorical_cells") else row
        for index, row in enumerate(rows)
    ],
    "envelope in format twelve": lambda rows: [{**row, "moments_format": 12} for row in rows],
    "format thirteen without envelopes": lambda rows: [
        row for row in rows if row.get("record_kind") != "cuped_evidence"
    ],
    "invalid cells": lambda rows: _replace_payload(
        rows, lambda p: p["evidence"]["outcome_magnitudes"][0].__setitem__(0, -1.0)
    ),
}


def _replace_payload(rows, change):
    index = _first_envelope(rows, "categorical_cells")
    return [
        _rewrite(row, change) if position == index else row for position, row in enumerate(rows)
    ]


def test_evidence_arm_roster_must_match_ordinary_metric_rows(tmp_path):
    rows, spec = _categorical_rows(tmp_path)
    scalar = next(
        index
        for index, row in enumerate(rows)
        if row.get("record_kind") == "cuped_evidence"
        and json.loads(row["cuped_evidence"])["kind"] == "scalar_moments"
    )
    rows[scalar] = _rewrite(
        rows[scalar],
        lambda payload: payload["evidence"]["rows"].pop(),
    )
    with pytest.raises(CodedError) as raised:
        Analysis.from_moments(rows, metrics=[spec], control="control", experiment_id="compact")
    assert raised.value.code == "moments.cuped_evidence.invalid"


def test_categorical_envelope_arm_roster_must_match_ordinary_rows(tmp_path):
    rows, spec = _categorical_rows(tmp_path)
    index = _first_envelope(rows, "categorical_cells")
    rows[index] = _rewrite(
        rows[index],
        lambda payload: payload["evidence"]["arms"].__setitem__(0, "renamed"),
    )
    with pytest.raises(CodedError) as raised:
        Analysis.from_moments(rows, metrics=[spec], control="control", experiment_id="compact")
    assert raised.value.code == "moments.cuped_evidence.invalid"


@pytest.mark.parametrize("mismatch", ["scalar rows", "categorical covariate"])
def test_evidence_payload_must_match_its_declared_method(tmp_path, mismatch):
    rows, spec = _categorical_rows(tmp_path)
    if mismatch == "scalar rows":
        scalar_indexes = [
            index
            for index, row in enumerate(rows)
            if row.get("record_kind") == "cuped_evidence"
            and json.loads(row["cuped_evidence"])["kind"] == "scalar_moments"
        ]
        second = json.loads(rows[scalar_indexes[1]]["cuped_evidence"])
        rows[scalar_indexes[0]] = _rewrite(
            rows[scalar_indexes[0]], lambda payload: payload.update(evidence=second["evidence"])
        )

    else:
        index = _first_envelope(rows, "categorical_cells")
        rows[index] = _rewrite(
            rows[index],
            lambda payload: payload["evidence"]["covariate_names"].__setitem__(0, "different"),
        )
    with pytest.raises(CodedError) as raised:
        Analysis.from_moments(rows, metrics=[spec], control="control", experiment_id="compact")
    assert raised.value.code == "moments.cuped_evidence.invalid"


def test_cancellation_magnitude_underreport_is_rejected_on_envelope_import():
    from increment._cuped_evidence import categorical_payload, parse_cuped_evidence
    from increment.estimation.cuped.cells import categorical_cells

    method = Method.cuped_multicovariate(
        name="category", covariates=CupedCovariate(name="category", kind="categorical")
    )
    parts = []
    for k in (2.0, 3.0):
        outcomes = np.array([-1e100, 1e100, 1.0, -1e100, 1e100, k])
        arms = np.array(["control"] * 3 + ["treatment"] * 3)
        parts.append(
            categorical_cells(
                outcomes,
                arms,
                identity=("cancellation",),
                control="control",
                arm_order=("control", "treatment"),
                covariates={"category": np.repeat("constant", 6)},
            )
        )
    left, right = parts
    payload = categorical_payload(
        "outcome",
        method,
        left,
        population="assigned",
        source={"source": "counterexample"},
    )
    body = json.loads(payload)
    for arm, _values in enumerate(body["evidence"]["outcome_magnitudes"][0]):
        count = sum(body["evidence"]["counts"][arm])
        reference = math.ldexp(
            body["evidence"]["outcome_references"][0][arm],
            -body["evidence"]["outcome_exponents"][0][arm],
        )
        total = math.fsum(
            [
                count * reference,
                *body["evidence"]["centered_first_moments"][arm][0],
            ]
        )
        body["evidence"]["outcome_magnitudes"][0][arm] = abs(total)
    with pytest.raises(CodedError) as raised:
        parse_cuped_evidence(canonical_json_bytes(body).decode())
    assert raised.value.code == "moments.cuped_evidence.invalid"
    with pytest.raises(CodedError) as merged:
        left.merge(right)
    assert merged.value.code == "estimation.cuped_linear.merge_precision"


def test_ratio_categorical_component_count_is_rejected_on_admission(tmp_path):
    analysis, spec = _case(ratio=True, family="categorical")
    try:
        rows = _export_rows(analysis, tmp_path, "ratio-components.parquet")[1]
    finally:
        analysis.close()
    index = _first_envelope(rows, "categorical_cells")

    def drop_denominator(payload):
        cells = payload["evidence"]
        cells["outcome_references"] = [cells["outcome_references"][0]]
        cells["centered_first_moments"] = [[arm[0]] for arm in cells["centered_first_moments"]]
        cells["symmetric_cross_moments"] = [
            [[arm[0][0]]] for arm in cells["symmetric_cross_moments"]
        ]
        cells["outcome_exponents"] = [cells["outcome_exponents"][0]]
        cells["outcome_magnitudes"] = [cells["outcome_magnitudes"][0]]

    rows[index] = _rewrite(rows[index], drop_denominator)
    with pytest.raises(CodedError) as raised:
        Analysis.from_moments(
            rows,
            metrics=[spec],
            control="control",
            experiment_id="compact",
        )
    assert raised.value.code == "moments.cuped_evidence.invalid"


@pytest.mark.parametrize("case", sorted(_MALFORMED))
def test_malformed_cuped_evidence_refuses_before_estimation(tmp_path, case):
    rows, spec = _categorical_rows(tmp_path)
    with pytest.raises(CodedError) as raised:
        Analysis.from_moments(
            _MALFORMED[case](rows), metrics=[spec], control="control", experiment_id="compact"
        )
    assert isinstance(raised.value, WireFormatError) or raised.value.code.startswith(
        "moments.cuped_evidence."
    )
    assert raised.value.code.startswith(("moments.cuped_evidence.", "moments.format."))


@pytest.mark.parametrize(
    ("family", "replacement", "code"),
    [
        (
            "categorical",
            Method.cuped_multicovariate(
                name="crossed_categories",
                covariates=(CupedCovariate(name="segment", kind="categorical"),),
            ),
            "facade.analysis_config.cuped_linear_moments",
        ),
        (
            "scalar",
            Method.cuped(
                name="other_history", covariates=CupedCovariate(name="segment", kind="continuous")
            ),
            "readout.scalar_method_evidence",
        ),
    ],
)
def test_changed_request_without_matching_evidence_refuses(tmp_path, family, replacement, code):
    analysis, spec = _case(family=family)
    try:
        rows = _export_rows(analysis, tmp_path)[1]
    finally:
        analysis.close()
    methods = (*_methods(family)[:-1], replacement)
    changed = _spec(False, methods)
    with pytest.raises(CodedError) as raised:
        Analysis.from_moments(rows, metrics=[changed], control="control", experiment_id="compact")
    assert raised.value.code == code


def test_equivalent_requests_with_different_labels_share_one_envelope(tmp_path):
    first = Method.cuped(
        name="first", covariates=CupedCovariate(name="history_a", kind="continuous")
    )
    renamed = Method.cuped(
        name="renamed", covariates=CupedCovariate(name="history_a", kind="continuous")
    )
    other = Method.cuped(
        name="other", covariates=CupedCovariate(name="history_b", kind="continuous")
    )
    spec = _spec(False, (first, renamed, other))
    analysis = Analysis.from_unit_summary(
        pa.Table.from_pylist(_records()),
        unit="unit",
        group="arm",
        control="control",
        experiment_id="compact",
        metrics=[spec],
    )
    try:
        expected = analysis.run()
        rows = _export_rows(analysis, tmp_path)[1]
        assert sum(row.get("record_kind") == "cuped_evidence" for row in rows) == 2
        restored = Analysis.from_moments(
            rows, metrics=[spec], control="control", experiment_id="compact"
        )
        try:
            _same_result(_dumps(restored.run()), _dumps(expected))
        finally:
            restored.close()
    finally:
        analysis.close()


@pytest.mark.parametrize("ratio", [False, True])
def test_partition_evidence_merges_to_the_whole_export(tmp_path, ratio):
    from increment._cuped_evidence import parse_cuped_evidence
    from increment.estimation.armstats import ArmStats
    from increment.estimation.cuped.cells import fit_categorical_cells
    from increment.estimation.engine import _df_to_arms

    records = _records()
    halves = (records[0::2], records[1::2])

    def evidence(rows):
        decoded = [
            parse_cuped_evidence(row["cuped_evidence"])
            for row in rows
            if row.get("record_kind") == "cuped_evidence"
        ]
        return {(item.kind, item.method_key): item for item in decoded}

    exports = []
    for index, part in enumerate((records, *halves)):
        analysis, _ = _case(ratio=ratio, family="combined", records=part)
        try:
            exports.append(evidence(_export_rows(analysis, tmp_path, f"part-{index}.parquet")[1]))
        finally:
            analysis.close()
    whole, left, right = exports
    assert whole.keys() == left.keys() == right.keys()
    for key, item in whole.items():
        if item.kind == "categorical_cells":
            merged = left[key].cells.merge(right[key].cells)
            actual = fit_categorical_cells(merged, control="control")
            expected = fit_categorical_cells(item.cells, control="control")
            assert np.asarray(actual.adjusted_means) == pytest.approx(
                np.asarray(expected.adjusted_means), rel=1e-12, abs=0
            )
            assert np.asarray(actual.covariance) == pytest.approx(
                np.asarray(expected.covariance), rel=1e-10, abs=0
            )
        else:
            parts = {arm.group_id: arm for arm in _df_to_arms(left[key].rows)}
            for arm in _df_to_arms(right[key].rows):
                combined = ArmStats.combine([parts[arm.group_id], arm])
                reference = next(
                    stats for stats in _df_to_arms(item.rows) if stats.group_id == arm.group_id
                )
                assert combined.n == reference.n
                assert combined.mean_y() == pytest.approx(reference.mean_y(), rel=1e-13)
                assert combined.var_y() == pytest.approx(reference.var_y(), rel=1e-11)
