from __future__ import annotations

import copy
import pickle

import pandas as pd
import polars as pl
import pytest

from increment.analysis import Analysis
from increment.errors import CapabilityError, InvalidRequestError
from increment.semantics.design import Randomized
from increment.sequential_source import validate_frame_registration
from tests.sequential_cases import registration


def test_old_pandas_boolean_sequential_control_label_refuses_with_coded_reason():
    legacy = registration().model_copy(update={"control_group": "True"})
    with pytest.raises(CapabilityError) as raised:
        validate_frame_registration(
            legacy,
            metrics=[],
            specs=[],
            design=Randomized(control_group="true"),
            configs=[],
            source_id="experiment",
            source_mapping={},
        )
    assert raised.value.code == "sequential.source.invalid"
    assert "labels changed" in str(raised.value)
    for copied in (copy.deepcopy(raised.value), pickle.loads(pickle.dumps(raised.value))):
        assert copied.code == raised.value.code
        assert "labels changed" in str(copied)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_boolean_groups_use_canonical_labels_for_every_control_spelling(backend):
    data = {"unit": ["c1", "c2", "t1", "t2"], "arm": [True, True, False, False], "y": [1, 2, 3, 4]}
    frame = pd.DataFrame(data) if backend == "pandas" else pl.DataFrame(data)
    results = []
    for control in (True, "True", "true"):
        analysis = Analysis.from_unit_summary(
            frame, unit="unit", group="arm", control=control, metrics={"y": "mean"}
        )
        rows = analysis.run()
        assert {getattr(row, "group_id", None) for row in rows} == {"false"}
        results.append((analysis.unit_counts(), rows))
    assert results[0] == results[1] == results[2]
    assert results[0][0] == {"true": 2, "false": 2}


@pytest.mark.parametrize("constructor", ["summary", "panel"])
def test_boolean_labels_normalize_explicit_design_and_allocation(constructor):
    frame = pd.DataFrame(
        {
            "unit": ["c1", "c2", "t1", "t2"],
            "arm": [True, True, False, False],
            "date": ["2025-01-01"] * 4,
            "y": [1, 2, 3, 4],
        }
    )
    factory = Analysis.from_unit_summary if constructor == "summary" else Analysis.from_unit_panel
    kwargs = {"date": "date"} if constructor == "panel" else {}
    analysis = factory(
        frame,
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="True",
            allocation={"True": 1, "False": 1},
            allocation_scheme="independent",
        ),
        metrics={"y": "mean"},
        **kwargs,
    )
    assert analysis.unit_counts() == {"true": 2, "false": 2}
    assert analysis.srm().model_dump().get("expected") == {"true": 0.5, "false": 0.5}


def test_boolean_moments_control_coerces_only_for_boolean_labels(tmp_path):
    original = Analysis.from_unit_summary(
        pd.DataFrame(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": [True, True, False, False],
                "y": [1, 2, 3, 4],
            }
        ),
        unit="unit",
        group="arm",
        control=True,
        metrics={"y": "mean"},
    )
    path = tmp_path / "moments.parquet"
    original.export(path)
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    replayed = [
        Analysis.from_moments(rows, metrics={"y": "mean"}, control=value).run()
        for value in (True, "true")
    ]
    assert replayed[0] == replayed[1] == original.run()
    with pytest.raises(InvalidRequestError) as noncanonical:
        Analysis.from_moments(rows, metrics={"y": "mean"}, control="True")
    explicit = Analysis.from_moments(
        rows,
        metrics={"y": "mean"},
        design=Randomized(
            control_group="true",
            allocation={True: 1, False: 1},
            allocation_scheme="independent",
        ),
    )
    assert explicit.srm().model_dump().get("expected") == {"true": 0.5, "false": 0.5}
    assert noncanonical.value.code == "source.frame.control_missing"
    with pytest.raises(InvalidRequestError) as explicit_design:
        Analysis.from_moments(
            rows,
            metrics={"y": "mean"},
            design=Randomized(control_group="True"),
        )
    assert explicit_design.value.code == "source.frame.control_missing"
    with pytest.raises(InvalidRequestError) as raised:
        Analysis.from_moments(rows, metrics={"y": "mean"}, control=["true"])
    assert raised.value.code == "source.frame.control_invalid"


def test_boolean_moments_group_ids_are_canonicalized_at_ingress(tmp_path):
    original = Analysis.from_unit_summary(
        pd.DataFrame(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": [True, True, False, False],
                "y": [1, 2, 3, 4],
            }
        ),
        unit="unit",
        group="arm",
        control=True,
        metrics={"y": "mean"},
    )
    path = tmp_path / "boolean-group-id-ingress.parquet"
    original.export(path)
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    boolean_rows = copy.deepcopy(rows)
    string_rows = copy.deepcopy(rows)
    for boolean_row, string_row in zip(boolean_rows, string_rows, strict=True):
        boolean_row.pop("assignment_counts", None)
        string_row.pop("assignment_counts", None)
        boolean_row["group_id"] = boolean_row["group_id"] == "true"
    expected = Analysis.from_moments(string_rows, metrics={"y": "mean"}, control="true").run()
    actual = Analysis.from_moments(boolean_rows, metrics={"y": "mean"}, control=True).run()
    assert actual == expected
    with pytest.raises(InvalidRequestError) as wrong_case:
        Analysis.from_moments(boolean_rows, metrics={"y": "mean"}, control="True")
    assert wrong_case.value.code == "source.frame.control_missing"


def test_dropped_boolean_arm_remains_in_enrollment_counts_after_export(tmp_path):
    from increment import IncrementWarning, MetricSpec

    frame = pd.DataFrame(
        {
            "unit": ["c1", "c2", "t1", "t2"],
            "arm": [True, True, False, False],
            "y": [1.0, 2.0, None, None],
        }
    )
    with pytest.warns(IncrementWarning):
        original = Analysis.from_unit_summary(
            frame,
            unit="unit",
            group="arm",
            control=True,
            metrics=[MetricSpec(name="y", missing="drop")],
        )
    path = tmp_path / "dropped-boolean-arm.parquet"
    original.export(path)
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    replay = Analysis.from_moments(
        rows,
        metrics=[MetricSpec(name="y", missing="drop")],
        control=True,
    )
    with pytest.raises(InvalidRequestError) as no_treatment:
        replay.run()
    assert no_treatment.value.code == "readout.arms.no_treatment"
    assert no_treatment.value.context["control_group"] == "true"


def test_assignment_counts_match_canonical_moments_group_identities(tmp_path):
    import json

    original = Analysis.from_unit_summary(
        pd.DataFrame(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": [True, True, False, False],
                "y": [1, 2, 3, 4],
            }
        ),
        unit="unit",
        group="arm",
        control=True,
        metrics={"y": "mean"},
    )
    path = tmp_path / "boolean-count-identities.parquet"
    original.export(path)
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    boolean_rows = copy.deepcopy(rows)
    for boolean_row in boolean_rows:
        boolean_row["group_id"] = boolean_row["group_id"] == "true"
        boolean_row["assignment_counts"] = json.dumps({"True": 2, "False": 2})

    boolean_analysis = Analysis.from_moments(boolean_rows, metrics={"y": "mean"}, control=True)
    expected_rows = copy.deepcopy(rows)
    for row in expected_rows:
        row["assignment_counts"] = json.dumps({"true": 2, "false": 2})
    expected = Analysis.from_moments(expected_rows, metrics={"y": "mean"}, control="true")
    assert boolean_analysis.unit_counts() == {"true": 2, "false": 2}
    assert boolean_analysis.run() == expected.run()
    with pytest.raises(InvalidRequestError) as wrong_case:
        Analysis.from_moments(boolean_rows, metrics={"y": "mean"}, control="True")
    assert wrong_case.value.code == "source.frame.control_missing"


def test_string_boolean_looking_moments_labels_stay_case_sensitive(tmp_path):
    original = Analysis.from_unit_summary(
        pd.DataFrame(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["True", "True", "False", "False"],
                "y": [1, 2, 3, 4],
            }
        ),
        unit="unit",
        group="arm",
        control="True",
        metrics={"y": "mean"},
    )
    path = tmp_path / "string-moments.parquet"
    original.export(path)
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    assert (
        Analysis.from_moments(rows, metrics={"y": "mean"}, control="True").run() == original.run()
    )


def test_lowercase_string_moments_labels_do_not_infer_boolean_provenance(tmp_path):
    original = Analysis.from_unit_summary(
        pd.DataFrame(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["true", "true", "false", "false"],
                "y": [1, 2, 3, 4],
            }
        ),
        unit="unit",
        group="arm",
        control="true",
        metrics={"y": "mean"},
    )
    path = tmp_path / "lowercase-string-moments.parquet"
    original.export(path)
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    assert (
        Analysis.from_moments(rows, metrics={"y": "mean"}, control="true").run() == original.run()
    )
    with pytest.raises(InvalidRequestError) as raised:
        Analysis.from_moments(rows, metrics={"y": "mean"}, control="True")
    assert raised.value.code == "source.frame.control_missing"


def test_design_summary_boolean_controls_use_only_python_bool_alias(tmp_path):
    from increment import Encouragement, UptakeSpec

    frame = pd.DataFrame(
        {
            "unit": ["c1", "c2", "t1", "t2"],
            "arm": [True, True, False, False],
            "clicked": [0, 1, 0, 1],
        }
    )
    design = Encouragement(control_group="true", uptake=UptakeSpec(fact="clicked"))
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        metrics=[],
        design=design,
        uptake="clicked",
    )
    path = tmp_path / "boolean-design-summary.parquet"
    analysis.export(path)
    import pyarrow.parquet as pq

    payload = pq.read_table(path).to_pylist()
    assert payload[0]["record_kind"] == "design_summary"
    with pytest.raises(InvalidRequestError) as wrong_case:
        Analysis.from_moments(
            payload,
            metrics=[],
            design=Encouragement(control_group="True", uptake=UptakeSpec(fact="clicked")),
        )
    assert wrong_case.value.code == "source.frame.control_missing"
    assert wrong_case.value.context["observed"] == ("false", "true")
    assert isinstance(wrong_case.value.context["route"], str)
    assert wrong_case.value.context["route"]
    replay = Analysis.from_moments(payload, metrics=[], control=True)
    assert replay.unit_counts() == {"true": 2, "false": 2}


def test_numeric_moments_control_matches_stringified_group_label(tmp_path):
    original = Analysis.from_unit_summary(
        pd.DataFrame(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": [0, 0, "treatment", "treatment"],
                "y": [1, 2, 3, 4],
            }
        ),
        unit="unit",
        group="arm",
        control=0,
        metrics={"y": "mean"},
    )
    path = tmp_path / "numeric-control-moments.parquet"
    original.export(path)
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row.pop("assignment_counts", None)
        if row["group_id"] == "0":
            row["group_id"] = 0
    replay = Analysis.from_moments(rows, metrics={"y": "mean"}, control=0)
    assert len(replay.run()) == 1


def test_design_summary_control_only_reload_accepts_mapping_compliance(tmp_path):
    import json

    from increment import Encouragement, UptakeSpec

    frame = pd.DataFrame(
        {
            "unit": ["c1", "c2", "t1", "t2"],
            "arm": [True, True, False, False],
            "clicked": [0, 1, 0, 1],
        }
    )
    original = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        metrics=[],
        design=Encouragement(control_group="true", uptake=UptakeSpec(fact="clicked")),
        uptake="clicked",
    )
    path = tmp_path / "boolean-design-summary-mapping.parquet"
    original.export(path)
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    rows[0]["compliance_summary"] = json.loads(rows[0]["compliance_summary"])
    replay = Analysis.from_moments(rows, metrics=[], control=True)
    assert replay.unit_counts() == {"true": 2, "false": 2}


def test_boolean_design_allocation_is_frozen_and_collisions_are_coded():
    data = pd.DataFrame(
        {"unit": ["c1", "c2", "t1", "t2"], "arm": [True, True, False, False], "y": [1, 2, 3, 4]}
    )
    design = Randomized(
        control_group="True",
        allocation={"True": 1, "False": 1},
        allocation_scheme="independent",
    )
    analysis = Analysis.from_unit_summary(
        data, unit="unit", group="arm", design=design, metrics={"y": "mean"}
    )
    allocation = analysis._src.context.design.allocation
    with pytest.raises(TypeError):
        allocation["true"] = 3
    assert analysis.srm().model_dump().get("expected") == {"true": 0.5, "false": 0.5}

    colliding = Randomized(
        control_group="True",
        allocation={"True": 1, "true": 1},
    )
    with pytest.raises(InvalidRequestError) as raised:
        Analysis.from_unit_summary(
            data, unit="unit", group="arm", design=colliding, metrics={"y": "mean"}
        )
    assert raised.value.code == "frame.validation.group_label_collision"


def test_public_constructor_refuses_old_boolean_registration_with_replay_route():
    from increment import SequentialRegistration
    from increment.frame import MetricSpec, synthesise_metric
    from increment.semantics.models import AnalysisPlan, InferenceSpec
    from increment.sequential_source import frame_observation_mapping, sequential_definition_id

    design = Randomized(control_group="True")
    spec = MetricSpec(name="outcome")
    base = registration("bernoulli")
    legacy = SequentialRegistration.model_validate(
        {
            **base.model_dump(),
            "control_group": "True",
            "definitions_id": sequential_definition_id(
                [synthesise_metric(spec)],
                design,
                transformations=[spec],
                source_mapping=frame_observation_mapping(unit="unit", group="arm"),
            ),
        }
    )
    plan = AnalysisPlan(
        primary="outcome",
        inference=InferenceSpec(kind="always_valid", registration=legacy),
    )
    with pytest.raises(CapabilityError) as raised:
        Analysis.from_unit_summary(
            pd.DataFrame({"unit": ["c1", "c2"], "arm": [True, False], "outcome": [0, 1]}),
            unit="unit",
            group="arm",
            design=design,
            metrics=[spec],
            plan=plan,
        )
    assert raised.value.code == "sequential.source.invalid"
    assert "replay" in str(raised.value).lower()


def test_string_boolean_looking_group_values_are_unchanged():
    analysis = Analysis.from_unit_summary(
        pd.DataFrame(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["True", "True", "False", "False"],
                "y": [1, 2, 3, 4],
            }
        ),
        unit="unit",
        group="arm",
        control="True",
        metrics={"y": "mean"},
    )
    assert {getattr(row, "group_id", None) for row in analysis.run()} == {"False"}


def test_integer_group_accepts_integer_control_and_missing_control_is_coded():
    analysis = Analysis.from_unit_summary(
        pd.DataFrame({"unit": ["c1", "c2", "t1", "t2"], "arm": [0, 0, 1, 1], "y": [1, 2, 3, 4]}),
        unit="unit",
        group="arm",
        control=0,
        metrics={"y": "mean"},
    )
    assert {getattr(row, "group_id", None) for row in analysis.run()} == {"1"}
    with pytest.raises(InvalidRequestError) as raised:
        Analysis.from_unit_summary(
            pd.DataFrame(
                {"unit": ["c1", "c2", "t1", "t2"], "arm": [0, 0, 1, 1], "y": [1, 2, 3, 4]}
            ),
            unit="unit",
            group="arm",
            control=2,
            metrics={"y": "mean"},
        )
    assert raised.value.code == "source.frame.control_missing"
    assert raised.value.context["observed"] == ("0", "1")


def test_non_scalar_control_is_coded():
    with pytest.raises(InvalidRequestError) as raised:
        Analysis.from_unit_summary(
            pd.DataFrame({"unit": ["c"], "arm": ["control"], "y": [1]}),
            unit="unit",
            group="arm",
            control=["control"],
            metrics={"y": "mean"},
        )
    assert raised.value.code == "source.frame.control_invalid"


def test_explicit_design_can_omit_control_argument():
    from increment.semantics.design import Randomized

    analysis = Analysis.from_unit_summary(
        pd.DataFrame(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["control", "control", "treatment", "treatment"],
                "y": [1, 2, 3, 4],
            }
        ),
        unit="unit",
        group="arm",
        design=Randomized(control_group="control"),
        metrics={"y": "mean"},
    )
    assert analysis.unit_counts() == {"control": 2, "treatment": 2}


def test_exclusion_justification_refusal_is_coded_and_roundtrips():
    from increment.semantics.design import ExclusionRestriction

    with pytest.raises(InvalidRequestError) as raised:
        ExclusionRestriction(acknowledged=True, justification="short")
    error = raised.value
    assert error.code == "design.exclusion_restriction.justification"
    assert pickle.loads(pickle.dumps(error)).code == error.code
    assert copy.deepcopy(error).code == error.code
