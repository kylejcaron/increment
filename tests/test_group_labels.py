from __future__ import annotations

import copy
import pickle

import pandas as pd
import polars as pl
import pytest

from increment.analysis import Analysis
from increment.errors import InvalidRequestError


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("control", [True, "True", "true"])
def test_boolean_groups_use_canonical_labels_for_every_control_spelling(backend, control):
    data = {"unit": ["c1", "c2", "t1", "t2"], "arm": [True, True, False, False], "y": [1, 2, 3, 4]}
    frame = pd.DataFrame(data) if backend == "pandas" else pl.DataFrame(data)
    analysis = Analysis.from_unit_summary(
        frame, unit="unit", group="arm", control=control, metrics={"y": "mean"}
    )
    rows = analysis.run()
    assert {getattr(row, "group_id", None) for row in rows} == {"false"}


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


def test_exclusion_justification_refusal_is_coded_and_roundtrips():
    from increment.semantics.design import ExclusionRestriction

    with pytest.raises(InvalidRequestError) as raised:
        ExclusionRestriction(acknowledged=True, justification="short")
    error = raised.value
    assert error.code == "design.exclusion_restriction.justification"
    assert pickle.loads(pickle.dumps(error)).code == error.code
    assert copy.deepcopy(error).code == error.code
