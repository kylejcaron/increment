"""Encouragement/Observational designs declared on from_definitions."""

from datetime import datetime
from pathlib import Path

import ibis
import pytest

from increment.analysis import Analysis
from increment.semantics.models import AnalysisPlan, Definitions
from tests.analysis_factory import _native_source, lift_rows


@pytest.fixture
def con():
    return ibis.duckdb.connect()


_PLAN = AnalysisPlan(secondaries=["revenue"])


def _encouragement_defs() -> dict:
    return {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "select * from native_encouragement_events",
                "timestamp_column": "ts",
                "entities": ["user_id"],
                "facts": [
                    {"name": "exposed", "column": None},
                    {"name": "clicked", "column": None},
                    {"name": "purchase", "column": "revenue"},
                ],
            }
        ],
        "exposures": [{"name": "e", "fact": "exposed"}],
        "metrics": [
            {
                "type": "mean",
                "name": "revenue",
                "entity": "user_id",
                "fact": "purchase",
                "aggregation": "sum",
                "window_days": 7,
            }
        ],
        "experiments": [
            {
                "name": "native_encouragement_exp",
                "exposure": "e",
                "unit": "user_id",
                "start": "2025-01-01",
                "end": "2025-01-31",
                "control_group": "control",
                "plan": {"secondaries": ["revenue"]},
                "design": {
                    "mechanism": "encouragement",
                    "uptake": {"fact": "clicked"},
                    "one_sided": True,
                    "exclusion_restriction": {
                        "acknowledged": True,
                        "justification": "assignment moves revenue only via uptake",
                    },
                },
            }
        ],
    }


def _write_defs_yaml(defs_dict: dict, tmp_path) -> Path:
    import yaml

    path = Path(tmp_path) / "defs.yaml"
    path.write_text(yaml.safe_dump(defs_dict, sort_keys=False))
    return path


def _revenue(i: int, *, clicked: bool) -> float:
    # Genuine within-arm variance: a constant control arm makes
    # family.evidence.incomplete refuse instead of exercising the row.
    return 8.0 + (i % 4) + (5.0 if clicked else 0.0)


def _seed(con):
    if "native_encouragement_events" in con.list_tables():
        return
    control = [f"c{i}" for i in range(1, 21)]
    treat = [f"t{i}" for i in range(1, 21)]
    clickers = {f"t{i}" for i in range(1, 13)}
    rows = (
        [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 1, 9, 0, 0),
                "event": "exposed",
                "group_id": "control",
                "revenue": None,
            }
            for u in control
        ]
        + [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 1, 9, 0, 0),
                "event": "exposed",
                "group_id": "treatment",
                "revenue": None,
            }
            for u in treat
        ]
        + [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 2, 9, 0, 0),
                "event": "clicked",
                "group_id": None,
                "revenue": None,
            }
            for u in clickers
        ]
        + [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 3, 9, 0, 0),
                "event": "purchase",
                "group_id": None,
                "revenue": _revenue(i, clicked=u in clickers),
            }
            for i, u in enumerate(control + treat, start=1)
        ]
        + [
            # Pushes the table's observed date extent well past the metric's
            # 7-day window (fe_date + 7 = Jan 8) so the maturity check does
            # not censor every unit for lack of "future" data in this small
            # synthetic fixture; zero revenue leaves per-unit sums unchanged.
            {
                "user_id": "c1",
                "ts": datetime(2025, 1, 10, 9, 0, 0),
                "event": "purchase",
                "group_id": None,
                "revenue": 0.0,
            }
        ]
    )
    con.create_table("native_encouragement_events", obj=rows)


@pytest.mark.parametrize("exclusion_declared", [False, True])
def test_native_encouragement_declaration_runs_and_matches_dataframe_oracle(
    con, tmp_path, exclusion_declared
):
    """from_definitions with a declared Encouragement design produces the
    same itt/compliance/late rows as the equivalent from_unit_summary
    call -- with the plan actually declared, an itt row must be present,
    not only compliance/late."""
    import pandas as pd

    from increment import Encouragement, ExclusionRestriction, UptakeSpec

    _seed(con)
    defs_dict = _encouragement_defs()
    if not exclusion_declared:
        del defs_dict["experiments"][0]["design"]["exclusion_restriction"]
    defs_path = _write_defs_yaml(defs_dict, tmp_path)
    estimands = None if exclusion_declared else ("itt", "compliance")
    native = Analysis.from_definitions("native_encouragement_exp", defs_path, con, store="none")
    native_rows = {
        (r.metric, r.group_id, r.estimand): r.require_lift().value
        for r in lift_rows(native.run(estimands=estimands))
    }
    assert any(estimand == "itt" for _, _, estimand in native_rows), (
        "a declared plan must still report itt, not only compliance/late"
    )

    records = []
    for i, u in enumerate(
        [f"c{i}" for i in range(1, 21)] + [f"t{i}" for i in range(1, 21)], start=1
    ):
        clicked = u.startswith("t") and int(u[1:]) <= 12
        records.append(
            {
                "user_id": u,
                "group": "control" if u.startswith("c") else "treatment",
                "clicked": int(clicked),
                "revenue": _revenue(i, clicked=clicked),
            }
        )
    oracle = Analysis.from_unit_summary(
        pd.DataFrame(records),
        unit="user_id",
        group="group",
        metrics={"revenue": "mean"},
        design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked"),
            exclusion_restriction=(
                ExclusionRestriction(
                    acknowledged=True,
                    justification="assignment moves revenue only via uptake",
                )
                if exclusion_declared
                else None
            ),
            one_sided=True,
        ),
        uptake="clicked",
        plan=_PLAN,
    )
    oracle_rows = {
        (r.metric, r.group_id, r.estimand): r.require_lift().value
        for r in lift_rows(oracle.run(estimands=estimands))
    }
    assert set(native_rows) == set(oracle_rows)
    for key, value in oracle_rows.items():
        assert native_rows[key] == pytest.approx(value, rel=1e-9)


def test_observational_design_declares_and_reads_the_covariate(con, tmp_path):
    """Observational parses (via the {property, source} covariate shape) and
    resolved_design() returns it; the definitions unit frame serves the
    declared covariate."""
    from increment.semantics.design import Observational
    from tests.analysis_factory import _native_source

    _seed(con)
    defs_dict = {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "select *, 30.0 as tenure from native_encouragement_events",
                "timestamp_column": "ts",
                "entities": ["user_id"],
                "facts": [
                    {"name": "exposed", "column": None},
                    {"name": "purchase", "column": "revenue"},
                ],
                "properties": [
                    {"name": "tenure", "column": "tenure", "dtype": "float", "as_of": "static"}
                ],
            }
        ],
        "exposures": [{"name": "e", "fact": "exposed"}],
        "metrics": [
            {
                "type": "mean",
                "name": "revenue",
                "entity": "user_id",
                "fact": "purchase",
                "aggregation": "sum",
                "window_days": 7,
            }
        ],
        "experiments": [
            {
                "name": "native_observational_exp",
                "exposure": "e",
                "unit": "user_id",
                "start": "2025-01-01",
                "end": "2025-01-31",
                "control_group": "control",
                "plan": {"secondaries": ["revenue"]},
                "design": {
                    "mechanism": "observational",
                    "covariates": [{"property": "tenure", "source": "events"}],
                },
            }
        ],
    }
    exp = Definitions.model_validate(defs_dict).experiment("native_observational_exp")
    assert exp is not None
    design = exp.resolved_design()
    assert isinstance(design, Observational)
    assert design.control_group == "control"

    defs_path = _write_defs_yaml(defs_dict, tmp_path)
    native = Analysis.from_definitions("native_observational_exp", defs_path, con, store="none")
    source = _native_source(native)
    table = source.unit_frame(source.context.metrics[0], covariates=["tenure"])
    rows = table.to_pylist()
    assert rows
    assert all(row["tenure"] == pytest.approx(30.0) for row in rows)


def test_native_encouragement_artifact_publish_and_reopen_carries_design(con, tmp_path):
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.design import Encouragement

    _seed(con)
    defs_dict = _encouragement_defs()
    defs = Definitions.model_validate(defs_dict)
    experiment = defs.experiment("native_encouragement_exp")
    assert experiment is not None
    defs_path = _write_defs_yaml(defs_dict, tmp_path)
    native = Analysis.from_definitions("native_encouragement_exp", defs_path, con, store="none")
    context = artifact_context(defs, experiment, "error")
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    ref = native.publish_unit_day_artifact(store)
    reopened = Analysis.from_unit_day_artifact(store, ref, expected_context=context)

    assert isinstance(_native_source(reopened).context.design, Encouragement)
    native_rows = {
        (r.metric, r.group_id, r.estimand): r.require_lift().value for r in lift_rows(native.run())
    }
    reopened_rows = {
        (r.metric, r.group_id, r.estimand): r.require_lift().value
        for r in lift_rows(reopened.run())
    }
    assert native_rows.keys() == reopened_rows.keys()
    for key, value in native_rows.items():
        assert reopened_rows[key] == pytest.approx(value, rel=1e-9)


def test_native_encouragement_export_import_moments_carries_design(con, tmp_path):
    """Matches the shape of
    tests/test_realistic_warehouse.py::test_export_from_moments_round_trips_run_exactly:
    export()/pq.read_table()/from_moments(rows, metrics=[MetricSpec...],
    design=...), asserting the full itt/compliance/late round trip."""
    import pyarrow.parquet as pq

    from increment.frame import MetricSpec
    from increment.semantics.design import Encouragement

    _seed(con)
    defs_path = _write_defs_yaml(_encouragement_defs(), tmp_path)
    native = Analysis.from_definitions("native_encouragement_exp", defs_path, con, store="none")
    baseline = {
        (r.metric, r.group_id, r.estimand): r.require_lift() for r in lift_rows(native.run())
    }

    path = tmp_path / "moments.parquet"
    native.export(path)
    rows = pq.read_table(path).to_pylist()
    specs = [MetricSpec(name="revenue", type="mean")]
    reimported = Analysis.from_moments(
        rows, metrics=specs, design=native.experiment.resolved_design()
    )

    assert isinstance(_native_source(reimported).context.design, Encouragement)
    reimported_rows = {
        (r.metric, r.group_id, r.estimand): r.require_lift() for r in lift_rows(reimported.run())
    }
    assert set(reimported_rows) == set(baseline)
    for key, base_lift in baseline.items():
        lift = reimported_rows[key]
        assert lift.value is not None and base_lift.value is not None
        assert abs(lift.value - base_lift.value) < 1e-9


def test_declared_encouragement_context_equals_the_separately_supplied_design_form():
    import json

    from increment import Encouragement
    from increment.query.artifact_publish import artifact_context

    definitions = Definitions.model_validate(_encouragement_defs())
    experiment = definitions.experiment("native_encouragement_exp")
    assert experiment is not None
    context = artifact_context(definitions, experiment, "error")
    explicit = artifact_context(
        definitions,
        experiment,
        "error",
        encouragement_uptake=experiment.resolved_design(),
    )
    assert context.sha256 == explicit.sha256
    design = json.loads(context.canonical_json)["experiment"]["design"]
    assert design["mechanism"] == "encouragement"
    resolved = experiment.resolved_design()
    assert isinstance(resolved, Encouragement)
    assert design["uptake"]["fact"] == resolved.uptake.fact


def test_design_less_experiment_context_omits_the_design_key():
    """A design-less experiment's context omits the undeclared design key
    instead of serializing ``"design": null``, so no Randomized design is invented."""
    from increment.query.artifact_contract import compile_unit_day_artifact_context

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "select 1 as user_id, current_timestamp as ts",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "exposed", "column": None},
                        {"name": "purchase", "column": "revenue"},
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "exposed"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "revenue",
                    "entity": "user_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                }
            ],
            "experiments": [
                {
                    "name": "exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-01-01T00:00:00+00:00",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue"]},
                }
            ],
        }
    )
    context = compile_unit_day_artifact_context("exp", defs, on_mixed_assignment="error")
    import json

    assert "design" not in json.loads(context.canonical_json)["experiment"]


def test_observational_artifact_publish_and_reopen_restores_design(con, tmp_path):
    """Reopening a unit-day artifact published from a declared-Observational
    analysis restores Observational, not the Randomized fallback earlier
    versions built regardless of the experiment's actual mechanism."""
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.design import Observational

    _seed(con)
    defs_dict = {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "select *, 30.0 as tenure from native_encouragement_events",
                "timestamp_column": "ts",
                "entities": ["user_id"],
                "facts": [
                    {"name": "exposed", "column": None},
                    {"name": "purchase", "column": "revenue"},
                ],
                "properties": [
                    {"name": "tenure", "column": "tenure", "dtype": "float", "as_of": "static"}
                ],
            }
        ],
        "exposures": [{"name": "e", "fact": "exposed"}],
        "metrics": [
            {
                "type": "mean",
                "name": "revenue",
                "entity": "user_id",
                "fact": "purchase",
                "aggregation": "sum",
                "window_days": 7,
            }
        ],
        "experiments": [
            {
                "name": "native_observational_exp",
                "exposure": "e",
                "unit": "user_id",
                "start": "2025-01-01",
                "end": "2025-01-31",
                "control_group": "control",
                "plan": {"secondaries": ["revenue"]},
                "design": {
                    "mechanism": "observational",
                    "covariates": [{"property": "tenure", "source": "events"}],
                },
            }
        ],
    }
    defs = Definitions.model_validate(defs_dict)
    experiment = defs.experiment("native_observational_exp")
    assert experiment is not None
    defs_path = _write_defs_yaml(defs_dict, tmp_path)
    native = Analysis.from_definitions("native_observational_exp", defs_path, con, store="none")
    context = artifact_context(defs, experiment, "error")
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    ref = native.publish_unit_day_artifact(store)
    reopened = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    reopened_design = _native_source(reopened).context.design
    assert isinstance(reopened_design, Observational)
    assert reopened_design.control_group == "control"


def test_compile_unit_day_artifact_context_refuses_conflicting_encouragement_uptake():
    """A caller-supplied encouragement_uptake= that disagrees with the
    experiment's own declared design is a genuine contradiction, refused
    by name rather than silently overridden either way."""
    from increment.query.artifact_contract import (
        ArtifactContractError,
        compile_unit_day_artifact_context,
    )
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "select 1 as user_id, current_timestamp as ts",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "exposed", "column": None},
                        {"name": "clicked", "column": None},
                        {"name": "purchase", "column": "revenue"},
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "exposed"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "revenue",
                    "entity": "user_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                }
            ],
            "experiments": [
                {
                    "name": "exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-01-01T00:00:00+00:00",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue"]},
                    "design": {
                        "mechanism": "encouragement",
                        "uptake": {"fact": "clicked"},
                        "exclusion_restriction": {
                            "acknowledged": True,
                            "justification": "assignment moves revenue only via uptake",
                        },
                    },
                }
            ],
        }
    )
    conflicting = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="a_different_fact"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="a different, conflicting declaration"
        ),
    )
    with pytest.raises(ArtifactContractError) as exc:
        compile_unit_day_artifact_context(
            "exp", defs, on_mixed_assignment="error", encouragement_uptake=conflicting
        )
    assert exc.value.code == "artifact_contract.encouragement_uptake_conflict"
