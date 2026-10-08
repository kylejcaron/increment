"""Declared assignment laws and experiment-level integrity diagnostics."""

import pickle
from copy import deepcopy

import pytest

from increment import Encouragement, Randomized, UptakeSpec
from increment.errors import InvalidRequestError
from increment.semantics.design import AdjustmentSet, Observational


@pytest.mark.parametrize("model", [Randomized, Encouragement])
@pytest.mark.parametrize("scheme", ["independent", "blocked", "adaptive", "quota", "fixed_counts"])
def test_allocation_scheme_round_trips_without_inference(model, scheme):
    kwargs = {"control_group": "control", "allocation": {"control": 4, "treatment": 1}}
    if model is Encouragement:
        kwargs["uptake"] = UptakeSpec(fact="uptake")
    old = model(**kwargs)
    declared = model(**kwargs, allocation_scheme=scheme)
    assert "allocation_scheme" not in old.model_dump(mode="json")
    assert model.model_validate_json(declared.model_dump_json()).allocation_scheme == scheme
    assert declared != old
    assert hash(declared) != hash(old)


def test_observational_refuses_assignment_scheme_with_stable_code():
    with pytest.raises(InvalidRequestError) as error:
        Observational(
            control_group="control",
            adjustment=AdjustmentSet(covariates=("x",)),
            allocation_scheme="independent",  # ty: ignore[unknown-argument]
        )
    assert error.value.code == "design.allocation_scheme.incompatible"


def _design(scheme="independent", allocation=None):
    return Randomized(
        control_group="control",
        allocation=allocation or {"control": 0.5, "treatment": 0.5},
        allocation_scheme=scheme,
    )


def test_declared_severe_mismatch_uses_existing_e_process_and_survives_replay():
    from increment.estimation.assignment_integrity import assignment_integrity

    result = assignment_integrity(
        _design(), {"control": 900, "treatment": 100}, randomization_grain="unit"
    )
    assert result.status == "failed"
    assert result.construction == "always_valid"
    assert result.alpha == 0.001
    assert result.observed == {"control": 900, "treatment": 100}
    assert result.expected == {"control": 0.5, "treatment": 0.5}
    assert result.context["log_e_value"] == pytest.approx(364.325138, abs=1e-6)
    assert type(result).model_validate_json(result.model_dump_json()) == result
    assert deepcopy(result) == result
    assert pickle.loads(pickle.dumps(result)) == result


@pytest.mark.parametrize(
    "scheme,status,code",
    [
        (None, "not_checked_missing_declaration", "integrity.allocation_scheme_missing"),
        ("blocked", "unsupported_assignment", "integrity.allocation_scheme_unsupported"),
        ("adaptive", "unsupported_assignment", "integrity.allocation_scheme_unsupported"),
        ("quota", "unsupported_assignment", "integrity.allocation_scheme_unsupported"),
        ("fixed_counts", "unsupported_assignment", "integrity.allocation_scheme_unsupported"),
        ("independent", "not_checked_missing_counts", "integrity.counts_missing"),
    ],
)
def test_declaration_precedence_even_without_counts(scheme, status, code):
    from increment.estimation.assignment_integrity import assignment_integrity

    result = assignment_integrity(_design(scheme), None)
    assert result.status == status
    assert result.code == code
    assert result.alpha is None
    assert result.construction == "none"


@pytest.mark.parametrize("kwargs", [{}, {"constant_allocation": False}])
def test_missing_or_nonconstant_allocation_is_not_inferred(kwargs):
    from increment.estimation.assignment_integrity import assignment_integrity

    design = (
        _design()
        if kwargs
        else Randomized(control_group="control", allocation_scheme="independent")
    )
    result = assignment_integrity(design, {"control": 900, "treatment": 100}, **kwargs)
    assert result.status == "not_checked_missing_declaration"
    assert result.code == "integrity.allocation_missing_or_nonconstant"


def test_unequal_cluster_allocation_checks_assigned_clusters_only():
    from increment.estimation.assignment_integrity import assignment_integrity

    result = assignment_integrity(
        _design(allocation={"control": 4, "treatment": 1}),
        {"control": 800, "treatment": 200},
        randomization_grain="cluster",
    )
    assert result.status == "not_rejected"
    assert result.randomization_grain == "cluster"
    assert result.expected == {"control": 0.8, "treatment": 0.2}


def test_encouragement_uses_randomized_assignment_not_uptake():
    from increment.estimation.assignment_integrity import assignment_integrity

    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="uptake"),
        allocation={"control": 0.5, "treatment": 0.5},
        allocation_scheme="independent",
    )
    assert assignment_integrity(design, {"control": 900, "treatment": 100}).status == "failed"


def test_observational_and_nonparallel_laws_are_distinct():
    from increment.estimation.assignment_integrity import assignment_integrity

    design = Observational(control_group="control", adjustment=AdjustmentSet(covariates=("x",)))
    assert assignment_integrity(design, None).status == "not_applicable"
    result = assignment_integrity(None, None)
    assert result.status == "unsupported_assignment"
    assert result.code == "integrity.switchback_assignment_law"


def test_repeated_cumulative_looks_match_existing_stateless_e_process():
    from increment.estimation.assignment_integrity import assignment_integrity
    from increment.estimation.diagnostics import sample_ratio_mismatch

    for counts in [
        {"control": 9, "treatment": 1},
        {"control": 90, "treatment": 10},
        {"control": 900, "treatment": 100},
    ]:
        expected = sample_ratio_mismatch(counts, expected={"control": 0.5, "treatment": 0.5})
        result = assignment_integrity(_design(), counts)
        assert (result.status == "failed") == expected.is_srm
        assert result.context["log_e_value"] == expected.log_e_value


def test_incompatible_count_prefix_is_explicitly_unchecked():
    from increment.estimation.assignment_integrity import assignment_integrity

    assert (
        assignment_integrity(_design(), {"control": 900, "treatment": 100}, cumulative=False).status
        == "not_checked_missing_counts"
    )


def test_missing_scheme_keeps_declared_allocation_unknown():
    from increment.estimation.assignment_integrity import assignment_integrity

    result = assignment_integrity(_design(None), {"control": 900, "treatment": 100})
    assert result.status == "not_checked_missing_declaration"
    assert result.code == "integrity.allocation_scheme_missing"
    assert result.expected == {"control": 0.5, "treatment": 0.5}
    assert result.observed == {"control": 900, "treatment": 100}
    assert result.construction == "none"


def test_independent_assignment_with_unequal_expected_shares_uses_declared_ratio():
    from increment.estimation.assignment_integrity import assignment_integrity

    result = assignment_integrity(
        _design(allocation={"control": 4, "treatment": 1}),
        {"control": 900, "treatment": 100},
    )
    assert result.status == "failed"
    assert result.expected == {"control": 0.8, "treatment": 0.2}


def test_experiment_declaration_round_trips_and_resolves_to_randomized_design():
    import datetime as dt

    from increment.semantics.models import Definitions

    payload = {
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM events",
                "timestamp_column": "ts",
                "entities": ["unit"],
                "facts": [
                    {"name": "purchase", "column": "value"},
                    {"name": "uptake", "column": "uptake"},
                ],
            }
        ],
        "exposures": [{"name": "assigned", "fact": "purchase"}],
        "metrics": [
            {
                "name": "metric",
                "type": "mean",
                "entity": "unit",
                "fact": "purchase",
                "aggregation": "sum",
                "window_days": 7,
            }
        ],
        "experiments": [
            {
                "name": "exp",
                "exposure": "assigned",
                "unit": "unit",
                "control_group": "control",
                "allocation": {"control": 4, "treatment": 1},
                "allocation_scheme": "independent",
                "start": dt.datetime(2025, 1, 1),
                "plan": {"secondaries": ["metric"]},
            }
        ],
    }
    definitions = Definitions.model_validate(payload)
    experiment = definitions.experiment("exp")
    assert experiment is not None
    resolved_design = experiment.resolved_design()
    assert isinstance(resolved_design, (Randomized, Encouragement))
    assert resolved_design.allocation_scheme == "independent"
    restored = Definitions.model_validate_json(definitions.model_dump_json())
    restored_experiment = restored.experiment("exp")
    assert restored_experiment is not None
    assert restored_experiment.allocation_scheme == "independent"
    legacy_experiment = {
        key: value for key, value in payload["experiments"][0].items() if key != "allocation_scheme"
    }
    legacy_payload = {**payload, "experiments": [legacy_experiment]}
    legacy = Definitions.model_validate(legacy_payload)
    assert "allocation_scheme" not in legacy.model_dump(mode="json")["experiments"][0]


def test_switchback_is_explicitly_unsupported_instead_of_using_parallel_srm():
    from increment.estimation.assignment_integrity import assignment_integrity

    result = assignment_integrity(object(), {"control": 900, "treatment": 100})
    assert result.status == "unsupported_assignment"
    assert result.code == "integrity.switchback_assignment_law"
    assert result.construction == "none"
    assert result.alpha is None


def test_experiment_allocation_scheme_reaches_encouragement_design():
    import datetime as dt

    from increment.semantics.design import Encouragement
    from increment.semantics.models import Definitions

    definitions = Definitions.model_validate(
        {
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM events",
                    "timestamp_column": "ts",
                    "entities": ["unit"],
                    "facts": [
                        {"name": "purchase", "column": "value"},
                        {"name": "uptake", "column": "uptake"},
                    ],
                }
            ],
            "exposures": [{"name": "assigned", "fact": "purchase"}],
            "metrics": [
                {
                    "name": "metric",
                    "type": "mean",
                    "entity": "unit",
                    "fact": "purchase",
                    "aggregation": "sum",
                    "window_days": 7,
                }
            ],
            "experiments": [
                {
                    "name": "exp",
                    "exposure": "assigned",
                    "unit": "unit",
                    "control_group": "control",
                    "allocation": {"control": 4, "treatment": 1},
                    "allocation_scheme": "independent",
                    "start": dt.datetime(2025, 1, 1),
                    "plan": {"secondaries": ["metric"]},
                    "design": {"mechanism": "encouragement", "uptake": {"fact": "uptake"}},
                }
            ],
        }
    )
    experiment = definitions.experiment("exp")
    assert experiment is not None
    design = experiment.resolved_design()
    assert isinstance(design, Encouragement)
    assert design.allocation_scheme == "independent"


def test_clustered_run_keeps_mixed_and_unassigned_audit_counts_in_scope(tmp_path):
    import ibis
    import pyarrow as pa

    from increment import Analysis

    definitions = tmp_path / "clustered.yml"
    definitions.write_text(
        """
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: enrolled
        column: null
      - name: revenue
        column: value
exposures:
  - name: assignment
    fact: enrolled
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: revenue
    aggregation: sum
    window_days: 7
experiments:
  - name: exp
    exposure: assignment
    unit: user_id
    cluster: store_id
    allocation: {C: 0.5, T: 0.5}
    allocation_scheme: independent
    start: 2024-01-01T00:00:00
    end: 2024-01-10T00:00:00
    control_group: C
    plan:
      secondaries: [revenue]
"""
    )

    def rows(*, include_audit_units: bool):
        events = []
        for arm in ("C", "T"):
            for index in range(20):
                unit = f"{arm}{index}"
                store = f"{arm}-store-{index}"
                events.extend(
                    [
                        {
                            "user_id": unit,
                            "group_id": arm,
                            "store_id": store,
                            "event": "enrolled",
                            "ts": "2024-01-02",
                            "value": None,
                        },
                        {
                            "user_id": unit,
                            "group_id": arm,
                            "store_id": store,
                            "event": "revenue",
                            "ts": "2024-01-09",
                            "value": float(index + (arm == "T")),
                        },
                    ]
                )
        if include_audit_units:
            for arm, store in (("C", "C-mixed"), ("T", "T-mixed")):
                events.append(
                    {
                        "user_id": "mixed",
                        "group_id": arm,
                        "store_id": store,
                        "event": "enrolled",
                        "ts": "2024-01-02",
                        "value": None,
                    }
                )
            events.append(
                {
                    "user_id": "unassigned",
                    "group_id": None,
                    "store_id": None,
                    "event": "enrolled",
                    "ts": "2024-01-02",
                    "value": None,
                }
            )
        return pa.Table.from_pylist(events)

    def run(include_audit_units: bool):
        con = ibis.duckdb.connect()
        con.create_table("events", obj=rows(include_audit_units=include_audit_units))
        analysis = Analysis.from_definitions("exp", definitions, con, on_mixed_assignment="exclude")
        return analysis.run()

    audited = run(True)
    baseline = run(False)
    scope = next(iter(audited.metadata.scope.by_source.values()))
    (integrity,) = scope.integrity
    assert integrity.randomization_grain == "cluster"
    assert integrity.observed == {"C": 20, "T": 20}
    assert integrity.context["mixed_assignment_units"] == 1
    assert integrity.context["unassigned_units"] == 1
    assert scope.source_snapshot_id != next(iter(baseline.metadata.scope.by_source))
    assert [row.group_id for row in audited] == ["T"]
