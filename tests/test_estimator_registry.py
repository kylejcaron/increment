from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from increment._evidence_types import (
    CountsRequest,
    MethodFitRequest,
    MomentsRequest,
    UnitRowsRequest,
)
from increment._metric_specs import MetricSpec
from increment.declarations import ADJUSTMENTS, EstimatorRef, Method, MomentsSpec, UnitRowsSpec
from increment.errors import CodedError
from increment.estimators.registry import (
    ESTIMATORS,
    estimator_id,
    load,
    observational_estimator_id,
    randomized_estimator_id,
    register,
    routes,
)


def test_every_legacy_name_resolves_to_its_registered_adjustment():
    for name in ("iptw", "dml", "aipw"):
        assert load(estimator_id(Method(name=name))) is ADJUSTMENTS.get(name)


def test_custom_adjustment_registration_is_a_first_class_estimator():
    from tests.estimation.registry_helpers import registered_adjustment

    def fn(*_args, **_kwargs):
        return []

    name = "registry_test_custom_adjustment"
    estimator = f"observational.{name}"
    with registered_adjustment(name, fn):
        assert Method(name=name).is_observational
        assert estimator_id(Method(name=name)) == estimator
        assert load(estimator) is fn
    assert not Method(name=name).is_observational
    assert estimator not in ESTIMATORS
    with pytest.raises(CodedError):
        ADJUSTMENTS.get(name)


def test_unknown_observational_name_is_refused_instead_of_becoming_unadjusted():
    with pytest.raises(CodedError):
        observational_estimator_id(Method(name="registry_test_unknown"))


def test_same_method_resolves_differently_by_path():
    method = Method(name="registry_test_unknown")
    assert randomized_estimator_id(method) == "randomized.unadjusted"
    with pytest.raises(CodedError):
        observational_estimator_id(method)


def test_registry_routes_build_doorway_requests_with_population_and_covariates():
    metric = MetricSpec(name="outcome", type="mean", value_column="y")
    method = Method(name="iptw")
    ref = ESTIMATORS["observational.iptw"]
    ctx = {
        "metric": metric,
        "population": "triggered",
        "method": method,
        "control": "control",
        "arm_order": ("control", "treatment"),
        "design_covariates": ("age", "site"),
        "method_covariates": (),
    }
    all_routes = routes(ref, ctx)
    assert all(
        isinstance(request, (MomentsRequest, UnitRowsRequest, MethodFitRequest, CountsRequest))
        for route in all_routes
        for request in route
    )
    assert all(request.population == "triggered" for route in all_routes for request in route)
    row_requests = [
        request for route in all_routes for request in route if isinstance(request, UnitRowsRequest)
    ]
    assert row_requests and row_requests[0].covariates == ("age", "site")


def test_routes_refuse_when_design_covariates_are_absent_from_context():
    ref = ESTIMATORS["observational.iptw"]
    with pytest.raises(CodedError):
        routes(
            ref,
            {
                "metric": MetricSpec(name="outcome", type="mean", value_column="y"),
                "method": Method(name="iptw"),
            },
        )


def test_custom_needs_keep_population_and_do_not_import_implementation():
    import sys

    from increment.declarations import CountsSpec, MethodFitSpec, MomentsSpec, UnitRowsSpec

    ref = EstimatorRef(
        estimator_id="observational.registry_population",
        module="not_imported_registry_estimator",
        attribute="estimate",
        contract="panel",
        estimand="ate",
        selector="registry_population",
        needs=(
            (
                MomentsSpec("total", method_specific=True, include_covariate=True),
                UnitRowsSpec("raw", "method"),
                CountsSpec("triggered"),
                MethodFitSpec(),
            ),
            (MomentsSpec("total"), UnitRowsSpec(covariates="design")),
        ),
    )
    register(ref)
    try:
        built = routes(ref, {**_context(), "population": "triggered"})
        assert "not_imported_registry_estimator" not in sys.modules
        assert all(request.population == "triggered" for route in built for request in route)
        assert isinstance(built[0][0], MomentsRequest)
        assert built[0][0].method is not None and built[0][0].include_covariate
        assert built[0][1].outcome_stage == "raw"
        assert built[0][1].covariates == ("pre",)
        assert isinstance(built[0][2], CountsRequest) and built[0][2].kind == "triggered"
        assert isinstance(built[0][3], MethodFitRequest)
        assert isinstance(built[1][1], UnitRowsRequest)
        assert built[1][1].covariates == ("age",)
    finally:
        from increment.declarations import unregister_estimator

        unregister_estimator(ref.estimator_id)


def test_observational_estimator_registration_requires_total_moments_and_design_rows():
    from increment.declarations import UnitRowsSpec

    invalid = EstimatorRef(
        estimator_id="observational.missing_base_route",
        module="module",
        attribute="estimate",
        contract="panel",
        estimand="ate",
        selector="missing_base_route",
        needs=((UnitRowsSpec(covariates="design"),),),
    )
    with pytest.raises(ValueError):
        register(invalid)


def test_distinct_same_setting_sensitivity_methods_produce_distinct_rows():
    import pandas as pd

    from increment import Analysis

    analysis = Analysis.from_unit_summary(
        pd.DataFrame(
            {
                "unit": range(8),
                "arm": ["c"] * 4 + ["t"] * 4,
                "outcome": [1, 2, 3, 4, 2, 3, 4, 5],
            }
        ),
        unit="unit",
        group="arm",
        control="c",
        metrics={"outcome": "mean"},
    )
    rows = analysis.run(
        sensitivity_methods=(
            Method.unadjusted(name="same_setting_a"),
            Method.unadjusted(name="same_setting_b"),
        )
    )
    assert len(rows) == 3
    assert {row.method for row in rows} == {"unadjusted", "same_setting_a", "same_setting_b"}


@pytest.mark.integration
def test_observational_decision_roles_resolve_in_a_fresh_process():
    code = """\\
import sys
import pandas as pd
from increment import Analysis, Method
from increment.semantics.design import AdjustmentSet, IdentificationGate, Observational
assert 'increment.estimation._adjust.iptw' not in sys.modules
frame = pd.DataFrame({'unit': range(100), 'arm': ['c'] * 50 + ['t'] * 50, 'y': [float(i % 7) for i in range(100)], 'x': [i % 2 for i in range(100)]})
analysis = Analysis.from_unit_summary(frame, unit='unit', group='arm', metrics={'y': 'mean'}, design=Observational(control_group='c', adjustment=AdjustmentSet(covariates=('x',)), gate=IdentificationGate(overlap='trim')))
rows = analysis.run(decision_method=Method(name='iptw'), sensitivity_methods=[Method.unadjusted(name='none')])
roles = {(row.method, row.method_role) for row in rows}
assert ('iptw', 'decision') in roles and ('none', 'sensitivity') in roles
try:
    analysis.run(decision_method=Method(name='nope'))
except Exception as exc:
    assert exc.code == 'estimation.variance.registry.no_registered_available'
else:
    raise AssertionError('unknown observational estimator accepted')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_each_observational_route_requires_total_moments_and_design_rows():
    for ref in ESTIMATORS.values():
        if ref.estimator_id.startswith("observational."):
            assert any(
                any(isinstance(spec, MomentsSpec) and spec.grain == "total" for spec in route)
                and any(
                    isinstance(spec, UnitRowsSpec) and spec.covariates == "design" for spec in route
                )
                for route in ref.needs
            )


def test_observational_routes_require_total_moments_and_unit_rows_for_acquisition():
    from increment._evidence_refusals import require_evidence
    from increment._source_types import EvidenceCapabilities
    from increment.errors import CapabilityError

    route = routes(ESTIMATORS["observational.iptw"], _context())[0]

    def provider(grains):
        return type(
            "Provider",
            (),
            {
                "evidence": EvidenceCapabilities(
                    kind="unit_summary",
                    grains=frozenset(grains),
                    breakouts=(),
                    shape="unit_summary",
                    operations=frozenset(),
                    unit_rows=True,
                    raw_outcomes=False,
                    method_fit="none",
                    compliance_axis="none",
                    count_grains=frozenset(),
                    populations=frozenset({"assigned"}),
                    sequential_capture="none",
                    scalar=None,
                    available_covariates=frozenset({"age"}),
                )
            },
        )()

    with pytest.raises(CapabilityError):
        for request in route:
            require_evidence(provider(()), request=request)
    available = provider(("total",))
    for request in route:
        require_evidence(available, request=request)


def test_scalar_cuped_route_is_supported_by_moments_only_evidence():
    from increment._evidence_refusals import require_evidence
    from increment._source_types import EvidenceCapabilities

    request = routes(ESTIMATORS["randomized.cuped"], _context())[0][0]
    source = type(
        "MomentsOnly",
        (),
        {
            "evidence": EvidenceCapabilities(
                kind="moments",
                grains=frozenset({"total"}),
                breakouts=(),
                shape=None,
                operations=frozenset(),
                unit_rows=False,
                raw_outcomes=False,
                method_fit="none",
                compliance_axis="none",
                count_grains=frozenset(),
                populations=frozenset({"assigned"}),
                sequential_capture="none",
                scalar=None,
                available_covariates=None,
            )
        },
    )()
    require_evidence(source, request=request)


def test_cuped_alternative_routes_match_provider_fit_and_local_capabilities():
    from increment._evidence_refusals import require_evidence
    from increment._source_types import EvidenceCapabilities

    all_requests = routes(ESTIMATORS["randomized.cuped_multicovariate"], _context())
    fit_source = type(
        "FitSource",
        (),
        {
            "evidence": EvidenceCapabilities(
                kind="unit_summary",
                grains=frozenset({"total"}),
                breakouts=(),
                shape="unit_summary",
                operations=frozenset(),
                unit_rows=False,
                raw_outcomes=False,
                method_fit="warehouse",
                compliance_axis="none",
                count_grains=frozenset(),
                populations=frozenset({"assigned"}),
                sequential_capture="none",
                scalar=None,
                available_covariates=None,
            )
        },
    )()
    local_source = type(
        "LocalSource",
        (),
        {
            "evidence": EvidenceCapabilities(
                kind="unit_summary",
                grains=frozenset({"total"}),
                breakouts=(),
                shape="unit_summary",
                operations=frozenset(),
                unit_rows=True,
                raw_outcomes=False,
                method_fit="none",
                compliance_axis="none",
                count_grains=frozenset(),
                populations=frozenset({"assigned"}),
                sequential_capture="none",
                scalar=None,
                available_covariates=frozenset({"age"}),
            )
        },
    )()
    assert isinstance(all_requests[0][0], MethodFitRequest)
    require_evidence(fit_source, request=all_requests[0][0])
    for request in all_requests[1]:
        require_evidence(local_source, request=request)


def test_routes_preserve_triggered_population_as_an_acquisition_requirement():
    from increment._evidence_refusals import require_evidence
    from increment._source_types import EvidenceCapabilities
    from increment.errors import CapabilityError

    request = routes(ESTIMATORS["randomized.cuped"], {**_context(), "population": "triggered"})[0][
        0
    ]
    source = type(
        "AssignedOnly",
        (),
        {
            "evidence": EvidenceCapabilities(
                kind="moments",
                grains=frozenset({"total"}),
                breakouts=(),
                shape=None,
                operations=frozenset(),
                unit_rows=False,
                raw_outcomes=False,
                method_fit="none",
                compliance_axis="none",
                count_grains=frozenset(),
                populations=frozenset({"assigned"}),
                sequential_capture="none",
                scalar=None,
                available_covariates=None,
            )
        },
    )()
    with pytest.raises(CapabilityError):
        require_evidence(source, request=request)


def test_route_spec_literals_build_their_matching_request_values():
    from increment.declarations import CountsSpec, MomentsSpec, UnitRowsSpec

    ref = EstimatorRef(
        estimator_id="randomized.registry_literal_specs",
        module="module",
        attribute="estimate",
        contract="panel",
        estimand="ate",
        selector=None,
        needs=(
            (MomentsSpec("daily"), UnitRowsSpec("transformed", "none"), CountsSpec("unit")),
            (MomentsSpec("asof"), UnitRowsSpec("raw", "method"), CountsSpec("cluster")),
            (CountsSpec("assignment"), CountsSpec("triggered")),
        ),
    )
    built = routes(ref, _context())
    moments = [
        request for route in built for request in route if isinstance(request, MomentsRequest)
    ]
    rows = [request for route in built for request in route if isinstance(request, UnitRowsRequest)]
    counts = [request for route in built for request in route if isinstance(request, CountsRequest)]
    assert [request.grain for request in moments] == ["daily", "asof"]
    assert [(request.outcome_stage, request.covariates) for request in rows] == [
        ("transformed", ()),
        ("raw", ("pre",)),
    ]
    assert [request.kind for request in counts] == ["unit", "cluster", "assignment", "triggered"]


def _context():
    return {
        "metric": MetricSpec(name="outcome", type="mean", value_column="y"),
        "population": "assigned",
        "method": Method(name="naive"),
        "control": "control",
        "arm_order": ("control", "treatment"),
        "design_covariates": ("age",),
        "method_covariates": ("pre",),
    }


def test_cuped_declares_fit_or_local_routes_and_scalar_cuped_moments_only():
    multi = ESTIMATORS["randomized.cuped_multicovariate"]
    built = routes(multi, _context())
    assert any(any(isinstance(request, MethodFitRequest) for request in route) for route in built)
    assert any(any(isinstance(request, UnitRowsRequest) for request in route) for route in built)
    scalar = routes(ESTIMATORS["randomized.cuped"], _context())
    assert len(scalar) == 1
    assert len(scalar[0]) == 1 and isinstance(scalar[0][0], MomentsRequest)
    assert scalar[0][0].method is not None


def test_custom_estimator_needs_flow_to_routes_and_manifest():
    from increment.declarations import CountsSpec, MomentsSpec, UnitRowsSpec
    from scripts import render_estimator_manifest

    custom_id = "observational.registry_custom_needs"
    ref = EstimatorRef(
        estimator_id=custom_id,
        module="test_module",
        attribute="estimate",
        contract="panel",
        estimand="ate",
        selector=None,
        needs=((MomentsSpec("total"), UnitRowsSpec(covariates="design")), (CountsSpec("cluster"),)),
    )
    register(ref)
    try:
        generated = routes(ref, _context())
        assert isinstance(generated[0][0], MomentsRequest)
        assert isinstance(generated[0][1], UnitRowsRequest)
        assert isinstance(generated[1][0], CountsRequest)
        assert generated[1][0].kind == "cluster"
        manifest = render_estimator_manifest.render(refs=(ref,))
        assert custom_id in manifest
    finally:
        from increment.declarations import unregister_estimator

        unregister_estimator(custom_id)


def test_manifest_is_current():
    from scripts import render_estimator_manifest

    assert render_estimator_manifest.check(Path(__file__).parents[1]) == []


@pytest.mark.integration
def test_builtins_load_in_fresh_process_without_preimporting_adjust_or_engine():
    code = """\
import sys
from increment.estimators.registry import load
fn = load('observational.iptw')
from increment.declarations import ADJUSTMENTS
assert fn is ADJUSTMENTS.get('iptw')
assert 'increment.estimation.adjust' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.integration
def test_known_and_unknown_legacy_adjustment_lookups_resolve_lazily():
    code = """\
import sys
from increment.declarations import ADJUSTMENTS
assert ADJUSTMENTS.get('iptw')
assert 'increment.estimation.adjust' not in sys.modules
try:
    ADJUSTMENTS.get('never_registered_registry_probe')
except Exception as exc:
    assert exc.code == 'estimation.variance.registry.no_registered_available'
    assert {'iptw', 'aipw', 'dml'} <= set(exc.context['available'])
else:
    raise AssertionError('unknown name accepted')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.integration
def test_new_observational_estimator_runs_through_analysis_in_fresh_process(tmp_path):
    module = tmp_path / "dummy_estimator.py"
    module.write_text("""\\
from increment.declarations import EstimatorRef, MomentsSpec, UnitRowsSpec, register, register_implementation
from increment.estimation.results import Estimate, LiftEstimate

ESTIMATOR_ID = 'observational.dummy'
CONTRACT = 'panel'
NEEDS = ((MomentsSpec('total'), UnitRowsSpec(covariates='design')),)
def estimate(frame, moments, cluster, metric, design, **kwargs):
    groups = sorted(str(group) for group in frame['group_id'].unique())
    return [LiftEstimate(metric=metric.name, group_id=group, method='dummy', method_role='decision', lift=Estimate(value=0.1), note='registry-sentinel') for group in groups if group != design.control_group]
register(EstimatorRef(ESTIMATOR_ID, __name__, 'estimate', CONTRACT, 'ate', 'dummy', NEEDS))
register_implementation(ESTIMATOR_ID, estimate)
""")
    code = """\\
import importlib.util, sys
from increment import Analysis, Method
from increment.semantics.design import Observational, AdjustmentSet, IdentificationGate
import pandas as pd
spec = importlib.util.spec_from_file_location('dummy_estimator', sys.argv[1])
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
frame = pd.DataFrame({'unit': range(8), 'arm': ['c'] * 4 + ['t'] * 4, 'y': [1, 2, 3, 4, 2, 3, 4, 5], 'x': [0, 1, 0, 1] * 2})
analysis = Analysis.from_unit_summary(frame, unit='unit', group='arm', metrics={'y': 'mean'}, design=Observational(control_group='c', adjustment=AdjustmentSet(covariates=('x',)), gate=IdentificationGate(overlap='trim')))
rows = analysis.run(decision_method=Method(name='dummy'))
assert any(row.note and 'registry-sentinel' in row.note for row in rows)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(module)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
