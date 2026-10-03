"""Source operation refusals and runtime protocol checks."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from increment._source_operations import PanelSQLOperation, SummarySqlOperation
from increment.analysis import Analysis
from increment.errors import CapabilityError
from increment.query._native_triggered import TriggeredPopulationSource
from increment.sources import (
    MomentSource,
    SourceOperation,
    require_operation,
)
from tests.test_native_core_operations import _assert_operation_refusal

if TYPE_CHECKING:
    from pydantic import BaseModel

    from increment._readout_request import ReadoutView
    from increment._source_types import Grain
    from increment.semantics.models import Metric


def test_triggered_population_source_limits_grains_and_uses_triggered_counts() -> None:
    source = SimpleNamespace(
        capabilities=frozenset({"total", "daily", "asof"}),
        breakouts=(),
        shape=None,
        _validate_trigger_capability=lambda *, operation: None,
        triggered_counts=lambda: (
            "cluster",
            {"control": 2, "treatment": 3},
            {"control": 7, "treatment": 11},
        ),
    )

    triggered = TriggeredPopulationSource(source)

    assert triggered.capabilities == frozenset({"total"})
    assert triggered.unit_counts() == {"control": 7, "treatment": 11}
    assert triggered.cluster_counts() == {"control": 2, "treatment": 3}
    with pytest.raises(CapabilityError) as raised:
        triggered.sql()
    _assert_operation_refusal(
        raised.value,
        operation="sql",
        request={"grain": "total", "population": "triggered"},
        offered=("total",),
    )


def test_triggered_population_source_uses_primary_counts_for_unclustered_units() -> None:
    source = SimpleNamespace(
        capabilities=frozenset({"total", "daily", "asof"}),
        breakouts=(),
        shape=None,
        _validate_trigger_capability=lambda *, operation: None,
        triggered_counts=lambda: ("unit", {"control": 7, "treatment": 11}, {}),
    )

    assert TriggeredPopulationSource(source).unit_counts() == {
        "control": 7,
        "treatment": 11,
    }


class _OperationsDescriptorSource:
    """Arm source whose `operations` descriptor raises when it is read."""

    context = SimpleNamespace(study_id="descriptor", design=None)
    descriptor_reads = 0

    @property
    def operations(self) -> frozenset[SourceOperation]:
        type(self).descriptor_reads += 1
        raise AttributeError("operations descriptor failed")


def test_from_source_does_not_invoke_operations_descriptor() -> None:
    source = cast(MomentSource, _OperationsDescriptorSource())

    Analysis._from_source(source)

    assert _OperationsDescriptorSource.descriptor_reads == 0
    # The descriptor is genuinely armed: touching it raises and counts.
    with pytest.raises(AttributeError):
        _ = source.operations
    assert _OperationsDescriptorSource.descriptor_reads == 1


def test_require_operation_rejects_declaration_drift() -> None:
    from tests.source_conformance import sql_totals_source

    source: MomentSource = sql_totals_source()
    assert require_operation(source, "summary_sql", SummarySqlOperation) is source
    with pytest.raises(CapabilityError) as raised:
        require_operation(source, "panel_sql", PanelSQLOperation)
    assert raised.value.code == "source.operation.unsupported"
    assert raised.value.context == {
        "operation": "panel_sql",
        "source": "SqlPanelSource",
    }


# Capability policy at the validation seam: metric type x view x option x substrate.
# Construction rules are raised by the shared gate (`GATE_POLICY`); source limits derive
# from each source's declared `capabilities` and `breakouts`. Axes are read from code.

_VIEW_GRAIN: dict[ReadoutView, Grain] = {
    "run": "total",
    "breakout": "total",
    "daily": "daily",
    "asof": "asof",
}
_OPTIONS = (
    "none",
    "percentile_winsorization",
    "fixed_winsorization",
    "unbounded_band",
    "cluster",
    "cuped",
)
_QUANTILE_GRAIN = "readout.metric.quantile_grain"
# Raised by `validate_readout_source` after the grain and dimension checks, so a source
# that lacks the grain reports the source limit first.
_SOURCE_STAGE_CONSTRUCTION_CODES = frozenset({_QUANTILE_GRAIN})
_DAILY_WINSOR = "readout.metric.daily_winsorization"
_PERCENTILE_WINSOR = "readout.metric.percentile_winsorization"
_QUANTILE_BREAKOUT = "readout.metric.quantile_breakout"
_QUANTILE_CLUSTER = "arm.metric.quantile_cluster"
_QUANTILE_CUPED = "arm.metric.quantile_cuped"
_UNBOUNDED_RETENTION = "breakout.retention.unbounded"
# The gate accepts the request; an explicit decision, distinct from an undeclared cell.
_ACCEPTED = "accepted"


def _by_view(*, run: str, breakout: str, daily: str, asof: str) -> dict[str, str]:
    return {"run": run, "breakout": breakout, "daily": daily, "asof": asof}


def _accepted_everywhere() -> dict[str, str]:
    return _by_view(run=_ACCEPTED, breakout=_ACCEPTED, daily=_ACCEPTED, asof=_ACCEPTED)


# metric type -> option -> view -> `_ACCEPTED` or the refusal code the shared gate raises.
# Every applicable cell is declared: `_policy_gaps` fails on a missing or stale one.
GATE_POLICY: dict[str, dict[str, dict[str, str]] | None] = {
    "mean": {
        "none": _accepted_everywhere(),
        "percentile_winsorization": _by_view(
            run=_ACCEPTED, breakout=_PERCENTILE_WINSOR, daily=_DAILY_WINSOR, asof=_PERCENTILE_WINSOR
        ),
        "fixed_winsorization": _by_view(
            run=_ACCEPTED, breakout=_ACCEPTED, daily=_DAILY_WINSOR, asof=_ACCEPTED
        ),
        "cluster": _accepted_everywhere(),
        "cuped": _accepted_everywhere(),
    },
    "conversion": {
        "none": _accepted_everywhere(),
        "cluster": _accepted_everywhere(),
        "cuped": _accepted_everywhere(),
    },
    "ratio": {
        "none": _accepted_everywhere(),
        "cluster": _accepted_everywhere(),
        "cuped": _accepted_everywhere(),
    },
    "retention": {
        "none": _accepted_everywhere(),
        "unbounded_band": _by_view(
            run=_ACCEPTED, breakout=_ACCEPTED, daily=_UNBOUNDED_RETENTION, asof=_ACCEPTED
        ),
        "cluster": _accepted_everywhere(),
        "cuped": _accepted_everywhere(),
    },
    "quantile": {
        "none": _by_view(
            run=_ACCEPTED, breakout=_QUANTILE_BREAKOUT, daily=_QUANTILE_GRAIN, asof=_QUANTILE_GRAIN
        ),
        "cluster": _by_view(
            run=_QUANTILE_CLUSTER,
            breakout=_QUANTILE_CLUSTER,
            daily=_QUANTILE_GRAIN,
            asof=_QUANTILE_CLUSTER,
        ),
        "cuped": _by_view(
            run=_QUANTILE_CUPED,
            breakout=_QUANTILE_BREAKOUT,
            daily=_QUANTILE_GRAIN,
            asof=_QUANTILE_CUPED,
        ),
    },
    # Report-layer types cannot be held by a source (`MATRIX["estimate"]` is NA).
    "total": None,
    "active": None,
}

# `MATRIX` capability column -> the (view, option) cell of this seam it also declares.
_MATRIX_SEAM_CELLS: dict[str, tuple[tuple[ReadoutView, str], ...]] = {
    "breakout": (("breakout", "none"),),
    "daily_asof": (("daily", "none"), ("asof", "none")),
}


def _metric_model(metric_type: str) -> type[BaseModel]:
    from typing import get_args

    from increment.semantics import models

    union, _field = get_args(models.Metric)
    for member in get_args(union):
        if get_args(member.model_fields["type"].annotation) == (metric_type,):
            return member
    raise AssertionError(f"no semantics model for metric type {metric_type!r}")


def _option_applies(metric_type: str, option: str) -> bool:
    """Whether the metric model can express the option, read from its fields."""
    fields = _metric_model(metric_type).model_fields
    if option in ("percentile_winsorization", "fixed_winsorization"):
        return "winsorization" in fields
    if option == "unbounded_band":
        return "threshold_days" in fields
    return True


def _synthetic_metric(metric_type: str, option: str) -> Metric:
    from increment.semantics import models

    winsorization = {
        "percentile_winsorization": models.Winsorization(upper_percentile=0.99),
        "fixed_winsorization": models.Winsorization(upper_value=100.0),
    }.get(option)
    name, entity = "outcome", "user_id"
    if metric_type == "mean":
        return models.MeanMetric(
            name=name, entity=entity, fact="f", aggregation="sum", winsorization=winsorization
        )
    if metric_type == "conversion":
        return models.ConversionMetric(name=name, entity=entity, fact="f")
    if metric_type == "retention":
        return models.RetentionMetric(
            name=name,
            entity=entity,
            fact="f",
            threshold_days=1 if option == "unbounded_band" else (1, 3),
        )
    if metric_type == "ratio":
        return models.RatioMetric(
            name=name,
            entity=entity,
            numerator=models.Measure(fact="a", aggregation="sum"),
            denominator=models.Measure(fact="b", aggregation="sum"),
        )
    if metric_type == "quantile":
        return models.QuantileMetric(
            name=name, entity=entity, fact="f", aggregation="sum", quantile=0.5
        )
    raise AssertionError(
        f"metric type {metric_type!r} has no synthetic builder: declare one here and "
        "its policy in GATE_POLICY"
    )


def _gate_request(source, metric_type: str, view: ReadoutView, option: str):
    import dataclasses

    from increment._analysis_config import ResolvedMetricConfig
    from increment._readout_request import ReadoutRequest
    from increment.estimation.engine import Method

    metric = _synthetic_metric(metric_type, option)
    context = source.context
    procedure = next(iter(context.plan.procedures.values()))
    plan = context.plan.model_copy(
        update={"procedures": {metric.name: procedure.model_copy(update={"metric": metric.name})}}
    )
    method = (
        Method(name="cuped", variance_reduction="cuped")
        if option == "cuped"
        else Method(name="unadjusted")
    )
    config = ResolvedMetricConfig(
        metric=metric,
        decision_method=method,
        sensitivity_methods=(),
        prior=None,
        prior_is_global=False,
        decision_defaulted=True,
    )
    breakouts = tuple(getattr(source, "breakouts", ()))
    dimension = (
        (breakouts[0] if breakouts else "undeclared_dimension") if view == "breakout" else None
    )
    request = ReadoutRequest.from_source(
        source,
        metrics=[metric],
        configs=[config],
        view=view,
        grain=_VIEW_GRAIN[view],
        by=(dimension,) if dimension else (),
        dimension=dimension,
    )
    return dataclasses.replace(
        request,
        context=dataclasses.replace(
            context,
            plan=plan,
            metrics=(metric,),
            configs=(config,),
            cluster="cluster_id" if option == "cluster" else context.cluster,
        ),
        cluster="cluster_id" if option == "cluster" else context.cluster,
    )


def _declared_gate_outcome(source, metric_type: str, view: ReadoutView, option: str) -> str | None:
    """The code `validate_request` must raise: construction rules, else the source's limit."""
    policy = GATE_POLICY[metric_type]
    assert policy is not None
    declared = policy[option][view]
    construction = None if declared == _ACCEPTED else declared
    request_by = tuple(getattr(source, "breakouts", ()))
    if _VIEW_GRAIN[view] not in source.capabilities:
        source_limit = "readout.source.grain"
    elif view == "breakout" and not request_by:
        source_limit = "readout.source.dimension"
    else:
        source_limit = None
    if construction in _SOURCE_STAGE_CONSTRUCTION_CODES:
        return source_limit or construction
    return construction or source_limit


@pytest.fixture(scope="module")
def _substrates(tmp_path_factory: pytest.TempPathFactory):
    from tests.source_conformance import arm_adapters

    return {
        adapter.name: adapter.build()[0]
        for adapter in arm_adapters(tmp_path_factory.mktemp("substrates"))
    }


def _policy_gaps() -> list[tuple[object, ...]]:
    """Undeclared or stale cells of the metric type x option x view product."""
    from typing import get_args

    from increment._readout_request import ReadoutView

    views = set(get_args(ReadoutView))
    gaps: list[tuple[object, ...]] = []
    if set(_VIEW_GRAIN) != views:
        gaps.append(("views without a grain", views ^ set(_VIEW_GRAIN)))
    for metric_type, options in GATE_POLICY.items():
        if options is None:
            continue
        applicable = {option for option in _OPTIONS if _option_applies(metric_type, option)}
        gaps.extend((metric_type, option, "undeclared") for option in applicable - set(options))
        gaps.extend((metric_type, option, "stale") for option in set(options) - applicable)
        for option, outcomes in options.items():
            gaps.extend((metric_type, option, view, "undeclared") for view in views - set(outcomes))
            gaps.extend((metric_type, option, view, "stale") for view in set(outcomes) - views)
    return gaps


def test_gate_policy_covers_the_code_derived_axes() -> None:
    from tests.compatibility_catalog import MATRIX
    from tests.test_composition_matrix import METRIC_TYPES

    assert set(GATE_POLICY) == set(METRIC_TYPES), (
        "decide the validation-seam policy for every metric type: "
        f"missing {set(METRIC_TYPES) - set(GATE_POLICY)}, "
        f"stale {set(GATE_POLICY) - set(METRIC_TYPES)}"
    )
    for metric_type, options in GATE_POLICY.items():
        report_only = MATRIX["estimate"][metric_type].status == "na"
        assert (options is None) == report_only, metric_type
    _assert_policy_complete()


def _assert_policy_complete() -> None:
    gaps = _policy_gaps()
    assert not gaps, (
        "decide the validation-seam outcome (_ACCEPTED or a refusal code) for every "
        f"applicable metric x option x view: {gaps}"
    )


@pytest.mark.parametrize("metric_type", list(GATE_POLICY))
def test_gate_outcome_matches_declared_policy_on_every_substrate(
    metric_type: str, _substrates: dict[str, object]
) -> None:
    """Every registered substrate raises the declared code, or its own grain/dimension limit.

    A new metric type fails on its missing builder or `GATE_POLICY` entry, a new view or
    option on its missing declarations, and a new substrate is swept with the derived
    rules. Options a source fixes at construction (sequential inference, observational and
    encouragement designs) are not request-level variants; the arm-contract sweeps in
    `test_refusal_uniqueness.py` and the pair cells in `tests/compatibility_catalog.py`
    own them.
    """
    from increment._readout_request import validate_request
    from increment.errors import CodedError

    _assert_policy_complete()
    if GATE_POLICY[metric_type] is None:
        pytest.skip("report-layer type: no source can hold it (MATRIX 'estimate' is NA)")
    mismatches = []
    for name, source in _substrates.items():
        for view in _VIEW_GRAIN:
            for option in _OPTIONS:
                if not _option_applies(metric_type, option):
                    continue
                expected = _declared_gate_outcome(source, metric_type, view, option)
                try:
                    validate_request(_gate_request(source, metric_type, view, option))
                    actual = None
                except CodedError as error:
                    actual = error.code
                if actual != expected:
                    mismatches.append((name, view, option, actual, expected))
    assert not mismatches, (
        f"{metric_type}: (substrate, view, option, actual, declared) differ: {mismatches}"
    )


def test_gate_declarations_agree_with_the_composition_matrix() -> None:
    """`MATRIX` cells for the day-axis and breakout capabilities name the same code the
    gate raises; a supported cell means the gate accepts the plain request."""
    from tests.compatibility_catalog import MATRIX

    for capability, seam_cells in _MATRIX_SEAM_CELLS.items():
        for metric_type, options in GATE_POLICY.items():
            if options is None:
                continue
            assert metric_type in MATRIX[capability], (
                f"MATRIX[{capability!r}] declares no cell for {metric_type!r}: decide it"
            )
            cell = MATRIX[capability][metric_type]
            for view, option in seam_cells:
                declared = options[option][view]
                if cell.status == "refused":
                    assert cell.code == declared, (capability, metric_type, view, option)
                else:
                    assert declared == _ACCEPTED, (capability, metric_type, view, option)
