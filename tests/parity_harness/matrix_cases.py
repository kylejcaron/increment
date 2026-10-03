"""Build one parity-matrix cell on each of the six ingress constructors.

``build_case`` returns the ``ParityCase`` for the five matched-arm ingresses of a
cell and, separately, one single-ingress case for ``from_switchback_panel`` (which
identifies a different estimand and is never row-compared with the other five).
Every builder derives its inputs from the one event log in ``matrix_data``.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from increment import Analysis
from increment.estimation.engine import Method
from increment.frame import MetricSpec
from increment.plan import bind_automatic_sequential_plan
from increment.semantics.design import AdjustmentSet, Observational, Randomized
from increment.semantics.models import (
    AnalysisPlan,
    Definitions,
    ExperimentMetric,
    InferenceSpec,
    MethodSpec,
    Winsorization,
)
from increment.sequential_source import native_observation_mapping
from tests.analysis_factory import make_analysis

from . import matrix_data as md
from .cases import (
    CONSTRUCTORS,
    ParityCase,
    _export_and_replay,
    _publish_and_adopt,
    _track_connection,
)
from .matrix import Cell, Outcome, Refuses, Runs, StructuralAbsence

METRIC_NAME = "m"
_CUPED = Method(name="cuped", variance_reduction="cuped")
_CUPED_SPEC = MethodSpec(name="cuped", variance_reduction="cuped")
_REPORT_LAYER = ("total", "active")
_MARGIN = 0.05
_ALLOCATION = {"control": 0.5, "treatment": 0.5}
_SEQUENTIAL_AS_OF = dt.date(2025, 1, 19)
_RANDOMIZED = Randomized(control_group="control", allocation=_ALLOCATION)
_OBSERVATIONAL = Observational(
    control_group="control", adjustment=AdjustmentSet(covariates=("tenure",))
)


def _window(cell: Cell) -> dict[str, int]:
    return {"window_days": md.WINDOW_DAYS} if cell.windowed else {}


def _winsorization(cell: Cell) -> dict[str, float] | None:
    if cell.option == "winsor_fixed":
        return {"upper_value": md.WINSOR_FIXED_UPPER}
    if cell.option == "winsor_percentile":
        return {"upper_percentile": md.WINSOR_PERCENTILE_UPPER}
    return None


def metric_definition(cell: Cell) -> dict[str, Any]:
    """The warehouse ``Metric`` declaration of the cell's metric type."""
    window = _window(cell)
    common = {"name": METRIC_NAME, "preferred_direction": "increase"}
    out: dict[str, Any]
    entity = {"entity": "user_id"}
    match cell.base:
        case "mean":
            out = {
                **common,
                **entity,
                "type": "mean",
                "fact": "purchase",
                "aggregation": "sum",
                **window,
            }
        case "conversion":
            out = {**common, **entity, "type": "conversion", "fact": "purchase", **window}
        case "ratio":
            out = {
                **common,
                **entity,
                "type": "ratio",
                "numerator": {"fact": "purchase", "aggregation": "sum", **window},
                "denominator": {"fact": "session_end", "aggregation": "count", **window},
            }
        case "retention":
            out = {
                **common,
                **entity,
                "type": "retention",
                "fact": "purchase",
                "threshold_days": list(md.RETENTION_BAND),
                **window,
            }
        case "quantile":
            out = {
                **common,
                **entity,
                "type": "quantile",
                "fact": "latency",
                "aggregation": "sum",
                "quantile": md.QUANTILE,
                **window,
            }
        case "total":
            out = {
                "name": METRIC_NAME,
                "type": "total",
                "fact": "purchase",
                "aggregation": "sum",
                **window,
            }
        case "active":
            out = {**common, **entity, "type": "active", "fact": "purchase", **window}
        case other:
            raise AssertionError(other)
    winsorization = _winsorization(cell)
    if winsorization is not None:
        out["winsorization"] = winsorization
    return out


def plan_for(cell: Cell, *, frame: bool) -> AnalysisPlan:
    """The plan naming the metric the way the ingress family declares methods."""
    if cell.option == "cuped" and not frame:
        return AnalysisPlan(
            primary=ExperimentMetric(metric=METRIC_NAME, decision_method=_CUPED_SPEC)
        )
    if cell.option == "ni_margin":
        return AnalysisPlan(guardrails=(ExperimentMetric(metric=METRIC_NAME, margin=_MARGIN),))
    if cell.option == "sequential":
        return AnalysisPlan(primary=METRIC_NAME, inference=_sequential_inference(cell))
    return AnalysisPlan(primary=METRIC_NAME)


def _sequential_inference(cell: Cell) -> InferenceSpec:
    """Exact Bernoulli monitoring for a binary metric, asymptotic mean otherwise."""
    if cell.base in ("conversion", "retention"):
        return InferenceSpec(kind="always_valid")
    return InferenceSpec(kind="asymptotic_mean", expected_decision_sample_size=100)


def definitions_payload(cell: Cell) -> dict[str, Any]:
    """The ``Definitions`` payload; one fact source over the shared ``events`` table."""
    experiment: dict[str, Any] = {
        "name": "exp",
        "exposure": "assignment",
        "unit": "user_id",
        "start": md.WINDOW_START,
        "end": md.WINDOW_END,
        "control_group": "control",
        "day_boundary": md.BOUNDARY_SPELLING[cell.day_boundary],
        "plan": plan_for(cell, frame=False).model_dump(mode="json"),
    }
    if cell.option == "cuped":
        experiment["n_pre_periods"] = 7
    if cell.option == "observational":
        experiment["n_pre_periods"] = 14
    if cell.option == "sequential":
        experiment["allocation"] = dict(_ALLOCATION)
    if cell.view == "breakout":
        experiment["breakouts"] = [{"property": "store"}]
    if cell.option == "cluster":
        experiment["cluster"] = "cluster_id"
    if cell.option == "observational":
        experiment["design"] = {
            "mechanism": "observational",
            "covariates": [{"property": "tenure", "source": "events"}],
        }
    return {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM events",
                "timestamp_column": "event_at",
                "entities": ["user_id"],
                "facts": [
                    {"name": "exposure", "column": None},
                    {"name": "purchase", "column": "revenue"},
                    {"name": "session_end", "column": "sess"},
                    {"name": "latency", "column": "latency"},
                ],
                "properties": [
                    {"name": "store", "column": "store_id", "dtype": "string", "as_of": "static"},
                    {
                        "name": "cluster_id",
                        "column": "cluster_id",
                        "dtype": "string",
                        "as_of": "static",
                    },
                    {
                        "name": "tenure",
                        "column": "tenure",
                        "dtype": "float",
                        "as_of": "pre_exposure",
                    },
                ],
            }
        ],
        "exposures": [{"name": "assignment", "fact": "exposure"}],
        "metrics": [metric_definition(cell)],
        "experiments": [experiment],
    }


def metric_spec(cell: Cell) -> MetricSpec | dict[str, str]:
    """The frame-side declaration of the cell's metric (terse for report-layer types)."""
    if cell.base in _REPORT_LAYER:
        return {METRIC_NAME: cell.base}
    kwargs: dict[str, Any] = {
        "name": METRIC_NAME,
        "preferred_direction": "increase",
        "missing": cell.missing,
        **_window(cell),
    }
    match cell.base:
        case "mean":
            kwargs |= {"type": "mean", "value_column": "revenue"}
        case "conversion":
            kwargs |= {"type": "conversion", "value_column": "converted"}
        case "ratio":
            kwargs |= {"type": "ratio", "numerator": "revenue", "denominator": "sessions"}
        case "retention":
            kwargs |= {
                "type": "retention",
                "value_column": "returned",
                "threshold_days": md.RETENTION_BAND,
            }
        case "quantile":
            kwargs |= {"type": "quantile", "value_column": "latency", "quantile": md.QUANTILE}
        case other:
            raise AssertionError(other)
    if cell.option == "cuped":
        kwargs |= {
            "covariate": "pre_converted" if cell.base == "conversion" else "pre_revenue",
            "decision_method": _CUPED,
        }
    winsorization = _winsorization(cell)
    if winsorization is not None:
        kwargs["winsorization"] = Winsorization.model_validate(winsorization)
    return MetricSpec(**kwargs)


def _replay_spec(cell: Cell) -> MetricSpec | dict[str, str]:
    """The re-declaration a moments cube carries: the type and band, no per-unit levers."""
    spec = metric_spec(cell)
    if isinstance(spec, dict):
        return spec
    return MetricSpec(
        name=spec.name,
        type=spec.type,
        preferred_direction="increase",
        numerator=spec.numerator,
        denominator=spec.denominator,
        threshold_days=spec.threshold_days,
        quantile=spec.quantile,
        window_days=spec.window_days,
        winsorization=spec.winsorization,
    )


def _extension_kinds(cell: Cell) -> tuple[str, ...]:
    """The unit-day artifact extensions the cell's request reads."""
    kinds = []
    if cell.option == "cuped":
        kinds.append("cuped_preperiod")
    if cell.view == "breakout":
        kinds.append("breakout_dimension")
    if cell.option == "cluster":
        kinds.append("cluster_identity")
    if cell.option == "observational":
        kinds.append("unit_covariate")
    return tuple(kinds)


class _Ingress:
    """One cell on the five matched-arm constructors."""

    def __init__(self, cell: Cell) -> None:
        self.cell = cell
        self.nulls = cell.missing in ("zero", "drop")
        self.payload = definitions_payload(cell)
        if cell.missing in ("drop", "impute"):
            # The Definitions schema has no missing-policy field: declaring one is
            # the structural-absence attempt.
            self.payload["metrics"][0]["missing"] = cell.missing
        self.positive = cell.option in ("winsor_fixed", "winsor_percentile")
        self.rows = md.event_rows(positive=self.positive)
        self.plan = plan_for(cell, frame=True)

    def _con(self) -> Any:
        return md.duckdb_connection(self.rows)

    def _native(self, con: Any) -> Analysis:
        defs = Definitions.model_validate(copy.deepcopy(self.payload))
        if self.cell.option != "sequential":
            return make_analysis(con, defs, experiment="exp")
        exp = defs.experiment("exp")
        assert exp is not None
        # `Analysis.__init__` binds an automatic sequential plan's registration before
        # building the source; `make_analysis` compiles the plan as given.
        bound = bind_automatic_sequential_plan(
            exp.plan,
            [m for m in defs.metrics if m.name in exp.metric_names],
            design=_RANDOMIZED,
            source_id=exp.name,
            source_mapping=native_observation_mapping(defs, exp, on_mixed_assignment="error"),
            pre_period_covariate=exp.n_pre_periods > 0,
        )
        analysis = make_analysis(con, defs, experiment="exp", plan=bound)
        analysis._sequential_as_of = _SEQUENTIAL_AS_OF  # ty: ignore[unresolved-attribute]
        return analysis

    def definitions(self) -> Analysis:
        con = self._con()
        try:
            analysis = self._native(con)
        except BaseException:
            con.disconnect()
            raise
        return _track_connection(analysis, con)

    def artifact(self) -> Analysis:
        con = self._con()
        try:
            native = self._native(con)
        except BaseException:
            con.disconnect()
            raise
        adopted = _publish_and_adopt(con, native, _extension_kinds(self.cell))
        if self.cell.option == "sequential":
            adopted._sequential_as_of = _SEQUENTIAL_AS_OF  # ty: ignore[unresolved-attribute]
        return adopted

    def _frame_kwargs(self) -> dict[str, Any]:
        spec = metric_spec(self.cell)
        kwargs: dict[str, Any] = {
            "unit": "user_id",
            "group": "variant",
            "metrics": spec if isinstance(spec, dict) else [spec],
            "plan": self.plan,
        }
        if self.cell.option == "observational":
            kwargs["design"] = _OBSERVATIONAL
        elif self.cell.option == "sequential":
            kwargs |= {"design": _RANDOMIZED, "experiment_id": "exp"}
        else:
            kwargs["control"] = "control"
        return kwargs

    def summary(self) -> Analysis:
        kwargs = self._frame_kwargs()
        if self.cell.option == "cluster":
            kwargs["cluster"] = "cluster_id"
        if self.cell.option == "sequential":
            kwargs["exposure_date"] = "exposed_on"
        analysis = Analysis.from_unit_summary(
            md.frame(
                md.summary_rows(self.cell.day_boundary, nulls=self.nulls, positive=self.positive)
            ),
            **kwargs,
        )
        if self.cell.option == "sequential":
            analysis._sequential_as_of = _SEQUENTIAL_AS_OF  # ty: ignore[unresolved-attribute]
        return analysis

    def panel(self) -> Analysis:
        kwargs = self._frame_kwargs()
        if self.cell.option == "cluster":
            kwargs["cluster"] = "cluster_id"
        analysis = Analysis.from_unit_panel(
            md.frame(
                md.panel_rows(
                    self.cell.day_boundary,
                    nulls=self.nulls,
                    positive=self.positive,
                    daily_conversion=self.cell.view == "daily",
                )
            ),
            date="date",
            exposure_date="exposed_on",
            observation_end=md.observation_end_day(self.cell.day_boundary),
            breakouts=("store",) if self.cell.view == "breakout" else (),
            **kwargs,
        )
        if self.cell.option == "sequential":
            analysis._sequential_as_of = _SEQUENTIAL_AS_OF  # ty: ignore[unresolved-attribute]
        return analysis

    def moments(self) -> Analysis:
        cell = self.cell
        source = self.panel if cell.windowed or cell.base == "retention" else self.summary
        exported = source()
        if cell.option == "sequential":
            # A cube replays a captured checkpoint; it cannot start a sequential process.
            exported.capture_sequential(finalized=True, as_of=_SEQUENTIAL_AS_OF)
        if cell.option == "observational":
            return _export_and_replay_observational(exported, _replay_metrics(cell), self.plan)
        return _export_and_replay(exported, _replay_metrics(cell))


def _export_and_replay_observational(
    analysis: Analysis, metrics: Any, plan: AnalysisPlan
) -> Analysis:
    """`_export_and_replay` reimports as Randomized; an observational design must
    survive the replay to reach the covariate read."""
    import tempfile
    from pathlib import Path

    import pyarrow.parquet as pq

    try:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "moments.parquet"
            analysis.export(path)
            rows = pq.read_table(path).to_pylist()
        return Analysis.from_moments(rows, metrics=metrics, design=_OBSERVATIONAL, plan=plan)
    finally:
        analysis.close()


def _replay_metrics(cell: Cell) -> Any:
    spec = _replay_spec(cell)
    return spec if isinstance(spec, dict) else [spec]


def switchback_frame() -> pd.DataFrame:
    """A complete switchback schedule: eight units, two cycles, two periods, a washout
    step then one observed step, carrying every column the cell's metric may read."""
    rows: list[dict[str, Any]] = []
    for u in range(8):
        for cycle in range(2):
            order = ("control", "treatment") if (u + cycle) % 2 == 0 else ("treatment", "control")
            for period, arm in enumerate(order):
                for step in range(2):
                    value = 2.0 + (u % 3) + cycle + (1.5 if arm == "treatment" else 0.0) + 0.25 * u
                    rows.append(
                        {
                            "unit": f"u{u}",
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "arm": arm,
                            "revenue": value if step else 999.0,
                            "converted": int(step == 1 and value > 4.0),
                            "sessions": 1 + (u % 2),
                            "returned": float(step),
                            "latency": 100.0 + u + (10.0 if arm == "treatment" else 0.0),
                            "pre_revenue": 1.0 + (u % 4),
                            "pre_converted": u % 2,
                            "cluster_id": f"k{u // 2}",
                            "tenure": float(u),
                        }
                    )
    return pd.DataFrame(rows)


def _switchback_builder(cell: Cell) -> Callable[[], Analysis]:
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )

    def build() -> Analysis:
        spec = metric_spec(cell)
        kwargs: dict[str, Any] = {
            "unit": "unit",
            "cycle": "cycle",
            "period": "period",
            "step": "step",
            "group": "arm",
            "metrics": spec if isinstance(spec, dict) else [spec],
            "identification": _OBSERVATIONAL
            if cell.option == "observational"
            else Randomized(control_group="control", allocation=_ALLOCATION),
            "assignment": SwitchbackAssignment(
                sequence=IndependentBernoulliOrder(probability_ct=0.5),
                window=SwitchbackWindow(washout_steps=1, observation_steps=1),
            ),
        }
        if cell.option in ("sequential", "ni_margin"):
            kwargs["plan"] = plan_for(cell, frame=True)
        if cell.option == "cluster":
            # The signature has no cluster parameter: the unsupported keyword is the attempt.
            kwargs["cluster"] = "cluster_id"
        return Analysis.from_switchback_panel(switchback_frame(), **kwargs)

    return build


def _case_fields(cell: Cell) -> dict[str, Any]:
    return {
        "breakout_dimension": "store" if cell.view == "breakout" else None,
        "view": cell.view if cell.view in ("daily", "asof") else None,
    }


_NOT_ATTEMPTED = "SOURCE: not attempted for this request; the other ingress cases carry it"


def _expectations(outcomes: Mapping[str, Outcome], attempted: tuple[str, ...]) -> dict[str, Any]:
    """The waiver fields that make the runner assert each recorded outcome."""
    waive: dict[str, str] = {}
    codes: dict[str, str] = {}
    absence: dict[str, type[Exception]] = {}
    for name in CONSTRUCTORS:
        if name not in attempted:
            waive[name] = _NOT_ATTEMPTED
            continue
        outcome = outcomes[name]
        if isinstance(outcome, Refuses):
            waive[name] = f"refuses with {outcome.code}"
            codes[name] = outcome.code
        elif isinstance(outcome, StructuralAbsence):
            waive[name] = f"structurally absent: {outcome.fact}"
            absence[name] = outcome.error
    return {
        "waive": waive,
        "waived_refusal_codes": codes,
        "expected_absence": absence,
        "refusal_only": not any(isinstance(outcomes[n], Runs) for n in attempted),
    }


MATCHED = (
    "from_definitions",
    "from_unit_day_artifact",
    "from_unit_summary",
    "from_unit_panel",
    "from_moments",
)


def build_case(cell: Cell, outcomes: Mapping[str, Outcome]) -> ParityCase:
    """The matched-arm `ParityCase` for *cell*: the five ingresses compared with each other.

    *outcomes* are the recorded outcomes; the runner asserts each one, so a stale
    refusal code, a refusal that became a number or a number that became a refusal fails.
    """
    ingress = _Ingress(cell)
    builders: dict[str, Callable[[], Analysis]] = {
        "from_definitions": ingress.definitions,
        "from_unit_day_artifact": ingress.artifact,
        "from_unit_summary": ingress.summary,
        "from_unit_panel": ingress.panel,
        "from_moments": ingress.moments,
    }
    return ParityCase(
        id=cell.id,
        build=builders,
        sequential=cell.option == "sequential",
        **_expectations(outcomes, MATCHED),
        **_case_fields(cell),
    )


def build_switchback_case(cell: Cell, outcomes: Mapping[str, Outcome]) -> ParityCase:
    """`from_switchback_panel` on its own: a different estimand, never row-compared."""
    return ParityCase(
        id=f"{cell.id}-switchback",
        build={"from_switchback_panel": _switchback_builder(cell)},
        **_expectations(outcomes, ("from_switchback_panel",)),
        **_case_fields(cell),
    )


def memo_key(cell: Cell, name: str) -> tuple[Any, ...] | None:
    """Identical warehouse inputs: the digest of what the builder will read.

    ``missing`` error and zero declare the same Definitions payload, so a cell that
    differs only there reads the same events through the same request and reaches the same
    result; the executor runs it once per process.
    """
    if name not in ("from_definitions", "from_unit_day_artifact"):
        return None
    payload = definitions_payload(cell)
    if cell.missing in ("drop", "impute"):
        payload["metrics"][0]["missing"] = cell.missing
    return (
        name,
        json.dumps(payload, sort_keys=True),
        cell.view,
        cell.option,
        cell.option in ("winsor_fixed", "winsor_percentile"),
        _extension_kinds(cell),
    )
