"""Declared raw sequential cases shared by public Bernoulli and diagnostic tests."""

from fractions import Fraction
from typing import NoReturn

from increment import (
    AlwaysValid,
    JointReveal,
    PredictivePrior,
    ScalarMeanModel,
    SequentialCell,
    SequentialModel,
    SequentialRegistration,
    capture_sequential_snapshot,
)
from increment import (
    estimate_sequential as _public_estimate_sequential,
)
from increment.estimation.sequential_runtime import _evaluate_sequential_diagnostic
from increment.semantics.models import InferenceSpec, MeanMetric
from increment.sequential_state import _capture_sequential_diagnostic_snapshot


def registration(law="bernoulli", *, cells=None, alpha=Fraction(1, 20), models=None):
    prior = (
        PredictivePrior(kind="beta", a=1, b=1)
        if law == "bernoulli"
        else PredictivePrior(kind="nig", kappa=1, nu=2, mean=(0,), scale=((2,),))
        if law == "gaussian"
        else PredictivePrior(kind="niw", kappa=1, nu=4, mean=(0, 1), scale=((2, 0), (0, 2)))
    )
    model = SequentialModel(
        metric="outcome",
        law=law,
        control_prior=prior,
        treatment_prior=prior,
        positive_population_control=True,
        positive_population_denominators=law == "gaussian_ratio",
    )
    return SequentialRegistration(
        source_id="experiment",
        definitions_id="immutable-definition-v1",
        control_group="control",
        committed_before_data=True,
        reveal=JointReveal(
            filtration_id="joint-units-v1",
            independent_unit_vectors=True,
            simultaneous_metrics=True,
            outcome_independent_order=True,
            immutable_finalized_outcomes=True,
            longest_window_days=14,
        ),
        models=models or (model,),
        roster=cells or (SequentialCell(metric="outcome", group_id="treatment", alpha=alpha),),
    )


def records(control, treatment, *, offset=0, extra=None, segments=None):
    result = []
    for i, (c, t) in enumerate(zip(control, treatment, strict=True)):
        for arm, value in (("control", c), ("treatment", t)):
            values = {"outcome": value}
            if extra:
                values.update(dict.fromkeys(extra, value))
            result.append(
                {
                    "unit_id": f"{offset + i:08d}-{arm}",
                    "group_id": arm,
                    "values": values,
                    "segments": segments or {},
                }
            )
    return result


def capture(reg, rows, previous=None, *, append=False):
    capture_fn = (
        capture_sequential_snapshot
        if all(model.law in ("bernoulli", "scalar_mean") for model in reg.models)
        else _capture_sequential_diagnostic_snapshot
    )
    return capture_fn(
        reg,
        rows,
        source_id=reg.source_id,
        definitions_id=reg.definitions_id,
        finalized=True,
        previous=previous,
        append=append,
    )


class _DiagnosticComputation:
    def __init__(self, results):
        self.results = tuple(results)
        self.evidence = {}


def estimate_sequential(snapshot, inference):
    """Use typed public inference or retain the private Gaussian diagnostics."""
    if all(model.law in ("bernoulli", "scalar_mean") for model in inference.registration.models):
        return _public_estimate_sequential(snapshot, inference)
    return _DiagnosticComputation(_evaluate_sequential_diagnostic(snapshot, inference))


class UnreadFrame:
    """NativeDataFrame-shaped input that fails on any pre-refusal data access."""

    def __len__(self) -> int:
        raise AssertionError("source length read before proof refusal")

    @property
    def columns(self) -> list[str]:
        raise AssertionError("source columns read before proof refusal")

    def join(self, *args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("source join before proof refusal")

    def drop(self, *args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("source projection before proof refusal")

    def __getattribute__(self, name: str) -> NoReturn:
        raise AssertionError("source accessed before proof refusal")


def raw_gaussian(*, n=500, segments=(), alpha=0.05, family=False, label=None):
    """Build an explicit Gaussian research-diagnostic checkpoint, not evidence."""
    base = registration("gaussian")
    model = SequentialModel.model_validate({**base.models[0].model_dump(), "metric": "rev"})
    cells = tuple(
        SequentialCell(
            metric="rev",
            group_id="treatment",
            segment=segment,
            alpha=Fraction(alpha),
            family=family,
        )
        for segment in (segments or ((),))
    )
    reg = SequentialRegistration.model_validate(
        {
            **base.model_dump(),
            "models": (model,),
            "roster": cells,
        }
    )
    rows = []
    for i, segment in enumerate(segments or ((),)):
        rows.extend(
            records([8, 12] * (n // 2), [9, 13] * (n // 2), offset=i * n, segments=dict(segment))
        )
    for row in rows:
        row["values"] = {"rev": row["values"]["outcome"]}
    snapshot = capture(reg, rows)
    if label is not None:
        snapshot = type(snapshot).model_validate({**snapshot.model_dump(), "reveal_cursor": label})
    metric = MeanMetric(
        name="rev", entity="unit_id", fact="revenue", aggregation="sum", window_days=7
    )
    return snapshot, [metric], AlwaysValid(registration=reg)


def registered_bernoulli(*, n=400, segments=(), alternative="two-sided", alpha=Fraction(1, 20)):
    """Build admitted public Bernoulli evidence for contract-facing tests."""
    base = registration("bernoulli")
    model = SequentialModel.model_validate({**base.models[0].model_dump(), "metric": "revenue"})
    cells = tuple(
        SequentialCell(
            metric="revenue",
            group_id="treatment",
            segment=segment,
            alternative=alternative,
            alpha=alpha,
        )
        for segment in (segments or ((),))
    )
    reg = SequentialRegistration.model_validate(
        {**base.model_dump(), "models": (model,), "roster": cells}
    )
    policy = AlwaysValid(registration=reg)
    rows = []
    for i, segment in enumerate(segments or ((),)):
        rows.extend(
            records(
                [0, 1] * (n // 2),
                [1, 1] * (n // 2),
                offset=i * n,
                segments=dict(segment),
            )
        )
    rows = [{**row, "values": {"revenue": row["values"]["outcome"]}} for row in rows]
    return capture(reg, rows), policy


def registered_spec():
    """Predeclared policy for tests whose operation is refused before source access."""
    return InferenceSpec(kind="always_valid", registration=registration("gaussian"))


def declared_plan(
    metrics,
    *,
    source_id,
    design,
    transformations=(),
    primary=None,
    source_mapping=None,
    public_mean=False,
):
    """Declare the test's unit sampling models from its semantic metric targets.

    ``public_mean=True`` swaps the private Gaussian diagnostic for the
    public ``scalar_mean`` asymptotic law on ``mean`` metrics, so the
    registration actually runs through ``Analysis.run()`` instead of only
    exercising the ``sequential.route.unsupported`` refusal. Only pass it
    when every declared metric is a ``mean`` metric under a fixed
    Randomized mechanism -- mixing a public asymptotic law with a private
    or Bernoulli one in the same family is not supported, and the
    scalar-mean law admits only a fixed randomized assignment mechanism.
    """
    from increment.semantics.models import AnalysisPlan
    from increment.sequential_source import frame_observation_mapping, sequential_definition_id

    models = []
    for metric in metrics:
        if public_mean:
            if metric.type != "mean":
                raise ValueError("public_mean=True only supports mean metrics")
            models.append(
                ScalarMeanModel(
                    metric=metric.name,
                    rho=Fraction(1, 10),
                    start_count=2,
                    assignment="iid_fixed_bernoulli_randomization",
                    treatment_probability=Fraction(1, 2),
                    consistency_and_no_interference=True,
                    segment_membership="pre_assignment",
                    unit_model="iid_stationary_potential_outcomes",
                    moments="finite_2_plus_delta",
                    positive_limiting_variance=True,
                    positive_population_control=True,
                )
            )
            continue
        law = (
            "bernoulli"
            if metric.type in ("conversion", "retention")
            else "gaussian_ratio"
            if metric.type == "ratio"
            else "gaussian"
        )
        base = registration(law).models[0]
        models.append(SequentialModel.model_validate({**base.model_dump(), "metric": metric.name}))
    names = [metric.name for metric in metrics]
    reg = SequentialRegistration.model_validate(
        {
            **registration("gaussian").model_dump(),
            "source_id": source_id,
            "control_group": design.control_group,
            "definitions_id": sequential_definition_id(
                metrics,
                design,
                transformations=transformations,
                source_mapping=source_mapping
                if source_mapping is not None
                else frame_observation_mapping(
                    unit="user_id", group="variant", exposure_date="exposure_date"
                ),
            ),
            "models": models,
            "roster": tuple(
                SequentialCell(metric=name, group_id="treatment", family=name != primary)
                for name in names
            ),
        }
    )
    return AnalysisPlan(
        primary=primary,
        secondaries=[name for name in names if name != primary],
        inference=InferenceSpec(
            kind="asymptotic_mean" if public_mean else "always_valid", registration=reg
        ),
    )


def registered_native(analysis, *, metrics=None, public_mean=False):
    """Bind declared models to a native fixture before requesting raw outcomes."""
    from tests.analysis_factory import _native_source, make_analysis_like

    source = _native_source(analysis)
    context = source.context
    metrics = tuple(metrics if metrics is not None else context.metrics)
    plan = declared_plan(
        metrics,
        source_id=context.study_id,
        design=context.design,
        source_mapping=source._sequential_observation_mapping(),
        public_mean=public_mean,
    )
    return make_analysis_like(analysis, list(metrics), plan=plan)
