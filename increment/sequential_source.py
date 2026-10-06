"""Sequential registration and snapshot access on existing moment sources."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from increment._analysis_config import effective_methods
from increment._window import resolve_window_days
from increment.errors import DefinitionError, RefusalSpec, refuse
from increment.semantics.sequential import (
    BINARY_METRIC_TYPES,
    RATIO_LAWS,
    SCALAR_METRIC_TYPES,
    AsymptoticLaw,
    ScalarMeanModel,
    refuse_segmented_family_compliance,
    sequential_family_size,
    validate_predeclared_segments,
)
from increment.sequential_state import (
    SequentialSnapshot,
    adjustment_kind,
    canonical_id,
    fixed_horizon_alternative,
    registration_id,
    require_public_laws,
    sequential_refuse,
    validate_sequential_methods,
    validate_sequential_transform,
)

if TYPE_CHECKING:
    from increment.semantics.models import MultiplicitySpec
    from increment.semantics.sequential import (
        PredeclaredAdjustment,
        PredictivePrior,
        SequentialCell,
        SequentialCompliancePolicy,
    )
    from increment.sources import SourceContext

BreakoutCorrection = Literal["bh", "bonferroni", "none"]

# Pseudo-observation weight of the declared-baseline Beta prior in automatic exact
# Bernoulli registration: the largest weight that beats a flat prior at the declared
# rate and keeps power if the true rate doubles (calibration/bernoulli_prior.py;
# table in docs/guides/sequential-inference.md). Validity holds at every weight.
DEFAULT_BERNOULLI_PRIOR_WEIGHT = 10


def frame_observation_mapping(
    *,
    unit: str,
    group: str,
    uptake: str | None = None,
    date: str | None = None,
    exposure_date: str | None = None,
) -> dict[str, object]:
    """Bind physical observation roles before adapting a frame."""
    return {
        "unit": unit,
        "group": group,
        "uptake": uptake,
        "date": date,
        "exposure_date": exposure_date,
    }


def native_observation_mapping(definitions, experiment, *, on_mixed_assignment="error"):
    """Bind native observation recipes to a tagged digest, independently of decision policy."""
    import datetime as dt

    from increment.semantics.models import window_days

    def normalize(value):
        if isinstance(value, dt.datetime):
            return value.replace(tzinfo=value.tzinfo or dt.UTC).astimezone(dt.UTC).isoformat()
        if isinstance(value, dt.date):
            return value.isoformat()
        if isinstance(value, Mapping):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [normalize(item) for item in value]
        return value

    raw_mapping = normalize(
        {
            "definitions": definitions.model_dump(
                include={"dialect", "day_boundary", "fact_sources", "dim_sources", "exposures"}
            ),
            "experiment": experiment.model_dump(exclude={"plan"}),
            # Dumps keep the declared spelling only as UTC, so the derived days are bound too.
            "window_days": window_days(experiment),
            "on_mixed_assignment": on_mixed_assignment,
        }
    )
    # The raw recipe holds source SQL, so only its digest may travel with a registration.
    return {"source_mapping_format": 2, "recipe_sha256": canonical_id(raw_mapping)}


def sequential_definition_id(
    metrics, design, *, source_mapping: Mapping[str, object], transformations=()
) -> str:
    """Compute the pre-data metric/window/assignment/population binding.

    Frame MetricSpecs belong in transformations so column bindings and missing
    value policies cannot change while a process continues.
    """
    return canonical_id(
        {
            "metrics": sorted(
                (m.model_dump(mode="json") for m in metrics), key=lambda m: m["name"]
            ),
            "design": design.model_dump(mode="json") if design is not None else None,
            "transformations": sorted(
                (t.model_dump(mode="json") for t in transformations), key=canonical_id
            ),
            "population": "assigned",
            "source_mapping": dict(source_mapping),
        }
    )


def _automatic_law(metric, *, wants_cuped: bool, covariate: bool, adjustment) -> AsymptoticLaw:
    """The asymptotic law a metric's declarations select, refusing what cannot compute.

    A conversion or retention metric is a 0/1 scalar and takes the same laws as
    a mean metric. A CUPED method with a per-unit covariate selects the joint
    law that retains (Y, X) or (N, D, X) per unit; a predeclared coefficient
    keeps the scalar law with the adjusted scalar; a ratio metric selects the
    (N, D) law.
    """
    if metric.type not in (*SCALAR_METRIC_TYPES, "ratio"):
        sequential_refuse(
            "route.unsupported",
            "automatic asymptotic inference requires mean, ratio, conversion or retention metrics",
        )
    ratio = metric.type == "ratio"
    if adjustment is not None:
        if ratio:
            sequential_refuse(
                "route.unsupported",
                f"metric {metric.name!r}: a predeclared coefficient adjusts one scalar; a "
                "ratio metric under sequential inference fits both component coefficients "
                "from retained moments (declare a CUPED method with covariate= instead)",
            )
        return "scalar_mean"
    if not wants_cuped:
        return "ratio_mean" if ratio else "scalar_mean"
    if not covariate:
        sequential_refuse(
            "route.unsupported",
            f"metric {metric.name!r}: CUPED under asymptotic sequential inference retains the "
            "per-unit pre-period covariate alongside the outcome; declare one with "
            "MetricSpec(covariate=...) on a unit-summary frame or n_pre_periods > 0 on a "
            "definitions experiment",
        )
    return "adjusted_ratio_mean" if ratio else "adjusted_mean"


def _declared_cuped(methods) -> bool:
    return any(getattr(method, "variance_reduction", "none") == "cuped" for method in methods)


def _law_inputs(metric, spec, binding, *, pre_period_covariate: bool) -> tuple[bool, bool]:
    """``(wants_cuped, covariate)`` from the frame spec, else the plan binding.

    A frame spec carries its own methods and covariate column; a definitions
    metric declares methods on its plan binding and its covariate through the
    experiment's pre-period.
    """
    if spec is not None:
        methods = (spec.decision_method, *spec.sensitivity_methods)
        return _declared_cuped(methods), spec.covariate is not None
    methods = () if binding is None else (binding.decision_method, *binding.sensitivity_methods)
    return _declared_cuped(methods), pre_period_covariate


def _treatment_arms(design, *, route: str, multi_arm: bool) -> tuple[str, ...]:
    """The declared treatment arms, in canonical order, from the assignment allocation."""
    allocation = getattr(design, "allocation", None)
    if allocation is None:
        sequential_refuse("source.invalid", f"{route} needs declared assignment allocation")
    control = str(design.control_group)
    arms = tuple(sorted(str(arm) for arm in allocation if str(arm) != control))
    if not multi_arm and len(arms) != 1:
        sequential_refuse("route.unsupported", f"{route} requires one treatment arm")
    if not arms:
        sequential_refuse("route.unsupported", f"{route} requires at least one treatment arm")
    return arms


def _breakout_family(
    *,
    exact: bool,
    view_multiplicity: MultiplicitySpec | None,
    mechanism: str | None,
    q: Fraction,
) -> tuple[BreakoutCorrection, Fraction]:
    """The breakout view's ``(correction, q)`` an automatic segmented roster is fixed against.

    Mirrors the compiled breakout view policy: BH at the plan's ``q`` by
    default under a randomized design and uncorrected under encouragement,
    overridden by a declared ``view_multiplicity`` (whose own ``q`` is the
    e-BH level the readout then requires the registration to carry); an
    asymptotic family takes fixed-roster Bonferroni at the plan's ``q`` in
    place of the BH default and refuses an explicit BH, exactly as
    compilation does.
    """
    correction: BreakoutCorrection = "none" if mechanism == "encouragement" else "bh"
    if view_multiplicity is not None:
        correction = view_multiplicity.correction
    if not exact and correction == "bh" and mechanism != "encouragement":
        if view_multiplicity is not None:
            sequential_refuse(
                "route.unsupported", "asymptotic views require explicit Bonferroni, not BH"
            )
        correction = "bonferroni"
    if exact and correction == "bh" and view_multiplicity is not None:
        # A BH view always carries its level (MultiplicitySpec refuses one without).
        assert view_multiplicity.q is not None
        q = Fraction(view_multiplicity.q)
    return correction, q


def _automatic_allocations(
    ceilings: Mapping[str, Fraction],
    *,
    in_family: Mapping[str, bool],
    family_cells: int,
    q: Fraction,
    n_arms: int,
    n_levels: int,
    correction: BreakoutCorrection | None,
    continuous: bool,
) -> dict[str, tuple[Fraction, bool]]:
    """Each metric's per-cell ``(alpha, family)``, fixed from the compiled plan.

    ``ceilings`` are the per-cell levels the compiled procedures allow (a
    primary's level split across the arms; the uptake policy's level split
    across its cells). Whole window (``correction is None``): a family member's
    cells share ``q`` equally, never above the metric's own level, over the
    ``family_cells`` of the roster; every other cell sits at its level.
    Segmented, the family is the breakout view's: under BH every cell is a
    member sharing ``q`` over all ``m`` cells; under Bonferroni an asymptotic
    family keeps fixed per-cell allocations (an equal share of ``q`` across
    metrics, capped at each level, split across the levels) while an exact
    family divides each level across the levels uncorrected; ``none`` leaves
    each cell at its level.
    """
    if correction is None:
        return {
            name: (min(level, q / family_cells), True) if in_family[name] else (level, False)
            for name, level in ceilings.items()
        }
    if correction == "none":
        return {name: (level, False) for name, level in ceilings.items()}
    if correction == "bonferroni":
        if not continuous:
            return {name: (level / n_levels, False) for name, level in ceilings.items()}
        share = q / len(ceilings)
        return {name: (min(level, share) / n_levels, True) for name, level in ceilings.items()}
    m = len(ceilings) * n_arms * n_levels
    return {name: (min(level, q / m), True) for name, level in ceilings.items()}


def _automatic_cells(
    metric: str,
    *,
    arms: Sequence[str],
    allocation: tuple[Fraction, bool],
    alternative,
    null_lift: Fraction,
    segments: Mapping[str, tuple[str, ...]] | None,
    estimand: Literal["itt", "compliance"] = "itt",
) -> list[SequentialCell]:
    """One retained hypothesis per arm, and per predeclared level when segmented."""
    from increment.semantics.sequential import SequentialCell

    alpha, family = allocation
    keys = [
        ((dimension, level),) for dimension, levels in (segments or {}).items() for level in levels
    ] or [()]
    return [
        SequentialCell(
            metric=metric,
            group_id=arm,
            estimand=estimand,
            segment=segment,
            alpha=alpha,
            family=family,
            alternative=alternative,
            null_lift=null_lift,
        )
        for arm in arms
        for segment in keys
    ]


def _segment_family(segments) -> tuple[dict[str, tuple[str, ...]] | None, int]:
    """The admitted predeclared segment family and its level count (1 unsegmented)."""
    if not segments:
        return None, 1
    admitted = validate_predeclared_segments(segments)
    return admitted, sum(len(levels) for levels in admitted.values())


def _compliance_composition(compliance, design, ordered):
    """Whether the design's Bernoulli uptake cell joins the automatic roster."""
    if compliance is not None and getattr(design, "mechanism", None) != "encouragement":
        sequential_refuse("source.invalid", "compliance requires an encouragement design")
    compose_uptake = compliance is not None
    if compose_uptake and any(metric.name == "uptake" for metric in ordered):
        sequential_refuse(
            "source.invalid",
            "an outcome metric named 'uptake' collides with the design uptake cell; rename the "
            "metric or declare an explicit SequentialRegistration",
        )
    return compose_uptake


def _automatic_roster_plan(
    ordered,
    procedures,
    *,
    q: Fraction,
    arms: Sequence[str],
    compliance,
    compose_uptake: bool,
    segments,
    design,
    view_multiplicity,
    continuous: bool,
) -> tuple[dict[str, tuple[Fraction, bool]], dict[str, tuple[str, ...]] | None, Fraction]:
    """Each cell's allocation and family membership, fixed before any outcome is read.

    The whole-window family is the compiled secondary family (plus the uptake
    cell when its policy shares it) at the plan's ``q``; a predeclared segment
    family is judged by the breakout view's policy and level instead, so the
    returned ``q`` is the level the registration carries. Both are one cell
    per arm, so a primary's compiled level is split across the arms and the
    compliance policy's level across its cells, as the request validators
    require.
    """
    admitted, n_levels = _segment_family(segments)
    correction: BreakoutCorrection | None = None
    if admitted is not None:
        correction, q = _breakout_family(
            exact=not continuous,
            view_multiplicity=view_multiplicity,
            mechanism=getattr(design, "mechanism", None),
            q=q,
        )
    ceilings: dict[str, Fraction] = {}
    in_family: dict[str, bool] = {}
    for metric in ordered:
        procedure = procedures[metric.name]
        role_split = len(arms) if procedure.role == "primary" else 1
        ceilings[metric.name] = Fraction(procedure.alpha) / role_split
        in_family[metric.name] = bool(getattr(procedure, "in_family", False))
    if compose_uptake:
        ceilings["uptake"] = Fraction(compliance.alpha) / (len(arms) * n_levels)
        in_family["uptake"] = compliance.family
    family_cells = len(arms) * sequential_family_size(
        sum(1 for metric in ordered if in_family[metric.name]), compliance
    )
    allocations = _automatic_allocations(
        ceilings,
        in_family=in_family,
        family_cells=family_cells,
        q=q,
        n_arms=len(arms),
        n_levels=n_levels,
        correction=correction,
        continuous=continuous,
    )
    return allocations, admitted, q


def registration_carries_segments(registration, segments) -> bool:
    """Whether every declared level of every declared dimension is a retained cell.

    The automatic metadata is cleared from a plan only once this holds: the
    immutable roster, not the declaration, is what continues.
    """
    if not segments:
        return not any(cell.segment for cell in registration.roster)
    retained: dict[str, set[str]] = {}
    for cell in registration.roster:
        if len(cell.segment) != 1:
            return False
        dimension, level = cell.segment[0]
        retained.setdefault(dimension, set()).add(level)
    return retained == {dimension: set(levels) for dimension, levels in segments.items()}


def _automatic_registration(
    *,
    source_id: str,
    metrics,
    design,
    definitions_id: str,
    models,
    roster,
    q: Fraction,
    arms: Sequence[str],
    compliance,
    uptake_prior,
    uptake_allocation,
    segments,
):
    """Close the roster with the uptake cells and the joint reveal, then register."""
    from increment.semantics.sequential import JointReveal, SequentialModel, SequentialRegistration

    if compliance is not None:
        models.append(
            SequentialModel(
                metric="uptake",
                observable="uptake",
                law="bernoulli",
                control_prior=uptake_prior,
                treatment_prior=uptake_prior,
                positive_population_control=True,
            )
        )
        roster.extend(
            _automatic_cells(
                "uptake",
                arms=arms,
                allocation=uptake_allocation,
                alternative=compliance.alternative,
                null_lift=compliance.null_lift,
                segments=segments,
                estimand="compliance",
            )
        )
    uptake_window = (design.uptake.window_days or 0) if compliance is not None else 0
    reveal = JointReveal(
        filtration_id=canonical_id({"binding": definitions_id, "reveal": "joint_units_v1"}),
        independent_unit_vectors=True,
        simultaneous_metrics=True,
        outcome_independent_order=True,
        immutable_finalized_outcomes=True,
        longest_window_days=max(
            uptake_window, max((resolve_window_days(metric) or 0 for metric in metrics), default=0)
        ),
    )
    registration = SequentialRegistration(
        source_id=source_id,
        definitions_id=definitions_id,
        control_group=str(design.control_group),
        committed_before_data=True,
        reveal=reveal,
        models=tuple(models),
        roster=tuple(roster),
        q=q,
    )
    if not registration_carries_segments(registration, segments):
        sequential_refuse(
            "source.invalid", "retained roster does not carry the declared segment family"
        )
    return registration


# Binding metadata stays explicit, matching the existing public registration API.
def auto_register_scalar_mean(  # noqa: PLR0913
    *,
    source_id: str,
    metrics,
    design,
    source_mapping: Mapping[str, object],
    resolved,
    transformations=(),
    bindings: Mapping[str, object] | None = None,
    pre_period_covariate: bool = False,
    expected_decision_sample_size: int = 5000,
    adjustments: Mapping[str, PredeclaredAdjustment] | None = None,
    compliance: SequentialCompliancePolicy | None = None,
    segments: Mapping[str, Sequence[str]] | None = None,
    view_multiplicity: MultiplicitySpec | None = None,
):
    """Build strict asymptotic registration from metadata before source reads.

    ``bindings`` are the plan's per-metric method declarations and
    ``pre_period_covariate`` whether the source derives a per-unit pre-period
    covariate for every metric (a definitions experiment with
    ``n_pre_periods > 0``); a frame declares both on its ``transformations``.
    ``segments`` predeclares one segment dimension and its levels: the roster
    then retains one cell per metric and level under the breakout view's
    fixed-roster Bonferroni family (or uncorrected when ``view_multiplicity``
    says so), every level a monitored cell whether or not it is ever observed.
    Segment membership stays the declaration ``ScalarMeanModel`` carries:
    the column is asserted fixed before assignment, never inferred from data.
    """
    if compliance is not None and compliance.family and segments:
        refuse_segmented_family_compliance(segments, route="automatic scalar registration")
    if getattr(design, "mechanism", None) not in ("randomized", "encouragement"):
        sequential_refuse(
            "route.unsupported",
            "scalar mean requires randomized assignment (randomized or encouragement design; "
            "assignment to encouragement is itself randomized)",
        )
    arms = _treatment_arms(design, route="scalar registration", multi_arm=False)
    allocation = design.allocation
    total = sum((Fraction(value) for value in allocation.values()), Fraction(0))
    treatment_probability = Fraction(allocation[arms[0]]) / total
    ordered = tuple(sorted(metrics, key=lambda m: m.name))
    specs = {spec.name: spec for spec in transformations}
    laws = {}
    for metric in ordered:
        wants_cuped, covariate = _law_inputs(
            metric,
            specs.get(metric.name),
            (bindings or {}).get(metric.name),
            pre_period_covariate=pre_period_covariate,
        )
        laws[metric.name] = _automatic_law(
            metric,
            wants_cuped=wants_cuped,
            covariate=covariate,
            adjustment=(adjustments or {}).get(metric.name),
        )
    unknown = sorted(set(adjustments or ()) - {m.name for m in ordered})
    if unknown:
        sequential_refuse(
            "source.invalid", f"pre-period adjustments name undeclared metrics {unknown}"
        )
    procedures = resolved.procedures
    if not ordered or expected_decision_sample_size < 2:
        sequential_refuse(
            "source.invalid", "scalar registration needs mean metrics and expected N >= 2"
        )
    if any(procedures.get(metric.name) is None for metric in ordered):
        sequential_refuse("source.invalid", "compiled scalar procedure is missing")
    compose_uptake = _compliance_composition(compliance, design, ordered)
    allocations, admitted, q = _automatic_roster_plan(
        ordered,
        procedures,
        q=Fraction(resolved.q),
        arms=arms,
        compliance=compliance,
        compose_uptake=compose_uptake,
        segments=segments,
        design=design,
        view_multiplicity=view_multiplicity,
        continuous=True,
    )
    # Tuning only: inference retains the exact Fraction and certified boundary.
    from increment.estimation.asymptotic_mean import mixture_r_star

    models, roster = [], []
    for metric in ordered:
        procedure = procedures[metric.name]
        alpha, _ = allocations[metric.name]
        rho = Fraction(math.sqrt(float(mixture_r_star(alpha) / expected_decision_sample_size)))
        models.append(
            ScalarMeanModel(
                metric=metric.name,
                law=laws[metric.name],
                rho=rho,
                start_count=2,
                assignment="iid_fixed_bernoulli_randomization",
                treatment_probability=treatment_probability,
                consistency_and_no_interference=True,
                segment_membership="pre_assignment",
                unit_model="iid_stationary_potential_outcomes",
                moments="finite_2_plus_delta",
                positive_limiting_variance=True,
                positive_population_control=True,
                positive_population_denominators=laws[metric.name] in RATIO_LAWS,
                adjustment=(adjustments or {}).get(metric.name),
            )
        )
        roster.extend(
            _automatic_cells(
                metric.name,
                arms=arms,
                allocation=allocations[metric.name],
                alternative=procedure.alternative,
                null_lift=Fraction(procedure.null_lift),
                segments=admitted,
            )
        )
    return _automatic_registration(
        source_id=source_id,
        metrics=metrics,
        design=design,
        definitions_id=sequential_definition_id(
            metrics, design, source_mapping=source_mapping, transformations=transformations
        ),
        models=models,
        roster=roster,
        q=q,
        arms=arms,
        compliance=compliance,
        uptake_prior=bernoulli_prior(None),
        uptake_allocation=allocations.get("uptake"),
        segments=admitted,
    )


_ALWAYS_VALID_METRIC_TYPE = RefusalSpec(
    "definition.inference.always_valid_metric_type",
    DefinitionError,
    template="metric {metric!r} has type {metric_type!r}; InferenceSpec(kind='always_valid') without a registration monitors conversion and retention metrics with the exact Bernoulli e-process. Monitor this metric with InferenceSpec(kind='asymptotic_mean'), or declare an explicit SequentialRegistration",
)


def bernoulli_prior(baseline_rate: Fraction | None) -> PredictivePrior:
    """The Beta prior an automatic exact registration commits before any outcome is read.

    ``Beta(w * p0, w * (1 - p0))`` places ``w`` pseudo-observations at the
    declared control rate; without a declared rate the prior is flat. The
    e-process is valid under every proper prior, so the weight moves power only.
    """
    from increment.semantics.sequential import PredictivePrior

    if baseline_rate is None:
        return PredictivePrior(kind="beta", a=Fraction(1), b=Fraction(1))
    weight = Fraction(DEFAULT_BERNOULLI_PRIOR_WEIGHT)
    return PredictivePrior(kind="beta", a=weight * baseline_rate, b=weight * (1 - baseline_rate))


def auto_register_bernoulli(
    *,
    source_id: str,
    metrics,
    design,
    source_mapping: Mapping[str, object],
    resolved,
    transformations=(),
    bindings: Mapping[str, object] | None = None,
    baseline_rate: Fraction | None = None,
    compliance: SequentialCompliancePolicy | None = None,
    segments: Mapping[str, Sequence[str]] | None = None,
    view_multiplicity: MultiplicitySpec | None = None,
):
    """Build the exact Bernoulli registration from metadata before source reads.

    Every metric must be a conversion or retention metric; each is modelled by
    the Bernoulli law with the same committed Beta prior on both arms. The
    roster retains one cell per metric and declared treatment arm: a primary's
    compiled level is split across the arms, family members are allocated
    ``min(alpha, q / m)`` over the ``m`` family cells exactly as the scalar
    route does, and the design's uptake cell (when a compliance policy is
    declared) takes the policy's level split across its cells. ``segments``
    predeclares one segment dimension and its levels; the roster then retains
    one cell per metric, arm and level under the breakout view's family
    (e-BH by default, or the ``view_multiplicity`` declared), every level a
    monitored cell whether or not it is ever observed.
    """
    if compliance is not None and compliance.family and segments:
        refuse_segmented_family_compliance(segments, route="automatic Bernoulli registration")
    from increment.semantics.sequential import SequentialModel

    if getattr(design, "mechanism", None) not in ("randomized", "encouragement"):
        sequential_refuse(
            "route.unsupported",
            "automatic Bernoulli registration requires randomized assignment (randomized or "
            "encouragement design; assignment to encouragement is itself randomized)",
        )
    arms = _treatment_arms(design, route="automatic Bernoulli registration", multi_arm=True)
    ordered = tuple(sorted(metrics, key=lambda m: m.name))
    if not ordered and compliance is None:
        sequential_refuse(
            "source.invalid",
            "automatic Bernoulli registration needs conversion or retention metrics",
        )
    specs = {spec.name: spec for spec in transformations}
    for metric in ordered:
        if metric.type not in BINARY_METRIC_TYPES:
            refuse(_ALWAYS_VALID_METRIC_TYPE, metric=metric.name, metric_type=metric.type)
        validate_sequential_transform(metric)
        wants_cuped, _ = _law_inputs(
            metric,
            specs.get(metric.name),
            (bindings or {}).get(metric.name),
            pre_period_covariate=False,
        )
        if wants_cuped:
            sequential_refuse(
                "transform.unpredictable",
                f"metric {metric.name!r}: the exact Bernoulli e-process admits no in-experiment "
                "CUPED coefficient; monitor under InferenceSpec(kind='asymptotic_mean') or "
                "drop the CUPED method",
            )
    procedures = resolved.procedures
    compose_uptake = _compliance_composition(compliance, design, ordered)
    if any(procedures.get(metric.name) is None for metric in ordered):
        sequential_refuse("source.invalid", "compiled Bernoulli procedure is missing")
    allocations, admitted, q = _automatic_roster_plan(
        ordered,
        procedures,
        q=Fraction(resolved.q),
        arms=arms,
        compliance=compliance,
        compose_uptake=compose_uptake,
        segments=segments,
        design=design,
        view_multiplicity=view_multiplicity,
        continuous=False,
    )
    prior = bernoulli_prior(baseline_rate)
    models, roster = [], []
    for metric in ordered:
        procedure = procedures[metric.name]
        models.append(
            SequentialModel(
                metric=metric.name,
                law="bernoulli",
                control_prior=prior,
                treatment_prior=prior,
                positive_population_control=True,
            )
        )
        roster.extend(
            _automatic_cells(
                metric.name,
                arms=arms,
                allocation=allocations[metric.name],
                alternative=procedure.alternative,
                null_lift=Fraction(procedure.null_lift),
                segments=admitted,
            )
        )
    return _automatic_registration(
        source_id=source_id,
        metrics=metrics,
        design=design,
        definitions_id=sequential_definition_id(
            metrics, design, source_mapping=source_mapping, transformations=transformations
        ),
        models=models,
        roster=roster,
        q=q,
        arms=arms,
        compliance=compliance,
        uptake_prior=prior if not ordered else bernoulli_prior(None),
        uptake_allocation=allocations.get("uptake"),
        segments=admitted,
    )


def validate_compliance_policy(plan, design, *, required=False):
    """Require the declared uptake estimand and its compiled testing allocation."""
    registration = getattr(plan.inference, "registration", None)
    if registration is None:
        return
    needed = required or any(m.observable == "uptake" for m in registration.models)
    if needed and getattr(design, "mechanism", None) != "encouragement":
        sequential_refuse("source.invalid", "compliance requires an encouragement design")
    validate_compliance_allocation(
        registration, plan.compliance, plan.alpha, plan.q, required=required
    )


def validate_compliance_allocation(registration, policy, alpha, q, *, required=False):
    """Shared declaration, compiled-plan and portable-policy invariant."""
    from fractions import Fraction

    models = [m for m in registration.models if m.observable == "uptake" and m.law == "bernoulli"]
    cells = [c for c in registration.roster if c.estimand == "compliance"]
    if not models and not cells and not required and policy is None:
        return
    if (
        len(models) != 1
        or not cells
        or policy is None
        or {c.metric for c in cells} != {models[0].metric}
    ):
        sequential_refuse(
            "source.invalid",
            "compliance requires a Bernoulli uptake model, retained cells and an explicit compiled compliance policy",
        )
    if policy.alpha > Fraction(alpha):
        sequential_refuse("source.invalid", "compliance alpha exceeds the compiled plan allocation")
    if policy.family and (float(registration.q) != q or registration.q > Fraction(q)):
        sequential_refuse(
            "source.invalid", "registered compliance family q differs from the compiled plan"
        )
    allocation = policy.alpha / len(cells)
    if any(
        c.alpha > allocation
        or c.alternative != policy.alternative
        or c.null_lift != policy.null_lift
        or c.family != policy.family
        for c in cells
    ):
        sequential_refuse(
            "source.invalid",
            "registered compliance cells differ from their compiled policy allocation",
        )


def validate_scalar_mean_design(registration, design):
    """Check observable design restrictions; distributional assumptions are declarations."""
    from fractions import Fraction

    models = [m for m in registration.models if isinstance(m, ScalarMeanModel)]
    if not models:
        return
    if getattr(design, "mechanism", None) not in ("randomized", "encouragement"):
        sequential_refuse(
            "route.unsupported",
            "scalar mean requires fixed randomized parallel assignment (randomized "
            "or encouragement design; assignment to encouragement is itself randomized)",
        )
    allocation = getattr(design, "allocation", None)
    if allocation is not None:
        treatment = registration.roster[0].group_id
        if set(allocation) != {registration.control_group, treatment}:
            sequential_refuse(
                "source.invalid", "scalar mean allocation and registered arms disagree"
            )
        total = sum((Fraction(v) for v in allocation.values()), Fraction(0))
        probability = Fraction(allocation[treatment]) / total
        if any(float(m.treatment_probability) != float(probability) for m in models):
            sequential_refuse(
                "source.invalid", "declared treatment probability differs from assignment"
            )


def validate_sequential_plan(plan, metrics, design):
    """Check the complete registered decision before any producer reads data."""
    from fractions import Fraction

    registration = getattr(plan.inference, "registration", None)
    if registration is None:
        return
    validate_scalar_mean_design(registration, design)
    if any(isinstance(m, ScalarMeanModel) for m in registration.models):
        if float(registration.q) != plan.q or registration.q > Fraction(plan.q):
            sequential_refuse(
                "source.invalid", "asymptotic family budget differs from the compiled plan"
            )
    require_public_laws(registration.models, "sequential plans")
    validate_compliance_policy(plan, design)
    if design is None or getattr(design, "mechanism", None) == "observational":
        sequential_refuse(
            "route.unsupported",
            "sequential likelihoods require a declared randomized or encouragement design"
            + (
                ""
                if design is None
                else "; "
                + fixed_horizon_alternative(metrics, "an observational design", "observational")
            ),
        )
    if str(design.control_group) != registration.control_group:
        sequential_refuse(
            "source.invalid", "registered control differs from the declared assignment"
        )
    catalog = {metric.name: metric for metric in metrics}
    for model in registration.models:
        if model.observable == "uptake":
            if getattr(design, "one_sided", False):
                sequential_refuse(
                    "route.unsupported",
                    "structural-zero control uptake is a rate target, not a positive-control relative likelihood; use fixed-horizon compliance (valid for one planned analysis, not repeated looks)",
                )
            if getattr(design, "mechanism", None) != "encouragement":
                sequential_refuse(
                    "source.invalid", "uptake registration requires an encouragement design"
                )
            continue
        metric = catalog.get(model.metric)
        if metric is None:
            sequential_refuse("source.invalid", "registered outcome has no declared metric")
        if model.law in ("scalar_mean", "adjusted_mean") and metric.type not in SCALAR_METRIC_TYPES:
            sequential_refuse(
                "route.unsupported",
                "scalar mean inference requires a mean, conversion or retention metric",
            )
        validate_sequential_transform(metric)
        if (metric.type == "ratio") != (model.law in ("gaussian_ratio", *RATIO_LAWS)):
            sequential_refuse("source.invalid", "ratio target and joint sampling model disagree")
        if metric.type in BINARY_METRIC_TYPES and model.law not in (
            "bernoulli",
            "scalar_mean",
            "adjusted_mean",
        ):
            sequential_refuse(
                "source.invalid", "binary outcomes require a Bernoulli or scalar mean declaration"
            )
        procedure = plan.procedures[model.metric]
        validate_sequential_methods(
            registration,
            model.metric,
            (procedure.decision_method, *procedure.sensitivity_methods),
            prior=procedure.prior,
        )
        for cell in registration.roster:
            if cell.metric != model.metric:
                continue
            if (
                cell.alternative != procedure.alternative
                or float(cell.null_lift) != procedure.null_lift
                or getattr(procedure, "null_abs", None) is not None
            ):
                sequential_refuse(
                    "source.invalid",
                    "registered null or direction differs from the compiled decision",
                )
            arm_count = len({c.group_id for c in registration.roster if c.metric == cell.metric})
            allowance = Fraction(procedure.alpha) / (
                arm_count if procedure.role == "primary" else 1
            )
            if cell.alpha > allowance:
                sequential_refuse(
                    "source.invalid",
                    f"registered cell alpha exceeds its compiled allocation -- "
                    f"{cell.metric!r} holds the {procedure.role} role, so register it at "
                    f"alpha <= {float(allowance)!r}",
                )


def validate_breakout_registration(
    registration,
    *,
    control_group,
    dimension,
    correction,
    q,
    alpha_by_metric,
):
    """Check the retained segment family before any likelihood is evaluated."""
    from fractions import Fraction

    continuous = any(isinstance(m, ScalarMeanModel) for m in registration.models)
    if continuous and correction == "bh":
        sequential_refuse("route.unsupported", "asymptotic breakout uses predeclared Bonferroni")
    family_correction = "bonferroni" if continuous else "bh"
    if registration.control_group != control_group or any(
        len(c.segment) != 1
        or c.segment[0][0] != dimension
        or c.family != (correction == family_correction)
        for c in registration.roster
    ):
        sequential_refuse(
            "source.invalid", "breakout request differs from its retained registration"
        )
    if (
        q is not None
        and correction in ("bh", "bonferroni")
        and (float(registration.q) != q or registration.q > Fraction(q))
    ):
        sequential_refuse("source.invalid", "breakout family level differs from registration")
    segments = {c.segment for c in registration.roster}
    divisor = len(segments) if correction == "bonferroni" else 1
    if any(c.alpha > Fraction(alpha_by_metric[c.metric]) / divisor for c in registration.roster):
        sequential_refuse(
            "source.invalid", "registered segment alpha exceeds the requested allocation"
        )


def _validate_request_breakout(request, registration):
    from fractions import Fraction

    allocations = {}
    for model in registration.models:
        arms = {c.group_id for c in registration.roster if c.metric == model.metric}
        if model.observable == "uptake":
            allocations[model.metric] = request.plan.compliance.alpha / len(arms)
        else:
            procedure = request.plan.procedures[model.metric]
            allocations[model.metric] = Fraction(procedure.alpha) / (
                len(arms) if procedure.role == "primary" else 1
            )
    validate_breakout_registration(
        registration,
        control_group=str(request.design.control_group),
        dimension=request.dimension,
        correction=request.correction,
        q=request.q,
        alpha_by_metric=allocations,
    )


def validate_sequential_request(request):
    from fractions import Fraction

    from increment.estimation.sequential import ASYMPTOTIC_PROCEDURE_POLICIES, SEQUENTIAL_POLICIES

    inference = request.plan.inference
    if not isinstance(inference, SEQUENTIAL_POLICIES) or request.view == "daily":
        return
    registration = inference.registration
    validate_scalar_mean_design(registration, request.design)
    validate_compliance_policy(
        request.plan,
        request.design,
        required=request.estimands is not None and "compliance" in request.estimands,
    )
    if registration.source_id != request.context.study_id:
        sequential_refuse(
            "source.invalid", "registered source identity differs from requested source"
        )
    if request.cluster is not None or getattr(request.design, "mechanism", None) == "observational":
        sequential_refuse(
            "route.unsupported",
            "clustered and observational sequential inference are unsupported; "
            + fixed_horizon_alternative(request.metrics, "both", "clustered or observational"),
        )
    if registration.control_group != str(request.design.control_group):
        sequential_refuse("source.invalid", "registered assignment control differs from design")
    if request.population != "assigned":
        sequential_refuse(
            "route.unsupported",
            "trigger-selected units require a separate conditional sampling law; use raw assigned ITT",
        )
    if request.value_scale:
        sequential_refuse("route.unsupported", "registered likelihoods target relative effects")
    if request.estimands is None and getattr(request.design, "mechanism", None) == "encouragement":
        sequential_refuse(
            "route.unsupported",
            "declare estimands=('itt',) or ('compliance',); binary-uptake LATE has no matching likelihood proof",
        )
    if request.estimands is not None and any(
        e not in ("itt", "compliance") for e in request.estimands
    ):
        sequential_refuse(
            "route.unsupported", "binary-uptake LATE is not a Gaussian joint observation model"
        )
    models = {m.metric: m for m in registration.models}
    if request.view == "breakout":
        _validate_request_breakout(request, registration)
    for metric, config in zip(request.metrics, request.configs, strict=True):
        if request.estimands is not None and set(request.estimands) == {"compliance"}:
            continue
        validate_sequential_methods(
            registration,
            metric.name,
            effective_methods(config, design=request.design),
            prior=config.prior,
        )
        validate_sequential_transform(metric)
        if metric.name not in models:
            sequential_refuse("source.invalid", "metric has no pre-data sampling-law declaration")
        model = models[metric.name]
        if (metric.type == "ratio") != (model.law in ("gaussian_ratio", *RATIO_LAWS)):
            sequential_refuse("source.invalid", "ratio target and joint sampling model disagree")
        if metric.type in BINARY_METRIC_TYPES and model.law not in (
            "bernoulli",
            "scalar_mean",
            "adjusted_mean",
        ):
            sequential_refuse(
                "source.invalid", "binary outcome needs a Bernoulli or scalar mean model"
            )
        procedure = request.plan.procedures[metric.name]
        for cell in registration.roster:
            if cell.metric != metric.name:
                continue
            if cell.alternative != procedure.alternative or float(cell.null_lift) != getattr(
                procedure, "null_lift", 0
            ):
                sequential_refuse(
                    "source.invalid", "registered null or direction differs from compiled decision"
                )
            if getattr(procedure, "null_abs", None) is not None:
                sequential_refuse(
                    "route.unsupported",
                    "this likelihood registration targets relative effects, not additive margins",
                )
            arms = {
                c.group_id
                for c in registration.roster
                if c.metric == cell.metric and c.estimand == cell.estimand
            }
            allocated = Fraction(procedure.alpha)
            if procedure.role == "primary":
                allocated /= len(arms)
            if cell.alpha > allocated:
                sequential_refuse(
                    "source.invalid",
                    "registered cell alpha exceeds the compiled metric/arm allocation",
                )
            expected_family = (
                bool(getattr(procedure.family, "member", False))
                if request.view != "breakout"
                else request.correction
                == ("bonferroni" if isinstance(inference, ASYMPTOTIC_PROCEDURE_POLICIES) else "bh")
            )
            if cell.family != expected_family:
                sequential_refuse(
                    "source.invalid",
                    "registered family membership differs from the compiled view policy",
                )


def validate_frame_registration(
    registration, *, metrics, specs, design, configs, source_id, source_mapping, cluster=None
):
    """Validate the observation mapping before a dataframe adapter reads it."""
    if registration is None:
        return
    validate_scalar_mean_design(registration, design)
    require_public_laws(registration.models, "frame registration")
    expected = sequential_definition_id(
        metrics, design, transformations=specs, source_mapping=source_mapping
    )
    if registration.source_id != source_id or registration.definitions_id != expected:
        sequential_refuse(
            "source.invalid", "registered source/definitions differ before frame access"
        )
    if cluster is not None or getattr(design, "mechanism", None) == "observational":
        sequential_refuse(
            "route.unsupported",
            "clustered and observational sequential routes are unsupported; "
            + fixed_horizon_alternative(metrics, "both", "clustered or observational"),
        )
    if registration.control_group != str(design.control_group):
        sequential_refuse("source.invalid", "registered assignment control differs from design")
    modeled = {model.metric for model in registration.models if model.observable == "outcome"}
    for config in configs:
        if config.metric.name not in modeled:
            continue
        validate_sequential_methods(
            registration,
            config.metric.name,
            effective_methods(config, design=design),
            prior=config.prior,
        )
    declared = {s.name: s for s in specs}
    for model in registration.models:
        if model.observable == "uptake":
            if getattr(design, "one_sided", False):
                sequential_refuse(
                    "route.unsupported",
                    "structural-zero control uptake is a rate target, not a positive-control relative likelihood; use fixed-horizon compliance (valid for one planned analysis, not repeated looks)",
                )
            if getattr(design, "mechanism", None) != "encouragement":
                sequential_refuse("source.invalid", "uptake requires an encouragement design")
            continue
        spec = declared.get(model.metric)
        if spec is None:
            sequential_refuse(
                "source.invalid", "registered metric has no frame observation mapping"
            )
        if model.law in ("scalar_mean", "adjusted_mean") and spec.type not in SCALAR_METRIC_TYPES:
            sequential_refuse(
                "route.unsupported",
                "scalar mean inference requires a mean, conversion or retention metric",
            )
        if spec.missing == "drop":
            sequential_refuse(
                "route.unsupported",
                "outcome-dependent missing-row deletion breaks joint reveal; use error or a declared zero outcome",
            )
        validate_sequential_transform(spec)
        if adjustment_kind(registration, model.metric) is not None and spec.covariate is None:
            sequential_refuse(
                "source.invalid",
                f"metric {model.metric!r}: the registered covariate adjustment needs the "
                "frame's covariate column (MetricSpec.covariate)",
            )
        if (spec.type == "ratio") != (model.law in ("gaussian_ratio", *RATIO_LAWS)):
            sequential_refuse(
                "source.invalid",
                "ratio declaration needs genuine numerator/denominator observations",
            )
        if spec.type in BINARY_METRIC_TYPES and model.law not in (
            "bernoulli",
            "scalar_mean",
            "adjusted_mean",
        ):
            sequential_refuse(
                "source.invalid", "binary outcomes require Bernoulli or scalar mean registration"
            )


def validate_source_mapping(context, source_mapping):
    """Bind a native/artifact construction before resolving any relation."""
    registration = getattr(context.plan.inference, "registration", None)
    if registration is None:
        return
    expected = sequential_definition_id(
        context.metrics, context.design, source_mapping=source_mapping
    )
    if registration.source_id != context.study_id or registration.definitions_id != expected:
        sequential_refuse(
            "source.invalid", "registered source/definitions differ before relation access"
        )
    validate_sequential_plan(context.plan, context.metrics, context.design)


def link_snapshot(snapshot, previous):
    """Prove extension from retained immutable unit proofs, never a generation hash.

    Every freeze either side records survives: ``previous``'s freezes carry
    forward, and a freeze ``snapshot`` declared after ``previous`` is kept.
    """
    if previous is None:
        return snapshot
    if snapshot.registration_id != previous.registration_id:
        sequential_refuse("continuation.rewrite", "registration changed")
    if snapshot.prefix_id == previous.prefix_id:
        return _with_cursor(previous, snapshot.reveal_cursor)
    from increment.sequential_state import (
        SequentialAncestor,
        _PrefixContent,
        declare_sequential_freeze_cells,
    )

    parent = SequentialAncestor(
        prefix_id=previous.prefix_id,
        n_records=len(previous.records),
        states=previous.states,
        frozen=previous.frozen,
    )
    chain = [a.prefix_id for a in snapshot.ancestors]
    if previous.prefix_id in chain:
        # The snapshot's own history already passes through previous's content:
        # keep everything it recorded after that point, freezes included.
        after = snapshot.ancestors[chain.index(previous.prefix_id) + 1 :]
        linked = SequentialSnapshot.model_validate(
            {**snapshot.model_dump(), "ancestors": (*previous.ancestors, parent, *after)}
        )
        linked.verify_parent(previous)
        return linked
    if (
        len(snapshot.records) < len(previous.records)
        or snapshot.records[: len(previous.records)] != previous.records
        or (len(snapshot.records) == len(previous.records) and snapshot.states != previous.states)
    ):
        sequential_refuse(
            "continuation.rewrite", "prior finalized records were removed, reordered or corrected"
        )
    carried = {c.cell: c for c in previous.frozen}
    if any(carried.get(c.cell, c) != c for c in snapshot.frozen):
        sequential_refuse(
            "freeze.dropped", "the snapshot records a different freeze than its proof parent"
        )
    declared = [c for c in snapshot.frozen if c.cell not in carried]
    if any(c.revealed_units != len(snapshot.records) for c in declared):
        sequential_refuse(
            "continuation.rewrite",
            "the snapshot froze a cell at a look its proof parent does not contain; "
            "prove continuation from a parent captured at or after that look",
        )
    if len(snapshot.records) == len(previous.records):
        linked = _with_cursor(previous, snapshot.reveal_cursor)
    else:
        content = _PrefixContent(
            snapshot.version,
            snapshot.registration_id,
            snapshot.records,
            (a.n_records for a in previous.ancestors),
        )
        linked = SequentialSnapshot.model_validate(
            {
                **snapshot.model_dump(),
                "parent_id": previous.prefix_id,
                "ancestors": (*previous.ancestors, parent),
                "frozen": previous.frozen,
                "prefix_id": content.digest(
                    len(snapshot.records), snapshot.states, previous.frozen
                ),
            }
        )
        linked.verify_parent(previous)
    if declared:
        # Re-declared at this same look, the checkpoints differ only in the
        # pre-freeze prefix they cite, which is now on the proven chain.
        linked = declare_sequential_freeze_cells(linked, [c.cell for c in declared])
        linked.verify_parent(previous)
    return linked


def _with_cursor(snapshot, reveal_cursor):
    if reveal_cursor is None:
        return snapshot
    return snapshot.model_copy(update={"reveal_cursor": reveal_cursor})


@runtime_checkable
class SequentialSnapshotSource(Protocol):
    def sequential_snapshot(
        self, *, previous: SequentialSnapshot | None = None
    ) -> SequentialSnapshot: ...


def source_snapshot(
    source: object, *, previous: SequentialSnapshot | None = None
) -> SequentialSnapshot:
    if not isinstance(source, SequentialSnapshotSource):
        sequential_refuse(
            "continuation.legacy",
            "this source has no exact finalized checkpoint; use a raw-record, frame, native or current artifact capture",
        )
    return source.sequential_snapshot(previous=previous)


class SequentialSourceMixin:
    """Existing source adapters retain immutable snapshots independently of moments."""

    _context: SourceContext
    _sequential_snapshot: SequentialSnapshot | None = None

    @property
    def context(self) -> SourceContext:
        return self._context

    def sequential_snapshot(
        self, *, previous: SequentialSnapshot | None = None
    ) -> SequentialSnapshot:
        snapshot = getattr(self, "_sequential_snapshot", None)
        if snapshot is None:
            sequential_refuse(
                "continuation.legacy",
                "source has no exact finalized joint-unit checkpoint; capture or adopt a current sequential snapshot",
            )
        registration = getattr(self.context.plan.inference, "registration", None)
        if registration is not None:
            require_public_laws(registration.models, "sequential source access")
        linked = link_snapshot(snapshot, previous)
        self._sequential_snapshot = linked
        return linked

    def adopt_sequential_snapshot(self, snapshot: SequentialSnapshot) -> SequentialSnapshot:
        """Adopt a trusted producer checkpoint with the same construction context.

        The producer must already have captured actual finalized source records.
        This method does not turn ordinary rounded moment exports into evidence.
        """
        inference = self.context.plan.inference
        registration = getattr(inference, "registration", None)
        if registration is None or snapshot.registration_id != registration_id(registration):
            sequential_refuse(
                "source.invalid", "adopted checkpoint has a different registered source plan"
            )
        require_public_laws(registration.models, "sequential snapshot adoption")
        if snapshot.registration.source_id != self.context.study_id:
            sequential_refuse("source.invalid", "adopted checkpoint belongs to another source")
        previous = getattr(self, "_sequential_snapshot", None)
        linked = link_snapshot(snapshot, previous)
        self._sequential_snapshot = linked
        return linked


def adopt_source_snapshot(source: object, snapshot: SequentialSnapshot) -> SequentialSnapshot:
    """Retain ``snapshot`` on ``source``, proving it continues the prefix held there."""
    if not isinstance(source, SequentialSourceMixin):
        sequential_refuse(
            "continuation.legacy",
            "this source has no exact finalized checkpoint; use a raw-record, frame, native or current artifact capture",
        )
    return source.adopt_sequential_snapshot(snapshot)
