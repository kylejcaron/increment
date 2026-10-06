"""Compatibility cells, scenario outcomes, and evidence for the public reference.

Scenarios evaluate real requests; declared outcomes and evidence are validated.
"""

from __future__ import annotations

import ast
import importlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

from increment._compatibility_inspector import (
    CompatibilityCheck,
    CompatibilityReport,
    FamilyStatus,
    RuntimeStatus,
    contextual_decision_observe,
    default_support_observe,
    inspect_compatibility,
)
from increment._plan_compatibility import (
    PlanFamilyCompatibilityRequest,
    plan_family_compatibility,
)
from increment.errors import CapabilityError, UnsupportedRequestError
from increment.estimation.arm_contract import (
    ARM_EVIDENCE_CONTRACT,
    AnalysisAxes,
    ArmCompatibilityRequest,
    FamilyPolicy,
    MethodCapability,
    MetricCapabilities,
    RelativeDecisionPolicy,
)
from increment.estimation.sequential import AlwaysValid
from increment.semantics.assignment import ParallelAssignment
from tests.sequential_cases import registration

ROOT = Path(__file__).resolve().parents[1]

# Axes

CAPABILITIES = [
    "estimate",
    "cuped",
    "cluster",
    "sequential",
    "observational",
    "encouragement",
    "cate",
    "breakout",
    "daily_asof",
    "report_calendar",
    "report_window",
    "from_moments",
    "export",
    "sitewide",
    "rollout",
]


# Cell declarations


@dataclass(frozen=True)
class Cell:
    status: Literal["supported", "refused", "na", "silent"]
    fragment: str | None = None
    raises: type[Exception] | None = None
    warns: str | None = None
    reason: str | None = None
    note: str | None = None
    advisory: str | None = None
    code: str | None = None

    @property
    def runtime_status(self) -> str:
        # Advisory prose makes a supported cell limited rather than unconditional.
        if self.status == "supported":
            return "limited" if self.advisory else "supported"
        return {
            "refused": "refused",
            "na": "not_applicable",
            "silent": "limited",
        }[self.status]

    @property
    def refusal_code_display(self) -> str | None:
        """Return the refusal code or explicit unavailable marker; None if not refused."""
        if self.status != "refused":
            return None
        return self.code if self.code is not None else "unavailable"


def S(note: str | None = None, warns: str | None = None, advisory: str | None = None) -> Cell:
    return Cell("supported", note=note, warns=warns, advisory=advisory)


def R(
    fragment: str,
    raises: type[Exception],
    warns: str | None = None,
    advisory: str | None = None,
    code: str | None = None,
) -> Cell:
    return Cell(
        "refused", fragment=fragment, raises=raises, warns=warns, advisory=advisory, code=code
    )


def NA(reason: str) -> Cell:
    return Cell("na", reason=reason)


def SILENT(note: str) -> Cell:
    return Cell("silent", note=note)


_NA_REPORT_LAYER = NA("report-layer type: refused by MetricSpec and by experiment validation")
_SEAM = "type='retention' is not supported on from_unit_summary"


# The declared truth, one cell per (capability x metric type). Fragments are
# short on purpose: refusals may be reworded, but must not disappear.
MATRIX: dict[str, dict[str, Cell]] = {
    "estimate": {
        "mean": S(),
        "conversion": S(),
        "ratio": S(),
        "retention": S(note="native path; the frame summary seam refuses (pinned separately)"),
        "quantile": S(note="served through unit_frame, not moments"),
        "total": _NA_REPORT_LAYER,
        "active": _NA_REPORT_LAYER,
    },
    "cuped": {
        "mean": S(),
        "conversion": S(),
        "ratio": S(note="numerator and denominator each adjusted; induced covariance retained"),
        "retention": S(note="native path via Experiment.n_pre_periods"),
        "quantile": R(
            "CUPED does not apply to quantile",
            ValueError,
            code="frame.metric.cuped_does_apply",
        ),
        "total": _NA_REPORT_LAYER,
        "active": _NA_REPORT_LAYER,
    },
    "cluster": {
        "mean": S(note=">=40 clusters keeps the small-K advisory out of this cell"),
        "conversion": S(),
        "ratio": S(note="den family carries the metric's own per-cluster denominator"),
        "retention": S(note="native path; declared Experiment.cluster"),
        "quantile": R(
            "not supported for quantile metrics",
            CapabilityError,
            code="source.frame.cluster_capability",
        ),
        "total": _NA_REPORT_LAYER,
        "active": _NA_REPORT_LAYER,
    },
    "sequential": {
        "mean": R(
            "public sequential plans admits Bernoulli or registered asymptotic mean, adjusted "
            "mean and ratio observations",
            CapabilityError,
            code="sequential.route.unsupported",
            advisory=(
                "This probe uses a Gaussian likelihood registration. Use "
                "InferenceSpec(kind='asymptotic_mean') for qualified asymptotic "
                "continuous-mean sequential inference."
            ),
        ),
        "conversion": S(note="registered raw Bernoulli likelihood"),
        "ratio": R(
            "public sequential plans admits Bernoulli or registered asymptotic mean, adjusted "
            "mean and ratio observations",
            CapabilityError,
            code="sequential.route.unsupported",
            advisory=(
                "This probe uses the exact Gaussian ratio likelihood, which remains a "
                "private diagnostic. Use InferenceSpec(kind='asymptotic_mean') for the "
                "qualified asymptotic ratio_mean law (see the sequential x cuped pair)."
            ),
        ),
        "retention": S(note="native finalized bounded retention with Bernoulli likelihood"),
        "quantile": R(
            "quantiles need a matching sequential sampling proof",
            CapabilityError,
            code="sequential.route.unsupported",
        ),
        "total": _NA_REPORT_LAYER,
        "active": _NA_REPORT_LAYER,
    },
    "observational": {
        "mean": S(note="iptw, dml and aipw all probed"),
        "conversion": S(),
        "ratio": R(
            "is not supported for ratio metric",
            UnsupportedRequestError,
            code="estimation.adjust_common.supported_ratio_metric",
        ),
        "retention": R(_SEAM, ValueError),
        "quantile": R(
            "has no observational estimator",
            UnsupportedRequestError,
            code="readout.observational.quantile",
        ),
        "total": _NA_REPORT_LAYER,
        "active": _NA_REPORT_LAYER,
    },
    "encouragement": {
        "mean": S(),
        "conversion": S(),
        "ratio": R(
            "LATE is not supported for ratio metric",
            NotImplementedError,
            code="estimation.encouragement.late.ratio",
        ),
        "retention": R(_SEAM, ValueError),
        "quantile": R("have no moments representation", CapabilityError, code="breakout.quantile"),
        "total": _NA_REPORT_LAYER,
        "active": _NA_REPORT_LAYER,
    },
    "cate": {
        "mean": S(),
        "conversion": S(),
        "ratio": R(
            "estimate_cate does not support ratio metric",
            NotImplementedError,
            code="cate.does_support_ratio",
        ),
        "retention": R(_SEAM, ValueError),
        "quantile": R(
            "CATE is not supported for quantile metric",
            NotImplementedError,
            code="cate.cate_supported_quantile",
        ),
        "total": _NA_REPORT_LAYER,
        "active": _NA_REPORT_LAYER,
    },
    "breakout": {
        "mean": S(note="native run_breakout; frame sources refuse by substrate"),
        "conversion": S(),
        "ratio": S(),
        "retention": S(note="native path; bounded band"),
        "quantile": R(
            "quantiles do not decompose over segment moments",
            CapabilityError,
            code="readout.metric.quantile_breakout",
        ),
        "total": _NA_REPORT_LAYER,
        "active": _NA_REPORT_LAYER,
    },
    "daily_asof": {
        "mean": S(note="panel daily grain"),
        "conversion": S(),
        "ratio": S(),
        "retention": S(note="native as-of view; the independent-per-day view refuses by design"),
        "quantile": R(
            "quantiles do not decompose into per-day moments",
            CapabilityError,
            code="readout.metric.quantile_grain",
        ),
        "total": _NA_REPORT_LAYER,
        "active": _NA_REPORT_LAYER,
    },
    "report_calendar": {
        "mean": S(),
        "conversion": S(),
        "ratio": S(note="point-only: value is served, the interval is NULL"),
        "retention": R(
            "retention needs a cohort anchor", CapabilityError, code="report.metric.unsupported"
        ),
        "quantile": R(
            "no moments representation", CapabilityError, code="report.metric.unsupported"
        ),
        "total": S(note="value only; n and the interval are NULL by design"),
        "active": S(note="value only; n and the interval are NULL by design"),
    },
    "report_window": {
        "mean": R(
            "rolling window= applies to total/active metrics only",
            CapabilityError,
            code="report.metric.unsupported",
        ),
        "conversion": R(
            "rolling window= applies to total/active metrics only",
            CapabilityError,
            code="report.metric.unsupported",
        ),
        "ratio": R(
            "rolling window= applies to total/active metrics only",
            CapabilityError,
            code="report.metric.unsupported",
        ),
        "retention": R(
            "retention needs a cohort anchor", CapabilityError, code="report.metric.unsupported"
        ),
        "quantile": R(
            "no moments representation", CapabilityError, code="report.metric.unsupported"
        ),
        "total": S(),
        "active": S(),
    },
    "from_moments": {
        "mean": S(),
        "conversion": S(),
        "ratio": S(note="denominator moment family rides the wire"),
        "retention": S(
            note="explicit MetricSpec re-declare round-trips a real export; the "
            "terse form refuses pointing at the explicit form"
        ),
        "quantile": R("needs unit-grain rows", CapabilityError, code="source.moments.unit_grain"),
        "total": NA("no MetricSpec type exists; the terse form refuses by name"),
        "active": NA("no MetricSpec type exists; the terse form refuses by name"),
    },
    "export": {
        "mean": S(),
        "conversion": S(),
        "ratio": S(),
        "retention": S(note="guardrail rows ride the export"),
        "quantile": R(
            "no moments representation", CapabilityError, code="source.frame.quantile_no_moments"
        ),
        "total": _NA_REPORT_LAYER,
        "active": _NA_REPORT_LAYER,
    },
    "sitewide": {
        "mean": S(),
        "conversion": S(),
        "ratio": S(note="numerator and denominator get independent site sums"),
        "retention": R(
            "site volume is undefined for a retention metric",
            CapabilityError,
            code="query.builders.site_volume_metric_type",
        ),
        "quantile": R(
            "site volume is undefined for a quantile metric",
            CapabilityError,
            code="query.builders.site_volume_metric_type",
        ),
        "total": _NA_REPORT_LAYER,
        "active": _NA_REPORT_LAYER,
    },
    "rollout": {
        "mean": S(note="priced over run_breakout output; relative log moments only"),
        "conversion": S(),
        "ratio": S(note="the ratio's own relative log moments carry it, like any other type"),
        "retention": S(note="native path; a guardrail's segments price like any other"),
        "quantile": NA(
            "no quantile BreakoutEstimate can exist: breakout refuses at the "
            "segment-moment seam, and this capability consumes that output only"
        ),
        "total": _NA_REPORT_LAYER,
        "active": _NA_REPORT_LAYER,
    },
}


# Capability x capability crosses: a pair enters when the code carries a
# refusal or support claim for it; the __all__ tripwire cannot police this table.

PAIRS: dict[tuple[str, str], Cell] = {
    ("cluster", "cuped"): R(
        "cannot combine with a CUPED covariate",
        CapabilityError,
        code="source.frame.cluster_capability",
    ),
    ("cluster", "sequential"): R(
        "clustered and observational sequential routes are unsupported",
        CapabilityError,
        code="sequential.route.unsupported",
    ),
    ("cluster", "prior"): R(
        "an informative prior is not supported with a declared cluster",
        CapabilityError,
        code="arm.adjustment.cluster_prior",
    ),
    ("cluster", "breakout"): R(
        "does not support breakout dimension 'seg'; declared breakouts: []",
        CapabilityError,
        code="readout.source.dimension",
    ),
    ("cluster", "daily_asof"): R(
        "from_unit_panel(cluster=...) is unavailable on a per-day panel",
        CapabilityError,
        code="source.frame_panel.cluster_grain",
    ),
    ("cluster", "export"): R(
        "wire format carries no cluster marker",
        CapabilityError,
        code="source.moments.cluster_grain",
    ),
    ("cluster", "sitewide"): S(
        warns="enrolled units exhaust the treated population",
        note="mean/conversion/retention/ratio decompose over cluster-grain moments via from_clusters",
        advisory=(
            "Clustered sitewide can miss treated units with no exposure row, "
            "so the relative-impact estimate is a lower bound; the "
            "absolute-impact point estimate is unaffected."
        ),
    ),
    ("cluster", "encouragement"): S(
        note="ITT + additive LATE cluster-robust; CUPED and complier-relative LATE still refuse"
    ),
    ("cluster", "observational"): S(note="adjusted estimators cluster their variance"),
    ("sequential", "observational"): R(
        "sequential likelihoods require a declared randomized or encouragement design",
        CapabilityError,
        code="sequential.route.unsupported",
    ),
    ("sequential", "encouragement"): S(
        note="registered raw ITT and Bernoulli uptake compliance",
        advisory="Support depends on the estimand: raw ITT and registered Bernoulli uptake compliance are supported; binary-uptake LATE is refused. Declare estimands explicitly.",
    ),
    ("breakout", "observational"): R(
        "breakout is not supported for an observational design",
        NotImplementedError,
        code="readout.view.observational",
    ),
    ("cuped", "encouragement"): S(note="numerator-only CUPED LATE"),
    ("sequential", "cuped"): S(
        note="asymptotic_mean registers adjusted_mean from retained (Y, X) moments",
        advisory=(
            "Admitted on InferenceSpec(kind='asymptotic_mean') only: the coefficient is "
            "fitted from retained joint moments at every look (Lindon et al. 2022). "
            "A pre-period coefficient is an alternative on the asymptotic scalar_mean "
            "route, not on AlwaysValid's exact Bernoulli route."
        ),
    ),
}


# Probe evidence for each MATRIX capability (one dedicated probe in
# tests/test_composition_matrix.py) and PAIRS key (the shared
# test_capability_pairs probe). `owner` is the production entry point it calls.


@dataclass(frozen=True)
class Provenance:
    probe: str
    owner: str


_COMPOSITION_PROBE = "tests/test_composition_matrix.py"
_PAIR_PROBE = f"{_COMPOSITION_PROBE}::test_capability_pairs"

CAPABILITY_PROVENANCE: dict[str, Provenance] = {
    "estimate": Provenance(f"{_COMPOSITION_PROBE}::test_estimate", "increment.readouts"),
    "cuped": Provenance(f"{_COMPOSITION_PROBE}::test_cuped", "increment.readouts"),
    "cluster": Provenance(f"{_COMPOSITION_PROBE}::test_cluster", "increment.readouts"),
    "sequential": Provenance(f"{_COMPOSITION_PROBE}::test_sequential", "increment.readouts"),
    "observational": Provenance(f"{_COMPOSITION_PROBE}::test_observational", "increment.readouts"),
    "encouragement": Provenance(f"{_COMPOSITION_PROBE}::test_encouragement", "increment.readouts"),
    "cate": Provenance(f"{_COMPOSITION_PROBE}::test_cate", "increment.cate"),
    "breakout": Provenance(f"{_COMPOSITION_PROBE}::test_breakout", "increment.readouts"),
    "daily_asof": Provenance(f"{_COMPOSITION_PROBE}::test_daily_asof", "increment.readouts"),
    "report_calendar": Provenance(
        f"{_COMPOSITION_PROBE}::test_report_calendar", "increment.reporting"
    ),
    "report_window": Provenance(f"{_COMPOSITION_PROBE}::test_report_window", "increment.reporting"),
    "from_moments": Provenance(f"{_COMPOSITION_PROBE}::test_from_moments", "increment.analysis"),
    "export": Provenance(f"{_COMPOSITION_PROBE}::test_export", "increment.analysis"),
    "sitewide": Provenance(f"{_COMPOSITION_PROBE}::test_sitewide", "increment.analysis"),
    "rollout": Provenance(f"{_COMPOSITION_PROBE}::test_rollout", "increment.breakout.rollout"),
}

PAIR_PROVENANCE: dict[tuple[str, str], Provenance] = {
    ("cluster", "cuped"): Provenance(_PAIR_PROBE, "increment._frame_validation"),
    ("cluster", "sequential"): Provenance(_PAIR_PROBE, "increment.readouts"),
    ("cluster", "prior"): Provenance(_PAIR_PROBE, "increment.readouts"),
    ("cluster", "breakout"): Provenance(_PAIR_PROBE, "increment.readouts"),
    ("cluster", "daily_asof"): Provenance(_PAIR_PROBE, "increment.frame"),
    ("cluster", "export"): Provenance(_PAIR_PROBE, "increment.analysis"),
    ("cluster", "sitewide"): Provenance(_PAIR_PROBE, "increment.analysis"),
    ("cluster", "encouragement"): Provenance(_PAIR_PROBE, "increment.readouts"),
    ("cluster", "observational"): Provenance(_PAIR_PROBE, "increment.readouts"),
    ("sequential", "observational"): Provenance(_PAIR_PROBE, "increment.readouts"),
    ("sequential", "encouragement"): Provenance(_PAIR_PROBE, "increment.readouts"),
    ("breakout", "observational"): Provenance(_PAIR_PROBE, "increment.readouts"),
    ("cuped", "encouragement"): Provenance(_PAIR_PROBE, "increment.readouts"),
    ("sequential", "cuped"): Provenance(_PAIR_PROBE, "increment.sequential_source"),
}


# Scenario/evidence catalog: user-facing higher-order scenarios, evaluated
# against real production callables rather than accepted declarations.

EvidenceKind = Literal["unit", "integration", "parameter_recovery"]


@dataclass(frozen=True)
class EvidenceRef:
    kind: EvidenceKind
    path: str
    test: str


@dataclass(frozen=True)
class Scenario:
    id: str
    axes: tuple[tuple[str, str], ...]
    expected_runtime: RuntimeStatus
    expected_family: FamilyStatus
    explanation: str
    alternative: str | None
    evidence: tuple[EvidenceRef, ...]
    evaluate: Callable[[], CompatibilityReport]

    def __post_init__(self) -> None:
        names = [name for name, _ in self.axes]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"{self.id!r}: duplicate axis name(s) {duplicates!r} in axes")


def _scenario_axis_signature(scenario: Scenario) -> tuple[tuple[str, str], ...]:
    """Order-insensitive identity for a scenario's axes. Two scenarios
    declaring the same axis/value pairs in a different order are the same
    scenario for uniqueness purposes; the renderer's server-side filter
    matching and the browser's client-side ``exactScenarioMatch`` must
    never see two indistinguishable scenarios and resolve between them
    arbitrarily."""
    return tuple(sorted(scenario.axes))


def _always_valid_quantile_report(role: Literal["primary", "secondary"]):
    family = (
        FamilyPolicy(
            kind="e_bh",
            axes=("metric", "arm"),
            nominal_alpha=0.05,
        )
        if role == "secondary"
        else FamilyPolicy(kind="none", axes=(), nominal_alpha=0.05)
    )
    arm_request = ArmCompatibilityRequest(
        assignment=ParallelAssignment(),
        analysis=AnalysisAxes(
            identification="randomized",
            view="total",
            segmented=False,
            completed_windows_only=False,
            population="assigned",
            variance_adjustment="none",
        ),
        dependence="iid",
        inference=AlwaysValid(registration=registration()),
        estimand="relative_lift",
        metric=MetricCapabilities(
            metric_type="quantile",
            value_scale="relative",
            winsorization="none",
            outcome_window="bounded",
            uptake_window="not_applicable",
        ),
        decision=RelativeDecisionPolicy(
            alternative="two-sided",
            null_lift=0.0,
            family=family,
        ),
        methods=(
            MethodCapability(
                role="decision",
                estimator="unadjusted",
                variance_reduction="none",
            ),
        ),
        prior_present=False,
    )
    plan_request = PlanFamilyCompatibilityRequest(
        metric="latency",
        metric_type="quantile",
        role=role,
        inference="always_valid",
    )
    return inspect_compatibility(
        (
            CompatibilityCheck(
                capability="arm_moments",
                request=arm_request,
                evaluate=ARM_EVIDENCE_CONTRACT.runtime_support,
                observe=partial(
                    default_support_observe,
                    capability="arm_moments",
                    evidence_source="contract",
                ),
            ),
            CompatibilityCheck(
                capability="plan_family",
                request=plan_request,
                evaluate=plan_family_compatibility,
                observe=partial(
                    contextual_decision_observe,
                    capability="plan_family",
                    evidence_source="contract",
                    reference="plan_family",
                ),
            ),
        )
    )


SCENARIOS = (
    Scenario(
        id="always-valid-quantile",
        axes=(
            ("capability", "sequential"),
            ("metric_type", "quantile"),
            ("inference", "always_valid"),
            ("family_role", "primary"),
            ("multiplicity", "disabled"),
        ),
        expected_runtime="refused",
        expected_family="not_applicable",
        explanation=(
            "Quantile sequential inference has no matching raw likelihood and is refused before data."
        ),
        alternative=(
            "Use fixed-horizon quantile inference (valid for one planned analysis, not repeated looks)."
        ),
        evidence=(
            EvidenceRef(
                kind="integration",
                path="tests/test_analysis_quantile.py",
                test="test_declared_plan_always_valid_inference_reaches_quantile_readout",
            ),
            EvidenceRef(
                kind="integration",
                path="tests/estimation/test_quantile_sequential_coverage.py",
                test="test_quantile_sequential_historical_tail_grid_refuses_before_outcomes",
            ),
        ),
        evaluate=partial(_always_valid_quantile_report, "primary"),
    ),
    Scenario(
        id="always-valid-quantile-secondary-family",
        axes=(
            ("capability", "sequential"),
            ("metric_type", "quantile"),
            ("inference", "always_valid"),
            ("family_role", "secondary"),
            ("multiplicity", "enabled"),
        ),
        expected_runtime="refused",
        expected_family="excluded",
        explanation=(
            "The unsupported quantile likelihood refuses the analysis before family evidence."
        ),
        alternative=(
            "Use fixed-horizon quantile inference for family participation (valid for one planned analysis, not repeated looks)."
        ),
        evidence=(
            EvidenceRef(
                kind="unit",
                path="tests/test_plan_resolver.py",
                test="TestAlwaysValidQuantileWarning::test_quantile_refuses_before_plan_can_claim_sequential_evidence",
            ),
            EvidenceRef(
                kind="integration",
                path="tests/test_analysis_quantile.py",
                test="test_declared_plan_always_valid_inference_reaches_quantile_readout",
            ),
        ),
        evaluate=partial(_always_valid_quantile_report, "secondary"),
    ),
)


# Both pytest and the renderer validate evidence before publishing coverage claims.


def _find_evidence_function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Resolve only a module's direct function or a direct class method."""
    class_name, sep, method_name = name.partition("::")
    if sep:
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for child in ast.iter_child_nodes(node):
                    if (
                        isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
                        and child.name == method_name
                    ):
                        return child
                raise ValueError(f"class {class_name!r} has no method named {method_name!r}")
        raise ValueError(f"no class named {class_name!r}")
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
            return node
    raise ValueError(f"no top-level function named {name!r}")


def _module_dotted_name(path: Path) -> str:
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def _runtime_marks(module: object, name: str) -> tuple[set[str], set[str]]:
    """Collect module, class-MRO, and function marks as pytest does."""
    class_name, sep, method_name = name.partition("::")
    marks_sources: list[object] = [module]
    if sep:
        cls = getattr(module, class_name)
        marks_sources.extend(reversed(cls.__mro__))
        marks_sources.append(getattr(cls, method_name))
    else:
        marks_sources.append(getattr(module, name))
    marks: set[str] = set()
    ids: set[str] = set()
    for source in marks_sources:
        source_marks = vars(source).get("pytestmark", [])
        if not isinstance(source_marks, list):
            source_marks = [source_marks]
        for mark in source_marks:
            marks.add(mark.name)
            if mark.name == "compatibility" and mark.args:
                ids.add(mark.args[0])
    return marks, ids


def validate_evidence(scenarios: tuple[Scenario, ...] = SCENARIOS) -> None:
    """Validate that every evidence reference names a real, correctly marked test.

    Function existence is checked statically. Parameter-recovery markers are
    checked on imported objects, including inherited class marks, using the
    same ``pytestmark`` metadata pytest collects.
    """
    for scenario in scenarios:
        for ref in scenario.evidence:
            path = ROOT / ref.path
            if not path.is_file():
                raise ValueError(f"{scenario.id}: {ref.path} does not exist")
            tree = ast.parse(path.read_text(), filename=str(path))
            _find_evidence_function(tree, ref.test)
            if ref.kind != "parameter_recovery":
                continue
            module = importlib.import_module(_module_dotted_name(path))
            marks, ids = _runtime_marks(module, ref.test)
            if "parameter_recovery" not in marks:
                raise ValueError(
                    f"{scenario.id}: {ref.test} is missing the parameter_recovery marker"
                )
            if "compatibility" not in marks:
                raise ValueError(f"{scenario.id}: {ref.test} is missing the compatibility marker")
            if scenario.id not in ids:
                raise ValueError(
                    f"{scenario.id}: {ref.test}'s compatibility marker id does not match"
                )


# Both pytest and the renderer compare declared outcomes with real inspection results.


def validate_scenario_outcome(scenario: Scenario, report: CompatibilityReport) -> None:
    """Refuse a real report that disagrees with its scenario's declared outcomes."""
    if (
        report.overall != scenario.expected_runtime
        or report.overall_family != scenario.expected_family
    ):
        raise ValueError(
            f"{scenario.id}: observed (overall={report.overall!r}, "
            f"overall_family={report.overall_family!r}) does not match declared "
            f"(expected_runtime={scenario.expected_runtime!r}, "
            f"expected_family={scenario.expected_family!r})"
        )
