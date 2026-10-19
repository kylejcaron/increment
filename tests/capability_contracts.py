"""Metadata shared by capability rendering and the tests owning each consumer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type OutcomeKind = Literal[
    "supported",
    "refused",
    "unavailable",
    "source_limited",
    "not_expressible",
    "not_comparable",
    "replay_only",
    "unfinished",
    "unsound",
]


@dataclass(frozen=True)
class ConsumerOutcome:
    kind: OutcomeKind
    codes: tuple[str, ...] = ()
    reason: str = ""
    stage: str = "read"

    def __post_init__(self) -> None:
        if self.kind in {"refused", "unavailable"} and len(self.codes) != 1:
            raise ValueError("a refused or unavailable consumer outcome requires one exact code")
        if self.kind not in {"supported", "refused", "unavailable"} and self.codes:
            raise ValueError("only supported warnings, refusals and unavailable rows carry codes")
        if self.kind != "supported" and not self.reason:
            raise ValueError("unavailable consumer outcomes require their reason")
        if not self.stage:
            raise ValueError("consumer outcomes require their evaluation stage")
        if len(self.codes) != len(set(self.codes)):
            raise ValueError("consumer outcome codes must be unique")


@dataclass(frozen=True)
class ConsumerExpectation:
    id: str
    operation: str
    ingress: str
    condition: str
    expected: ConsumerOutcome
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if not all((self.id, self.operation, self.ingress, self.condition, self.evidence)):
            raise ValueError("consumer expectations require subject, condition and evidence")


@dataclass(frozen=True)
class ConsumerContract:
    id: str
    outcomes: tuple[ConsumerExpectation, ...]

    def __post_init__(self) -> None:
        if not self.id or not self.outcomes:
            raise ValueError("consumer contracts require an identity and outcomes")
        ids = [outcome.id for outcome in self.outcomes]
        if len(ids) != len(set(ids)):
            raise ValueError("consumer outcome identities must be unique within a contract")

    def outcome(self, outcome_id: str) -> ConsumerExpectation:
        for outcome in self.outcomes:
            if outcome.id == outcome_id:
                return outcome
        raise KeyError(f"unknown consumer outcome {self.id}:{outcome_id}")


@dataclass(frozen=True)
class ParityDeclaration:
    attempted: tuple[str, ...]
    waive: tuple[tuple[str, str], ...] = ()
    refusals: tuple[tuple[str, str], ...] = ()
    absences: tuple[tuple[str, str, str], ...] = ()
    warnings: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        if len(self.attempted) != len(set(self.attempted)):
            raise ValueError("parity attempts must be unique")
        waived = dict(self.waive)
        attempted = set(self.attempted)
        if not set(dict(self.refusals)) <= attempted & waived.keys():
            raise ValueError("refused parity paths must be attempted and explained")
        if not {ingress for ingress, _, _ in self.absences} <= attempted & waived.keys():
            raise ValueError("absent parity paths must be attempted and explained")


CONSTRUCTORS = (
    "from_definitions",
    "from_unit_day_artifact",
    "from_unit_summary",
    "from_unit_panel",
    "from_switchback_panel",
    "from_moments",
)

InteractionStatus = Literal["supported", "source_limited", "unfinished", "mathematically_unsound"]


@dataclass(frozen=True)
class InteractionDisposition:
    status: InteractionStatus
    code: str | None = None
    route_forward: str = ""
    stage: Literal["read", "construction"] = "construction"
    evidence: str | None = None


@dataclass(frozen=True)
class MethodInteraction:
    """Measured compatibility of mixed-covariate CUPED with another method axis."""

    name: str
    status: InteractionStatus
    code: str | None
    evidence: str
    route_forward: str = ""
    ingress_dispositions: tuple[tuple[str, InteractionDisposition], ...] = ()

    def for_ingress(self, ingress: str) -> InteractionDisposition:
        override = next(
            (disposition for name, disposition in self.ingress_dispositions if name == ingress),
            None,
        )
        if override is not None:
            return override
        return InteractionDisposition(
            self.status,
            self.code,
            self.route_forward,
            "read" if self.status == "supported" else "construction",
            self.evidence,
        )


_JOURNEY = "tests/test_cuped_portable_journey.py"
_PORTABLE_CATEGORICAL = InteractionDisposition(
    "supported",
    route_forward="Categorical-only mixed requests replay from format-13 joint cells; continuous mixed covariates refuse with facade.analysis_config.cuped_linear_moments.",
    stage="read",
    evidence=f"{_JOURNEY}::test_definitions_artifact_moments_journey_replays_companions_without_unit_transfer",
)


CUPED_LINEAR_INTERACTIONS = (
    MethodInteraction(
        "scalar_cuped",
        "supported",
        None,
        "tests/test_warehouse_cuped_parity.py::test_scalar_cuped_composes_with_mixed_cuped_on_all_four_ingresses",
        "Frames declare scalar and mixed covariates on their respective Method.cuped(covariates=...) declarations; definitions derive the scalar covariate from n_pre_periods > 0 and bind mixed covariates separately. Publish the cuped_preperiod artifact extension.",
        (("from_moments", _PORTABLE_CATEGORICAL),),
    ),
    MethodInteraction(
        "ratio",
        "supported",
        None,
        "tests/test_warehouse_cuped_parity.py::test_mixed_cuped_mean_ratio_conversion_rows_match_all_supported_ingresses",
        ingress_dispositions=(
            (
                "from_moments",
                InteractionDisposition(
                    "supported",
                    route_forward=_PORTABLE_CATEGORICAL.route_forward,
                    stage="read",
                    evidence="tests/test_cuped_evidence.py::test_format13_method_scoped_export_reloads_exact_inference",
                ),
            ),
        ),
    ),
    MethodInteraction(
        "clustering",
        "unfinished",
        "estimation.engine.cuped_linear_cluster",
        "tests/test_cuped_linear_acceptance.py::test_composition_gates_refuse_before_source_acquisition",
        "Implement cluster-robust HC2/Bell-McCaffrey inference for the WLS fit.",
        (
            (
                "from_unit_panel",
                InteractionDisposition(
                    "source_limited",
                    "source.frame_panel.cluster_grain",
                    "The panel is collapsed to one row per unit before clustering could apply.",
                    "construction",
                    "tests/test_cuped_linear_acceptance.py::test_cuped_linear_interactions_preserve_ingress_specific_refusals",
                ),
            ),
        ),
    ),
    MethodInteraction(
        "sequential",
        "unfinished",
        "estimation.engine.cuped_linear_sequential",
        "tests/test_cuped_linear_acceptance.py::test_composition_gates_refuse_before_source_acquisition",
        "Implement predictable multivariate sequential construction; tracked by kata 6rhs.",
    ),
    MethodInteraction(
        "winsorization_fixed_value",
        "supported",
        None,
        "tests/test_warehouse_cuped_parity.py::test_fixed_value_winsorization_with_both_bounds_matches_all_supported_ingresses",
        "Fixed lower and upper value bounds are exercised on each supported unit ingress and compared with shared row and retained-scope normalization.",
        (
            (
                "from_moments",
                InteractionDisposition(
                    "unfinished",
                    "query.cuped.general_qr_required",
                    "Warehouse exports of bounded outcomes need general warehouse QR, which is not implemented; supply per-unit data through from_unit_summary or from_unit_panel.",
                    "construction",
                    f"{_JOURNEY}::test_fixed_caps_route_the_portable_export_to_general_qr_without_transfer",
                ),
            ),
        ),
    ),
    MethodInteraction(
        "winsorization_percentile",
        "unfinished",
        "estimation.engine.cuped_linear_winsorization",
        "tests/test_cuped_linear_acceptance.py::test_composition_gates_refuse_before_source_acquisition",
        "Implement winsorized CUPED fitting and interval construction for percentile bounds.",
    ),
    MethodInteraction(
        "breakouts",
        "supported",
        None,
        "tests/test_warehouse_cuped_parity.py::test_mixed_cuped_breakout_rows_match_panel_definitions_and_artifact",
        ingress_dispositions=(
            (
                "from_unit_summary",
                InteractionDisposition(
                    "source_limited",
                    "facade.analysis.operation",
                    "A unit-summary frame has no breakout dimension.",
                    "read",
                    "tests/test_cuped_linear_acceptance.py::test_cuped_linear_interactions_preserve_ingress_specific_refusals",
                ),
            ),
            (
                "from_unit_panel",
                InteractionDisposition(
                    "supported",
                    evidence="tests/test_cuped_linear_acceptance.py::test_cuped_linear_breakout_emits_complete_per_cell_family_evidence",
                ),
            ),
            (
                "from_moments",
                InteractionDisposition(
                    "source_limited",
                    "facade.analysis.operation",
                    "Portable moments retain no per-unit breakout dimension.",
                    "read",
                    f"{_JOURNEY}::test_definitions_artifact_moments_journey_replays_companions_without_unit_transfer",
                ),
            ),
        ),
    ),
    MethodInteraction(
        "multiplicity_roles",
        "supported",
        None,
        "tests/test_warehouse_cuped_parity.py::test_mixed_cuped_mean_ratio_conversion_rows_match_all_supported_ingresses",
        ingress_dispositions=(("from_moments", _PORTABLE_CATEGORICAL),),
    ),
    MethodInteraction(
        "triggered_population",
        "supported",
        None,
        "tests/test_cuped_breakout_review.py::test_triggered_cuped_definition_and_artifact_runs_match",
        ingress_dispositions=tuple(
            (
                ingress,
                InteractionDisposition(
                    "source_limited",
                    "facade.analysis.trigger_unsupported",
                    "Dataframe ingresses do not retain the source trigger chain.",
                    "read",
                    "tests/test_cuped_linear_acceptance.py::test_cuped_linear_interactions_preserve_ingress_specific_refusals",
                ),
            )
            for ingress in ("from_unit_summary", "from_unit_panel")
        ),
    ),
)
