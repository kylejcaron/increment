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


CUPED_LINEAR_INTERACTIONS = (
    MethodInteraction(
        "scalar_cuped",
        "supported",
        None,
        "tests/test_cuped_covariates.py::test_scalar_cuped_can_share_a_declared_covariate_with_cuped_linear",
        "Declare the same covariate by scalar name and as a member of the mixed covariate set.",
    ),
    MethodInteraction(
        "ratio",
        "supported",
        None,
        "tests/test_warehouse_cuped_parity.py::test_mixed_cuped_mean_ratio_conversion_rows_match_all_supported_ingresses",
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
        "tests/test_cuped_linear_acceptance.py::test_cuped_linear_fixed_upper_value_clipping_matches_preclipped_fit",
        "Fixed upper and lower value bounds are clipped before fitting.",
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
        "tests/test_cuped_breakout_review.py::test_triggered_native_breakout_linear_matches_triggered_unit_oracle",
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
        ),
    ),
    MethodInteraction(
        "multiplicity_roles",
        "supported",
        None,
        "tests/test_warehouse_cuped_parity.py::test_mixed_cuped_mean_ratio_conversion_rows_match_all_supported_ingresses",
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
