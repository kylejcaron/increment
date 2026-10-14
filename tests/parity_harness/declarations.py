"""Authoritative expected outcomes consumed by the registered row-parity cases."""

from types import MappingProxyType

from tests.capability_consumers_auxiliary import PARITY_DECLARATIONS as _AUXILIARY_DECLARATIONS
from tests.capability_consumers_scalar import PARITY_DECLARATIONS as _SCALAR_DECLARATIONS
from tests.capability_contracts import ParityDeclaration

PARITY_DECLARATIONS = MappingProxyType(
    {
        "triggered_sequential_asymptotic_mean": ParityDeclaration(
            attempted=("from_definitions", "from_unit_day_artifact", "from_moments"),
            waive=(
                (
                    "from_unit_summary",
                    "SOURCE: unit summaries carry no trigger declaration or per-unit trigger anchors.",
                ),
                (
                    "from_unit_panel",
                    "SOURCE: unit panels carry no trigger declaration or per-unit trigger anchors.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: switchback contrasts carry no triggered arm evidence.",
                ),
            ),
        ),
        "artifact-breakout-exposure-coverage": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
                ("from_moments", "SOURCE: portable moments retain no per-unit breakout dimension"),
            ),
            refusals=(("from_moments", "facade.analysis.operation"),),
        ),
        "artifact_context_format_one_refusal": ParityDeclaration(
            attempted=("from_unit_day_artifact",),
            waive=(
                (
                    "from_unit_summary",
                    "SOURCE: no warehouse artifact context is carried by a frame",
                ),
                ("from_unit_panel", "SOURCE: no warehouse artifact context is carried by a frame"),
                (
                    "from_switchback_panel",
                    "SOURCE: no warehouse artifact context is carried by a switchback",
                ),
                (
                    "from_definitions",
                    "SOURCE: definitions are a publisher, not a stored-context reader",
                ),
                ("from_moments", "SOURCE: portable moments carry no warehouse artifact context"),
                (
                    "from_unit_day_artifact",
                    "COMBINATION: the current reader refuses a format-1 context",
                ),
            ),
            refusals=(("from_unit_day_artifact", "artifact.format.unsupported"),),
        ),
        "artifact_context_format_two": ParityDeclaration(
            attempted=("from_definitions", "from_unit_day_artifact"),
            waive=(
                (
                    "from_unit_summary",
                    "SOURCE: no warehouse artifact context is carried by a frame",
                ),
                ("from_unit_panel", "SOURCE: no warehouse artifact context is carried by a frame"),
                (
                    "from_switchback_panel",
                    "SOURCE: no warehouse artifact context is carried by a switchback",
                ),
                ("from_moments", "SOURCE: portable moments carry no warehouse artifact context"),
            ),
        ),
        "audit-adjusted-ratio-anchor": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
                (
                    "from_unit_panel",
                    "SOURCE: registered panel capture cannot supply the per-unit pre-period covariate for adjusted_ratio_mean; use a unit-summary frame.",
                ),
            ),
            refusals=(("from_unit_panel", "frame.validation.from_unit_panel"),),
        ),
        "audit-adjusted-ratio-anchor-resolved": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
                (
                    "from_unit_panel",
                    "SOURCE: registered panel capture cannot supply the per-unit pre-period covariate for adjusted_ratio_mean; use a unit-summary frame.",
                ),
            ),
            refusals=(("from_unit_panel", "frame.validation.from_unit_panel"),),
        ),
        "audit-artifact-rounded-aggregates": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "audit-binary-breakout-ancillary-uptake": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
                ("from_moments", "SOURCE: portable moments retain no per-unit breakout dimension"),
            ),
            refusals=(("from_moments", "facade.analysis.operation"),),
        ),
        "audit-cluster-export-refusal": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_moments",
            ),
            waive=(
                (
                    "from_moments",
                    "SOURCE: export_source_moments refuses a clustered, non-Encouragement source ('cluster transport requires a complete encouragement compliance payload') -- the moments wire format carries no cluster marker, so from_moments would rehydrate the cluster rows as unit-grain ratio moments and silently misread them.",
                ),
                (
                    "from_unit_panel",
                    "SOURCE cannot supply a clustered arm-lift ratio readout: Analysis.from_unit_panel(cluster=...) refuses unconditionally with source.frame_panel.cluster_grain -- the panel is collapsed to one row per unit before clustering could apply, and the collapse is ambiguous on this shape (increment/frame.py); build the clustered arm totals with from_unit_summary(cluster=...) instead.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE cannot supply a clustered arm-lift ratio readout: a switchback panel declares a shared-schedule contrast (ContrastResults), a structurally different estimand shape from the LiftEstimate rows this case compares.",
                ),
            ),
            refusals=(("from_moments", "source.moments.cluster_grain"),),
        ),
        "audit-compliance-only-empty-catalog": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                ("from_switchback_panel", "COMBINATION: no switchback schedule in this dataset"),
            ),
        ),
        "audit-compliance-only-outcome-missingness": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                ("from_switchback_panel", "COMBINATION: no switchback schedule in this dataset"),
                (
                    "from_unit_panel",
                    "CONSTRUCTION: panel densification has no complete-case missing='drop' path; zero-filling would change the declared outcome policy",
                ),
            ),
            refusals=(("from_unit_panel", "frame.missing_policy.panel_drop"),),
        ),
        "audit-compliance-only-sequential-empty-catalog": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "audit-compliance-only-sequential-unbounded-retention-catalog": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
                (
                    "from_unit_summary",
                    "SOURCE: a one-row-per-unit summary has no per-unit dates to resolve a retention band against, so the catalog cannot be declared.",
                ),
                (
                    "from_unit_panel",
                    "COMBINATION: a unit panel refuses a retention metric under an Encouragement design at construction, before any request is read.",
                ),
            ),
            refusals=(
                ("from_unit_summary", "source.frame.constructor"),
                ("from_unit_panel", "readout.encouragement.retention"),
            ),
        ),
        "audit-cuped-covariate-rescaling": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "audit-encouragement-multi-metric-breakout": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
                ("from_moments", "SOURCE: portable moments retain no per-unit breakout dimension"),
            ),
            refusals=(("from_moments", "facade.analysis.operation"),),
        ),
        "audit-panel-component-horizon": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "audit-panel-component-horizon-ratio": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
            warnings=(
                ("from_definitions", ("frame.censoring.dropped_units",)),
                ("from_unit_panel", ("frame.censoring.dropped_units",)),
            ),
        ),
        "audit-panel-component-horizon-ratio-complete": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "audit-panel-missing-zero-inferred": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "audit-panel-pre-exposure-and-selection": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_moments",
                    "SOURCE: a quantile has no moments representation -- export() itself refuses before a from_moments cube could ever exist.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: switchback metrics must be type='mean' or type='conversion'; quantile is refused before any frame is read (confirmed live: source.frame.switchback.metric).",
                ),
            ),
            refusals=(("from_moments", "source.frame.quantile_no_moments"),),
        ),
        "audit-pinned-dimension-freshness": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "audit-prior-reset-family": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "audit-quantile-family": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_moments",
                    "SOURCE: a quantile has no moments representation -- export() itself refuses before a from_moments cube could ever exist.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: switchback metrics must be type='mean' or type='conversion'; quantile is refused before any frame is read (confirmed live: source.frame.switchback.metric).",
                ),
            ),
            refusals=(("from_moments", "source.frame.quantile_no_moments"),),
        ),
        "audit-ratio-neighboring-means": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "audit-retention-breakout-cohorts": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_panel",
                "from_unit_summary",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
                (
                    "from_unit_summary",
                    "SOURCE: unit summaries cannot supply retention exposure cohorts.",
                ),
                (
                    "from_moments",
                    "SOURCE: portable moments retain no breakout operation or cohort identity.",
                ),
            ),
            refusals=(
                ("from_unit_summary", "source.frame.constructor"),
                ("from_moments", "facade.analysis.operation"),
            ),
        ),
        "audit-sequential-missing-zero": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "audit-sequential-messy-mean-conversion": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "COMBINATION: unequal declared allocation is refused by the switchback constructor with source.frame.switchback.identification.",
                ),
                (
                    "from_unit_panel",
                    "COMBINATION: sequential panel capture refuses the CUPED metric with `window_days=None`; panel metrics require bounded windows (`sequential.route.unsupported`).",
                ),
            ),
            refusals=(("from_unit_panel", "sequential.route.unsupported"),),
        ),
        "audit-switchback-neighboring-integers": ParityDeclaration(
            attempted=("from_switchback_panel",),
            waive=(
                (
                    "from_definitions",
                    "SOURCE: parallel-arm and portable-moment constructors carry no switchback schedule",
                ),
                (
                    "from_unit_day_artifact",
                    "SOURCE: parallel-arm and portable-moment constructors carry no switchback schedule",
                ),
                (
                    "from_unit_summary",
                    "SOURCE: parallel-arm and portable-moment constructors carry no switchback schedule",
                ),
                (
                    "from_unit_panel",
                    "SOURCE: parallel-arm and portable-moment constructors carry no switchback schedule",
                ),
                (
                    "from_moments",
                    "SOURCE: parallel-arm and portable-moment constructors carry no switchback schedule",
                ),
            ),
        ),
        "audit-uptake-timestamp-boundaries": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "breakout_boolean_store_dimension": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_moments",
                    "SOURCE: a moments cube contains aggregate rows, not unit-level breakout dimensions.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
            refusals=(("from_moments", "facade.analysis.operation"),),
        ),
        "breakout_store_dimension": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_moments",
                    "SOURCE: a moments cube contains aggregate rows, not unit-level breakout dimensions.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
            refusals=(("from_moments", "facade.analysis.operation"),),
        ),
        "clustered_assignment_integrity_scope": ParityDeclaration(
            attempted=("from_definitions", "from_unit_day_artifact", "from_unit_summary"),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: a switchback contrast does not expose parallel-arm assignment counts.",
                ),
                (
                    "from_unit_panel",
                    "SOURCE: the panel constructor has no cluster key, so it cannot preserve the declared randomization grain.",
                ),
                (
                    "from_moments",
                    "SOURCE: this clustered case has no portable cluster-identity payload.",
                ),
            ),
        ),
        "clustered_negative_control_mean_produces_a_fieller_set": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_moments",
            ),
            waive=(
                (
                    "from_moments",
                    "SOURCE: export_source_moments refuses a clustered, non-Encouragement source ('cluster transport requires a complete encouragement compliance payload') -- the moments wire format carries no cluster marker, so from_moments would rehydrate the cluster rows as unit-grain ratio moments and silently misread them.",
                ),
                (
                    "from_unit_panel",
                    "SOURCE cannot supply a clustered arm-lift ratio readout: Analysis.from_unit_panel(cluster=...) refuses unconditionally with source.frame_panel.cluster_grain -- the panel is collapsed to one row per unit before clustering could apply, and the collapse is ambiguous on this shape (increment/frame.py); build the clustered arm totals with from_unit_summary(cluster=...) instead.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE cannot supply a clustered arm-lift ratio readout: a switchback panel declares a shared-schedule contrast (ContrastResults), a structurally different estimand shape from the LiftEstimate rows this case compares.",
                ),
            ),
            refusals=(("from_moments", "source.moments.cluster_grain"),),
        ),
        "clustered_zero_relative_variance_preserves_point_and_additive_interval": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_moments",
            ),
            waive=(
                (
                    "from_moments",
                    "SOURCE: export_source_moments refuses a clustered, non-Encouragement source ('cluster transport requires a complete encouragement compliance payload') -- the moments wire format carries no cluster marker, so from_moments would rehydrate the cluster rows as unit-grain ratio moments and silently misread them.",
                ),
                (
                    "from_unit_panel",
                    "SOURCE cannot supply a clustered arm-lift ratio readout: Analysis.from_unit_panel(cluster=...) refuses unconditionally with source.frame_panel.cluster_grain -- the panel is collapsed to one row per unit before clustering could apply, and the collapse is ambiguous on this shape (increment/frame.py); build the clustered arm totals with from_unit_summary(cluster=...) instead.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE cannot supply a clustered arm-lift ratio readout: a switchback panel declares a shared-schedule contrast (ContrastResults), a structurally different estimand shape from the LiftEstimate rows this case compares.",
                ),
            ),
            refusals=(("from_moments", "source.moments.cluster_grain"),),
        ),
        "conversion_route_breakout_segments": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
                ("from_moments", "SOURCE: portable moments retain no per-unit breakout dimension"),
            ),
        ),
        "conversion_route_encouragement_family": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "conversion_route_observational_family": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: a per-unit adjustment set has no analogue on a switchback block/period schedule, and this dataset has none.",
                ),
            ),
        ),
        "conversion_route_straddles_the_dense_threshold": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "dashboard-breakout-reads": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_moments",
                    "SOURCE: Analysis.from_moments's own docstring names run_breakout among the methods it refuses -- a moments cube enters at moment grain and carries no per-unit breakout dimension.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
            refusals=(("from_moments", "facade.analysis.operation"),),
        ),
        "dashboard-group-data": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "encouragement_declared_definitions": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                ("from_switchback_panel", "COMBINATION: no switchback schedule in this dataset"),
            ),
        ),
        "encouragement_fcr_reestimation": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                ("from_switchback_panel", "COMBINATION: no switchback schedule in this dataset"),
            ),
        ),
        "encouragement_guardrail_late_only_request": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                ("from_switchback_panel", "COMBINATION: no switchback schedule in this dataset"),
            ),
        ),
        "encouragement_guardrail_late_rows": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                ("from_switchback_panel", "COMBINATION: no switchback schedule in this dataset"),
            ),
        ),
        "encouragement_itt_dense_takes_the_delta_method_route": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "encouragement_itt_keeps_binomial_reference_kind": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "encouragement_margin_guardrail": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                ("from_switchback_panel", "COMBINATION: no switchback schedule in this dataset"),
            ),
        ),
        "encouragement_margin_guardrail_late_only": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                ("from_switchback_panel", "COMBINATION: no switchback schedule in this dataset"),
                ("from_definitions", "CONSTRUCTION: the request excludes the ITT guardrail result"),
                (
                    "from_unit_day_artifact",
                    "CONSTRUCTION: the request excludes the ITT guardrail result",
                ),
                (
                    "from_unit_summary",
                    "CONSTRUCTION: the request excludes the ITT guardrail result",
                ),
                ("from_unit_panel", "CONSTRUCTION: the request excludes the ITT guardrail result"),
                ("from_moments", "CONSTRUCTION: the request excludes the ITT guardrail result"),
            ),
            refusals=(
                ("from_definitions", "readout.encouragement.margin"),
                ("from_unit_day_artifact", "readout.encouragement.margin"),
                ("from_unit_summary", "readout.encouragement.margin"),
                ("from_unit_panel", "readout.encouragement.margin"),
                ("from_moments", "readout.encouragement.margin"),
            ),
        ),
        "encouragement_optional_exclusion": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                ("from_switchback_panel", "COMBINATION: no switchback schedule in this dataset"),
            ),
        ),
        "encouragement_retention_probe": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                ("from_switchback_panel", "COMBINATION: no switchback schedule in this dataset"),
            ),
        ),
        "encouragement_supplied_design_definitions": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                ("from_switchback_panel", "COMBINATION: no switchback schedule in this dataset"),
            ),
        ),
        "exact_binomial_beyond_the_former_arm_ceiling": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "exact_counts_conversion_window_boundary_export_from_definitions": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel contrasts per-cycle block outcomes of one switching population and emits no arm-level unit-grain moments, so it has no exact arm `successes`/`n` pair to compare with the parallel-arm ingresses.",
                ),
            ),
        ),
        "exact_counts_conversion_window_boundary_export_from_unit_panel": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel contrasts per-cycle block outcomes of one switching population and emits no arm-level unit-grain moments, so it has no exact arm `successes`/`n` pair to compare with the parallel-arm ingresses.",
                ),
            ),
        ),
        "exact_counts_retention_cohort_boundary_export_from_definitions": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel contrasts per-cycle block outcomes of one switching population and emits no arm-level unit-grain moments, so it has no exact arm `successes`/`n` pair to compare with the parallel-arm ingresses.",
                ),
                (
                    "from_unit_summary",
                    "SOURCE: unit summaries cannot supply retention exposure cohorts.",
                ),
            ),
            refusals=(("from_unit_summary", "source.frame.constructor"),),
        ),
        "exact_counts_retention_cohort_boundary_export_from_unit_panel": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel contrasts per-cycle block outcomes of one switching population and emits no arm-level unit-grain moments, so it has no exact arm `successes`/`n` pair to compare with the parallel-arm ingresses.",
                ),
                (
                    "from_unit_summary",
                    "SOURCE: unit summaries cannot supply retention exposure cohorts.",
                ),
            ),
            refusals=(("from_unit_summary", "source.frame.constructor"),),
        ),
        "exploratory-metrics": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "lift_prior_mixture_mean_ratio_conversion_cuped": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "lift_prior_normal_mean_ratio_conversion_cuped": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "lift_prior_student_t_mean_ratio_conversion_cuped": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "mean_ratio_conversion_cuped_fixed_horizon": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "multiplicity_primary_secondaries_guardrail": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "nonpositive_mean_additive_row_matches_across_sources": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "observational_aipw_conversion_diagnostics": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_moments",
                    "SOURCE: a moments cube holds no per-unit rows to attach a covariate to.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: a per-unit adjustment set has no analogue on a switchback block/period schedule, and this dataset has none.",
                ),
            ),
            refusals=(("from_moments", "source.moments.covariate_unavailable"),),
        ),
        "observational_aipw_dml_covariate_adjusted": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_moments",
                    "SOURCE: a moments cube holds no per-unit rows to attach a covariate to.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: a per-unit adjustment set has no analogue on a switchback block/period schedule, and this dataset has none.",
                ),
            ),
            refusals=(("from_moments", "source.moments.covariate_unavailable"),),
        ),
        "observational_categorical_covariate_adjusted": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_moments",
                    "SOURCE: a moments cube holds no per-unit rows to attach a covariate to.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: a per-unit adjustment set has no analogue on a switchback block/period schedule, and this dataset has none.",
                ),
            ),
            refusals=(("from_moments", "source.moments.covariate_unavailable"),),
        ),
        "observational_categorical_covariate_missing_levels": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_moments",
                    "SOURCE: a moments cube holds no per-unit rows to attach a covariate to.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: a per-unit adjustment set has no analogue on a switchback block/period schedule, and this dataset has none.",
                ),
                (
                    "from_definitions",
                    "SOURCE: the definitions grammar declares no missing-covariate policy, so this path resolves the default missing='refuse' and must name the NULL level.",
                ),
                (
                    "from_unit_day_artifact",
                    "SOURCE: the definitions grammar declares no missing-covariate policy, so this path resolves the default missing='refuse' and must name the NULL level.",
                ),
            ),
            refusals=(
                ("from_moments", "source.moments.covariate_unavailable"),
                ("from_definitions", "adjust.identification.missing_covariates"),
                ("from_unit_day_artifact", "adjust.identification.missing_covariates"),
            ),
        ),
        "observational_default_with_unadjusted_sensitivity": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_moments",
                    "SOURCE: a moments cube holds no per-unit rows to attach a covariate to.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: a per-unit adjustment set has no analogue on a switchback block/period schedule, and this dataset has none.",
                ),
            ),
            refusals=(("from_moments", "source.moments.covariate_unavailable"),),
        ),
        "observational_iptw_conversion_diagnostics": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_moments",
                    "SOURCE: a moments cube holds no per-unit rows to attach a covariate to.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: a per-unit adjustment set has no analogue on a switchback block/period schedule, and this dataset has none.",
                ),
            ),
            refusals=(("from_moments", "source.moments.covariate_unavailable"),),
        ),
        "observational_iptw_covariate_adjusted_ate": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_moments",
                    "SOURCE: a moments cube holds no per-unit rows to attach a covariate to.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: a per-unit adjustment set has no analogue on a switchback block/period schedule, and this dataset has none.",
                ),
            ),
            refusals=(("from_moments", "source.moments.covariate_unavailable"),),
        ),
        "observational_multiplicity_primary_secondary_guardrail": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_moments",
                    "SOURCE: a moments cube holds no per-unit rows to attach a covariate to.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: a per-unit adjustment set has no analogue on a switchback block/period schedule, and this dataset has none.",
                ),
            ),
            refusals=(("from_moments", "source.moments.covariate_unavailable"),),
        ),
        "observational_quantile_refused_at_the_readout_seam": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_definitions",
                    "CONSTRUCTION: an observational design has no quantile estimator (readout.observational.quantile)",
                ),
                (
                    "from_unit_day_artifact",
                    "CONSTRUCTION: no quantile estimator under an observational design (readout.observational.quantile)",
                ),
                (
                    "from_unit_summary",
                    "CONSTRUCTION: no quantile estimator under an observational design (readout.observational.quantile)",
                ),
                (
                    "from_unit_panel",
                    "CONSTRUCTION: no quantile estimator under an observational design (readout.observational.quantile)",
                ),
                (
                    "from_moments",
                    "CONSTRUCTION: the quantile declaration over scalar moments must refuse its absent observational estimator before using those moments (readout.observational.quantile).",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: no design= parameter, and a quantile metric is refused at the switchback metric-type gate (source.frame.switchback.metric).",
                ),
            ),
            refusals=(
                ("from_definitions", "readout.observational.quantile"),
                ("from_unit_day_artifact", "readout.observational.quantile"),
                ("from_unit_summary", "readout.observational.quantile"),
                ("from_unit_panel", "readout.observational.quantile"),
                ("from_moments", "readout.observational.quantile"),
            ),
        ),
        "portable_subset_with_winsorized_sibling": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "quantile_tied_rounded_outcomes": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_moments",
                    "SOURCE: a quantile has no moments representation -- export() itself refuses before a from_moments cube could ever exist.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: switchback metrics must be type='mean' or type='conversion'; quantile is refused before any frame is read (confirmed live: source.frame.switchback.metric).",
                ),
            ),
            refusals=(("from_moments", "source.frame.quantile_no_moments"),),
        ),
        "sequential-mixed-quantile-catalog-checkpoint": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: switchback accepts only mean/conversion metrics and refuses a quantile catalog (source.frame.switchback.metric); it also has no registered sequential construction. This dataset has no switchback schedule.",
                ),
            ),
        ),
        "sequential_asymptotic_mean_revenue": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "sequential_automatic_breakout_continuous_bonferroni": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_definitions",
                    "SOURCE: given the same segmented registration as from_unit_summary, a relational source (validate_relational_capture, shared by native_source.py and artifact_reader.py) refuses that segmented sequential roster outright -- 'relational segment capture needs a registered immutable property relation; use finalized joint records or frame segments'.",
                ),
                (
                    "from_unit_day_artifact",
                    "SOURCE: publish/adopt re-derives the experiment's declared plan's own automatic registration -- segmented from the predeclared family, so the relational capture check above refuses it for the same reason.",
                ),
                (
                    "from_moments",
                    "SOURCE: MomentsSource declares no breakout catalog at all -- a replayed checkpoint carries the segmented registration intact (export survives it), but run_breakout still refuses because a moments cube has no breakout dimension to serve.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
            refusals=(
                ("from_definitions", "sequential.route.unsupported"),
                ("from_unit_day_artifact", "sequential.route.unsupported"),
                ("from_moments", "readout.source.dimension"),
            ),
        ),
        "sequential_automatic_breakout_discrete_bh": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_definitions",
                    "SOURCE: given the same segmented registration as from_unit_summary, a relational source (validate_relational_capture, shared by native_source.py and artifact_reader.py) refuses that segmented sequential roster outright -- 'relational segment capture needs a registered immutable property relation; use finalized joint records or frame segments'.",
                ),
                (
                    "from_unit_day_artifact",
                    "SOURCE: publish/adopt re-derives the experiment's declared plan's own automatic registration -- segmented from the predeclared family, so the relational capture check above refuses it for the same reason.",
                ),
                (
                    "from_moments",
                    "SOURCE: MomentsSource declares no breakout catalog at all -- a replayed checkpoint carries the segmented registration intact (export survives it), but run_breakout still refuses because a moments cube has no breakout dimension to serve.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
            refusals=(
                ("from_definitions", "sequential.route.unsupported"),
                ("from_unit_day_artifact", "sequential.route.unsupported"),
                ("from_moments", "readout.source.dimension"),
            ),
        ),
        "sequential_automatic_multi_arm_exact": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "sequential_boolean_segments_automatic_asymptotic_mean": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_definitions",
                    "SOURCE: given the same segmented registration as from_unit_summary, a relational source (validate_relational_capture, shared by native_source.py and artifact_reader.py) refuses that segmented sequential roster outright -- 'relational segment capture needs a registered immutable property relation; use finalized joint records or frame segments'.",
                ),
                (
                    "from_unit_day_artifact",
                    "SOURCE: publish/adopt re-derives the experiment's declared plan's own automatic registration -- segmented from the predeclared family, so the relational capture check above refuses it for the same reason.",
                ),
                (
                    "from_moments",
                    "SOURCE: MomentsSource declares no breakout catalog at all -- a replayed checkpoint carries the segmented registration intact (export survives it), but run_breakout still refuses because a moments cube has no breakout dimension to serve.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
            refusals=(
                ("from_definitions", "sequential.route.unsupported"),
                ("from_unit_day_artifact", "sequential.route.unsupported"),
                ("from_moments", "readout.source.dimension"),
            ),
        ),
        "sequential_boolean_segments_automatic_exact_bernoulli": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_definitions",
                    "SOURCE: given the same segmented registration as from_unit_summary, a relational source (validate_relational_capture, shared by native_source.py and artifact_reader.py) refuses that segmented sequential roster outright -- 'relational segment capture needs a registered immutable property relation; use finalized joint records or frame segments'.",
                ),
                (
                    "from_unit_day_artifact",
                    "SOURCE: publish/adopt re-derives the experiment's declared plan's own automatic registration -- segmented from the predeclared family, so the relational capture check above refuses it for the same reason.",
                ),
                (
                    "from_moments",
                    "SOURCE: MomentsSource declares no breakout catalog at all -- a replayed checkpoint carries the segmented registration intact (export survives it), but run_breakout still refuses because a moments cube has no breakout dimension to serve.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
            refusals=(
                ("from_definitions", "sequential.route.unsupported"),
                ("from_unit_day_artifact", "sequential.route.unsupported"),
                ("from_moments", "readout.source.dimension"),
            ),
        ),
        "sequential_boolean_segments_registered_asymptotic_mean": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_definitions",
                    "SOURCE: given the same segmented registration as from_unit_summary, a relational source (validate_relational_capture, shared by native_source.py and artifact_reader.py) refuses that segmented sequential roster outright -- 'relational segment capture needs a registered immutable property relation; use finalized joint records or frame segments'.",
                ),
                (
                    "from_unit_day_artifact",
                    "SOURCE: publish/adopt always re-derives an automatic (unsegmented) registration from the experiment's declared plan -- a caller-built segmented registration never reaches it -- and the relational capture check above would refuse it regardless.",
                ),
                (
                    "from_moments",
                    "SOURCE: MomentsSource declares no breakout catalog at all -- a replayed checkpoint carries the segmented registration intact (export survives it), but run_breakout still refuses because a moments cube has no breakout dimension to serve.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
            refusals=(
                ("from_definitions", "sequential.route.unsupported"),
                ("from_unit_day_artifact", "sequential.route.unsupported"),
                ("from_moments", "readout.source.dimension"),
            ),
        ),
        "sequential_boolean_segments_registered_exact_bernoulli": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_definitions",
                    "SOURCE: given the same segmented registration as from_unit_summary, a relational source (validate_relational_capture, shared by native_source.py and artifact_reader.py) refuses that segmented sequential roster outright -- 'relational segment capture needs a registered immutable property relation; use finalized joint records or frame segments'.",
                ),
                (
                    "from_unit_day_artifact",
                    "SOURCE: publish/adopt always re-derives an automatic (unsegmented) registration from the experiment's declared plan -- a caller-built segmented registration never reaches it -- and the relational capture check above would refuse it regardless.",
                ),
                (
                    "from_moments",
                    "SOURCE: MomentsSource declares no breakout catalog at all -- a replayed checkpoint carries the segmented registration intact (export survives it), but run_breakout still refuses because a moments cube has no breakout dimension to serve.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
            refusals=(
                ("from_definitions", "sequential.route.unsupported"),
                ("from_unit_day_artifact", "sequential.route.unsupported"),
                ("from_moments", "readout.source.dimension"),
            ),
        ),
        "sequential_composed_family_ebh": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "sequential_composed_itt_and_uptake": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "sequential_encouragement_optional_exclusion": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
        ),
        "sequential_explicit_registration_native_artifact_identity": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
                (
                    "from_unit_summary",
                    "SOURCE: a registration declared against the native source mapping is bound to that source's identity and is never adopted by a frame source",
                ),
                (
                    "from_unit_panel",
                    "SOURCE: a registration declared against the native source mapping is bound to that source's identity and is never adopted by a frame source",
                ),
            ),
            refusals=(
                ("from_unit_summary", "sequential.source.invalid"),
                ("from_unit_panel", "sequential.source.invalid"),
            ),
        ),
        "sequential_registered_breakout_continuous_bonferroni": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_definitions",
                    "SOURCE: given the same segmented registration as from_unit_summary, a relational source (validate_relational_capture, shared by native_source.py and artifact_reader.py) refuses that segmented sequential roster outright -- 'relational segment capture needs a registered immutable property relation; use finalized joint records or frame segments'.",
                ),
                (
                    "from_unit_day_artifact",
                    "SOURCE: publish/adopt always re-derives an automatic (unsegmented) registration from the experiment's declared plan -- a caller-built segmented registration never reaches it -- and the relational capture check above would refuse it regardless.",
                ),
                (
                    "from_moments",
                    "SOURCE: MomentsSource declares no breakout catalog at all -- a replayed checkpoint carries the segmented registration intact (export survives it), but run_breakout still refuses because a moments cube has no breakout dimension to serve.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
            refusals=(
                ("from_definitions", "sequential.route.unsupported"),
                ("from_unit_day_artifact", "sequential.route.unsupported"),
                ("from_moments", "readout.source.dimension"),
            ),
        ),
        "sequential_registered_breakout_discrete_bh": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_definitions",
                    "SOURCE: given the same segmented registration as from_unit_summary, a relational source (validate_relational_capture, shared by native_source.py and artifact_reader.py) refuses that segmented sequential roster outright -- 'relational segment capture needs a registered immutable property relation; use finalized joint records or frame segments'.",
                ),
                (
                    "from_unit_day_artifact",
                    "SOURCE: publish/adopt always re-derives an automatic (unsegmented) registration from the experiment's declared plan -- a caller-built segmented registration never reaches it -- and the relational capture check above would refuse it regardless.",
                ),
                (
                    "from_moments",
                    "SOURCE: MomentsSource declares no breakout catalog at all -- a replayed checkpoint carries the segmented registration intact (export survives it), but run_breakout still refuses because a moments cube has no breakout dimension to serve.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
            ),
            refusals=(
                ("from_definitions", "sequential.route.unsupported"),
                ("from_unit_day_artifact", "sequential.route.unsupported"),
                ("from_moments", "readout.source.dimension"),
            ),
        ),
        "sequential_registered_registration_native_artifact_identity": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
                (
                    "from_unit_summary",
                    "SOURCE: a registration declared against the native source mapping is bound to that source's identity and is never adopted by a frame source",
                ),
                (
                    "from_unit_panel",
                    "SOURCE: a registration declared against the native source mapping is bound to that source's identity and is never adopted by a frame source",
                ),
            ),
            refusals=(
                ("from_unit_summary", "sequential.source.invalid"),
                ("from_unit_panel", "sequential.source.invalid"),
            ),
        ),
        "triggered_asof": ParityDeclaration(
            attempted=("from_definitions", "from_unit_day_artifact"),
            waive=(
                (
                    "from_unit_summary",
                    "SOURCE: unit summaries carry no trigger declaration or per-unit trigger anchors.",
                ),
                (
                    "from_unit_panel",
                    "SOURCE: unit panels carry no trigger declaration or per-unit trigger anchors.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: switchback contrasts carry no triggered arm day-axis evidence.",
                ),
                ("from_moments", "SOURCE: portable moments carry no per-unit trigger anchors."),
            ),
        ),
        "triggered_asof_lift": ParityDeclaration(
            attempted=("from_definitions", "from_unit_day_artifact"),
            waive=(
                (
                    "from_unit_summary",
                    "SOURCE: unit summaries carry no trigger declaration or per-unit trigger anchors.",
                ),
                (
                    "from_unit_panel",
                    "SOURCE: unit panels carry no trigger declaration or per-unit trigger anchors.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: switchback contrasts carry no triggered arm day-axis evidence.",
                ),
                ("from_moments", "SOURCE: portable moments carry no per-unit trigger anchors."),
            ),
        ),
        "triggered_daily": ParityDeclaration(
            attempted=("from_definitions", "from_unit_day_artifact"),
            waive=(
                (
                    "from_unit_summary",
                    "SOURCE: unit summaries carry no trigger declaration or per-unit trigger anchors.",
                ),
                (
                    "from_unit_panel",
                    "SOURCE: unit panels carry no trigger declaration or per-unit trigger anchors.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: switchback contrasts carry no triggered arm day-axis evidence.",
                ),
                ("from_moments", "SOURCE: portable moments carry no per-unit trigger anchors."),
            ),
        ),
        "triggered_daily_lift": ParityDeclaration(
            attempted=("from_definitions", "from_unit_day_artifact"),
            waive=(
                (
                    "from_unit_summary",
                    "SOURCE: unit summaries carry no trigger declaration or per-unit trigger anchors.",
                ),
                (
                    "from_unit_panel",
                    "SOURCE: unit panels carry no trigger declaration or per-unit trigger anchors.",
                ),
                (
                    "from_switchback_panel",
                    "SOURCE: switchback contrasts carry no triggered arm day-axis evidence.",
                ),
                ("from_moments", "SOURCE: portable moments carry no per-unit trigger anchors."),
            ),
        ),
        "triggered_encouragement_compliance_asof": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
                "from_switchback_panel",
            ),
            waive=(
                (
                    "from_unit_summary",
                    "SOURCE: unit summaries have no trigger declaration or trigger anchors.",
                ),
                (
                    "from_unit_panel",
                    "SOURCE: unit panels have no trigger declaration or trigger anchors.",
                ),
                ("from_moments", "SOURCE: portable moments have no per-unit trigger anchors."),
                (
                    "from_switchback_panel",
                    "SOURCE: switchback contrasts have no triggered assignment population.",
                ),
            ),
            refusals=(
                ("from_unit_summary", "facade.analysis.trigger_unsupported"),
                ("from_unit_panel", "facade.analysis.trigger_unsupported"),
                ("from_moments", "facade.analysis.trigger_unsupported"),
                ("from_switchback_panel", "facade.analysis.trigger_unsupported"),
            ),
        ),
        "window_spelling_offset_vs_offset": ParityDeclaration(
            attempted=("from_definitions", "from_unit_day_artifact"),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
                (
                    "from_unit_summary",
                    "SOURCE: from_unit_summary has no experiment-window lever -- the day boundary is applied by the caller pre-bucketing ds/exposure_date (see tests/test_frame_window_parity.py), so the window spelling is not an input.",
                ),
                (
                    "from_unit_panel",
                    "SOURCE: from_unit_panel has no experiment-window lever -- the day boundary is applied by the caller pre-bucketing ds/exposure_date (see tests/test_frame_window_parity.py), so the window spelling is not an input.",
                ),
                (
                    "from_moments",
                    "SOURCE: from_moments replays an exported moment cube; it carries no declared experiment window to spell.",
                ),
            ),
        ),
        "window_spelling_zulu_vs_offset": ParityDeclaration(
            attempted=("from_definitions", "from_unit_day_artifact"),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: from_switchback_panel needs a switchback-shaped frame (unit/cycle/period/step columns) and only accepts a Randomized identification -- no case in this module builds that shape.",
                ),
                (
                    "from_unit_summary",
                    "SOURCE: from_unit_summary has no experiment-window lever -- the day boundary is applied by the caller pre-bucketing ds/exposure_date (see tests/test_frame_window_parity.py), so the window spelling is not an input.",
                ),
                (
                    "from_unit_panel",
                    "SOURCE: from_unit_panel has no experiment-window lever -- the day boundary is applied by the caller pre-bucketing ds/exposure_date (see tests/test_frame_window_parity.py), so the window spelling is not an input.",
                ),
                (
                    "from_moments",
                    "SOURCE: from_moments replays an exported moment cube; it carries no declared experiment window to spell.",
                ),
            ),
        ),
        "winsor-percentile-size-routed-analytic": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: switchback aggregation does not provide percentile-winsor independent-unit inference",
                ),
                (
                    "from_unit_panel",
                    "SOURCE: panel collapse does not retain exact pre-winsor unit outcomes",
                ),
                (
                    "from_moments",
                    "SOURCE: portable moments do not retain exact pre-winsor unit outcomes",
                ),
            ),
            refusals=(
                ("from_unit_panel", "estimation.winsor.raw_state_required"),
                ("from_moments", "estimation.winsor.raw_state_required"),
            ),
        ),
        "winsor-percentile-three-arm": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: switchback aggregation does not provide percentile-winsor independent-unit inference",
                ),
                (
                    "from_unit_panel",
                    "SOURCE: panel collapse does not retain exact pre-winsor unit outcomes",
                ),
                (
                    "from_moments",
                    "SOURCE: portable moments do not retain exact pre-winsor unit outcomes",
                ),
            ),
            refusals=(
                ("from_unit_panel", "estimation.winsor.raw_state_required"),
                ("from_moments", "estimation.winsor.raw_state_required"),
            ),
        ),
        "winsor-percentile-zero-inclusive-2-arm": ParityDeclaration(
            attempted=(
                "from_definitions",
                "from_unit_day_artifact",
                "from_unit_summary",
                "from_unit_panel",
                "from_moments",
            ),
            waive=(
                (
                    "from_switchback_panel",
                    "SOURCE: switchback aggregation does not provide percentile-winsor independent-unit inference",
                ),
                (
                    "from_unit_panel",
                    "SOURCE: panel collapse does not retain exact pre-winsor unit outcomes",
                ),
                (
                    "from_moments",
                    "SOURCE: portable moments do not retain exact pre-winsor unit outcomes",
                ),
            ),
            refusals=(
                ("from_unit_panel", "estimation.winsor.raw_state_required"),
                ("from_moments", "estimation.winsor.raw_state_required"),
            ),
        ),
    }
)


_merged_declarations = dict(PARITY_DECLARATIONS)
_merged_declarations.update(
    {
        "audit-retention-breakout-cohorts-single-segment": PARITY_DECLARATIONS[
            "audit-retention-breakout-cohorts"
        ],
        "winsor-percentile-zero-inclusive-3-arm": PARITY_DECLARATIONS[
            "winsor-percentile-zero-inclusive-2-arm"
        ],
    }
)
for _consumer_declarations in (_SCALAR_DECLARATIONS, _AUXILIARY_DECLARATIONS):
    _duplicate_ids = _merged_declarations.keys() & _consumer_declarations.keys()
    if _duplicate_ids:
        raise ValueError(f"duplicate parity declaration IDs: {sorted(_duplicate_ids)}")
    _merged_declarations.update(_consumer_declarations)
PARITY_DECLARATIONS = MappingProxyType(_merged_declarations)
