from __future__ import annotations

from typing import Literal

from increment.estimation.diagnostics import (
    NotApplicable,
    SRMResult,
    complete_srm_support,
    resolve_srm_expected,
    sample_ratio_mismatch,
)
from increment.readouts._common import _raise
from increment.sources import MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL, MomentSource


def srm(
    src: MomentSource,
    *,
    expected: dict[str, float] | None = None,
    alpha: float = 0.001,
    inference: Literal["always_valid", "fixed"] = "always_valid",
) -> SRMResult | NotApplicable:
    """Sample-ratio-mismatch check on the source's per-group counts.

    Design is read off *src* (``src.design``) rather than passed in - a
    `MomentSource` owns it as construction state, per every other readout
    entry point.

    The default is an anytime-valid check for cumulative prefixes under a
    known allocation with the same conditional arm probabilities at every
    assignment. Independent categorical assignment suffices; blocked,
    adaptive, dependent, quota, exact-balance, and without-replacement
    protocols do not. Pass ``inference="fixed"`` for one predeclared
    Pearson look.
    The source must also declare ``allocation_scheme="independent"``; a
    missing or unsupported scheme returns ``NotApplicable`` before counts are read.

    Applies under `Encouragement` too - encouragement assignment IS
    randomized, only uptake is not; an SRM test on the assignment counts
    is exactly as meaningful as under `Randomized`. Not applicable under
    an `Observational` design: an SRM test presumes a target randomized
    allocation to compare observed counts against, which does not exist
    for a non-randomized comparison.

    When the source declares a randomization cluster, the chi-square runs
    over DISTINCT CLUSTER counts per arm (`SRMResult.grain == "cluster"`).
    Independently assigned clusters with known, constant conditional arm
    probabilities satisfy the anytime-valid contract. Per-arm unit counts ride
    along on `SRMResult.unit_counts` as context: unit imbalance under a clustered
    design is cluster-SIZE imbalance, which the randomizer never controlled.

    For ``inference="always_valid"``, pass ``expected`` or declare
    ``design.allocation``: its support is static and known before the
    cumulative prefix is observed. Fixed mode with neither expected nor
    design allocation uses equal observed-arm shares and returns
    ``log_e_value=None``; the fixed Pearson p-value controls ``is_srm``.

    ``alpha`` (default 0.001) is a diagnostic significance threshold for
    this randomization-integrity check, not part of the declared
    `AnalysisPlan` - unlike every other ``alpha=`` this effort removed
    from `run`/`breakout`/`asof_lift`/`daily`, it stays a call-time
    parameter deliberately: an SRM check answers "did the randomizer
    misbehave", a question with no plan-declared role/alpha to read.
    """
    design = src.context.design
    if design is None:
        _raise("readout.srm_source_declared")
    if design.mechanism not in ("randomized", "encouragement"):
        return NotApplicable(
            check="srm",
            reason=(
                "assignment was not randomized; a sample-ratio test presumes "
                "a target allocation to mismatch"
            ),
        )
    if inference == "always_valid":
        scheme = getattr(design, "allocation_scheme", None)
        if scheme != "independent":
            code = (
                "integrity.allocation_scheme_missing"
                if scheme is None
                else "integrity.allocation_scheme_unsupported"
            )
            return NotApplicable(
                check="srm",
                reason=f"{code}: always-valid SRM requires independent assignment",
            )
    expected = resolve_srm_expected(
        expected,
        allocation=getattr(design, "allocation", None),
        inference=inference,
    )
    counts = complete_srm_support(src.unit_counts(), expected=expected)
    if src.context.cluster is None:
        return sample_ratio_mismatch(counts, expected=expected, alpha=alpha, inference=inference)
    # Accounting keys stay in the tested dict - sample_ratio_mismatch lifts
    # them onto their own fields with no degree of freedom.
    accounting = {
        key: counts.pop(key) for key in (UNASSIGNED_LABEL, MIXED_ASSIGNMENT_LABEL) if key in counts
    }
    return sample_ratio_mismatch(
        {
            **complete_srm_support(src.cluster_counts(), expected=expected),
            **accounting,
        },
        expected=expected,
        alpha=alpha,
        inference=inference,
        grain="cluster",
        unit_counts=counts,
    )
