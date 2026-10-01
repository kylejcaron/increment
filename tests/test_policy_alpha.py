"""Contract tests for the single per-cell alpha resolver.

Each case pins one row of the policy enumeration: the display alpha a cell gets
as a function of its role, its arms, and the plan's own segment policy. No
warehouse, no estimation -- the plan is compiled in-process and the resolver is
pure. The segment correction is never supplied by the caller; it is read from
the compiled plan, so a cell can only be levelled by a policy the plan fixed.
"""

import pytest

from increment._policy_alpha import resolve_cell_alpha
from increment.compatibility import _conservative_divide
from increment.plan import compile_decision_plan
from increment.semantics.models import AnalysisPlan, MeanMetric, MultiplicitySpec

ALPHA = 0.1
BONFERRONI = MultiplicitySpec(correction="bonferroni")
BH = MultiplicitySpec(correction="bh")


def _mean(name: str, **kwargs) -> MeanMetric:
    return MeanMetric(name=name, entity="user", fact=name, **kwargs)


def _declared_plan(view_multiplicity: MultiplicitySpec | None = None):
    """3 primaries, 1 secondary, 1 guardrail, 1 unassigned."""
    metrics = [
        _mean("p0"),
        _mean("p1"),
        _mean("p2"),
        _mean("s0"),
        _mean("g0", preferred_direction="increase"),
        _mean("u0"),
    ]
    plan = AnalysisPlan(
        alpha=ALPHA,
        primary=("p0", "p1", "p2"),
        secondaries=("s0",),
        guardrails=("g0",),
        view_multiplicity=view_multiplicity,
    )
    return compile_decision_plan(plan, metrics, path="warehouse")


def _undeclared_plan():
    metrics = [_mean("p0"), _mean("s0")]
    return compile_decision_plan(None, metrics, path="warehouse")


def test_primary_splits_across_primaries_and_arms():
    plan = _declared_plan()
    got = resolve_cell_alpha(plan, plan.procedures["p0"], n_arms=2)
    assert got == _conservative_divide(ALPHA, 3 * 2)


def test_primary_with_no_arm_falls_back_to_compiled_alpha():
    plan = _declared_plan()
    proc = plan.procedures["p0"]
    assert resolve_cell_alpha(plan, proc, n_arms=0) == proc.alpha


def test_secondary_uses_nominal_plan_alpha():
    plan = _declared_plan()
    assert resolve_cell_alpha(plan, plan.procedures["s0"], n_arms=1) == ALPHA


@pytest.mark.parametrize("name", ["g0", "u0"])
def test_guardrail_and_unassigned_use_compiled_alpha(name):
    plan = _declared_plan()
    proc = plan.procedures[name]
    assert resolve_cell_alpha(plan, proc, n_arms=1) == proc.alpha


@pytest.mark.parametrize("name", ["p0", "s0"])
def test_undeclared_plan_uses_compiled_default_for_every_role(name):
    plan = _undeclared_plan()
    assert plan.declared is False
    proc = plan.procedures[name]
    assert resolve_cell_alpha(plan, proc, n_arms=2) == proc.alpha


def test_segmented_primary_under_bonferroni_carries_the_segment_divisor():
    plan = _declared_plan(BONFERRONI)
    assert plan.view_policies.asof.correction == "bonferroni"
    got = resolve_cell_alpha(plan, plan.procedures["p0"], n_arms=2, n_segments=4, view="asof")
    assert got == _conservative_divide(ALPHA, 3 * 2 * 4)


def test_segmented_primary_is_one_combined_division_not_nested():
    # Conservative division does not associate, so the resolver takes the plan
    # and divides once by the combined denominator; nesting yields another float.
    plan = _declared_plan(BONFERRONI)
    got = resolve_cell_alpha(plan, plan.procedures["p0"], n_arms=7, n_segments=3, view="asof")
    combined = _conservative_divide(ALPHA, 3 * 7 * 3)
    nested = _conservative_divide(_conservative_divide(ALPHA, 3 * 7), 3)
    assert got == combined
    assert combined != nested


@pytest.mark.parametrize(
    ("view_multiplicity", "view"),
    [
        (BONFERRONI, None),  # unsegmented readout: no segment axis at all
        (None, "asof"),  # plan's as-of family defaults to uncorrected
        (BH, "asof"),  # an FDR family corrects by selection, not by division
    ],
)
def test_segments_do_not_split_a_primary_outside_a_bonferroni_family(view_multiplicity, view):
    plan = _declared_plan(view_multiplicity)
    got = resolve_cell_alpha(plan, plan.procedures["p0"], n_arms=2, n_segments=5, view=view)
    assert got == _conservative_divide(ALPHA, 3 * 2)


def test_caller_cannot_force_a_segment_split_the_plan_did_not_declare():
    # Even asking for the as-of view with many segments yields no split when the
    # plan's own as-of family is uncorrected -- the correction is the plan's, not
    # the caller's.
    plan = _declared_plan(view_multiplicity=None)
    assert plan.view_policies.asof.correction == "none"
    got = resolve_cell_alpha(plan, plan.procedures["p0"], n_arms=2, n_segments=9, view="asof")
    assert got == _conservative_divide(ALPHA, 3 * 2)


def test_secondary_carries_the_segment_divisor():
    plan = _declared_plan(BONFERRONI)
    got = resolve_cell_alpha(plan, plan.procedures["s0"], n_arms=1, n_segments=4, view="asof")
    assert got == _conservative_divide(ALPHA, 4)


@pytest.mark.parametrize("name", ["g0", "u0"])
def test_non_primary_carries_the_segment_divisor(name):
    # The Bonferroni segment correction is family-wide: it applies to every
    # role's base level, not only primaries. Primaries additionally carry the
    # arm/primary split (see the combined-division test above).
    plan = _declared_plan(BONFERRONI)
    proc = plan.procedures[name]
    got = resolve_cell_alpha(plan, proc, n_arms=1, n_segments=4, view="asof")
    assert got == _conservative_divide(proc.alpha, 4)
