"""Compose each cell's display alpha from its compiled procedure.

Centralizing divisor composition keeps whole-window, as-of, and daily
readouts numerically consistent. Conservative division is not associative,
so primary cells divide the plan alpha once by the combined
``primary_count * arm_count * K`` product.

The composition first resolves a family-wide segment divisor ``K``: under the
readout's compiled Bonferroni segment family, ``K`` is the segment count;
otherwise it is one. Then, for a declared plan:

- primary, with arms present: one conservative division of the plan alpha by
  ``primary_count * arm_count * K``.
- primary, with no non-control arm present: the procedure's compiled alpha
  divided by ``K``.
- secondary: the plan's nominal alpha divided by ``K``; family selection still
  happens downstream.
- guardrail or unassigned: the procedure's compiled alpha divided by ``K``.

An undeclared plan uses each procedure's compiled default divided by ``K``.
When ``K == 1`` the helper returns the base float unchanged, so unsegmented
readouts are bit-identical to no segment correction.

The segment correction is read from the compiled plan's own view policies,
keyed by a typed view name -- never a correction string the caller supplies --
so a cell can only be levelled by a policy the plan itself fixed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from increment.compatibility import _conservative_divide

if TYPE_CHECKING:
    from increment.decision import CompiledDecisionPlan, DecisionProcedure


def resolve_cell_alpha(
    plan: CompiledDecisionPlan,
    procedure: DecisionProcedure,
    *,
    n_arms: int,
    n_segments: int = 1,
    view: Literal["asof", "breakout"] | None = None,
    mechanism: str | None = None,
) -> float:
    """Display alpha for one metric cell.

    ``n_arms`` is the number of non-control arms present in this cell's rows and
    ``n_segments`` the number of segments present in this readout (1 when
    unsegmented). ``view`` names which segmented readout this is, so the segment
    correction can be looked up from the plan's own compiled view policies;
    whole-window readouts are unsegmented and pass ``None``. ``mechanism``
    disambiguates the breakout family (randomized vs encouragement).
    """
    segments = _segment_divisor(plan, view, mechanism, n_segments)
    if not plan.declared:
        return _with_segments(procedure.alpha, segments)
    role = procedure.role
    if role == "primary":
        if n_arms < 1:
            return _with_segments(procedure.alpha, segments)
        n_primaries = sum(p.role == "primary" for p in plan.procedures.values())
        return _conservative_divide(plan.alpha, n_primaries * n_arms * segments)
    if role == "secondary":
        return _with_segments(plan.alpha, segments)
    return _with_segments(procedure.alpha, segments)


def _with_segments(base: float, segments: int) -> float:
    """Apply the family-wide segment divisor. Unsegmented (``segments == 1``) is
    a no-op, so an unsegmented readout is bit-identical to no correction."""
    return base if segments == 1 else _conservative_divide(base, segments)


def _segment_divisor(
    plan: CompiledDecisionPlan,
    view: Literal["asof", "breakout"] | None,
    mechanism: str | None,
    n_segments: int,
) -> int:
    """Segments split the alpha only under the plan's own Bonferroni family.

    The correction comes from ``plan.view_policies`` for this view, not from the
    caller: an FDR family (BH/e-BH) controls its own error rate through
    selection, and ``none`` accepts the cross-segment inflation, so neither
    divides the per-cell level here.
    """
    if view is None or n_segments <= 1:
        return 1
    policy = plan.view_policies.for_view(view, mechanism=mechanism, segmented=True)
    return max(n_segments, 1) if policy.correction == "bonferroni" else 1
