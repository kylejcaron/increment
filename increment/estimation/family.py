"""Multiple-testing family procedures: FDR control across a batch of
tests (BH, e-BH) and the interval-level correction after a data-dependent
selection (FCR). Certified sequential selection consumes raw-likelihood log
certificates on a retained registered roster.

``select_family`` is the shared typed-evidence selection procedure used by
``readouts.run()`` and ``run_breakout()``.  Presentation rows identify the
family cells and receive verdict metadata only after selection succeeds;
selection itself consumes one typed evidence value per compiled hypothesis.

How multiplicity axes compose
-----------------------------
A test cell is a ``(metric, arm, segment)`` triple, with any axis a given
read does not vary collapsed. Three axes can carry multiplicity, and the
rule for combining them is:

1. **Every correction declares its family.** A procedure states exactly
   which cells it covers -- the cross-product of the axes it spans -- and
   each corrected row records that on ``family_axes``, with the level on
   ``family_q`` and the realized cutoff on ``family_threshold``.

2. **One procedure per cell.** A cell is corrected by exactly one
   procedure. This holds structurally rather than by check: a metric is
   either primary or secondary, never both, and ``run_breakout``'s
   segment division and its BH family are exclusive branches on
   ``correction`` (``k`` is 1 unless ``correction == "bonferroni"``).

3. **Alpha splitting nests; procedures do not.** Dividing alpha down an
   axis hierarchy composes, because the shares sum to no more than alpha --
   which is why a primary's ``plan.alpha / n_primaries`` may be split again
   across its own arms and still hold FWER at alpha. Two procedures of
   DIFFERENT kinds over overlapping axes do not compose that way; pool
   the cells into one joint family instead, which is what
   ``correction="bh"`` does across metric x arm x segment in a single
   ``select_family`` call rather than correcting each axis in turn.

4. **An FDR level is not an FWER level.** A family selected at ``q``
   bounds the false DISCOVERY rate over that family; it says nothing
   about the familywise error rate at the nominal alpha, and the two are
   not nested. Concretely, BH's FCR cutoff ``R*q/m`` is bounded by ``q``
   and not by alpha, so it can exceed the nominal level -- see
   ``select_family`` for why that is capped rather than emitted.

A consequence worth stating: ``run()`` and ``breakout()`` deliberately
hold the same metric to different bars. ``run()`` is confirmatory and
splits alpha down metric then arm (FWER); ``breakout()`` is exploratory,
stamps every row ``role="exploratory"``, and corrects with an FDR family
at ``q`` over its own cells. Exploration should be more permissive than
confirmation, so the divergence is intended, not an inconsistency to
reconcile -- and each row's ``family_*`` fields say which bar it met.

``select_exploratory_family`` applies the same flat-family rule to finished rows: whole-window
and segment decision rows, uncorrected, are one BH family, and each selected interval is
reissued at the FCR level from the construction the row persists (see ``_fcr_reinterval``).

Pure functions - no I/O, no ibis, no warehouse query construction.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

from increment.compatibility import _conservative_ratio
from increment.errors import (
    CapabilityError,
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    raiser,
    refusals,
    refuse,
)
from increment.estimation._certified import log_interval
from increment.estimation.decision_types import (
    ArmHypothesisKey,
    DecisionComputation,
    DecisionFailure,
    EValueEvidence,
    FixedInference,
    PValueEvidence,
    SegmentHypothesisKey,
    exact_fraction,
)
from increment.estimation.sequential import AlwaysValid, AsymptoticMean, MixedFamily

if TYPE_CHECKING:
    from pydantic import BaseModel

    from increment.breakout.estimates import BreakoutEstimate
    from increment.decision import (
        HypothesisKey,
        TestEvidence,
    )
    from increment.estimation._fcr_reinterval import Construction
    from increment.estimation.results import LiftEstimate


def _row_labels(rows: Sequence[str]) -> str:
    return ", ".join(rows)


#: Shared by ``run_breakout(correction="bh")`` and the exploratory family: the same hazard
#: (no frequentist p-value to select on) carries the same code on both paths.
BH_EXCLUDES_PRIOR = RefusalSpec(
    "breakout.run_breakout_bh_excludes_prior",
    InvalidRequestError,
    template="an informative prior cannot enter a BH family: BH/e-BH selection needs frequentist p-values/e-values, and a posterior tail probability is neither",
)
_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.family.bh_select_q_finite": "q must be finite and in (0, 1], got {q!r}",
        "estimation.family.p_values_finite": "p_values[{i}] must be finite and in [0, 1], got {p!r}",
        "estimation.family.e_values_finite": "log_e_values[{i}] must be an exact lower log bound or tagged infinity, got {e!r}",
        "estimation.family.nominal_alpha_finite": "nominal_alpha must be finite and in (0, 1), got {nominal_alpha!r}",
        "estimation.family.exploratory_row": RefusalSpec(
            "estimation.family.exploratory_row",
            InvalidRequestError,
            lambda *, rows, **_: (
                f"exploratory family rows must be whole-window LiftEstimate or segment "
                f"BreakoutEstimate rows; {_row_labels(rows)} are not (a day-axis or other row "
                "type is not a hypothesis of one whole-window family)"
            ),
        ),
        "estimation.family.exploratory_pre_corrected": RefusalSpec(
            "estimation.family.exploratory_pre_corrected",
            InvalidRequestError,
            lambda *, rows, **_: (
                f"exploratory family rows must be uncorrected: {_row_labels(rows)} already carry "
                "a family correction, and a cell is corrected by exactly one procedure"
            ),
        ),
        "estimation.family.exploratory_non_decision": RefusalSpec(
            "estimation.family.exploratory_non_decision",
            InvalidRequestError,
            lambda *, rows, **_: (
                f"exploratory family rows must be decision rows: {_row_labels(rows)} are "
                "sensitivity rows, which are companions of a decision row, not hypotheses"
            ),
        ),
        "estimation.family.exploratory_sequential": RefusalSpec(
            "estimation.family.exploratory_sequential",
            UnsupportedRequestError,
            lambda *, rows, **_: (
                "exploratory family selection is not available for sequential inference: "
                f"{_row_labels(rows)} carry always-valid or registered intervals, and e-BH over "
                "arbitrary segment cells has no validity argument yet. Pass fixed-horizon rows"
            ),
        ),
        "estimation.family.exploratory_construction": RefusalSpec(
            "estimation.family.exploratory_construction",
            UnsupportedRequestError,
            lambda *, rows, constructions, **_: (
                "exploratory family selection cannot reissue these intervals at the FCR level "
                "without approximating them: "
                + "; ".join(
                    f"{row} ({kind})" for row, kind in zip(rows, constructions, strict=True)
                )
                + ". Leave these rows out of the family"
            ),
        ),
    },
)
_raise = raiser(_REFUSALS)


def decision_cells[K, E](
    cells: Sequence[tuple[K, E]],
) -> list[tuple[K, E]]:
    """Return only decision-role rows eligible for family selection.

    Sensitivity estimates are useful companion readouts, but they are not
    independent hypotheses and must never affect family size, selection, or
    FCR re-estimation. Objects predating role stamping are treated as
    decision rows for compatibility with the pure selector's callers.
    """
    return [
        (key, estimate)
        for key, estimate in cells
        if getattr(estimate, "method_role", "decision") == "decision"
    ]


def _bh_step_up(values: Sequence[float], q: float) -> tuple[list[int], float]:
    """Shared BH step-up core: largest ``k`` with ``value_(k) <= k*q/m``,
    selecting every index whose value is at or below that threshold (so
    ties at the boundary are all included, regardless of sort order).
    Returns ``([], 0.0)`` when no ``k`` qualifies. No input validation -
    ``bh_select`` and ``e_bh_select`` each validate their own domain.
    """
    m = len(values)
    if m == 0:
        return [], 0.0
    order = sorted(range(m), key=lambda i: values[i])
    for k in range(m, 0, -1):
        threshold = _conservative_ratio(q, k, m)
        if values[order[k - 1]] <= threshold:
            selected = [i for i in range(m) if values[i] <= threshold]
            return selected, threshold
    return [], 0.0


def _require_q(q: float) -> None:
    if not math.isfinite(q) or not (0.0 < q <= 1.0):
        _raise("estimation.family.bh_select_q_finite", q=q)


def bh_select(p_values: Sequence[float], q: float) -> tuple[list[int], float]:
    """BH step-up. Returns (sorted selected indices, realized threshold t = k*q/m;
    t = 0.0 when nothing selected). Refuses non-finite or out-of-[0,1] inputs."""
    _require_q(q)
    for i, p in enumerate(p_values):
        if not math.isfinite(p) or not (0.0 <= p <= 1.0):
            _raise("estimation.family.p_values_finite", i=i, p=p)
    return _bh_step_up(p_values, q)


def e_bh_select(log_e_values: Sequence[Fraction | float], q: Fraction | float) -> list[int]:
    """Log-domain e-BH with an upward-enclosed canonical threshold.

    Inputs are current/frozen lower log evidence; -infinity denotes abstention.
    Running maxima are not valid inputs. Retained missing cells remain in m.
    Step ``k`` compares against ``(-log_interval(q * k / m)).hi``, the same
    enclosure ``count_boundary`` builds at level ``q * k / m``, so evidence
    certified against a boundary at that level is compared with the identical
    rational threshold.
    """
    try:
        q = Fraction(q)
    except (ValueError, TypeError, OverflowError):
        _raise("estimation.family.bh_select_q_finite", q=q)
    if not 0 < q <= 1:
        _raise("estimation.family.bh_select_q_finite", q=str(q))
    for i, value in enumerate(log_e_values):
        if (
            isinstance(value, bool)
            or not isinstance(value, (Fraction, int, float))
            or isinstance(value, float)
            and math.isnan(value)
        ):
            _raise("estimation.family.e_values_finite", i=i, e=value)
    m = len(log_e_values)
    if not m:
        return []
    order = sorted(range(m), key=lambda i: log_e_values[i], reverse=True)
    for k in range(m, 0, -1):
        # Refuse a step threshold below float64 resolution before selection:
        # the realized threshold is later reported as a float and must not read 0.0.
        _conservative_ratio(float(q), k, m)
        threshold = (-log_interval(q * k / m)).hi
        if log_e_values[order[k - 1]] >= threshold:
            return sorted(i for i in order if log_e_values[i] >= threshold)
    return []


@dataclass(frozen=True)
class FamilyOutcome[K: Hashable]:
    """What a family procedure decided, and at what level.

    ``fcr_alpha`` is the level a caller should re-estimate SELECTED
    cells' intervals at, or ``None`` when no re-estimation should happen.
    ``realized_threshold`` is BH's own cutoff ``R*q/m`` before capping -
    kept separately because it is the number that explains the decision,
    while ``fcr_alpha`` is the number the interval was actually cut at.
    The two differ exactly when ``capped`` is True.

    ``canonical_method`` records, per key, which method's row supplied the
    decision statistic that drove selection ("unadjusted" wins when
    present, else the first-encountered method for that key -- see
    `select_family`). It is evidence provenance, not a gate among
    family-stamped decision rows: every such row for a selected compiled key
    carries the same family verdict, even when its own interval does not
    exclude the null. Sensitivity rows receive no family verdict or metadata.
    """

    selected: frozenset[K]
    fcr_alpha: Fraction | float | None
    q: float
    n_family: int
    realized_threshold: float | None
    capped: bool
    canonical_method: Mapping[K, str]
    guarantee: Literal["finite_sample", "asymptotic_sequential"] | None = None

    def __post_init__(self):
        object.__setattr__(self, "selected", frozenset(self.selected))
        object.__setattr__(self, "canonical_method", MappingProxyType(dict(self.canonical_method)))


def family_discovery[K: Hashable](outcome: FamilyOutcome[K], key: K) -> bool:
    """Return whether a family-stamped decision row's compiled key was selected."""
    return key in outcome.selected


def _unique_family_keys[K: Hashable](cells: Sequence[tuple[K, object]]) -> list[K]:
    """Return compiled family keys once, preserving compilation order."""
    keys: list[K] = []
    seen: set[K] = set()
    for key, _row in cells:
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


#: Codes for outcome-based cell degeneracy (zero variance, unusable SE, non-positive mean), not
#: invalid requests: a key failing only with these stays in a fixed-horizon family as a sure
#: non-rejection. Breakout keys the same hazards so run() and run_breakout() agree; design-based
#: few_units/no_control_arm are absent, and estimate_lift's non-positive means get additive rows.
_CONSERVATIVE_NONREJECTION_CODES = frozenset(
    {
        "estimation.engine.lift_guard",
        "breakout.nonpositive_mean",
        "breakout.zero_variance",
        "breakout.extreme_ratio",
    }
)


def _conservative_nonrejection_pvalue(
    key: HypothesisKey, failure: DecisionFailure
) -> PValueEvidence:
    """The most conservative (never-selectable) p-value evidence for a
    degenerate cell: p=1.0 can never fall below any BH/Bonferroni
    threshold (bh_select's ``values[order[k-1]] <= threshold`` test), so
    the cell is retained in the family's size but can never be selected.
    """
    method = failure.context.get("method")
    return PValueEvidence(
        key,
        method if isinstance(method, str) else "unknown",
        1.0,
        "degenerate_arm_conservative_nonrejection",
    )


def _require_complete_evidence[K: Hashable](
    keys: Sequence[K],
    computation: DecisionComputation,
    *,
    inference: AlwaysValid | object | None,
    expected_types: Mapping[K, type] | None = None,
) -> list[TestEvidence]:
    """Return one typed evidence value per key, or refuse the whole family.

    ``expected_types`` overrides the single-class dispatch below with a
    per-key type when the family mixes evidence kinds (a ``MixedFamily``
    roster carries both a Bernoulli e-value cell and an asymptotic
    confidence-set cell, so no one class fits every key).

    Outcome-based guard failures, including a sampled zero-relative-variance
    reason, remain in a fixed-horizon family as guaranteed non-rejections.
    Other unavailable evidence is still a hard failure; the reason check
    must not admit missing raw statistics or malformed requests.
    Registered sequential families
    (EValueEvidence/AsymptoticSequentialEvidence) and mixed families
    (``expected_types`` set) are unaffected: this hazard cannot occur
    on those paths.
    """
    from increment.estimation.decision_types import (
        AsymptoticSequentialEvidence,
        PValueEvidence,
    )

    evidence_map: dict[object, TestEvidence] = dict(computation.evidence.items())
    if expected_types is None:
        expected_type = (
            AsymptoticSequentialEvidence
            if isinstance(inference, AsymptoticMean)
            else EValueEvidence
            if isinstance(inference, AlwaysValid)
            else PValueEvidence
        )
        if expected_type is PValueEvidence:
            for key in keys:
                if key in evidence_map or key not in computation.failures:
                    continue
                failure = computation.failures[cast("HypothesisKey", key)]
                if failure.code in _CONSERVATIVE_NONREJECTION_CODES or (
                    failure.code == "evidence.p_value.unavailable"
                    and failure.context.get("reason") == "zero_relative_variance"
                ):
                    evidence_map[key] = _conservative_nonrejection_pvalue(
                        cast("HypothesisKey", key), failure
                    )
    failed = tuple(key for key in keys if key in computation.failures and key not in evidence_map)
    missing = tuple(
        key for key in keys if key not in evidence_map and key not in computation.failures
    )
    if expected_types is not None:
        invalid = tuple(
            key
            for key in keys
            if key in evidence_map and not isinstance(evidence_map[key], expected_types[key])
        )
    else:
        invalid = tuple(
            key
            for key in keys
            if key in evidence_map and not isinstance(evidence_map[key], expected_type)
        )
    if failed or missing or invalid:
        labels = []
        if failed:
            labels.append(f"failed={failed!r}")
        if missing:
            labels.append(f"missing={missing!r}")
        if invalid:
            labels.append(f"invalid={invalid!r}")
        context: dict[str, object] = {"failed": failed, "missing": missing}
        if invalid:
            context["invalid"] = invalid
        raise CapabilityError(
            "family selection requires complete typed evidence: " + ", ".join(labels),
            code="family.evidence.incomplete",
            context=context,
        )
    return [evidence_map[key] for key in keys]


SequentialHypothesisKey = ArmHypothesisKey | SegmentHypothesisKey


def _select_registered_breakout_family(
    cells: Sequence[tuple[object, object]],
    q: Fraction | float,
    inference: AsymptoticMean | AlwaysValid | MixedFamily,
    *,
    computation: DecisionComputation,
) -> FamilyOutcome[SequentialHypothesisKey]:
    """Fixed preallocated per-cell allocation, never reinverted: the
    breakout/as-of view axis registers its own segment family with
    correction="bonferroni" (validate_breakout_registration), a real FWER
    promise over the segments -- e-BH's FDR guarantee is a DIFFERENT,
    weaker promise, so applying it here would silently loosen what the
    registration already declared. Each cell is judged independently at
    its own registered allocation, exactly as it always was."""
    from increment.estimation.decision_types import (
        AsymptoticSequentialEvidence,
        sequential_hypothesis_key,
    )
    from increment.sequential_state import registration_id, sequential_refuse

    registration = inference.registration
    roster = {sequential_hypothesis_key(c): c for c in registration.roster if c.family}
    if not roster or not {key for key, _ in cells}.issubset(roster):
        sequential_refuse("source.invalid", "family keys differ from the registered roster")
    if Fraction(q) != registration.q and float(registration.q) != q:
        sequential_refuse("source.invalid", "family level differs from registration")
    snapshot = computation.sequential_snapshot
    if snapshot is None or snapshot.registration_id != registration_id(registration):
        sequential_refuse("source.invalid", "family needs the registered finalized prefix")
    models = {model.metric: model for model in registration.models}
    order = list(roster)
    values = _require_complete_evidence(
        order,
        computation,
        inference=inference,
        expected_types=_sequential_evidence_types(roster, models),
    )
    selected = set()
    methods: dict[SequentialHypothesisKey, str] = {}
    regimes: set[str] = set()
    for key, value in zip(order, values, strict=True):
        if not isinstance(value, (EValueEvidence, AsymptoticSequentialEvidence)):
            sequential_refuse(
                "source.invalid", "family evidence type unsupported for a fixed-roster family"
            )
        value.checkpoint.verify_snapshot(snapshot)
        if value.checkpoint.cell != roster[key]:
            sequential_refuse("source.invalid", "family allocation changed")
        methods[key] = value.method
        if isinstance(value, EValueEvidence):
            regimes.add("finite_sample")
            rejected = value.rejects()
        else:
            regimes.add("asymptotic_sequential")
            rejected = value.result.rejects()
        if rejected:
            selected.add(key)
    guarantee = (
        "asymptotic_sequential"
        if "asymptotic_sequential" in regimes
        else "finite_sample"
        if regimes
        else None
    )
    return FamilyOutcome(
        selected=frozenset(selected),
        fcr_alpha=None,
        q=float(registration.q),
        n_family=len(roster),
        realized_threshold=None,
        capped=False,
        canonical_method=methods,
        guarantee=guarantee,
    )


def _sequential_evidence_types(roster, models) -> dict[object, type]:
    """Each registered family key's evidence type, by the law its model declares."""
    from increment.estimation.decision_types import AsymptoticSequentialEvidence

    return {
        key: EValueEvidence
        if models[cell.metric].law == "bernoulli"
        else AsymptoticSequentialEvidence
        for key, cell in roster.items()
    }


def select_sequential_family(
    cells: Sequence[tuple[object, object]],
    q: Fraction | float,
    inference: AsymptoticMean | AlwaysValid | MixedFamily,
    nominal_alpha: Fraction | float,
    *,
    computation: DecisionComputation,
) -> FamilyOutcome[SequentialHypothesisKey]:
    """e-BH over the retained roster's per-cell log evidence: exact for a
    Bernoulli cell, direction-respecting asymptotic for a count-clock cell.
    A cell without usable evidence at its stopped state -- a missing arm, too
    few observations, zero variance or a nonpositive ratio denominator --
    carries log evidence of minus infinity: it stays in the family size and is
    never selected. Any recorded failure refuses the family.
    The family's guarantee is the weakest regime present across all cells.

    A ratio cell's evidence is capped by its denominators' stability, so it
    can clear the step threshold at ``q * R / m`` only where its set resolves
    the denominators at that level, whatever its registered allocation; a
    ratio whose denominators are not resolved at ``nominal_alpha`` enters as
    minus infinity, again without leaving the roster. A selection's reporting
    level ``min(q * R / m, nominal_alpha)`` is one of those two, so every
    selected ratio is available there. The capped evidence is dominated by
    the contrast's own, so e-BH over it keeps the same guarantee, and a
    selected set is still the contrast's inversion at the reporting level.
    Selection is not individual significance: a selected row whose reporting
    level is below ``q * R / m`` need not exclude its null.

    A continuous breakout family -- every family cell segmented, under the
    count-clock asymptotic construction of any ``ScalarMeanModel`` law -- is
    dispatched to the fixed, never-reinverted per-cell allocation instead: it
    is registered with correction="bonferroni" (validate_breakout_registration),
    a familywise promise e-BH's false discovery rate must not silently loosen.
    ``sequential_family_rule`` is the one statement of that dispatch. The as-of
    view reuses the same registration and roster as the ordinary secondary
    family, so it gets the same e-BH answer.
    """
    from increment.estimation.decision_types import (
        AsymptoticSequentialEvidence,
        sequential_hypothesis_key,
    )
    from increment.semantics.sequential import sequential_family_rule
    from increment.sequential_state import registration_id, sequential_refuse

    registration = inference.registration
    registered = {sequential_hypothesis_key(cell): cell for cell in registration.roster}
    roster = {key: cell for key, cell in registered.items() if cell.family}
    order = list(roster)
    if not order or not {key for key, _ in cells}.issubset(roster):
        sequential_refuse("source.invalid", "family keys differ from the registered roster")
    if sequential_family_rule(registration.models, registration.roster) == "bonferroni":
        return _select_registered_breakout_family(cells, q, inference, computation=computation)
    if not 0 < nominal_alpha < 1:
        _raise("estimation.family.nominal_alpha_finite", nominal_alpha=nominal_alpha)
    if Fraction(q) != registration.q and float(registration.q) != q:
        sequential_refuse("source.invalid", "family level differs from registration")
    snapshot = computation.sequential_snapshot
    if snapshot is None:
        sequential_refuse("source.invalid", "family selection requires a verified joint snapshot")
    models = {model.metric: model for model in registration.models}
    values = _require_complete_evidence(
        order,
        computation,
        inference=inference,
        expected_types=_sequential_evidence_types(roster, models),
    )
    log_e_values: list[Fraction | float] = []
    regimes: set[str] = set()
    methods: dict[SequentialHypothesisKey, str] = {}
    prefixes = set()
    nominal = exact_fraction(nominal_alpha)
    for key, value in zip(order, values, strict=True):
        if not isinstance(value, (EValueEvidence, AsymptoticSequentialEvidence)):
            sequential_refuse(
                "source.invalid", "registered family requires typed sequential evidence"
            )
        cp = value.checkpoint
        cp.verify_snapshot(snapshot)
        if (
            cp.registration_id != registration_id(registration)
            or cp.cell != registered[key]
            or cp.model != models[cp.cell.metric]
            or cp.filtration_id != registration.reveal.filtration_id
            or cp.control.group_id != registration.control_group
        ):
            sequential_refuse("source.invalid", "family model, roster or filtration changed")
        if cp.status != "frozen":
            prefixes.add((cp.prefix_id, cp.revealed_units))
        methods[key] = value.method
        if isinstance(value, EValueEvidence):
            log_e_values.append(value.log_e)
            regimes.add("finite_sample")
        else:
            result = value.result
            log_e_values.append(
                result.log_e if result._denominator_resolved(nominal) else float("-inf")
            )
            regimes.add("asymptotic_sequential")
    if len(prefixes) > 1:
        sequential_refuse("source.invalid", "current cells use different joint reveal prefixes")
    guarantee = (
        "asymptotic_sequential"
        if "asymptotic_sequential" in regimes
        else "finite_sample"
        if regimes
        else None
    )
    selected_indices = e_bh_select(log_e_values, registration.q)
    selected = frozenset(order[i] for i in selected_indices)
    realized = registration.q * len(selected) / len(order) if selected else None
    alpha = min(realized, nominal) if realized is not None else None
    return FamilyOutcome(
        selected=selected,
        fcr_alpha=alpha,
        q=float(q),
        n_family=len(order),
        realized_threshold=float(realized) if realized is not None else None,
        capped=alpha is not None and realized is not None and alpha < realized,
        canonical_method=methods,
        guarantee=guarantee,
    )


def select_family[K: Hashable](
    cells: Sequence[tuple[K, object]],
    q: float | Fraction,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | FixedInference | None,
    nominal_alpha: float,
    *,
    computation: DecisionComputation,
) -> FamilyOutcome[K]:
    """Fixed-horizon family selection from complete typed p-value evidence.

    Registered likelihood families use select_sequential_family so roster keys
    and exact alpha allocations cannot be inferred from presentation rows.
    """
    if not math.isfinite(nominal_alpha) or not 0 < nominal_alpha < 1:
        _raise("estimation.family.nominal_alpha_finite", nominal_alpha=nominal_alpha)
    if inference is not None and not isinstance(inference, FixedInference):
        from increment.sequential_state import sequential_refuse

        sequential_refuse(
            "route.unsupported", "registered evidence requires select_sequential_family"
        )
    order = _unique_family_keys(cells)
    evidence = _require_complete_evidence(order, computation, inference=inference)
    p_values = []
    for value in evidence:
        if not isinstance(value, PValueEvidence):
            from increment.sequential_state import sequential_refuse

            sequential_refuse("source.invalid", "fixed family requires p-value evidence")
        p_values.append(value.p_value)
    canonical_method = {key: value.method for key, value in zip(order, evidence, strict=True)}
    fixed_q = float(q)
    if Fraction(fixed_q) > Fraction(q):
        fixed_q = math.nextafter(fixed_q, 0.0)
    selected_idx, threshold = bh_select(p_values, fixed_q)
    selected = frozenset(order[i] for i in selected_idx)
    realized = _conservative_ratio(fixed_q, len(selected_idx), len(order)) if selected else None
    alpha = min(realized, nominal_alpha) if realized is not None else None
    return FamilyOutcome(
        selected=selected,
        fcr_alpha=alpha,
        q=fixed_q,
        n_family=len(order),
        realized_threshold=threshold if selected else None,
        capped=alpha is not None and realized is not None and alpha < realized,
        canonical_method=canonical_method,
    )


#: Fields a family correction writes; a row carrying any of them is already corrected.
_FAMILY_FIELDS = (
    "discovery",
    "family_axes",
    "family_q",
    "family_threshold",
    "family_guarantee",
    "family_nominal_alpha",
    "family_size",
)
#: Reissued interval fields and their disclosures; everything else is the row's own.
_INTERVAL_FIELDS = (
    "lift",
    "binomial_set",
    "relative_confidence_set",
    "abs_lb",
    "abs_ub",
    "abs_alpha",
    "note",
)
#: Each selected row is capped at its own nominal level below, so selection runs uncapped.
_UNCAPPED = math.nextafter(1.0, 0.0)
#: Breakout exclusions that condition only on arm counts, ancillary to the outcome: such a cell
#: is no hypothesis, so leaving it out of the family cannot bias selection. Outcome-based
#: exclusions (zero variance, non-positive mean, extreme ratio) stay in as non-rejections.
_DESIGN_EXCLUSIONS = frozenset({"few_units", "no_control_arm"})
_ADMISSION_ORDER = (
    "estimation.family.exploratory_row",
    "estimation.family.exploratory_pre_corrected",
    "estimation.family.exploratory_non_decision",
    "estimation.family.exploratory_sequential",
    BH_EXCLUDES_PRIOR.code,
)


#: A segment cell's scope: (dimension, segment value, fact source).
_Scope = tuple[str, str, str | None]


@dataclass(frozen=True, slots=True)
class _SourcedSegmentKey(SegmentHypothesisKey):
    """A segment hypothesis that also names the fact source it was read from."""

    source: str | None


@dataclass(frozen=True, slots=True)
class _Member:
    """One exploratory-family row with its hypothesis key and the estimate evidence reads."""

    row: LiftEstimate | BreakoutEstimate
    key: ArmHypothesisKey | _SourcedSegmentKey
    scope: _Scope | None
    view: LiftEstimate | None
    construction: Construction
    tested: bool = True


def _label(index: int, row: object) -> str:
    metric = getattr(row, "metric", None)
    if metric is None:
        return f"rows[{index}] ({type(row).__name__})"
    label = f"{metric}/{getattr(row, 'group_id', '?')}"
    dimension = getattr(row, "dimension", None)
    if dimension is None:
        return label
    source = getattr(row, "source", None)
    where = f"{dimension}={getattr(row, 'dimension_value', '?')}"
    return f"{label}[{where}{'' if source is None else f' @{source}'}]"


#: The stable reason each admission hazard reports, per row, in its refusal's ``reasons``.
_ADMISSION_REASONS = {
    "estimation.family.exploratory_row": "not a whole-window or segment estimate row",
    "estimation.family.exploratory_pre_corrected": "already family-corrected",
    "estimation.family.exploratory_non_decision": "sensitivity row, not a decision row",
    "estimation.family.exploratory_sequential": "sequential inference",
    BH_EXCLUDES_PRIOR.code: "informative prior",
}


def _admission_hazard(row: object) -> str | None:
    """The first reason *row* cannot join an exploratory family, as its refusal code."""
    if not callable(getattr(row, "_family_view", None)) or getattr(row, "ds", None) is not None:
        return "estimation.family.exploratory_row"
    if any(getattr(row, name, None) is not None for name in _FAMILY_FIELDS):
        return "estimation.family.exploratory_pre_corrected"
    if getattr(row, "method_role", None) != "decision":
        return "estimation.family.exploratory_non_decision"
    if (
        getattr(row, "inference", "fixed") != "fixed"
        or getattr(row, "reference_kind", None) == "sequential"
        or getattr(row, "sequential_result", None) is not None
    ):
        return "estimation.family.exploratory_sequential"
    if getattr(row, "prior_shrunk", False) or getattr(row, "prior_spec", None) is not None:
        return BH_EXCLUDES_PRIOR.code
    return None


def _require_admissible(rows: Sequence[object]) -> None:
    """Refuse, naming every offending row, the first hazard that applies to any row."""
    offenders: dict[str, list[str]] = {}
    for index, row in enumerate(rows):
        code = _admission_hazard(row)
        if code is not None:
            offenders.setdefault(code, []).append(_label(index, row))
    for code in _ADMISSION_ORDER:
        if code in offenders:
            spec = BH_EXCLUDES_PRIOR if code == BH_EXCLUDES_PRIOR.code else _REFUSALS[code]
            labels = tuple(offenders[code])
            refuse(spec, rows=labels, reasons=(_ADMISSION_REASONS[code],) * len(labels))


def exploratory_family_exclusion(row: LiftEstimate | BreakoutEstimate) -> str | None:
    """Why *row* cannot join an exploratory family, or ``None`` when it can.

    This is the predicate ``select_exploratory_family`` applies, so a caller can leave an
    excluded row out of the family and report the reason instead of letting one hazardous cell
    refuse every cell. The string is the reason the refusal carries in its ``reasons`` context
    (``constructions`` for an interval that cannot be reissued). A cell excluded by design
    (too few units, no control arm) is not excluded here: the family returns it unchanged.
    """
    from increment.estimation._fcr_reinterval import classify

    code = _admission_hazard(row)
    if code is not None:
        return _ADMISSION_REASONS[code]
    view = row._family_view()
    if view is None:
        return None
    construction, reason = classify(view)
    return reason if construction is None else None


def _scope_of(row: object) -> _Scope | None:
    """A segment cell's scope: its dimension, segment value and fact source.

    The same segment value read from two fact sources is two cells, so the source is part of the
    identity. Whole-window rows have no scope.
    """
    dimension = getattr(row, "dimension", None)
    if dimension is None:
        return None
    return (dimension, cast("Any", row).dimension_value, getattr(row, "source", None))


def _members(rows: Sequence[LiftEstimate | BreakoutEstimate]) -> list[_Member]:
    """Key every row and classify how its interval is reissued, refusing what cannot be."""
    from increment.estimation._fcr_reinterval import classify

    members: list[_Member] = []
    obstructed: list[tuple[str, str]] = []
    for index, row in enumerate(rows):
        scope = _scope_of(row)
        key = _in_scope(ArmHypothesisKey(row.metric, row.group_id, row.estimand), scope)
        view = row._family_view()
        construction, reason = ("unavailable", "") if view is None else classify(view)
        if construction is None:
            obstructed.append((_label(index, row), reason))
            continue
        tested = view is not None or getattr(row, "excluded", None) not in _DESIGN_EXCLUSIONS
        members.append(_Member(row, key, scope, view, construction, tested))
    if obstructed:
        refuse(
            _REFUSALS["estimation.family.exploratory_construction"],
            rows=tuple(label for label, _ in obstructed),
            constructions=tuple(reason for _, reason in obstructed),
        )
    return members


def _in_scope(
    hypothesis: HypothesisKey, scope: _Scope | None
) -> ArmHypothesisKey | _SourcedSegmentKey:
    arm = cast("ArmHypothesisKey", hypothesis)
    if scope is None:
        return arm
    return _SourcedSegmentKey(arm.metric, arm.group_id, arm.estimand, *scope)


def _exploratory_computation(members: Sequence[_Member]) -> DecisionComputation[Any]:
    """Typed p-value evidence for every member, from the engine's own derivation.

    Each scope's rows (one segment of one fact source, or the whole window) go through
    ``_lift_decision_bundle`` together, as ``run_breakout`` does, and are re-keyed to that scope;
    an excluded cell becomes the failure that run keeps.
    """
    from increment.estimation.engine import _lift_decision_bundle

    views: dict[_Scope | None, list[LiftEstimate]] = {}
    for member in members:
        if member.view is not None:
            views.setdefault(member.scope, []).append(member.view)
    evidence: dict[HypothesisKey, TestEvidence] = {}
    failures: dict[HypothesisKey, DecisionFailure] = {}
    for scope, scope_views in views.items():
        bundle = _lift_decision_bundle(scope_views, inference=None)
        for value in bundle.evidence.values():
            key = _in_scope(value.hypothesis, scope)
            evidence[key] = replace(value, hypothesis=key)
        for failure in bundle.failures.values():
            key = _in_scope(failure.hypothesis, scope)
            failures[key] = DecisionFailure(key, failure.code, failure.context)
    for member in members:
        if member.view is None:
            reason = getattr(member.row, "excluded", None)
            failures[member.key] = DecisionFailure(
                member.key,
                f"breakout.{reason}",
                {"metric": member.row.metric, "group_id": member.row.group_id, "reason": reason},
            )
    return DecisionComputation(results=(), evidence=evidence, failures=failures)


def _stamped[R: BaseModel](row: R, view: LiftEstimate | None, **family: object) -> R:
    """*row* with *view*'s reissued interval and the family record, revalidated as a whole."""
    fields = type(row).model_fields
    data: dict[str, Any] = {name: getattr(row, name) for name in fields}
    if view is not None:
        data.update({name: getattr(view, name) for name in _INTERVAL_FIELDS})
    data.update(family)
    return type(row)(**data)


def select_exploratory_family(
    rows: Sequence[LiftEstimate | BreakoutEstimate], *, q: float
) -> tuple[LiftEstimate | BreakoutEstimate, ...]:
    """Correct whole-window and segment decision rows as one Benjamini-Hochberg family.

    ``rows`` are uncorrected ``LiftEstimate`` rows (one per metric and arm) and
    ``BreakoutEstimate`` rows (one per metric, arm and segment). Every decision row is one
    hypothesis of a single family at FDR level ``q``; the rows come back in the same order and
    types. Each carries ``discovery``, ``family_axes`` (``("metric", "arm")``, plus
    ``"segment"`` when any row is a segment cell), ``family_q``, ``family_threshold`` (the
    realized ``R*q/m``, ``None`` when nothing is selected) and ``family_size`` (``m``), so a row
    read alone states which family corrected it.

    Selection is ``select_family`` over the p-values the estimators themselves emit, so
    discovery equals ``bh_select`` on those p-values, ties at the threshold included. A selected
    row's interval is reissued at the Benjamini-Yekutieli level ``1 - R*q/m``, capped at the
    row's own nominal level, from the construction the row persists: the Wald interval from its
    raw statistics and reference, the exact binomial set from its counts, or the Fieller set
    from its joint reference, each with its one-sided geometry. That equals the interval
    ``run_breakout(correction="bh")`` returns for the same cell. A row whose relative interval is
    unavailable (a non-positive arm mean) reports only its additive interval; an absolute margin
    can select it, and that interval is reissued from its persisted ``abs_alpha`` at the same
    level, never narrower than the nominal one. An unselected row keeps its interval.

    A cell excluded for an outcome-based reason (zero variance, non-positive mean, extreme
    ratio) stays in ``m`` as a non-rejection, as in ``run_breakout``. A cell excluded by design
    (too few units, no control arm) is no hypothesis: it is returned unchanged, outside the
    family.

    The call refuses before any work, naming every offending row: rows that are not whole-window
    or segment rows (``estimation.family.exploratory_row``), rows already corrected
    (``estimation.family.exploratory_pre_corrected``), sensitivity rows
    (``estimation.family.exploratory_non_decision``), sequential rows
    (``estimation.family.exploratory_sequential``), informative-prior rows
    (``breakout.run_breakout_bh_excludes_prior``), and rows whose interval cannot be reissued
    without approximation (``estimation.family.exploratory_construction``: quantile, percentile
    winsorized, additive-scale and similar constructions, and an additive-only interval whose
    ``abs_alpha`` was not persisted). A cell whose evidence is unavailable
    refuses the family as ``run_breakout`` does (``family.evidence.incomplete``). An empty input
    returns ``()``.
    """
    _require_q(q)
    rows = tuple(rows)
    _require_admissible(rows)
    members = _members(rows)
    tested = [member for member in members if member.tested]
    if not tested:
        return tuple(member.row for member in members)
    outcome = select_family(
        [(member.key, member.row) for member in tested],
        q,
        None,
        _UNCAPPED,
        computation=_exploratory_computation(tested),
    )
    from increment.estimation._fcr_reinterval import reinterval_selected

    axes = ("metric", "arm", "segment") if any(m.scope for m in tested) else ("metric", "arm")
    family = {
        "family_axes": axes,
        "family_q": outcome.q,
        "family_threshold": outcome.realized_threshold,
        "family_size": outcome.n_family,
    }
    corrected: list[LiftEstimate | BreakoutEstimate] = []
    for member in members:
        if not member.tested:
            corrected.append(member.row)
            continue
        selected = family_discovery(outcome, member.key)
        view = None
        if selected and member.view is not None:
            assert outcome.realized_threshold is not None
            view = reinterval_selected(member.view, member.construction, outcome.realized_threshold)
        corrected.append(_stamped(member.row, view, discovery=selected, **family))
    return tuple(corrected)
