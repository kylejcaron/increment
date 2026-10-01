"""Portable stopped state, evidence and confidence geometry for one hypothesis."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from increment.errors import CodedModel, InvalidRequestError, RefusalSpec, refuse
from increment.estimation._certified import log_interval
from increment.estimation._sequential_inversion import (
    ConfidenceBounds,
    bernoulli_confidence_sequence,
    gaussian_confidence_sequence,
)
from increment.estimation._sequential_likelihood import (
    BernoulliState,
    BetaPrior,
    GaussianPrior,
    GaussianState,
    LikelihoodCertificate,
    bernoulli_evidence,
    gaussian_evidence,
)
from increment.estimation.asymptotic_joint import (
    JointPreparation,
    invert_joint,
    joint_means,
    prepare_joint,
)
from increment.estimation.asymptotic_mean import (
    AsymptoticMeanSet,
    asymptotic_mean_set,
    boundary_alpha,
    count_boundary,
    count_boundary_log_e,
)
from increment.semantics.sequential import (
    RATIO_LAWS,
    PredictivePrior,
    ScalarMeanModel,
    SequentialCell,
    SequentialModel,
    SequentialSamplingModel,
)
from increment.sequential_state import (
    SequentialArmState,
    SequentialCheckpoint,
    sequential_refuse,
)


class SequentialInferenceResult(BaseModel):
    """Ratio-coordinate bounds are authoritative, including empty/full sets.

    Point availability is independent of confidence-set availability. Frozen
    results retain exactly the state and stopping time used by their evidence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    checkpoint: SequentialCheckpoint
    certificate: LikelihoodCertificate
    bounds: ConfidenceBounds
    decision_alpha: Fraction
    alpha_ceiling: Fraction
    point_reason: str | None = None

    @model_validator(mode="after")
    def _levels(self):
        if not 0 < self.decision_alpha <= self.alpha_ceiling:
            sequential_refuse(
                "source.invalid", "decision alpha must not exceed its recorded ceiling"
            )
        if self.bounds.alpha != self.decision_alpha:
            sequential_refuse("source.invalid", "interval and decision allocations disagree")
        if self.bounds.alternative != self.checkpoint.cell.alternative:
            sequential_refuse(
                "source.invalid", "confidence geometry and hypothesis direction differ"
            )

        if (
            self.certificate != checkpoint_certificate(self.checkpoint)
            or self.bounds != checkpoint_bounds(self.checkpoint, self.decision_alpha)
            or self.point_reason != _point(self.checkpoint)[1]
        ):
            sequential_refuse("source.invalid", "portable evidence or confidence geometry changed")
        return self

    @property
    def log_e(self) -> Fraction | float:
        if self.certificate.status == "zero":
            return float("-inf")
        if self.certificate.status == "infinite":
            return float("inf")
        assert self.certificate.log_e is not None
        return self.certificate.log_e.lo

    def rejects(self) -> bool:
        """Equality-null evidence at or above 1/alpha.

        This is the same event as the registered null ratio lying outside
        ``bounds``: the interval inverts each tail against the same 1/alpha and
        the equality statistic equals the larger tail e-process at that ratio.
        """
        return self.log_e >= (-log_interval(self.decision_alpha)).hi


class AsymptoticSequentialResult(CodedModel, BaseModel):
    """Portable asymptotic evidence and confidence geometry for one stopped state
    (asymptotic, never exact).

    ``log_e`` is the contrast's direction-respecting plug-in mixture value,
    not a finite-sample e-value. Its estimated variance does not preserve the
    oracle martingale's expectation bound. The value reads the stopped state
    alone and is unchanged by freezing, replay or alpha reinversion.
    A ratio law caps it by denominator stability, clearing ``-log(alpha)``
    only where the denominators are resolved. ``bounds`` invert the uncapped contrast at
    ``decision_alpha``: where that set is available, a family member rejects
    when the uncapped value exceeds ``-log(decision_alpha)``. A ratio set's
    availability, unlike ``log_e``, depends on that alpha."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")
    construction: Literal["direct_shifted_contrast_v1", "linearised_shifted_contrast_v1"] = (
        "direct_shifted_contrast_v1"
    )
    validity_regime: Literal["asymptotic_sequential"] = "asymptotic_sequential"
    checkpoint: SequentialCheckpoint
    bounds: AsymptoticMeanSet
    decision_alpha: Fraction
    alpha_ceiling: Fraction
    point_reason: str | None = None

    @model_validator(mode="after")
    def _replay(self):
        if not isinstance(self.checkpoint.model, ScalarMeanModel):
            sequential_refuse(
                "source.invalid", "asymptotic evidence requires the scalar mean model"
            )
        if self.construction != self.checkpoint.model.construction:
            sequential_refuse(
                "source.invalid",
                "result construction differs from its retained law; evaluate the checkpoint again",
            )
        if not 0 < self.decision_alpha <= self.alpha_ceiling:
            sequential_refuse(
                "source.invalid", "asymptotic decision alpha must not exceed its recorded ceiling"
            )
        if (
            self.bounds != checkpoint_mean_bounds(self.checkpoint, self.decision_alpha)
            or self.point_reason != _point(self.checkpoint)[1]
        ):
            sequential_refuse(
                "source.invalid", "asymptotic checkpoint geometry or allocation changed"
            )
        return self

    @property
    def log_e(self) -> Fraction | float:
        model = self.checkpoint.model
        assert isinstance(model, ScalarMeanModel)
        if model.law != "scalar_mean":
            return checkpoint_joint(self.checkpoint).log_e(self.checkpoint.cell.alternative)
        if not self.bounds.available:
            return float("-inf")
        assert self.bounds.estimator_contrast is not None
        assert self.bounds.estimator_variance is not None
        return count_boundary_log_e(
            count=self.bounds.count,
            rho=model.rho,
            estimator_contrast=self.bounds.estimator_contrast,
            estimator_variance=self.bounds.estimator_variance,
            alternative=self.checkpoint.cell.alternative,
        )

    def _denominator_resolved(self, alpha: Fraction) -> bool:
        """Whether a ratio law's denominators are separated from zero at the
        boundary this cell's set uses at ``alpha``; other laws have none.

        A ratio state that never reaches the denominator guard -- not ready,
        or not linearisable -- has nothing resolved.
        """
        model, cell = self.checkpoint.model, self.checkpoint.cell
        if model.law not in RATIO_LAWS:
            return True
        prepared = checkpoint_joint(self.checkpoint)
        if prepared.reason is not None:
            return False
        level = boundary_alpha(alpha, cell.alternative, e_value_dual=cell.family)
        return prepared.resolves(count_boundary(prepared.count, level, prepared.rho))

    def rejects(self) -> bool:
        return self.bounds.rejects()


SequentialResult = SequentialInferenceResult | AsymptoticSequentialResult


_MEMO_INVALID = RefusalSpec(
    "sequential.memo.invalid", InvalidRequestError, lambda *, reason: reason
)

# Entries hold exact rationals plus a validated cell/model copy. At 500 Bernoulli states and
# 500 units per arm, certificate and bounds entries measured 10.3 kB and 17.1 kB, so 512 of
# each use about 14 MB. Larger replay workloads can call `set_checkpoint_memo_size`.
_DEFAULT_MEMO_ENTRIES = 512


@dataclass(frozen=True, slots=True)
class MemoUsage:
    """One checkpoint memo's observed reuse, so a caller can size the bound."""

    name: str
    hits: int
    misses: int
    entries: int
    bound: int


class _StateMemo:
    """Bounded memo over the sufficient state a checkpoint computation reads.

    A checkpoint also carries provenance: registration, prefix and filtration
    identity, the revealed prefix length and the retention status. None of that
    reaches evidence or inversion, and prefix identity changes with every new
    reveal, so a memo keyed on the whole checkpoint cannot hit across two
    reveals of an identical retained state.
    """

    __slots__ = ("_compute", "_memo", "bound", "name")

    def __init__(self, name: str, compute: Callable[..., object]) -> None:
        self.name = name
        self.bound = _DEFAULT_MEMO_ENTRIES
        self._compute = compute
        self._memo = lru_cache(maxsize=_DEFAULT_MEMO_ENTRIES)(compute)

    def __call__(self, *key):
        return self._memo(*key)

    def resize(self, entries: int) -> None:
        """Rebind at a new bound; retained values are exact, so dropping is free."""
        self.bound = entries
        self._memo = lru_cache(maxsize=entries)(self._compute)

    def clear(self) -> None:
        self._memo.cache_clear()

    def usage(self) -> MemoUsage:
        info = self._memo.cache_info()
        return MemoUsage(self.name, info.hits, info.misses, info.currsize, self.bound)


def _beta_prior(prior: PredictivePrior) -> BetaPrior:
    if prior.kind != "beta" or prior.a is None or prior.b is None:
        sequential_refuse("source.invalid", "Bernoulli evidence requires a proper Beta prior")
    return BetaPrior(prior.a, prior.b)


def _gaussian_prior(prior: PredictivePrior) -> GaussianPrior:
    if prior.kind == "beta" or prior.kappa is None or prior.nu is None:
        sequential_refuse("source.invalid", "Gaussian evidence requires a proper NIG or NIW prior")
    return GaussianPrior(prior.kappa, prior.nu, prior.mean, prior.scale)


def _certificate(
    cell: SequentialCell,
    model: SequentialSamplingModel,
    control_state: SequentialArmState,
    treatment_state: SequentialArmState,
) -> LikelihoodCertificate:
    """Evidence for one retained state under one declared law and hypothesis."""
    control, treatment = control_state.kernel(), treatment_state.kernel()
    if isinstance(model, ScalarMeanModel):
        sequential_refuse("route.unsupported", "asymptotic evidence is not an e-value")
    if not control.n or not treatment.n:
        return LikelihoodCertificate("zero", None, None, None, "missing retained cell")
    if model.law == "bernoulli":
        if not isinstance(control, BernoulliState) or not isinstance(treatment, BernoulliState):
            sequential_refuse("source.invalid", "Bernoulli model requires exact event counts")
        return bernoulli_evidence(
            control,
            treatment,
            _beta_prior(model.control_prior),
            _beta_prior(model.treatment_prior),
            ratio=1 + cell.null_lift,
            alternative=cell.alternative,
        )
    if not isinstance(control, GaussianState) or not isinstance(treatment, GaussianState):
        sequential_refuse("source.invalid", "Gaussian model requires exact joint state")
    return gaussian_evidence(
        control,
        treatment,
        _gaussian_prior(model.control_prior),
        _gaussian_prior(model.treatment_prior),
        ratio=1 + cell.null_lift,
        alternative=cell.alternative,
    )


_certificate_memo = _StateMemo("certificate", _certificate)


def checkpoint_certificate(checkpoint: SequentialCheckpoint) -> LikelihoodCertificate:
    """Recompute portable evidence from its exact stopped state."""
    return _certificate_memo(
        checkpoint.cell, checkpoint.model, checkpoint.control, checkpoint.treatment
    )


def _bounds(
    cell: SequentialCell,
    model: SequentialSamplingModel,
    control_state: SequentialArmState,
    treatment_state: SequentialArmState,
    alpha: Fraction,
) -> ConfidenceBounds:
    """Invert the same process the certificate scores, at one error level."""
    control, treatment = control_state.kernel(), treatment_state.kernel()
    if isinstance(model, SequentialModel) and model.law == "bernoulli":
        if not isinstance(control, BernoulliState) or not isinstance(treatment, BernoulliState):
            sequential_refuse("source.invalid", "Bernoulli model requires exact event counts")
        return bernoulli_confidence_sequence(
            control,
            treatment,
            _beta_prior(model.control_prior),
            _beta_prior(model.treatment_prior),
            alpha=alpha,
            alternative=cell.alternative,
        )
    if not isinstance(control, GaussianState) or not isinstance(treatment, GaussianState):
        sequential_refuse("source.invalid", "Gaussian model requires exact joint state")
    if isinstance(model, ScalarMeanModel):
        sequential_refuse("route.unsupported", "asymptotic sets use their registered allocation")
    return gaussian_confidence_sequence(
        control,
        treatment,
        _gaussian_prior(model.control_prior),
        _gaussian_prior(model.treatment_prior),
        alpha=alpha,
        alternative=cell.alternative,
    )


_bounds_memo = _StateMemo("bounds", _bounds)


def checkpoint_bounds(checkpoint: SequentialCheckpoint, alpha: Fraction) -> ConfidenceBounds:
    """Invert the same process; bounded memoisation also serves portable replay."""
    return _bounds_memo(
        checkpoint.cell, checkpoint.model, checkpoint.control, checkpoint.treatment, alpha
    )


def _point(checkpoint: SequentialCheckpoint) -> tuple[float | None, str | None]:
    c, t = checkpoint.control, checkpoint.treatment
    if not c.n or not t.n:
        return None, "missing arm in retained cell"
    if c.law == "bernoulli":
        assert c.successes is not None and t.successes is not None
        mc, mt = Fraction(c.successes, c.n), Fraction(t.successes, t.n)
    elif c.law in ("gaussian", "scalar_mean"):
        mc, mt = c.mean[0], t.mean[0]
    elif c.law == "gaussian_ratio":
        if not c.mean[1] or not t.mean[1]:
            return None, "observed denominator mean is zero"
        mc, mt = c.mean[0] / c.mean[1], t.mean[0] / t.mean[1]
    else:
        model, control, treatment = checkpoint.model, c.kernel(), t.kernel()
        if (
            not isinstance(model, ScalarMeanModel)
            or not isinstance(control, GaussianState)
            or not isinstance(treatment, GaussianState)
        ):
            sequential_refuse(
                "source.invalid", "joint asymptotic checkpoint requires centered joint moments"
            )
        means, reason = joint_means(control, treatment, model)
        if means is None:
            return None, reason
        mc, mt = means
    if mc == 0:
        return None, "observed control mean is zero"
    try:
        value = float(mt / mc - 1)
    except OverflowError:
        return None, "relative point exceeds binary64 range"
    return (
        (value, None) if math.isfinite(value) else (None, "relative point exceeds binary64 range")
    )


def _centered(
    model: SequentialSamplingModel,
    control_state: SequentialArmState,
    treatment_state: SequentialArmState,
) -> tuple[ScalarMeanModel, GaussianState, GaussianState]:
    """The scalar mean declaration and its centered arm moments, or a refusal."""
    control, treatment = control_state.kernel(), treatment_state.kernel()
    if (
        not isinstance(model, ScalarMeanModel)
        or not isinstance(control, GaussianState)
        or not isinstance(treatment, GaussianState)
    ):
        sequential_refuse(
            "source.invalid", "scalar mean checkpoint requires centered scalar moments"
        )
    return model, control, treatment


def _joint(
    model: SequentialSamplingModel,
    control_state: SequentialArmState,
    treatment_state: SequentialArmState,
    null_lift: Fraction,
) -> JointPreparation:
    """A joint law's stopped contrast before any error level: every inversion
    alpha and the family evidence read this one preparation."""
    declaration, control, treatment = _centered(model, control_state, treatment_state)
    return prepare_joint(control, treatment, declaration=declaration, null_lift=null_lift)


_joint_memo = _StateMemo("joint", _joint)


def checkpoint_joint(checkpoint: SequentialCheckpoint) -> JointPreparation:
    """The joint preparation of a checkpoint's stopped state at its registered null."""
    return _joint_memo(
        checkpoint.model, checkpoint.control, checkpoint.treatment, checkpoint.cell.null_lift
    )


def _mean_bounds(
    cell: SequentialCell,
    model: SequentialSamplingModel,
    control_state: SequentialArmState,
    treatment_state: SequentialArmState,
    alpha: Fraction,
) -> AsymptoticMeanSet:
    # Available sets invert the uncapped contrast; ratio family evidence
    # also caps it by the denominator guard at each reporting level.
    if model.law != "scalar_mean":
        return invert_joint(
            _joint_memo(model, control_state, treatment_state, cell.null_lift),
            alpha=alpha,
            alternative=cell.alternative,
            e_value_dual=cell.family,
        )
    declaration, control, treatment = _centered(model, control_state, treatment_state)
    return asymptotic_mean_set(
        control,
        treatment,
        declaration=declaration,
        alpha=alpha,
        null_lift=cell.null_lift,
        alternative=cell.alternative,
        e_value_dual=cell.family,
    )


_mean_bounds_memo = _StateMemo("mean_bounds", _mean_bounds)


def checkpoint_mean_bounds(
    checkpoint: SequentialCheckpoint, alpha: Fraction | None = None
) -> AsymptoticMeanSet:
    """Recover the asymptotic set at the registered cell alpha by default, or
    an explicit FCR-selected alpha for a reinverted selected row."""
    resolved = checkpoint.cell.alpha if alpha is None else alpha
    return _mean_bounds_memo(
        checkpoint.cell, checkpoint.model, checkpoint.control, checkpoint.treatment, resolved
    )


_CHECKPOINT_MEMOS = (_certificate_memo, _bounds_memo, _joint_memo, _mean_bounds_memo)


def set_checkpoint_memo_size(entries: int) -> None:
    """Rebound every checkpoint memo; replay-heavy callers raise the default.

    The memos are keyed on user state, so the bound is a memory policy the
    embedding process owns. Resizing discards the retained values; every one of
    them is recomputable from its key, so no result changes.
    """
    if type(entries) is not int or entries < 1:
        refuse(_MEMO_INVALID, reason="checkpoint memo size must be a positive entry count")
    for memo in _CHECKPOINT_MEMOS:
        memo.resize(entries)


def checkpoint_memo_usage() -> tuple[MemoUsage, ...]:
    """Report reuse for every checkpoint memo, in declaration order."""
    return tuple(memo.usage() for memo in _CHECKPOINT_MEMOS)


def clear_checkpoint_memos() -> None:
    """Drop every retained value; the next call recomputes the same result."""
    for memo in _CHECKPOINT_MEMOS:
        memo.clear()
