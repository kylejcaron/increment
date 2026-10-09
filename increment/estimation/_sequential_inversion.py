"""Certified, outward directional confidence sequences in ratio coordinates.

Invert Q/sup(theta<=r)L for a lower endpoint and Q/sup(theta>=r)L for
an upper endpoint. Their nulls are nested even when equality evidence is not
unimodal. Closed outward bounds include the threshold boundary conservatively.
The observation law, committed priors, and prefix provenance belong to callers.

Both tails compare with the same threshold 1/alpha, two-sided or not: at the
true parameter each tail's composite null contains the truth, so each tail
e-process is at most Q/L_truth, one test martingale whose crossing probability
Ville bounds by alpha. The equality-null statistic a decision uses equals the
larger tail e-process at the null ratio, so "rejects" and "null outside the
sequence" are one event.
"""

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from increment._literals import ALTERNATIVE_VALUES
from increment.errors import InvalidRequestError, RefusalSpec, refuse
from increment.estimation._certified import (
    _INEXACT_VALUE,
    _NONPOSITIVE_ROOT_WIDTH,
    Interval,
    log_interval,
)
from increment.estimation._sequential_likelihood import (
    _INVALID_CERTIFICATE_REASON,
    _STATE_PRIOR_DIMENSION_MISMATCH,
    _UNKNOWN_ALTERNATIVE,
    Alternative,
    BernoulliState,
    BetaPrior,
    GaussianPrior,
    GaussianState,
    LikelihoodCertificate,
    bernoulli_evidence,
    gaussian_evidence,
)

_Direction = Literal["greater", "less"]
_EndpointStatus = Literal["finite", "domain", "unbounded", "unresolved", "abstained", "empty"]
_Evaluator = Callable[[Fraction, _Direction, Fraction], LikelihoodCertificate]
_ZERO = Fraction(0)
_ERRORS = (Fraction(1, 10**10), Fraction(1, 10**20), Fraction(1, 10**40))
_INVALID_EVIDENCE_TYPE = RefusalSpec(
    "estimation.sequential_inversion.invalid_evidence_type",
    InvalidRequestError,
    template="evidence must be a likelihood certificate, got {value_type}",
)
_UNKNOWN_ENDPOINT_STATUS = RefusalSpec(
    "estimation.sequential_inversion.unknown_endpoint_status",
    InvalidRequestError,
    template="unknown endpoint status",
    keys=frozenset({"status"}),
)
_INVALID_BRACKET_LENGTH = RefusalSpec(
    "estimation.sequential_inversion.invalid_bracket_length",
    InvalidRequestError,
    template="endpoint bracket requires two coordinates",
    keys=frozenset({"bracket"}),
)
_REVERSED_BRACKET = RefusalSpec(
    "estimation.sequential_inversion.reversed_bracket",
    InvalidRequestError,
    template="endpoint bracket is reversed",
    keys=frozenset({"lower", "upper"}),
)
_INVALID_WITNESS_TYPE = RefusalSpec(
    "estimation.sequential_inversion.invalid_witness_type",
    InvalidRequestError,
    template="{witness} endpoint witness must be a typed evaluation, got {value_type}",
)
_WITNESS_OUTSIDE_BRACKET = RefusalSpec(
    "estimation.sequential_inversion.witness_outside_bracket",
    InvalidRequestError,
    template="endpoint witness lies outside its bracket",
    keys=frozenset({"bracket", "ratio", "witness"}),
)
_INCOMPLETE_FINITE_ENDPOINT = RefusalSpec(
    "estimation.sequential_inversion.incomplete_finite_endpoint",
    InvalidRequestError,
    template="finite endpoint requires two certified witnesses",
    keys=frozenset({"accepted", "bracket", "rejected"}),
)
_MISSING_ENDPOINT_REASON = RefusalSpec(
    "estimation.sequential_inversion.missing_endpoint_reason",
    InvalidRequestError,
    template="nonfinite endpoint status requires a reason",
    keys=frozenset({"status"}),
)
_UNKNOWN_DOMAIN = RefusalSpec(
    "estimation.sequential_inversion.unknown_domain",
    InvalidRequestError,
    template="unknown ratio domain",
    keys=frozenset({"domain"}),
)
_INVALID_THRESHOLD_TYPE = RefusalSpec(
    "estimation.sequential_inversion.invalid_threshold_type",
    InvalidRequestError,
    template="threshold must be a certified interval, got {value_type}",
)
_INVALID_DOMAIN_EVIDENCE_TYPE = RefusalSpec(
    "estimation.sequential_inversion.invalid_domain_evidence_type",
    InvalidRequestError,
    template="domain evidence must be a certified interval, got {value_type}",
)
_INVALID_ENDPOINT_TYPE = RefusalSpec(
    "estimation.sequential_inversion.invalid_endpoint_type",
    InvalidRequestError,
    template="{side} endpoint must have a typed certificate, got {value_type}",
)
_ENDPOINT_WIDTH_EXCEEDED = RefusalSpec(
    "estimation.sequential_inversion.endpoint_width_exceeded",
    InvalidRequestError,
    template="finite endpoint exceeds max_width",
    keys=frozenset({"bracket", "max_width", "side"}),
)
_UNCERTIFIED_REJECTION = RefusalSpec(
    "estimation.sequential_inversion.uncertified_rejection",
    InvalidRequestError,
    template="rejected endpoint lacks certified rejection",
    keys=frozenset({"evidence", "side", "threshold"}),
)
_REJECTED_COORDINATE_MISMATCH = RefusalSpec(
    "estimation.sequential_inversion.rejected_coordinate_mismatch",
    InvalidRequestError,
    template="rejected witness must be the outward bracket endpoint",
    keys=frozenset({"expected", "ratio", "side"}),
)
_UNCERTIFIED_NONREJECTION = RefusalSpec(
    "estimation.sequential_inversion.uncertified_nonrejection",
    InvalidRequestError,
    template="accepted endpoint lacks certified nonrejection",
    keys=frozenset({"evidence", "side", "threshold"}),
)
_ACCEPTED_COORDINATE_MISMATCH = RefusalSpec(
    "estimation.sequential_inversion.accepted_coordinate_mismatch",
    InvalidRequestError,
    template="accepted witness must be the inward bracket endpoint",
    keys=frozenset({"expected", "ratio", "side"}),
)
_UNCERTIFIED_UNIFORM_BOUND = RefusalSpec(
    "estimation.sequential_inversion.uncertified_uniform_bound",
    InvalidRequestError,
    template="uniform evidence bound does not prove nonrejection",
    keys=frozenset({"log_e_upper", "side", "threshold"}),
)
_BOUND_COORDINATE_MISMATCH = RefusalSpec(
    "estimation.sequential_inversion.bound_coordinate_mismatch",
    InvalidRequestError,
    template="bound must equal the outward certificate coordinate",
    keys=frozenset({"bound", "expected", "side"}),
)
_UNCERTIFIED_EMPTY_ENDPOINT = RefusalSpec(
    "estimation.sequential_inversion.uncertified_empty_endpoint",
    InvalidRequestError,
    template="empty endpoint requires a whole-domain rejection certificate",
    keys=frozenset({"domain_log_e", "side", "threshold"}),
)
_NEGATIVE_BERNOULLI_BOUND = RefusalSpec(
    "estimation.sequential_inversion.negative_bernoulli_bound",
    InvalidRequestError,
    template="Bernoulli lower endpoint must be nonnegative",
    keys=frozenset({"lower"}),
)
_UNCERTIFIED_REVERSED_BOUNDS = RefusalSpec(
    "estimation.sequential_inversion.uncertified_reversed_bounds",
    InvalidRequestError,
    template="reversed bounds require certified emptiness",
    keys=frozenset({"lower", "upper"}),
)
_INVALID_ALPHA = RefusalSpec(
    "estimation.sequential_inversion.invalid_alpha",
    InvalidRequestError,
    template="alpha must lie strictly between zero and one",
    keys=frozenset({"alpha"}),
)


def _exact(value: Fraction | int) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int)):
        refuse(_INEXACT_VALUE, value_type=type(value).__name__)
    return Fraction(value)


@dataclass(frozen=True)
class EndpointEvaluation:
    """A finite ratio and its actual directional kernel certificate."""

    ratio: Fraction
    evidence: LikelihoodCertificate

    def __post_init__(self) -> None:
        object.__setattr__(self, "ratio", _exact(self.ratio))
        if not isinstance(self.evidence, LikelihoodCertificate):
            refuse(_INVALID_EVIDENCE_TYPE, value_type=type(self.evidence).__name__)


@dataclass(frozen=True)
class EndpointCertificate:
    """Endpoint bracket, with separate rejection and nonrejection witnesses.

    None in bracket[0]/bracket[1] means minus/plus infinity. An unresolved
    bracket is still outward: its rejected witness, if any, bounds the exact
    crossing on the outside. A missing rejected witness retains the domain
    endpoint. Only finite status certifies a bracket of the requested width.
    log_e_upper is an optional uniform bound from a common feasible null face;
    it proves no rejection anywhere, rather than extrapolating a finite probe.
    """

    status: _EndpointStatus
    bracket: tuple[Fraction | None, Fraction | None]
    rejected: EndpointEvaluation | None = None
    accepted: EndpointEvaluation | None = None
    reason: str | None = None
    log_e_upper: Fraction | None = None

    def __post_init__(self) -> None:
        if self.status not in ("finite", "domain", "unbounded", "unresolved", "abstained", "empty"):
            refuse(_UNKNOWN_ENDPOINT_STATUS, status=self.status)
        bracket = tuple(None if x is None else _exact(x) for x in self.bracket)
        if len(bracket) != 2:
            refuse(_INVALID_BRACKET_LENGTH, bracket=bracket)
        lo, hi = bracket
        if lo is not None and hi is not None and lo > hi:
            refuse(_REVERSED_BRACKET, lower=lo, upper=hi)
        object.__setattr__(self, "bracket", bracket)
        for witness, evaluation in (("rejected", self.rejected), ("accepted", self.accepted)):
            if evaluation is not None and not isinstance(evaluation, EndpointEvaluation):
                refuse(
                    _INVALID_WITNESS_TYPE,
                    witness=witness,
                    value_type=type(evaluation).__name__,
                )
            if evaluation is not None and (
                (lo is not None and evaluation.ratio < lo)
                or (hi is not None and evaluation.ratio > hi)
            ):
                refuse(
                    _WITNESS_OUTSIDE_BRACKET,
                    witness=witness,
                    ratio=evaluation.ratio,
                    bracket=bracket,
                )
        if self.status == "finite" and (
            lo is None or hi is None or self.rejected is None or self.accepted is None
        ):
            refuse(
                _INCOMPLETE_FINITE_ENDPOINT,
                bracket=bracket,
                rejected=self.rejected,
                accepted=self.accepted,
            )
        if self.reason is not None and (not isinstance(self.reason, str) or not self.reason):
            refuse(_INVALID_CERTIFICATE_REASON, reason=self.reason)
        if self.status != "finite" and self.reason is None:
            refuse(_MISSING_ENDPOINT_REASON, status=self.status)
        if self.log_e_upper is not None:
            object.__setattr__(self, "log_e_upper", _exact(self.log_e_upper))


@dataclass(frozen=True)
class ConfidenceBounds:
    """Immutable closed outer interval; coordinates are ratios, never lifts.

    lower/upper None represent -/+ infinity. Inspect status and certificates:
    an unresolved infinite bound is not a claim of mathematical unboundedness.
    empty also covers an empty intersection proved by overlapping rejected
    tails; its stored coordinates need not form an ordered nonempty interval.
    max_width controls each crossing bracket, not the confidence interval width.
    """

    lower: Fraction | None
    upper: Fraction | None
    lower_certificate: EndpointCertificate
    upper_certificate: EndpointCertificate
    domain: Literal["nonnegative", "real"]
    alpha: Fraction
    alternative: Alternative
    max_width: Fraction
    log_threshold: Interval
    domain_log_e: Interval | None = None

    def __post_init__(self) -> None:
        alpha, width = _arguments(self.alpha, self.max_width, self.alternative)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "max_width", width)
        for name in ("lower", "upper"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _exact(value))
        if self.domain not in ("nonnegative", "real"):
            refuse(_UNKNOWN_DOMAIN, domain=self.domain)
        if not isinstance(self.log_threshold, Interval):
            refuse(_INVALID_THRESHOLD_TYPE, value_type=type(self.log_threshold).__name__)
        if self.domain_log_e is not None and not isinstance(self.domain_log_e, Interval):
            refuse(_INVALID_DOMAIN_EVIDENCE_TYPE, value_type=type(self.domain_log_e).__name__)
        for lower, endpoint in ((True, self.lower_certificate), (False, self.upper_certificate)):
            if not isinstance(endpoint, EndpointCertificate):
                refuse(
                    _INVALID_ENDPOINT_TYPE,
                    side="lower" if lower else "upper",
                    value_type=type(endpoint).__name__,
                )
            lo, hi = endpoint.bracket
            if endpoint.status == "finite" and lo is not None and hi is not None:
                if hi - lo > width:
                    refuse(
                        _ENDPOINT_WIDTH_EXCEEDED,
                        side="lower" if lower else "upper",
                        bracket=endpoint.bracket,
                        max_width=width,
                    )
            if endpoint.rejected is not None:
                if _classify(endpoint.rejected.evidence, self.log_threshold) != 1:
                    refuse(
                        _UNCERTIFIED_REJECTION,
                        side="lower" if lower else "upper",
                        evidence=endpoint.rejected.evidence,
                        threshold=self.log_threshold,
                    )
                if endpoint.rejected.ratio != (lo if lower else hi):
                    refuse(
                        _REJECTED_COORDINATE_MISMATCH,
                        side="lower" if lower else "upper",
                        ratio=endpoint.rejected.ratio,
                        expected=lo if lower else hi,
                    )
            if endpoint.accepted is not None:
                if _classify(endpoint.accepted.evidence, self.log_threshold) != -1:
                    refuse(
                        _UNCERTIFIED_NONREJECTION,
                        side="lower" if lower else "upper",
                        evidence=endpoint.accepted.evidence,
                        threshold=self.log_threshold,
                    )
                if endpoint.accepted.ratio != (hi if lower else lo):
                    refuse(
                        _ACCEPTED_COORDINATE_MISMATCH,
                        side="lower" if lower else "upper",
                        ratio=endpoint.accepted.ratio,
                        expected=hi if lower else lo,
                    )
            if endpoint.log_e_upper is not None and endpoint.log_e_upper >= self.log_threshold.lo:
                refuse(
                    _UNCERTIFIED_UNIFORM_BOUND,
                    side="lower" if lower else "upper",
                    log_e_upper=endpoint.log_e_upper,
                    threshold=self.log_threshold,
                )
            bound = self.lower if lower else self.upper
            expected = (
                (lo if lower else hi)
                if endpoint.status in ("finite", "unresolved")
                else (_ZERO if lower and self.domain == "nonnegative" else None)
            )
            if bound != expected:
                refuse(
                    _BOUND_COORDINATE_MISMATCH,
                    side="lower" if lower else "upper",
                    bound=bound,
                    expected=expected,
                )
            if endpoint.status == "empty" and not self.empty:
                refuse(
                    _UNCERTIFIED_EMPTY_ENDPOINT,
                    side="lower" if lower else "upper",
                    domain_log_e=self.domain_log_e,
                    threshold=self.log_threshold,
                )
        if self.domain == "nonnegative" and (self.lower is None or self.lower < 0):
            refuse(_NEGATIVE_BERNOULLI_BOUND, lower=self.lower)
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            if not self.empty:
                refuse(_UNCERTIFIED_REVERSED_BOUNDS, lower=self.lower, upper=self.upper)

    @property
    def empty(self) -> bool:
        """Whether the whole domain or both overlapping tails were rejected."""
        if self.domain_log_e is not None and self.domain_log_e.lo >= self.log_threshold.hi:
            return True
        left, right = self.lower_certificate.rejected, self.upper_certificate.rejected
        return left is not None and right is not None and left.ratio >= right.ratio

    @property
    def status(self) -> Literal["empty", "full-domain", "interval", "unresolved", "abstained"]:
        if self.empty:
            return "empty"
        endpoints = (self.lower_certificate, self.upper_certificate)
        if any(e.status == "abstained" for e in endpoints):
            return "abstained"
        if any(e.status == "unresolved" for e in endpoints):
            return "unresolved"
        if self.upper is None and self.lower == (_ZERO if self.domain == "nonnegative" else None):
            return "full-domain"
        return "interval"

    def __repr__(self) -> str:
        from increment._display import format_interval

        interval = format_interval(self.lower, self.upper, empty=self.empty)
        confidence = float(1 - self.alpha)
        return (
            f"ConfidenceBounds({interval} ratio, confidence={confidence:.1%}, "
            f"status={self.status!r})"
        )


def _arguments(
    alpha: Fraction, max_width: Fraction, alternative: Alternative
) -> tuple[Fraction, Fraction]:
    alpha, width = _exact(alpha), _exact(max_width)
    if not 0 < alpha < 1:
        refuse(_INVALID_ALPHA, alpha=alpha)
    if width <= 0:
        refuse(_NONPOSITIVE_ROOT_WIDTH, max_width=width)
    if alternative not in ALTERNATIVE_VALUES:
        refuse(_UNKNOWN_ALTERNATIVE, alternative=alternative)
    return alpha, width


def _classify(evidence: LikelihoodCertificate, threshold: Interval) -> int:
    if evidence.status == "infinite":
        return 1
    if evidence.status == "zero":
        # Zero can mean only a lower-evidence abstention, not an upper bound.
        return 0
    assert evidence.log_e is not None
    if evidence.log_e.lo >= threshold.hi:
        return 1
    if evidence.log_e.hi < threshold.lo:
        return -1
    return 0


def _domain_endpoint(*, lower: bool, nonnegative: bool, reason: str) -> EndpointCertificate:
    if lower and nonnegative:
        return EndpointCertificate("domain", (_ZERO, _ZERO), reason=reason)
    return EndpointCertificate("unbounded", (None, None), reason=reason)


def _invert_tail(
    evaluate: _Evaluator,
    direction: _Direction,
    threshold: Interval,
    width: Fraction,
    *,
    nonnegative: bool,
    scale: Fraction,
    first: EndpointEvaluation,
    uniform_upper: Fraction | None,
) -> EndpointCertificate:
    lower = direction == "greater"
    if uniform_upper is not None and uniform_upper < threshold.lo:
        endpoint = _domain_endpoint(
            lower=lower, nonnegative=nonnegative, reason="common null face proves no rejection"
        )
        return EndpointCertificate(
            endpoint.status, endpoint.bracket, reason=endpoint.reason, log_e_upper=uniform_upper
        )
    rejected = accepted = None
    last_reason = None

    def probe(ratio: Fraction) -> int:
        nonlocal rejected, accepted, last_reason
        for error in _ERRORS:
            evidence = (
                first.evidence
                if ratio == first.ratio and error == _ERRORS[0]
                else evaluate(ratio, direction, error)
            )
            decision = _classify(evidence, threshold)
            if decision:
                point = EndpointEvaluation(ratio, evidence)
                if decision > 0:
                    if rejected is None or (
                        ratio > rejected.ratio if lower else ratio < rejected.ratio
                    ):
                        rejected = point
                elif accepted is None or (
                    ratio < accepted.ratio if lower else ratio > accepted.ratio
                ):
                    accepted = point
                return decision
            last_reason = evidence.reason or "log evidence overlaps the certified log threshold"
            if evidence.status == "zero":
                break
        return 0

    initial = probe(_ZERO)
    if lower and nonnegative and initial < 0:
        return EndpointCertificate(
            "domain", (_ZERO, _ZERO), accepted=accepted, reason="zero ratio is not rejected"
        )

    # Search separately for both signs of the comparison. A ratio with an
    # ambiguous certificate never replaces either certified bracket endpoint.
    for sign in (1,) if nonnegative else (-1, 1):
        for step in range(32):
            if rejected is not None and accepted is not None:
                break
            if sign < 0 and (
                (lower and rejected is not None) or (not lower and accepted is not None)
            ):
                break
            if sign > 0 and (
                (lower and accepted is not None) or (not lower and rejected is not None)
            ):
                break
            probe(sign * scale * 2**step)

    reason = "endpoint search exhausted 32 geometric expansions per side"
    if rejected is not None and accepted is not None:
        for _ in range(96):
            lo, hi = sorted((rejected.ratio, accepted.ratio))
            if hi - lo <= width:
                return EndpointCertificate("finite", (lo, hi), rejected, accepted)
            midpoint = (lo + hi) / 2
            if probe(midpoint) == 0:
                # Exact threshold equality may stay ambiguous at every precision.
                # Quarter probes can still enclose the crossing without guessing.
                before = (rejected, accepted)
                probe((lo + midpoint) / 2)
                probe((midpoint + hi) / 2)
                if before == (rejected, accepted):
                    reason = "threshold comparison unresolved after arithmetic refinement and quarter probes"
                    break
        else:
            lo, hi = sorted((rejected.ratio, accepted.ratio))
            if hi - lo <= width:
                return EndpointCertificate("finite", (lo, hi), rejected, accepted)
            reason = "endpoint bisection exhausted 96 subdivisions before max_width was proved"

    lo = rejected if lower else accepted
    hi = accepted if lower else rejected
    bracket = (
        lo.ratio if lo is not None else (_ZERO if nonnegative else None),
        hi.ratio if hi is not None else None,
    )
    if last_reason is not None:
        reason += "; " + last_reason
    return EndpointCertificate("unresolved", bracket, rejected, accepted, reason)


def _gaussian_face_log(state: GaussianState, residual: Fraction) -> Interval:
    """Reduced likelihood on a fixed mean face; residual is squared Mahalanobis distance."""
    if not state.n:
        return Interval.exact(0)
    n = state.n
    if len(state.mean) == 1:
        return -Fraction(n, 2) * (1 + log_interval(state.scatter[0][0] * (1 + n * residual) / n))
    s = state.scatter
    determinant = s[0][0] * s[1][1] - s[0][1] ** 2
    return (
        -n
        + n * log_interval(Fraction(n))
        - Fraction(n, 2) * log_interval(determinant * (1 + n * residual))
    )


def _face_residual(
    state: GaussianState,
    face: Literal[
        "vertical", "positive", "negative", "full", "horizontal_positive", "horizontal_negative"
    ],
) -> Fraction:
    """Exact distance to axis faces used only for domain and infinity certificates.

    In two dimensions positive/negative constrain the numerator sign and a
    nonnegative denominator; full constrains only the denominator. A quadrant
    projection is its vertex, either axis projection, or the feasible mean.
    """
    if not state.n:
        return _ZERO
    if len(state.mean) == 1:
        mean = state.mean[0]
        feasible = (
            face == "full"
            or (face == "positive" and mean >= 0)
            or (face == "negative" and mean <= 0)
        )
        return _ZERO if feasible else mean * mean / state.scatter[0][0]
    a, b = state.mean
    c, d, e = state.scatter[0][0], state.scatter[0][1], state.scatter[1][1]
    if face == "full":
        return _ZERO if b >= 0 else b * b / e
    determinant = c * e - d * d
    vertex = (e * a * a - 2 * d * a * b + c * b * b) / determinant
    vertical = vertex - max(_ZERO, c * b - d * a) ** 2 / (c * determinant)
    if face == "vertical":
        return vertical
    sign = -1 if face in ("negative", "horizontal_negative") else 1
    if face in ("positive", "negative") and sign * a >= 0 and b >= 0:
        return _ZERO
    horizontal = vertex - max(_ZERO, sign * (e * a - d * b)) ** 2 / (e * determinant)
    if face in ("horizontal_positive", "horizontal_negative"):
        return horizontal
    return min(vertical, horizontal)


def _assemble(
    evaluate: _Evaluator,
    *,
    alpha: Fraction,
    alternative: Alternative,
    width: Fraction,
    nonnegative: bool,
    scale: Fraction,
    prefix_reason: str | None = None,
    abstain: bool = False,
    no_lower: bool = False,
    no_upper: bool = False,
    gaussian_states: tuple[GaussianState, GaussianState] | None = None,
) -> ConfidenceBounds:
    # One threshold for every direction. At the true ratio both tail
    # e-processes are dominated by the single test martingale Q/L_theta*, so
    # the union "either tail >= 1/alpha" already has probability <= alpha by
    # Ville; halving alpha across the tails would only widen the sequence.
    threshold = -log_interval(alpha)
    left = _domain_endpoint(
        lower=True, nonnegative=nonnegative, reason=prefix_reason or "lower tail not requested"
    )
    right = _domain_endpoint(
        lower=False, nonnegative=nonnegative, reason=prefix_reason or "upper tail not requested"
    )
    domain_log_e = None
    if abstain:
        left = EndpointCertificate("abstained", left.bracket, reason=prefix_reason)
        right = EndpointCertificate("abstained", right.bracket, reason=prefix_reason)
    elif prefix_reason is None:
        directions: tuple[_Direction, ...] = ("greater", "less")
        for direction in directions:
            if alternative not in ("two-sided", direction):
                continue
            if (direction == "greater" and no_lower) or (direction == "less" and no_upper):
                endpoint = _domain_endpoint(
                    lower=direction == "greater",
                    nonnegative=nonnegative,
                    reason="zero successes provide a common maximizing Bernoulli null face",
                )
            else:
                first = EndpointEvaluation(_ZERO, evaluate(_ZERO, direction, _ERRORS[0]))
                uniform_upper = None
                if gaussian_states is not None:
                    control, treatment = gaussian_states
                    q = first.evidence.log_predictive
                    assert q is not None
                    full_control = _gaussian_face_log(control, _face_residual(control, "positive"))
                    full = full_control + _gaussian_face_log(
                        treatment, _face_residual(treatment, "full")
                    )
                    domain_log_e = q - full
                    common = _gaussian_face_log(control, _face_residual(control, "vertical"))
                    common += _gaussian_face_log(
                        treatment,
                        _face_residual(
                            treatment, "negative" if direction == "greater" else "positive"
                        ),
                    )
                    uniform_upper = (q - common).hi
                    if len(treatment.mean) == 2:
                        # A vanishing treatment denominator gives a second common
                        # null face without forcing the control numerator to zero.
                        horizontal = full_control + _gaussian_face_log(
                            treatment,
                            _face_residual(
                                treatment,
                                "horizontal_negative"
                                if direction == "greater"
                                else "horizontal_positive",
                            ),
                        )
                        uniform_upper = min(uniform_upper, (q - horizontal).hi)
                    if domain_log_e.lo >= threshold.hi:
                        # Every directional null is contained in the full domain.
                        left = right = EndpointCertificate(
                            "empty",
                            (None, None),
                            reason="full ratio domain rejected; see domain_log_e",
                        )
                        break
                endpoint = _invert_tail(
                    evaluate,
                    direction,
                    threshold,
                    width,
                    nonnegative=nonnegative,
                    scale=scale,
                    first=first,
                    uniform_upper=uniform_upper,
                )
            if direction == "greater":
                left = endpoint
            else:
                right = endpoint
    lower = (
        left.bracket[0]
        if left.status in ("finite", "unresolved")
        else (_ZERO if nonnegative else None)
    )
    upper = right.bracket[1] if right.status in ("finite", "unresolved") else None
    return ConfidenceBounds(
        lower,
        upper,
        left,
        right,
        "nonnegative" if nonnegative else "real",
        alpha,
        alternative,
        width,
        threshold,
        domain_log_e,
    )


def bernoulli_confidence_sequence(
    control: BernoulliState,
    treatment: BernoulliState,
    prior_control: BetaPrior,
    prior_treatment: BetaPrior,
    *,
    alpha: Fraction,
    alternative: Alternative = "two-sided",
    max_width: Fraction = Fraction(1, 10**8),
) -> ConfidenceBounds:
    """Invert Beta evidence on [0,+infinity); both tails compare with 1/alpha."""
    alpha, width = _arguments(alpha, max_width, alternative)

    def evaluate(ratio: Fraction, direction: _Direction, error: Fraction) -> LikelihoodCertificate:
        return bernoulli_evidence(
            control,
            treatment,
            prior_control,
            prior_treatment,
            ratio=ratio,
            alternative=direction,
            max_error=error,
        )

    scale = Fraction(1)
    if control.successes and treatment.n:
        scale = max(
            scale, Fraction(treatment.successes * control.n, treatment.n * control.successes)
        )
    return _assemble(
        evaluate,
        alpha=alpha,
        alternative=alternative,
        width=width,
        nonnegative=True,
        scale=scale,
        prefix_reason="empty prefix has Q=L=1" if not control.n and not treatment.n else None,
        no_lower=treatment.successes == 0,
        no_upper=control.successes == 0,
    )


def gaussian_confidence_sequence(
    control: GaussianState,
    treatment: GaussianState,
    prior_control: GaussianPrior,
    prior_treatment: GaussianPrior,
    *,
    alpha: Fraction,
    alternative: Alternative = "two-sided",
    max_width: Fraction = Fraction(1, 10**8),
) -> ConfidenceBounds:
    """Invert NIG/NIW directional evidence over the entire real ratio line.

    Nonempty singular scatters temporarily abstain; subsequent calls use all
    accumulated observations once scatters are positive definite. Axis-face
    likelihoods certify some infinite endpoints and full-domain rejections.
    Other unsuccessful tail searches retain explicit unresolved outer bounds.
    """
    alpha, width = _arguments(alpha, max_width, alternative)
    dimension = len(control.mean)
    if any(len(obj.mean) != dimension for obj in (treatment, prior_control, prior_treatment)):
        refuse(
            _STATE_PRIOR_DIMENSION_MISMATCH,
            control_dimension=dimension,
            treatment_dimension=len(treatment.mean),
            prior_control_dimension=len(prior_control.mean),
            prior_treatment_dimension=len(prior_treatment.mean),
        )
    singular = any(
        state.n
        and (
            state.scatter[0][0] == 0
            if dimension == 1
            else state.scatter[0][0] * state.scatter[1][1] == state.scatter[0][1] ** 2
        )
        for state in (control, treatment)
    )
    reason = None
    if not control.n and not treatment.n:
        reason = "empty prefix has Q=L=1"
    elif singular:
        reason = "temporary singular-prefix abstention: nonempty Gaussian scatter is not positive definite"

    def evaluate(ratio: Fraction, direction: _Direction, error: Fraction) -> LikelihoodCertificate:
        return gaussian_evidence(
            control,
            treatment,
            prior_control,
            prior_treatment,
            ratio=ratio,
            alternative=direction,
            max_error=error,
        )

    scale = Fraction(1)
    if control.mean[0]:
        if dimension == 1:
            scale = max(scale, abs(treatment.mean[0] / control.mean[0]))
        elif treatment.mean[1]:
            scale = max(
                scale,
                abs(treatment.mean[0] * control.mean[1] / (treatment.mean[1] * control.mean[0])),
            )
    return _assemble(
        evaluate,
        alpha=alpha,
        alternative=alternative,
        width=width,
        nonnegative=False,
        scale=scale,
        prefix_reason=reason,
        abstain=bool(singular),
        gaussian_states=(control, treatment),
    )
