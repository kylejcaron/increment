"""Conjugate evidence kernels for Bernoulli and Gaussian sequences.

Inputs are exact observation sufficient statistics, not rational conversions of
rounded warehouse aggregates. Gaussian logs omit the same common normalizer in
both Q and L. All numerical enclosures and root isolation belong to _certified.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Literal

from increment._literals import ALTERNATIVE_VALUES, Alternative
from increment.errors import InvalidRequestError, RefusalSpec, refuse
from increment.estimation._certified import (
    _INEXACT_VALUE,
    _INVALID_COUNT,
    Interval,
    isolate_roots,
    log_gamma_half_step,
    log_interval,
    log_rising,
)

_Polynomial = tuple[Fraction, ...]
_Matrix = tuple[tuple[Fraction, ...], ...]
_ZERO = Fraction(0)
_ONE = Fraction(1)
_HALF = Fraction(1, 2)
_INVALID_DIMENSION = RefusalSpec(
    "estimation.sequential_likelihood.invalid_dimension",
    InvalidRequestError,
    template="dimension must be a positive integer",
    keys=frozenset({"dimension"}),
)
_MATRIX_DIMENSION_MISMATCH = RefusalSpec(
    "estimation.sequential_likelihood.matrix_dimension_mismatch",
    InvalidRequestError,
    template="matrix dimensions must match the mean (dimension {dimension}): got row lengths {row_lengths}",
)
_ASYMMETRIC_MATRIX = RefusalSpec(
    "estimation.sequential_likelihood.asymmetric_matrix",
    InvalidRequestError,
    template="matrix must be symmetric",
    keys=frozenset({"matrix"}),
)
_NONPOSITIVE_DEFINITE_MATRIX = RefusalSpec(
    "estimation.sequential_likelihood.nonpositive_definite_matrix",
    InvalidRequestError,
    lambda *, matrix, principal_minors: (
        "matrix must be positive definite: principal minors "
        f"({', '.join(str(minor) for minor in principal_minors)})"
    ),
)
_NON_PSD_MATRIX = RefusalSpec(
    "estimation.sequential_likelihood.non_psd_matrix",
    InvalidRequestError,
    lambda *, matrix, principal_minors: (
        "matrix must be PSD: principal minors "
        f"({', '.join(str(minor) for minor in principal_minors)})"
    ),
)
_SUCCESSES_EXCEED_COUNT = RefusalSpec(
    "estimation.sequential_likelihood.successes_exceed_count",
    InvalidRequestError,
    template="successes cannot exceed n",
    keys=frozenset({"n", "successes"}),
)
_NONZERO_SMALL_SAMPLE_SCATTER = RefusalSpec(
    "estimation.sequential_likelihood.nonzero_small_sample_scatter",
    InvalidRequestError,
    template="n=0 or n=1 requires zero centered scatter",
    keys=frozenset({"n", "scatter"}),
)
_NONZERO_EMPTY_MEAN = RefusalSpec(
    "estimation.sequential_likelihood.nonzero_empty_mean",
    InvalidRequestError,
    template="empty state requires canonical zero mean",
    keys=frozenset({"mean"}),
)
_SCATTER_RANK_EXCEEDS_COUNT = RefusalSpec(
    "estimation.sequential_likelihood.scatter_rank_exceeds_count",
    InvalidRequestError,
    lambda *, n, scatter: f"{n} observations require scatter rank at most {n - 1}",
)
_ROW_DIMENSION_MISMATCH = RefusalSpec(
    "estimation.sequential_likelihood.row_dimension_mismatch",
    InvalidRequestError,
    template="row dimension {row_dimension} does not match the declared dimension {dimension}",
)
_NONFINITE_OBSERVATION = RefusalSpec(
    "estimation.sequential_likelihood.nonfinite_observation",
    InvalidRequestError,
    template="observations must be finite",
    keys=frozenset({"value"}),
)
_MERGE_DIMENSION_MISMATCH = RefusalSpec(
    "estimation.sequential_likelihood.merge_dimension_mismatch",
    InvalidRequestError,
    template="cannot merge different dimensions: {left_dimension} vs {right_dimension}",
)
_NONPOSITIVE_BETA_PARAMETER = RefusalSpec(
    "estimation.sequential_likelihood.nonpositive_beta_parameter",
    InvalidRequestError,
    template="Beta hyperparameters must be positive",
    keys=frozenset({"parameter", "value"}),
)
_NONPOSITIVE_PRIOR_KAPPA = RefusalSpec(
    "estimation.sequential_likelihood.nonpositive_prior_kappa",
    InvalidRequestError,
    template="prior requires kappa>0",
    keys=frozenset({"kappa"}),
)
_INSUFFICIENT_PRIOR_NU = RefusalSpec(
    "estimation.sequential_likelihood.insufficient_prior_nu",
    InvalidRequestError,
    template="prior requires nu>dimension-1",
    keys=frozenset({"dimension", "nu"}),
)
_INVALID_LOG_INTERVAL = RefusalSpec(
    "estimation.sequential_likelihood.invalid_log_interval",
    InvalidRequestError,
    template="log field {field!r} must be a certified interval, got {value_type}",
)
_INVALID_CERTIFICATE_REASON = RefusalSpec(
    "estimation.sequential_likelihood.invalid_certificate_reason",
    InvalidRequestError,
    template="reason must be a nonempty string",
    keys=frozenset({"reason"}),
)
_MISSING_FINITE_LOGS = RefusalSpec(
    "estimation.sequential_likelihood.missing_finite_logs",
    InvalidRequestError,
    template="finite evidence requires all three intervals",
    keys=frozenset({"log_e", "log_null_sup", "log_predictive"}),
)
_EVIDENCE_ENCLOSURE_MISMATCH = RefusalSpec(
    "estimation.sequential_likelihood.evidence_enclosure_mismatch",
    InvalidRequestError,
    template="log_e must enclose predictive minus null supremum",
    keys=frozenset({"log_e", "required"}),
)
_INCONSISTENT_NONFINITE_LOGS = RefusalSpec(
    "estimation.sequential_likelihood.inconsistent_nonfinite_logs",
    InvalidRequestError,
    template="nonfinite evidence requires a reason and absent null/e logs",
    keys=frozenset({"log_e", "log_null_sup", "reason", "status"}),
)
_UNKNOWN_CERTIFICATE_STATUS = RefusalSpec(
    "estimation.sequential_likelihood.unknown_certificate_status",
    InvalidRequestError,
    template="unknown certificate status",
    keys=frozenset({"status"}),
)
_UNKNOWN_ALTERNATIVE = RefusalSpec(
    "estimation.sequential_likelihood.unknown_alternative",
    InvalidRequestError,
    template="unknown alternative",
    keys=frozenset({"alternative"}),
)
_NONPOSITIVE_MAX_ERROR = RefusalSpec(
    "estimation.sequential_likelihood.nonpositive_max_error",
    InvalidRequestError,
    template="max_error must be positive",
    keys=frozenset({"max_error"}),
)
_NEGATIVE_BERNOULLI_RATIO = RefusalSpec(
    "estimation.sequential_likelihood.negative_bernoulli_ratio",
    InvalidRequestError,
    template="Bernoulli ratio must be nonnegative",
    keys=frozenset({"ratio"}),
)
_SINGULAR_CANDIDATE_RESIDUAL = RefusalSpec(
    "estimation.sequential_likelihood.singular_candidate_residual",
    InvalidRequestError,
    template="scalar candidate crosses a singular residual",
    keys=frozenset({"candidate", "mean", "n", "residual", "scatter"}),
)
_STATE_PRIOR_DIMENSION_MISMATCH = RefusalSpec(
    "estimation.sequential_likelihood.state_prior_dimension_mismatch",
    InvalidRequestError,
    template="states and priors must have the same dimension: control={control_dimension}, treatment={treatment_dimension}, prior_control={prior_control_dimension}, prior_treatment={prior_treatment_dimension}",
)


def _rational(value: Fraction | int) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int)):
        refuse(_INEXACT_VALUE, value_type=type(value).__name__)
    return Fraction(value)


def _count(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        refuse(_INVALID_COUNT, count=value)


def _dimension(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        refuse(_INVALID_DIMENSION, dimension=value)


def _det(matrix: _Matrix) -> Fraction:
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    return sum(
        (
            (-1) ** j * matrix[0][j] * _det(tuple(row[:j] + row[j + 1 :] for row in matrix[1:]))
            for j in range(len(matrix))
        ),
        _ZERO,
    )


def _adjugate(matrix: _Matrix) -> _Matrix:
    return ((matrix[1][1], -matrix[0][1]), (-matrix[1][0], matrix[0][0]))


def _principal_minors(matrix: _Matrix, order: int) -> tuple[Fraction, ...]:
    """Determinants of every principal submatrix of the given order."""
    return tuple(
        _det(tuple(tuple(matrix[i][j] for j in rows) for i in rows))
        for rows in combinations(range(len(matrix)), order)
    )


def _vector_matrix(
    mean: Sequence[Fraction | int], matrix: Sequence[Sequence[Fraction | int]], *, positive: bool
) -> tuple[tuple[Fraction, ...], _Matrix]:
    vector = tuple(_rational(v) for v in mean)
    _dimension(len(vector))
    copied = tuple(tuple(_rational(v) for v in row) for row in matrix)
    d = len(vector)
    if len(copied) != d or any(len(row) != d for row in copied):
        refuse(
            _MATRIX_DIMENSION_MISMATCH,
            dimension=d,
            row_lengths=tuple(len(row) for row in copied),
        )
    if any(copied[i][j] != copied[j][i] for i in range(d) for j in range(d)):
        refuse(_ASYMMETRIC_MATRIX, matrix=copied)
    # Every principal minor: the diagonal, the intermediate orders and the
    # determinant, which is the whole PSD criterion (Sylvester needs only the
    # leading minors for definiteness, but zero eigenvalues need them all).
    minors = (
        *tuple(copied[i][i] for i in range(d)),
        *(minor for order in range(2, d) for minor in _principal_minors(copied, order)),
        _det(copied),
    )
    if any(v <= 0 if positive else v < 0 for v in minors):
        if positive:
            refuse(_NONPOSITIVE_DEFINITE_MATRIX, matrix=copied, principal_minors=minors)
        refuse(_NON_PSD_MATRIX, matrix=copied, principal_minors=minors)
    return vector, copied


@dataclass(frozen=True)
class BernoulliState:
    """Exact sequence counts (n, successes), including n=0."""

    n: int
    successes: int

    def __post_init__(self) -> None:
        _count(self.n)
        _count(self.successes)
        if self.successes > self.n:
            refuse(_SUCCESSES_EXCEED_COUNT, n=self.n, successes=self.successes)


@dataclass(frozen=True)
class GaussianState:
    """Exact n, mean, and S=Σ(x-mean)(x-mean)' in any dimension d>=1.

    The constructor accepts exact summaries with caller-established provenance.
    Only from_rows establishes exact accumulation of the supplied observations.
    """

    n: int
    mean: tuple[Fraction, ...]
    scatter: _Matrix

    def __init__(
        self, n: int, mean: Sequence[Fraction | int], scatter: Sequence[Sequence[Fraction | int]]
    ) -> None:
        _count(n)
        mean, scatter = _vector_matrix(mean, scatter, positive=False)
        if n <= 1 and any(v for row in scatter for v in row):
            refuse(_NONZERO_SMALL_SAMPLE_SCATTER, n=n, scatter=scatter)
        if n == 0 and any(mean):
            refuse(_NONZERO_EMPTY_MEAN, mean=mean)
        # Rank(S) <= n-1: for a PSD matrix the order-n principal minors sum to
        # the n-th elementary symmetric polynomial of its eigenvalues.
        if 2 <= n <= len(mean) and any(_principal_minors(scatter, n)):
            refuse(_SCATTER_RANK_EXCEEDS_COUNT, n=n, scatter=scatter)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scatter", scatter)

    @classmethod
    def empty(cls, dimension: int) -> "GaussianState":
        """Canonical zero state for the empty prefix."""
        _dimension(dimension)
        return cls(0, (_ZERO,) * dimension, ((_ZERO,) * dimension,) * dimension)

    @classmethod
    def from_rows(
        cls, rows: Sequence[Sequence[float | Fraction | int]], *, dimension: int
    ) -> "GaussianState":
        """Accumulate centered moments exactly, interpreting binary64 as dyadics.

        Chan's centered merge with a singleton is the exact Welford update.
        No decimal-string conversion or rounded raw-second-moment subtraction occurs.
        """
        state = cls.empty(dimension)
        zero_scatter = state.scatter
        for row in rows:
            if len(row) != dimension:
                refuse(_ROW_DIMENSION_MISMATCH, dimension=dimension, row_dimension=len(row))
            values = []
            for value in row:
                if isinstance(value, float):
                    try:
                        values.append(Fraction(*value.as_integer_ratio()))
                    except (ValueError, OverflowError):
                        refuse(_NONFINITE_OBSERVATION, value=value)
                else:
                    values.append(_rational(value))
            state = state.merge(cls(1, tuple(values), zero_scatter))
        return state

    def merge(self, other: "GaussianState") -> "GaussianState":
        """Chan merge: S=S1+S2+n1*n2/(n1+n2)*ΔΔ'."""
        if len(self.mean) != len(other.mean):
            refuse(
                _MERGE_DIMENSION_MISMATCH,
                left_dimension=len(self.mean),
                right_dimension=len(other.mean),
            )
        if not self.n:
            return other
        if not other.n:
            return self
        n = self.n + other.n
        delta = tuple(b - a for a, b in zip(self.mean, other.mean, strict=True))
        weight = Fraction(self.n * other.n, n)
        mean = tuple(a + Fraction(other.n, n) * d for a, d in zip(self.mean, delta, strict=True))
        scatter = tuple(
            tuple(
                self.scatter[i][j] + other.scatter[i][j] + weight * delta[i] * delta[j]
                for j in range(len(mean))
            )
            for i in range(len(mean))
        )
        return GaussianState(n, mean, scatter)


@dataclass(frozen=True)
class BetaPrior:
    """Proper Beta(a,b), Q=(a)_s(b)_(n-s)/(a+b)_n."""

    a: Fraction
    b: Fraction

    def __init__(self, a: Fraction | int, b: Fraction | int) -> None:
        for name, parameter in (("a", a), ("b", b)):
            value = _rational(parameter)
            if value <= 0:
                refuse(_NONPOSITIVE_BETA_PARAMETER, parameter=name, value=value)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class GaussianPrior:
    """Proper normal–inverse-gamma or normal–inverse-Wishart prior.

    Scalar inverse-gamma shape is nu/2 and scale is scale[0][0]/2;
    mu|variance is N(mean, variance/kappa). In dimension 2 this is
    NIW(kappa, nu, mean, Psi=scale), with nu>1 and positive definite Psi.
    """

    kappa: Fraction
    nu: Fraction
    mean: tuple[Fraction, ...]
    scale: _Matrix

    def __init__(
        self,
        kappa: Fraction | int,
        nu: Fraction | int,
        mean: Sequence[Fraction | int],
        scale: Sequence[Sequence[Fraction | int]],
    ) -> None:
        mean, scale = _vector_matrix(mean, scale, positive=True)
        kappa, nu = _rational(kappa), _rational(nu)
        if kappa <= 0:
            refuse(_NONPOSITIVE_PRIOR_KAPPA, kappa=kappa)
        if nu <= len(mean) - 1:
            refuse(_INSUFFICIENT_PRIOR_NU, nu=nu, dimension=len(mean))
        for name, value in (("mean", mean), ("scale", scale), ("kappa", kappa), ("nu", nu)):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class LikelihoodCertificate:
    """Outward log(Q/sup L) certificate.

    Finite logs use the same removed normalizer: log_e contains [q.lo-L.hi,
    q.hi-L.lo]. Zero/infinite tag evidence, not the null likelihood. A zero
    singular-prefix abstention is identified by reason, not a finite surrogate.
    """

    status: Literal["finite", "zero", "infinite"]
    log_predictive: Interval | None
    log_null_sup: Interval | None
    log_e: Interval | None
    reason: str | None

    def __post_init__(self) -> None:
        for field, value in (
            ("log_predictive", self.log_predictive),
            ("log_null_sup", self.log_null_sup),
            ("log_e", self.log_e),
        ):
            if value is not None and not isinstance(value, Interval):
                refuse(_INVALID_LOG_INTERVAL, field=field, value_type=type(value).__name__)
        if self.reason is not None and (not isinstance(self.reason, str) or not self.reason):
            refuse(_INVALID_CERTIFICATE_REASON, reason=self.reason)
        if self.status == "finite":
            if self.log_predictive is None or self.log_null_sup is None or self.log_e is None:
                refuse(
                    _MISSING_FINITE_LOGS,
                    log_predictive=self.log_predictive,
                    log_null_sup=self.log_null_sup,
                    log_e=self.log_e,
                )
            difference = self.log_predictive - self.log_null_sup
            if self.log_e.lo > difference.lo or self.log_e.hi < difference.hi:
                refuse(_EVIDENCE_ENCLOSURE_MISMATCH, log_e=self.log_e, required=difference)
        elif self.status in ("zero", "infinite"):
            if self.log_e is not None or self.log_null_sup is not None or not self.reason:
                refuse(
                    _INCONSISTENT_NONFINITE_LOGS,
                    status=self.status,
                    log_null_sup=self.log_null_sup,
                    log_e=self.log_e,
                    reason=self.reason,
                )
        else:
            refuse(_UNKNOWN_CERTIFICATE_STATUS, status=self.status)


def _trim(coefficients: Sequence[Fraction]) -> _Polynomial:
    result = list(coefficients)
    while result and result[-1] == 0:
        result.pop()
    return tuple(result)


def _add(a: Sequence[Fraction], b: Sequence[Fraction]) -> _Polynomial:
    return _trim(
        tuple(
            (a[i] if i < len(a) else _ZERO) + (b[i] if i < len(b) else _ZERO)
            for i in range(max(len(a), len(b)))
        )
    )


def _scale(a: Sequence[Fraction], scalar: Fraction | int) -> _Polynomial:
    return _trim(tuple(scalar * value for value in a))


def _multiply(a: Sequence[Fraction], b: Sequence[Fraction]) -> _Polynomial:
    if not a or not b:
        return ()
    result = [_ZERO] * (len(a) + len(b) - 1)
    for i, left in enumerate(a):
        for j, right in enumerate(b):
            result[i + j] += left * right
    return _trim(result)


def _polynomial(coefficients: Sequence[Fraction], x: Interval) -> Interval:
    result = Interval.exact(0)
    for coefficient in reversed(coefficients):
        result = result * x + coefficient
    return result


def _roots(
    coefficients: Sequence[Fraction], lower: Fraction, upper: Fraction | None, width: Fraction
) -> tuple[Interval, ...]:
    polynomial = _trim(coefficients)
    if len(polynomial) <= 1:
        return ()
    return isolate_roots(polynomial, lower=lower, upper=upper, max_width=width)


def _maximum(candidates: Sequence[Interval]) -> Interval:
    return Interval(max(v.lo for v in candidates), max(v.hi for v in candidates))


def _intersection(old: Interval | None, new: Interval) -> Interval:
    if old is None:
        return new
    return Interval(max(old.lo, new.lo), min(old.hi, new.hi))


def _candidate(evaluate: Callable[[Interval], Interval], root: Interval) -> Interval:
    enclosure = evaluate(root)
    witness = evaluate(Interval.exact(root.midpoint))
    return Interval(witness.lo, enclosure.hi)


def _arguments(
    ratio: Fraction, alternative: Alternative, max_error: Fraction
) -> tuple[Fraction, Fraction]:
    ratio, max_error = _rational(ratio), _rational(max_error)
    if alternative not in ALTERNATIVE_VALUES:
        refuse(_UNKNOWN_ALTERNATIVE, alternative=alternative)
    if max_error <= 0:
        refuse(_NONPOSITIVE_MAX_ERROR, max_error=max_error)
    return ratio, max_error


def _empty_certificate() -> LikelihoodCertificate:
    zero = Interval.exact(0)
    return LikelihoodCertificate("finite", zero, zero, zero, None)


def _refine(
    predictive: Callable[[int], Interval],
    null: Callable[[int, Fraction], Interval | None],
    max_error: Fraction,
) -> LikelihoodCertificate:
    q_best = d_best = None
    for iteration in range(12):
        precision = 60 * 2**iteration
        width = Fraction(1, 2 ** (40 * 2**iteration))
        q_best = _intersection(q_best, predictive(precision))
        denominator = null(precision, width)
        if denominator is None:
            return LikelihoodCertificate(
                "infinite", q_best, None, None, "null likelihood is identically zero"
            )
        d_best = _intersection(d_best, denominator)
        evidence = q_best - d_best
        if evidence.width <= max_error:
            return LikelihoodCertificate("finite", q_best, d_best, evidence, None)
    return LikelihoodCertificate(
        "finite",
        q_best,
        d_best,
        evidence,
        "numerical resolution not achieved within iteration budget; bounds remain certified",
    )


def _beta_predictive(state: BernoulliState, prior: BetaPrior, precision: int) -> Interval:
    if not state.n:
        return Interval.exact(0)
    return (
        log_rising(prior.a, state.successes, precision=precision)
        + log_rising(prior.b, state.n - state.successes, precision=precision)
        - log_rising(prior.a + prior.b, state.n, precision=precision)
    )


def _bernoulli_vanishes(state: BernoulliState, probability: Fraction) -> bool:
    """Whether a positive count meets a zero probability, so the likelihood is zero."""
    return bool(
        (state.successes and probability == 0) or (state.n - state.successes and probability == 1)
    )


def _bernoulli_log(state: BernoulliState, probability: Fraction, precision: int) -> Interval | None:
    if _bernoulli_vanishes(state, probability):
        return None
    result = Interval.exact(0)
    for count, p in ((state.successes, probability), (state.n - state.successes, 1 - probability)):
        if count:
            result += count * log_interval(p, precision=precision)
    return result


def _bernoulli_null(
    control: BernoulliState,
    treatment: BernoulliState,
    ratio: Fraction,
    alternative: Alternative,
    precision: int,
    width: Fraction,
) -> Interval | None:
    c_range = (Fraction(control.successes, control.n),) * 2 if control.n else (_ZERO, _ONE)
    t_range = (Fraction(treatment.successes, treatment.n),) * 2 if treatment.n else (_ZERO, _ONE)
    feasible = (alternative == "greater" and t_range[0] <= ratio * c_range[1]) or (
        alternative == "less" and t_range[1] >= ratio * c_range[0]
    )
    if feasible:
        c = _bernoulli_log(control, c_range[0], precision)
        t = _bernoulli_log(treatment, t_range[0], precision)
        assert c is not None and t is not None
        return c + t
    if ratio == 0:
        if treatment.successes:
            return None
        return _bernoulli_log(control, c_range[0], precision)
    upper = min(_ONE, 1 / ratio)
    successes = control.successes + treatment.successes
    coefficients = (
        Fraction(successes),
        -(
            successes * (1 + ratio)
            + ratio * (treatment.n - treatment.successes)
            + control.n
            - control.successes
        ),
        ratio * (control.n + treatment.n),
    )
    points = (Interval.exact(0), Interval.exact(upper), *_roots(coefficients, _ZERO, upper, width))
    candidates = []
    for point in points:
        midpoint = point.midpoint
        if _bernoulli_vanishes(control, midpoint) or _bernoulli_vanishes(
            treatment, ratio * midpoint
        ):
            continue
        c = _bernoulli_log(control, midpoint, precision)
        t = _bernoulli_log(treatment, ratio * midpoint, precision)
        assert c is not None and t is not None
        # Each term's upper endpoint bounds its value even if an interval
        # touches log(0); a separate rational midpoint supplies the lower bound.
        high = Interval.exact(0)
        for state, probability in ((control, point), (treatment, ratio * point)):
            for count, p in (
                (state.successes, probability),
                (state.n - state.successes, 1 - probability),
            ):
                if count:
                    argument = p if p.lo > 0 else p.hi
                    high += count * log_interval(argument, precision=precision)
        candidates.append(Interval((c + t).lo, high.hi))
    return _maximum(candidates) if candidates else None


def bernoulli_evidence(
    control: BernoulliState,
    treatment: BernoulliState,
    prior_control: BetaPrior,
    prior_treatment: BetaPrior,
    *,
    ratio: Fraction,
    alternative: Alternative,
    max_error: Fraction = Fraction(1, 10**12),
) -> LikelihoodCertificate:
    """Beta Q divided by the supremum-null Bernoulli sequence likelihood.

    Equality uses every root of A-[A(1+r)+r*fT+fC]u+r(nT+nC)u²,
    plus endpoints. Greater/less use pT<=r*pC / pT>=r*pC respectively.
    Zero-probability positive-count nulls yield infinite evidence, not NaN.
    """
    ratio, max_error = _arguments(ratio, alternative, max_error)
    if ratio < 0:
        refuse(_NEGATIVE_BERNOULLI_RATIO, ratio=ratio)
    if not control.n and not treatment.n:
        return _empty_certificate()
    return _refine(
        lambda p: (
            _beta_predictive(control, prior_control, p)
            + _beta_predictive(treatment, prior_treatment, p)
        ),
        lambda p, w: _bernoulli_null(control, treatment, ratio, alternative, p, w),
        max_error,
    )


def _gaussian_predictive(state: GaussianState, prior: GaussianPrior, precision: int) -> Interval:
    """Conjugate log predictive density with a common normalizer removed.

    With Ψn=Ψ0+S+κn/(κ+n)(m-m0)(m-m0)', the scalar expression is
    log(κ/(κ+n))/2 + logΓ((ν+n)/2)-logΓ(ν/2)
    + (ν/2)log(Ψ0/2) - ((ν+n)/2)log(Ψn/2).
    The vector expression is log(κ/(κ+n)) + (ν/2)log|Ψ0|
    - ((ν+n)/2)log|Ψn| + log((ν-1)_n).
    """
    if not state.n:
        return Interval.exact(0)
    n, d = state.n, len(state.mean)
    kappa_n = prior.kappa + n
    delta = tuple(a - b for a, b in zip(state.mean, prior.mean, strict=True))
    psi = tuple(
        tuple(
            prior.scale[i][j]
            + state.scatter[i][j]
            + prior.kappa * n / kappa_n * delta[i] * delta[j]
            for j in range(d)
        )
        for i in range(d)
    )
    log_kappa = log_interval(prior.kappa / kappa_n, precision=precision)
    if d == 1:
        a, k = prior.nu / 2, n // 2
        gamma = log_rising(a, k, precision=precision)
        if n % 2:
            # Γ(a+k+1/2)/Γ(a) = [Γ(a+k+1/2)/Γ(a+k)] * (a)_k.
            gamma += log_gamma_half_step(a + k, precision=precision)
        return (
            _HALF * log_kappa
            + gamma
            + a * log_interval(prior.scale[0][0] / 2, precision=precision)
            - (a + Fraction(n, 2)) * log_interval(psi[0][0] / 2, precision=precision)
        )
    # π^(-n) Γ₂((ν+n)/2)/Γ₂(ν/2) = (2π)^(-n) (ν-1)_n.
    # Remove (2π)^(-n) here and the identical likelihood factor below.
    return (
        log_kappa
        + prior.nu / 2 * log_interval(_det(prior.scale), precision=precision)
        - (prior.nu + n) / 2 * log_interval(_det(psi), precision=precision)
        + log_rising(prior.nu - 1, n, precision=precision)
    )


def _scalar_log(state: GaussianState, mean: Interval, precision: int) -> Interval:
    if not state.n:
        return Interval.exact(0)
    delta = mean - state.mean[0]
    residual = state.scatter[0][0] + state.n * delta * delta
    # The centered residual is at least S, even when dependency in interval
    # multiplication makes the enclosure's lower endpoint negative.
    residual = Interval(max(state.scatter[0][0], residual.lo), residual.hi)
    if residual.lo <= 0:
        # For a finite singular branch the mean cannot cross the sample mean.
        refuse(
            _SINGULAR_CANDIDATE_RESIDUAL,
            n=state.n,
            mean=state.mean[0],
            scatter=state.scatter[0][0],
            candidate=mean,
            residual=residual,
        )
    return -Fraction(state.n, 2) * (1 + log_interval(residual / state.n, precision=precision))


def _scalar_singular(
    control: GaussianState, treatment: GaussianState, ratio: Fraction, alternative: Alternative
) -> bool:
    if control.n and control.scatter[0][0] == 0 and control.mean[0] >= 0:
        return True
    if not treatment.n or treatment.scatter[0][0] != 0:
        return False
    m = treatment.mean[0]
    if alternative == "two-sided":
        return (ratio == 0 and m == 0) or (ratio != 0 and m / ratio >= 0)
    if alternative == "greater":
        return ratio > 0 or m <= 0
    return ratio < 0 or m >= 0


def _scalar_null(
    control: GaussianState,
    treatment: GaussianState,
    ratio: Fraction,
    alternative: Alternative,
    precision: int,
    width: Fraction,
) -> Interval:
    nc, nt = control.n, treatment.n
    mc, mt = control.mean[0], treatment.mean[0]
    c = (control.scatter[0][0] + nc * mc * mc, -2 * nc * mc, Fraction(nc)) if nc else (_ONE,)
    t = (
        (treatment.scatter[0][0] + nt * mt * mt, -2 * nt * ratio * mt, nt * ratio * ratio)
        if nt
        else (_ONE,)
    )
    # Cleared derivative of g: nc²(u-mc)At + nt²r(ru-mt)Ac.
    polynomial = _add(
        _scale(_multiply((-mc, _ONE), t), nc * nc),
        _scale(_multiply((-mt, ratio), c), nt * nt * ratio),
    )

    def evaluate(u: Interval) -> Interval:
        return _scalar_log(control, u, precision) + _scalar_log(treatment, ratio * u, precision)

    candidates = [evaluate(Interval.exact(0))]
    candidates.extend(_candidate(evaluate, root) for root in _roots(polynomial, _ZERO, None, width))
    # A varying nonempty arm makes the +infinity tail -infinity. Otherwise
    # the whole equality profile is constant and u=0 already represents it.
    if alternative != "two-sided":
        boundary = min(mt, _ZERO) if alternative == "greater" else max(mt, _ZERO)
        candidates.append(
            _scalar_log(control, Interval.exact(0), precision)
            + _scalar_log(treatment, Interval.exact(boundary), precision)
        )
        feasible = mt <= ratio * mc if alternative == "greater" else mt >= ratio * mc
        if (not treatment.n or feasible) and mc >= 0:
            candidates.append(
                _scalar_log(control, Interval.exact(mc), precision)
                + _scalar_log(treatment, Interval.exact(mt), precision)
            )
    return _maximum(candidates)


@dataclass(frozen=True)
class _Ray:
    state: GaussianState
    slope: Fraction
    denominator: _Polynomial
    numerator: _Polynomial
    stationary: _Polynomial
    positivity: tuple[Fraction, Fraction]
    vertex: Fraction
    floor: Fraction


def _ray(state: GaussianState, slope: Fraction) -> _Ray:
    if not state.n:
        return _Ray(state, slope, (_ONE,), (_ONE,), (), (_ZERO, _ZERO), _ZERO, _ONE)
    a, mean_d = state.mean
    c, cross, e = state.scatter[0][0], state.scatter[0][1], state.scatter[1][1]
    delta = _det(state.scatter)
    adj = _adjugate(state.scatter)
    vertex = (
        sum(state.mean[i] * adj[i][j] * state.mean[j] for i in range(2) for j in range(2)) / delta
    )
    b, d, effective_e = slope * mean_d, slope * cross, slope * slope * e
    denominator = (c, -2 * d, effective_e)
    numerator = _add(denominator, _scale(_multiply((a, -b), (a, -b)), state.n))
    u, w = a * d - b * c, b * d - a * effective_e
    stationary = _multiply((a, -b), (u, w))
    positivity = (c * mean_d - cross * a, slope * (e * a - cross * mean_d))
    return _Ray(
        state,
        slope,
        denominator,
        numerator,
        stationary,
        positivity,
        vertex,
        delta / e if slope else c,
    )


def _ray_residual(ray: _Ray, x: Interval, *, interior: bool) -> Interval:
    if not ray.state.n or not interior:
        return Interval.exact(ray.vertex)
    denominator = _polynomial(ray.denominator, x)
    denominator = Interval(max(ray.floor, denominator.lo), denominator.hi)
    difference = ray.state.mean[0] - ray.slope * ray.state.mean[1] * x
    square = difference * difference
    square = Interval(max(_ZERO, square.lo), square.hi)
    return square / denominator


def _ratio_log(state: GaussianState, residual: Interval, precision: int) -> Interval:
    if not state.n:
        return Interval.exact(0)
    n = state.n
    return (
        -n
        + n * log_interval(Fraction(n), precision=precision)
        - Fraction(n, 2) * log_interval(_det(state.scatter), precision=precision)
        - Fraction(n, 2) * log_interval(1 + n * residual, precision=precision)
    )


def _ray_tail(ray: _Ray, *, interior: bool) -> Fraction:
    if not ray.state.n or not interior:
        return ray.vertex
    if not ray.slope:
        return ray.state.mean[0] ** 2 / ray.state.scatter[0][0]
    return ray.state.mean[1] ** 2 / ray.state.scatter[1][1]


def _ratio_candidates(
    control: _Ray,
    treatment: _Ray,
    *,
    constant_treatment: Fraction | None,
    lower: Fraction,
    upper: Fraction | None,
    precision: int,
    width: Fraction,
) -> list[Interval]:
    """Design §6: Σj nj² Hj Π(k!=j) Dk Nk has degree at most six.

    Only interior rays enter this polynomial. Constant faces retain their
    likelihood contribution, and every affine cut and tail is evaluated.
    """
    rays = (control, treatment)
    cuts = {lower}
    if upper is not None:
        cuts.add(upper)
    for index, ray in enumerate(rays):
        if index == 1 and constant_treatment is not None:
            continue
        a, b = ray.positivity
        if b:
            cut = -a / b
            if cut > lower and (upper is None or cut < upper):
                cuts.add(cut)
    endpoints = sorted(cuts)
    domains: list[tuple[Fraction, Fraction | None]] = list(
        zip(endpoints[:-1], endpoints[1:], strict=True)
    )
    if upper is None:
        domains.append((endpoints[-1], None))
    if lower == upper:
        domains = [(lower, upper)]
    candidates = []
    for left, right in domains:
        sample = (left + right) / 2 if right is not None else left + 1
        active = tuple(ray.positivity[0] + ray.positivity[1] * sample > 0 for ray in rays)

        def evaluate(x: Interval, flags: tuple[bool, ...] = active) -> Interval:
            qc = _ray_residual(control, x, interior=flags[0])
            qt = (
                _ray_residual(treatment, x, interior=flags[1])
                if constant_treatment is None
                else Interval.exact(constant_treatment)
            )
            return _ratio_log(control.state, qc, precision) + _ratio_log(
                treatment.state, qt, precision
            )

        varying = [
            i
            for i, flag in enumerate(active)
            if flag and rays[i].state.n and not (i == 1 and constant_treatment is not None)
        ]
        polynomial: _Polynomial = ()
        for i in varying:
            term = _scale(rays[i].stationary, rays[i].state.n ** 2)
            for j in varying:
                if j != i:
                    term = _multiply(term, _multiply(rays[j].denominator, rays[j].numerator))
            polynomial = _add(polynomial, term)
        candidates.append(evaluate(Interval.exact(left)))
        if right is not None:
            candidates.append(evaluate(Interval.exact(right)))
        else:
            qt = (
                _ray_tail(treatment, interior=active[1])
                if constant_treatment is None
                else constant_treatment
            )
            candidates.append(
                _ratio_log(
                    control.state, Interval.exact(_ray_tail(control, interior=active[0])), precision
                )
                + _ratio_log(treatment.state, Interval.exact(qt), precision)
            )
        # An identically zero derivative is a constant branch, represented by
        # its endpoints; nonzero branches require every distinct real root.
        for root in _roots(polynomial, left, right, width):
            candidates.append(_candidate(evaluate, root))
    return candidates


def _linear_domain(a: Fraction, b: Fraction) -> tuple[Fraction, Fraction | None] | None:
    """Intersection of x>=0 with a+b*x>=0, including its finite closure."""
    if not b:
        return (_ZERO, None) if a >= 0 else None
    cut = -a / b
    if b > 0:
        return max(_ZERO, cut), None
    return (_ZERO, cut) if cut >= 0 else None


def _bivariate_null(
    control: GaussianState,
    treatment: GaussianState,
    ratio: Fraction,
    alternative: Alternative,
    precision: int,
    width: Fraction,
) -> Interval:
    c, t = _ray(control, _ONE), _ray(treatment, ratio)
    candidates = _ratio_candidates(
        c, t, constant_treatment=None, lower=_ZERO, upper=None, precision=precision, width=width
    )
    if alternative != "two-sided":
        direction = -1 if alternative == "greater" else 1
        if treatment.n:
            adj = _adjugate(treatment.scatter)
            projection = direction * (adj[0][0] * treatment.mean[0] + adj[0][1] * treatment.mean[1])
            horizontal = t.vertex - max(_ZERO, projection) ** 2 / (
                adj[0][0] * _det(treatment.scatter)
            )
        else:
            horizontal = _ZERO
        candidates.extend(
            _ratio_candidates(
                c,
                t,
                constant_treatment=horizontal,
                lower=_ZERO,
                upper=None,
                precision=precision,
                width=width,
            )
        )
        # The observed treatment mean is feasible inside the cone precisely
        # when mD>=0 and direction*(mY-r*x*mD)>=0.
        if treatment.mean[1] >= 0:
            domain = _linear_domain(
                direction * treatment.mean[0], -direction * ratio * treatment.mean[1]
            )
            if domain is not None:
                candidates.extend(
                    _ratio_candidates(
                        c,
                        t,
                        constant_treatment=_ZERO,
                        lower=domain[0],
                        upper=domain[1],
                        precision=precision,
                        width=width,
                    )
                )
    return _maximum(candidates)


def gaussian_evidence(
    control: GaussianState,
    treatment: GaussianState,
    prior_control: GaussianPrior,
    prior_treatment: GaussianPrior,
    *,
    ratio: Fraction,
    alternative: Alternative,
    max_error: Fraction = Fraction(1, 10**12),
) -> LikelihoodCertificate:
    """Conjugate Q divided by the supremum-null Gaussian likelihood.

    Scalar: NIG Q and the full cubic, positive-control boundary, and feasible MLE.
    Its reduced profile is -n/2*[1+log((S+n*(m-mu)²)/n)].
    Bivariate: NIW Q, degree-six coupled rays, quadratic single rays, affine
    clipping cuts, tails, and all one-sided cone faces. Its reduced profile is
    -n+n*log(n)-n/2*log|S|-n/2*log(1+n*q), with q the clipped ray residual.
    Population control mean (scalar) or control ratio (vector) is positive;
    treatment need not be positive. Both vector denominator means are positive.
    Supremization includes closures; _gaussian_predictive gives the Q formulas.
    """
    ratio, max_error = _arguments(ratio, alternative, max_error)
    dimension = len(control.mean)
    if any(len(obj.mean) != dimension for obj in (treatment, prior_control, prior_treatment)):
        refuse(
            _STATE_PRIOR_DIMENSION_MISMATCH,
            control_dimension=dimension,
            treatment_dimension=len(treatment.mean),
            prior_control_dimension=len(prior_control.mean),
            prior_treatment_dimension=len(prior_treatment.mean),
        )
    if not control.n and not treatment.n:
        return _empty_certificate()

    def predictive(precision: int) -> Interval:
        return _gaussian_predictive(control, prior_control, precision) + _gaussian_predictive(
            treatment, prior_treatment, precision
        )

    if dimension == 1 and _scalar_singular(control, treatment, ratio, alternative):
        return LikelihoodCertificate(
            "zero",
            predictive(60),
            None,
            None,
            "null likelihood is unbounded at a feasible zero scalar residual",
        )
    if dimension == 2 and any(
        state.n and _det(state.scatter) == 0 for state in (control, treatment)
    ):
        return LikelihoodCertificate(
            "zero",
            predictive(60),
            None,
            None,
            "temporary singular-prefix abstention: bivariate scatter is not positive definite",
        )
    optimizer = _scalar_null if dimension == 1 else _bivariate_null
    return _refine(
        predictive, lambda p, w: optimizer(control, treatment, ratio, alternative, p, w), max_error
    )
