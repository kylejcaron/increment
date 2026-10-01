"""Experiment diagnostics — sample-ratio mismatch (SRM) detection and allocation bands.

``sample_ratio_mismatch`` performs a chi-square goodness-of-fit test of
observed group counts against expected allocation proportions.

``allocation_posterior_bands`` computes per-arm Beta posterior credible
bands on allocation share over time — a descriptive complement to the SRM
test, not a substitute for it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Literal, cast

import narwhals as nw
from narwhals.typing import IntoDataFrame
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from scipy.stats import beta as beta_dist
from scipy.stats import chi2 as chi2_dist

from increment._labels import MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL
from increment.errors import InvalidRequestError, RefusalSpec, raiser, refusals
from increment.estimation._tails import tail_isf

_ALWAYS_SRM_PREDECLARED = RefusalSpec(
    "estimation.diagnostics.always_srm_predeclared",
    InvalidRequestError,
    template="inference='always_valid' requires predeclared allocation support; {route}",
)
_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.diagnostics.expected_allocation_contain": "expected allocation cannot contain accounting labels {accounting}",
        "estimation.diagnostics.always_srm_predeclared": _ALWAYS_SRM_PREDECLARED,
        "estimation.diagnostics.alpha": "alpha must be in (0, 1), got {alpha}",
        "estimation.diagnostics.inference_always_valid": "inference must be 'always_valid' or 'fixed', got {inference!r}",
        "estimation.diagnostics.counts_whole_unit": "counts must be whole unit counts, got fractional values {fractional}",
        "estimation.diagnostics.expected_keys_do": "expected keys {expected_keys} do not match counts keys {counts_keys}",
        "estimation.diagnostics.need_least_groups": "Need at least 2 groups for SRM test",
        "estimation.diagnostics.total_observed_count": "Total observed count must be > 0",
        "estimation.diagnostics.counts_non_negative": "counts must be non-negative, got {negative}",
        "estimation.diagnostics.predeclared_allocation_support": "predeclared allocation support via expected is required for inference='always_valid'; validity also requires known, constant conditional assignment probabilities",
        "estimation.diagnostics.every_expected_share_finite": "every expected share must be finite, got {nonfinite}",
        "estimation.diagnostics.every_expected_share_positive": "every expected share must be > 0 -- a zero-share arm makes the chi-square statistic undefined for any observed count there (and negative shares are meaningless); got {expected}",
        "estimation.diagnostics.expected_share_normalization": "expected-share normalization produced nonpositive or nonfinite shares; got {invalid_normalized}",
        "estimation.diagnostics.credible_level": "credible_level must be in (0, 1), got {credible_level!r}",
        "estimation.diagnostics.prior_alpha_finite": "prior alpha must be finite and positive, got {prior_a!r}",
        "estimation.diagnostics.prior_beta_finite": "prior beta must be finite and positive, got {prior_b!r}",
        "estimation.diagnostics.credible_level_too": "credible_level={credible_level!r} is too close to 1 to resolve a tail probability in floating point -- request a less extreme credible_level",
        "estimation.diagnostics.counts_missing_columns": "counts missing required columns: {missing}",
        "estimation.diagnostics.n_cumulative_whole": "n_cumulative must be a whole unit count, got {n_raw!r} for (ds={ds!r}, group_id={group_id!r}) -- silently truncating would misstate the posterior",
        "estimation.diagnostics.n_cumulative_non": "n_cumulative must be non-negative, got {n_raw!r} for (ds={ds!r}, group_id={group_id!r})",
        "estimation.diagnostics.allocation_posterior_bands": "allocation_posterior_bands computes bands for one experiment at a time (it pools n_total per ds across every row it sees), but got rows spanning multiple experiment_id values: {experiment_ids!r}. Filter to a single experiment before calling.",
        "estimation.diagnostics.duplicate_ds_group": "duplicate (ds, group_id) row {ds_group!r}: n_total would double-count this arm and every band that day would be wrong",
        "estimation.diagnostics.allocation_band_ds": "allocation band (ds={ds!r}, group_id={group_id!r}) is [{lower!r}, {upper!r}], which touches 0 or 1 -- at credible_level={credible_level!r} this posterior (Beta({a}, {b})) cannot resolve a proper interval, and a bound of exactly 0 or 1 would read as certainty about the allocation; request a less extreme credible_level",
    },
)
_raise = raiser(_REFUSALS)


class SRMResult(BaseModel):
    """Result of an allocation sample-ratio-mismatch check.

    ``fixed_p_value`` is the ordinary Pearson fixed-look p-value.
    ``log_e_value`` is the current uniform-Dirichlet mixture evidence
    for a predeclared allocation (unavailable when fixed inference
    infers equal shares from observed arms). ``inference`` states
    which quantity controls ``is_srm``.

    ``unassigned_units`` and ``mixed_assignment_units`` are accounting
    entries, not arms: they contribute no chi-square degree of freedom
    and appear in neither ``observed`` nor ``expected``.
    ``mixed_assignment_units`` counts units observed in more than one
    arm, which shrinks every arm's count symmetrically and so must be
    surfaced separately to be seen.

    ``grain`` is ``"cluster"`` when randomization happened over
    clusters, so the chi-square belongs there; ``unit_counts`` then
    carries per-arm unit counts as descriptive context only (cluster
    size imbalance earns no degree of freedom). At ``grain="unit"``,
    ``unit_counts`` is empty.

    ``min_expected_count`` is the smallest per-arm expected count
    (the minimum over arms of ``expected[k] * total``), or ``None`` when
    unavailable. ``low_expected_count`` is True when that minimum falls
    below 5, the standard Cochran rule of thumb, flagging that the fixed
    asymptotic chi-square p-value may be unreliable there.
    """

    model_config = ConfigDict(frozen=True, validate_default=True)

    inference: Literal["always_valid", "fixed"]
    chi2_stat: float
    fixed_p_value: float
    log_e_value: float | None
    df: int
    is_srm: bool
    alpha: float
    observed: Mapping[str, int]
    expected: Mapping[str, float]
    unassigned_units: int = 0
    mixed_assignment_units: int = 0
    grain: Literal["unit", "cluster"] = "unit"
    unit_counts: Mapping[str, int] = {}
    min_expected_count: float | None = None
    low_expected_count: bool = False

    @field_validator("observed", "expected", "unit_counts")
    @classmethod
    def _own_counts(cls, value: Mapping[str, int | float]) -> Mapping[str, int | float]:
        return MappingProxyType(dict(value))

    @field_serializer("observed", "expected", "unit_counts")
    def _serialize_counts(self, value: Mapping[str, int | float]) -> dict[str, int | float]:
        return dict(value)


class NotApplicable(BaseModel):
    """A diagnostic check that does not apply to the current design.

    Returned in place of a check's normal result (e.g. ``SRMResult``)
    when the design makes the check meaningless - a sample-ratio test
    presumes a target randomized allocation an observational design
    does not have.
    """

    model_config = ConfigDict(frozen=True)

    check: str
    reason: str


def _validate_srm_support(expected: Mapping[str, float] | None) -> None:
    if expected is None:
        return
    accounting = set(expected) & {UNASSIGNED_LABEL, MIXED_ASSIGNMENT_LABEL}
    if accounting:
        _raise("estimation.diagnostics.expected_allocation_contain", accounting=accounting)


def resolve_srm_expected(
    expected: dict[str, float] | None,
    *,
    allocation: dict[str, float] | None,
    inference: Literal["always_valid", "fixed"],
) -> dict[str, float] | None:
    """Select SRM allocation support and require it for anytime-valid looks."""
    resolved = expected if expected is not None else allocation
    _validate_srm_support(resolved)
    if resolved is None and inference == "always_valid":
        _raise(
            "estimation.diagnostics.always_srm_predeclared",
            route="pass expected=... or declare design.allocation",
        )
    return resolved


def complete_srm_support(
    counts: Mapping[str, int],
    *,
    expected: Mapping[str, float] | None,
) -> dict[str, int]:
    """Add zero-count declared arms without hiding observed or accounting keys."""
    if expected is None:
        return dict(counts)
    return {**dict.fromkeys(expected, 0), **counts}


def _log_dirichlet_e_value(
    counts: Mapping[str, int],
    expected: Mapping[str, float],
) -> float:
    """Uniform-Dirichlet mixture log likelihood ratio against ``expected``."""
    n_total = sum(counts.values())
    n_groups = len(counts)
    terms = [
        math.lgamma(counts[group] + 1.0)
        - math.lgamma(1.0)
        - counts[group] * math.log(expected[group])
        for group in counts
    ]
    return math.lgamma(float(n_groups)) - math.lgamma(float(n_groups + n_total)) + math.fsum(terms)


def sample_ratio_mismatch(
    counts: Mapping[str, int],
    expected: Mapping[str, float] | None = None,
    alpha: float = 0.001,
    *,
    inference: Literal["always_valid", "fixed"] = "always_valid",
    grain: Literal["unit", "cluster"] = "unit",
    unit_counts: Mapping[str, int] | None = None,
) -> SRMResult:
    """Detect allocation sample-ratio mismatch with fixed or anytime-valid evidence.

    ``inference="always_valid"`` (default) evaluates the exact
    uniform-Dirichlet mixture e-process against the normalized expected
    allocation and alarms when its log evidence reaches ``-log(alpha)``.
    The time-uniform guarantee needs cumulative prefixes with the same
    known conditional arm probabilities at every assignment; static
    marginal shares alone do not suffice, and blocked, adaptive,
    dependent, quota, exact-balance, without-replacement, and ramped or
    reset assignment streams are unsupported (independently assigned
    clusters satisfy the contract at cluster grain). ``log_e_value`` is
    stateless and may decrease, since this result retains no historical
    maximum.

    ``inference="fixed"`` uses the ordinary Pearson chi-square p-value
    for one predeclared look or a caller-managed scheduled-look alpha
    budget; without ``expected`` it falls back to equal observed-arm
    shares and ``log_e_value`` is unavailable.

    Apply to all assigned or targeted units, or to a demonstrably
    pre-treatment, arm-invariant exposure - a treatment-affected
    triggered subset is selection or telemetry evidence, not evidence
    that randomization failed.

    ``counts``' ``"(unassigned)"``/``"(mixed assignment)"`` accounting
    keys are split out and never treated as arms. ``expected`` is
    required for ``always_valid``; its keys declare the arms, and
    missing observed arms receive zero counts. ``unit_counts`` is
    descriptive per-arm context for a cluster-grain test, never a
    second allocation sample.
    """
    if not 0.0 < alpha < 1.0:
        _raise("estimation.diagnostics.alpha", alpha=alpha)
    if inference not in ("always_valid", "fixed"):
        _raise("estimation.diagnostics.inference_always_valid", inference=inference)
    unassigned = int(counts.get(UNASSIGNED_LABEL, 0))
    mixed = int(counts.get(MIXED_ASSIGNMENT_LABEL, 0))
    counts = {
        k: v for k, v in counts.items() if k not in (UNASSIGNED_LABEL, MIXED_ASSIGNMENT_LABEL)
    }
    fractional = {k: v for k, v in counts.items() if float(v) != int(v)}
    if fractional:
        _raise("estimation.diagnostics.counts_whole_unit", fractional=fractional)
    counts = {k: int(v) for k, v in counts.items()}
    expected_is_predeclared = expected is not None
    _validate_srm_support(expected)
    if expected is not None:
        unexpected_observed = set(counts) - set(expected)
        if unexpected_observed:
            _raise(
                "estimation.diagnostics.expected_keys_do",
                expected_keys=set(expected.keys()),
                counts_keys=set(counts.keys()),
            )
        counts = complete_srm_support(counts, expected=expected)

    n_groups = len(counts)
    if n_groups < 2:
        _raise("estimation.diagnostics.need_least_groups")

    total = sum(counts.values())
    if total <= 0:
        _raise("estimation.diagnostics.total_observed_count")
    negative = {k: v for k, v in counts.items() if v < 0}
    if negative:
        _raise("estimation.diagnostics.counts_non_negative", negative=negative)
    if expected is None and inference == "always_valid":
        _raise("estimation.diagnostics.predeclared_allocation_support")

    # Default: equal proportions
    if expected is None:
        expected = dict.fromkeys(counts, 1.0 / n_groups)
    else:
        nonfinite = {k: v for k, v in expected.items() if not math.isfinite(v)}
        if nonfinite:
            _raise("estimation.diagnostics.every_expected_share_finite", nonfinite=nonfinite)
        if any(v <= 0 for v in expected.values()):
            _raise("estimation.diagnostics.every_expected_share_positive", expected=expected)
        # Normalize exactly for both statistics. Scaling first keeps the total
        # finite even when individually finite shares would overflow in a sum.
        scale = max(expected.values())
        scaled_total = math.fsum(v / scale for v in expected.values())
        expected = {k: (v / scale) / scaled_total for k, v in expected.items()}
        invalid_normalized = {k: v for k, v in expected.items() if not math.isfinite(v) or v <= 0}
        if invalid_normalized:
            _raise(
                "estimation.diagnostics.expected_share_normalization",
                invalid_normalized=invalid_normalized,
            )

    # Chi-square statistic (every expected share is validated > 0 above,
    # so no term is ever dropped from the sum)
    chi2 = 0.0
    min_expected_count = math.inf
    for k in counts:
        e = expected[k] * total
        o = counts[k]
        chi2 += (o - e) ** 2 / e
        min_expected_count = min(min_expected_count, e)
    low_expected_count = min_expected_count < 5.0

    df = n_groups - 1
    # Survival function, not 1-cdf: real p-values routinely sit below
    # float epsilon of 1.0, and the magnitude is the severity signal.
    fixed_p_value = float(chi2_dist.sf(chi2, df))
    if inference == "always_valid":
        log_e_value = _log_dirichlet_e_value(counts, expected)
        is_srm = log_e_value >= -math.log(alpha)
    else:
        log_e_value = _log_dirichlet_e_value(counts, expected) if expected_is_predeclared else None
        is_srm = fixed_p_value < alpha

    return SRMResult(
        inference=inference,
        chi2_stat=chi2,
        fixed_p_value=fixed_p_value,
        log_e_value=log_e_value,
        df=df,
        is_srm=is_srm,
        alpha=alpha,
        observed=counts,
        expected=expected,
        unassigned_units=unassigned,
        mixed_assignment_units=mixed,
        grain=grain,
        unit_counts=dict(unit_counts or {}),
        min_expected_count=min_expected_count,
        low_expected_count=low_expected_count,
    )


class AllocationBand(BaseModel):
    """A Beta posterior credible band on one arm's allocation share at one ``ds``.

    Purely descriptive: unlike ``SRMResult``, this carries no pass/fail
    flag - ``sample_ratio_mismatch`` is the actual gate. ``posterior_a``/
    ``posterior_b`` are the Beta parameters the interval was derived
    from, carried so a caller can reconstruct the full posterior (they
    cannot be recovered from ``n``/``n_total`` alone without the prior,
    a call-site argument).
    """

    model_config = ConfigDict(frozen=True)

    ds: str
    group_id: str
    n: int
    n_total: int
    posterior_a: float
    posterior_b: float
    mean: float
    lower: float
    upper: float
    credible_level: float


def allocation_posterior_bands(
    counts: IntoDataFrame | Iterable[Mapping[str, Any]],
    *,
    credible_level: float = 0.95,
    prior: tuple[float, float] = (1.0, 1.0),
) -> list[AllocationBand]:
    """Compute per-arm Beta posterior credible bands on allocation share.

    ``counts`` carries ``ds``/``group_id``/``n_cumulative`` (the shape
    ``daily_exposure_counts`` produces, minus ``experiment_id``/
    ``n_daily``); accepts any narwhals-supported frame or an iterable
    of row mappings. ``prior`` is the ``Beta(a, b)`` prior on each
    arm's share (default uniform); both parameters must be finite and
    positive.

    Every ``ds`` is pooled across all arms present to compute that
    day's ``n_total`` - rows spanning more than one ``experiment_id``
    raise ``ValueError``, since silently mixing experiments sharing a
    date would inflate the total and produce spuriously narrow, wrong
    bands. Returns one band per input row, in input order.
    """
    if not (math.isfinite(credible_level) and 0.0 < credible_level < 1.0):
        _raise("estimation.diagnostics.credible_level", credible_level=credible_level)
    prior_a, prior_b = prior
    if not (math.isfinite(prior_a) and prior_a > 0.0):
        _raise("estimation.diagnostics.prior_alpha_finite", prior_a=prior_a)
    if not (math.isfinite(prior_b) and prior_b > 0.0):
        _raise("estimation.diagnostics.prior_beta_finite", prior_b=prior_b)
    # A structural check on the requested tail only: it must be a resolvable
    # probability in (0, 0.5). It cannot detect resolution loss, because every
    # representable credible_level below 1 yields a positive lower_q -- the
    # bounds themselves are judged where they are computed, below.
    lower_q = (1.0 - credible_level) / 2.0
    if not (math.isfinite(lower_q) and 0.0 < lower_q < 0.5):
        _raise("estimation.diagnostics.credible_level_too", credible_level=credible_level)

    required = ["ds", "group_id", "n_cumulative"]

    frame = nw.from_native(counts, eager_only=True, pass_through=True)
    if isinstance(frame, nw.DataFrame):
        missing = [c for c in required if c not in frame.columns]
        if missing:
            _raise("estimation.diagnostics.counts_missing_columns", missing=missing)
        rows: Iterable[Mapping[str, Any]] = frame.iter_rows(named=True)
    else:
        rows = cast("Iterable[Mapping[str, Any]]", counts)

    parsed: list[tuple[str, str, int]] = []
    experiment_ids: set[Any] = set()
    for row in rows:
        n_raw = row["n_cumulative"]
        if float(n_raw) != int(n_raw):
            _raise(
                "estimation.diagnostics.n_cumulative_whole",
                n_raw=n_raw,
                ds=row["ds"],
                group_id=row["group_id"],
            )
        if int(n_raw) < 0:
            _raise(
                "estimation.diagnostics.n_cumulative_non",
                n_raw=n_raw,
                ds=row["ds"],
                group_id=row["group_id"],
            )
        parsed.append((str(row["ds"]), str(row["group_id"]), int(n_raw)))
        if "experiment_id" in row:
            experiment_ids.add(row["experiment_id"])

    if len(experiment_ids) > 1:
        _raise(
            "estimation.diagnostics.allocation_posterior_bands",
            experiment_ids=sorted(experiment_ids),
        )

    totals: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for ds, group_id, n_k in parsed:
        if (ds, group_id) in seen:
            _raise("estimation.diagnostics.duplicate_ds_group", ds_group=(ds, group_id))
        seen.add((ds, group_id))
        totals[ds] = totals.get(ds, 0) + n_k

    bands: list[AllocationBand] = []
    for ds, group_id, n_k in parsed:
        n_total = totals[ds]
        a = prior_a + n_k
        b = prior_b + (n_total - n_k)
        dist = beta_dist(a, b)
        lower = float(dist.ppf(lower_q))
        upper = tail_isf(
            dist.isf, lower_q, what=f"allocation band (ds={ds!r}, group_id={group_id!r})"
        )
        # Judge the output, not just the requested tail: every credible_level below 1 gives a
        # positive lower_q, yet an imbalanced Beta(N+1, 1) posterior can still collapse to a
        # bound of exactly 0 or 1, falsely claiming certainty about the allocation.
        if not (0.0 < lower <= upper < 1.0):
            _raise(
                "estimation.diagnostics.allocation_band_ds",
                ds=ds,
                group_id=group_id,
                lower=lower,
                upper=upper,
                credible_level=credible_level,
                a=a,
                b=b,
            )
        bands.append(
            AllocationBand(
                ds=ds,
                group_id=group_id,
                n=n_k,
                n_total=n_total,
                posterior_a=a,
                posterior_b=b,
                mean=a / (a + b),
                lower=lower,
                upper=upper,
                credible_level=credible_level,
            )
        )
    return bands


ESTIMATION_DIAGNOSTICS_ALPHA = _REFUSALS["estimation.diagnostics.alpha"]
