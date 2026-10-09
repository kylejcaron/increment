"""Joint cutoff/mean confidence sets and separate influence diagnostics.

Inference uses arm-wise rank events under fixed independent iid arm sizes.
The empirical statistic uses a linear quantile; its population target is
the generalized inverse of the allocation-weighted population CDF.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from increment._literals import PreferredDirection
from increment._winsor_rank import (
    RankBand,
    _mean,
)
from increment._winsor_rank import (
    _linear_cutoff as _linear_cutoff,
)
from increment._winsor_rank import (
    _mean_fraction as _mean_fraction,
)
from increment._winsor_rank import (
    pooled_max_error as pooled_max_error,
)
from increment._winsor_rank import (
    project_bands as _project_bands,
)
from increment.estimation._winsor_bootstrap import (
    bootstrap_confidence_set as bootstrap_confidence_set,
)
from increment.estimation._winsor_bootstrap import (
    fit_positive_log_pilot as fit_positive_log_pilot,
)
from increment.estimation._winsor_bootstrap import (
    full_procedure_bootstrap_reference as full_procedure_bootstrap_reference,
)
from increment.estimation._winsor_influence import (
    influence_confidence_set as influence_confidence_set,
)
from increment.estimation._winsor_influence import (
    influence_reference as influence_reference,
)
from increment.estimation._winsor_permutation import (
    conditional_permutation_test as conditional_permutation_test,
)
from increment.winsor import (
    BootstrapReference,
    InfluenceReference,
    RawArm,
    SetInterval,
    WinsorConfidenceSet,
    WinsorRawState,
    _bootstrap_reference_context,
    _rank_interval,
    _rank_region_fields,
    winsor_refuse,
)

ComputedWinsorReference = BootstrapReference | InfluenceReference
if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

    from increment._source_types import MomentSource, RawOutcomeSource
    from increment.estimation.results import LiftEstimate
    from increment.semantics.models import Metric, Winsorization


def _pandas_identity_has_missing_sentinel(native_frame: Any) -> bool:
    import pandas as pd

    for column in ("unit_id", "group_id"):
        for value in native_frame[column].array:
            if value is None or value is pd.NA or value is pd.NaT:
                return True
    return False


def raw_state_from_source(source: MomentSource, metric: Metric) -> WinsorRawState:
    """Capture raw outcomes only from the declared source."""
    return _raw_state_from_source(source, metric)


def _raw_state_from_source(
    source: MomentSource, metric: Metric, *, native: IntoDataFrame | None = None
) -> WinsorRawState:
    """Extract once through the common source protocol; no query construction."""
    import narwhals as nw

    config = validate_winsor_metric(metric)
    context = source.context
    if metric not in context.metrics:
        winsor_refuse(
            "pool_mismatch", "Metric differs from the source's immutable construction catalog."
        )
    if context.cluster is not None or getattr(context.design, "mechanism", None) != "randomized":
        winsor_refuse(
            "design_unsupported", "Pooled winsor inference requires independent randomized units."
        )
    if native is None:
        raw_source = cast("RawOutcomeSource", source)
        native = raw_source.unit_frame(metric, outcome_stage="raw")
    frame = nw.from_native(native, eager_only=True)
    raw_frame = frame.select("unit_id", "group_id", "y")
    unit_ids = raw_frame.get_column("unit_id")
    group_ids = raw_frame.get_column("group_id")
    outcomes = raw_frame.get_column("y")
    pandas_like = frame.implementation.is_pandas_like()
    unit_values = unit_ids.to_numpy() if pandas_like else None
    group_values = group_ids.to_numpy() if pandas_like else None
    if pandas_like:
        assert native is not None
        missing_identity = _pandas_identity_has_missing_sentinel(cast(Any, native))
    else:
        missing_identity = bool(unit_ids.is_null().any() or group_ids.is_null().any())
    if missing_identity:
        winsor_refuse("invalid_state", "Raw unit and arm identities must be present.")
    if unit_ids.dtype == nw.String and not pandas_like:
        duplicate_ids = int(unit_ids.n_unique()) != raw_frame.shape[0]
    else:
        if unit_values is None:
            unit_values = unit_ids.to_numpy()
        identities: set[str] = set()
        duplicate_ids = False
        for unit_id in unit_values:
            identity = str(unit_id)
            if identity in identities:
                duplicate_ids = True
                break
            identities.add(identity)
    if duplicate_ids:
        winsor_refuse("invalid_state", "Raw outcome frame must contain exactly one row per unit.")
    values = outcomes.to_numpy()
    if outcomes.is_null().any():
        winsor_refuse(
            "invalid_state", "Raw outcomes must be finite after declared missingness handling."
        )
    finite_outcomes = (
        np.isfinite(values).all()
        if values.dtype.kind in "biuf"
        else all(math.isfinite(value) for value in values)
    )
    if not finite_outcomes:
        winsor_refuse(
            "invalid_state", "Raw outcomes must be finite after declared missingness handling."
        )
    if group_values is None:
        group_values = group_ids.to_numpy()
    labels = group_values.tolist()
    if set(map(type, labels)) != {str}:
        labels = [str(label) for label in labels]
    label_array = np.asarray(labels, dtype=object)
    outcome_array = np.asarray(values, dtype=np.float64)
    captured_arms = [
        RawArm(group_id=group_id, values=tuple(outcome_array[label_array == group_id].tolist()))
        for group_id in dict.fromkeys(labels)
    ]
    del (
        labels,
        label_array,
        outcome_array,
        group_values,
        unit_values,
        values,
        unit_ids,
        group_ids,
        outcomes,
        raw_frame,
        frame,
        native,
    )
    spec = getattr(source, "_specs_by_name", {}).get(metric.name)
    assert config.upper_percentile is not None
    return WinsorRawState(
        metric=metric.name,
        study_id=context.study_id,
        population=getattr(source, "population", getattr(source, "_population", "assigned")),
        missingness=(
            spec.missing
            if spec is not None
            else f"measure-unit-inclusion-v1:{getattr(metric, 'aggregation', 'undefined')}:observable-windows"
        ),
        quantile=config.upper_percentile,
        support=config.support,
        inference=config.inference,
        arms=tuple(captured_arms),
    )


def validate_winsor_metric(metric: Metric) -> Winsorization:
    config = getattr(metric, "winsorization", None)
    if (
        config is None
        or config.upper_percentile is None
        or config.lower_percentile is not None
        or config.lower_value is not None
        or config.upper_value is not None
    ):
        winsor_refuse(
            "design_unsupported",
            "Pooled winsor inference supports a single pooled upper percentile.",
        )
    if config.inference.method == "joint-rank-projection-v1" and config.support is None:
        winsor_refuse(
            "support_required",
            "Declare lower outcome support and independent provenance for pooled winsor inference.",
        )
    return config


def joint_confidence_set(
    raw: WinsorRawState, control: str, treatment: str, alpha: float = 0.05
) -> WinsorConfidenceSet:
    """Project calibrated rank bands and validate all portable region fields."""
    reference, relative, additive, cutoff, point, additive_point = _rank_region_fields(
        raw, control, treatment, alpha
    )
    return WinsorConfidenceSet(
        raw=raw,
        control=control,
        treatment=treatment,
        alpha=alpha,
        reference=reference,
        relative=relative,
        additive=additive,
        cutoff=cutoff,
        point=point,
        additive_point=additive_point,
    )


def project_bands(
    raw: WinsorRawState,
    control: str,
    treatment: str,
    bands: Sequence[RankBand],
    cutoff_upper: float = math.inf,
) -> tuple[SetInterval, SetInterval, SetInterval]:
    """Allocation-aware projection, including atoms and both gap limits."""
    relative, additive, cutoff = _project_bands(raw, control, treatment, bands, cutoff_upper)
    return _rank_interval(relative), _rank_interval(additive), _rank_interval(cutoff)


@dataclass(frozen=True)
class InfluenceArm:
    n: int
    weight: float
    mean: float
    variance: float
    cdf: float

    def __post_init__(self):
        if (
            isinstance(self.n, bool)
            or not isinstance(self.n, int)
            or self.n < 1
            or not all(math.isfinite(x) for x in (self.weight, self.mean, self.variance, self.cdf))
            or not 0 < self.weight <= 1
            or self.variance < 0
            or not 0 <= self.cdf <= 1
        ):
            winsor_refuse("invalid_state", "Invalid population influence arm parameters.")


@dataclass(frozen=True)
class InfluenceVariance:
    fixed: float
    cutoff: float
    cross: float

    @property
    def total(self) -> float:
        return math.fsum((self.fixed, self.cutoff, self.cross))


def influence_variance(
    arms: Mapping[str, InfluenceArm],
    control: str,
    treatment: str,
    cutoff: float,
    density: float,
    contrast: Literal["log_ratio", "difference"],
) -> InfluenceVariance:
    """Full contrast-specific population influence variance; no normal guarantee."""
    if control == treatment or control not in arms or treatment not in arms:
        winsor_refuse("pool_mismatch", "Influence contrast requires two distinct pool arms.")
    if contrast not in ("log_ratio", "difference") or not math.isfinite(cutoff):
        winsor_refuse("invalid_state", "Influence contrast and cutoff must be defined.")
    total = sum(arm.n for arm in arms.values())
    if any(arm.weight != arm.n / total for arm in arms.values()):
        winsor_refuse("pool_mismatch", "Influence weights must use every arm's allocation.")
    if not math.isfinite(density) or density <= 0:
        winsor_refuse(
            "density_required", "Influence studentization requires positive pooled density."
        )
    if contrast == "log_ratio" and (arms[control].mean <= 0 or arms[treatment].mean <= 0):
        winsor_refuse("invalid_state", "Log-ratio influence requires positive means.")
    coefficients = dict.fromkeys(arms, 0.0)
    coefficients[control] = -1 / arms[control].mean if contrast == "log_ratio" else -1.0
    coefficients[treatment] = 1 / arms[treatment].mean if contrast == "log_ratio" else 1.0
    A = math.fsum(coefficients[g] * (1 - arm.cdf) for g, arm in arms.items())
    fixed, nuisance, cross = [], [], []
    for g, arm in arms.items():
        a, b = coefficients[g], A * arm.weight / density
        fixed.append(a * a * arm.variance / arm.n)
        nuisance.append(b * b * arm.cdf * (1 - arm.cdf) / arm.n)
        cross.append(2 * a * b * (1 - arm.cdf) * (cutoff - arm.mean) / arm.n)
    return InfluenceVariance(math.fsum(fixed), math.fsum(nuisance), math.fsum(cross))


def influence_studentization(
    raw: WinsorRawState,
    control: str,
    treatment: str,
    *,
    density: float,
    contrast: Literal["log_ratio", "difference"] = "log_ratio",
) -> float:
    """Centered empirical score variance, including every cutoff-pool arm."""
    raw.arm(control)
    raw.arm(treatment)
    if control == treatment:
        winsor_refuse("pool_mismatch", "Studentization requires two distinct pool arms.")
    if contrast not in ("log_ratio", "difference"):
        winsor_refuse("invalid_state", "Unknown influence contrast.")
    if density <= 0 or not math.isfinite(density):
        winsor_refuse("density_required", "A positive pooled density estimate is required.")
    c = _linear_cutoff(raw)
    means = {a.group_id: _mean(np.minimum(a.values, c)) for a in raw.arms}
    if contrast == "log_ratio" and (means[control] <= 0 or means[treatment] <= 0):
        winsor_refuse("invalid_state", "Log-ratio influence requires positive means.")
    coefs = dict.fromkeys(means, 0.0)
    coefs[control] = -1 / means[control] if contrast == "log_ratio" else -1.0
    coefs[treatment] = 1 / means[treatment] if contrast == "log_ratio" else 1.0
    cdfs = {a.group_id: float(np.mean(np.asarray(a.values) <= c)) for a in raw.arms}
    A = math.fsum(coefs[g] * (1 - cdfs[g]) for g in coefs)
    terms = []
    for arm, (_, w) in zip(raw.arms, raw.weights, strict=True):
        if len(arm.values) < 2:
            winsor_refuse("invalid_state", "Studentization requires two observations per arm.")
        y = np.asarray(arm.values)
        g = arm.group_id
        scores = coefs[g] * (np.minimum(y, c) - means[g]) + A * w / density * (cdfs[g] - (y <= c))
        terms.append(float(np.var(scores, ddof=1)) / len(y))
    if not all(math.isfinite(term) for term in terms):
        winsor_refuse("invalid_state", "Empirical influence variance exceeds numeric range.")
    return math.fsum(terms)


def build_winsor_references(
    raw: WinsorRawState,
    control: str,
    treatments: tuple[str, ...],
    *,
    validation_context: Mapping[object, object] | None = None,
) -> dict[str, ComputedWinsorReference]:
    """Build one reference per treatment with the method the raw state resolves to.

    Rank inference carries no precomputed reference and returns an empty map.
    """
    executed = raw.executed_method
    if executed == "positive-log-kernel-bootstrap-t-v1":
        from increment.estimation._winsor_bootstrap import full_procedure_bootstrap_references

        return dict(
            full_procedure_bootstrap_references(
                raw, control, treatments, validation_context=validation_context
            )
        )
    if executed == "influence-normal-v1":
        from increment.estimation._winsor_influence import influence_references

        return dict(
            influence_references(raw, control, treatments, validation_context=validation_context)
        )
    return {}


def estimate_winsor_lift(
    raw: WinsorRawState,
    control: str,
    treatment: str,
    *,
    alpha: float = 0.05,
    method: str = "unadjusted",
    method_role: Literal["decision", "sensitivity"] = "decision",
    null_lift: float = 0.0,
    null_abs: float | None = None,
    preferred_direction: PreferredDirection | None = None,
    reference: ComputedWinsorReference | None = None,
) -> LiftEstimate:
    """Build the public row without manufacturing a posterior or a standard error."""
    from increment.estimation.results import Estimate, LiftEstimate

    executed = raw.executed_method
    validation_context: dict[object, object] | None = None
    if executed == "joint-rank-projection-v1":
        if reference is not None:
            winsor_refuse("pool_mismatch", "Rank inference cannot consume a stored reference.")
        region = joint_confidence_set(raw, control, treatment, alpha)
    else:
        # One exact point summary serves the reference, the set and the row.
        _, validation_context = _bootstrap_reference_context(raw)
        if reference is None:
            reference = build_winsor_references(
                raw, control, (treatment,), validation_context=validation_context
            )[treatment]
        if (
            reference.method != executed
            or reference.raw != raw
            or (reference.control, reference.treatment) != (control, treatment)
        ):
            winsor_refuse(
                "pool_mismatch", "Stored reference differs from the requested raw contrast."
            )
        region = (
            bootstrap_confidence_set(reference, alpha, validation_context=validation_context)
            if isinstance(reference, BootstrapReference)
            else influence_confidence_set(reference, alpha, validation_context=validation_context)
        )
    cutoff = reference.observed_cutoff if reference is not None else _linear_cutoff(raw)
    point = None
    if region.point is not None:
        if region.lower is not None and region.upper is not None and alpha / 2 > 0:
            point = Estimate(
                value=region.point,
                lb=region.lower,
                ub=region.upper,
                alpha=alpha,
                level=region.level,
            )
        else:
            point = Estimate(value=region.point)
    row = {
        "metric": raw.metric,
        "group_id": treatment,
        "method": method,
        "method_role": method_role,
        "reference_kind": "confidence_set",
        "confidence_set": region,
        "lift": point,
        "abs_diff": region.additive_point,
        "abs_lb": region.additive.lower.value,
        "abs_ub": region.additive.upper.value,
        "abs_alpha": (
            region.alpha
            if region.additive.lower.value is not None and region.additive.upper.value is not None
            else None
        ),
        "null_lift": null_lift,
        "null_abs": null_abs,
        "preferred_direction": preferred_direction,
        "analysis_population": raw.population,
        "winsor_upper_percentile": raw.quantile,
        "winsor_upper_bound": cutoff,
        "winsor_control_n_lower": 0,
        "winsor_treatment_n_lower": 0,
        "winsor_control_n_upper": _count_above(raw.arm(control).values, cutoff),
        "winsor_treatment_n_upper": _count_above(raw.arm(treatment).values, cutoff),
        "winsor_control_n": len(raw.arm(control).values),
        "winsor_treatment_n": len(raw.arm(treatment).values),
    }
    return LiftEstimate.model_validate(row, context=validation_context)


def _count_above(values: tuple[float, ...], cutoff: float) -> int:
    return int(np.count_nonzero(np.fromiter(values, dtype=np.float64, count=len(values)) > cutoff))
