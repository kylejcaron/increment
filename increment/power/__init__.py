"""Power analysis: sample-size, achieved-power, and minimum-detectable-effect.

The arm solvers invert the estimator's log-ratio variance with each arm
evaluated at its own mean under the alternative. Mean-like planning assumes
equal absolute effective arm variances; conversion and retention planning use
the Bernoulli variance shape at the implied treatment rate. Where the runtime
routes an unadjusted, unclustered, fixed-horizon conversion or retention
contrast by its counts (``conversion_inference``), planning follows the route: the
rejection probability of the runtime's union of the delta-method decision on dense counts
and the finite-sample binomial risk-ratio decision on the rest, summed over the count lattice
(``power_basis`` ``"exact"``, or ``"approximate"`` when the replay budget leaves mass
undecided and ``power`` is the certified lower figure), and the closed-form model where the
counts are dense with near certainty and the lattice is too large to enumerate
(``"asymptotic"``).
Segment-pairwise planning deliberately retains its separate baseline-only
approximation.

A valid supplied-effect result can have no admissible or numerically resolved
companion MDE. In that case ``mde_relative`` is ``None`` and
``mde_unavailable_reason`` records why.

Sequential planning is declared through ``ArmPlanningProcedure.standard(
inference=InferenceSpec(kind="asymptotic_mean"))``, the runtime's own
declaration object -- it builds ``GaussianScoreMixture`` internally, tuned
identically to the runtime's own boundary at every information fraction.
"""

from increment.estimation.arm_contract import ArmPlanningProcedure
from increment.power.core import (
    Baseline,
    PowerDesign,
    PowerResult,
    achieved_power,
    joint_q_power_fixed,
    joint_q_power_random,
    minimum_detectable_effect,
    required_sample_size,
    segment_pairwise_achieved_power,
    segment_pairwise_minimum_detectable_effect,
    segment_pairwise_required_sample_size,
)
from increment.power.curve import PowerCurve, PowerCurvePoint, power_curve
from increment.power.switchback import (
    SwitchbackBaseline,
    SwitchbackPowerResult,
    switchback_achieved_power,
    switchback_minimum_detectable_effect,
    switchback_required_blocks_or_units,
)
from increment.power.unit_cycle import (
    UnitCyclePowerLowerBoundResult,
    unit_cycle_power_lower_bound,
)

__all__ = [
    "SwitchbackBaseline",
    "SwitchbackPowerResult",
    "switchback_achieved_power",
    "switchback_minimum_detectable_effect",
    "switchback_required_blocks_or_units",
    "UnitCyclePowerLowerBoundResult",
    "unit_cycle_power_lower_bound",
    "Baseline",
    "ArmPlanningProcedure",
    "PowerDesign",
    "PowerResult",
    "PowerCurvePoint",
    "PowerCurve",
    "power_curve",
    "required_sample_size",
    "achieved_power",
    "minimum_detectable_effect",
    "segment_pairwise_required_sample_size",
    "segment_pairwise_achieved_power",
    "segment_pairwise_minimum_detectable_effect",
    "joint_q_power_fixed",
    "joint_q_power_random",
]
