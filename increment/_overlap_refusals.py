"""Coded refusals shared by the overlapping-experiment workflow."""

from increment.errors import CapabilityError, InvalidRequestError, RefusalSpec, WireFormatError

PLAN_INVALID = RefusalSpec(
    "overlap.plan.invalid",
    InvalidRequestError,
    template="Invalid overlapping-experiment plan field {field}: {reason}.",
)
LAW_UNSUPPORTED = RefusalSpec(
    "overlap.law.unsupported",
    CapabilityError,
    template="Cannot establish independent binary assignment for {factors}: {reason}.",
)
CAPTURE_INVALID = RefusalSpec(
    "overlap.capture.invalid",
    InvalidRequestError,
    template="Invalid source receipt for {role}: {reason}.",
)
EVIDENCE_INVALID = RefusalSpec(
    "overlap.evidence.invalid",
    InvalidRequestError,
    template="Invalid overlap evidence for {role}: {reason} ({count}).",
)
FORMAT_INVALID = RefusalSpec(
    "overlap.format.invalid",
    WireFormatError,
    template="Invalid serialized overlap field {field}: {reason}.",
)
NUMERIC_RANGE = RefusalSpec(
    "overlap.numeric.range",
    InvalidRequestError,
    template="Metric {metric} has an out-of-range {quantity}.",
)

OVERLAP_REFUSALS = {
    spec.code: spec
    for spec in (
        PLAN_INVALID,
        LAW_UNSUPPORTED,
        CAPTURE_INVALID,
        EVIDENCE_INVALID,
        FORMAT_INVALID,
        NUMERIC_RANGE,
    )
}
