"""Versioned, source-independent evidence fixtures, replayed at two levels.

Estimator-level transcripts (`transcript.py`, consumed by `replay.py` and `method_replay.py`)
freeze the compound answers one scenario ingress gives through `request_evidence` and feed
them to pure estimators. Use them to assert estimator math on frozen inputs: that the same
inputs give the same rows across consumers.

Journey-level primitive recordings (`recording.py`) freeze the primitive answers of one public
``run`` through the custom-provider seam and replay that run without its warehouse. Use them
to assert what downstream logic does with those answers: readout rows, breakouts, refusal
routing, multiplicity and the rest of assembly. A test that asserts SQL, a loader or an
artifact keeps its warehouse.
"""

from tests.evidence_fixtures.capture import (
    CaptureKey,
    EvidenceRecord,
    RequestCoordinates,
    assert_same_evidence,
    capture_acquired_evidence,
    method_identity,
)

__all__ = [
    "CaptureKey",
    "EvidenceRecord",
    "RequestCoordinates",
    "assert_same_evidence",
    "capture_acquired_evidence",
    "method_identity",
]
