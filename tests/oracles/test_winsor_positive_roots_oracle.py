"""The zero-inclusive bootstrap preserves the positive-only numerical reference.

The fixtures are generated from the construction that preceded the zero atom.
Floating-point roots and summaries use tight tolerances to allow platform-level
rounding differences; bandwidths and failure indices remain exact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "winsor_positive_roots.json").read_text()
)
_CASES = _FIXTURE["cases"]


@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_positive_outcomes_reproduce_the_frozen_bootstrap_reference(case):
    from increment.estimation._winsor_bootstrap import full_procedure_bootstrap_references
    from increment.winsor import RawArm, WinsorInferenceSpec, WinsorRawState

    raw = WinsorRawState(
        metric="revenue",
        study_id="oracle",
        missingness="error",
        quantile=case["quantile"],
        inference=WinsorInferenceSpec(stream=case["stream"]),
        arms=tuple(RawArm(group_id=g, values=tuple(v)) for g, v in case["arms"].items()),
    )
    assert raw.inference.method == "pooled-size-route-v1"
    assert raw.executed_method == "positive-log-kernel-bootstrap-t-v1"
    references = full_procedure_bootstrap_references(raw, "C", tuple(case["treatments"]))
    for treatment, expected in case["results"].items():
        reference = references[treatment]
        assert reference.method == "positive-log-kernel-bootstrap-t-v1"
        assert all(pilot.zero_count == 0 for pilot in reference.pilots)
        np.testing.assert_allclose(
            [p.bandwidth for p in reference.pilots],
            expected["bandwidths"],
            rtol=1e-12,
            atol=1e-14,
        )
        assert reference.observed_cutoff == pytest.approx(
            expected["observed_cutoff"], rel=1e-12, abs=1e-14
        )
        assert reference.pilot_cutoff == pytest.approx(
            expected["pilot_cutoff"], rel=1e-12, abs=1e-14
        )
        assert reference.log_relative.point == pytest.approx(
            expected["log_point"], rel=1e-12, abs=1e-14
        )
        assert reference.log_relative.pilot_target == pytest.approx(
            expected["log_target"], rel=1e-12, abs=1e-14
        )
        assert reference.log_relative.se == pytest.approx(expected["log_se"], rel=1e-12, abs=1e-14)
        assert reference.additive.point == pytest.approx(
            expected["additive_point"], rel=1e-12, abs=1e-14
        )
        assert reference.additive.pilot_target == pytest.approx(
            expected["additive_target"], rel=1e-12, abs=1e-14
        )
        assert reference.additive.se == pytest.approx(expected["additive_se"], rel=1e-12, abs=1e-14)
        assert list(reference.failure_indices) == expected["failure_indices"]
        np.testing.assert_allclose(
            np.asarray(reference.log_relative.roots, dtype=np.float64),
            np.asarray(expected["log_roots"], dtype=np.float64),
            rtol=1e-12,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            np.asarray(reference.additive.roots, dtype=np.float64),
            np.asarray(expected["additive_roots"], dtype=np.float64),
            rtol=1e-12,
            atol=1e-14,
        )
