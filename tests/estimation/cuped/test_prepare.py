import copy
import pickle
from types import SimpleNamespace

from increment.estimation import Method
from increment.estimation.cuped import (
    BreakoutCell,
    BreakoutIdentity,
    MethodFitAnswer,
    MethodFitRequest,
    prepare_multicovariate_fit,
)
from increment.semantics.models import MeanMetric


def test_prepare_selects_exact_breakout_cell_and_local_fallback():
    metric = MeanMetric(name="revenue", entity="user", fact="revenue")
    method = Method.cuped_multicovariate(name="mixed", covariates="segment")
    breakout = BreakoutIdentity("segment", "table.segment")
    first = BreakoutCell(breakout, "A", False)
    second = BreakoutCell(breakout, "B", False)
    base = {
        "metric": metric,
        "method": method,
        "control": "control",
        "arm_order": ("control", "treatment"),
        "population": "assigned",
    }
    answer_a = MethodFitAnswer(origin="retained_fitted", envelope=SimpleNamespace(label="A"))
    answer_b = MethodFitAnswer(origin="warehouse", envelope=SimpleNamespace(label="B"))

    prepared = prepare_multicovariate_fit(
        method,
        (
            (MethodFitRequest(**base, cell=first), answer_a),
            (MethodFitRequest(**base, cell=second), answer_b),
        ),
        cell=second,
    )

    assert prepared.origin == "warehouse"
    assert prepared.envelope is answer_b.envelope
    for restored in (pickle.loads(pickle.dumps(prepared)), copy.deepcopy(prepared)):
        assert restored.origin == prepared.origin
        envelope = restored.envelope
        assert envelope is not None
        assert envelope.label == "B"
    local = prepare_multicovariate_fit(method, (), cell=second)
    assert local.origin == "local"
    assert local.envelope is None and local.cells is None
