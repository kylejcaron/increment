from __future__ import annotations

import pytest
from _pytest.mark import Expression, MarkMatcher

from scripts.test_changed import _marker_expression


def test_unit_marker_expression_excludes_integration_tests() -> None:
    expression = Expression.compile(_marker_expression("unit"))

    assert expression.evaluate(MarkMatcher.from_markers([]))
    assert not expression.evaluate(MarkMatcher.from_markers([pytest.mark.integration.mark]))
