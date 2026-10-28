from datetime import UTC, datetime

import pytest

from increment.errors import CodedError
from increment.store import Run


def test_run_rejects_missing_roster():
    with pytest.raises(CodedError) as error:
        Run(
            experiment_id="exp",
            run_id="r1",
            analysis_spec_hash="spec",
            execution_identity_hash="exec",
            registration_hash=None,
            expected_roster=(),
            cells=(),
            metadata=None,
            captured_at=datetime.now(UTC),
            committed_at=datetime.now(UTC),
        )
    assert error.value.code == "store.run.roster_mismatch"
