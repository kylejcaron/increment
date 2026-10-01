"""Coded refusals raised directly by ``increment.query.source``."""

import pytest

from increment.errors import InvalidRequestError
from increment.query.artifact_reader import ArtifactMomentSource
from increment.query.source import open_artifact


def test_artifact_open_verification_refusal_uses_canonical_code() -> None:
    with pytest.raises(InvalidRequestError) as via_artifact_reader:
        ArtifactMomentSource.open(
            None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            expected_context=None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            verification="eager",  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(InvalidRequestError) as via_artifact_facade:
        open_artifact(
            None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            expected_context=None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            verification="eager",  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        )
    assert (
        via_artifact_reader.value.code
        == via_artifact_facade.value.code
        == "query.artifact_reader.artifact_moment.verification_lazy_digest"
    )
