"""Coded refusals raised directly by ``increment.query.artifact_reader``."""

import pytest

from increment.errors import InvalidRequestError
from increment.query.artifact_reader import ArtifactMomentSource, _rows


def test_rows_rejects_an_unsupported_relation_type():
    with pytest.raises(InvalidRequestError) as exc_info:
        _rows(object())
    assert exc_info.value.code == "query.artifact_reader.snapshot_execution_returned"
    assert exc_info.value.context["type_name"] == "object"


def test_open_rejects_a_verification_other_than_lazy_digest():
    with pytest.raises(InvalidRequestError) as exc_info:
        ArtifactMomentSource.open(
            None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            expected_context=None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            verification="eager",  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        )
    assert exc_info.value.code == "query.artifact_reader.artifact_moment.verification_lazy_digest"


class _FakeContext:
    canonical_json = "{}"


class _FakeManifest:
    context = _FakeContext()
    metric_measures: tuple[object, ...] = ()
    extensions: tuple[object, ...] = ()


def test_context_rejects_a_manifest_with_no_metric_bindings():
    source = ArtifactMomentSource(
        None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        _FakeManifest(),  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        _ = source.context
    assert exc_info.value.code == "query.artifact_reader.artifact_moment.manifest_no_metric"


class _CountingContext:
    def __init__(self) -> None:
        self.exits = 0

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: object) -> None:
        self.exits += 1


def test_a_directly_constructed_source_releases_a_plain_snapshot_context_once():
    context = _CountingContext()
    source = ArtifactMomentSource(
        None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        context,
        None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        _FakeManifest(),  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    )
    assert source.closed is False
    source.close()
    source.close()
    assert (source.closed, context.exits) == (True, 1)
