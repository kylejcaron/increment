"""Primitive evidence recordings: one public ``run`` answered again without its warehouse.

`RecordingProvider` wraps a scenario's built-in provider, binds through the custom-provider
route and keeps every primitive answer under its scope and request digest. `ReplayProvider`
binds the same way and answers the same run from the recording alone, refusing any request
the recording does not hold. A `CaptureManifest` names what one recording captured; its
digest is part of the file name, so recordings of different views never overwrite each other.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from increment import Experiment, RunResult, View, run
from increment._evidence_refusals import refuse_unsupported_request
from increment.errors import CodedError
from increment.evidence import EvidenceProvider, EvidenceScopeResult, evidence_provider
from tests.analysis_factory import bind_pair, declare_views, held_source
from tests.evidence_fixtures.capture import assert_same_payload
from tests.evidence_fixtures.transcript import (
    _TYPED_MAPPING,
    _TYPED_VALUE,
    _TYPED_VALUE_FIELDS,
    _canonical,
    _digest,
    _plain,
    _request_type,
    request_digest,
    typed_payload,
    typed_value,
)
from tests.parity_harness.cases import (
    PARITY_CASES,
    ParityCase,
    _close_parity_binding,
    _retention_breakout_cohorts_case,
)

if TYPE_CHECKING:
    from increment.evidence import (
        BreakoutMethodMomentsRequest,
        ComplianceRequest,
        ComplianceSeriesRequest,
        CountsRequest,
        EvidenceScope,
        MethodFitRequest,
        MomentsRequest,
        SequentialCaptureRequest,
        UnitRowsRequest,
    )

RECORDINGS_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "evidence_identity" / "primitive-recordings"
)
_SCHEMA_VERSION = 1
_ROOT_SCOPE = "root"
_TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "manifest", "providers", "scopes", "records", "records_sha256"}
)
_MANIFEST_FIELDS = frozenset({"scenario_id", "ingress", "views", "binding_digest"})
_SCOPE_FIELDS = frozenset(
    {"parent", "request_type", "request_digest", "request", "provider", "result", "streams"}
)
_RECORD_FIELDS = frozenset(
    {"scope", "request_type", "request_digest", "request", "outcome", "answer"}
)
# Optional provider operations, offered on replay only where the recording answered them.
_OPTIONAL_OPERATIONS = {
    "assignment": "AssignmentRequest",
    "sequential_snapshot": "SequentialSnapshotRequest",
}


def view_identity(view: View) -> str:
    """One view's canonical declaration, from which `View.from_payload` rebuilds it."""
    return _canonical(view.canonical_payload())


@dataclass(frozen=True, slots=True)
class CaptureManifest:
    """What one recording captured: a scenario ingress, its ordered views, and its binding.

    ``binding_digest`` is the experiment digest `bind_pair` produces for the recorded
    provider's declaration (`_binding_digest`); a replay bound to another declaration stops
    before it reads.
    """

    scenario_id: str
    ingress: str
    views: tuple[str, ...]
    binding_digest: str

    @property
    def digest8(self) -> str:
        return _digest(self.to_payload())[:8]

    def file_name(self) -> str:
        return f"{self.scenario_id}__{self.ingress}__{self.digest8}.json.gz"

    def scopes(self) -> tuple[View, ...]:
        return tuple(View.from_payload(json.loads(identity)) for identity in self.views)

    def to_payload(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "ingress": self.ingress,
            "views": list(self.views),
            "binding_digest": self.binding_digest,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> CaptureManifest:
        if not isinstance(payload, dict) or payload.keys() != _MANIFEST_FIELDS:
            raise ValueError("capture manifest fields are incomplete or unexpected")
        views = payload["views"]
        text = (payload["scenario_id"], payload["ingress"], payload["binding_digest"])
        if (
            not all(isinstance(value, str) and value for value in text)
            or not isinstance(views, list)
            or not views
            or not all(isinstance(view, str) for view in views)
        ):
            raise ValueError("capture manifest fields are invalid")
        return cls(
            scenario_id=payload["scenario_id"],
            ingress=payload["ingress"],
            views=tuple(views),
            binding_digest=payload["binding_digest"],
        )


class Recording:
    """Primitive answers of one run, keyed by scope and request digest.

    ``providers`` holds each distinct provider declaration (context and evidence), ``scopes``
    each entered scope's portable resolved metadata under an identity derived from its parent
    and its request digest, and ``records`` each answer or coded refusal in call order.
    """

    def __init__(
        self,
        manifest: CaptureManifest,
        *,
        providers: Mapping[str, Any] | None = None,
        scopes: Mapping[str, dict[str, Any]] | None = None,
        records: Sequence[dict[str, Any]] = (),
    ) -> None:
        self.manifest = manifest
        self.providers: dict[str, Any] = dict(providers or {})
        self.scopes: dict[str, dict[str, Any]] = dict(scopes or {})
        self.records: list[dict[str, Any]] = []
        self._index: dict[tuple[str, str], dict[str, Any]] = {}
        self._types: dict[str, set[str]] = {}
        self._declared: dict[str, tuple[Any, Any]] = {}
        for record in records:
            self._add(record)

    def _add(self, record: dict[str, Any]) -> None:
        key = (record["scope"], record["request_digest"])
        existing = self._index.get(key)
        if existing is None:
            self._index[key] = record
            self._types.setdefault(record["scope"], set()).add(record["request_type"])
            self.records.append(record)
        elif existing != record:
            raise ValueError(f"{record['request_type']} answered differently within one scope")

    def record(self, scope_id: str, request: object, outcome: str, answer: object) -> None:
        self._add(
            {
                "scope": scope_id,
                "request_type": _request_type(request),
                "request_digest": request_digest(request),
                "request": _plain(request),
                "outcome": outcome,
                "answer": typed_payload(answer),
            }
        )

    def declare_provider(self, scope_id: str, provider: Any) -> None:
        declaration = typed_payload({"context": provider.context, "evidence": provider.evidence})
        digest = _digest(declaration)
        self.providers.setdefault(digest, declaration)
        scope = self.scopes.setdefault(scope_id, _scope_entry())
        if scope["provider"] not in (None, digest):
            raise ValueError(f"scope {scope_id} yielded a different provider declaration")
        scope["provider"] = digest

    def open_scope(self, parent: str, request: EvidenceScope, result: EvidenceScopeResult) -> str:
        digest = request_digest(request)
        scope_id = _digest([parent, digest])
        entry = {
            **_scope_entry(),
            "parent": parent,
            "request_type": _request_type(request),
            "request_digest": digest,
            "request": _plain(request),
            "result": typed_payload(
                {
                    "scope": result.scope,
                    "streams": result.streams,
                    "capture_status": result.capture_status,
                    "breakout_streams": result.breakout_streams,
                    "covariate_streams": result.covariate_streams,
                    "stream_resolver": result.stream_resolver is not None,
                }
            ),
        }
        existing = self.scopes.setdefault(scope_id, entry)
        if {**existing, "provider": None, "streams": None} != entry:
            raise ValueError("one scope request resolved differently within its parent")
        return scope_id

    def resolve_streams(self, scope_id: str, streams: object) -> None:
        encoded = typed_payload(streams)
        scope = self.scopes[scope_id]
        if scope["streams"] not in (None, encoded):
            raise ValueError(f"scope {scope_id} resolved its streams differently")
        scope["streams"] = encoded

    def declared(self, scope_id: str) -> tuple[Any, Any]:
        digest = self.scopes[scope_id]["provider"]
        if digest not in self._declared:
            declaration = typed_value(self.providers[digest])
            self._declared[digest] = (declaration["context"], declaration["evidence"])
        return self._declared[digest]

    def answer(self, scope_id: str, request: object) -> dict[str, Any] | None:
        return self._index.get((scope_id, request_digest(request)))

    def request_types(self, scope_id: str) -> tuple[str, ...]:
        nested = any(scope["parent"] == scope_id for scope in self.scopes.values())
        recorded = self._types.get(scope_id, set())
        return tuple(sorted(recorded | ({"EvidenceScope"} if nested else set())))

    def to_payload(self) -> dict[str, Any]:
        records = [dict(record) for record in self.records]
        return {
            "schema_version": _SCHEMA_VERSION,
            "manifest": self.manifest.to_payload(),
            "providers": dict(self.providers),
            "scopes": {scope_id: dict(scope) for scope_id, scope in self.scopes.items()},
            "records": records,
            "records_sha256": _digest(records),
        }

    @classmethod
    def from_payload(cls, payload: Any) -> Recording:
        if not isinstance(payload, dict) or payload.keys() != _TOP_LEVEL_FIELDS:
            raise ValueError("recording fields are incomplete or unexpected")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != (
            _SCHEMA_VERSION
        ):
            raise ValueError("unsupported recording schema_version")
        providers, scopes, records = payload["providers"], payload["scopes"], payload["records"]
        if not isinstance(providers, dict) or any(
            _digest(declaration) != digest for digest, declaration in providers.items()
        ):
            raise ValueError("recording provider declarations do not match their digests")
        if (
            not isinstance(scopes, dict)
            or _ROOT_SCOPE not in scopes
            or any(
                not isinstance(scope, dict)
                or scope.keys() != _SCOPE_FIELDS
                or scope["provider"] not in (None, *providers)
                or (scope_id == _ROOT_SCOPE and scope["provider"] is None)
                or (
                    scope_id != _ROOT_SCOPE
                    and (
                        scope["parent"] not in scopes
                        or scope["request_digest"] != _declared_digest(scope)
                        or scope_id != _digest([scope["parent"], scope["request_digest"]])
                    )
                )
                for scope_id, scope in scopes.items()
            )
        ):
            raise ValueError("recording scopes are invalid")
        if (
            not isinstance(records, list)
            or any(
                not isinstance(record, dict)
                or record.keys() != _RECORD_FIELDS
                or record["scope"] not in scopes
                or record["outcome"] not in {"answer", "refusal"}
                or record["request_digest"] != _declared_digest(record)
                for record in records
            )
            or payload["records_sha256"] != _digest(records)
        ):
            raise ValueError("recording records are invalid")
        return cls(
            CaptureManifest.from_payload(payload["manifest"]),
            providers=providers,
            scopes={scope_id: dict(scope) for scope_id, scope in scopes.items()},
            records=records,
        )


def _scope_entry() -> dict[str, Any]:
    return dict.fromkeys(_SCOPE_FIELDS)


def _declared_digest(entry: Mapping[str, Any]) -> str:
    """`request_digest` of the request an entry's plain declaration records."""
    return _digest([entry["request_type"], entry["request"]])


class RecordingProvider(EvidenceProvider):
    """Answer each primitive from the wrapped provider, keeping its typed answer.

    A scope yields a nested recording provider over the wrapped scope's provider, so every
    scoped read is kept under that scope's identity.
    """

    def __init__(self, provider: Any, recording: Recording, scope_id: str = _ROOT_SCOPE) -> None:
        self._provider = provider
        self._recording = recording
        self._scope_id = scope_id
        self.evidence = provider.evidence
        self.context = provider.context
        recording.declare_provider(scope_id, provider)
        for operation in _OPTIONAL_OPERATIONS:
            if callable(getattr(provider, operation, None)):
                setattr(self, operation, self._recorded(operation))

    def _recorded(self, operation: str) -> Callable[[object], Any]:
        def answer(request: object) -> Any:
            return self._answer(operation, request)

        return answer

    def _answer(self, operation: str, request: object) -> Any:
        try:
            answer = getattr(self._provider, operation)(request)
        except CodedError as refusal:
            self._recording.record(self._scope_id, request, "refusal", refusal)
            raise
        self._recording.record(self._scope_id, request, "answer", answer)
        return answer

    def moments(self, request: MomentsRequest) -> Any:
        return self._answer("moments", request)

    def breakout_method_moments(self, request: BreakoutMethodMomentsRequest) -> Any:
        return self._answer("breakout_method_moments", request)

    def unit_rows(self, request: UnitRowsRequest) -> Any:
        return self._answer("unit_rows", request)

    def method_fit(self, request: MethodFitRequest) -> Any:
        return self._answer("method_fit", request)

    def compliance(self, request: ComplianceRequest) -> Any:
        return self._answer("compliance", request)

    def compliance_series(self, request: ComplianceSeriesRequest) -> Any:
        return self._answer("compliance_series", request)

    def counts(self, request: CountsRequest) -> Any:
        return self._answer("counts", request)

    def capture_sequential(self, request: SequentialCaptureRequest) -> Any:
        return self._answer("capture_sequential", request)

    def scope(self, request: EvidenceScope) -> EvidenceScopeResult:
        recording = self._recording
        try:
            result = self._provider.scope(request)
        except CodedError as refusal:
            recording.record(self._scope_id, request, "refusal", refusal)
            raise
        scope_id = recording.open_scope(self._scope_id, request, result)
        entered: list[Any] = []

        @contextmanager
        def recorded() -> Iterator[RecordingProvider]:
            with ExitStack() as stack:
                try:
                    pinned = stack.enter_context(result.source)
                except CodedError as refusal:
                    # Entering refused: kept in the scope it would have opened.
                    recording.record(scope_id, request, "refusal", refusal)
                    raise
                entered.append(pinned)
                yield RecordingProvider(evidence_provider(pinned), recording, scope_id)

        def resolve(_scoped: object) -> Any:
            # The wrapped scope resolves its streams from its own pinned provider.
            streams = result.stream_resolver(entered[-1])
            recording.resolve_streams(scope_id, streams)
            return streams

        return EvidenceScopeResult(
            recorded(),
            result.scope,
            result.streams,
            None if result.stream_resolver is None else resolve,
            result.capture_status,
            result.breakout_streams,
            result.covariate_streams,
        )


class ReplayProvider(EvidenceProvider):
    """Answer one recorded run's primitives from its recording alone.

    A request the recording does not hold is refused with the package's unsupported-request
    code; the optional operations are offered only where the recording answered them.
    """

    def __init__(self, recording: Recording, scope_id: str = _ROOT_SCOPE) -> None:
        self._recording = recording
        self._scope_id = scope_id
        self.context, self.evidence = recording.declared(scope_id)
        recorded = recording.request_types(scope_id)
        for operation, request_type in _OPTIONAL_OPERATIONS.items():
            if request_type in recorded:
                setattr(self, operation, self._answer)

    def _refuse(self, request: object) -> NoReturn:
        refuse_unsupported_request(self, request, self._recording.request_types(self._scope_id))

    def _raise_recorded(self, request: object, scope_id: str) -> NoReturn:
        """Raise the refusal recorded for ``request`` in ``scope_id``, else refuse it."""
        record = self._recording.answer(scope_id, request)
        if record is not None and record["outcome"] == "refusal":
            raise typed_value(record["answer"])
        self._refuse(request)

    def _answer(self, request: object) -> Any:
        record = self._recording.answer(self._scope_id, request)
        if record is None or record["outcome"] == "refusal":
            self._raise_recorded(request, self._scope_id)
        return typed_value(record["answer"])

    def moments(self, request: MomentsRequest) -> Any:
        return self._answer(request)

    def breakout_method_moments(self, request: BreakoutMethodMomentsRequest) -> Any:
        return self._answer(request)

    def unit_rows(self, request: UnitRowsRequest) -> Any:
        return self._answer(request)

    def method_fit(self, request: MethodFitRequest) -> Any:
        return self._answer(request)

    def compliance(self, request: ComplianceRequest) -> Any:
        return self._answer(request)

    def compliance_series(self, request: ComplianceSeriesRequest) -> Any:
        return self._answer(request)

    def counts(self, request: CountsRequest) -> Any:
        return self._answer(request)

    def capture_sequential(self, request: SequentialCaptureRequest) -> Any:
        return self._answer(request)

    def scope(self, request: EvidenceScope) -> EvidenceScopeResult:
        recording = self._recording
        scope_id = _digest([self._scope_id, request_digest(request)])
        entry = recording.scopes.get(scope_id)
        if entry is None:
            self._raise_recorded(request, self._scope_id)
        result = typed_value(entry["result"])

        @contextmanager
        def replayed() -> Iterator[ReplayProvider]:
            if entry["provider"] is None:
                self._raise_recorded(request, scope_id)
            yield ReplayProvider(recording, scope_id)

        def resolve(_scoped: object) -> Any:
            if entry["streams"] is None:
                self._refuse(request)
            return typed_value(entry["streams"])

        return EvidenceScopeResult(
            replayed(),
            result["scope"],
            result["streams"],
            resolve if result["stream_resolver"] else None,
            result["capture_status"],
            result["breakout_streams"],
            result["covariate_streams"],
        )


def _scenario(scenario_id: str) -> ParityCase:
    """The parity case a scenario id names: a registered case or a declared variant."""
    variants = (_retention_breakout_cohorts_case(single_segment=True),)
    case = next((case for case in (*PARITY_CASES, *variants) if case.id == scenario_id), None)
    if case is None:
        raise KeyError(f"unknown declared scenario: {scenario_id}")
    return case


@contextmanager
def _scenario_provider(scenario_id: str, ingress: str) -> Iterator[EvidenceProvider]:
    """The built-in provider of the scenario ingress's source, open until the block ends."""
    bound = _scenario(scenario_id).build[ingress]()
    try:
        yield evidence_provider(held_source(bound[1]))
    finally:
        _close_parity_binding(bound)


def _binding_digest(experiment: Experiment) -> str:
    """The experiment digest of a binding, leaving out the bound provider's class name.

    A recording and its replay bind different provider classes over one declaration.
    """
    declared = dict(experiment.analysis_context)
    declared.pop("source_name", None)
    return replace(experiment, analysis_context=declared).hash()


def _run(provider: EvidenceProvider, manifest: CaptureManifest, views: Sequence[View]) -> Any:

    experiment, resolver = bind_pair(provider)
    with resolver:
        if _binding_digest(experiment) != manifest.binding_digest:
            raise ValueError(f"{manifest.file_name()} declares a different binding")
        return run(declare_views(experiment, *views), resolver, views=tuple(views))


def journey_manifest(scenario_id: str, ingress: str, views: Sequence[View]) -> CaptureManifest:
    """The manifest of one journey, bound once to read its binding digest."""
    identities = tuple(map(view_identity, views))
    with _scenario_provider(scenario_id, ingress) as provider:
        unbound = CaptureManifest(scenario_id, ingress, identities, binding_digest="")
        experiment, resolver = bind_pair(RecordingProvider(provider, Recording(unbound)))
        resolver.close()
    return CaptureManifest(scenario_id, ingress, identities, _binding_digest(experiment))


def record_run(manifest: CaptureManifest) -> tuple[Recording, RunResult]:
    """Record the manifest's run on a freshly built warehouse, closed before returning."""
    recording = Recording(manifest)
    with _scenario_provider(manifest.scenario_id, manifest.ingress) as provider:
        result = _run(RecordingProvider(provider, recording), manifest, manifest.scopes())
    return recording, result


def record_journey(manifest: CaptureManifest) -> Recording:
    """Record the manifest's run afresh."""
    return record_run(manifest)[0]


def replay(recording: Recording, views: Sequence[View]) -> RunResult:
    """Run ``views`` over the recording's binding, answered only from the recording."""
    return _run(ReplayProvider(recording), recording.manifest, views)


def replay_journey(manifest: CaptureManifest) -> RunResult:
    """Replay the manifest's frozen recording with its views."""
    return replay(load_recording(manifest), manifest.scopes())


def write_recording(recording: Recording, directory: Path) -> Path:
    """Write ``recording`` under its manifest's file name; nothing else in ``directory`` changes."""
    path = directory / recording.manifest.file_name()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(_canonical(recording.to_payload()).encode(), mtime=0))
    return path


def _read_recording(path: Path) -> Recording:
    recording = Recording.from_payload(json.loads(gzip.decompress(path.read_bytes())))
    if path.name != recording.manifest.file_name():
        raise ValueError(f"{path.name} holds the recording of {recording.manifest.file_name()}")
    return recording


def load_recording(manifest: CaptureManifest, directory: Path = RECORDINGS_DIR) -> Recording:
    """The frozen recording of ``manifest``."""
    recording = _read_recording(directory / manifest.file_name())
    if recording.manifest != manifest:
        raise ValueError(f"{manifest.file_name()} holds a different manifest")
    return recording


def frozen_manifests(directory: Path = RECORDINGS_DIR) -> tuple[CaptureManifest, ...]:
    """The manifest of every frozen recording, in file-name order."""
    return tuple(_read_recording(path).manifest for path in sorted(directory.glob("*.json.gz")))


def _without_floats(value: Any) -> Any:
    if isinstance(value, float):
        return None
    if isinstance(value, list):
        return [_without_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: _without_floats(item) for key, item in value.items()}
    return value


def _row_key(row: Any) -> str:
    return _canonical(_without_floats(row))


def _entry_key(pair: list[Any]) -> str:
    return _canonical(pair[0])


def _comparable(value: Any) -> Any:
    """``value`` with mapping entries and row collections in a stable order.

    A warehouse returns rows and keyed counts in no declared order. Mapping entries sort by
    key; rows sort by their non-float values, so float noise cannot reorder them.
    """
    if isinstance(value, list):
        items = [_comparable(item) for item in value]
        if items and all(
            isinstance(item, dict) and item.keys() == {_TYPED_MAPPING} for item in items
        ):
            items.sort(key=_row_key)
        return items
    if not isinstance(value, dict):
        return value
    if value.keys() == {_TYPED_MAPPING}:
        return {_TYPED_MAPPING: sorted(_comparable(value[_TYPED_MAPPING]), key=_entry_key)}
    kind = value.get(_TYPED_VALUE)
    if value.keys() == _TYPED_VALUE_FIELDS and kind in {"proxy", "mapping"}:
        return {**value, "value": sorted(_comparable(value["value"]), key=_entry_key)}
    if value.keys() == _TYPED_VALUE_FIELDS and kind in {"pyarrow", "polars", "pandas"}:
        frame = value["value"]
        cells = zip(*frame["columns"], strict=True)
        rows = sorted(([_comparable(cell) for cell in row] for row in cells), key=_row_key)
        columns = [list(column) for column in zip(*rows, strict=True)] or frame["columns"]
        return {**value, "value": {**frame, "columns": columns}}
    return {key: _comparable(item) for key, item in value.items()}


def assert_same_recording(frozen: Recording, fresh: Recording, *, rtol: float = 1e-12) -> None:
    """The fresh recording asks the same requests and gets the same answers within ``rtol``.

    Answers compare with mapping entries and rows in a stable order (`_comparable`).
    """
    assert frozen.manifest == fresh.manifest

    def comparable(recording: Recording) -> dict[str, Any]:
        return _comparable(
            {
                "providers": recording.providers,
                "scopes": recording.scopes,
                "records": {
                    f"{scope}:{digest}": record
                    for (scope, digest), record in recording._index.items()
                },
            }
        )

    assert_same_payload(comparable(frozen), comparable(fresh), "recording", rtol=rtol)


def recorded_journeys() -> tuple[tuple[str, str, tuple[View, ...]], ...]:
    """The journeys whose recordings are frozen: scenario, ingress and views in run order."""
    return (
        (
            "audit-retention-breakout-cohorts-single-segment",
            "from_definitions",
            (View(view="run"), View(view="breakout")),
        ),
    )
