"""Public analysis constructors and readouts over experiment builders."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from inspect import Parameter, signature
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from increment import builders
from increment._analysis_execution import Analysis as _Analysis
from increment._analysis_execution import fit_predeclared_adjustment

if TYPE_CHECKING:
    from ibis.backends.sql import SQLBackend

    from increment._binding import _ArtifactOpenSpec
    from increment.artifacts import SourceSnapshotEvidence
    from increment.decision import ContrastSource
    from increment.query.session import WarehouseSession
    from increment.semantics.design import Encouragement, Observational, Randomized
    from increment.semantics.models import Definitions, Experiment
    from increment.sources import EvidenceSource

__all__ = ["Analysis", "fit_predeclared_adjustment"]


def _facade_builder[**P](
    builder_name: str, implementation: Callable[P, Any]
) -> classmethod[Analysis, P, Analysis]:
    @wraps(implementation)
    def construct(cls: type[Analysis], *args: P.args, **kwargs: P.kwargs) -> Analysis:
        options = dict(kwargs)
        name = builder_name
        if builder_name == "from_definitions":
            options["_record_schema"] = False
            name = "_from_definitions"
        experiment, resolver = getattr(builders, name)(*args, **options)
        analysis = cls.__new__(cls)
        analysis.__dict__.update(resolver._analysis.__dict__)
        analysis.__dict__["_kernel_binding"] = (experiment, resolver)
        return analysis

    factory_signature = signature(implementation)
    cast(Any, construct).__signature__ = factory_signature.replace(
        parameters=(
            Parameter("cls", Parameter.POSITIONAL_OR_KEYWORD),
            *factory_signature.parameters.values(),
        ),
        return_annotation="Analysis",
    )
    return classmethod(construct)


class Analysis(_Analysis):
    """Analyse an experiment built from definitions, frames or retained moments."""

    def __init__(
        self,
        experiment_name: str,
        definitions_path: str | Path,
        con: SQLBackend,
        *,
        backend: str | None = None,
        store: Literal["auto", "always", "none"] = "auto",
        on_mixed_assignment: Literal["error", "warn", "exclude"] = "error",
        source_snapshot_evidence: SourceSnapshotEvidence | None = None,
    ) -> None:
        binding = builders._bind_definitions(
            experiment_name,
            definitions_path,
            con,
            backend=backend,
            store=store,
            on_mixed_assignment=on_mixed_assignment,
            source_snapshot_evidence=source_snapshot_evidence,
        )
        _Analysis.__init__(self, binding)

    @classmethod
    def _from_source(
        cls,
        src: EvidenceSource | ContrastSource,
        design: Randomized | Encouragement | Observational | None = None,
        *,
        defs: Definitions | None = None,
        experiment: Experiment | None = None,
        con: SQLBackend | None = None,
        session: WarehouseSession | None = None,
        experiment_name: str | None = None,
        backend: str | None = None,
        store: Literal["auto", "always", "none"] = "auto",
        on_mixed_assignment: Literal["error", "warn", "exclude"] = "error",
        artifact_open_spec: _ArtifactOpenSpec | None = None,
    ) -> Analysis:
        binding = builders.bind_source(
            src,
            design,
            defs=defs,
            experiment=experiment,
            con=con,
            session=session,
            experiment_name=experiment_name,
            backend=backend,
            store=store,
            on_mixed_assignment=on_mixed_assignment,
            artifact_open_spec=artifact_open_spec,
        )
        analysis = cls.__new__(cls)
        _Analysis.__init__(analysis, binding)
        return analysis

    from_unit_day_artifact = _facade_builder("from_unit_day_artifact", builders._bind_artifact)
    from_definitions = _facade_builder("from_definitions", builders._bind_definitions)
    from_unit_summary = _facade_builder("from_unit_summary", builders._bind_summary)
    from_unit_panel = _facade_builder("from_unit_panel", builders._bind_panel)
    from_moments = _facade_builder("from_moments", builders._bind_moments)
    from_switchback_panel = _facade_builder("from_switchback_frame", builders._bind_switchback)
