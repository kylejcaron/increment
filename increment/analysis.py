"""Public analysis constructors and readouts over experiment builders."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from types import MethodType
from typing import cast

from increment import builders
from increment._analysis_execution import Analysis as _Analysis
from increment._analysis_execution import fit_predeclared_adjustment

__all__ = ["Analysis", "fit_predeclared_adjustment"]


def _facade_builder[**P](
    builder_name: str, implementation: Callable[P, _Analysis]
) -> classmethod[Analysis, P, Analysis]:
    @wraps(cast(MethodType, implementation).__func__)
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

    return classmethod(construct)


class Analysis(_Analysis):
    """Analyse an experiment built from definitions, frames or retained moments."""

    from_unit_day_artifact = _facade_builder(
        "from_unit_day_artifact", _Analysis._from_unit_day_artifact
    )
    from_definitions = _facade_builder("from_definitions", _Analysis._from_definitions)
    from_unit_summary = _facade_builder("from_unit_summary", _Analysis._from_unit_summary)
    from_unit_panel = _facade_builder("from_unit_panel", _Analysis._from_unit_panel)
    from_moments = _facade_builder("from_moments", _Analysis._from_moments)
    from_switchback_panel = _facade_builder(
        "from_switchback_frame", _Analysis._from_switchback_panel
    )
