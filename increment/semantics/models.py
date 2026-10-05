"""Semantic models for the increment A/B testing framework.

Structural vocabulary - fact sources, metrics, exposures, experiments -
for the metric types in docs/guides/metric-types.md.

Pure data layer: defines schema and validates internal consistency.
Must not import ibis, build queries, or touch a database.
"""

import math
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, NoReturn, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from increment._finite_sample_refusals import (
    FINITE_SAMPLE_METRIC_TYPE,
    refuse_finite_sample_cuped,
)
from increment._immutable import _FrozenMapping
from increment._literals import (
    ALTERNATIVE_VALUES,
    Alternative,
    ConversionInference,
    Correction,
    PreferredDirection,
)
from increment.errors import CodedModel, CodedValidationMixin, DefinitionError, RefusalSpec
from increment.semantics.design import (
    ExclusionRestriction,
    UptakeSpec,
    _freeze_allocation,
    _serialize_allocation,
    _validate_allocation_semantics,
)
from increment.semantics.rational import DeclaredRational
from increment.semantics.sequential import (
    ASYMPTOTIC_LAWS,
    PredeclaredAdjustment,
    SequentialCompliancePolicy,
    SequentialRegistration,
    refuse_segmented_family_compliance,
)
from increment.winsor import WinsorInferenceSpec, WinsorSupport

if TYPE_CHECKING:
    from increment.semantics.design import Encouragement, Observational, Randomized


def _render_definition_message(message: str, **_: object) -> str:
    return message


DEFINITION_REFUSALS: dict[str, RefusalSpec] = {
    "definition.active.metric_window_days": RefusalSpec(
        "definition.active.metric_window_days", DefinitionError, _render_definition_message
    ),
    "definition.already_canonical_json": RefusalSpec(
        "definition.already_canonical_json", DefinitionError, _render_definition_message
    ),
    "definition.analysis.appears_both_metric": RefusalSpec(
        "definition.analysis.appears_both_metric", DefinitionError, _render_definition_message
    ),
    "definition.analysis.appears_more_once": RefusalSpec(
        "definition.analysis.appears_more_once", DefinitionError, _render_definition_message
    ),
    "definition.artifact.context_format_disagrees": RefusalSpec(
        "definition.artifact.context_format_disagrees",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.artifact.context_sha256_does": RefusalSpec(
        "definition.artifact.context_sha256_does", DefinitionError, _render_definition_message
    ),
    "definition.artifact_datetimes_timezone": RefusalSpec(
        "definition.artifact_datetimes_timezone", DefinitionError, _render_definition_message
    ),
    "definition.artifact_extension.definition_sha256_does": RefusalSpec(
        "definition.artifact_extension.definition_sha256_does",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.artifact_extension.source_provenance_sha256": RefusalSpec(
        "definition.artifact_extension.source_provenance_sha256",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.artifact_extension.unsupported_version": RefusalSpec(
        "definition.artifact_extension.unsupported_version",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.artifact_json_values": RefusalSpec(
        "definition.artifact_json_values", DefinitionError, _render_definition_message
    ),
    "definition.artifact_relation.primary_key_contain": RefusalSpec(
        "definition.artifact_relation.primary_key_contain",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.artifact_relation.primary_key_members": RefusalSpec(
        "definition.artifact_relation.primary_key_members",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.artifact_relation.row_count_integer": RefusalSpec(
        "definition.artifact_relation.row_count_integer",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.assignment_populations_contain": RefusalSpec(
        "definition.assignment_populations_contain", DefinitionError, _render_definition_message
    ),
    "definition.assignment_populations_include": RefusalSpec(
        "definition.assignment_populations_include", DefinitionError, _render_definition_message
    ),
    "definition.base_relations.exposures_primary_key": RefusalSpec(
        "definition.base_relations.exposures_primary_key",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.base_relations.exposures_relation_use": RefusalSpec(
        "definition.base_relations.exposures_relation_use",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.base_relations.measure_stats_primary": RefusalSpec(
        "definition.base_relations.measure_stats_primary",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.base_relations.measure_stats_relation": RefusalSpec(
        "definition.base_relations.measure_stats_relation",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.base_relations.share_artifact_id": RefusalSpec(
        "definition.base_relations.share_artifact_id", DefinitionError, _render_definition_message
    ),
    "definition.base_relations.share_generation_id": RefusalSpec(
        "definition.base_relations.share_generation_id", DefinitionError, _render_definition_message
    ),
    "definition.breakout.ambiguous_source": RefusalSpec(
        "definition.breakout.ambiguous_source", DefinitionError, _render_definition_message
    ),
    "definition.check_observational.covariate_could": RefusalSpec(
        "definition.check_observational.covariate_could",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.check_observational.covariate_property_found": RefusalSpec(
        "definition.check_observational.covariate_property_found",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.check_observational.covariate_source_found": RefusalSpec(
        "definition.check_observational.covariate_source_found",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.check_observational.covariate_unit_but": RefusalSpec(
        "definition.check_observational.covariate_unit_but",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.contain_control_characters": RefusalSpec(
        "definition.contain_control_characters", DefinitionError, _render_definition_message
    ),
    "definition.cuped_preperiod.window_bounds_integers": RefusalSpec(
        "definition.cuped_preperiod.window_bounds_integers",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.cuped_preperiod.window_satisfy_window": RefusalSpec(
        "definition.cuped_preperiod.window_satisfy_window",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.day_boundary_accepts": RefusalSpec(
        "definition.day_boundary_accepts", DefinitionError, _render_definition_message
    ),
    "definition.dim.duplicate_property_name": RefusalSpec(
        "definition.dim.duplicate_property_name", DefinitionError, _render_definition_message
    ),
    "definition.dim.source_declares_no": RefusalSpec(
        "definition.dim.source_declares_no", DefinitionError, _render_definition_message
    ),
    "definition.dim.source_no_validity": RefusalSpec(
        "definition.dim.source_no_validity", DefinitionError, _render_definition_message
    ),
    "definition.dim.source_versioned_validity": RefusalSpec(
        "definition.dim.source_versioned_validity", DefinitionError, _render_definition_message
    ),
    "definition.dim.validity_mutually_exclusive_encoding": RefusalSpec(
        "definition.dim.validity_mutually_exclusive_encoding",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.dim.validity_ranges_need": RefusalSpec(
        "definition.dim.validity_ranges_need", DefinitionError, _render_definition_message
    ),
    "definition.dim.validity_requires_one_encoding": RefusalSpec(
        "definition.dim.validity_requires_one_encoding", DefinitionError, _render_definition_message
    ),
    "definition.dim.validity_valid_from": RefusalSpec(
        "definition.dim.validity_valid_from", DefinitionError, _render_definition_message
    ),
    "definition.duplicates": RefusalSpec(
        "definition.duplicates", DefinitionError, _render_definition_message
    ),
    "definition.encode_json_object": RefusalSpec(
        "definition.encode_json_object", DefinitionError, _render_definition_message
    ),
    "definition.encouragement_uptake.window_days_integer": RefusalSpec(
        "definition.encouragement_uptake.window_days_integer",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.expected_datetime_bool": RefusalSpec(
        "definition.expected_datetime_bool", DefinitionError, _render_definition_message
    ),
    "definition.expected_iso_8601_epoch": RefusalSpec(
        "definition.expected_iso_8601_epoch", DefinitionError, _render_definition_message
    ),
    "definition.expected_iso_8601_numeric_string": RefusalSpec(
        "definition.expected_iso_8601_numeric_string", DefinitionError, _render_definition_message
    ),
    "definition.experiment.binding_margin_margin": RefusalSpec(
        "definition.experiment.binding_margin_margin", DefinitionError, _render_definition_message
    ),
    "definition.experiment.cluster_combined_n": RefusalSpec(
        "definition.experiment.cluster_combined_n", DefinitionError, _render_definition_message
    ),
    "definition.experiment.cluster_names_same": RefusalSpec(
        "definition.experiment.cluster_names_same", DefinitionError, _render_definition_message
    ),
    "definition.experiment.design_tuning_key_not_yaml": RefusalSpec(
        "definition.experiment.design_tuning_key_not_yaml",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.experiment.end_before_start": RefusalSpec(
        "definition.experiment.end_before_start", DefinitionError, _render_definition_message
    ),
    "definition.experiment.intervention_grain_without_cluster": RefusalSpec(
        "definition.experiment.intervention_grain_without_cluster",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.experiment.metric_declares_cuped": RefusalSpec(
        "definition.experiment.metric_declares_cuped", DefinitionError, _render_definition_message
    ),
    "definition.experiment.observation_end_before_end": RefusalSpec(
        "definition.experiment.observation_end_before_end",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.experiment.observation_end_requires_end": RefusalSpec(
        "definition.experiment.observation_end_requires_end",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.experiment.timezone_aware_but": RefusalSpec(
        "definition.experiment.timezone_aware_but", DefinitionError, _render_definition_message
    ),
    "definition.exposure.filters_apply_fact": RefusalSpec(
        "definition.exposure.filters_apply_fact", DefinitionError, _render_definition_message
    ),
    "definition.exposure.needs_exactly_one": RefusalSpec(
        "definition.exposure.needs_exactly_one", DefinitionError, _render_definition_message
    ),
    "definition.exposure.sql_set_but": RefusalSpec(
        "definition.exposure.sql_set_but", DefinitionError, _render_definition_message
    ),
    "definition.fact.source_declares_propert": RefusalSpec(
        "definition.fact.source_declares_propert", DefinitionError, _render_definition_message
    ),
    "definition.fact.source_points_propert": RefusalSpec(
        "definition.fact.source_points_propert", DefinitionError, _render_definition_message
    ),
    "definition.fact.source_renames_its": RefusalSpec(
        "definition.fact.source_renames_its", DefinitionError, _render_definition_message
    ),
    "definition.fact.source_renames_propert": RefusalSpec(
        "definition.fact.source_renames_propert", DefinitionError, _render_definition_message
    ),
    "definition.filter.op_between_comparable": RefusalSpec(
        "definition.filter.op_between_comparable", DefinitionError, _render_definition_message
    ),
    "definition.filter.op_between_values": RefusalSpec(
        "definition.filter.op_between_values", DefinitionError, _render_definition_message
    ),
    "definition.filter.op_exactly_value": RefusalSpec(
        "definition.filter.op_exactly_value", DefinitionError, _render_definition_message
    ),
    "definition.filter.op_exactly_values": RefusalSpec(
        "definition.filter.op_exactly_values", DefinitionError, _render_definition_message
    ),
    "definition.filter.op_least_value": RefusalSpec(
        "definition.filter.op_least_value", DefinitionError, _render_definition_message
    ),
    "definition.filter.values_finite": RefusalSpec(
        "definition.filter.values_finite", DefinitionError, _render_definition_message
    ),
    "definition.inference.baseline_rate_domain": RefusalSpec(
        "definition.inference.baseline_rate_domain", DefinitionError, _render_definition_message
    ),
    "definition.inference.baseline_rate_route": RefusalSpec(
        "definition.inference.baseline_rate_route", DefinitionError, _render_definition_message
    ),
    "definition.invalid": RefusalSpec(
        "definition.invalid", DefinitionError, _render_definition_message
    ),
    "definition.json": RefusalSpec("definition.json", DefinitionError, _render_definition_message),
    "definition.lowercase_64_hex": RefusalSpec(
        "definition.lowercase_64_hex", DefinitionError, _render_definition_message
    ),
    "definition.method.methodspec_name_cuped": RefusalSpec(
        "definition.method.methodspec_name_cuped", DefinitionError, _render_definition_message
    ),
    "definition.metric_base.margin_margin_abs": RefusalSpec(
        "definition.metric_base.margin_margin_abs", DefinitionError, _render_definition_message
    ),
    "definition.metric_base.rollout_cost_preferred": RefusalSpec(
        "definition.metric_base.rollout_cost_preferred", DefinitionError, _render_definition_message
    ),
    "definition.metric_both_relative": RefusalSpec(
        "definition.metric_both_relative", DefinitionError, _render_definition_message
    ),
    "definition.metric_margin_abs_requires_direction": RefusalSpec(
        "definition.metric_margin_abs_requires_direction",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.metric_margin_abs_requires_non_neutral_direction": RefusalSpec(
        "definition.metric_margin_abs_requires_non_neutral_direction",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.metric_margin_implies_unrepresentable_null_lift": RefusalSpec(
        "definition.metric_margin_implies_unrepresentable_null_lift",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.metric_margin_non": RefusalSpec(
        "definition.metric_margin_non", DefinitionError, _render_definition_message
    ),
    "definition.metric_margin_requires_direction": RefusalSpec(
        "definition.metric_margin_requires_direction", DefinitionError, _render_definition_message
    ),
    "definition.metric_unknown_alternative": RefusalSpec(
        "definition.metric_unknown_alternative", DefinitionError, _render_definition_message
    ),
    "definition.models.reject_bool": RefusalSpec(
        "definition.models.reject_bool", DefinitionError, _render_definition_message
    ),
    "definition.multiplicity.bh_q_only_valid_for_bh": RefusalSpec(
        "definition.multiplicity.bh_q_only_valid_for_bh",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.multiplicity.bh_requires_q": RefusalSpec(
        "definition.multiplicity.bh_requires_q", DefinitionError, _render_definition_message
    ),
    "definition.non_empty_string": RefusalSpec(
        "definition.non_empty_string", DefinitionError, _render_definition_message
    ),
    "definition.observational.ambiguous_covariate_source": RefusalSpec(
        "definition.observational.ambiguous_covariate_source",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.quantile.metric_use_avg": RefusalSpec(
        "definition.quantile.metric_use_avg", DefinitionError, _render_definition_message
    ),
    "definition.ratio.metric_numerator_denominator": RefusalSpec(
        "definition.ratio.metric_numerator_denominator", DefinitionError, _render_definition_message
    ),
    "definition.ratio.metric_use_avg": RefusalSpec(
        "definition.ratio.metric_use_avg", DefinitionError, _render_definition_message
    ),
    "definition.ratio_metric.numerator_denominator_differ": RefusalSpec(
        "definition.ratio_metric.numerator_denominator_differ",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.relation_bind_manifest": RefusalSpec(
        "definition.relation_bind_manifest", DefinitionError, _render_definition_message
    ),
    "definition.relation_primary_key": RefusalSpec(
        "definition.relation_primary_key", DefinitionError, _render_definition_message
    ),
    "definition.relation_role": RefusalSpec(
        "definition.relation_role", DefinitionError, _render_definition_message
    ),
    "definition.resolve_breakout.experiment_property_could": RefusalSpec(
        "definition.resolve_breakout.experiment_property_could",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.resolve_breakout.experiment_property_found": RefusalSpec(
        "definition.resolve_breakout.experiment_property_found",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.resolve_breakout.experiment_property_references": RefusalSpec(
        "definition.resolve_breakout.experiment_property_references",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.resolve_breakout.experiment_unit_but": RefusalSpec(
        "definition.resolve_breakout.experiment_unit_but",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.retention.metric_window_days": RefusalSpec(
        "definition.retention.metric_window_days", DefinitionError, _render_definition_message
    ),
    "definition.retention.threshold_days_lower_bound_non_negative": RefusalSpec(
        "definition.retention.threshold_days_lower_bound_non_negative",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.retention.threshold_days_non_negative": RefusalSpec(
        "definition.retention.threshold_days_non_negative",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.retention.threshold_days_upper_exceeds_lower": RefusalSpec(
        "definition.retention.threshold_days_upper_exceeds_lower",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.single_sql_identifier": RefusalSpec(
        "definition.single_sql_identifier", DefinitionError, _render_definition_message
    ),
    "definition.site_volume.first_ds_last": RefusalSpec(
        "definition.site_volume.first_ds_last", DefinitionError, _render_definition_message
    ),
    "definition.site_volume.measure_keys_empty": RefusalSpec(
        "definition.site_volume.measure_keys_empty", DefinitionError, _render_definition_message
    ),
    "definition.site_volume.measure_keys_unique": RefusalSpec(
        "definition.site_volume.measure_keys_unique", DefinitionError, _render_definition_message
    ),
    "definition.site_volume.metric_names_empty": RefusalSpec(
        "definition.site_volume.metric_names_empty", DefinitionError, _render_definition_message
    ),
    "definition.site_volume.metric_names_unique": RefusalSpec(
        "definition.site_volume.metric_names_unique", DefinitionError, _render_definition_message
    ),
    "definition.source.sql_empty": RefusalSpec(
        "definition.source.sql_empty", DefinitionError, _render_definition_message
    ),
    "definition.surrounding_whitespace": RefusalSpec(
        "definition.surrounding_whitespace", DefinitionError, _render_definition_message
    ),
    "definition.total.metric_use_avg": RefusalSpec(
        "definition.total.metric_use_avg", DefinitionError, _render_definition_message
    ),
    "definition.total.metric_window_days": RefusalSpec(
        "definition.total.metric_window_days", DefinitionError, _render_definition_message
    ),
    "definition.unit_day.created_at_timezone": RefusalSpec(
        "definition.unit_day.created_at_timezone", DefinitionError, _render_definition_message
    ),
    "definition.unit_day.manifest_extension_absent": RefusalSpec(
        "definition.unit_day.manifest_extension_absent", DefinitionError, _render_definition_message
    ),
    "definition.unit_day.manifest_extensions_contain": RefusalSpec(
        "definition.unit_day.manifest_extensions_contain",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.unit_day.manifest_extensions_use": RefusalSpec(
        "definition.unit_day.manifest_extensions_use", DefinitionError, _render_definition_message
    ),
    "definition.unit_day.manifest_first_ds": RefusalSpec(
        "definition.unit_day.manifest_first_ds", DefinitionError, _render_definition_message
    ),
    "definition.unit_day.manifest_measure_keys": RefusalSpec(
        "definition.unit_day.manifest_measure_keys", DefinitionError, _render_definition_message
    ),
    "definition.unit_day.manifest_measures_use": RefusalSpec(
        "definition.unit_day.manifest_measures_use", DefinitionError, _render_definition_message
    ),
    "definition.unit_day.manifest_metric_bindings": RefusalSpec(
        "definition.unit_day.manifest_metric_bindings", DefinitionError, _render_definition_message
    ),
    "definition.unit_day.manifest_metric_bindings_canonical_order": RefusalSpec(
        "definition.unit_day.manifest_metric_bindings_canonical_order",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.unit_day.manifest_metric_names": RefusalSpec(
        "definition.unit_day.manifest_metric_names", DefinitionError, _render_definition_message
    ),
    "definition.unit_day.manifest_sha256_does": RefusalSpec(
        "definition.unit_day.manifest_sha256_does", DefinitionError, _render_definition_message
    ),
    "definition.unit_day.metric_references_undeclared": RefusalSpec(
        "definition.unit_day.metric_references_undeclared",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.unit_day.ratio_metric_references": RefusalSpec(
        "definition.unit_day.ratio_metric_references", DefinitionError, _render_definition_message
    ),
    "definition.unit_day.site_volume_keys": RefusalSpec(
        "definition.unit_day.site_volume_keys", DefinitionError, _render_definition_message
    ),
    "definition.unit_day.site_volume_keys_absent_from_manifest": RefusalSpec(
        "definition.unit_day.site_volume_keys_absent_from_manifest",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.unit_day.site_volume_request": RefusalSpec(
        "definition.unit_day.site_volume_request", DefinitionError, _render_definition_message
    ),
    "definition.utf": RefusalSpec("definition.utf", DefinitionError, _render_definition_message),
    "definition.winsorization.finite": RefusalSpec(
        "definition.winsorization.finite", DefinitionError, _render_definition_message
    ),
    "definition.winsorization.least_one_bound": RefusalSpec(
        "definition.winsorization.least_one_bound", DefinitionError, _render_definition_message
    ),
    "definition.winsorization.lower_percentile_below": RefusalSpec(
        "definition.winsorization.lower_percentile_below",
        DefinitionError,
        _render_definition_message,
    ),
    "definition.winsorization.lower_value_below": RefusalSpec(
        "definition.winsorization.lower_value_below", DefinitionError, _render_definition_message
    ),
    "definition.winsorization.set_percentile_fixed": RefusalSpec(
        "definition.winsorization.set_percentile_fixed", DefinitionError, _render_definition_message
    ),
}


def _definition_refusal(code: str, message: str, **context: object) -> NoReturn:
    """Raise a registered DefinitionError with its interpolated message."""
    spec = DEFINITION_REFUSALS[code]
    raise DefinitionError(spec.render(message=message), code=code, context=context)


def _require_nonblank_source_sql(kind: str, name: str, sql: str) -> None:
    """Refuse blank fact/dimension SQL; admission of real SQL stays in the loader."""
    if not sql.strip():
        _definition_refusal(
            "definition.source.sql_empty",
            f"{kind} source '{name}' has empty SQL -- declare a real query",
            source_kind=kind,
            source_name=name,
            route="supply a read-only SELECT or WITH query",
        )


# Base / mixin


class _Base(CodedValidationMixin, BaseModel):
    # Prevent silent field typos at every level, not just the top; freeze
    # every declaration so post-validation mutation (direct or through a
    # nested caller-owned instance) cannot silently invalidate it; deep-copy
    # a caller-supplied nested model instance instead of aliasing it.
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class _ExplicitOnlyFields:
    """Mixin for fields whose "declared vs defaulted" state is
    load-bearing downstream (checked via ``model_fields_set``).

    ``model_dump()`` writes out every field's resolved value regardless
    of whether it was explicitly supplied; reloading that dump then
    marks the field as declared even when it was only ever a default or
    an inherited value. Dropping the field from the dump whenever it
    was not explicitly set restores the same ``model_fields_set`` on
    reload that the original declaration had.
    """

    _explicit_only_fields: ClassVar[frozenset[str]] = frozenset()

    @model_serializer(mode="wrap")
    def _drop_unset_explicit_fields(
        self, handler: Callable[[Any], dict[str, Any]]
    ) -> dict[str, Any]:
        data = handler(self)
        # The mixin is only ever combined with BaseModel, which supplies
        # model_fields_set; the cast states that for the type checker.
        declared = cast("BaseModel", self).model_fields_set
        for field in self._explicit_only_fields - declared:
            data.pop(field, None)
        return data


def _reject_bool(v: Any) -> Any:
    """``bool`` is an ``int`` subclass in Python; a declared count/window
    field must never silently accept ``True``/``False`` as ``1``/``0``."""
    if isinstance(v, bool):
        _definition_refusal(
            "definition.models.reject_bool",
            "expected an int, got a bool",
        )
    return v


def _reject_numeric_datetime(v: Any) -> Any:
    """Reject a raw epoch number (or numeric-looking string) posing as a
    datetime -- pydantic's default coercion silently accepts both,
    turning ``start: 0`` into 1970-01-01 or ``start: '1700000000'`` into
    a wildly wrong date with no error."""
    if isinstance(v, bool):
        _definition_refusal(
            "definition.expected_datetime_bool",
            "expected a datetime, got a bool",
        )
    if isinstance(v, int | float):
        _definition_refusal(
            "definition.expected_iso_8601_epoch",
            "expected an ISO-8601 datetime, got a numeric epoch value",
        )
    if isinstance(v, str) and v.strip().lstrip("+-").isdigit():
        _definition_refusal(
            "definition.expected_iso_8601_numeric_string",
            "expected an ISO-8601 datetime, got a numeric-looking string",
        )
    return v


_SQL_ALIAS_PART = re.compile(r"[a-z0-9]+")

#: Reserved column names a dim property may not take (used internally by
#: `_rename_to_builder_cols` and the range-join projection).
_DIM_RESERVED_COLUMNS = frozenset(
    {"ts", "unit_id", "event", "experiment_id", "valid_from", "valid_to"}
)

#: Columns `_rename_to_builder_cols` creates on every fact source, so a
#: property with one of these names always collides. Versioned dim joins
#: refuse their own `valid_from`/`valid_to` collisions.
_ALWAYS_BUILT_COLUMNS = frozenset({"ts", "unit_id", "event", "experiment_id"})


def _sql_alias(name: str, prefix: str) -> str:
    alias = "_".join(_SQL_ALIAS_PART.findall(name.lower()))
    if alias[:1].isdigit():
        return f"{prefix.lower()}_{alias}"
    return alias


class AliasMixin:
    """SQL-safe alias generation."""

    # Provided by the concrete model this is mixed into (every consumer
    # declares a `name: str` field) - declared here so `.alias` type-checks.
    name: str

    @property
    def alias(self) -> str:
        return _sql_alias(self.name, type(self).__name__)


# Fact sources


class SourceColumn(AliasMixin, _Base):
    """A named, documented column exposed by a fact source."""

    name: str
    column: str
    description: str | None = None


class Fact(SourceColumn):
    # `column` is the fact's value column; None means an occurrence-only
    # event (page_view) - count/conversion/retention work, but sum/averages/etc. do not.
    column: str | None = None


class Property(SourceColumn):
    """A unit attribute exposed by a fact source.

    ``as_of`` declares when the value is measured; conditioning on a
    value measured after exposure can bias the result.

    - ``"event_time"`` (default): free to change; rejected as a breakout dimension.
    - ``"pre_exposure"``: latest value before first exposure; missing
      values land in the ``"__null__"`` bin.
    - ``"static"``: cannot change for the unit's lifetime.
    """

    dtype: Literal["string", "int", "float", "bool", "date"] = "string"
    as_of: Literal["event_time", "pre_exposure", "static"] = "event_time"


class DimValidity(_Base):
    """Time axis of a versioned dim source - two physical encodings.

    Either both range columns (``valid_from``/``valid_to``, one row per
    version) or a single ``changed_at`` column (a changelog, windowed
    into ranges by the query layer). Exactly one encoding must be given.
    """

    valid_from: str | None = None
    valid_to: str | None = None
    changed_at: str | None = None

    @model_validator(mode="after")
    def _exactly_one_encoding(self):
        has_range = self.valid_from is not None or self.valid_to is not None
        full_range = self.valid_from is not None and self.valid_to is not None
        has_changelog = self.changed_at is not None
        if has_range and has_changelog:
            _definition_refusal(
                "definition.dim.validity_mutually_exclusive_encoding",
                "validity takes either both 'valid_from'/'valid_to' or 'changed_at', not both",
            )
        if has_range and not full_range:
            _definition_refusal(
                "definition.dim.validity_ranges_need",
                "validity ranges need both 'valid_from' and 'valid_to'",
            )
        if not has_range and not has_changelog:
            _definition_refusal(
                "definition.dim.validity_requires_one_encoding",
                "validity takes either both 'valid_from'/'valid_to' or 'changed_at'",
            )
        if full_range and self.valid_from == self.valid_to:
            _definition_refusal(
                "definition.dim.validity_valid_from",
                f"validity 'valid_from' and 'valid_to' must be distinct columns, "
                f"both {self.valid_from!r}",
                valid_from=self.valid_from,
            )
        return self


class DimSource(AliasMixin, _Base):
    """A dimension table declared once and joined into fact sources on demand.

    Without ``validity`` it's a plain dim (one row per ``entity``, every
    property ``as_of: static``); with it, a versioned history with a
    temporal range predicate. Joins are LEFT joins - an unmatched fact
    row keeps NULL properties rather than being dropped.
    """

    name: str
    sql: str
    entity: str
    validity: DimValidity | None = None
    properties: tuple[Property, ...]

    @model_validator(mode="after")
    def _sql_is_nonblank(self):
        _require_nonblank_source_sql("dimension", self.name, self.sql)
        return self

    @model_validator(mode="after")
    def _properties_match_time_axis(self):
        if not self.properties:
            _definition_refusal(
                "definition.dim.source_declares_no",
                f"dim source '{self.name}' declares no properties: at least one property is required",
                name=self.name,
            )
        if self.validity is None:
            bad = [p.name for p in self.properties if p.as_of != "static"]
            if bad:
                _definition_refusal(
                    "definition.dim.source_no_validity",
                    f"dim source '{self.name}' has no validity (a plain dim), so every "
                    f"property must be as_of: static; got non-static: {bad}",
                    name=self.name,
                    bad=bad,
                )
        else:
            bad = [p.name for p in self.properties if p.as_of == "static"]
            if bad:
                _definition_refusal(
                    "definition.dim.source_versioned_validity",
                    f"dim source '{self.name}' is versioned (has validity), so no property "
                    f"may be as_of: static; got static: {bad}",
                    name=self.name,
                    bad=bad,
                )
        return self

    @model_validator(mode="after")
    def _no_duplicate_property_names(self):
        seen: set[str] = set()
        for p in self.properties:
            if p.name in seen:
                _definition_refusal(
                    "definition.dim.duplicate_property_name",
                    f"duplicate property name '{p.name}' in dim source '{self.name}'",
                    property_name=p.name,
                    source_name=self.name,
                )
            seen.add(p.name)
        return self


class FactSource(AliasMixin, _Base):
    name: str
    sql: str
    timestamp_column: str
    entities: tuple[str, ...] = Field(min_length=1)
    facts: tuple[Fact, ...]
    properties: tuple[Property, ...] = ()
    #: Names of DimSources to join in. Join key is the dim's ``entity``
    #: (must appear in ``entities``); versioned dims add a range predicate.
    dims: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _sql_is_nonblank(self) -> "FactSource":
        _require_nonblank_source_sql("fact", self.name, self.sql)
        return self

    @model_validator(mode="after")
    def _properties_dont_collide_with_builder_or_fact_columns(self) -> "FactSource":
        # `_rename_to_builder_cols` renames the entity and timestamp onto `unit_id`
        # and `ts`, so a reserved property NAME collides even under an identity
        # mapping. Either value could win; a property in `ts` would make window and
        # enrollment logic read it as a timestamp.
        bad_reserved = sorted({p.name for p in self.properties if p.name in _ALWAYS_BUILT_COLUMNS})
        if bad_reserved:
            _definition_refusal(
                "definition.fact.source_declares_propert",
                f"fact source '{self.name}' declares propert{'y' if len(bad_reserved) == 1 else 'ies'} "
                f"named {bad_reserved}, which the builders reserve -- rename the "
                f"propert{'y' if len(bad_reserved) == 1 else 'ies'}",
                name=self.name,
                bad_reserved=bad_reserved,
            )
        # A fact value column, by contrast, is only lost by an ACTUAL rename: an
        # identity mapping moves nothing, so it is the mapping that matters here.
        renames = [(p.column, p.name) for p in self.properties if p.column != p.name]
        fact_value_columns = {f.column for f in self.facts if f.column is not None}
        clobbered_dst = sorted({dst for _, dst in renames if dst in fact_value_columns})
        if clobbered_dst:
            _definition_refusal(
                "definition.fact.source_renames_propert",
                f"fact source '{self.name}' renames propert{'y' if len(clobbered_dst) == 1 else 'ies'} "
                f"onto its own fact value column(s) {clobbered_dst} -- the fact's value "
                f"would be overwritten",
                name=self.name,
                clobbered_dst=clobbered_dst,
            )
        renamed_away = sorted({src for src, _ in renames if src in fact_value_columns})
        if renamed_away:
            _definition_refusal(
                "definition.fact.source_renames_its",
                f"fact source '{self.name}' renames its own fact value column(s) "
                f"{renamed_away} away to a property name -- the fact's value would be lost",
                name=self.name,
                renamed_away=renamed_away,
            )
        # Builders always rename `timestamp_column` to `ts`, so no property may read
        # that column, even through an identity mapping. Entities are not checked:
        # only the query-selected unit becomes `unit_id`, so the query layer owns
        # that check.
        if any(p.column == self.timestamp_column for p in self.properties):
            stolen = sorted(
                {p.column for p in self.properties if p.column == self.timestamp_column}
            )
            _definition_refusal(
                "definition.fact.source_points_propert",
                f"fact source '{self.name}' points propert{'y' if len(stolen) == 1 else 'ies'} "
                f"at column(s) {stolen}, which the builders rename onto ts -- the "
                f"timestamp would be renamed away and the property would be "
                f"unavailable; expose a copy under a different name in the source SQL",
                name=self.name,
                stolen=stolen,
            )
        return self


# Filters


class Filter(_Base):
    property: str
    op: Literal[
        "equals",
        "not_equals",
        "in",
        "not_in",
        "gt",
        "gte",
        "lt",
        "lte",
        "between",
    ]
    values: tuple[str | int | float | bool, ...]

    @field_validator("values")
    @classmethod
    def _reject_nonfinite_values(
        cls, v: tuple[str | int | float | bool, ...]
    ) -> tuple[str | int | float | bool, ...]:
        bad = [item for item in v if isinstance(item, float) and not math.isfinite(item)]
        if bad:
            _definition_refusal(
                "definition.filter.values_finite",
                f"filter values must be finite, got {bad!r}",
                bad=bad,
            )
        return v

    @model_validator(mode="after")
    def _check_arity(self):
        n = len(self.values)
        if self.op == "between" and n != 2:
            _definition_refusal(
                "definition.filter.op_exactly_values",
                f"filter op '{self.op}' requires exactly 2 values, got {n}",
                op=self.op,
                n=n,
            )
        if self.op in ("equals", "not_equals", "gt", "gte", "lt", "lte") and n != 1:
            _definition_refusal(
                "definition.filter.op_exactly_value",
                f"filter op '{self.op}' requires exactly 1 value, got {n}",
                op=self.op,
                n=n,
            )
        if self.op in ("in", "not_in") and n < 1:
            _definition_refusal(
                "definition.filter.op_least_value",
                f"filter op '{self.op}' requires at least 1 value, got {n}",
                op=self.op,
                n=n,
            )
        return self

    @model_validator(mode="after")
    def _between_bounds_are_ordered(self):
        if self.op == "between" and len(self.values) == 2:
            lo, hi = self.values
            try:
                # Execution uses inclusive bounds, so [x, x] is a valid exact-value
                # range; only a reversed range is meaningless.
                reversed_range = bool(cast("Any", hi) < cast("Any", lo))
            except TypeError:
                _definition_refusal(
                    "definition.filter.op_between_comparable",
                    f"filter op 'between' requires comparable bounds, got {self.values!r}",
                    values=self.values,
                )
            if reversed_range:
                _definition_refusal(
                    "definition.filter.op_between_values",
                    f"filter op 'between' requires values[0] <= values[1], got {self.values!r}",
                    values=self.values,
                )
        return self


# Exposures


class Exposure(AliasMixin, _Base):
    """What marks a unit as exposed."""

    name: str
    description: str | None = None
    sql: str | None = None  # explicit query, XOR:
    fact: str | None = None  # fact-based exposure
    filters: tuple[Filter, ...] = ()

    @model_validator(mode="after")
    def _exactly_one_source(self):
        # `is None`, not truthiness: sql="" is a set-but-empty query (a
        # template bug) and must refuse, not silently fall back to fact.
        if (self.sql is None) == (self.fact is None):
            _definition_refusal(
                "definition.exposure.needs_exactly_one",
                "Exposure needs exactly one of `sql` or `fact` (an empty sql string counts as set)",
            )
        if self.sql is not None and not self.sql.strip():
            _definition_refusal(
                "definition.exposure.sql_set_but",
                "Exposure.sql is set but empty -- declare a real query, or "
                "use `fact` for a fact-based exposure",
            )
        if self.sql is not None and self.filters:
            _definition_refusal(
                "definition.exposure.filters_apply_fact",
                "`filters` apply only to a fact-based exposure",
            )
        return self


# Metrics


class FactRef(_Base):
    """Occurrence core shared by every metric."""

    fact: str
    filters: tuple[Filter, ...] = ()
    window_days: int | None = Field(default=None, ge=1)  # None = variable per-unit window

    @field_validator("window_days", mode="before")
    @classmethod
    def _validate_window_days(cls, v: Any) -> Any:
        return None if v is None else _reject_bool(v)


class Measure(FactRef):
    """Reduce a fact's events to one number per unit."""

    aggregation: Literal[
        "sum", "count", "count_distinct", "avg_event", "avg_calendar_day", "min", "max"
    ] = "count"


def _measure_identity(measure: "Measure") -> tuple[object, ...]:
    """Semantic identity of a measure, independent of filter declaration order.

    Conjunctive filters commute, so two measures listing the same filters in
    different orders select the same rows and must compare equal.
    """
    filters = tuple(sorted(f.model_dump_json() for f in measure.filters))
    return (measure.fact, measure.aggregation, measure.window_days, filters)


def resolve_margin(
    preferred_direction: PreferredDirection | None,
    # Direction explicitness is part of the public margin contract.
    explicitly_set: bool,  # noqa: FBT001
    margin: float,
    *,
    metric_name: str,
) -> tuple[float, Literal["greater", "less"]]:
    """Derive ``(null_lift, favorable_tail)`` from a margin and the
    metric's preferred direction.

    ``null_lift = adverse_sign * margin``; ``explicitly_set`` must come
    from ``"preferred_direction" in model_fields_set``, never inferred,
    since a default must never silently pick the adverse side. For
    "increase" the margin must be < 1, keeping the implied null above -1.
    """
    if not explicitly_set:
        _definition_refusal(
            "definition.metric_margin_requires_direction",
            f"metric '{metric_name}': margin={margin} requires preferred_direction "
            f"to be explicitly set (increase|decrease) -- a defaulted direction must "
            f"never silently pick a guardrail's adverse side",
            metric_name=metric_name,
            margin=margin,
        )
    if preferred_direction is None or preferred_direction == "neutral":
        _definition_refusal(
            "definition.metric_margin_non",
            f"metric '{metric_name}': margin={margin} requires a non-neutral "
            f"preferred_direction (increase|decrease) -- 'neutral' has no adverse side",
            metric_name=metric_name,
            margin=margin,
        )
    if preferred_direction == "increase":
        if margin >= 1.0:
            _definition_refusal(
                "definition.metric_margin_implies_unrepresentable_null_lift",
                f"metric '{metric_name}': margin={margin} with "
                f"preferred_direction='increase' implies null_lift={-margin} "
                "<= -1 -- a >=100% relative loss has no log-scale "
                "representation",
                metric_name=metric_name,
                margin=margin,
            )
        return -margin, "greater"
    return margin, "less"


def resolve_margin_abs(
    preferred_direction: PreferredDirection | None,
    # Direction explicitness is part of the public margin contract.
    explicitly_set: bool,  # noqa: FBT001
    margin_abs: float,
    *,
    metric_name: str,
) -> tuple[float, Literal["greater", "less"]]:
    """Derive ``(null_abs, favorable_tail)`` from an absolute-unit margin,
    the additive-scale counterpart to :func:`resolve_margin`.

    Same sign convention and explicit-direction guard, but no ``> -1``
    floor: the additive scale has no total-loss singularity.
    """
    if not explicitly_set:
        _definition_refusal(
            "definition.metric_margin_abs_requires_direction",
            f"metric '{metric_name}': margin_abs={margin_abs} requires preferred_direction "
            f"to be explicitly set (increase|decrease) -- a defaulted direction must "
            f"never silently pick a guardrail's adverse side",
            metric_name=metric_name,
            margin_abs=margin_abs,
        )
    if preferred_direction is None or preferred_direction == "neutral":
        _definition_refusal(
            "definition.metric_margin_abs_requires_non_neutral_direction",
            f"metric '{metric_name}': margin_abs={margin_abs} requires a non-neutral "
            f"preferred_direction (increase|decrease) -- 'neutral' has no adverse side",
            metric_name=metric_name,
            margin_abs=margin_abs,
        )
    if preferred_direction == "increase":
        return -margin_abs, "greater"
    return margin_abs, "less"


class MetricIdentity(AliasMixin, _ExplicitOnlyFields, _Base):
    """Name/description/direction - identity without an entity.

    Split from ``MetricBase`` so a report-only metric with no entity
    (``TotalMetric``) can share identity fields without inheriting ``entity``.
    """

    _explicit_only_fields: ClassVar[frozenset[str]] = frozenset({"preferred_direction"})

    name: str
    description: str = ""
    preferred_direction: PreferredDirection = "increase"

    @property
    def declared_preferred_direction(self) -> PreferredDirection | None:
        """``preferred_direction`` if explicitly set, else ``None``.

        A defaulted ``preferred_direction`` must never be mistaken for a
        declared one downstream (e.g. in favorability probabilities) --
        prefer this over reading the bare ``preferred_direction`` field
        whenever the DECLARED VALUE is what's needed; call sites that only
        need the boolean (e.g. ``resolve_margin``'s ``explicitly_set``)
        keep their own ``"preferred_direction" in model_fields_set`` check.
        """
        return self.preferred_direction if "preferred_direction" in self.model_fields_set else None


class MetricBase(MetricIdentity):
    """The named identity every entity-scoped metric carries.

    ``preferred_direction`` names the winning direction; it never gates
    sidedness alone, only the adverse side of a declared ``margin`` or
    ``margin_abs`` (mutually exclusive non-inferiority guardrails,
    requiring ``preferred_direction`` explicitly non-neutral).
    ``rollout_cost`` is the pre-declared per-segment break-even a
    rollout recommendation tests against.
    """

    entity: str
    margin: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    margin_abs: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    rollout_cost: float | None = Field(default=None, gt=-1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _margins_mutually_exclusive(self) -> "MetricBase":
        if self.margin is not None and self.margin_abs is not None:
            _definition_refusal(
                "definition.metric_base.margin_margin_abs",
                f"metric {self.name!r}: margin and margin_abs are mutually "
                "exclusive -- one guardrail, one scale",
                name=self.name,
            )
        return self

    @model_validator(mode="after")
    def _margin_requires_explicit_direction(self) -> "MetricBase":
        if self.margin is None:
            return self
        resolve_margin(
            self.preferred_direction,
            "preferred_direction" in self.model_fields_set,
            self.margin,
            metric_name=self.name,
        )
        return self

    @model_validator(mode="after")
    def _margin_abs_requires_explicit_direction(self) -> "MetricBase":
        if self.margin_abs is None:
            return self
        resolve_margin_abs(
            self.preferred_direction,
            "preferred_direction" in self.model_fields_set,
            self.margin_abs,
            metric_name=self.name,
        )
        return self

    @model_validator(mode="after")
    def _rollout_cost_requires_explicit_increase(self) -> "MetricBase":
        if self.rollout_cost is None:
            return self
        if (
            "preferred_direction" not in self.model_fields_set
            or self.preferred_direction != "increase"
        ):
            _definition_refusal(
                "definition.metric_base.rollout_cost_preferred",
                f"metric {self.name!r}: rollout_cost requires preferred_direction="
                "'increase' to be explicitly declared -- the rollout selection rule "
                "keeps segments whose lift EXCEEDS the cost, which presumes larger "
                "is better; a decrease-preferred rollout is not supported",
                name=self.name,
            )
        return self

    def resolved_null(self) -> tuple[float, Literal["greater", "less"] | None]:
        """Derive ``(null_lift, favorable_tail)`` from a declared ``margin``.

        Returns ``(0.0, None)`` when no margin is declared (two-sided vs zero).
        """
        if self.margin is None:
            return 0.0, None
        return resolve_margin(
            self.preferred_direction,
            "preferred_direction" in self.model_fields_set,
            self.margin,
            metric_name=self.name,
        )

    def resolved_null_abs(self) -> tuple[float | None, Literal["greater", "less"] | None]:
        """Derive ``(null_abs, favorable_tail)`` from a declared ``margin_abs``.

        Returns ``(None, None)`` when no absolute margin is declared.
        """
        if self.margin_abs is None:
            return None, None
        return resolve_margin_abs(
            self.preferred_direction,
            "preferred_direction" in self.model_fields_set,
            self.margin_abs,
            metric_name=self.name,
        )


class Winsorization(_Base):
    """Outcome bounds applied to a mean metric before aggregation."""

    lower_percentile: float | None = Field(default=None, gt=0.0, lt=1.0)
    upper_percentile: float | None = Field(default=None, gt=0.0, lt=1.0)
    lower_value: float | None = None
    upper_value: float | None = None
    support: WinsorSupport | None = Field(default=None, exclude_if=lambda v: v is None)
    inference: WinsorInferenceSpec = Field(default_factory=WinsorInferenceSpec)

    @property
    def has_percentile(self) -> bool:
        return self.lower_percentile is not None or self.upper_percentile is not None

    @model_validator(mode="after")
    def _valid_bounds(self) -> "Winsorization":
        if all(
            value is None
            for value in (
                self.lower_percentile,
                self.upper_percentile,
                self.lower_value,
                self.upper_value,
            )
        ):
            _definition_refusal(
                "definition.winsorization.least_one_bound",
                "winsorization requires at least one bound",
            )

        for side in ("lower", "upper"):
            percentile = getattr(self, f"{side}_percentile")
            value = getattr(self, f"{side}_value")
            if percentile is not None and value is not None:
                _definition_refusal(
                    "definition.winsorization.set_percentile_fixed",
                    f"winsorization cannot set percentile and fixed value on the same side ({side})",
                    side=side,
                )

        for name in ("lower_value", "upper_value"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                _definition_refusal(
                    "definition.winsorization.finite",
                    f"winsorization {name} must be finite",
                    name=name,
                )

        if (
            self.lower_percentile is not None
            and self.upper_percentile is not None
            and self.lower_percentile >= self.upper_percentile
        ):
            _definition_refusal(
                "definition.winsorization.lower_percentile_below",
                "winsorization lower_percentile must be below upper_percentile",
            )
        if (
            self.lower_value is not None
            and self.upper_value is not None
            and self.lower_value >= self.upper_value
        ):
            _definition_refusal(
                "definition.winsorization.lower_value_below",
                "winsorization lower_value must be below upper_value",
            )
        return self


def _reject_bool_days(v: Any) -> Any:
    """``_reject_bool`` over a day count or a ``(a, b)`` band: a ``bool`` is
    never a day count. Shared by ``RetentionMetric`` and ``MetricSpec`` so both
    ingresses refuse with the same code."""
    if isinstance(v, list | tuple):
        return tuple(_reject_bool(item) for item in v)
    return _reject_bool(v)


def _validate_retention_declaration(
    name: str, threshold_days: int | tuple[int, int], window_days: int | None
) -> None:
    """Refuse an unusable retention declaration: ``window_days`` set (both band
    edges live in ``threshold_days``), a negative edge, or an empty band. The
    one source of these refusals for ``RetentionMetric`` and ``MetricSpec``."""
    if window_days is not None:
        _definition_refusal(
            "definition.retention.metric_window_days",
            f"retention metric '{name}': window_days is not a retention "
            f"field -- the observation band, both edges, lives in threshold_days. "
            f"Declare threshold_days: [a, b] instead of threshold_days: a + "
            f"window_days: b.",
            name=name,
        )
    band = threshold_days
    if isinstance(band, int):
        if band < 0:
            _definition_refusal(
                "definition.retention.threshold_days_non_negative",
                f"retention metric '{name}': threshold_days={band} must be "
                f">= 0 -- days are counted from exposure (day 0)",
                name=name,
                band=band,
            )
        return
    a, b = band
    if a < 0:
        _definition_refusal(
            "definition.retention.threshold_days_lower_bound_non_negative",
            f"retention metric '{name}': threshold_days=[{a}, {b}] must have "
            f"a >= 0 -- days are counted from exposure (day 0)",
            name=name,
            a=a,
            b=b,
        )
    if b <= a:
        _definition_refusal(
            "definition.retention.threshold_days_upper_exceeds_lower",
            f"retention metric '{name}': threshold_days=[{a}, {b}] must have "
            f"b > a -- the half-open observation band [a, b) would otherwise be "
            f"empty and every unit would score 0",
            name=name,
            a=a,
            b=b,
        )


class MeanMetric(MetricBase, Measure):
    type: Literal["mean"] = "mean"
    winsorization: Winsorization | None = None


class ConversionMetric(MetricBase, FactRef):
    type: Literal["conversion"] = "conversion"


class RetentionMetric(MetricBase, FactRef):
    """Binary survival outcome: did the unit return inside the observation
    band declared by ``threshold_days``?

    An ``int`` N is the unbounded band ``[N, inf)`` (never matures); an
    ``[a, b]`` pair is the half-open band ``[a, b)``, final at
    ``first_exposure_date + b``, counted from day 0 in the experiment's
    ``day_boundary``. ``window_days`` is not a retention field: both
    edges already live in ``threshold_days``.
    """

    type: Literal["retention"] = "retention"
    threshold_days: int | tuple[int, int]
    # Not bounded (``FactRef`` requires ``ge=1``): any declared value, zero or
    # negative included, reaches ``_validate_retention_declaration`` and its
    # coded refusal, the same one ``MetricSpec`` raises.
    window_days: int | None = None

    @field_validator("threshold_days", mode="before")
    @classmethod
    def _validate_threshold_days(cls, v: Any) -> Any:
        return _reject_bool_days(v)

    @property
    def band(self) -> tuple[int, int | None]:
        """``threshold_days`` normalized to ``(band_start, band_end)`` - the
        half-open window in which an event counts as a return. ``band_end``
        is ``None`` when the band is open on the right.
        """
        band = self.threshold_days
        if isinstance(band, int):
            return band, None
        return band

    @model_validator(mode="after")
    def _band_is_non_empty(self):
        _validate_retention_declaration(self.name, self.threshold_days, self.window_days)
        return self


class RatioMetric(MetricBase):
    type: Literal["ratio"] = "ratio"
    numerator: Measure
    denominator: Measure

    @model_validator(mode="after")
    def _average_basis_is_event_only(self) -> "RatioMetric":
        for part_name, part in (("numerator", self.numerator), ("denominator", self.denominator)):
            if part.aggregation == "avg_calendar_day":
                _definition_refusal(
                    "definition.ratio.metric_use_avg",
                    f"ratio metric '{self.name}' {part_name} cannot use "
                    "'avg_calendar_day'; use 'avg_event' or a non-average aggregation",
                    name=self.name,
                    part_name=part_name,
                )
        return self

    @model_validator(mode="after")
    def _numerator_and_denominator_differ(self) -> "RatioMetric":
        if _measure_identity(self.numerator) == _measure_identity(self.denominator):
            _definition_refusal(
                "definition.ratio.metric_numerator_denominator",
                f"ratio metric '{self.name}': numerator and denominator must "
                "differ -- an identical measure always resolves to a ratio of 1",
                name=self.name,
            )
        return self


class TotalMetric(MetricIdentity, Measure):
    """Per-period aggregate of a fact column, with no entity denominator.

    Report-only: has no per-unit variance, so estimation rejects it in
    any experiment's metrics/guardrails.
    """

    type: Literal["total"] = "total"

    @model_validator(mode="after")
    def _no_window_days(self):
        if self.window_days is not None:
            _definition_refusal(
                "definition.total.metric_window_days",
                f"total metric '{self.name}': window_days is meaningless for "
                f"report-only metrics -- the calendar period is the window",
                name=self.name,
            )
        return self

    @model_validator(mode="after")
    def _average_basis_is_event_only(self) -> "TotalMetric":
        if self.aggregation == "avg_calendar_day":
            _definition_refusal(
                "definition.total.metric_use_avg",
                f"total metric '{self.name}' cannot use 'avg_calendar_day'; "
                "use 'avg_event' or a non-average aggregation",
                name=self.name,
            )
        return self


class ActiveMetric(MetricBase, FactRef):
    """Distinct entities with >=1 qualifying event in the period
    (DAU/WAU/MAU, distinct purchasers). Report-only, like ``TotalMetric``.
    ``conversion`` is this metric's rate twin: same event predicate,
    per-population share instead of raw count.
    """

    type: Literal["active"] = "active"

    @model_validator(mode="after")
    def _no_window_days(self):
        if self.window_days is not None:
            _definition_refusal(
                "definition.active.metric_window_days",
                f"active metric '{self.name}': window_days is meaningless for "
                f"report-only metrics -- the calendar period is the window",
                name=self.name,
            )
        return self


class QuantileMetric(MetricBase, Measure):
    """Distributional quantile of the per-unit totals - p50/p90/p99.

    Not additive across units, so only sources that can serve per-unit
    rows can estimate it; only ``run()`` serves it, not the moments-only
    surfaces.
    """

    type: Literal["quantile"] = "quantile"
    quantile: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _average_basis_is_event_only(self) -> "QuantileMetric":
        if self.aggregation == "avg_calendar_day":
            _definition_refusal(
                "definition.quantile.metric_use_avg",
                f"quantile metric '{self.name}' cannot use 'avg_calendar_day'; "
                "use 'avg_event' or a non-average aggregation",
                name=self.name,
            )
        return self


Metric = Annotated[
    MeanMetric
    | ConversionMetric
    | RetentionMetric
    | RatioMetric
    | QuantileMetric
    | TotalMetric
    | ActiveMetric,
    Field(discriminator="type"),
]


def resolve_null_and_alternative(
    metric: MetricBase,
    margins: Mapping[str, float] | None,
    null_lifts: Mapping[str, float] | None,
    alternative: str,
    margins_abs: Mapping[str, float] | None = None,
) -> tuple[float, float | None, str]:
    """Resolve one metric's effective ``(null_lift, null_abs, alternative)``.

    Relative axis: ``null_lifts=`` > ``margins=`` > the declared
    ``margin`` > ``0.0``. Absolute axis: ``margins_abs=`` > the declared
    ``margin_abs``; the two axes are mutually exclusive.
    """
    if alternative not in ALTERNATIVE_VALUES:
        _definition_refusal(
            "definition.metric_unknown_alternative",
            f"metric '{metric.name}': unknown alternative={alternative!r} -- must be "
            f'"two-sided", "greater", or "less". A typo here would silently discard '
            f"any margin-implied tail and forward the bogus string downstream.",
            name=metric.name,
            alternative=alternative,
        )
    override_null = (null_lifts or {}).get(metric.name)
    override_margin = (margins or {}).get(metric.name)
    override_margin_abs = (margins_abs or {}).get(metric.name)

    relative_active = (
        override_null is not None or override_margin is not None or metric.margin is not None
    )
    absolute_active = override_margin_abs is not None or metric.margin_abs is not None
    if relative_active and absolute_active:
        _definition_refusal(
            "definition.metric_both_relative",
            f"metric '{metric.name}': both a relative and an absolute null source "
            "resolve for this metric (null_lifts/margins/declared margin vs "
            "margins_abs/declared margin_abs) -- one guardrail, one scale",
            name=metric.name,
        )

    if absolute_active:
        if override_margin_abs is not None:
            null_abs, implied_tail_abs = resolve_margin_abs(
                metric.preferred_direction,
                "preferred_direction" in metric.model_fields_set,
                override_margin_abs,
                metric_name=metric.name,
            )
        else:
            null_abs, implied_tail_abs = metric.resolved_null_abs()
        effective_alt = (
            alternative if alternative != "two-sided" else (implied_tail_abs or "two-sided")
        )
        return 0.0, null_abs, effective_alt

    if override_null is not None:
        return override_null, None, alternative
    if override_margin is not None:
        null_lift, implied_tail = resolve_margin(
            metric.preferred_direction,
            "preferred_direction" in metric.model_fields_set,
            override_margin,
            metric_name=metric.name,
        )
    else:
        null_lift, implied_tail = metric.resolved_null()
    effective_alt = alternative if alternative != "two-sided" else (implied_tail or "two-sided")
    return null_lift, None, effective_alt


# Breakouts


class Breakout(_Base):
    """A group-by dimension applied to an experiment's metrics; ``source``
    is optional (inferred if omitted)."""

    property: str
    source: str | None = None
    skip_missing: bool = False


class Factor(_Base):
    """A discrete factor absorbed as a covariate to sharpen the ATE; unlike
    ``Breakout``, no per-level numbers are emitted."""

    property: str
    source: str | None = None


# Declared identification designs


_ENCOURAGEMENT_TUNING_KEYS = frozenset({"min_first_stage_z"})
_OBSERVATIONAL_TUNING_KEYS = frozenset({"gate", "missing"})


def _refuse_design_tuning_keys(mechanism: str, data: object, known: frozenset[str]) -> None:
    if not isinstance(data, Mapping):
        return
    present = sorted(known & set(data))
    if present:
        _definition_refusal(
            "definition.experiment.design_tuning_key_not_yaml",
            f"design: {{mechanism: {mechanism}, ...}} does not accept "
            f"{present} from YAML -- statistical tuning keeps its "
            f"library default here; declare a full "
            f"Encouragement/Observational object from "
            f"Analysis.from_unit_summary/from_unit_panel to override it",
            mechanism=mechanism,
            keys=present,
        )


class EncouragementDeclaration(_Base):
    """The mechanism-specific facts a randomized-encouragement experiment
    declares in YAML: which fact records uptake, whether encouragement is
    one-sided, and the optional exclusion acknowledgment required for LATE. The control
    arm and target allocation come from the experiment's own
    ``control_group``/``allocation`` -- see ``Experiment.resolved_design()``.
    Statistical tuning (``min_first_stage_z``) keeps its library default;
    declare it from ``Analysis.from_unit_summary``/``from_unit_panel`` with a
    full ``Encouragement`` object if you need to override it."""

    @model_validator(mode="before")
    @classmethod
    def _reject_tuning_keys(cls, data: object) -> object:
        _refuse_design_tuning_keys("encouragement", data, _ENCOURAGEMENT_TUNING_KEYS)
        return data

    mechanism: Literal["encouragement"] = "encouragement"
    uptake: UptakeSpec
    one_sided: bool = False
    exclusion_restriction: ExclusionRestriction | None = None


class AdjustmentCovariate(_Base):
    """One covariate binding for an observational design's adjustment
    set, named the same way a ``Breakout``/``Factor`` names its dimension:
    the declared ``Property`` and, when more than one source declares a
    property of that name, which source."""

    property: str
    source: str | None = None


class ObservationalDeclaration(_Base):
    """The mechanism-specific facts a non-randomized experiment declares
    in YAML: the covariates sufficient to identify the effect. The
    control arm comes from ``Experiment.control_group``. Overlap/positivity
    policy (``gate``) and missing-covariate handling (``adjustment.missing``)
    keep their library defaults; override them from
    ``Analysis.from_unit_summary`` with a full ``Observational`` object."""

    @model_validator(mode="before")
    @classmethod
    def _reject_tuning_keys(cls, data: object) -> object:
        _refuse_design_tuning_keys("observational", data, _OBSERVATIONAL_TUNING_KEYS)
        return data

    mechanism: Literal["observational"] = "observational"
    covariates: tuple[AdjustmentCovariate, ...] = Field(min_length=1)


ExperimentDesignDeclaration = Annotated[
    EncouragementDeclaration | ObservationalDeclaration, Field(discriminator="mechanism")
]


#: ``"UTC"`` or ``"UTC±HH:MM"`` at quarter-hour minutes, magnitude <=14h
#: (a few nonexistent offsets in between are accepted harmlessly).
_DAY_BOUNDARY_RE = re.compile(r"^UTC(?:([+-])(\d{2}):(00|15|30|45))?$")


def _validate_day_boundary(v: str) -> str:
    """Shared grammar check for the ``day_boundary`` field.

    Fixed offsets only - named (IANA) timezones are DST-aware and not
    supported yet.
    """
    m = _DAY_BOUNDARY_RE.match(v)
    valid = m is not None
    if valid and m.group(2) is not None:
        total_minutes = int(m.group(2)) * 60 + int(m.group(3))
        valid = total_minutes <= 14 * 60
    if not valid:
        _definition_refusal(
            "definition.day_boundary_accepts",
            f"day_boundary accepts 'UTC' or a fixed offset like 'UTC-05:00', "
            f"got {v!r}; named timezones (DST-aware) are not supported yet",
            v=v,
        )
    return v


# Experiment


class MethodSpec(_Base):
    """Serializable estimation-method declaration - the YAML-safe subset
    of ``increment.estimation.engine.Method`` (learners/folds are
    call-time-only, never declared here).

    ``conversion_inference`` mirrors ``Method.conversion_inference``:
    ``"auto"`` (the default) routes an unadjusted conversion or retention
    contrast by its counts, and ``"finite_sample"`` always uses the
    finite-sample binomial route. It is refused with CUPED and, on a
    metric that is not a conversion or retention rate, at definition load."""

    name: str
    variance_reduction: Literal["none", "cuped"] = "none"
    conversion_inference: ConversionInference = "auto"

    @model_validator(mode="after")
    def _no_mislabel(self) -> "MethodSpec":
        if self.name == "cuped" and self.variance_reduction != "cuped":
            _definition_refusal(
                "definition.method.methodspec_name_cuped",
                "MethodSpec(name='cuped') without variance_reduction='cuped' "
                "would label an unadjusted estimate as CUPED-adjusted",
            )
        if self.conversion_inference == "finite_sample" and self.variance_reduction == "cuped":
            refuse_finite_sample_cuped(self.name)
        return self


class NormalPriorSpec(_Base):
    """Serializable Normal prior declaration - the YAML-safe counterpart
    of ``increment.estimation.inference.Normal``."""

    mu: float = Field(allow_inf_nan=False)
    sigma: float = Field(gt=0, allow_inf_nan=False)


class ExperimentMetric(_Base):
    """One metric's method-role and prior declaration within an experiment.

    A bare metric NAME in a plan role (``primary``/``secondaries``/
    ``guardrails``) is shorthand for a binding with no overrides.  The
    decision method is optional because omission means the default
    unadjusted estimator; sensitivity methods are additional estimates
    reported after that decision estimate.
    """

    metric: str
    decision_method: MethodSpec | None = None
    sensitivity_methods: tuple[MethodSpec, ...] = ()
    prior: NormalPriorSpec | None = None
    margin: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    margin_abs: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _margins_mutually_exclusive(self) -> "ExperimentMetric":
        if self.margin is not None and self.margin_abs is not None:
            _definition_refusal(
                "definition.experiment.binding_margin_margin",
                f"experiment binding for {self.metric!r}: margin and margin_abs are "
                "mutually exclusive -- one guardrail, one scale",
                metric=self.metric,
            )
        return self

    @property
    def wants_cuped(self) -> bool:
        methods = (
            () if self.decision_method is None else (self.decision_method,)
        ) + self.sensitivity_methods
        return any(method.variance_reduction == "cuped" for method in methods)

    @property
    def wants_finite_sample(self) -> bool:
        methods = (
            () if self.decision_method is None else (self.decision_method,)
        ) + self.sensitivity_methods
        return any(method.conversion_inference == "finite_sample" for method in methods)


def _plan_entry_name(entry: "PlanEntry") -> str:
    """A ``PlanEntry`` is a bare metric name (``str``) or a binding
    (``ExperimentMetric``, keyed by its ``.metric``)."""
    return entry if isinstance(entry, str) else entry.metric


class InferenceSpec(_ExplicitOnlyFields, _Base):
    """Serializable exact likelihood or explicitly asymptotic scalar-mean policy.

    ``adjustments`` maps a mean metric to the pre-period CUPED coefficient and
    covariate centre its automatic scalar-mean registration binds; an explicit
    registration declares the same on ``ScalarMeanModel.adjustment`` instead.
    ``baseline_rate`` is the conversion rate you expect in the control arm,
    read off the same metric over the weeks before the experiment (the number
    the power analysis used); an automatic ``always_valid`` registration
    centres its Beta prior on it, and without it the prior is flat. A wrong
    value costs power, never validity. Both automatic kinds bind their
    registration from source metadata before any outcome is read.
    With no outcome metrics, ``baseline_rate`` instead describes control uptake
    and tunes its prior; composed outcome/compliance plans retain a flat uptake prior.
    ``segments`` predeclares the one segment dimension (a frame column) and the
    levels an automatic registration retains one hypothesis for, per metric and
    arm, before any outcome is read; a level never observed stays a monitored
    empty cell, and a value outside the declared levels joins no cell. Declaring
    it asserts that every unit's level was fixed before assignment: an
    ``asymptotic_mean`` registration records exactly that assertion as its
    ``segment_membership="pre_assignment"``, and nothing about membership is
    checked or inferred from outcomes. An explicit registration names each
    segment on its ``SequentialCell`` instead.
    """

    _explicit_only_fields: ClassVar[frozenset[str]] = frozenset({"expected_decision_sample_size"})

    kind: Literal["always_valid", "asymptotic_mean"]
    registration: SequentialRegistration | None = None
    expected_decision_sample_size: StrictInt | None = Field(default=None, ge=2)
    # A YAML plan carries the rate as a float; a trusted float binds the decimal as typed.
    baseline_rate: DeclaredRational | None = None
    adjustments: Mapping[str, PredeclaredAdjustment] = Field(
        default_factory=dict, validate_default=True, exclude_if=lambda value: not value
    )
    segments: Mapping[str, tuple[str, ...]] = Field(
        default_factory=dict, validate_default=True, exclude_if=lambda value: not value
    )

    @field_validator("adjustments")
    @classmethod
    def _snapshot_adjustments(
        cls, value: Mapping[str, PredeclaredAdjustment]
    ) -> Mapping[str, PredeclaredAdjustment]:
        # Caller-owned mapping; snapshot it so a later mutation cannot reach a frozen spec.
        return _FrozenMapping(value)

    @field_serializer("adjustments")
    def _dump_adjustments(
        self, value: Mapping[str, PredeclaredAdjustment]
    ) -> dict[str, PredeclaredAdjustment]:
        return dict(value)

    @field_validator("segments")
    @classmethod
    def _snapshot_segments(
        cls, value: Mapping[str, tuple[str, ...]]
    ) -> Mapping[str, tuple[str, ...]]:
        from increment.semantics.sequential import validate_predeclared_segments

        # Admit the family once (one dimension, distinct string labels) and
        # snapshot it so a later mutation cannot reach a frozen spec.
        return _FrozenMapping(validate_predeclared_segments(value) if value else {})

    @field_serializer("segments")
    def _dump_segments(self, value: Mapping[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
        return dict(value)

    @model_validator(mode="after")
    def _fields_match_kind(self):
        from increment.semantics.sequential import invalid_registration

        if self.expected_decision_sample_size is not None and (
            self.kind != "asymptotic_mean" or self.registration is not None
        ):
            invalid_registration(
                "expected sample size belongs to automatic asymptotic registration"
            )
        if self.adjustments and (self.kind != "asymptotic_mean" or self.registration is not None):
            invalid_registration(
                "pre-period adjustments belong to automatic asymptotic registration; an "
                "explicit registration declares them on ScalarMeanModel.adjustment"
            )
        if self.segments and self.registration is not None:
            invalid_registration(
                "predeclared segments belong to automatic registration; an explicit "
                "registration names each segment on its SequentialCell"
            )
        if self.baseline_rate is not None:
            if self.kind != "always_valid" or self.registration is not None:
                _definition_refusal(
                    "definition.inference.baseline_rate_route",
                    "baseline_rate centres the Beta prior of an automatic always_valid "
                    "registration; an explicit registration declares its priors on "
                    "SequentialModel, and asymptotic_mean uses no prior",
                )
            if not 0 < self.baseline_rate < 1:
                _definition_refusal(
                    "definition.inference.baseline_rate_domain",
                    f"baseline_rate={self.baseline_rate} must be a rate strictly between 0 and 1",
                )
        if self.registration is None:
            return self
        laws = {m.law for m in self.registration.models}
        has_asymptotic = bool(laws & set(ASYMPTOTIC_LAWS))
        if has_asymptotic != (self.kind == "asymptotic_mean"):
            if has_asymptotic and "bernoulli" in laws:
                from increment.errors import refuse
                from increment.semantics.sequential import MIXED_REQUIRES_ASYMPTOTIC_MEAN

                refuse(
                    MIXED_REQUIRES_ASYMPTOTIC_MEAN,
                    metrics=sorted(
                        m.metric for m in self.registration.models if m.law in ASYMPTOTIC_LAWS
                    ),
                )
            invalid_registration("asymptotic laws require explicit asymptotic_mean inference")
        return self


PlanEntry = str | ExperimentMetric


class MultiplicitySpec(_Base):
    """Multiplicity policy for a segmented readout family."""

    correction: Correction
    q: float | None = Field(default=None, gt=0, lt=1)

    @model_validator(mode="before")
    @classmethod
    def _default_bh_q(cls, value: object) -> object:
        if isinstance(value, dict) and value.get("correction") == "bh" and "q" not in value:
            return {**value, "q": 0.10}
        return value

    @model_validator(mode="after")
    def _q_only_for_bh(self) -> "MultiplicitySpec":
        if self.correction == "bh" and self.q is None:
            _definition_refusal(
                "definition.multiplicity.bh_requires_q",
                "BH multiplicity requires q",
            )
        if self.correction != "bh" and self.q is not None:
            _definition_refusal(
                "definition.multiplicity.bh_q_only_valid_for_bh",
                f"q is only valid for BH multiplicity, got {self.correction!r}",
                correction=self.correction,
            )
        return self


class AnalysisPlan(_Base):
    """The pre-registered decision rule: how evidence is judged.

    Bindings (``ExperimentMetric`` / ``MetricSpec``) say how estimates are
    computed; this says how they are judged.

    Every role entry is a ``PlanEntry``: either a bare metric name, or an
    ``ExperimentMetric`` binding carrying per-metric overrides
    (decision_method, sensitivity_methods, prior, guardrail margin). A metric
    may hold only one role.

    Error control by role
    ---------------------
    A role is a claim about which error rate a metric's verdict is protected
    against, so each role gets its own budget and its own procedure.

    - **primary** -- confirmatory, tested at ``alpha / n_primaries`` and split
      again across the metric's own non-control arms. Bonferroni familywise
      control at ``alpha`` across the confirmatory family.
    - **guardrail** -- a one-sided non-inferiority test at the full ``alpha``,
      never divided across guardrails, on the adverse side implied by the
      metric's ``preferred_direction`` and margin. A guardrail asks "is this
      materially worse", not "is this better": dividing ``alpha`` across
      guardrails would raise the false-negative rate for real harm, which is
      the error that matters here.
    - **secondary** -- the discovery family at ``q``, a level deliberately
      looser than ``alpha`` because these verdicts are not confirmatory. The
      procedure depends on ``inference``: BH/e-BH false-discovery control for
      fixed-horizon and ``AlwaysValid`` inference, and fixed-roster Bonferroni
      familywise control at ``q`` for ``AsymptoticMean``, whose look structure
      has no established step-up composition. Familywise control at ``q``
      implies false-discovery control at ``q``, so the weaker-procedure case
      never promises less than the step-up ones; it is only less powerful.
    - **unassigned** -- reported at ``alpha``, no multiplicity claim.

    Parameters
    ----------
    alpha : float
        Familywise level for the confirmatory roles. A primary's share is
        ``alpha / n_primaries``, split again across that metric's own
        non-control arms at estimation, so the shares sum to no more than
        ``alpha``. Guardrails test at ``alpha`` unsplit.
    q : float
        Secondary-family error level: BH/e-BH targets false discoveries for
        fixed-horizon/exact registered inference. ``AsymptoticMean`` instead
        uses fixed-roster Bonferroni familywise control at ``q``, allocating
        ``q / n_secondaries`` per secondary. It is not a quantile. These
        guarantees are distinct from confirmatory allocation at ``alpha``;
        see ``increment.estimation.family``.
    view_multiplicity : MultiplicitySpec | None
        Multiplicity for segmented as-of and breakout views. ``None`` keeps
        route defaults: randomized breakout BH at ``q`` (fixed-roster
        Bonferroni for ``AsymptoticMean``), encouragement breakout uncorrected,
        and as-of uncorrected unless segmented.
    alternative : {"two-sided", "greater", "less"}
        Direction of the test. A one-sided level is displayed through the
        standard alpha-doubling convention.
    primary : PlanEntry | tuple[PlanEntry, ...] | None
        The confirmatory metric(s). Accepts a single entry or a list;
        ``None`` declares no primary.
    secondaries : tuple[PlanEntry, ...] | None
        Metrics judged as a discovery family at ``q`` rather than against
        ``alpha``. A prior-bound entry sits outside the family.
    guardrails : tuple[PlanEntry, ...]
        Metrics that must not move adversely. A guardrail needs an
        explicit non-neutral ``preferred_direction`` on the metric, since
        a non-inferiority margin has no adverse side otherwise.
    inference : InferenceSpec | None
        Look policy. ``None`` means fixed-horizon: one analysis, no
        peeking guarantee.

    Examples
    --------
    >>> AnalysisPlan(
    ...     alpha=0.05,
    ...     q=0.10,
    ...     primary="revenue_per_user",
    ...     secondaries=["signups", "sessions"],
    ...     guardrails=[ExperimentMetric(metric="latency_p95", margin=0.01)],
    ...     inference=None,
    ... )  # doctest: +ELLIPSIS
    AnalysisPlan(alpha=0.05, q=0.1, ...)
    """

    alpha: float = Field(default=0.05, gt=0, lt=1)
    q: float = Field(default=0.10, gt=0, lt=1)
    view_multiplicity: MultiplicitySpec | None = None
    alternative: Alternative = "two-sided"
    primary: PlanEntry | tuple[PlanEntry, ...] | None = None
    secondaries: tuple[PlanEntry, ...] | None = None
    guardrails: tuple[PlanEntry, ...] = ()

    inference: InferenceSpec | None = None
    compliance: SequentialCompliancePolicy | None = None

    @property
    def primaries(self) -> list[PlanEntry]:
        if self.primary is None:
            return []
        if isinstance(self.primary, tuple):
            return list(self.primary)
        return [self.primary]

    def role_names(self) -> dict[str, str]:
        """Map every declared metric name to its role in the plan."""
        names: dict[str, str] = {}
        for entry in self.primaries:
            names[_plan_entry_name(entry)] = "primary"
        for entry in self.secondaries or []:
            names[_plan_entry_name(entry)] = "secondary"
        for entry in self.guardrails:
            names[_plan_entry_name(entry)] = "guardrail"
        return names

    def entry(self, name: str) -> ExperimentMetric | None:
        """The binding for *name*, or ``None`` for a bare-name entry or an
        unknown name."""
        for candidate in (*self.primaries, *(self.secondaries or []), *self.guardrails):
            if isinstance(candidate, ExperimentMetric) and candidate.metric == name:
                return candidate
        return None

    def entries(self, *, include_guardrails: bool = True) -> list[PlanEntry]:
        """Every declared entry across plan roles, in declaration order:
        primaries, then secondaries, then guardrails (omitted when
        *include_guardrails* is ``False``)."""
        groups = (
            self.primaries,
            self.secondaries or [],
            self.guardrails if include_guardrails else [],
        )
        return [entry for group in groups for entry in group]

    @model_validator(mode="after")
    def _no_duplicate_roles(self) -> "AnalysisPlan":
        roles: dict[str, list[str]] = {}
        for field, entries in (
            ("primary", self.primaries),
            ("secondaries", self.secondaries or []),
            ("guardrails", self.guardrails),
        ):
            for candidate in entries:
                roles.setdefault(_plan_entry_name(candidate), []).append(field)
        for name, fields in roles.items():
            distinct = list(dict.fromkeys(fields))
            if len(distinct) > 1:
                joined = " and ".join(f"{f!r}" for f in distinct)
                _definition_refusal(
                    "definition.analysis.appears_both_metric",
                    f"{name!r} appears in both {joined} -- a metric may hold only "
                    "one role in the analysis plan",
                    name=name,
                    joined=joined,
                )
            if len(fields) > 1:
                _definition_refusal(
                    "definition.analysis.appears_more_once",
                    f"{name!r} appears more than once in {distinct[0]!r} -- a metric "
                    "may hold only one role in the analysis plan",
                    name=name,
                    distinct=distinct[0],
                )
        return self

    @model_validator(mode="after")
    def _automatic_segmented_compliance_family(self) -> "AnalysisPlan":
        if (
            self.compliance is not None
            and self.compliance.family
            and self.inference is not None
            and self.inference.registration is None
            and self.inference.segments
        ):
            refuse_segmented_family_compliance(
                self.inference.segments, route="automatic sequential inference"
            )
        if self.compliance is not None and self.inference is None:
            from increment.semantics.sequential import invalid_registration

            invalid_registration("compliance requires a registered sequential inference")
        return self


def local_day(value: datetime, offset: timedelta) -> date:
    """Calendar day of a declared experiment time at the day boundary.

    A naive value is wall-clock time at the boundary; an aware value is converted to it.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        return value.date()
    return (value.astimezone(UTC) + offset).date()


class Experiment(AliasMixin, _ExplicitOnlyFields, _Base):
    _explicit_only_fields: ClassVar[frozenset[str]] = frozenset(
        {"day_boundary", "allocation", "design"}
    )

    name: str
    description: str | None = None
    exposure: str
    #: Declared exposure whose first occurrence marks eligibility. Narrows
    #: the analysis population; the assigned-population effect still reports too.
    trigger: str | None = None
    unit: str
    #: Randomization-grain column when assignment is coarser than ``unit``.
    #: Switches total-grain readouts to cluster-robust variance; every other grain refuses.
    cluster: str | None = None
    #: Grain at which a policy deployed from this experiment intervenes. ``"cluster"``
    #: requires an explicit declaration and a non-None ``cluster``; ``cluster``
    #: (a randomization or dependence grain) alone never implies it.
    intervention_grain: Literal["unit", "cluster"] = "unit"
    start: datetime
    #: Enrollment close, compared at DAY granularity: enrollment stays open
    #: through the end of this day, and any time-of-day on it is ignored.
    end: datetime | None = None
    #: When data collection stops; defaults to ``end``. Push past ``end``
    #: for a one-shot intervention so late enrollees still get a full window.
    observation_end: datetime | None = None
    plan: AnalysisPlan
    n_pre_periods: int = Field(default=0, ge=0)  # CUPED lookback in days; 0 disables
    control_group: str  # REQUIRED: which group_id is control
    #: Declared assignment weights, never estimated from observed group counts.
    allocation: Mapping[str, float] | None = None
    #: Optional non-randomized identification mechanism. Absent means
    #: Randomized, built from control_group/allocation above -- see
    #: resolved_design(). mechanism: "randomized" is not a valid value
    #: here: the top-level fields are randomized's only spelling.
    design: ExperimentDesignDeclaration | None = None
    breakouts: tuple[Breakout, ...] = ()
    factors: tuple[Factor, ...] = ()
    #: Fixed-offset day boundary for calendar-day bucketing (default "UTC");
    #: inherited from the definitions-level default unless declared here.
    day_boundary: str = "UTC"

    @field_validator("allocation")
    @classmethod
    def _snapshot_allocation(
        cls, value: Mapping[str, float] | None
    ) -> MappingProxyType[str, float] | None:
        return _freeze_allocation(value)

    @field_serializer("allocation")
    def _dump_allocation(
        self, value: Mapping[str, float] | None, info: object
    ) -> dict[str, float] | None:
        return _serialize_allocation(value, info)

    @model_validator(mode="after")
    def _check_allocation(self):
        _validate_allocation_semantics(self.control_group, self.allocation)
        return self

    @field_validator("start", "end", "observation_end", mode="before")
    @classmethod
    def _validate_schedule_type(cls, v: Any) -> Any:
        return None if v is None else _reject_numeric_datetime(v)

    @field_validator("n_pre_periods", mode="before")
    @classmethod
    def _validate_n_pre_periods(cls, v: Any) -> Any:
        return _reject_bool(v)

    @field_validator("day_boundary")
    @classmethod
    def _day_boundary_grammar(cls, v: str) -> str:
        return _validate_day_boundary(v)

    @property
    def day_boundary_offset(self) -> timedelta:
        """The declared day boundary as a signed offset from UTC."""
        m = _DAY_BOUNDARY_RE.match(self.day_boundary)
        assert m is not None  # validated
        if m.group(2) is None:
            return timedelta(0)
        sign = 1 if m.group(1) == "+" else -1
        return sign * timedelta(hours=int(m.group(2)), minutes=int(m.group(3)))

    @property
    def observation_horizon(self) -> datetime | None:
        """When observation stops: ``observation_end`` if declared, else ``end``.

        ``None`` means no declared bound - the caller falls back to the panel's own extent.
        """
        return self.observation_end or self.end

    @property
    def start_day(self) -> date:
        """The declared start as a day at ``day_boundary``."""
        return local_day(self.start, self.day_boundary_offset)

    @property
    def end_day(self) -> date | None:
        """The declared end as a day at ``day_boundary``, or ``None`` when open-ended."""
        return None if self.end is None else local_day(self.end, self.day_boundary_offset)

    @property
    def observation_horizon_day(self) -> date | None:
        """The observation horizon as a day at ``day_boundary``, or ``None`` when unbounded."""
        horizon = self.observation_horizon
        return None if horizon is None else local_day(horizon, self.day_boundary_offset)

    @property
    def metric_names(self) -> list[str]:
        """Declared metric names, in declaration order across every plan
        role - primaries, then secondaries, then guardrails."""
        return [
            _plan_entry_name(e)
            for e in (*self.plan.primaries, *(self.plan.secondaries or []), *self.plan.guardrails)
        ]

    @property
    def guardrail_names(self) -> list[str]:
        """Declared guardrail names, in declaration order - same
        shorthand/binding resolution as :attr:`metric_names`."""
        return [_plan_entry_name(g) for g in self.plan.guardrails]

    def resolved_design(self) -> "Randomized | Encouragement | Observational":
        """Compose this experiment's identification design from its
        always-present control_group/allocation plus the optional
        design declaration. The one place every production read site
        builds a design from an Experiment -- never construct
        Randomized(control_group=experiment.control_group, ...) ad hoc."""
        from increment.semantics.design import AdjustmentSet as _AdjustmentSet
        from increment.semantics.design import Encouragement as _Encouragement
        from increment.semantics.design import Observational as _Observational
        from increment.semantics.design import Randomized as _Randomized

        if self.design is None:
            return _Randomized(control_group=self.control_group, allocation=self.allocation)
        if isinstance(self.design, EncouragementDeclaration):
            return _Encouragement(
                control_group=self.control_group,
                allocation=self.allocation,
                uptake=self.design.uptake,
                exclusion_restriction=self.design.exclusion_restriction,
                one_sided=self.design.one_sided,
            )
        return _Observational(
            control_group=self.control_group,
            adjustment=_AdjustmentSet(covariates=tuple(c.property for c in self.design.covariates)),
        )

    @property
    def bindings(self) -> dict[str, "ExperimentMetric"]:
        """Name -> binding for every non-shorthand entry across every plan
        role; a bare-string entry has no binding here."""
        return {
            entry.metric: entry
            for entry in (
                *self.plan.primaries,
                *(self.plan.secondaries or []),
                *self.plan.guardrails,
            )
            if isinstance(entry, ExperimentMetric)
        }

    @model_validator(mode="after")
    def _validate_cuped_bindings(self):
        if self.n_pre_periods > 0:
            return self
        for name, binding in self.bindings.items():
            if binding.wants_cuped:
                _definition_refusal(
                    "definition.experiment.metric_declares_cuped",
                    f"experiment '{self.name}': metric '{name}' declares a CUPED "
                    f"method but n_pre_periods=0 -- no pre-period covariate can "
                    f"be materialized",
                    experiment_name=self.name,
                    metric_name=name,
                )
        return self

    @model_validator(mode="after")
    def _validate_cluster(self):
        if self.cluster is None:
            if self.intervention_grain == "cluster":
                _definition_refusal(
                    "definition.experiment.intervention_grain_without_cluster",
                    f"experiment '{self.name}': intervention_grain='cluster' declares a "
                    f"whole-cluster policy deployment, but no cluster= column is "
                    f"declared -- there is no cluster grain to intervene on.",
                    name=self.name,
                )
            return self
        if self.cluster == self.unit:
            _definition_refusal(
                "definition.experiment.cluster_names_same",
                f"experiment '{self.name}': cluster '{self.cluster}' names the same "
                f"column as unit -- when the randomization grain IS the analysis "
                f"grain, declare unit alone; cluster= exists for a coarser "
                f"randomization grain.",
                name=self.name,
                cluster=self.cluster,
            )
        if self.n_pre_periods > 0 and any(b.wants_cuped for b in self.bindings.values()):
            _definition_refusal(
                "definition.experiment.cluster_combined_n",
                f"experiment '{self.name}': cluster '{self.cluster}' cannot be "
                f"combined with n_pre_periods={self.n_pre_periods} (CUPED) -- the "
                f"clustered collapse carries no per-unit covariate moments, so the "
                f"adjustment would be silently wrong. Drop cluster= or set "
                f"n_pre_periods: 0.",
                name=self.name,
                cluster=self.cluster,
                n_pre_periods=self.n_pre_periods,
            )
        return self

    @model_validator(mode="after")
    def _validate_schedule(self):
        # All-or-nothing timezone consistency, checked first: a mixed
        # naive/aware pair fails downstream datetime arithmetic with a confusing TypeError.
        declared = [
            (name, v)
            for name, v in (
                ("start", self.start),
                ("end", self.end),
                ("observation_end", self.observation_end),
            )
            if v is not None
        ]
        awareness = {name: v.tzinfo is not None for name, v in declared}
        if len(set(awareness.values())) > 1:
            aware = sorted(n for n, a in awareness.items() if a)
            naive = sorted(n for n, a in awareness.items() if not a)
            _definition_refusal(
                "definition.experiment.timezone_aware_but",
                f"experiment '{self.name}': {', '.join(aware)} "
                f"{'is' if len(aware) == 1 else 'are'} timezone-aware but "
                f"{', '.join(naive)} {'is' if len(naive) == 1 else 'are'} naive -- "
                f"mixed datetimes fail downstream date arithmetic. Declare "
                f"start/end/observation_end all naive or all timezone-aware.",
                name=self.name,
            )
        # Edges compare as days at the boundary, the grain enrollment uses: an earlier
        # day than `start` is an inverted range.
        if self.end_day is not None and self.end_day < self.start_day:
            _definition_refusal(
                "definition.experiment.end_before_start",
                f"experiment '{self.name}': end={self.end} is before "
                f"start={self.start} -- an inverted enrollment window "
                f"enrolls no units; swap or correct the dates.",
                name=self.name,
                end=self.end,
                start=self.start,
            )
        if self.observation_end is None:
            return self
        if self.end is None:
            _definition_refusal(
                "definition.experiment.observation_end_requires_end",
                f"experiment '{self.name}': observation_end is set but end is not. "
                f"observation_end extends the enrollment end; declare end first.",
                experiment=self.name,
            )
        offset = self.day_boundary_offset
        if local_day(self.observation_end, offset) < local_day(self.end, offset):
            _definition_refusal(
                "definition.experiment.observation_end_before_end",
                f"experiment '{self.name}': observation_end={self.observation_end} "
                f"is before end={self.end}. Observation cannot stop before enrollment "
                f"does -- that would discard units enrolled in between.",
                experiment=self.name,
            )
        return self


def window_days(experiment: Experiment) -> dict[str, date | None]:
    """Boundary-local day of each declared window edge (None when the edge is absent)."""
    return {
        "start": experiment.start_day,
        "end": experiment.end_day,
        "observation_horizon": experiment.observation_horizon_day,
    }


def _check_dtype_compatibility(
    flt: Filter,
    prop: Property,
    label: str,
    source_name: str,
    errors: list[tuple[str, str]],
):
    """Check that every value in *flt.values* is compatible with *prop.dtype*."""
    bad: list[str] = []
    for v in flt.values:
        ok = True
        if prop.dtype == "string":
            ok = isinstance(v, str)
        elif prop.dtype == "int":
            ok = isinstance(v, int) and not isinstance(v, bool)
        elif prop.dtype == "float":
            ok = isinstance(v, int | float) and not isinstance(v, bool)
        elif prop.dtype == "bool":
            ok = isinstance(v, bool)
        elif prop.dtype == "date":
            # ISO-parseable strings only: Filter.values is typed str|int|float|bool,
            # so a raw date object never reaches here; quote dates in YAML.
            if isinstance(v, str):
                try:
                    date.fromisoformat(v)
                    ok = True
                except ValueError:
                    ok = False
            else:
                ok = False
        if not ok:
            bad.append(repr(v))
    if bad:
        errors.append(
            (
                "definition.check_dtype.filter_property_incompatible",
                f"{label} filter on property '{prop.name}' (dtype={prop.dtype}) "
                f"has incompatible value(s): {', '.join(bad)}",
            )
        )


@dataclass(frozen=True, slots=True)
class _DefinitionIndex:
    """Immutable lookup tables shared by definition validation phases."""

    dim_by_name: Mapping[str, DimSource]
    fact_names: frozenset[str]
    fact_to_source: Mapping[str, str]
    fact_to_column: Mapping[str, str | None]
    source_entities: Mapping[str, list[str]]
    source_properties: Mapping[str, Mapping[str, Property]]
    source_filter_properties: Mapping[str, Mapping[str, Property]]
    metric_names: frozenset[str]
    metric_entities: Mapping[str, str]
    metric_by_name: Mapping[str, Metric]
    exposure_names: frozenset[str]
    exposure_by_name: Mapping[str, Exposure]
    report_only: Mapping[str, str]


def _with_inherited_day_boundary(experiment: Experiment, day_boundary: str) -> Experiment:
    """Copy *experiment* with an inherited day boundary unless it declares its own.

    Restoring the original ``model_fields_set`` keeps an undeclared boundary unset, so a
    dump/reload re-inherits later changes.
    """
    if "day_boundary" in experiment.model_fields_set:
        return experiment
    original_fields_set = set(experiment.model_fields_set)
    copied = experiment.model_copy(update={"day_boundary": day_boundary})
    object.__setattr__(copied, "__pydantic_fields_set__", original_fields_set)
    return copied


# Definitions: the top-level container loaded from YAML


# Definitions is the direct coded-error boundary for nested semantic declarations.
# Schema-driven validation uses unwrap_coded at the caller's boundary.
class Definitions(CodedModel, _Base):
    """Top-level semantic definitions loaded from a directory or YAML file.

    Construct it with the validated source, metric, exposure, and experiment
    declarations, or obtain it from :func:`increment.semantics.load`.
    """

    dialect: str | None = None
    week_start: Literal["monday", "sunday"] = "monday"
    #: Org-level default day boundary, inherited by every experiment that
    #: doesn't declare its own.
    day_boundary: str = "UTC"
    fact_sources: tuple[FactSource, ...] = ()
    dim_sources: tuple[DimSource, ...] = ()
    exposures: tuple[Exposure, ...] = ()
    metrics: tuple[Metric, ...] = ()
    experiments: tuple[Experiment, ...] = ()

    @field_validator("day_boundary")
    @classmethod
    def _day_boundary_grammar(cls, v: str) -> str:
        return _validate_day_boundary(v)

    @model_validator(mode="after")
    def _inherit_day_boundary(self) -> "Definitions":
        # The model owns inheritance so every construction path uses the same day
        # bucketing. Restoring each experiment's original `model_fields_set` keeps an
        # undeclared day_boundary unset, so a dump/reload re-inherits later changes.
        inherited = [
            _with_inherited_day_boundary(exp, self.day_boundary) for exp in self.experiments
        ]
        changed = any(new is not old for new, old in zip(inherited, self.experiments, strict=True))
        if not changed:
            return self
        # Mutate in place and return `self`: pydantic does not install a
        # replacement instance returned from a model-level after-validator on
        # the direct-construction path, so `Definitions(...)` would silently keep
        # the experiment's default boundary while `model_validate` inherited.
        object.__setattr__(self, "experiments", tuple(inherited))
        return self

    # ── runtime lookups ────────────────────────────────────────────────

    def metric(self, name: str) -> Metric | None:
        for m in self.metrics:
            if m.name == name:
                return m
        return None

    def experiment(self, name: str) -> Experiment | None:
        for e in self.experiments:
            if e.name == name:
                return e
        return None

    def fact_source_for(self, fact_or_source_name: str) -> FactSource | None:
        """Return the FactSource that owns *fact_or_source_name*.

        Source names take precedence globally, followed by fact names across
        every source, and finally properties (including dimension properties).
        Keeping the namespaces separate prevents a property on one source
        from shadowing a fact on another source.
        """
        for fs in self.fact_sources:
            if fs.name == fact_or_source_name:
                return fs
        for fs in self.fact_sources:
            if any(f.name == fact_or_source_name for f in fs.facts):
                return fs
        for fs in self.fact_sources:
            if any(p.name == fact_or_source_name for p in self.properties_of(fs)):
                return fs
        return None

    def properties_of(self, fs: FactSource) -> tuple[Property, ...]:
        """*fs*'s own properties plus those contributed by its dims.

        The single read path for "what properties does this source expose";
        an unknown dim name is skipped rather than raising.
        """
        if not fs.dims:
            return fs.properties
        dim_by_name = {d.name: d for d in self.dim_sources}
        merged = list(fs.properties)
        for dim_name in fs.dims:
            dim = dim_by_name.get(dim_name)
            if dim is not None:
                merged.extend(dim.properties)
        return tuple(merged)

    # ── cross-reference validator ──────────────────────────────────────

    def _raise_errors(self, errors: list[tuple[str, str]]) -> None:
        """Refuse once, carrying every accumulated (code, message) pair."""
        if errors:
            message = "Definition consistency errors:\n  - " + "\n  - ".join(
                msg for _, msg in errors
            )
            _definition_refusal("definition.invalid", message, errors=tuple(errors))

    @model_validator(mode="after")
    def _cross_check(self):
        errors: list[tuple[str, str]] = []
        index = self._index_sources_and_validate_names(errors)
        self._validate_metrics(index, errors)
        self._validate_exposures(index, errors)
        self._validate_experiments(index, errors)
        self._raise_errors(errors)
        return self

    def _index_sources_and_validate_names(self, errors: list[tuple[str, str]]) -> _DefinitionIndex:
        dim_names: set[str] = set()
        for dim in self.dim_sources:
            if dim.name in dim_names:
                errors.append(
                    (
                        "definition.index_sources.duplicate_dim_source",
                        f"duplicate dim source name '{dim.name}'",
                    )
                )
            dim_names.add(dim.name)
        dim_by_name = {dim.name: dim for dim in self.dim_sources}

        fact_names: set[str] = set()
        fact_to_source: dict[str, str] = {}
        fact_to_column: dict[str, str | None] = {}
        source_entities: dict[str, list[str]] = {}
        source_properties: dict[str, dict[str, Property]] = {}
        source_filter_properties: dict[str, dict[str, Property]] = {}
        seen_source_props: dict[str, set[str]] = {}
        for source in self.fact_sources:
            if source.name in source_entities:
                _definition_refusal(
                    "definition.duplicates",
                    f"duplicate fact_source name '{source.name}'",
                    name=source.name,
                )
            source_entities[source.name] = list(source.entities)
            for dim_name in source.dims:
                dim = dim_by_name.get(dim_name)
                if dim is None:
                    errors.append(
                        (
                            "definition.index_sources.fact_source_references",
                            f"fact source '{source.name}' references unknown dim source '{dim_name}'",
                        )
                    )
                    continue
                if dim.entity not in source.entities:
                    errors.append(
                        (
                            "definition.index_sources.fact_source_joins",
                            f"fact source '{source.name}' joins dim source '{dim_name}' on "
                            f"entity '{dim.entity}', which is not in its entities {source.entities}",
                        )
                    )
                for prop in dim.properties:
                    if prop.name in _DIM_RESERVED_COLUMNS:
                        errors.append(
                            (
                                "definition.index_sources.dim_source_property",
                                f"dim source '{dim_name}' property '{prop.name}' uses the "
                                f"reserved column name '{prop.name}' -- reserved: "
                                f"{sorted(_DIM_RESERVED_COLUMNS)}",
                            )
                        )
            merged_props = list(source.properties) + [
                prop
                for dim_name in source.dims
                if (dim := dim_by_name.get(dim_name)) is not None
                for prop in dim.properties
            ]
            properties_by_name = {prop.name: prop for prop in merged_props}
            source_properties[source.name] = properties_by_name
            source_filter_properties.setdefault(source.name, properties_by_name)
            for fact in source.facts:
                if fact.name in fact_names:
                    errors.append(
                        (
                            "definition.index_sources.duplicate_fact_name",
                            f"duplicate fact name '{fact.name}' across fact sources",
                        )
                    )
                fact_names.add(fact.name)
                fact_to_source[fact.name] = source.name
                fact_to_column[fact.name] = fact.column

            seen_props = seen_source_props.setdefault(source.name, set())
            for prop in merged_props:
                if prop.name in seen_props:
                    errors.append(
                        (
                            "definition.index_sources.duplicate_property_name",
                            f"duplicate property name '{prop.name}' in fact source '{source.name}'",
                        )
                    )
                seen_props.add(prop.name)

        for fact_name, owner in fact_to_source.items():
            if fact_name in source_entities and owner != fact_name:
                errors.append(
                    (
                        "definition.index_sources.fact_owned_by",
                        f"fact '{fact_name}' (owned by source '{owner}') shares its "
                        f"name with a different fact source -- fact and source names "
                        f"share one lookup namespace (fact_source_for); rename one",
                    )
                )
        for source_name, props in source_properties.items():
            for prop_name in props:
                if prop_name in source_entities and prop_name != source_name:
                    errors.append(
                        (
                            "definition.index_sources.property_source_shares",
                            f"property '{prop_name}' (in source '{source_name}') shares its "
                            f"name with a different fact source -- property and source "
                            f"names share one lookup namespace (fact_source_for); rename one",
                        )
                    )

        metric_name_set: set[str] = set()
        metric_entities: dict[str, str] = {}
        metric_by_name: dict[str, Metric] = {}
        report_only: dict[str, str] = {}
        seen_metric_names: set[str] = set()
        for metric in self.metrics:
            metric_name_set.add(metric.name)
            metric_by_name.setdefault(metric.name, metric)
            if metric.name not in seen_metric_names:
                seen_metric_names.add(metric.name)
                if isinstance(metric, MetricBase):
                    metric_entities[metric.name] = metric.entity
            if metric.type in ("total", "active"):
                report_only[metric.name] = metric.type
        exposure_name_set: set[str] = set()
        exposure_by_name: dict[str, Exposure] = {}
        for exposure in self.exposures:
            exposure_name_set.add(exposure.name)
            exposure_by_name.setdefault(exposure.name, exposure)
        metric_names = frozenset(metric_name_set)
        exposure_names = frozenset(exposure_name_set)
        return _DefinitionIndex(
            dim_by_name=MappingProxyType(dim_by_name),
            fact_names=frozenset(fact_names),
            fact_to_source=MappingProxyType(fact_to_source),
            fact_to_column=MappingProxyType(fact_to_column),
            source_entities=MappingProxyType(source_entities),
            source_properties=MappingProxyType(
                {name: MappingProxyType(props) for name, props in source_properties.items()}
            ),
            source_filter_properties=MappingProxyType(
                {name: MappingProxyType(props) for name, props in source_filter_properties.items()}
            ),
            metric_names=metric_names,
            metric_entities=MappingProxyType(metric_entities),
            metric_by_name=MappingProxyType(metric_by_name),
            exposure_names=exposure_names,
            exposure_by_name=MappingProxyType(exposure_by_name),
            report_only=MappingProxyType(report_only),
        )

    def _validate_metrics(self, index: _DefinitionIndex, errors: list[tuple[str, str]]):
        seen_metric_names: set[str] = set()
        for metric in self.metrics:
            if metric.name in seen_metric_names:
                errors.append(
                    (
                        "definition.validate_metrics.duplicate_metric_name",
                        f"duplicate metric name '{metric.name}'",
                    )
                )
                continue
            seen_metric_names.add(metric.name)
            if metric.type != "ratio":
                self._check_metric_fact(
                    metric.name,
                    metric.fact,
                    index.fact_names,
                    index.fact_to_source,
                    errors,
                )
                self._validate_filters_on_source(
                    metric.name,
                    metric.fact,
                    index,
                    metric.filters,
                    errors,
                )
                if isinstance(metric, MeanMetric | TotalMetric | QuantileMetric):
                    self._check_value_aggregation(
                        metric.name,
                        metric.fact,
                        metric.aggregation,
                        index.fact_to_column,
                        errors,
                    )
                continue
            for part_name, part in (
                ("numerator", metric.numerator),
                ("denominator", metric.denominator),
            ):
                label = f"{metric.name} ({part_name})"
                self._check_metric_fact(
                    label,
                    part.fact,
                    index.fact_names,
                    index.fact_to_source,
                    errors,
                )
                self._validate_filters_on_source(
                    label,
                    part.fact,
                    index,
                    part.filters,
                    errors,
                )
                self._check_value_aggregation(
                    label,
                    part.fact,
                    part.aggregation,
                    index.fact_to_column,
                    errors,
                )

    def _validate_exposures(self, index: _DefinitionIndex, errors: list[tuple[str, str]]):
        seen_exposure_names: set[str] = set()
        for exposure in self.exposures:
            if exposure.name in seen_exposure_names:
                errors.append(
                    (
                        "definition.validate_exposures.duplicate_exposure_name",
                        f"duplicate exposure name '{exposure.name}'",
                    )
                )
            seen_exposure_names.add(exposure.name)
            if exposure.fact:
                if exposure.fact not in index.fact_to_source:
                    errors.append(
                        (
                            "definition.validate_exposures.exposure_references_unknown",
                            f"exposure '{exposure.name}' references unknown fact '{exposure.fact}'",
                        )
                    )
                else:
                    self._validate_filters_on_source(
                        f"exposure '{exposure.name}'",
                        exposure.fact,
                        index,
                        exposure.filters,
                        errors,
                    )

    def _validate_experiments(self, index: _DefinitionIndex, errors: list[tuple[str, str]]):
        experiment_names: set[str] = set()
        for experiment in self.experiments:
            if experiment.name in experiment_names:
                errors.append(
                    (
                        "definition.validate_experiments.duplicate_experiment_name",
                        f"duplicate experiment name '{experiment.name}'",
                    )
                )
            experiment_names.add(experiment.name)
            self._validate_experiment(experiment, index, errors)

    def _validate_experiment(
        self,
        experiment: Experiment,
        index: _DefinitionIndex,
        errors: list[tuple[str, str]],
    ):
        self._validate_experiment_exposure_and_trigger(experiment, index, errors)
        self._validate_experiment_plan(experiment, index, errors)
        self._validate_experiment_breakouts_and_factors(experiment, index, errors)
        self._validate_experiment_cluster(experiment, errors)
        self._validate_experiment_design(experiment, index, errors)

    def _validate_experiment_exposure_and_trigger(
        self,
        experiment: Experiment,
        index: _DefinitionIndex,
        errors: list[tuple[str, str]],
    ):
        for metric_name in experiment.metric_names:
            if metric_name in index.report_only:
                errors.append(
                    (
                        "definition.validate_experiment.metric_report_type",
                        f"experiment '{experiment.name}': metric '{metric_name}' is report-only "
                        f"(type '{index.report_only[metric_name]}') -- it has no per-unit "
                        f"variance, so the estimation engine cannot serve it; use it via "
                        f"increment.Report instead",
                    )
                )

        if experiment.exposure not in index.exposure_names:
            valid = sorted(index.exposure_names) if index.exposure_names else ["(no exposures)"]
            errors.append(
                (
                    "definition.validate_experiment.references_unknown_exposure",
                    f"experiment '{experiment.name}' references unknown exposure "
                    f"'{experiment.exposure}'; valid exposures: {valid}",
                )
            )
        else:
            exposure = index.exposure_by_name[experiment.exposure]
            if exposure.fact and exposure.fact in index.fact_to_source:
                _check_unit_in_source_entities(
                    experiment.name,
                    experiment.unit,
                    f"exposure fact '{exposure.fact}'",
                    index.fact_to_source[exposure.fact],
                    index.source_entities,
                    errors,
                )

        if experiment.trigger is None:
            return
        if experiment.trigger not in index.exposure_names:
            valid = sorted(index.exposure_names) if index.exposure_names else ["(no exposures)"]
            errors.append(
                (
                    "definition.validate_experiment.references_unknown_trigger",
                    f"experiment '{experiment.name}' references unknown trigger "
                    f"'{experiment.trigger}'; valid exposures: {valid}",
                )
            )
        elif experiment.trigger == experiment.exposure:
            errors.append(
                (
                    "definition.validate_experiment.trigger_also_exposure",
                    f"experiment '{experiment.name}': trigger '{experiment.trigger}' is also the "
                    "exposure -- that narrows the population to itself and "
                    "reports the assigned effect twice. Declare a separate "
                    "eligibility fact, or drop the trigger.",
                )
            )
        else:
            trigger = index.exposure_by_name[experiment.trigger]
            if trigger.fact and trigger.fact in index.fact_to_source:
                _check_unit_in_source_entities(
                    experiment.name,
                    experiment.unit,
                    f"trigger fact '{trigger.fact}'",
                    index.fact_to_source[trigger.fact],
                    index.source_entities,
                    errors,
                )

    def _validate_experiment_plan(
        self,
        experiment: Experiment,
        index: _DefinitionIndex,
        errors: list[tuple[str, str]],
    ):
        for metric_name in (
            _plan_entry_name(entry) for entry in experiment.plan.entries(include_guardrails=False)
        ):
            self._validate_experiment_metric(
                experiment,
                metric_name,
                "metric",
                index,
                errors,
            )
        for guardrail_name in experiment.guardrail_names:
            self._validate_experiment_metric(
                experiment,
                guardrail_name,
                "guardrail",
                index,
                errors,
            )
        for name, binding in experiment.bindings.items():
            metric = index.metric_by_name.get(name)
            if (
                binding.wants_finite_sample
                and metric is not None
                and metric.type not in ("conversion", "retention")
            ):
                errors.append(
                    (
                        FINITE_SAMPLE_METRIC_TYPE.code,
                        f"experiment '{experiment.name}': metric '{name}': "
                        + FINITE_SAMPLE_METRIC_TYPE.render(metric_type=metric.type),
                    )
                )

    @staticmethod
    def _validate_experiment_metric(
        experiment: Experiment,
        name: str,
        role: str,
        index: _DefinitionIndex,
        errors: list[tuple[str, str]],
    ):
        if name not in index.metric_names:
            valid = sorted(index.metric_names) if index.metric_names else ["(no metrics)"]
            errors.append(
                (
                    "definition.validate_experiment.references_unknown_metrics",
                    f"experiment '{experiment.name}' references unknown {role} "
                    f"'{name}'; valid metrics: {valid}",
                )
            )
            return
        metric = index.metric_by_name.get(name)
        if metric is not None:
            _visit_facts_for_metric(
                metric,
                experiment.name,
                experiment.unit,
                index.fact_to_source,
                index.source_entities,
                errors,
            )
        if name not in index.report_only and index.metric_entities.get(name) != experiment.unit:
            errors.append(
                (
                    "definition.validate_experiment.unit_but_entity",
                    f"experiment '{experiment.name}' has unit '{experiment.unit}' but {role} "
                    f"'{name}' has entity '{index.metric_entities.get(name)}'",
                )
            )

    def _validate_experiment_breakouts_and_factors(
        self,
        experiment: Experiment,
        index: _DefinitionIndex,
        errors: list[tuple[str, str]],
    ):
        self._check_breakouts(
            experiment,
            index.source_entities,
            index.source_properties,
            index.fact_to_source,
            index.metric_by_name,
            errors,
        )
        self._check_factors(
            experiment,
            index.source_entities,
            index.source_properties,
            errors,
        )
        self._check_observational_covariates(
            experiment,
            index.source_entities,
            index.source_properties,
            errors,
        )

    @staticmethod
    def _validate_experiment_cluster(experiment: Experiment, errors: list[tuple[str, str]]):
        # A CUPED binding with n_pre_periods == 0 refuses on the Experiment
        # itself (definition.experiment.metric_declares_cuped), ratio or not: a
        # ratio metric's warehouse covariate is its numerator's pre-period total.
        if experiment.cluster is not None and (experiment.breakouts or experiment.factors):
            errors.append(
                (
                    "definition.validate_experiment.declares_cluster_alongside",
                    f"experiment '{experiment.name}' declares cluster '{experiment.cluster}' "
                    f"alongside breakouts/factors -- cluster-robust inference "
                    f"is total-grain only, so these could never be served; "
                    f"drop them or drop cluster",
                )
            )

    def _validate_experiment_design(
        self,
        experiment: Experiment,
        index: _DefinitionIndex,
        errors: list[tuple[str, str]],
    ):
        """Bind an EncouragementDeclaration's uptake fact to the fact
        catalogue, the same existence + unit-membership check
        exposure/trigger already get. ObservationalDeclaration's
        covariates are not bound here -- the covariate-join work that
        validates {property, source} against declared Properties happens
        separately."""
        design = experiment.design
        if not isinstance(design, EncouragementDeclaration):
            return
        fact = design.uptake.fact
        self._check_metric_fact(
            f"experiment '{experiment.name}' design.uptake",
            fact,
            index.fact_names,
            index.fact_to_source,
            errors,
        )
        if fact in index.fact_to_source:
            _check_unit_in_source_entities(
                experiment.name,
                experiment.unit,
                f"design.uptake fact '{fact}'",
                index.fact_to_source[fact],
                index.source_entities,
                errors,
            )

    # ── internal helpers ───────────────────────────────────────────────
    @staticmethod
    def _check_metric_fact(
        label: str,
        fact: str,
        fact_names: Collection[str],
        fact_to_source: Mapping[str, str],
        errors: list[tuple[str, str]],
    ):
        if fact not in fact_to_source:
            valid = sorted(fact_names) if fact_names else ["(no facts defined)"]
            errors.append(
                (
                    "definition.check_metric.references_unknown_fact",
                    f"metric '{label}' references unknown fact '{fact}'; valid facts: {valid}",
                )
            )

    @staticmethod
    def _check_value_aggregation(
        label: str,
        fact: str,
        aggregation: str,
        fact_to_column: Mapping[str, str | None],
        errors: list[tuple[str, str]],
    ):
        """Reject value aggregations on facts with column=None (occurrence-only)."""
        value_aggs = frozenset(
            {"sum", "avg_event", "avg_calendar_day", "min", "max", "count_distinct"}
        )
        if aggregation not in value_aggs:
            return
        col = fact_to_column.get(fact)
        if col is not None:
            return
        errors.append(
            (
                "definition.check_value.metric_uses_aggregation",
                f"metric '{label}' uses aggregation '{aggregation}' on fact "
                f"'{fact}' which has column=None (occurrence-only); "
                f"only 'count' aggregation is supported",
            )
        )

    def _validate_filters_on_source(
        self,
        label: str,
        fact: str,
        index: _DefinitionIndex,
        filters: Sequence[Filter],
        errors: list[tuple[str, str]],
    ):
        if fact not in index.fact_to_source:
            return  # already reported
        source_name = index.fact_to_source[fact]
        source_props = index.source_filter_properties[source_name]
        for flt in filters:
            if flt.property not in source_props:
                errors.append(
                    (
                        "definition.validate_filters.filter_references_unknown",
                        f"{label} filter references unknown property "
                        f"'{flt.property}' on source '{source_name}'",
                    )
                )
                continue
            prop = source_props[flt.property]
            _check_dtype_compatibility(flt, prop, label, source_name, errors)

    def _check_breakouts(
        self,
        e: Experiment,
        source_entities: Mapping[str, list[str]],
        source_properties: Mapping[str, Mapping[str, Property]],
        fact_to_source: Mapping[str, str],
        metric_by_name: Mapping[str, Metric],
        errors: list[tuple[str, str]],
    ):
        """Validate *e.breakouts*: source resolution, dtype, duplicates, and
        cross-source coverage against every metric in the experiment."""
        seen_pairs: set[tuple[str, str]] = set()
        for b in e.breakouts:
            source_name = _resolve_breakout_source(e, b, source_entities, source_properties, errors)
            if source_name is None:
                continue

            pair = (b.property, source_name)
            if pair in seen_pairs:
                errors.append(
                    (
                        "definition.check_breakouts.experiment_duplicate_breakout",
                        f"experiment '{e.name}' has duplicate breakout on property "
                        f"'{b.property}' from source '{source_name}'",
                    )
                )
                continue
            seen_pairs.add(pair)

            prop = source_properties[source_name][b.property]
            if prop.as_of == "event_time":
                errors.append(
                    (
                        "definition.check_breakouts.experiment_breakout_property",
                        f"experiment '{e.name}' breakout property '{b.property}' "
                        f"(source '{source_name}') has as_of='event_time' -- "
                        f"conditioning a segment on a value measured at/after "
                        f"exposure biases the result (collider / differential "
                        f"composition). Declare as_of='pre_exposure' to scope the "
                        f"lookup to before first exposure, or as_of='static' only "
                        f"if the value cannot change for the unit's lifetime",
                    )
                )

            if prop.dtype not in ("string", "int", "bool"):
                errors.append(
                    (
                        "definition.check_breakouts.experiment_breakout_property_unsupported_dtype",
                        f"experiment '{e.name}' breakout property '{b.property}' "
                        f"(source '{source_name}') has dtype '{prop.dtype}'; only "
                        f"string/int/bool properties support group-by breakouts",
                    )
                )

            for metric_name in e.metric_names:
                m_obj = metric_by_name.get(metric_name)
                if m_obj is None:
                    continue  # unknown metric already reported
                for src_name in dict.fromkeys(_breakout_metric_sources(m_obj, fact_to_source)):
                    props = source_properties.get(src_name, {})
                    entities = source_entities.get(src_name, [])
                    if b.property in props and e.unit in entities:
                        continue
                    if b.skip_missing:
                        continue
                    errors.append(
                        (
                            "definition.check_breakouts.metric_source_does",
                            f"metric '{metric_name}' (source '{src_name}') does not "
                            f"carry breakout property '{b.property}'",
                        )
                    )

    def _check_factors(
        self,
        e: Experiment,
        source_entities: Mapping[str, list[str]],
        source_properties: Mapping[str, Mapping[str, Property]],
        errors: list[tuple[str, str]],
    ):
        """Validate *e.factors*: source resolution, dtype, duplicates, pre-exposure."""
        seen: set[tuple[str, str]] = set()
        for f in e.factors:
            source_name = _resolve_breakout_source(e, f, source_entities, source_properties, errors)
            if source_name is None:
                continue
            pair = (f.property, source_name)
            if pair in seen:
                errors.append(
                    (
                        "definition.check_factors.experiment_duplicate_factor",
                        f"experiment '{e.name}' has duplicate factor on property "
                        f"'{f.property}' from source '{source_name}'",
                    )
                )
                continue
            seen.add(pair)

            prop = source_properties[source_name][f.property]
            if prop.as_of == "event_time":
                errors.append(
                    (
                        "definition.check_factors.experiment_factor_property",
                        f"experiment '{e.name}' factor property '{f.property}' "
                        f"(source '{source_name}') has as_of='event_time' -- "
                        "absorbing a value that can change during the experiment "
                        "biases the overall effect. Declare as_of='pre_exposure' "
                        "or as_of='static'.",
                    )
                )

            if prop.dtype not in ("string", "int", "bool"):
                errors.append(
                    (
                        "definition.check_factors.experiment_factor_property_unsupported_dtype",
                        f"experiment '{e.name}' factor property '{f.property}' "
                        f"(source '{source_name}') has dtype '{prop.dtype}'; only "
                        f"string/int/bool properties can be absorbed as a categorical factor",
                    )
                )

    def _check_observational_covariates(
        self,
        e: Experiment,
        source_entities: Mapping[str, list[str]],
        source_properties: Mapping[str, Mapping[str, Property]],
        errors: list[tuple[str, str]],
    ):
        """Validate an observational design's covariates: each must resolve to
        one pre-exposure/static adjustable property on a source carrying the
        experiment's unit -- breakout source resolution, then a numeric
        (int/float/bool) or categorical (string) dtype."""
        design = e.design
        if not isinstance(design, ObservationalDeclaration):
            return
        for covariate in design.covariates:
            name = covariate.property
            source_name = _resolve_covariate_source(
                e, covariate, source_entities, source_properties, errors
            )
            if source_name is None:
                continue
            prop = source_properties[source_name][name]
            if prop.as_of == "event_time":
                errors.append(
                    (
                        "definition.check_observational.covariate_as_of",
                        f"experiment '{e.name}' observational covariate '{name}' "
                        f"(source '{source_name}') has as_of='event_time' -- "
                        f"conditioning on a value measured at/after exposure biases "
                        f"the adjustment. Declare as_of='pre_exposure' or 'static'",
                    )
                )
            if prop.dtype not in ("int", "float", "bool", "string"):
                errors.append(
                    (
                        "definition.check_observational.covariate_dtype",
                        f"experiment '{e.name}' observational covariate '{name}' "
                        f"(source '{source_name}') has dtype '{prop.dtype}'; "
                        f"IPTW/AIPW/DML adjust on a numeric (int/float/bool) or "
                        f"categorical (string) column -- a date carries no "
                        f"adjustment meaning. Derive a numeric or categorical "
                        f"pre-exposure property",
                    )
                )


def _check_unit_in_source_entities(
    exp_name: str,
    unit: str,
    context: str,
    source_name: str,
    source_entities: Mapping[str, list[str]],
    errors: list[tuple[str, str]],
):
    """Check *unit* is in *source_name*'s entities list."""
    entities = source_entities.get(source_name, [])
    if unit not in entities:
        errors.append(
            (
                "definition.check_unit.experiment_but_source",
                f"experiment '{exp_name}' has unit '{unit}' but source "
                f"'{source_name}' ({context}) has entities {entities}",
            )
        )


# ── module-level helpers ──────────────────────────────────────────────


def _visit_facts_for_metric(
    m: Metric,
    exp_name: str,
    unit: str,
    fact_to_source: Mapping[str, str],
    source_entities: Mapping[str, list[str]],
    errors: list[tuple[str, str]],
):
    """Check *unit* is in the entities of every source backing metric *m*."""
    if m.type == "ratio":
        for part_name, part in [("numerator", m.numerator), ("denominator", m.denominator)]:
            if part.fact in fact_to_source:
                src = fact_to_source[part.fact]
                _check_unit_in_source_entities(
                    exp_name,
                    unit,
                    f"metric '{m.name}' ({part_name} fact '{part.fact}')",
                    src,
                    source_entities,
                    errors,
                )
    else:
        if m.fact in fact_to_source:
            src = fact_to_source[m.fact]
            _check_unit_in_source_entities(
                exp_name,
                unit,
                f"metric '{m.name}' (fact '{m.fact}')",
                src,
                source_entities,
                errors,
            )


@dataclass(frozen=True, slots=True)
class _SourceResolutionCodes:
    unknown_source: str
    property_missing: str
    unit_missing: str
    unresolved: str
    ambiguous: str


_BREAKOUT_SOURCE_CODES = _SourceResolutionCodes(
    unknown_source="definition.resolve_breakout.experiment_property_references",
    property_missing="definition.resolve_breakout.experiment_property_found",
    unit_missing="definition.resolve_breakout.experiment_unit_but",
    unresolved="definition.resolve_breakout.experiment_property_could",
    ambiguous="definition.breakout.ambiguous_source",
)
_COVARIATE_SOURCE_CODES = _SourceResolutionCodes(
    unknown_source="definition.check_observational.covariate_source_found",
    property_missing="definition.check_observational.covariate_property_found",
    unit_missing="definition.check_observational.covariate_unit_but",
    unresolved="definition.check_observational.covariate_could",
    ambiguous="definition.observational.ambiguous_covariate_source",
)


def _resolve_breakout_source(
    e: Experiment,
    b: Breakout | Factor,
    source_entities: Mapping[str, list[str]],
    source_properties: Mapping[str, Mapping[str, Property]],
    errors: list[tuple[str, str]],
) -> str | None:
    """Resolve (and validate) the fact source backing breakout/factor *b*.

    Returns the resolved source name, or ``None`` if resolution failed (an
    error has already been appended to *errors*). Raises ``DefinitionError``
    when *b.source* is unset and more than one source qualifies.
    """
    label = "factor" if isinstance(b, Factor) else "breakout"
    return _resolve_property_source(
        e,
        b.property,
        b.source,
        label=label,
        codes=_BREAKOUT_SOURCE_CODES,
        source_entities=source_entities,
        source_properties=source_properties,
        errors=errors,
    )


def _resolve_covariate_source(
    e: Experiment,
    covariate: AdjustmentCovariate,
    source_entities: Mapping[str, list[str]],
    source_properties: Mapping[str, Mapping[str, Property]],
    errors: list[tuple[str, str]],
) -> str | None:
    """Resolve an observational covariate's fact source by the breakout rule."""
    return _resolve_property_source(
        e,
        covariate.property,
        covariate.source,
        label="observational covariate",
        codes=_COVARIATE_SOURCE_CODES,
        source_entities=source_entities,
        source_properties=source_properties,
        errors=errors,
    )


def _resolve_property_source(
    e: Experiment,
    property_name: str,
    requested: str | None,
    *,
    label: str,
    codes: _SourceResolutionCodes,
    source_entities: Mapping[str, list[str]],
    source_properties: Mapping[str, Mapping[str, Property]],
    errors: list[tuple[str, str]],
) -> str | None:
    """An explicit source wins; otherwise exactly one source carrying the
    property and the experiment's unit must qualify."""
    if requested is not None:
        if requested not in source_properties:
            errors.append(
                (
                    codes.unknown_source,
                    f"experiment '{e.name}' {label} property '{property_name}' "
                    f"references unknown source '{requested}'",
                )
            )
            return None
        if property_name not in source_properties[requested]:
            errors.append(
                (
                    codes.property_missing,
                    f"experiment '{e.name}' {label} property '{property_name}' not "
                    f"found on source '{requested}'",
                )
            )
            return None
        if e.unit not in source_entities.get(requested, []):
            errors.append(
                (
                    codes.unit_missing,
                    f"experiment '{e.name}' has unit '{e.unit}' but {label} "
                    f"source '{requested}' has entities {source_entities.get(requested, [])}",
                )
            )
            return None
        return requested

    candidates = [
        src_name
        for src_name, props in source_properties.items()
        if property_name in props and e.unit in source_entities.get(src_name, [])
    ]
    if not candidates:
        errors.append(
            (
                codes.unresolved,
                f"experiment '{e.name}' {label} property '{property_name}' could not "
                f"be resolved to any fact source with unit '{e.unit}' as an entity",
            )
        )
        return None
    if len(candidates) > 1:
        _definition_refusal(
            codes.ambiguous,
            f"experiment '{e.name}' {label} property '{property_name}' matches "
            f"{len(candidates)} fact sources ({', '.join(sorted(candidates))}) with "
            f"unit '{e.unit}' as an entity -- declare 'source: <name>' on the "
            f"{label} to disambiguate",
            experiment=e.name,
            property=property_name,
            candidates=tuple(sorted(candidates)),
        )
    return candidates[0]


def _breakout_metric_sources(m: Metric, fact_to_source: Mapping[str, str]) -> list[str]:
    """Return the fact source name(s) backing metric *m* (both parts for ratio)."""
    if m.type == "ratio":
        return [
            fact_to_source[part.fact]
            for part in (m.numerator, m.denominator)
            if part.fact in fact_to_source
        ]
    if m.fact in fact_to_source:
        return [fact_to_source[m.fact]]
    return []


# Re-exported for one release, then deleted: importers retarget to
# increment.semantics.artifact. Lazy so this module never imports it at load.
if TYPE_CHECKING:
    # Typed view of the lazy re-exports below; keeps legacy imports checkable.
    from increment.semantics.artifact import (  # noqa: F401
        ArtifactContext,
        ArtifactExtensionCatalogEntry,
        ArtifactExtensionRef,
        ArtifactExtensionRequest,
        ArtifactRelationRef,
        AssignmentCountsExtension,
        AssignmentCountsRequest,
        BaseRelations,
        BreakoutDimensionExtension,
        BreakoutDimensionRequest,
        ClusterIdentityExtension,
        ClusterIdentityRequest,
        CupedPreperiodExtension,
        CupedPreperiodRequest,
        EncouragementUptakeExtension,
        EncouragementUptakeRequest,
        ExtensionRefBase,
        ExtensionRequestBase,
        FactorDimensionExtension,
        FactorDimensionRequest,
        Freshness,
        MeasureManifest,
        MetricMeasure,
        RatioMetricMeasure,
        RelationLocator,
        RelationRole,
        SimpleMetricMeasure,
        SiteVolumeExtension,
        SiteVolumeRequest,
        TriggerPopulationExtension,
        TriggerPopulationRequest,
        UnitCovariateExtension,
        UnitCovariateLevelExtension,
        UnitCovariateLevelRequest,
        UnitCovariateRequest,
        UnitDayArtifactManifest,
        UnitDayArtifactRef,
    )

_ARTIFACT_REEXPORTS = frozenset(
    {
        "ArtifactContext",
        "ArtifactExtensionCatalogEntry",
        "ArtifactExtensionRef",
        "ArtifactExtensionRequest",
        "ArtifactRelationRef",
        "AssignmentCountsExtension",
        "AssignmentCountsRequest",
        "BaseRelations",
        "BreakoutDimensionExtension",
        "BreakoutDimensionRequest",
        "ClusterIdentityExtension",
        "ClusterIdentityRequest",
        "CupedPreperiodExtension",
        "CupedPreperiodRequest",
        "EncouragementUptakeExtension",
        "EncouragementUptakeRequest",
        "ExtensionRefBase",
        "ExtensionRequestBase",
        "FactorDimensionExtension",
        "FactorDimensionRequest",
        "Freshness",
        "MeasureManifest",
        "MetricMeasure",
        "RatioMetricMeasure",
        "RelationLocator",
        "RelationRole",
        "SimpleMetricMeasure",
        "SiteVolumeExtension",
        "SiteVolumeRequest",
        "UnitCovariateExtension",
        "UnitCovariateLevelExtension",
        "UnitCovariateLevelRequest",
        "UnitCovariateRequest",
        "TriggerPopulationExtension",
        "TriggerPopulationRequest",
        "UnitDayArtifactManifest",
        "UnitDayArtifactRef",
    }
)

# Keep the pre-split wildcard surface and include the lazy compatibility
# names without importing artifact.py while this module initializes.
__all__ = sorted({name for name in globals() if not name.startswith("_")} | _ARTIFACT_REEXPORTS)


def __getattr__(name: str) -> Any:
    if name in _ARTIFACT_REEXPORTS:
        from increment.semantics import artifact

        return getattr(artifact, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _ARTIFACT_REEXPORTS)
