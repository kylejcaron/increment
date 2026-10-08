"""Canonical column sets for every pipeline stage.

Every builder declares its output columns here.  Downstream tasks
**must** use these names — they are the contract.
"""

from types import MappingProxyType

from increment._moment_plan import DAY_GRAIN, UNIT_GRAIN

EXPOSURES = frozenset(
    {
        "unit_id",
        "experiment_id",
        "group_id",
        "first_exposure_ts",
    }
)

METRIC_EVENTS = frozenset(
    {
        "unit_id",
        "ts",
        "metric",
        "value",
    }
)

SITE_VOLUME = frozenset(
    {
        "metric",
        "y",
        "y_den",
    }
)

UNIT_DAY_PANEL = frozenset(
    {
        "unit_id",
        "ds",
        "experiment_id",
        "group_id",
        "metric",
        # Mergeable per-unit-day sufficient state -- the same shape
        # `unit_day_stats` produces -- so a caller can recombine it under
        # any declared aggregation (avg_event/min/max/count_distinct, not
        # just sum), never a single pre-summed `value` column.
        "n_events",
        "sum_value",
        "min_value",
        "max_value",
        # Internal plumbing: unit_totals uses these for late-enrollee
        # censoring, not part of the public 6-column model.
        "first_exposure_ts",
        "first_exposure_date",
    }
)

UNIT_TOTALS = frozenset(
    {
        "unit_id",
        "experiment_id",
        "group_id",
        "metric",
        "y",
        "x",
        "y_den",
        "d",
    }
)

# Keys, `n`, the unit-grain plan's sixteen slots (what each carries, `cxden`
# included, is documented on increment._moment_plan.SLOTS), x_role, and the plan's
# passthrough columns (winsor metadata, exact binary `successes`).
GROUP_SUMMARY = frozenset(
    {
        "experiment_id",
        "metric",
        "group_id",
        *UNIT_GRAIN.names.values(),
        "x_role",
        *(passthrough.column for passthrough in UNIT_GRAIN.passthrough),
    }
)

DAILY_GROUP_SUMMARY = frozenset(
    {
        "ds",
        "experiment_id",
        "metric",
        "group_id",
        *DAY_GRAIN.names.values(),
        "x_role",
        *(passthrough.column for passthrough in DAY_GRAIN.passthrough),
    }
)

DAILY_EXPOSURE_COUNTS = frozenset(
    {
        "experiment_id",
        "ds",
        "group_id",
        "n_daily",
        "n_cumulative",
    }
)

CLUSTER_EXPOSURE_COUNTS = frozenset(
    {
        "experiment_id",
        "group_id",
        "n_clusters",
        "n_units",
    }
)
# Frozen format-1 unit-day artifact relation declarations.  These are
# logical wire schemas: producers and readers must use the listed field order,
# physical type tags, and primary keys when computing canonical digests.

UNIT_DAY_ARTIFACT_RELATION_ROLES = (
    "exposures",
    "measure_stats",
    "breakout_dimension",
    "factor_dimension",
    "cluster_identity",
    "cuped_preperiod",
    "assignment_counts",
    "trigger_population",
    "trigger_measure_stats",
    "encouragement_uptake",
    "site_volume",
    "unit_covariate",
    "unit_covariate_level",
)

# The two mandatory relations are deliberately separate from optional
# extension roles.  A manifest without either base relation is invalid.
UNIT_DAY_ARTIFACT_BASE_RELATION_ROLES = ("exposures", "measure_stats")

UNIT_DAY_ARTIFACT_EXPOSURES_FIELDS = (
    "experiment_id",
    "unit_id",
    "group_id",
    "first_exposure_ts",
    "first_exposure_date",
)
UNIT_DAY_ARTIFACT_MEASURE_STATS_FIELDS = (
    "experiment_id",
    "unit_id",
    "ds",
    "measure_key",
    "n_events",
    "sum_value",
    "min_value",
    "max_value",
)
UNIT_DAY_ARTIFACT_BREAKOUT_DIMENSION_FIELDS = (
    "experiment_id",
    "unit_id",
    "value_is_missing",
    "dimension_value",
)
UNIT_DAY_ARTIFACT_FACTOR_DIMENSION_FIELDS = (
    "experiment_id",
    "unit_id",
    "value_is_missing",
    "factor_value",
)
UNIT_DAY_ARTIFACT_CLUSTER_IDENTITY_FIELDS = ("experiment_id", "unit_id", "cluster_id")
UNIT_DAY_ARTIFACT_CUPED_PREPERIOD_FIELDS = ("experiment_id", "unit_id", "x")
UNIT_DAY_ARTIFACT_ASSIGNMENT_COUNTS_FIELDS = (
    "experiment_id",
    "population",
    "group_id",
    "n_units",
    "n_randomization_units",
)
UNIT_DAY_ARTIFACT_TRIGGER_POPULATION_FIELDS = (
    "experiment_id",
    "unit_id",
    "first_trigger_ts",
)
UNIT_DAY_ARTIFACT_TRIGGER_MEASURE_STATS_FIELDS = (
    "experiment_id",
    "unit_id",
    "ds",
    "measure_key",
    "n_events",
    "sum_value",
    "min_value",
    "max_value",
)
UNIT_DAY_ARTIFACT_ENCOURAGEMENT_UPTAKE_FIELDS = (
    "experiment_id",
    "unit_id",
    "uptake",
    "first_uptake_ts",
)
UNIT_DAY_ARTIFACT_SITE_VOLUME_FIELDS = (
    "experiment_id",
    "ds",
    "measure_key",
    "n_events",
    "sum_value",
    "min_value",
    "max_value",
)

# Physical tags are the six canonical format-1 tags consumed by the digest
# codec. Nullability is explicit; first_uptake_ts is null exactly when
# uptake is false. Physical schemas must preserve this distinction.
UNIT_DAY_ARTIFACT_RELATION_SCHEMAS = MappingProxyType(
    {
        "exposures": tuple(
            (field, type_tag, False)
            for field, type_tag in (
                ("experiment_id", "STRING"),
                ("unit_id", "STRING"),
                ("group_id", "STRING"),
                ("first_exposure_ts", "TIMESTAMP_UTC_US"),
                ("first_exposure_date", "DATE"),
            )
        ),
        "measure_stats": tuple(
            (field, type_tag, False)
            for field, type_tag in (
                ("experiment_id", "STRING"),
                ("unit_id", "STRING"),
                ("ds", "DATE"),
                ("measure_key", "STRING"),
                ("n_events", "INT64"),
                ("sum_value", "FLOAT64"),
                ("min_value", "FLOAT64"),
                ("max_value", "FLOAT64"),
            )
        ),
        "breakout_dimension": tuple(
            (field, type_tag, False)
            for field, type_tag in (
                ("experiment_id", "STRING"),
                ("unit_id", "STRING"),
                ("value_is_missing", "BOOLEAN"),
                ("dimension_value", "STRING"),
            )
        ),
        "factor_dimension": tuple(
            (field, type_tag, False)
            for field, type_tag in (
                ("experiment_id", "STRING"),
                ("unit_id", "STRING"),
                ("value_is_missing", "BOOLEAN"),
                ("factor_value", "STRING"),
            )
        ),
        "cluster_identity": tuple(
            (field, type_tag, False)
            for field, type_tag in (
                ("experiment_id", "STRING"),
                ("unit_id", "STRING"),
                ("cluster_id", "STRING"),
            )
        ),
        "cuped_preperiod": tuple(
            (field, type_tag, False)
            for field, type_tag in (
                ("experiment_id", "STRING"),
                ("unit_id", "STRING"),
                ("x", "FLOAT64"),
            )
        ),
        "assignment_counts": tuple(
            (field, type_tag, False)
            for field, type_tag in (
                ("experiment_id", "STRING"),
                ("population", "STRING"),
                ("group_id", "STRING"),
                ("n_units", "INT64"),
                ("n_randomization_units", "INT64"),
            )
        ),
        "trigger_population": tuple(
            (field, type_tag, False)
            for field, type_tag in (
                ("experiment_id", "STRING"),
                ("unit_id", "STRING"),
                ("first_trigger_ts", "TIMESTAMP_UTC_US"),
            )
        ),
        "trigger_measure_stats": tuple(
            (field, type_tag, False)
            for field, type_tag in (
                ("experiment_id", "STRING"),
                ("unit_id", "STRING"),
                ("ds", "DATE"),
                ("measure_key", "STRING"),
                ("n_events", "INT64"),
                ("sum_value", "FLOAT64"),
                ("min_value", "FLOAT64"),
                ("max_value", "FLOAT64"),
            )
        ),
        "encouragement_uptake": tuple(
            (field, type_tag, False)
            for field, type_tag in (
                ("experiment_id", "STRING"),
                ("unit_id", "STRING"),
                ("uptake", "BOOLEAN"),
            )
        )
        + (("first_uptake_ts", "TIMESTAMP_UTC_US", True),),
        "site_volume": tuple(
            (field, type_tag, False)
            for field, type_tag in (
                ("experiment_id", "STRING"),
                ("ds", "DATE"),
                ("measure_key", "STRING"),
                ("n_events", "INT64"),
                ("sum_value", "FLOAT64"),
                ("min_value", "FLOAT64"),
                ("max_value", "FLOAT64"),
            )
        ),
        # A covariate can be genuinely missing for a unit, so its value is
        # nullable. A numeric covariate travels as `value`; a categorical
        # one as its own relation role carrying the level label, never a
        # sentinel string -- a NULL level is a missing value, not a level.
        "unit_covariate": (
            ("experiment_id", "STRING", False),
            ("unit_id", "STRING", False),
            ("value", "FLOAT64", True),
        ),
        "unit_covariate_level": (
            ("experiment_id", "STRING", False),
            ("unit_id", "STRING", False),
            ("level", "STRING", True),
        ),
    }
)

UNIT_DAY_ARTIFACT_PRIMARY_KEYS = MappingProxyType(
    {
        "exposures": ("experiment_id", "unit_id"),
        "measure_stats": ("experiment_id", "unit_id", "ds", "measure_key"),
        "breakout_dimension": ("experiment_id", "unit_id"),
        "factor_dimension": ("experiment_id", "unit_id"),
        "cluster_identity": ("experiment_id", "unit_id"),
        "cuped_preperiod": ("experiment_id", "unit_id"),
        "assignment_counts": ("experiment_id", "population", "group_id"),
        "trigger_population": ("experiment_id", "unit_id"),
        "trigger_measure_stats": ("experiment_id", "unit_id", "ds", "measure_key"),
        "encouragement_uptake": ("experiment_id", "unit_id"),
        "site_volume": ("experiment_id", "ds", "measure_key"),
        "unit_covariate": ("experiment_id", "unit_id"),
        "unit_covariate_level": ("experiment_id", "unit_id"),
    }
)

# Short aliases match the existing query schema vocabulary while the
# UNIT_DAY_ARTIFACT_* names remain the canonical, unambiguous declarations.
ARTIFACT_RELATION_ROLES = UNIT_DAY_ARTIFACT_RELATION_ROLES
ARTIFACT_BASE_RELATION_ROLES = UNIT_DAY_ARTIFACT_BASE_RELATION_ROLES
ARTIFACT_RELATION_SCHEMAS = UNIT_DAY_ARTIFACT_RELATION_SCHEMAS
ARTIFACT_RELATION_PRIMARY_KEYS = UNIT_DAY_ARTIFACT_PRIMARY_KEYS

# Version-1 uptake had no event time; retained only for legacy digest fixtures.
LEGACY_ENCOURAGEMENT_UPTAKE_SCHEMA = (
    ("experiment_id", "STRING", False),
    ("unit_id", "STRING", False),
    ("uptake", "BOOLEAN", False),
)
