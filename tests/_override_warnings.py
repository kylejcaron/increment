"""Test helpers for intentional call-time overrides of declared analysis methods."""

from __future__ import annotations

from tests.warning_codes import allow_warning_codes

#: Allow only declared-override warnings during a comparison test.
allow_declared_overrides = allow_warning_codes("analysis.run.override_replaces_declared")
