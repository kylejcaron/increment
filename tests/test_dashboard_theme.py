"""CSS-safe visual configuration and immutable caller-owned themes."""

import copy
import dataclasses
import pickle

import pytest

from increment.errors import InvalidRequestError

pytest.importorskip("coeftable")
pytest.importorskip("marimo")


@pytest.mark.parametrize(
    "category,values",
    [
        ("DashboardPalette", {"accent": "#fff; background:url(https://example.com)"}),
        ("DashboardPalette", {"text": "</style><script>evil()</script>"}),
        ("DashboardTypography", {"heading_font": "Georgia; } body { display:none"}),
        ("DashboardTypography", {"heading_font": '"Arial'}),
        ("DashboardTypography", {"body_font": "Arial,, sans-serif"}),
        ("DashboardTypography", {"body_size": float("nan")}),
        ("DashboardTypography", {"table_size": 10**400}),
        ("DashboardLayout", {"page_padding": -1}),
        ("DashboardLayout", {"dashboard_width": 10**400}),
        ("DashboardCharts", {"time_width": 0}),
        ("DashboardCharts", {"forest_height": True}),
        ("DashboardCharts", {"forest_width": 10**400}),
        ("DashboardPrint", {"min_font_size_pt": float("inf")}),
        ("DashboardTheme", {"light": {"accent": "#123456"}}),
    ],
)
def test_unusable_visual_configuration_has_portable_structured_refusal(category, values):
    import increment.dashboard as dashboard

    with pytest.raises(InvalidRequestError) as caught:
        getattr(dashboard, category)(**values)
    original = caught.value
    assert original.code == "dashboard.invalid_theme"
    for restored in (copy.deepcopy(original), pickle.loads(pickle.dumps(original))):
        assert restored.code == original.code
        assert restored.context == original.context
        with pytest.raises(TypeError):
            restored.context["field"] = "changed"  # ty: ignore[invalid-assignment]


def test_custom_preset_is_immutable_without_modifying_its_base():
    from increment.dashboard import MIDNIGHT, DashboardConfig

    original = MIDNIGHT.light.accent
    custom = dataclasses.replace(
        MIDNIGHT,
        name="Editorial",
        light=dataclasses.replace(MIDNIGHT.light, accent="#9b4d18"),
        typography=dataclasses.replace(MIDNIGHT.typography, heading_font="Arial, sans-serif"),
    )
    config = DashboardConfig(expected_allocation={"baseline": 7, "candidate": 3}, theme=custom)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.theme.light.accent = "#000000"  # ty: ignore[invalid-assignment]
    assert MIDNIGHT.light.accent == original
    assert copy.deepcopy(config.theme) == custom
    assert pickle.loads(pickle.dumps(config.theme)) == custom


def test_dashboard_rejects_unstructured_theme_before_source_access():
    from increment.dashboard import DashboardConfig

    with pytest.raises(InvalidRequestError) as caught:
        DashboardConfig(expected_allocation={"baseline": 7, "candidate": 3}, theme={})  # ty: ignore[invalid-argument-type]
    assert caught.value.code == "dashboard.invalid_theme"


def test_standalone_styles_refuse_unstructured_theme():
    from increment.dashboard import dashboard_styles

    with pytest.raises(InvalidRequestError) as caught:
        dashboard_styles(theme={})  # ty: ignore[invalid-argument-type]
    assert caught.value.code == "dashboard.invalid_theme"
