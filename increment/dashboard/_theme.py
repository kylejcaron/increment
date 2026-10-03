"""Immutable visual presets shared by native sections and the browser shell."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, NoReturn

from increment.errors import InvalidRequestError, RefusalSpec, refuse

if TYPE_CHECKING:
    from coeftable import Theme

_INVALID_THEME = RefusalSpec(
    "dashboard.invalid_theme",
    InvalidRequestError,
    template="invalid dashboard theme {field}: {reason}",
)
_HEX = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})\Z")
_FONT_NAME = r"""(?:[\w-]+(?: +[\w-]+)*|"[\w ,'-]+"|'[\w ,"-]+')"""
_FONT = re.compile(rf"{_FONT_NAME}(?: *, *{_FONT_NAME})*\Z")


def _invalid(field: str, reason: str) -> NoReturn:
    refuse(_INVALID_THEME, field=field, reason=reason)


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _sizes(value: DashboardLayout | DashboardCharts | DashboardPrint) -> None:
    for item in fields(value):
        number = getattr(value, item.name)
        minimum = (
            0 if item.name in {"page_padding", "radius", "cell_padding_x", "cell_padding_y"} else 1
        )
        if not _finite_number(number) or number < minimum:
            _invalid(item.name, f"must be a finite number at least {minimum}")


@dataclass(frozen=True, slots=True)
class DashboardPalette:
    """Semantic colors; arm identity remains separate from evidence direction."""

    text: str = "#1b2a43"
    muted: str = "#586b84"
    accent: str = "#386ba8"
    canvas: str = "#f4f6fa"
    panel: str = "#ffffff"
    rule: str = "#dbe3ef"
    column: str = "#e7edf7"
    band: str = "#f8fafe"
    header: str = "#dbe5f3"
    header_text: str = "#244060"
    control: str = "#8170b1"
    treatment: str = "#3776b9"
    weak: str = "#3c5f95"
    favorable: str = "#087f52"
    favorable_bg: str = "#eafbf2"
    favorable_line: str = "#a7ddc1"
    unfavorable: str = "#be3547"
    unfavorable_bg: str = "#fff0f2"
    inconclusive: str = "#906527"
    inconclusive_bg: str = "#fff7e9"
    on_accent: str = "#ffffff"

    def __post_init__(self) -> None:
        for item in fields(self):
            color = getattr(self, item.name)
            if not isinstance(color, str) or _HEX.fullmatch(color) is None:
                _invalid(item.name, "must be a hexadecimal CSS color")


@dataclass(frozen=True, slots=True)
class DashboardTypography:
    """Font stacks and native table text sizes, in CSS pixels."""

    body_font: str = '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    heading_font: str = 'Georgia, "Times New Roman", serif'
    mono_font: str = "ui-monospace, SFMono-Regular, Menlo, monospace"
    body_size: float = 16
    table_size: float = 14
    value_size: float = 14
    interval_size: float = 12

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name.endswith("_font"):
                if not isinstance(value, str) or _FONT.fullmatch(value.strip()) is None:
                    _invalid(item.name, "must be a CSS-safe font stack")
            elif not _finite_number(value) or value < 1:
                _invalid(item.name, "must be a finite font size at least 1px")


@dataclass(frozen=True, slots=True)
class DashboardLayout:
    """Container widths, whitespace and corner radius, in CSS pixels."""

    dashboard_width: float = 1368
    section_width: float = 1180
    page_padding: float = 36
    radius: float = 6
    cell_padding_x: float = 12
    cell_padding_y: float = 8

    def __post_init__(self) -> None:
        _sizes(self)


@dataclass(frozen=True, slots=True)
class DashboardCharts:
    """Native chart dimensions; these never change a statistical domain."""

    forest_width: int = 300
    forest_height: int = 42
    segment_forest_width: int = 220
    segment_forest_height: int = 48
    time_width: int = 560
    time_height: int = 156
    absolute_width: int = 600
    absolute_height: int = 160
    allocation_width: int = 420
    allocation_height: int = 76

    def __post_init__(self) -> None:
        _sizes(self)
        for item in fields(self):
            if not isinstance(getattr(self, item.name), int):
                _invalid(item.name, "must be an integer pixel dimension")


@dataclass(frozen=True, slots=True)
class DashboardPrint:
    """Readable report text floor in points; larger reports can span pages."""

    min_font_size_pt: float = 8

    def __post_init__(self) -> None:
        _sizes(self)


_DARK = DashboardPalette(
    text="#eef3ff",
    muted="#aabbd5",
    accent="#78b6ff",
    canvas="#0c1322",
    panel="#151f32",
    rule="#2d3c55",
    column="#20304b",
    band="#18243a",
    header="#253b5c",
    header_text="#f3f7ff",
    control="#60a5fa",
    treatment="#c4a3ff",
    weak="#8fa7cf",
    favorable="#5ae0a0",
    favorable_bg="#173b2d",
    favorable_line="#27664a",
    unfavorable="#ff899a",
    unfavorable_bg="#412432",
    inconclusive="#b68b50",
    inconclusive_bg="#30271c",
    on_accent="#0c1322",
)


@dataclass(frozen=True, slots=True)
class DashboardTheme:
    """A named preset with independently adjustable visual categories."""

    name: str = "Midnight"
    light: DashboardPalette = DashboardPalette()
    dark: DashboardPalette = _DARK
    typography: DashboardTypography = DashboardTypography()
    layout: DashboardLayout = DashboardLayout()
    charts: DashboardCharts = DashboardCharts()
    printing: DashboardPrint = DashboardPrint()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            _invalid("name", "must be a non-empty preset name")
        for name, expected in (
            ("light", DashboardPalette),
            ("dark", DashboardPalette),
            ("typography", DashboardTypography),
            ("layout", DashboardLayout),
            ("charts", DashboardCharts),
            ("printing", DashboardPrint),
        ):
            if not isinstance(getattr(self, name), expected):
                _invalid(name, f"must be {expected.__name__}")


MIDNIGHT = DashboardTheme()


def require_theme(theme: object) -> DashboardTheme:
    if not isinstance(theme, DashboardTheme):
        _invalid("theme", "must be DashboardTheme")
    return theme


def theme_from_payload(raw: object) -> DashboardTheme:
    """Reconstruct validated configuration when rendering an exported payload."""
    if raw is None:
        return MIDNIGHT
    if not isinstance(raw, Mapping):
        _invalid("theme", "must be a structured theme payload")
    categories = {
        "light": DashboardPalette,
        "dark": DashboardPalette,
        "typography": DashboardTypography,
        "layout": DashboardLayout,
        "charts": DashboardCharts,
        "printing": DashboardPrint,
    }
    try:
        values = dict(raw)
        for name, category in categories.items():
            if name in values:
                if not isinstance(values[name], Mapping):
                    _invalid(name, "must be a structured visual category")
                values[name] = category(**values[name])
        return DashboardTheme(**values)
    except TypeError as exc:
        _invalid("theme", str(exc))


def _palette_css(palette: DashboardPalette) -> str:
    return (
        ";".join(
            f"--inc-dashboard-theme-{item.name.replace('_', '-')}: {getattr(palette, item.name)}"
            for item in fields(palette)
        )
        + ";"
    )


def theme_css(theme: DashboardTheme, *, selector: str = ":root") -> str:
    """Generate the preset once at the document or standalone-section boundary."""
    theme = require_theme(theme)
    visual = []
    for category in (theme.typography, theme.layout, theme.charts):
        for item in fields(category):
            value = getattr(category, item.name)
            suffix = "" if isinstance(value, str) else "px"
            visual.append(f"--inc-dashboard-{item.name.replace('_', '-')}: {value}{suffix}")
    visual.append(
        f"--inc-dashboard-print-min-font-size: {theme.printing.min_font_size_pt * 4 / 3:g}px"
    )
    dark = (
        ':root[data-theme="dark"]'
        if selector == ":root"
        else f':root[data-theme="dark"] {selector}, {selector}[data-theme="dark"]'
    )
    return (
        f"{selector}{{{_palette_css(theme.light)}{';'.join(visual)};color-scheme:light;}}"
        f"@media not print{{{dark}{{{_palette_css(theme.dark)}color-scheme:dark;}}}}"
    )


def standalone_css(theme: DashboardTheme) -> str:
    """Scope a preset to notebook sections rather than changing the notebook canvas."""
    return theme_css(theme, selector=".inc-dashboard-root")


def coeftable_theme(theme: DashboardTheme) -> Theme:
    """Native colors resolve the active mode, with the preset's light fallback."""
    from coeftable import Theme

    def color(role: str) -> str:
        return f"var(--inc-dashboard-theme-{role.replace('_', '-')}, {getattr(theme.light, role)})"

    return Theme(
        favorable=color("favorable"),
        unfavorable=color("unfavorable"),
        inconclusive=color("weak"),
        neutral=color("accent"),
        header_bg=color("panel"),
        header_fg=color("header_text"),
        column_label_bg=color("header"),
        band=color("band"),
        surface=color("panel"),
        rule=color("rule"),
        border_color=color("rule"),
        axis=color("muted"),
        muted=color("muted"),
        text=color("text"),
        value_size=f"{theme.typography.value_size:g}px",
        ci_size=f"{theme.typography.interval_size:g}px",
        table_font_size=f"{theme.typography.table_size:g}px",
        border_style="minimal",
        na_text="N/A",
        series_palette=(color("control"), color("treatment")),
    )
