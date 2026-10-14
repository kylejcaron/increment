# A/B testing dashboard

This notebook binds the synthetic `checkout_redesign` experiment to a shared
summary and four tabs: **Readout**, **Explore**, **Health**, and a printable
**Report** with every captured metric result and interval. Explore compares
native CoefTable relative-lift or per-arm absolute trajectories across declared
breakouts. Report provides the full-readout CSV and **Print / Save PDF**;
browser controls and CSV downloads also work in the static export.

The masthead switches between **Midnight · Daylight** and **Midnight** and
remembers your choice without changing captured evidence. PDFs always use the
light palette. Small reports fit one A4 page; larger families paginate with
repeated headers rather than shrinking their textual evidence below the
configured font floor. Presets expose independently adjustable palettes,
typography, layout, native-chart dimensions, and print readability in
`DashboardConfig.theme`.

The Health tab keeps the allocation summary and applicable checks in view;
not-applicable trigger checks are summarized, with codes and context in a
collapsed technical-details section. For an experiment with a population
selector, see the
[triggered dashboard notebook](triggered_population_dashboard.md).

[Open the notebook in a full browser tab](ab_testing_dashboard.html)

<iframe src="../ab_testing_dashboard.html" style="width:100%; height:80vh; border:1px solid var(--md-default-fg-color--lightest); border-radius:4px;"></iframe>

## Next step

Use the reusable renderers with your own experiment; see the
[dashboard guide](../guides/dashboard.md).
