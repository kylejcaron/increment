# A/B testing dashboard

This notebook binds a synthetic `checkout_redesign` experiment to four
dashboard tabs: **Readout**, **Explore**, **Health**, and a printable
**Report**. Explore shows relative-lift or per-arm trajectories across declared
breakouts. Report includes every captured metric result and interval, plus a
full-readout CSV and **Print / Save PDF**. Browser controls and CSV downloads
also work in the static export.

The masthead switches between **Midnight · Daylight** and **Midnight** and
remembers your choice without changing captured evidence. PDFs always use the
light palette. Small reports fit one A4 page; longer reports paginate with
repeated headers and readable text. `DashboardConfig.theme` exposes palette,
typography, layout, chart size, and print settings.

Health shows allocation and applicable checks. Not-applicable trigger checks
appear with their codes and context in collapsed technical details. For an
experiment with a population selector, see the
[triggered dashboard notebook](triggered_population_dashboard.md).

[Open the notebook in a full browser tab](ab_testing_dashboard.html)

<iframe src="../ab_testing_dashboard.html" style="width:100%; height:80vh; border:1px solid var(--md-default-fg-color--lightest); border-radius:4px;"></iframe>

## Next step

Use the reusable renderers with your own experiment; see the
[dashboard guide](../guides/dashboard.md).
