# A/B testing dashboard

This notebook composes reusable `increment.dashboard` helpers into one
readable report: allocation health, the full role-grouped CoefTable readout,
metric time trends and segments, definitions and provenance, and a
downloadable CSV for the bundled synthetic `checkout_redesign` experiment.

This static export includes Overview, Health, Results, cumulative-lift Explore
charts, Metric details, and Details. Open the notebook live to change the
metric-detail selector, Explore view, or maturity controls.


[Open the notebook in a full browser tab](ab_testing_dashboard.html)

<iframe src="../ab_testing_dashboard.html" style="width:100%; height:80vh; border:1px solid var(--md-default-fg-color--lightest); border-radius:4px;"></iframe>

## Next step

Bind your own experiment to the same reusable renderers in the
[dashboard guide](../guides/dashboard.md).
