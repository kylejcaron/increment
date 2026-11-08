# A/B testing dashboard

This notebook binds an experiment from the synthetic realistic-demo warehouse
to a shared summary and four tabs: **Readout**, **Explore**, **Health**, and a
printable **Report** with every captured metric result and interval. The
**Experiment** dropdown picks which declared experiment to show; the same
`prepare_dashboard` and `render_dashboard` calls serve both, and the dashboard
adapts to what each one declares:

- `checkout_redesign` analyzes users from the moment they reach the redesigned
  checkout, with declared breakouts by country and plan.
- `express_checkout` assigns users on their first visit and declares a
  trigger: reaching checkout, where the one-tap button appears. Its dashboard
  opens on **Triggered** and its **Population** control switches every
  readout, allocation check and Explore view between **Assigned** and
  **Triggered** together; populations are never pooled.

Both experiments declare their allocation, so the readout's assignment check
and the sample-ratio check run for the assigned population.

!!! warning "Triggered results need a treatment-independent trigger"
    Triggered-cohort inference is appropriate only when treatment does not
    affect triggering. Assigned allocation checks assess assignment
    integrity; triggered checks describe balance among triggered users and
    are a separate status.

A triggered population is read up to an explicit event-time cutoff, so the
notebook passes the generated snapshot's cutoff to `from_definitions` when the
selected experiment declares a trigger.

Explore compares native CoefTable relative-lift or per-arm absolute
trajectories across declared breakouts. Report provides the full-readout CSV
and **Print / Save PDF**; browser controls and CSV downloads also work in the
static export, which shows the experiment selected when it was exported.

The masthead switches between **Midnight · Daylight** and **Midnight** and
remembers your choice without changing captured evidence. PDFs always use the
light palette. Small reports fit one A4 page; larger families paginate with
repeated headers rather than shrinking their textual evidence below the
configured font floor. Presets expose independently adjustable palettes,
typography, layout, native-chart dimensions, and print readability in
`DashboardConfig.theme`.

The Health tab keeps the allocation summary and applicable checks in view;
not-applicable trigger checks are summarized, with codes and context in a
collapsed technical-details section.

[Open the notebook in a full browser tab](dashboard.html)

<iframe src="../dashboard.html" style="width:100%; height:80vh; border:1px solid var(--md-default-fg-color--lightest); border-radius:4px;"></iframe>

## Next step

Use the reusable renderers with your own experiment; see the
[dashboard guide](../guides/dashboard.md).
