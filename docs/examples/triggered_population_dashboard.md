# Triggered population dashboard

This notebook builds a deterministic two-arm experiment with assignment events,
trigger events, and outcomes. Its dashboard defaults to the triggered cohort and
switches every readout together between **Assigned** and **Triggered**. The
triggered-only caveat is always visible: triggered-cohort inference is
appropriate only when triggering is unaffected by treatment. When the source
refuses a triggered daily/as-of view, its refusal remains visible and the
assignment-level view is a separate route.

Assigned-population allocation checks describe assignment integrity; triggered
checks describe balance in the triggered cohort. They are separate statuses,
not substitutes for each other. Registered sequential checkpoints stay
assignment-scoped and are not borrowed for triggered group evidence.

[Open the notebook in a full browser tab](triggered_population_dashboard.html)

<iframe src="../triggered_population_dashboard.html" style="width:100%; height:80vh; border:1px solid var(--md-default-fg-color--lightest); border-radius:4px;"></iframe>

## Next step

See the [dashboard guide](../guides/dashboard.md) for binding a dashboard to your own experiment.
