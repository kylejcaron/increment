# Triggered population dashboard

This notebook builds a deterministic two-arm experiment with assignment events,
trigger events, and outcomes. The dashboard defaults to the triggered cohort
and switches all readouts together between **Assigned** and **Triggered**.

!!! warning "Triggered results need a treatment-independent trigger"
    Triggered-cohort inference is appropriate only when treatment does not
    affect triggering.

Definitions and unit-day artifacts with trigger evidence also support
triggered daily and as-of encouragement compliance. Cohort membership starts
at each trigger, but uptake windows remain anchored at assignment. If a source
cannot provide triggered membership, the notebook shows that refusal instead of
substituting assigned users.

Assigned-population allocation checks assess assignment integrity; triggered
checks describe balance among triggered users. They are separate statuses.
Registered sequential checkpoints apply to assigned users only and are not
used as evidence for the triggered group.

[Open the notebook in a full browser tab](triggered_population_dashboard.html)

<iframe src="../triggered_population_dashboard.html" style="width:100%; height:80vh; border:1px solid var(--md-default-fg-color--lightest); border-radius:4px;"></iframe>

## Next step

See the [dashboard guide](../guides/dashboard.md) to connect a dashboard to your experiment.
