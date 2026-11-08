# Encouragement design

Analyze a randomized encouragement experiment. The notebook reports assignment,
uptake, intent-to-treat effect (ITT), compliance, and local average treatment
effect (LATE).

Its last section monitors LATE over time for a cohort whose exposure is
staggered over three weeks: each date's as-of LATE counts only users whose
outcome and uptake windows have closed, so the interval narrows as cohorts
mature. The per-day view is refused under an encouragement design, because a
single day's clicks are too small a first stage; the notebook shows that
refusal and monitors relative uptake with always-valid inference instead.

[Open the notebook in a full browser tab](encouragement.html)

<iframe src="../encouragement.html" style="width:100%; height:80vh; border:1px solid var(--md-default-fg-color--lightest); border-radius:4px;"></iframe>
