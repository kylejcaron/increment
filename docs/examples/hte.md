# Heterogeneous Effects

This notebook fits a CATE model on simulated data with a known per-unit
effect. On a second cohort with no heterogeneity, the in-sample top group
overstates the truth, while held-out re-estimation recovers it and
`targeting_rule` refuses to target.

[Open the notebook in a full browser tab](hte.html)

<iframe src="../hte.html" style="width:100%; height:80vh; border:1px solid var(--md-default-fg-color--lightest); border-radius:4px;"></iframe>

## References and assumptions

The heterogeneous-treatment-effect discussion follows Chernozhukov et al., [“Generic Machine Learning Inference on Heterogeneous Treatment Effects in Randomized Experiments”](https://www.nber.org/papers/w24678). Segment effects and targeting decisions require separate validation on held-out or otherwise independent data; a credible overall average effect does not validate a subgroup ranking or targeting rule.
