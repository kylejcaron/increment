# Heterogeneous effects

Fit a conditional average treatment effect (CATE) model on simulated data with
known effects. In a second cohort with no true heterogeneity, the in-sample top
group overstates the effect; held-out re-estimation corrects it, and
`targeting_rule` refuses to target.

[Open the notebook in a full browser tab](hte.html)

<iframe src="../hte.html" style="width:100%; height:80vh; border:1px solid var(--md-default-fg-color--lightest); border-radius:4px;"></iframe>

## References and assumptions

The discussion follows Chernozhukov et al.,
[“Generic Machine Learning Inference on Heterogeneous Treatment Effects in Randomized Experiments”](https://www.nber.org/papers/w24678).
Validate segment effects and targeting decisions on held-out or independent
data. A credible overall average effect does not validate subgroup rankings or
a targeting rule.
