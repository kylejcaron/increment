#!/usr/bin/env Rscript
# Oracle generator for increment/estimation/meta.py's DerSimonian-Laird
# tau^2, Hartung-Knapp-Sidik-Jonkman pooled mean, and the Higgins-Thompson
# (2002) test-based I^2 confidence interval.
#
# Run once; commit the JSON it writes. No R is needed at test time --
# tests/oracles/test_meta_oracle.py reads the committed fixture only.
#
# After running, replace the two version strings below with the actual
# installed versions (`R.version.string`, `packageVersion("metafor")`);
# the script also writes them into the fixture for anyone re-verifying.
#
# Usage (from the repository root):
#   Rscript tests/oracles/generate/gen_meta_oracle.R

library(metafor)
library(jsonlite)

alpha <- 0.05

# Two cases chosen to exercise both branches of the Higgins-Thompson
# ln(H) standard error (Q > k and Q <= k; see meta.py's
# _higgins_thompson_i2_ci docstring) and both a small (k=5, the
# threshold at which meta.py starts reporting a point I^2) and a larger
# k.
cases <- list(
  list(
    id = "k5_moderate_heterogeneity",
    est = c(0.12, 0.28, -0.05, 0.31, 0.09),
    var = c(0.020, 0.015, 0.025, 0.012, 0.018)
  ),
  list(
    id = "k8_high_heterogeneity",
    est = c(0.05, 0.45, -0.20, 0.38, 0.02, 0.50, -0.15, 0.33),
    var = c(0.030, 0.020, 0.025, 0.018, 0.028, 0.022, 0.030, 0.016)
  )
)

result_for_case <- function(case) {
  fit_dl <- rma(yi = case$est, vi = case$var, method = "DL")
  fit_knha <- rma(yi = case$est, vi = case$var, method = "DL", test = "knha")
  # type="HT": Higgins & Thompson (2002) method III test-based CI for
  # tau^2/I^2/H^2 -- the exact method meta.py's _higgins_thompson_i2_ci
  # implements by hand. metafor's default (Q-profile) CI is a DIFFERENT,
  # exact-coverage method and is deliberately not used here: comparing
  # against it would not check the HT formula this repo implements.
  ht <- confint(fit_dl, type = "HT")

  list(
    id = case$id,
    input = list(est = case$est, var = case$var, alpha = alpha),
    dl_tau2 = unname(fit_dl$tau2),
    hksj = list(
      mu_hat = unname(fit_knha$b[1, 1]),
      se = unname(fit_knha$se),
      lower = unname(fit_knha$ci.lb),
      upper = unname(fit_knha$ci.ub),
      dof = unname(fit_knha$k - 1)
    ),
    i2_ci = list(
      q = unname(fit_dl$QE),
      i2_lb_pct = unname(ht$random["I^2(%)", "ci.lb"]),
      i2_ub_pct = unname(ht$random["I^2(%)", "ci.ub"])
    )
  )
}

out <- list(
  generator = "tests/oracles/generate/gen_meta_oracle.R",
  package_versions = list(
    r = R.version.string,
    metafor = as.character(packageVersion("metafor"))
  ),
  tolerance = list(
    dl_tau2_rel = 1e-8,
    hksj_abs = 1e-6,
    i2_ci_abs = 1e-6,
    note = paste(
      "DL tau^2 and HKSJ's t-based interval are closed-form (no root",
      "finding), so both sides round the identical formula to double",
      "precision; 1e-8 relative / 1e-6 absolute leaves headroom for",
      "R-vs-numpy summation-order differences without hiding a real",
      "formula mismatch. The HT I^2 CI additionally exponentiates a",
      "normal CI on ln(H), so its tolerance stays absolute",
      "(percentage points, /100 to compare against meta.py's 0-1 fraction)."
    )
  ),
  cases = lapply(cases, result_for_case)
)

writeLines(
  toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
  "tests/oracles/fixtures/meta_dl_hksj.json"
)
cat("wrote tests/oracles/fixtures/meta_dl_hksj.json\n")
