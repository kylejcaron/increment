#!/usr/bin/env Rscript
# Oracle generator for increment/estimation/binomial_rr.py's
# clopper_pearson(x, n, beta).
#
# Matched estimand: the exact two-sided single-arm Clopper-Pearson (1 - beta)
# interval for a binomial rate, versus R's stats::binom.test()$conf.int (which
# inverts the beta distribution with R's own qbeta()). Production rounds
# outward by a documented relative slack so the interval is never narrower
# than the exact one; the test therefore asserts enclosure first and
# agreement within that slack (relative to the lower bound, and to one minus
# the upper bound) second. Estimator level, one arm, moderate n;
# two-arm and risk-ratio sets and arms beyond about 2^26 are NOT compared.
#
# Run once; commit the JSON it writes. No R is needed at test time --
# tests/oracles/test_clopper_pearson_oracle.py reads the fixture only.
#
# Usage (from the repository root):
#   Rscript tests/oracles/generate/gen_clopper_pearson_oracle.R

library(jsonlite)

counts <- list(
  c(0, 20), c(1, 20), c(7, 40), c(39, 40), c(40, 40), c(500, 1000)
)
betas <- c(0.05, 0.001)

cases <- list()
for (xn in counts) {
  for (beta in betas) {
    ci <- binom.test(xn[1], xn[2], conf.level = 1 - beta)$conf.int
    cases[[length(cases) + 1]] <- list(
      id = sprintf("x%d_n%d_beta%s", xn[1], xn[2], format(beta, scientific = FALSE)),
      x = xn[1],
      n = xn[2],
      beta = beta,
      lower = ci[1],
      upper = ci[2]
    )
  }
}

commit <- tryCatch(
  trimws(system2("git", c("rev-parse", "HEAD"), stdout = TRUE, stderr = FALSE)),
  error = function(e) NA_character_
)

out <- list(
  generator = "tests/oracles/generate/gen_clopper_pearson_oracle.R",
  package_versions = list(
    r = R.version.string,
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  generated_on = format(Sys.Date(), "%Y-%m-%d"),
  increment_commit = commit,
  estimand = paste(
    "Exact two-sided single-arm Clopper-Pearson (1 - beta) interval",
    "for a binomial rate."
  ),
  tolerance = list(
    slack_rel = 1e-6,
    headroom = 1.01,
    enclosure_abs = 1e-15,
    note = paste(
      "Production rounds outward by a documented relative slack",
      "(_CP_RELATIVE_SLACK = 1e-6) applied to the lower bound and to one",
      "minus the upper bound, so the permitted outward gap is",
      "slack_rel * lower below and slack_rel * (1 - upper) above, times",
      "headroom. Measured on this fixture: every gap equals the slack to",
      "within 5e-6 of itself, so enclosure plus the slack band is the",
      "contract rather than a numerical-agreement tolerance."
    )
  ),
  cases = cases
)

writeLines(
  toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
  "tests/oracles/fixtures/clopper_pearson.json"
)
cat("wrote tests/oracles/fixtures/clopper_pearson.json\n")
