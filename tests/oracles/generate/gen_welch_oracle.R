#!/usr/bin/env Rscript
# Oracle generator for the default fixed-horizon mean-difference interval
# (Analysis.from_unit_summary(...).run(), absolute-axis fields).
#
# Matched estimand: the unweighted two-sample difference of arm means
# (treatment - control) with the Welch/Satterthwaite variance estimator,
# Welch degrees of freedom and a two-sided central t interval, versus R's
# stats::t.test(var.equal = FALSE). Cluster-robust, weighted,
# covariate-adjusted, ratio and sequential contrasts are NOT compared.
#
# The two sample vectors are literals below (no RNG), so both languages see
# bit-identical inputs. Run once; commit the JSON it writes. No R is needed at
# test time -- tests/oracles/test_welch_oracle.py reads the fixture only.
#
# Usage (from the repository root):
#   Rscript tests/oracles/generate/gen_welch_oracle.R

library(jsonlite)

alpha <- 0.05

cases <- list(
  list(
    id = "unequal_n_unequal_variance",
    control = c(
      9.41, 10.87, 8.12, 11.35, 9.96, 10.2, 12.04, 7.88, 10.51, 9.07,
      11.62, 8.74, 10.09, 9.33
    ),
    treatment = c(
      12.5, 14.8, 9.1, 17.3, 11.7, 13.4, 19.6, 10.2, 15.9, 12.8,
      8.4, 16.1, 13.0, 18.2, 11.1, 14.4, 9.8, 15.3, 12.2, 20.5,
      10.9, 13.7
    )
  ),
  list(
    id = "near_equal_variance_small_n",
    control = c(5.2, 6.1, 4.8, 5.9, 5.5),
    treatment = c(6.0, 6.9, 5.4, 6.6, 6.3, 5.8)
  )
)

result_for_case <- function(case) {
  tt <- t.test(case$treatment, case$control, var.equal = FALSE, conf.level = 1 - alpha)
  list(
    id = case$id,
    input = list(control = case$control, treatment = case$treatment, alpha = alpha),
    diff = unname(tt$estimate[1] - tt$estimate[2]),
    se = unname(tt$stderr),
    df = unname(tt$parameter),
    lower = unname(tt$conf.int[1]),
    upper = unname(tt$conf.int[2])
  )
}

commit <- tryCatch(
  trimws(system2("git", c("rev-parse", "HEAD"), stdout = TRUE, stderr = FALSE)),
  error = function(e) NA_character_
)

out <- list(
  generator = "tests/oracles/generate/gen_welch_oracle.R",
  package_versions = list(
    r = R.version.string,
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  generated_on = format(Sys.Date(), "%Y-%m-%d"),
  increment_commit = commit,
  estimand = paste(
    "Treatment-minus-control difference of arm means, Welch variance,",
    "Welch-Satterthwaite df, two-sided 95% t interval."
  ),
  tolerance = list(
    rel = 1e-12,
    note = paste(
      "Relative tolerance on diff, se, df, lower and upper. Measured after",
      "the first generation run: the largest observed relative difference",
      "was 6.7e-15 (the lower endpoint of near_equal_variance_small_n,",
      "where the interval nearly straddles zero), so 1e-12 leaves about two",
      "orders of magnitude of floating-point headroom, not a methodological",
      "band."
    )
  ),
  cases = lapply(cases, result_for_case)
)

writeLines(
  toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
  "tests/oracles/fixtures/welch_t.json"
)
cat("wrote tests/oracles/fixtures/welch_t.json\n")
