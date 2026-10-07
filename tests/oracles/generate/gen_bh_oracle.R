#!/usr/bin/env Rscript
# Oracle generator for increment/estimation/family.py's bh_select.
#
# Matched estimand: the Benjamini-Hochberg step-up rejection set at FDR level
# q over a fixed family of p-values (ties selected together), versus R's
# stats::p.adjust(method = "BH"): index i is rejected iff its BH-adjusted
# p-value is <= q. Estimator level only -- family construction, dependence
# conditions, e-BH and sequential selection are NOT compared.
#
# The vectors are literals (no RNG) and avoid p-values within floating point
# of a k * q / m boundary, where exact rational thresholds and floating-point
# adjusted values may legitimately differ in the last bit. The `rejected`
# field is a sorted list of 1-based indices (R convention); the test
# converts to 0-based.
#
# Run once; commit the JSON it writes. No R is needed at test time --
# tests/oracles/test_bh_oracle.py reads the fixture only.
#
# Usage (from the repository root):
#   Rscript tests/oracles/generate/gen_bh_oracle.R

library(jsonlite)

families <- list(
  list(
    id = "mixed_unsorted",
    p = c(0.212, 0.001, 0.074, 0.039, 0.008, 0.205, 0.042, 0.06, 0.0004, 0.041)
  ),
  list(id = "all_null", p = c(0.2, 0.4, 0.6, 0.8, 0.95)),
  list(id = "all_selected", p = c(1e-6, 3e-7, 2e-5, 4e-8)),
  list(id = "tied_values", p = c(0.01, 0.5, 0.01, 0.01, 0.19)),
  list(id = "tied_at_cutoff", p = c(0.5, 0.012, 0.012, 0.9, 0.012, 0.3, 0.012)),
  list(id = "single", p = c(0.01)),
  list(
    id = "step_up_rescues_larger_p",
    p = c(0.011, 0.02, 0.0195, 0.0199, 0.2, 0.7, 0.0199, 0.012)
  )
)
levels <- c(0.05, 0.1, 0.25)

cases <- list()
for (fam in families) {
  adj <- p.adjust(fam$p, method = "BH")
  for (q in levels) {
    cases[[length(cases) + 1]] <- list(
      id = sprintf("%s_q%s", fam$id, format(q, scientific = FALSE)),
      p = I(fam$p),
      q = q,
      rejected = I(as.integer(which(adj <= q)))
    )
  }
}

commit <- tryCatch(
  trimws(system2("git", c("rev-parse", "HEAD"), stdout = TRUE, stderr = FALSE)),
  error = function(e) NA_character_
)

out <- list(
  generator = "tests/oracles/generate/gen_bh_oracle.R",
  package_versions = list(
    r = R.version.string,
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  generated_on = format(Sys.Date(), "%Y-%m-%d"),
  increment_commit = commit,
  estimand = paste(
    "Benjamini-Hochberg step-up rejection set at FDR level q over a fixed",
    "p-value family, ties selected together."
  ),
  tolerance = list(
    exact = TRUE,
    note = paste(
      "The compared quantity is a set of indices, so agreement is exact;",
      "there is no numeric tolerance. Vectors avoid p-values within",
      "floating point of a k * q / m boundary."
    )
  ),
  cases = cases
)

writeLines(
  toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
  "tests/oracles/fixtures/bh_p_adjust.json"
)
cat("wrote tests/oracles/fixtures/bh_p_adjust.json\n")
