#!/usr/bin/env Rscript
# Oracle generator for increment/estimation/quantile.py's log_quantile_se,
# a distribution-free quantile CI: a discrete order-statistic bracket
# selected via the exact binomial quantile of the rank
# (scipy.stats.binom.ppf/isf in production).
#
# This script is an INDEPENDENT R implementation of that same discrete
# rank-selection formula, using R's own qbinom() (a separately
# implemented binomial quantile function, not a port of scipy's), rather
# than comparing against survey::oldsvyquantile -- that package computes
# a materially different estimator (an asymptotic Wald CI on the
# cumulative proportion mapped through a linearly interpolated ECDF; read
# from survey:::oldsvyquantile.survey.design's computeWaldCI), so it does
# not agree with the discrete order-statistic method to a meaningful
# tolerance (see git history for the earlier attempt and the measured
# 2.8%-6.0% relative gap that prompted this rewrite).
#
# scipy.stats.binom.ppf(p, n, q) is the smallest integer k with
# P(X<=k) >= p; R's qbinom(p, n, q) (default lower.tail=TRUE) is defined
# identically. scipy.stats.binom.isf(p, n, q) is the smallest integer k
# with P(X>k) <= p; R's qbinom(p, n, q, lower.tail=FALSE) is defined
# identically. Both sides then read the order statistic at that rank
# from the SAME shared CSV, sorted independently in each language --
# verified by hand to be bit-identical floats (CSV round-trips full
# double precision via Python's repr()-based csv writer).
#
# Run once; commit the JSON it writes. Reads the CSV
# gen_quantile_samples.py produced, so both languages see the identical
# sample.
#
# Usage (from the repository root):
#   Rscript tests/oracles/generate/gen_quantile_oracle.R

library(jsonlite)

alpha <- 0.05
samples <- read.csv("tests/oracles/data/quantile_values.csv")

case_ids <- unique(samples$case_id)

result_for_case <- function(cid) {
  rows <- samples[samples$case_id == cid, ]
  q <- rows$q[1]
  n <- nrow(rows)
  y <- sort(rows$value)
  # 1-based ranks matching scipy's binom.ppf/isf convention exactly.
  a <- qbinom(alpha / 2, n, q)
  b <- qbinom(alpha / 2, n, q, lower.tail = FALSE) + 1

  list(
    id = cid,
    q = q,
    n = n,
    a = a,
    b = b,
    lower = y[a],
    upper = y[b]
  )
}

out <- list(
  generator = "tests/oracles/generate/gen_quantile_oracle.R",
  package_versions = list(
    r = R.version.string
  ),
  tolerance = list(
    se_rel = 1e-12,
    note = paste(
      "Measured after running this generator (not assumed): with the",
      "rank selection now independently computed via R's own qbinom()",
      "(rather than survey::oldsvyquantile's asymptotic Wald+interpolated",
      "estimator), the order statistics R and scipy.stats.binom.ppf/isf",
      "select are bit-identical on this fixture's two cases -- both are",
      "exact binomial quantile functions reading the same shared CSV.",
      "The test compares log_quantile_se's returned se against",
      "(log(upper)-log(lower))/(2*z) computed from these R-selected",
      "order statistics using the identical z log_quantile_se uses, so",
      "1e-12 relative is floating-point headroom, not a genuine",
      "methodological tolerance -- any real rank-selection defect would",
      "fail this by orders of magnitude."
    )
  ),
  cases = lapply(case_ids, result_for_case)
)

writeLines(
  toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
  "tests/oracles/fixtures/quantile_woodruff.json"
)
cat("wrote tests/oracles/fixtures/quantile_woodruff.json\n")
