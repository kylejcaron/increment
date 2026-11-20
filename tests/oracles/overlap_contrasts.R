# Run with Rscript --vanilla tests/oracles/overlap_contrasts.R.
# Saturated OLS with HC2 covariance, transformed from regression coding.
counts <- c(41, 53, 37, 64)
a <- rep(c(0, 1, 0, 1), counts)
b <- rep(c(0, 0, 1, 1), counts)
X <- model.matrix(~ a * b)
L <- rbind(c(0, 1, 0, .75), c(0, 0, 1, .25),
           c(0, 1, 0, 0), c(0, 1, 0, 1), c(0, 0, 1, 0), c(0, 0, 1, 1),
           c(0, 1, 1, 1), c(0, 0, 0, 1))
cell <- factor(rep(seq_along(counts), counts))
alpha <- .05
members <- 2 * nrow(L)

oracle <- function(location, scale, boundary) {
  y <- unlist(Map(function(m, s, n) m + s * ((2 * seq_len(n) - n - 1) / n),
                  location, scale, counts))
  fit <- lm(y ~ a * b)
  bread <- solve(crossprod(X))
  meat <- crossprod(X, X * (resid(fit)^2 / (1 - hatvalues(fit))))
  cov <- L %*% bread %*% meat %*% bread %*% t(L)
  estimate <- drop(L %*% coef(fit))
  se <- sqrt(diag(cov))
  independent <- sweep((L %*% solve(unique(X)))^2, 2, tapply(y, cell, var) / counts, `*`)
  df <- rowSums(independent)^2 / rowSums(sweep(independent^2, 2, counts - 1, `/`))
  width <- qt(alpha / (2 * members), df, lower.tail = FALSE) * se
  p <- 2 * pt(abs(estimate - boundary) / se, df, lower.tail = FALSE)
  list(rows = cbind(estimate, se, df, estimate - width, estimate + width, p, pmin(1, members * p)),
       covariance = cov)
}

# Base R JSON output for finite numeric vectors/matrices, with round-trip precision.
vec <- function(x) paste0("[", paste(sprintf("%.17g", x), collapse = ", "), "]")
mat <- function(x) paste0("[\n      ", paste(apply(x, 1, vec), collapse = ",\n      "), "\n    ]")
metric <- function(name, out) paste0('  "', name, '": {\n    "rows": ', mat(out$rows),
                                    ',\n    "covariance": ', mat(out$covariance), '\n  }')
cat('{\n  "columns": ["estimate", "se", "df", "lower", "upper", "p_boundary", "p_boundary_adj"],\n',
    metric("orders", oracle(c(10, 14, 13, 12), c(.8, 1.3, 2, 3), 0)), ',\n',
    metric("complaint", oracle(c(150, 149, 151, 149.2), c(.5, .4, .6, .6), .1)), '\n}\n', sep = '')
