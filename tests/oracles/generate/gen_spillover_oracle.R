# Reproduce the six-group point and variance frozen in tests/estimation/test_spillover.py.
# Supply the local path to RCT2 0.0.1 R/CalAPO.R from commit
# 5820e34f4a20ca116f0b7fd069ebdf735523324d; SHA-256:
# db9f91a162f366af68974757d0a485d2a02fd5b55c624ecf45f5a2ec3473d648
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1)
source(args[[1]])
n <- c(4, 5, 6, 8, 7, 9)
high <- c(1, 0, 1, 0, 0, 1)
treated <- c(3, 2, 4, 3, 2, 6)
events <- c(1, 1, 0, 2, 4, 1)
rows <- do.call(rbind, lapply(seq_along(n), function(g) {
  untreated <- n[g] - treated[g]
  data.frame(id = g, A = high[g], Z = rep(c(1, 0), c(treated[g], untreated)),
             Y = n[g] / mean(n) * rep(c(0, 1, 0), c(treated[g], events[g], untreated - events[g])))
}))
fit <- CalAPO(rows)
# RCT2 orders this contrast low minus high; we report high minus low.
print(c(estimate = -fit$ASE.est[2], variance = fit$var.hat.ASE[2, 2]), digits = 17)
