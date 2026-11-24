# Published group-average indirect-effect intervals on synthetic observations.
# Rscript --vanilla tests/oracles/generate/gen_spillover_published.R
# No package installation. Fetch and check upstream sources before evaluation.
pinned_source <- function(repo, commit, file, sha256) {
  url <- sprintf("https://raw.githubusercontent.com/cran/%s/%s/R/%s", repo, commit, file)
  path <- tempfile()
  on.exit(unlink(path))
  download.file(url, path, mode = "wb", quiet = TRUE)
  digest <- system2("shasum", c("-a", "256", shQuote(path)), stdout = TRUE)
  stopifnot(identical(substr(digest, 1, 64), sha256))
  parse(path)
}

upstream <- new.env()
eval(pinned_source("gtools", "d92e3e803b9102f41da2346d7f0d416a29cada1a", "combinations.R",
                   "f629e4a96de6d97c464dc88065b34ef608540462f9589e7875d9d1f9a03e2c01"), upstream)
code <- pinned_source("interferenceCI", "9352a8ec903a01f0abf4eb83f7e689e5ff1922dd", "interferenceCI.R",
                      "136270d242bcef02d4dbb31a3a27d7cbb758ce91b133e064541b31beda5fd7ce")
for (declaration in code) {
  # Its combinations() dependency is already in this environment.
  if (!identical(declaration, quote(library("gtools")))) eval(declaration, upstream)
}

alpha <- 0.05
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
cat(R.version.string, "; RNG:", RNGkind(), "\n")

compare <- function(label, n, policy, quota0, quota1, events, offered_events = 0 * n,
                    sample_eit = TRUE) {
  q <- c(mean(policy == 0), mean(policy == 1))
  offers <- ifelse(policy == 0, quota0, quota1)
  untreated <- n - offers
  stopifnot(all(quota0 < n & quota1 < n), all(0 <= events & events <= untreated),
            all(0 <= offered_events & offered_events <= offers))
  # Rows: offered/unoffered. Columns: event/no event. Offered outcomes unused by IE.
  table <- array(rbind(offered_events, events, offers - offered_events, untreated - events),
                 dim = c(2, 2, length(n)))
  rate <- events / untreated
  point <- mean(rate[policy == 1]) - mean(rate[policy == 0])
  L <- pmax(-expm1(-lchoose(n, quota0)) / q[1], -expm1(-lchoose(n, quota1)) / q[2])
  radius <- sqrt((max(q^-2) + mean(L^2)) * log(2 / alpha) / (2 * length(n)))
  tv <- upstream$TV("IE", policy, table, quota0, quota1, alpha)
  stopifnot(isTRUE(all.equal(point, -tv$est)))  # upstream reports policy 0 minus 1
  cat(sprintf("%s: point %.9f; Th.3 [%.9f, %.9f]; R TV [%.9f, %.9f]\n",
              label, point, max(-1, point - radius), min(1, point + radius), -tv$upper, -tv$lower))
  if (sample_eit) {
    for (seed in c(519L, 520L)) {
      set.seed(seed)
      eit <- upstream$exactCI("IE", policy, table, quota0, quota1, 100, 100, alpha)
      cat(sprintf("  R sampled EIT %d: [%.9f, %.9f]\n", seed, -eit$upper, -eit$lower))
    }
  }
}

for (label in c("equal", "unequal")) {
  sizes <- if (label == "equal") rep(3, 4) else c(2, 3, 2, 4)
  compare(label, sizes, c(0, 0, 1, 1), rep(0, 4), rep(1, 4), c(0, 1, 1, 2))
}
# Event counts from the package's example (interferenceCI.R:6-8).
compare("10x10", rep(10, 10), c(1, 0, 0, 0, 1, 1, 0, 1, 1, 0), rep(3, 10), rep(6, 10),
        c(1, 6, 4, 5, 1, 2, 5, 3, 1, 5), offered_events = c(1, 0, 0, 0, 1, 2, 1, 2, 1, 1))
# TV is design-only apart from its center; do not allocate exactCI's assignment matrix here.
compare("240x59", rep(59, 240), rep(0:1, each = 120), rep(0, 240), rep(1, 240),
        c(rep(14, 120), rep(58, 4), rep(13, 28), rep(14, 88)),
        offered_events = c(rep(0, 120), rep(1, 32), rep(0, 88)), sample_eit = FALSE)
