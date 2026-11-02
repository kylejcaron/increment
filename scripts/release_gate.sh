#!/usr/bin/env bash
# Refuse a release unless its exact candidate commit carries the evidence the
# version requires.
#
# Every version needs a successful ci-ok, a successful nightly run, no other
# unfinished or failed check, and successful legacy statuses when present. A
# stable version also needs a scheduled or manually dispatched nightly (which
# runs certification), the weekly dependency-floor job on every interpreter
# with floor sessions, and core-install (3.15) in the run whose ci-ok succeeded.
# FORCE=true skips these checks for a prerelease and is refused for a stable version.
#
# Environment: VERSION and SHA (required); GITHUB_REPOSITORY (defaults to the
# checkout's GitHub repository); RUN_URL, the running workflow's URL, whose own
# check runs are exempt; FORCE. From a checkout:
#
#   VERSION=1.0.0 SHA="$(git rev-parse origin/main)" scripts/release_gate.sh
# shellcheck disable=SC2016  # Backticks in the messages are literal.
set -euo pipefail

# Interpreters with dependency-floor sessions: the weekly test-all-floor matrix.
floor_pythons=(3.12 3.13 3.14)
floor_job="full suite at dependency floors"
core_install_job="core-install (3.15)"

: "${VERSION:?VERSION is required}"
: "${SHA:?SHA is required}"
repo=${GITHUB_REPOSITORY:-}
if [ -z "$repo" ]; then
  repo='{owner}/{repo}'
fi

prerelease_pattern='(a|b|rc|\.dev)[0-9]+'
if [[ $VERSION =~ $prerelease_pattern ]]; then
  stage=prerelease
else
  stage=stable
fi

if [ "${FORCE:-false}" = true ]; then
  if [ "$stage" = stable ]; then
    echo "::error::force applies only to prereleases; stable version $VERSION must pass the release gate"
    exit 1
  fi
  echo "::warning::force skipped the release gate for prerelease $VERSION on $SHA"
  exit 0
fi

# Include every page, but not the running workflow: its own job is running
# while this gate executes. Other unfinished checks must block release.
runs=$(gh api --paginate --slurp \
  "repos/${repo}/commits/${SHA}/check-runs?per_page=100&filter=latest" \
  | jq --arg run_url "${RUN_URL:-}" \
    '[.[].check_runs[] | select(($run_url != "" and .app.slug == "github-actions" and
      ((.details_url // "") | startswith($run_url))) | not)]')

if ! printf '%s' "$runs" | jq -e --arg sha "$SHA" \
  'any(.[]; .name == "ci-ok" and .app.slug == "github-actions" and
    .head_sha == $sha and .status == "completed" and .conclusion == "success")' >/dev/null; then
  echo "::error::ci-ok must finish successfully on $SHA before release"
  exit 1
fi

nightlies=$(gh api --paginate --slurp \
  "repos/${repo}/actions/workflows/nightly.yml/runs?head_sha=${SHA}&status=completed&per_page=100")
# A labeled-PR nightly skips the scientific tiers, so it cannot supply stable
# certification evidence. Scheduled and manually dispatched runs include them.
if ! printf '%s' "$nightlies" | jq -e --arg sha "$SHA" --arg stage "$stage" \
  '[.[].workflow_runs[] | select(.head_sha == $sha and .status == "completed"
    and .conclusion == "success" and ($stage == "prerelease"
      or .event == "schedule" or .event == "workflow_dispatch"))] | length > 0' >/dev/null; then
  echo "::error::a successful nightly run is required on $SHA before release (stable versions require certification from a scheduled or manually dispatched nightly)"
  printf '::error::run `gh workflow run nightly.yml --ref main` while main is at %s, or wait for the scheduled run\n' "$SHA"
  exit 1
fi

if [ "$stage" = stable ]; then
  # An interpreter qualifies when its weekly floor job, or every part
  # "(<python>, part <g>/<n>)" of a split job, succeeded on the candidate.
  weekly_suites=$(gh api --paginate --slurp \
    "repos/${repo}/actions/workflows/weekly.yml/runs?head_sha=${SHA}&per_page=100" \
    | jq --arg sha "$SHA" \
      '[.[].workflow_runs[] | select(.head_sha == $sha) | .check_suite_id | select(. != null)]')
  missing=()
  for python in "${floor_pythons[@]}"; do
    if ! printf '%s' "$runs" | jq -e --arg sha "$SHA" \
      --argjson suites "$weekly_suites" \
      --arg single "$floor_job ($python)" --arg prefix "$floor_job ($python, part " '
      [.[] | select(.name == $single or (.name | startswith($prefix)))] as $jobs
      | [$jobs[] | .name | select(startswith($prefix))] as $split
      | [$split[] | capture("part (?<g>[0-9]+)/(?<n>[0-9]+)\\)$")] as $parts
      | ($jobs | length) > 0
        and all($jobs[]; .app.slug == "github-actions" and .head_sha == $sha
          and .status == "completed" and .conclusion == "success"
          and (.check_suite.id as $suite | $suites | index($suite)) != null)
        and (($split | length) == 0
          or (($parts | length) == ($split | length)
            and ($parts | map(.n) | unique | length) == 1
            and ($parts | map(.g | tonumber) | unique)
              == [range(1; ($parts[0].n | tonumber) + 1)]))' >/dev/null; then
      missing+=("$python")
    fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "::error::stable release requires \"$floor_job\" to succeed on $SHA for Python ${missing[*]}"
    printf '::error::run `gh workflow run weekly.yml --ref main` while main is at %s, or wait for the scheduled run\n' "$SHA"
    exit 1
  fi

  if ! printf '%s' "$runs" | jq -e --arg sha "$SHA" --arg job "$core_install_job" '
    [.[] | select(.app.slug == "github-actions" and .head_sha == $sha
      and .status == "completed" and .conclusion == "success")] as $green
    | any($green[] | select(.name == "ci-ok") | .check_suite.id | select(. != null);
      . as $suite | any($green[]; .name == $job and .check_suite.id == $suite))' >/dev/null; then
    echo "::error::stable release requires \"$core_install_job\" to succeed in the ci run whose ci-ok succeeded on $SHA"
    printf '::error::run `gh workflow run ci.yml --ref main` while main is at %s\n' "$SHA"
    exit 1
  fi
fi

bad=$(printf '%s' "$runs" | jq -r \
  '.[] | select(.status != "completed" or
    (.conclusion != "success" and .conclusion != "neutral" and .conclusion != "skipped")) | .name')
if [ -n "$bad" ]; then
  echo "::error::these checks are unfinished or did not pass on $SHA:"
  printf '%s\n' "$bad"
  exit 1
fi

# Legacy statuses are independent of check runs; an absent status is
# fine, but an existing pending or failed status is not.
status_json=$(gh api --paginate --slurp \
  "repos/${repo}/commits/${SHA}/status?per_page=100")
status_total=$(printf '%s' "$status_json" | jq '[.[].statuses[]] | length')
if [ "$status_total" -gt 0 ] && printf '%s' "$status_json" \
  | jq -e 'any(.[]; .state != "success")' >/dev/null; then
  echo "::error::legacy commit statuses are not all successful on $SHA"
  printf '%s' "$status_json" | jq -r \
    '.[].statuses[] | select(.state != "success") | .context'
  exit 1
fi
checks_total=$(printf '%s' "$runs" | jq 'length')
echo "$stage $VERSION: $checks_total check run(s), $status_total commit status(es), all green"
