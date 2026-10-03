# Release-triggered Website Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Notify the documentation website after stable package publication and expose its URL in the package README.

**Architecture:** The package sends a small cross-repository event only after both PyPI publication and GitHub Release creation succeed. The website independently validates the published release and exact tag commit, exports and synchronizes documentation, and opens a draft update PR; its main-branch CI owns production deployment. Package publishing has no dependency on the website build or deployment.

**Tech Stack:** GitHub Actions, GitHub CLI, short-lived GitHub App installation tokens, Python, Astro/Starlight, Cloudflare Pages.

## Global Constraints

- Worktree: `/Users/kylejcaron/increment/.worktrees/release-website-sync`; branch: `release-website-sync`; base: `694ed7f4e109c193759b46a211fa73ddb2400aa6`.
- Website companion: `/Users/kylejcaron/open_source/getincrement.io/.worktrees/release-docs-deployment/docs/superpowers/plans/2026-10-03-release-docs-deployment.md`.
- Draft PRs only. Do not merge, tag, release, dispatch live workflows, push main, configure secret values, or deploy.
- Preserve user changes in the original checkout and the existing explicit bump-to-release dispatch, exact-commit CI gate, protected PyPI environment, OIDC publishing, and artifact checks.
- Only published stable releases can update production documentation. Prereleases do not silently replace stable docs.
- Event type is `increment_release`; payload contains exactly `tag` and `sha`. Source repository is fixed to `kylejcaron/increment`; destination is fixed to `kylejcaron/getincrement.io`.
- Authentication: variable `DOCS_APP_ID`, secret `DOCS_APP_PRIVATE_KEY`, and opt-in variable `DOCS_SYNC_ENABLED=true`. The App installation must include only the website repository; sender token requests only contents write. No Cloudflare credentials in the package repository.
- All actions use immutable SHA pins. Dynamic inputs enter commands through environment variables and quoted arguments, never direct expression interpolation in shell source.
- Secret values are never read or printed. Tokens are masked and revoked by the token action, never persisted in source, payloads, artifacts, or PR descriptions.
- No new permanent tests for workflow wiring or copied JSON fields. Exercise the actual shell in an isolated throwaway smoke and use existing workflow behavioral checks.

## Task 1: Stable-release notification and README website link

**Files:** Modify `.github/workflows/release.yml`, `README.md`, and `CONTRIBUTING.md`.

**Consumes:** Existing `build.outputs.prerelease`, `github.ref_name`, `github.sha`, and successful `github-release` job.

**Produces:** `repository_dispatch` to the website with this contract:

```json
{"event_type":"increment_release","client_payload":{"tag":"v1.2.3","sha":"0123456789abcdef0123456789abcdef01234567"}}
```

- [ ] Read current-main workflows. The release handoff repair is already merged; do not repeat the stale branch's defect.
- [ ] Add a separate `website-notify` job depending on `[build, github-release]`, conditional on stable output and `DOCS_SYNC_ENABLED == 'true'`. Give it a short timeout and no repository write permissions. The website job never becomes a dependency of package publishing.
- [ ] Mint a narrowly scoped token with `actions/create-github-app-token@fee1f7d63c2ff003460e3d139729b119787bc349`, `owner: kylejcaron`, `repositories: getincrement.io`, `permission-contents: write`.
- [ ] Generate and dispatch JSON with this shell, using the minted token only in that step's `GH_TOKEN` environment:

```bash
set -euo pipefail
jq -n --arg tag "$TAG" --arg sha "$SHA" \
  '{event_type:"increment_release",client_payload:{tag:$tag,sha:$sha}}' \
  | gh api --method POST repos/kylejcaron/getincrement.io/dispatches --input -
```

- [ ] Set `TAG` from `github.ref_name` and `SHA` from `github.sha`. A notification error is reported independently; it cannot undo or prevent the already-completed package publication.
- [ ] Add `[Website](https://incrementdocs.pages.dev/)` and a direct hosted documentation link near the README's existing navigation. Keep local source guide links intact.
- [ ] Document the opt-in, App installation, variable/secret names, exact release pinning, stable-only policy, manual website receiver recovery, companion PR dependency, and the absence of Cloudflare credentials here. Include names only, not sample private keys or tokens.
- [ ] Exercise the actual new workflow shell locally with a throwaway `gh` executable that captures stdin and parses the event. Verify the destination, tag, SHA, and absence of token material; send no network POST. Remove the throwaway file.
- [ ] Run existing focused workflow/release tests, actionlint, and `make check` after integration. Commit only scoped changes with a product-language message.

## Task 2: Cross-repository verification and draft PR delivery

**Files:** No unrelated package changes. This task reviews both branches and uses the website companion's completed code.

**Consumes:** Task 1 sender, website exact-release validator, version-aware docs surfaces, credential-isolated CI, and security setup documentation.

**Produces:** Two linked draft PRs, complete local evidence, and explicit setup prerequisites.

- [ ] Validate the sender/receiver event contract, stable-only policy, published-release tag/SHA matching, duplicate notification behavior, controlled generated-file paths, and absence of automatic merging.
- [ ] Run website Python/Node regressions and a production build. Run actual release preparation against public release metadata without credentials; if no stable release exists, verify the refusal and exercise accepted stable data locally without dispatching.
- [ ] Smoke both development and release modes on the actual built site: homepage install command, docs banner, Markdown notice, and `llms.txt` must agree with version metadata. No UI design change is intended.
- [ ] Review both workflows for untrusted-PR credential exposure, shell injection, artifact content, token lifetime, and production branch/environment checks.
- [ ] Run redacted secret scans against both staged changes and final branch commit ranges. Exclude no changed production files; retain only safe scanner summaries in evidence.
- [ ] Run branch-pinned review cadence checks before issue closure. Preserve any findings that cannot be verified rather than claiming hosted success.
- [ ] Commit, push only the two feature branches, and create PRs with `gh pr create --draft --base main`. Link the companion PRs and describe configuration prerequisites using variable and secret names only.
- [ ] Verify both PRs are draft through the GitHub API. Do not merge or execute deployment. Record verification evidence and commit SHAs in the issue ledger.

## Acceptance

A successful stable package release can notify the website without coupling package publication to website availability. The website receives only tag/SHA provenance, prepares docs from that exact published revision, creates a reviewed draft update PR, and deploys main only through its own protected job. The package README links the hosted website. The delivered implementation remains in two draft PRs; no live secret, release, or deployment mutation occurs during development.
