# Contributing to increment

Thank you for your interest in contributing.

## License and the CLA

increment is licensed under the [Apache License 2.0](LICENSE).

Before your first pull request can be merged, you must sign the
[Contributor License Agreement](.github/CLA.md). Comment the following on your PR and
the CLA assistant records it:

```
I have read the CLA Document and I hereby sign the CLA
```

You sign once and it covers every future contribution.

**You keep the copyright in your work.** The CLA is a license, not an
assignment -- you may reuse, sell, or relicense your own contributions freely.
What it grants increment is the right to sublicense contributed code, which is
what makes commercial licensing and any future relicensing of the project
possible. Without it, a single unreachable contributor can permanently block
both.

Contributing on behalf of a company whose legal team requires its own
agreement? Email kyle.j.caron@gmail.com before opening a pull request.

## Development

This repository currently holds the project's licensing, contributor
agreement, and release automation. The package source is not here yet, so
there is no build, test suite, or lint config to run against it.

What is here and how to check it:

- GitHub workflows under `.github/workflows/` -- validate with
  [`actionlint`](https://github.com/rhysd/actionlint) before pushing.
- Release process -- `.github/workflows/bump.yml` creates the tag,
  `.github/workflows/release.yml` builds and publishes it. `pyproject.toml`
  is in place, but a build produces an empty wheel until the package source
  lands; `release.yml` detects that and refuses to publish.
- Agent and issue-tracker conventions -- [AGENTS.md](AGENTS.md).
