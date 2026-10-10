<!-- Keep it concise; omit fields that do not apply, but retain evidence and checks. -->

## Change map
- **Behavior:** What does the user observe before and after this change? (For docs-only changes, what guidance changed?)
- **Owner:** Which subsystem owns the behavior?
- **Affected paths:** Which production, public/wire, test, or documentation paths changed?
- **Evidence:** What evidence was needed, and what evidence did you collect?
- **Assumptions / limits:** What assumptions or unsupported combinations matter? Omit if none.
- **Migration:** Is a public API or persisted/wire contract changing? State the affected paths and migration route, or “none.”

## Checks
- **Run:** Which focused checks did you run, and what did they show?
- **Not run / deferred:** What relevant or costly evidence remains, and why?

## Review
- **Human review:** Link the existing [contributor review questions](../CONTRIBUTING.md#review-and-handoffs). Request human statistical review separately when the change needs it; automated review does not replace it.
- **Public API docs:** For a nonbreaking public-API change outside automatic strict-docs paths, request the `docs-required` label once available. `documentation` is for docs-only changes.
