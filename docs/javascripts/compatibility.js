(() => {
  // root, dataNode, and drawer are every required interactive anchor
  // this script dereferences unconditionally -- checked together, before
  // any parsing/wiring/hiding-the-fallback happens, so a page missing
  // any one of them (however that happened) leaves initialization a
  // strict no-op: no enhancement class, no listeners, nothing to throw
  // later when a reader interacts with a page JS never actually wired.
  const root = document.querySelector("#compatibility-reference");
  const dataNode = document.querySelector("#compatibility-data");
  const drawer = root?.querySelector("#compatibility-drawer");
  if (!root || !dataNode || !drawer) return;

  const payload = JSON.parse(dataNode.textContent);
  let selectedCell = null;

  // Structural identity axes: a matrix scenario is identified by
  // capability/metric_type (chosen via the matrix cell button), a pair
  // scenario by left/right (chosen via the pair button). Every other axis
  // a scenario declares is "context" -- what actually distinguishes one
  // cataloged interaction from another for the same cell. This set is a
  // fixed structural fact about the payload shape, not a per-scenario or
  // per-value dependency: adding a new scenario or a new context axis
  // needs no change here.
  const IDENTITY_AXES = new Set(["capability", "metric_type", "left", "right"]);

  // Same transform scripts/render_compatibility.py's axis-label rendering
  // uses (`axis.replace('_', ' ').title()`) -- kept in sync by
  // inspection, not by import, since this is a generic snake_case ->
  // Title Case transform of the axis's own name, never a per-axis
  // hardcoded label table.
  const axisLabel = (axis) =>
    axis
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");

  // A scenario's own context axes, human-labeled and value-formatted,
  // joined into one line -- e.g. "Family Role: primary, Inference:
  // always valid, Multiplicity: disabled". Derived purely from the
  // scenario's own declared axes; never a hardcoded per-scenario label.
  const interactionLabel = (scenario) =>
    Object.entries(scenario.axes)
      .filter(([axis]) => !IDENTITY_AXES.has(axis))
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([axis, value]) => `${axisLabel(axis)}: ${value.replace(/_/g, " ")}`)
      .join(", ");

  const floorText = (floor) => {
    const { kind, ...rest } = floor;
    const extra = Object.entries(rest)
      .map(([key, value]) => `${key}=${value}`)
      .join(", ");
    return extra ? `${kind} (${extra})` : kind;
  };

  // Human-readable status/family label, looked up in the payload's own
  // `status_labels` -- the single explicit mapping
  // scripts/render_compatibility.py's `STATUS_LABELS` renders from and
  // embeds verbatim. JS never reimplements the snake_case-to-label rule;
  // there is one mapping, not two equivalent-looking ones that could
  // silently drift apart. A value absent from the mapping falls back to
  // plain uppercasing -- deliberately NOT the underscore-to-space rule --
  // so an incomplete mapping degrades visibly (a raw underscore stays
  // visible, e.g. "NOT_APPLICABLE") instead of silently still looking
  // correct. Internal lookups (payload keys, scenario axis comparisons)
  // stay snake_case; only user-facing text goes through this.
  const displayStatus = (value) => payload.status_labels[value] ?? value.toUpperCase();

  const findingSummary = (finding) => {
    const parts = [`${finding.capability}: runtime=${displayStatus(finding.runtime)}`];
    if (finding.warning) parts.push(`warning: ${finding.warning}`);
    if (finding.refusal_code) parts.push(`refusal code: ${finding.refusal_code}`);
    if (finding.assumptions?.length) parts.push(`assumptions: ${finding.assumptions.join(", ")}`);
    if (finding.floor) parts.push(`sampling floor: ${floorText(finding.floor)}`);
    if (finding.reference) parts.push(`reference: ${finding.reference}`);
    return parts.join("; ");
  };

  const contractProvenance = (findings) =>
    findings
      .map(
        (finding) =>
          `${finding.capability} (${finding.evidence_source}` +
          (finding.reference ? `, ref=${finding.reference}` : "") +
          ")"
      )
      .join("; ");

  // `status`, when given, is the raw internal snake_case value (never
  // the display label) tagged onto the `dd` as `data-status` so CSS can
  // color it semantically -- see docs/stylesheets/compatibility.css.
  // Only the Runtime/Family fields below pass one; every other field
  // (Meaning, Owner, Evidence, ...) is plain text with no status to color.
  const addField = (list, name, value, status) => {
    const term = document.createElement("dt");
    term.textContent = name;
    const description = document.createElement("dd");
    description.textContent = value;
    if (status !== undefined) description.setAttribute("data-status", status);
    list.append(term, description);
  };

  // Every field a base MATRIX/PAIRS record or a scenario record can
  // carry -- shared between the top-level selection and each nested
  // "Known interactions" entry below it, since both are the same record
  // shape (a scenario additionally carries `alternative`/`evidence`/
  // `findings`, which a plain cell/pair record never has; the
  // conditionals below simply skip whatever is absent).
  const buildRecordFields = (list, record) => {
    addField(list, "Runtime", displayStatus(record.runtime), record.runtime);
    addField(list, "Family", displayStatus(record.family), record.family);
    addField(list, "Meaning", record.explanation);
    if (record.owner) addField(list, "Owner", record.owner);
    if (record.probe) addField(list, "Evidence", record.probe);
    if (record.refusal) {
      addField(list, "Refusal code", record.refusal.code_display);
      addField(list, "Refusal exception", record.refusal.exception);
      addField(list, "Refusal fragment", record.refusal.fragment);
    }
    if (record.alternative) addField(list, "Alternative", record.alternative);
    if (record.evidence_source) addField(list, "Evidence source", record.evidence_source);
    if (record.evidence?.length) {
      addField(
        list,
        "Evidence",
        record.evidence.map((item) => `${item.kind}: ${item.path}::${item.test}`).join("; ")
      );
    }
    if (record.findings?.length) {
      addField(list, "Contract provenance", contractProvenance(record.findings));
      addField(list, "Findings", record.findings.map(findingSummary).join(" | "));
    }
  };

  // `interactions` is every SCENARIOS entry whose identity matches the
  // selected cell -- capability/metric_type for a matrix cell, left/right
  // for a pair -- possibly empty, but always a real array; both click
  // handlers below always compute and pass one. Each renders as a native
  // <details>/<summary> the reader opens independently, labeled from its
  // own context axes; opening it reveals the exact same field set
  // (status colors, evidence, findings, alternative) a full drawer
  // record would show. An empty list says so explicitly rather than
  // rendering nothing.
  const render = (title, record, interactions) => {
    drawer.replaceChildren();
    const heading = document.createElement("h3");
    heading.textContent = title;
    const list = document.createElement("dl");
    buildRecordFields(list, record);
    drawer.append(heading, list);

    const section = document.createElement("div");
    section.className = "compatibility-interactions";
    const subheading = document.createElement("h4");
    subheading.textContent = "Known interactions";
    section.append(subheading);
    if (interactions.length === 0) {
      const empty = document.createElement("p");
      empty.textContent = "No higher-order interactions are cataloged for this cell.";
      section.append(empty);
    } else {
      for (const scenario of interactions) {
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = interactionLabel(scenario);
        const nestedList = document.createElement("dl");
        buildRecordFields(nestedList, scenario);
        details.append(summary, nestedList);
        section.append(details);
      }
    }
    drawer.append(section);
  };

  const buttons = [...root.querySelectorAll(".compatibility-cell")];

  for (const button of buttons) {
    button.addEventListener("click", () => {
      selectedCell?.removeAttribute("aria-current");
      selectedCell = button;
      selectedCell.setAttribute("aria-current", "true");
      if (button.dataset.left !== undefined) {
        const { left, right } = button.dataset;
        const interactions = Object.values(payload.scenarios).filter(
          (scenario) => scenario.axes.left === left && scenario.axes.right === right
        );
        render(`${left} × ${right}`, payload.pairs[`${left}|${right}`], interactions);
        return;
      }
      const { capability, metric } = button.dataset;
      const interactions = Object.values(payload.scenarios).filter(
        (scenario) => scenario.axes.capability === capability && scenario.axes.metric_type === metric
      );
      render(`${capability} × ${metric}`, payload.cells[`${capability}|${metric}`], interactions);
    });
  }

  // Marks successful initialization -- reached only if every step above
  // ran without throwing (payload parsed, every DOM anchor found,
  // listeners wired). docs/stylesheets/compatibility.css scopes hiding
  // #compatibility-static-details (the no-JS fallback) to this class, so
  // it stays visible if the script fails anywhere before this point
  // instead of a half-initialized page hiding it prematurely.
  root.classList.add("compatibility-js-enhanced");
})();
