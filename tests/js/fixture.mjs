// Builds a synthetic page tree structurally equivalent to
// scripts/render_compatibility.py's output (same ids/classes/data-*
// attributes/payload shape) and runs docs/javascripts/compatibility.js
// against it inside a vm context, so tests exercise the real production
// script unmodified.

import vm from "node:vm";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createDocument, FakeElement } from "./dom_stub.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const SCRIPT_PATH = path.join(__dirname, "..", "..", "docs", "javascripts", "compatibility.js");

export const PAYLOAD = {
  cells: {
    "sequential|quantile": {
      runtime: "supported",
      family: "not_applicable",
      explanation: "Probed end to end.",
      evidence_source: "probe",
      owner: "increment.readouts",
      probe: "tests/test_composition_matrix.py::test_sequential",
      refusal: null,
    },
    // No cataloged scenario ever declares this identity -- the fixture's
    // stand-in for "an unrelated cell stays baseline and reports no
    // cataloged interactions".
    "estimate|mean": {
      runtime: "supported",
      family: "not_applicable",
      explanation: "Probed end to end.",
      evidence_source: "probe",
      owner: "increment.readouts",
      probe: "tests/test_composition_matrix.py::test_estimate",
      refusal: null,
    },
    "estimate|total": {
      runtime: "not_applicable",
      family: "not_applicable",
      explanation: "report-layer type: refused by MetricSpec and by experiment validation",
      evidence_source: "probe",
      owner: "increment.readouts",
      probe: "tests/test_composition_matrix.py::test_report_calendar",
      refusal: null,
    },
  },
  pairs: {
    "cluster|export": {
      runtime: "refused",
      family: "not_applicable",
      explanation: "wire format carries no cluster marker",
      evidence_source: "probe",
      owner: "increment.query.native_source",
      probe: "tests/test_composition_matrix.py::test_capability_pairs",
      refusal: {
        code: "source.moments.cluster_grain",
        code_display: "source.moments.cluster_grain",
        exception: "CapabilityError",
        fragment: "wire format carries no cluster marker",
      },
    },
    // No scenario ever declares this pair's left/right identity -- the
    // fixture's stand-in for "a pair with no cataloged interactions".
    "cate|estimate": {
      runtime: "refused",
      family: "not_applicable",
      explanation: "no shared join key between cate and estimate outputs",
      evidence_source: "probe",
      owner: "increment.query.native_source",
      probe: "tests/test_composition_matrix.py::test_capability_pairs",
      refusal: null,
    },
  },
  // Mirrors the real production catalog's shape: both scenarios share the
  // same sequential x quantile matrix identity but declare fully explicit,
  // mutually-distinguishable context (family_role/multiplicity), so a
  // reader who opens both interactions can tell them apart from the
  // summary label alone.
  scenarios: {
    "always-valid-quantile": {
      axes: {
        capability: "sequential",
        metric_type: "quantile",
        inference: "always_valid",
        family_role: "primary",
        multiplicity: "disabled",
      },
      runtime: "limited",
      family: "not_applicable",
      explanation: "Always-valid quantile inference runs with a statistical caveat.",
      alternative: "Use fixed-horizon quantile inference if the caveat is unacceptable.",
      evidence: [{ kind: "integration", path: "tests/test_analysis_quantile.py", test: "test_av" }],
      findings: [],
    },
    "always-valid-quantile-secondary-family": {
      axes: {
        capability: "sequential",
        metric_type: "quantile",
        inference: "always_valid",
        family_role: "secondary",
        multiplicity: "enabled",
      },
      runtime: "limited",
      family: "excluded",
      explanation: "The estimate is returned, but the secondary family does not cover it.",
      alternative: "Use fixed-horizon quantile inference for family participation.",
      evidence: [],
      findings: [],
    },
    // A pair scenario -- identified by left/right, exactly like a matrix
    // scenario is identified by capability/metric_type.
    "cluster-export-marker": {
      axes: { left: "cluster", right: "export", inference: "always_valid" },
      runtime: "limited",
      family: "not_applicable",
      explanation: "Synthetic pair-scenario explanation for cluster x export.",
      alternative: null,
      evidence: [],
      findings: [],
    },
  },
  // Mirrors scripts/render_compatibility.py's STATUS_LABELS exactly --
  // this is what real SSR embeds. Tests override this per-case to prove
  // displayStatus reads the payload's mapping rather than reimplementing
  // its own transform.
  status_labels: {
    supported: "SUPPORTED",
    limited: "LIMITED",
    refused: "REFUSED",
    not_applicable: "NOT APPLICABLE",
    unknown: "UNKNOWN",
    participates: "PARTICIPATES",
    excluded: "EXCLUDED",
  },
};

// Builds a fresh fixture (fresh DOM + fresh vm context) per test so tests
// stay isolated from each other's mutations.
export function loadFixture(payloadOverrides = {}) {
  const payload = { ...PAYLOAD, ...payloadOverrides };
  const document = createDocument();
  const root = new FakeElement("div", { id: "compatibility-reference" });

  const sequentialQuantileButton = new FakeElement("button", { className: "compatibility-cell" });
  sequentialQuantileButton.setAttribute("data-capability", "sequential");
  sequentialQuantileButton.setAttribute("data-metric", "quantile");
  sequentialQuantileButton.setAttribute("data-status", "supported");
  sequentialQuantileButton.setAttribute("aria-label", "sequential quantile: SUPPORTED");
  sequentialQuantileButton.textContent = "SUPPORTED";

  const estimateMeanButton = new FakeElement("button", { className: "compatibility-cell" });
  estimateMeanButton.setAttribute("data-capability", "estimate");
  estimateMeanButton.setAttribute("data-metric", "mean");
  estimateMeanButton.setAttribute("data-status", "supported");
  estimateMeanButton.setAttribute("aria-label", "estimate mean: SUPPORTED");
  estimateMeanButton.textContent = "SUPPORTED";

  const estimateTotalButton = new FakeElement("button", { className: "compatibility-cell" });
  estimateTotalButton.setAttribute("data-capability", "estimate");
  estimateTotalButton.setAttribute("data-metric", "total");
  estimateTotalButton.setAttribute("data-status", "not_applicable");
  estimateTotalButton.setAttribute("aria-label", "estimate total: NOT APPLICABLE");
  estimateTotalButton.textContent = "NOT APPLICABLE";

  const clusterExportButton = new FakeElement("button", {
    className: "compatibility-cell compatibility-pair",
  });
  clusterExportButton.setAttribute("data-left", "cluster");
  clusterExportButton.setAttribute("data-right", "export");
  clusterExportButton.setAttribute("data-status", "refused");
  clusterExportButton.setAttribute("aria-label", "cluster export: REFUSED");
  clusterExportButton.textContent = "REFUSED";

  const cateEstimateButton = new FakeElement("button", {
    className: "compatibility-cell compatibility-pair",
  });
  cateEstimateButton.setAttribute("data-left", "cate");
  cateEstimateButton.setAttribute("data-right", "estimate");
  cateEstimateButton.setAttribute("data-status", "refused");
  cateEstimateButton.setAttribute("aria-label", "cate estimate: REFUSED");
  cateEstimateButton.textContent = "REFUSED";

  const drawer = new FakeElement("section", { id: "compatibility-drawer" });

  root.append(
    sequentialQuantileButton,
    estimateMeanButton,
    estimateTotalButton,
    clusterExportButton,
    cateEstimateButton,
    drawer
  );

  const dataNode = new FakeElement("script", { id: "compatibility-data" });
  dataNode.textContent = JSON.stringify(payload);

  document.append(root, dataNode);

  const context = vm.createContext({ document });
  const source = fs.readFileSync(SCRIPT_PATH, "utf8");
  vm.runInContext(source, context, { filename: SCRIPT_PATH });

  return {
    root,
    sequentialQuantileButton,
    estimateMeanButton,
    estimateTotalButton,
    clusterExportButton,
    cateEstimateButton,
    drawer,
  };
}

// Simulates compatibility.js's `if (!root || !dataNode) return;` guard
// tripping -- the `#compatibility-data` script node is never appended, so
// the script must return before wiring anything or marking the page
// enhanced. Proves a failure before initialization completes leaves the
// no-JS static-details fallback visible (never hidden by a half-run
// script).
export function loadFixtureWithoutDataNode() {
  const document = createDocument();
  const root = new FakeElement("div", { id: "compatibility-reference" });
  document.append(root);

  const context = vm.createContext({ document });
  const source = fs.readFileSync(SCRIPT_PATH, "utf8");
  vm.runInContext(source, context, { filename: SCRIPT_PATH });

  return { root };
}

// Simulates the `#compatibility-drawer` anchor being missing while root
// and a valid dataNode both exist -- proves the guard treats every
// required interactive anchor (root, dataNode, drawer) as one atomic
// precondition, not just root/dataNode: init must stop, with no
// enhancement class and no listeners wired, before ever dereferencing a
// missing drawer.
export function loadFixtureWithoutDrawer(payloadOverrides = {}) {
  const payload = { ...PAYLOAD, ...payloadOverrides };
  const document = createDocument();
  const root = new FakeElement("div", { id: "compatibility-reference" });

  const sequentialQuantileButton = new FakeElement("button", { className: "compatibility-cell" });
  sequentialQuantileButton.setAttribute("data-capability", "sequential");
  sequentialQuantileButton.setAttribute("data-metric", "quantile");
  sequentialQuantileButton.setAttribute("data-status", "supported");
  sequentialQuantileButton.setAttribute("aria-label", "sequential quantile: SUPPORTED");
  sequentialQuantileButton.textContent = "SUPPORTED";

  // Deliberately no #compatibility-drawer appended.
  root.append(sequentialQuantileButton);

  const dataNode = new FakeElement("script", { id: "compatibility-data" });
  dataNode.textContent = JSON.stringify(payload);

  document.append(root, dataNode);

  const context = vm.createContext({ document });
  const source = fs.readFileSync(SCRIPT_PATH, "utf8");
  vm.runInContext(source, context, { filename: SCRIPT_PATH });

  return { root, sequentialQuantileButton };
}
