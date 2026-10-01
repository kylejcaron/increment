// Matrix/pair buttons render their SSR baseline status forever -- there is
// no global filter to mutate them. Clicking either a matrix cell or a
// pair button renders its base record plus a "Known interactions"
// section listing every SCENARIOS entry whose identity matches that
// cell/pair (capability/metric_type for a cell, left/right for a pair),
// each as an independently-openable native <details>/<summary>. Runs the
// real docs/javascripts/compatibility.js against a synthetic DOM
// (tests/js/dom_stub.mjs) -- no jsdom or other npm dependency exists in
// this project.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  loadFixture,
  loadFixtureWithoutDataNode,
  loadFixtureWithoutDrawer,
} from "./fixture.mjs";

test("matrix/pair buttons show their SSR baseline status and never mutate -- there is no filter", () => {
  const fixture = loadFixture();
  assert.equal(fixture.sequentialQuantileButton.textContent, "SUPPORTED");
  assert.equal(fixture.sequentialQuantileButton.getAttribute("data-status"), "supported");
  assert.equal(fixture.clusterExportButton.textContent, "REFUSED");
  assert.equal(fixture.clusterExportButton.getAttribute("data-status"), "refused");
  assert.equal(fixture.estimateMeanButton.textContent, "SUPPORTED");
  // No [data-axis] filter control exists anywhere on the page at all.
  assert.equal(fixture.root.querySelectorAll("[data-axis]").length, 0);
});

test("clicking a matrix cell with cataloged scenarios lists both as native details, each with its own explicit label", () => {
  const fixture = loadFixture();
  fixture.sequentialQuantileButton.dispatch("click");

  const sections = fixture.drawer.querySelectorAll(".compatibility-interactions");
  assert.equal(sections.length, 1);
  const [heading, ...detailsElements] = sections[0].children;
  assert.equal(heading.textContent, "Known interactions");
  assert.equal(detailsElements.length, 2);

  const labels = detailsElements.map((details) => details.children[0].textContent).sort();
  assert.deepEqual(labels, [
    "Family Role: primary, Inference: always valid, Multiplicity: disabled",
    "Family Role: secondary, Inference: always valid, Multiplicity: enabled",
  ]);
});

test("opening a known interaction reveals its own colored Runtime/Family, alternative, and evidence", () => {
  const fixture = loadFixture();
  fixture.sequentialQuantileButton.dispatch("click");
  const [, ...detailsElements] = fixture.drawer.querySelectorAll(".compatibility-interactions")[0].children;
  const secondaryDetails = detailsElements.find((details) =>
    details.children[0].textContent.includes("secondary")
  );
  const nestedList = secondaryDetails.children[1];
  const statusFields = nestedList.querySelectorAll("[data-status]");
  assert.equal(statusFields[0].getAttribute("data-status"), "limited");
  assert.equal(statusFields[0].textContent, "LIMITED");
  assert.equal(statusFields[1].getAttribute("data-status"), "excluded");
  assert.equal(statusFields[1].textContent, "EXCLUDED");
  assert.match(nestedList.textContent, /secondary family does not cover it/);
  assert.match(nestedList.textContent, /Use fixed-horizon quantile inference for family participation/);
});

test("clicking an unrelated cell stays at its baseline and reports no cataloged interactions", () => {
  const fixture = loadFixture();
  fixture.estimateMeanButton.dispatch("click");
  assert.match(fixture.drawer.textContent, /SUPPORTED/);
  const sections = fixture.drawer.querySelectorAll(".compatibility-interactions");
  assert.equal(sections.length, 1);
  const [heading, message] = sections[0].children;
  assert.equal(heading.textContent, "Known interactions");
  assert.equal(message.textContent, "No higher-order interactions are cataloged for this cell.");
  // Never the word UNKNOWN -- this cell has a real, classified baseline.
  assert.doesNotMatch(fixture.drawer.textContent, /UNKNOWN/);
});

test("clicking a pair with a cataloged scenario lists it as a native detail, labeled from its own context axes", () => {
  const fixture = loadFixture();
  fixture.clusterExportButton.dispatch("click");
  assert.match(fixture.drawer.textContent, /REFUSED/);
  const sections = fixture.drawer.querySelectorAll(".compatibility-interactions");
  assert.equal(sections.length, 1);
  const [heading, ...detailsElements] = sections[0].children;
  assert.equal(heading.textContent, "Known interactions");
  assert.equal(detailsElements.length, 1);
  const [summary, nestedList] = detailsElements[0].children;
  assert.equal(summary.textContent, "Inference: always valid");
  const statusFields = nestedList.querySelectorAll("[data-status]");
  assert.equal(statusFields[0].getAttribute("data-status"), "limited");
  assert.match(nestedList.textContent, /Synthetic pair-scenario explanation/);
});

test("clicking a pair with no cataloged scenario reports the same empty-state message as a matrix cell", () => {
  const fixture = loadFixture();
  fixture.cateEstimateButton.dispatch("click");
  assert.match(fixture.drawer.textContent, /REFUSED/);
  const sections = fixture.drawer.querySelectorAll(".compatibility-interactions");
  assert.equal(sections.length, 1);
  const [heading, message] = sections[0].children;
  assert.equal(heading.textContent, "Known interactions");
  assert.equal(message.textContent, "No higher-order interactions are cataloged for this cell.");
  assert.doesNotMatch(fixture.drawer.textContent, /UNKNOWN/);
});

test("switching the selection replaces the previous Known interactions section instead of appending to it", () => {
  const fixture = loadFixture();
  fixture.sequentialQuantileButton.dispatch("click");
  assert.equal(fixture.drawer.querySelectorAll(".compatibility-interactions")[0].children.length, 3);
  fixture.estimateMeanButton.dispatch("click");
  const sections = fixture.drawer.querySelectorAll(".compatibility-interactions");
  assert.equal(sections.length, 1);
  assert.equal(sections[0].children.length, 2);
});

test("compatibility.js marks the root enhanced once it successfully initializes", () => {
  const fixture = loadFixture();
  assert.equal(fixture.root.classList.contains("compatibility-js-enhanced"), true);
});

test("the enhancement class is never added when required DOM anchors are missing", () => {
  const fixture = loadFixtureWithoutDataNode();
  assert.equal(fixture.root.classList.contains("compatibility-js-enhanced"), false);
});

test("compatibility.js treats the drawer as a required anchor, atomically with root/dataNode, before wiring anything", () => {
  const fixture = loadFixtureWithoutDrawer();
  assert.equal(fixture.root.classList.contains("compatibility-js-enhanced"), false);
  assert.doesNotThrow(() => fixture.sequentialQuantileButton.dispatch("click"));
  assert.equal(fixture.sequentialQuantileButton.getAttribute("aria-current"), null);
  assert.equal(fixture.sequentialQuantileButton.textContent, "SUPPORTED");
});

test("displayStatus reads the payload's status_labels mapping, not a client-side reimplementation", () => {
  const fixture = loadFixture({
    status_labels: { supported: "AVAILABLE", limited: "LIMITED", refused: "REFUSED", not_applicable: "NOT APPLICABLE", unknown: "UNKNOWN", participates: "PARTICIPATES", excluded: "EXCLUDED" },
  });
  fixture.sequentialQuantileButton.dispatch("click");
  assert.match(fixture.drawer.textContent, /AVAILABLE/);
});
