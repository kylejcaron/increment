// Hosts the dashboard page in an iframe and relays the Explore picker to Python.
//
// The page posts {source: "inc-dashboard", type: "ready" | "set-exploratory-metrics" | "set-population"}.
// This widget answers {source: "inc-dashboard-host", type: "bridge" | "status" | "restore"}.
// Python re-renders the same snapshot when `exploratory_metrics` changes and replaces
// `document`; the replacement page restores the population and any active interaction selection.
function render({ model, el }) {
  const frame = document.createElement("iframe");
  frame.className = "inc-dashboard-frame";
  frame.title = "Experiment dashboard";
  let reopen = null;
  let selectedPopulation = null;
  let interactionSelection = null;
  let peer = null;

  const post = (message) => {
    if (peer) {
      peer.postMessage({ source: "inc-dashboard-host", ...message }, "*");
    }
  };

  const onMessage = (event) => {
    if (event.source !== frame.contentWindow) {
      // Shadow hosts can retarget WindowProxy identity; the exact iframe remains authoritative.
      try { if (event.source?.frameElement !== frame) return; } catch (error) { return; }
    }
    const message = event.data;
    if (!message || message.source !== "inc-dashboard") return;
    if (message.type === "ready") {
      peer = event.source;
      post({ type: "bridge" });
      const restore = { type: "restore" };
      if (reopen) restore.view = reopen;
      if (selectedPopulation) restore.population = selectedPopulation;
      if (reopen && interactionSelection) restore.interaction = interactionSelection;
      if (reopen || selectedPopulation) post(restore);
      interactionSelection = null;
      reopen = null;
    } else if (message.type === "set-exploratory-metrics" && Array.isArray(message.metrics)) {
      const interaction = message.interaction;
      const asString = (value) => typeof value === "string" ? value : null;
      interactionSelection = interaction && typeof interaction === "object" ? {
        metric: asString(interaction.metric),
        pair: asString(interaction.pair),
        scale: asString(interaction.scale),
        focus: asString(interaction.focus),
        focusPair: asString(interaction.focusPair),
        combination: asString(interaction.combination),
        combinationQuestion: asString(interaction.combinationQuestion),
        combinationFilter: asString(interaction.combinationFilter),
        combinationOpen: interaction.combinationOpen === true,
      } : null;
      model.set("exploratory_metrics", message.metrics.map(String));
      model.save_changes();
    } else if (message.type === "set-population" && typeof message.population === "string") {
      selectedPopulation = message.population;
    }
  };
  window.addEventListener("message", onMessage);

  model.on("change:status", () => post({ type: "status", status: model.get("status") }));
  model.on("change:document", () => {
    reopen = "explore";
    peer = null;
    frame.srcdoc = model.get("document");
  });

  frame.srcdoc = model.get("document");
  el.appendChild(frame);
  return () => {
    peer = null;
    window.removeEventListener("message", onMessage);
  };
}

export default { render };
