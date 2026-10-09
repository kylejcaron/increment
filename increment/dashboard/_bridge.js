// Hosts the dashboard page in an iframe and relays the Explore picker to Python.
//
// The page posts {source: "inc-dashboard", type: "ready" | "set-exploratory-metrics" | "set-population"}.
// This widget answers {source: "inc-dashboard-host", type: "bridge" | "status" | "restore"}.
// Python re-renders the same snapshot when `exploratory_metrics` changes and replaces
// `document`; the replacement page restores the active population and reopens Explore.
function render({ model, el }) {
  const frame = document.createElement("iframe");
  frame.className = "inc-dashboard-frame";
  frame.title = "Experiment dashboard";
  let reopen = null;
  let selectedPopulation = null;

  const post = (message) => {
    if (frame.contentWindow) {
      frame.contentWindow.postMessage({ source: "inc-dashboard-host", ...message }, "*");
    }
  };

  const onMessage = (event) => {
    if (event.source !== frame.contentWindow) return;
    const message = event.data;
    if (!message || message.source !== "inc-dashboard") return;
    if (message.type === "ready") {
      post({ type: "bridge" });
      const restore = { type: "restore" };
      if (reopen) restore.view = reopen;
      if (selectedPopulation) restore.population = selectedPopulation;
      if (reopen || selectedPopulation) post(restore);
      reopen = null;
    } else if (message.type === "set-exploratory-metrics" && Array.isArray(message.metrics)) {
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
    frame.srcdoc = model.get("document");
  });

  frame.srcdoc = model.get("document");
  el.appendChild(frame);
  return () => window.removeEventListener("message", onMessage);
}

export default { render };
