import { app } from "../../scripts/app.js";

async function post(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function widgetValues(node) {
  const values = {};
  for (const widget of node.widgets || []) {
    if (widget.name && widget.value !== undefined) values[widget.name] = widget.value;
  }
  return values;
}

function setWidgets(node, values) {
  for (const [name, value] of Object.entries(values)) {
    const widget = (node.widgets || []).find((item) => item.name === name);
    if (widget) widget.value = value;
  }
  node.setDirtyCanvas(true, true);
}

function showReport(text) {
  if (app.ui?.dialog?.show) {
    app.ui.dialog.show(text);
  } else {
    alert(text);
  }
}

app.registerExtension({
  name: "PromptForge.TestLoad",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!["PROMPTFORGE_Model", "PROMPTFORGE_ConfigFinder"].includes(nodeData.name)) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const isModel = nodeData.name === "PROMPTFORGE_Model";
      this.addWidget("button", isModel ? "Test load & apply" : "Scan local models", null, async () => {
        try {
          const body = widgetValues(this);
          if (isModel) {
            showReport("Loading the selected model to verify the settings...");
            const result = await post("/promptforge/test", body);
            if (result.ok) {
              setWidgets(this, result.settings || {});
            }
            showReport((result.ok ? "Verified. " : "Failed. ") + (result.message || ""));
          } else {
            const result = await post("/promptforge/scan", body);
            showReport(result.report);
          }
        } catch (error) {
          showReport("PromptForge: " + (error?.message || error));
        }
      });
    };
  },
});
