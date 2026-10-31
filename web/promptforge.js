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
    app.ui.dialog.element?.scrollTo?.(0, 0);
  } else {
    alert(text);
  }
}

app.registerExtension({
  name: "PromptForge.AutoConfig",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!["PROMPTFORGE_Model", "PROMPTFORGE_ConfigFinder"].includes(nodeData.name)) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const isModel = nodeData.name === "PROMPTFORGE_Model";
      this.addWidget("button", isModel ? "Find best config" : "Scan local models", null, async () => {
        try {
          const body = widgetValues(this);
          if (isModel) {
            const values = await post("/promptforge/recommend", body);
            const auto = (this.widgets || []).find((item) => item.name === "auto_config");
            if (auto) auto.value = false;
            setWidgets(this, values);
            showReport(
              `Applied: ${values.source} | ${values.quantization} | ctx ${values.context_size}` +
                ` | gpu_layers ${values.gpu_layers} | kv ${values.kv_cache_type}`,
            );
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
