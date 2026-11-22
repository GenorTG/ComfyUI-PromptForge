import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "PROMPTFORGE_ImageAction";
const RANDOM = "random";

app.registerExtension({
  name: "PromptForge.ImageAction",
  setup() {
    api.addEventListener("promptforge.random", (event) => {
      const node = app.graph?.getNodeById(Number(event.detail.node));
      node?.promptforgeShowRandom?.(event.detail);
    });
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const widget = (name) => this.widgets.find((item) => item.name === name);
      const category = widget("category");
      const action = widget("action");
      if (!category || !action) return;
      const all = [...action.options.values].filter((value) => value !== RANDOM);
      const groups = {};
      for (const label of all) {
        const prefix = label.split(" · ")[0];
        (groups[prefix] ||= []).push(label);
      }
      const apply = () => {
        const wanted = category.value === RANDOM ? all : (groups[category.value] || []);
        action.options.values = [RANDOM, ...(wanted.length ? wanted : all)];
        if (action.value !== RANDOM && !wanted.includes(action.value)) action.value = wanted[0];
        this.setDirtyCanvas(true, true);
      };
      this.promptforgeApplyActionFilter = apply;
      const callback = category.callback;
      category.callback = function () { callback?.apply(this, arguments); apply(); };

      // what the server rolled on the last run; display only, the widgets stay on "random"
      const frame = document.createElement("div");
      frame.style.cssText = "padding:2px 6px;font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;color:var(--input-text,#ddd);background:var(--comfy-input-bg,#333);border:1px solid var(--border-color,#555);border-radius:4px";
      frame.textContent = "Random: run the prompt to roll the pickers set to random.";
      const box = this.addDOMWidget("random_overview", "promptforge_random", frame, { serialize: false, getMinHeight: () => 24 });
      Object.defineProperty(box, "width", { get: () => this.size[0], set: () => {}, configurable: true });
      this.promptforgeShowRandom = ({ picks }) => { frame.textContent = picks.length ? "Random rolled: " + picks.map(([name, value]) => `${name} = ${value}`).join(", ") : ""; };
      apply();
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      const width = this.size[0], height = this.size[1];
      // the saved category and action are applied around this hook, and the action list still holds the
      // default category until it runs: rebuild it from the saved category, then keep the saved action
      setTimeout(() => {
        this.promptforgeApplyActionFilter?.();
        this.setSize([Math.max(width, this.size[0]), Math.max(height, this.size[1])]);
      }, 0);
      return result;
    };
  },
});
