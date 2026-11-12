import { app } from "../../scripts/app.js";

const NODE = "PROMPTFORGE_ImageAction";

app.registerExtension({
  name: "PromptForge.ImageAction",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const widget = (name) => this.widgets.find((item) => item.name === name);
      const category = widget("category");
      const action = widget("action");
      if (!category || !action) return;
      const all = [...action.options.values];
      const groups = {};
      for (const label of all) {
        const prefix = label.split(" · ")[0];
        (groups[prefix] ||= []).push(label);
      }
      const apply = () => {
        const wanted = (groups[category.value] || []);
        action.options.values = wanted.length ? wanted : all;
        if (!wanted.includes(action.value)) action.value = action.options.values[0];
        this.setDirtyCanvas(true, true);
      };
      this.promptforgeApplyActionFilter = apply;
      const callback = category.callback;
      category.callback = function () { callback?.apply(this, arguments); apply(); };
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
