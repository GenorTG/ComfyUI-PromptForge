import { app } from "../../scripts/app.js";

const ENDLESS = "PROMPTFORGE_H3Endless";
const OPTIONS = "PROMPTFORGE_H3Options";
// prompting lives on the Endless node; an Options node from an older workflow keeps these values but does not show them
const MOVED = ["parts", "part_ideas", "negative_prompt", "dialogue", "dialogue_language", "dialogue_language_other", "preset_style"];
const BOX_HEIGHT = 96;
const LABEL_HEIGHT = 18;

function el(tag, style = {}, props = {}) {
  const element = Object.assign(document.createElement(tag), props);
  Object.assign(element.style, style);
  return element;
}

function hide(widget) {
  widget.hidden = true;
  widget.computeSize = () => [0, -4];
  widget.options = { ...(widget.options || {}), hidden: true };
  if (widget.element) widget.element.style.display = "none";
}

app.registerExtension({
  name: "PromptForge.H3Endless",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === OPTIONS) {
      const onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onCreated?.apply(this, arguments);
        for (const name of MOVED) {
          const widget = this.widgets.find((entry) => entry.name === name);
          if (widget) hide(widget);
        }
        this.setSize([this.size[0], this.computeSize()[1]]);
      };
      return;
    }
    if (nodeData.name !== ENDLESS) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const node = this;
      const find = (name) => node.widgets.find((widget) => widget.name === name);
      const general = find("instructions");
      const countWidget = find("parts");
      const store = find("part_prompts");
      hide(countWidget);
      hide(store);
      general.label = "general prompt";

      let texts = [];
      const frame = el("div", { position: "relative", width: "100%", height: "100%", minWidth: "0", overflow: "hidden", contain: "layout paint" });
      const root = el("div", { position: "absolute", inset: "0", display: "flex", flexDirection: "column", gap: "4px", boxSizing: "border-box", overflowY: "auto", fontSize: "12px", color: "var(--input-text, #ddd)" });
      frame.append(root);
      frame.addEventListener("pointerdown", (event) => event.stopPropagation());
      root.addEventListener("wheel", (event) => event.stopPropagation());

      const header = el("div", { display: "flex", alignItems: "center", gap: "6px", flex: "0 0 auto" });
      const minus = el("button", { cursor: "pointer", width: "24px" }, { textContent: "−" });
      const plus = el("button", { cursor: "pointer", width: "24px" }, { textContent: "+" });
      const count = el("input", { width: "52px", textAlign: "center", background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #ddd)", border: "1px solid var(--border-color, #555)", borderRadius: "3px" }, { type: "number", min: 1, max: 200, step: 1 });
      header.append("Parts", minus, count, plus, el("span", { opacity: ".6", marginLeft: "6px" }, { textContent: "What happens in each part. Start a box with (10s) for a longer part, or raw: to use your own finished prompt." }));
      const list = el("div", { display: "flex", flexDirection: "column", gap: "4px" });
      root.append(header, list);

      const total = () => Math.max(1, Number(countWidget.value) || texts.length || 3);
      const save = () => {
        texts.length = total();
        store.value = JSON.stringify(Array.from(texts, (text) => text || ""));
      };
      const fit = () => {
        node.setSize([node.size[0], node.computeSize()[1]]);
        node.setDirtyCanvas(true, true);
      };

      function render() {
        const wanted = total();
        count.value = wanted;
        while (texts.length < wanted) texts.push("");
        list.replaceChildren(...Array.from({ length: wanted }, (_, index) => {
          const block = el("div", { display: "flex", flexDirection: "column", gap: "2px" });
          const box = el("textarea", { width: "100%", height: BOX_HEIGHT + "px", minHeight: "48px", resize: "vertical", boxSizing: "border-box", padding: "4px 6px", fontSize: "12px", borderRadius: "4px", border: "1px solid var(--border-color, #555)", background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #ddd)" },
            { value: texts[index] || "", placeholder: `Part ${index + 1}: what happens in this part` });
          box.addEventListener("input", () => { texts[index] = box.value; save(); });
          box.addEventListener("wheel", (event) => event.stopPropagation());
          block.append(el("div", { opacity: ".7" }, { textContent: `Part ${index + 1}` }), box);
          return block;
        }));
      }

      const setCount = (value) => {
        countWidget.value = Math.min(200, Math.max(1, Math.round(value)));
        render();
        save();
        fit();
      };
      minus.addEventListener("click", () => setCount(total() - 1));
      plus.addEventListener("click", () => setCount(total() + 1));
      count.addEventListener("change", () => setCount(Number(count.value) || 1));

      const widget = node.addDOMWidget("part_boxes", "promptforge_part_boxes", frame, { serialize: false, getMinHeight: () => 34 + total() * (BOX_HEIGHT + LABEL_HEIGHT + 4) });
      Object.defineProperty(widget, "width", { get: () => node.size[0], set: () => {}, configurable: true });
      // the part boxes go right under the general prompt
      node.widgets.splice(node.widgets.indexOf(widget), 1);
      node.widgets.splice(node.widgets.indexOf(general) + 1, 0, widget);

      const restore = () => {
        try { texts = JSON.parse(store.value || "[]"); } catch { texts = []; }
        if (!Array.isArray(texts)) texts = [];
        if (!Number(countWidget.value)) countWidget.value = Math.max(texts.length, 3);
        render();
        fit();
      };
      node.promptforgeRestoreParts = restore;
      if (!Number(countWidget.value)) countWidget.value = 3;
      render();
      node.setSize([Math.max(node.size[0], 520), Math.max(node.size[1], node.computeSize()[1])]);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      this.promptforgeRestoreParts?.();
      return result;
    };
  },
});
