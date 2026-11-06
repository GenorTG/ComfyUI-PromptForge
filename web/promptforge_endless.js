import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ENDLESS = "PROMPTFORGE_H3Endless";
const OPTIONS = "PROMPTFORGE_H3Options";
// prompting lives on the Endless node; an Options node from an older workflow keeps these values but does not show them
const MOVED = ["parts", "part_ideas", "negative_prompt", "dialogue", "dialogue_language", "dialogue_language_other", "preset_style", "seconds_per_part"];
const BOX_HEIGHT = 96;
const LABEL_HEIGHT = 18;

function el(tag, style = {}, props = {}) {
  const element = Object.assign(document.createElement(tag), props);
  Object.assign(element.style, style);
  return element;
}

// a text widget the user can make taller or shorter by dragging its corner; the height is kept in the workflow (node.properties)
function makeResizable(node, widget, key, fallback) {
  const input = widget?.inputEl || widget?.element;
  if (!input) return;
  node.properties = node.properties || {};
  const heights = (node.properties.heights = node.properties.heights || {});
  widget.options = widget.options || {};
  widget.options.getMinHeight = () => heights[key] || fallback;
  input.style.resize = "vertical";
  let settling = false;
  new ResizeObserver(() => {
    const height = Math.round(input.offsetHeight);
    if (settling || !height || Math.abs(height - (heights[key] || fallback)) < 4) return;
    heights[key] = height;
    settling = true;
    node.setSize([node.size[0], node.computeSize()[1]]);
    setTimeout(() => { settling = false; }, 80);
  }).observe(input);
}

function hide(widget) {
  widget.hidden = true;
  widget.computeSize = () => [0, -4];
  widget.options = { ...(widget.options || {}), hidden: true };
  if (widget.element) widget.element.style.display = "none";
}

// the prompt of a part, waiting for you: edit it, approve it, or ask for another one
let reviewPanel = null;
function closeReview() { reviewPanel?.remove(); reviewPanel = null; }
function showReview(detail) {
  closeReview();
  const send = (action, text) => { api.fetchApi("/promptforge/review", { method: "POST", body: JSON.stringify({ node: detail.node, action, text }) }); closeReview(); };
  reviewPanel = el("div", { position: "fixed", right: "16px", bottom: "16px", width: "min(620px, 92vw)", zIndex: 10010, display: "flex", flexDirection: "column", gap: "6px", padding: "10px", boxSizing: "border-box", background: "var(--comfy-menu-bg, #222)", color: "var(--input-text, #ddd)", border: "2px solid #4a90e2", borderRadius: "8px", boxShadow: "0 6px 24px rgba(0,0,0,.6)", fontSize: "13px" });
  reviewPanel.addEventListener("pointerdown", (event) => event.stopPropagation());
  const box = el("textarea", { width: "100%", height: "260px", minHeight: "80px", resize: "vertical", boxSizing: "border-box", padding: "6px 8px", fontSize: "12px", borderRadius: "4px", border: "1px solid var(--border-color, #555)", background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #ddd)" }, { value: detail.text });
  const buttons = el("div", { display: "flex", gap: "6px", flexWrap: "wrap" });
  const button = (label, action, accent) => {
    const element = el("button", { cursor: "pointer", padding: "4px 12px", borderRadius: "4px", border: "1px solid var(--border-color, #555)", background: accent ? "#2a5d9a" : "var(--comfy-input-bg, #333)", color: accent ? "#fff" : "var(--input-text, #ddd)" }, { textContent: label });
    element.addEventListener("click", () => send(action, box.value));
    return element;
  };
  buttons.append(button("Approve", "approve", true), button("Regenerate", "regenerate"), button("Approve this and all following", "approve_all"));
  reviewPanel.append(el("div", { fontWeight: "bold" }, { textContent: `Part ${detail.index + 1} of ${detail.total}: approve the prompt${detail.attempt ? ` (version ${detail.attempt + 1})` : ""}` }),
    el("div", { opacity: ".65", fontSize: "11px" }, { textContent: "Edit the text if you like; Approve uses what is in the box. Regenerate writes a new prompt for this part." }), box, buttons);
  document.body.append(reviewPanel);
}

app.registerExtension({
  name: "PromptForge.H3Endless",
  setup() {
    api.addEventListener("promptforge.review", (event) => showReview(event.detail));
    for (const name of ["execution_success", "execution_error", "execution_interrupted"]) api.addEventListener(name, closeReview);
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === OPTIONS) {
      const onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onCreated?.apply(this, arguments);
        const node = this;
        for (const name of MOVED) {
          const widget = node.widgets.find((entry) => entry.name === name);
          if (widget) hide(widget);
        }
        node.setSize([node.size[0], node.computeSize()[1]]);
      };
      // a node saved before options were added is too short to show them: grow it to fit
      const onConfigureOptions = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const result = onConfigureOptions?.apply(this, arguments);
        this.setSize([this.size[0], Math.max(this.size[1], this.computeSize()[1])]);
        return result;
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
      for (const [name, fallback] of [["instructions", 90], ["negative_prompt", 60], ["dialogue", 60]]) makeResizable(node, find(name), name, fallback);

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

      node.properties = node.properties || {};
      const heights = () => (node.properties.partHeights = node.properties.partHeights || []);
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
          const box = el("textarea", { width: "100%", height: (heights()[index] || BOX_HEIGHT) + "px", minHeight: "48px", resize: "vertical", boxSizing: "border-box", padding: "4px 6px", fontSize: "12px", borderRadius: "4px", border: "1px solid var(--border-color, #555)", background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #ddd)" },
            { value: texts[index] || "", placeholder: `Part ${index + 1}: what happens in this part` });
          box.addEventListener("input", () => { texts[index] = box.value; save(); });
          box.addEventListener("wheel", (event) => event.stopPropagation());
          let settling = false;
          new ResizeObserver(() => {
            const height = Math.round(box.offsetHeight);
            if (settling || !height || Math.abs(height - (heights()[index] || BOX_HEIGHT)) < 4) return;
            heights()[index] = height;
            settling = true;
            fit();
            setTimeout(() => { settling = false; }, 80);
          }).observe(box);
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

      const widget = node.addDOMWidget("part_boxes", "promptforge_part_boxes", frame, { serialize: false, getMinHeight: () => 34 + Array.from({ length: total() }, (_, index) => (heights()[index] || BOX_HEIGHT) + LABEL_HEIGHT + 4).reduce((sum, value) => sum + value, 0) });
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
