import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { addLivePreview } from "./promptforge_preview.js";

const STORY = "PROMPTFORGE_H3Story";
const SECOND_STAGE = "PROMPTFORGE_H3SecondStage";
const GENERATE = "PROMPTFORGE_H3Generate";
const SIZE_WIDGETS = { "scale by multiplier": ["scale"], megapixels: ["megapixels"], "target dimensions": ["width", "height"] };
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

function setShown(widget, shown) {
  if (!widget) return;
  if (widget.promptforgeSize === undefined) widget.promptforgeSize = widget.computeSize;
  widget.hidden = !shown;
  widget.computeSize = shown ? widget.promptforgeSize : () => [0, -4];
  if (widget.element) widget.element.style.display = shown ? "" : "none";
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
  name: "PromptForge.H3",
  setup() {
    api.addEventListener("promptforge.review", (event) => showReview(event.detail));
    for (const name of ["execution_success", "execution_error", "execution_interrupted"]) api.addEventListener(name, closeReview);
    const node = (event) => app.graph.getNodeById(Number(event.detail.node));
    api.addEventListener("promptforge.prompt", (event) => node(event)?.promptforgeShowPrompt?.(event.detail));
    api.addEventListener("promptforge.part", (event) => node(event)?.promptforgeShowPart?.(event.detail));
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === SECOND_STAGE) {
      // only the size setting that matches the size mode is shown
      const onCreated = nodeType.prototype.onNodeCreated;
      const refresh = (node) => {
        const wanted = SIZE_WIDGETS[node.widgets.find((widget) => widget.name === "size_mode")?.value] || [];
        for (const name of Object.values(SIZE_WIDGETS).flat()) setShown(node.widgets.find((widget) => widget.name === name), wanted.includes(name));
        node.setSize([node.size[0], node.computeSize()[1]]);
        node.setDirtyCanvas(true, true);
      };
      nodeType.prototype.onNodeCreated = function () {
        onCreated?.apply(this, arguments);
        const mode = this.widgets.find((widget) => widget.name === "size_mode");
        const callback = mode.callback;
        mode.callback = (...args) => { callback?.apply(mode, args); refresh(this); };
        refresh(this);
      };
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        refresh(this);
        return result;
      };
      return;
    }

    if (nodeData.name === GENERATE) {
      const onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onCreated?.apply(this, arguments);
        const node = this;
        addLivePreview(node);

        // the finished part, as a short clip (it is not saved anywhere)
        const holder = el("div", { position: "relative", width: "100%", height: "100%", overflow: "hidden", display: "none" });
        const video = el("video", { position: "absolute", inset: "0", width: "100%", height: "100%", objectFit: "contain", background: "rgba(0,0,0,.25)", borderRadius: "4px" }, { controls: true, muted: true, loop: true, autoplay: true, playsInline: true });
        const caption = el("div", { position: "absolute", left: "6px", top: "4px", padding: "1px 6px", fontSize: "11px", borderRadius: "3px", background: "rgba(0,0,0,.55)", color: "#fff", zIndex: 1 });
        holder.append(video, caption);
        holder.addEventListener("pointerdown", (event) => event.stopPropagation());
        let partShown = false;
        const part = node.addDOMWidget("finished_part", "promptforge_finished_part", holder, { serialize: false, getMinHeight: () => (partShown ? 240 : 0) });
        Object.defineProperty(part, "width", { get: () => node.size[0], set: () => {}, configurable: true });
        node.promptforgeShowPart = ({ index, mime, image }) => {
          node.promptforgeLiveHide?.();
          if (!partShown) { partShown = true; holder.style.display = ""; node.setSize([node.size[0], node.computeSize()[1]]); }
          caption.textContent = `part ${index + 1} finished`;
          video.src = `data:${mime};base64,${image}`;
          video.play().catch(() => {});
        };

        // every part's prompt, added as soon as it has been written
        const box = el("textarea", { width: "100%", height: "100%", resize: "none", boxSizing: "border-box", padding: "6px 8px", fontSize: "12px", borderRadius: "4px", border: "1px solid var(--border-color, #555)", background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #ddd)" },
          { readOnly: true, placeholder: "The prompt written for each part appears here as it is made." });
        box.addEventListener("wheel", (event) => event.stopPropagation());
        box.addEventListener("pointerdown", (event) => event.stopPropagation());
        const wrapper = el("div", { width: "100%", height: "100%" });
        wrapper.append(box);
        const prompts = node.addDOMWidget("written_prompts", "promptforge_written_prompts", wrapper, { serialize: false, getMinHeight: () => 200 });
        Object.defineProperty(prompts, "width", { get: () => node.size[0], set: () => {}, configurable: true });
        node.promptforgeShowPrompt = ({ index, text }) => {
          box.value = (index === 0 ? "" : box.value) + `=== Part ${index + 1} ===\n${text}\n\n`;
          box.scrollTop = box.scrollHeight;
        };
        node.setSize([Math.max(node.size[0], 420), node.computeSize()[1]]);
      };
      return;
    }

    if (nodeData.name !== STORY) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const node = this;
      const find = (name) => node.widgets.find((widget) => widget.name === name);
      const general = find("general_prompt");
      const countWidget = find("parts");
      const store = find("part_prompts");
      hide(countWidget);
      hide(store);
      for (const [name, fallback] of [["general_prompt", 90], ["negative_prompt", 60], ["dialogue", 60]]) makeResizable(node, find(name), name, fallback);

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
