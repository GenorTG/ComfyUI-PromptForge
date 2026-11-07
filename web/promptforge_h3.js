import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { addLivePreview } from "./promptforge_preview.js";

const IDEA = "PROMPTFORGE_H3Idea";
const PARTS = "PROMPTFORGE_H3Parts";
const MODES = "PROMPTFORGE_H3Modes";
const SECOND_STAGE = "PROMPTFORGE_H3SecondStage";
const GENERATE = "PROMPTFORGE_H3Generate";
const REFERENCES = "PROMPTFORGE_H3References";
// the description box of a slot is shown only while its picture / video / sound is connected and the description is not itself wired
const DESCRIPTIONS = { description_: "image_", video_description_: "video_", audio_description_: "audio_" };
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
  if (!("promptforgeSize" in widget)) widget.promptforgeSize = widget.computeSize || null;  // kept once; a widget may have no computeSize at all
  widget.hidden = !shown;
  if (widget.promptforgeSize) widget.computeSize = shown ? widget.promptforgeSize : () => [0, -4];
  if (widget.element) widget.element.style.display = shown ? "" : "none";
}


const IDEA_KEYS = ["general_prompt", "negative_prompt", "dialogue"];
const BUTTON = { cursor: "pointer", padding: "2px 8px" };
const INPUT = { background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #ddd)", border: "1px solid var(--border-color, #555)", borderRadius: "3px" };

// a DOM widget that is as wide as its node, is not saved in the workflow and is as tall as `height()`
function domWidget(node, name, element, height) {
  element.addEventListener("pointerdown", (event) => event.stopPropagation());
  element.addEventListener("wheel", (event) => event.stopPropagation());
  const widget = node.addDOMWidget(name, "promptforge_" + name, element, { serialize: false, getMinHeight: height });
  Object.defineProperty(widget, "width", { get: () => node.size[0], set: () => {}, configurable: true });
  return widget;
}

// the saved ideas of the control panel: your own texts, before the prompt writer turns them into prompts
function addIdeas(node, put) {
  const find = (name) => node.widgets.find((widget) => widget.name === name);
  const bar = el("div", { display: "flex", alignItems: "center", gap: "4px", boxSizing: "border-box", fontSize: "12px", color: "var(--input-text, #ddd)" });
  const select = el("select", { ...INPUT, flex: "1 1 auto", minWidth: "0", height: "24px" });
  const previous = el("button", BUTTON, { textContent: "◀", title: "previous idea" });
  const next = el("button", BUTTON, { textContent: "▶", title: "next idea" });
  const save = el("button", BUTTON, { textContent: "Save", title: "store the idea that is written now under the selected name" });
  const saveAs = el("button", BUTTON, { textContent: "Save as…", title: "store it under a new name" });
  const remove = el("button", BUTTON, { textContent: "Delete", title: "delete the selected idea" });
  bar.append(el("span", { opacity: ".7" }, { textContent: "Ideas" }), previous, select, next, save, saveAs, remove);
  let ideas = {};

  const refresh = (chosen) => {
    select.replaceChildren(el("option", {}, { value: "", textContent: "— saved ideas —" }), ...Object.keys(ideas).map((name) => el("option", {}, { value: name, textContent: name })));
    select.value = chosen && ideas[chosen] ? chosen : "";
  };
  const load = async (chosen) => { ideas = await (await api.fetchApi("/promptforge/ideas")).json(); refresh(chosen ?? select.value); };
  const current = () => Object.fromEntries(IDEA_KEYS.map((key) => [key, find(key)?.value]));
  const store = async (name) => {
    ideas = await (await api.fetchApi("/promptforge/ideas", { method: "POST", body: JSON.stringify({ name, data: current() }) })).json();
    refresh(name);
  };
  const apply = (name) => {
    const data = ideas[name];
    if (!data) return;
    for (const key of IDEA_KEYS) if (key in data && find(key)) find(key).value = data[key];
    node.promptforgeRestoreParts?.();
    find("generation")?.callback?.(find("generation").value);
    node.setDirtyCanvas(true, true);
  };
  const step = (by) => {
    const names = Object.keys(ideas);
    if (!names.length) return;
    const index = (names.indexOf(select.value) + by + names.length) % names.length;
    refresh(names[index]);
    apply(names[index]);
  };
  select.addEventListener("change", () => apply(select.value));
  previous.addEventListener("click", () => step(-1));
  next.addEventListener("click", () => step(1));
  save.addEventListener("click", () => { const name = select.value || window.prompt("Name of this idea:"); if (name?.trim()) store(name.trim()); });
  saveAs.addEventListener("click", () => { const name = window.prompt("Name of this idea:", select.value); if (name?.trim()) store(name.trim()); });
  remove.addEventListener("click", async () => {
    if (!select.value || !window.confirm(`Delete the idea "${select.value}"?`)) return;
    ideas = await (await api.fetchApi("/promptforge/ideas?name=" + encodeURIComponent(select.value), { method: "DELETE" })).json();
    refresh("");
  });
  put(domWidget(node, "saved_ideas", bar, () => 30));
  load("");
}

// which references are on, and a switch for each: the Load nodes keep the files, this is where you flip them
const REFERENCE_SLOTS = [["first_frame", "First frame"], ["last_frame", "Last frame"], ...Array.from({ length: 9 }, (_, index) => [`image_${index + 1}`, `Picture ${index + 1}`]),
  ...Array.from({ length: 3 }, (_, index) => [`video_${index + 1}`, `Video ${index + 1}`]), ...Array.from({ length: 3 }, (_, index) => [`audio_${index + 1}`, `Sound ${index + 1}`])];

// which modes use which slot
const USED_BY = (name) => (name === "first_frame" ? ["I2V", "FL2V", "REF2V"] : name === "last_frame" ? ["L2V", "FL2V"] : name === "extend" ? ["T2V", "I2V", "L2V", "FL2V", "REF2V"] : ["REF2V"]);

function addReferenceOverview(node, put) {
  const graph = () => node.graph;
  const linkOf = (id) => (graph().getLink ? graph().getLink(id) : graph().links[id]);
  const sourceOf = (target, name) => {
    const link = target?.inputs?.find((input) => input.name === name)?.link;
    return link == null ? null : graph().getNodeById(linkOf(link)?.origin_id);
  };
  const generator = () => node.outputs.flatMap((output) => output.links || []).map((id) => graph().getNodeById(linkOf(id)?.target_id)).find((target) => target?.type === GENERATE);
  const frame = el("div", { boxSizing: "border-box", fontSize: "12px", color: "var(--input-text, #ddd)", border: "1px solid var(--border-color, #555)", borderRadius: "4px", padding: "4px 6px", overflow: "auto" });
  let shown = "";

  const rows = () => {
    const target = generator();
    const collect = target ? sourceOf(target, "references") : null;
    const slots = REFERENCE_SLOTS.map(([name, label]) => [label, sourceOf(collect, name), name]);
    slots.push(["Video to extend", sourceOf(sourceOf(target, "continue_from"), "video"), "extend"]);
    return slots.filter(([, loader]) => loader);
  };
  const render = () => {
    if (!graph()) return;
    const slots = rows();
    const kind = String(node.widgets.find((widget) => widget.name === "mode")?.value || "T2V").split(" ")[0];
    const extending = !!node.widgets.find((widget) => widget.name === "extend_existing_video")?.value;
    const state = kind + extending + JSON.stringify(slots.map(([label, loader]) => [label, loader.title, loader.widgets?.map((widget) => widget.value).filter((value) => typeof value === "string").join("|"), loader.widgets?.find((widget) => widget.name === "enabled")?.value]));
    if (state === shown) return;
    shown = state;
    const used = slots.filter(([, loader]) => loader.widgets.find((widget) => ["image", "video", "audio"].includes(widget.name))?.value);
    const lines = used.map(([label, loader, name]) => {
      const file = loader.widgets.find((widget) => ["image", "video", "audio"].includes(widget.name))?.value || "";
      const toggle = loader.widgets.find((widget) => widget.name === "enabled");
      const line = el("label", { display: "flex", alignItems: "center", gap: "6px", cursor: "pointer", opacity: toggle?.value && (name === "extend" ? extending : USED_BY(name).includes(kind)) ? "1" : ".45", whiteSpace: "nowrap", overflow: "hidden" });
      const box = el("input", {}, { type: "checkbox", checked: !!toggle?.value });
      box.addEventListener("change", () => { toggle.value = box.checked; loader.setDirtyCanvas?.(true, true); node.setDirtyCanvas(true, true); shown = ""; });
      line.append(box, el("b", { minWidth: "96px" }, { textContent: label }), el("span", { overflow: "hidden", textOverflow: "ellipsis" }, { textContent: (file.split("/").pop() || "(no file)") + (name === "extend" ? (extending ? "" : "  (extend is off)") : USED_BY(name).includes(kind) ? "" : `  (not used in ${kind})`) }));
      return line;
    });
    const empty = slots.length - used.length;
    frame.replaceChildren(el("div", { opacity: ".7" }, { textContent: lines.length ? `Files chosen in the Load nodes (tick = on) - mode ${kind}` : "No files chosen: pick them in the Load nodes" }), ...lines,
      ...(slots.length && empty ? [el("div", { opacity: ".5" }, { textContent: `${empty} slot${empty > 1 ? "s" : ""} empty` })] : []));
    node.setSize([node.size[0], Math.max(node.size[1], node.computeSize()[1])]);
  };
  put(domWidget(node, "references_overview", frame, () => Math.min(300, 14 + 20 * Math.max(2, frame.childElementCount))));
  const timer = setInterval(render, 700);
  const removed = node.onRemoved;
  node.onRemoved = function () { clearInterval(timer); return removed?.apply(this, arguments); };
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

    if (nodeData.name === REFERENCES) {
      const refresh = (node) => {
        const linked = (name) => node.inputs?.find((input) => input.name === name)?.link != null;
        for (const widget of node.widgets) {
          const prefix = Object.keys(DESCRIPTIONS).find((key) => widget.name.startsWith(key));
          if (!prefix) continue;
          const number = widget.name.slice(prefix.length);
          setShown(widget, linked(DESCRIPTIONS[prefix] + number) && !linked(widget.name));
        }
        node.setSize([node.size[0], node.computeSize()[1]]);
        node.setDirtyCanvas(true, true);
      };
      const onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () { onCreated?.apply(this, arguments); refresh(this); };
      const onConnections = nodeType.prototype.onConnectionsChange;
      nodeType.prototype.onConnectionsChange = function () { const result = onConnections?.apply(this, arguments); refresh(this); return result; };
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () { const result = onConfigure?.apply(this, arguments); refresh(this); return result; };
      return;
    }

    if (nodeData.name === GENERATE) {
      const onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onCreated?.apply(this, arguments);
        const node = this;
        addLivePreview(node);

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

    if (nodeData.name === IDEA) {
      const onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onCreated?.apply(this, arguments);
        const node = this;
        const find = (name) => node.widgets.find((widget) => widget.name === name);
        for (const [name, fallback] of [["general_prompt", 150], ["negative_prompt", 60], ["dialogue", 60]]) makeResizable(node, find(name), name, fallback);
        const general = find("general_prompt");
        addIdeas(node, (added) => { node.widgets.splice(node.widgets.indexOf(added), 1); node.widgets.splice(node.widgets.indexOf(general), 0, added); });
        // the camera and the people only matter with an action; the style only while the template is automatic
        node.promptforgeRefreshIdea = () => {
          const action = String(find("action").value);
          const chosen = action !== "(none)";
          setShown(find("camera_angle"), chosen);
          setShown(find("person_a"), chosen);
          setShown(find("person_b"), chosen && !action.startsWith("Solo"));
          setShown(find("prompt_style"), String(find("prompt_template").value).startsWith("automatic") && !chosen);
          node.setSize([node.size[0], node.computeSize()[1]]);
          node.setDirtyCanvas(true, true);
        };
        for (const name of ["action", "prompt_template"]) {
          const widget = find(name);
          const callback = widget.callback;
          widget.callback = (...args) => { callback?.apply(widget, args); node.promptforgeRefreshIdea(); };
        }
        node.promptforgeRefreshIdea();
        node.setSize([Math.max(node.size[0], 560), Math.max(node.size[1], node.computeSize()[1])]);
      };
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () { const result = onConfigure?.apply(this, arguments); this.promptforgeRefreshIdea?.(); return result; };
      return;
    }

    if (nodeData.name === MODES) {
      const onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onCreated?.apply(this, arguments);
        addReferenceOverview(this, (added) => { this.widgets.splice(this.widgets.indexOf(added), 1); this.widgets.unshift(added); });
        this.setSize([Math.max(this.size[0], 520), Math.max(this.size[1], this.computeSize()[1])]);
      };
      return;
    }

    if (nodeData.name !== PARTS) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const node = this;
      const find = (name) => node.widgets.find((widget) => widget.name === name);
      const countWidget = find("parts");
      const store = find("part_prompts");
      hide(countWidget);
      hide(store);

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

      node.setSize([Math.max(node.size[0], 560), Math.max(node.size[1], node.computeSize()[1])]);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      this.promptforgeRestoreParts?.();
      return result;
    };
  },
});
