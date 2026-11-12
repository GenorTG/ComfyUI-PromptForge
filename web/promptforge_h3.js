import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { addLivePreview } from "./promptforge_preview.js";

const IDEA = "PROMPTFORGE_H3Idea";
const ACTION = "PROMPTFORGE_H3Action";
const WRITER = "PROMPTFORGE_H3Writer";
const PARTS = "PROMPTFORGE_H3Parts";
const MODES = "PROMPTFORGE_H3Modes";
const SECOND_STAGE = "PROMPTFORGE_H3SecondStage";
const GENERATE = "PROMPTFORGE_H3Generate";
const REFERENCES = "PROMPTFORGE_H3References";
const LIBRARY = "PROMPTFORGE_PromptLibrary";
const SUBJECT_DISABLED = "disabled (not in the video)";
const ANY_CAMERA = "(writer's choice)";
const RANDOM = "random";
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
    // never below the height the node already has: a measuring pass right after loading must not shrink it
    node.setSize([node.size[0], Math.max(node.size[1], node.computeSize()[1])]);
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
    const source = String(node.widgets.find((widget) => widget.name === "prompt_source")?.value || "");
    const state = kind + extending + source + JSON.stringify(slots.map(([label, loader]) => [label, loader.title, loader.widgets?.map((widget) => widget.value).filter((value) => typeof value === "string").join("|"), loader.widgets?.find((widget) => widget.name === "enabled")?.value]));
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
    frame.replaceChildren(el("div", { fontWeight: "bold", marginBottom: "2px" }, { textContent: `The prompt is written from: ${source.startsWith("action") ? "the ACTION node (action preset)" : "the IDEA node (my text)"}  -  change it in prompt_source below` }), el("div", { opacity: ".7" }, { textContent: lines.length ? `Files chosen in the Load nodes (tick = on) - mode ${kind}` : "No files chosen: pick them in the Load nodes" }), ...lines,
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
    api.addEventListener("promptforge.request", (event) => node(event)?.promptforgeShowRequest?.(event.detail));
    api.addEventListener("promptforge.random", (event) => node(event)?.promptforgeShowRandom?.(event.detail));
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
        const width = this.size[0], height = this.size[1];
        refresh(this);
        // the saved size was applied before this hook; keep it instead of snapping back to the widget minimum
        this.setSize([Math.max(width, this.size[0]), Math.max(height, this.size[1])]);
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
      nodeType.prototype.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        const width = this.size[0], height = this.size[1];
        refresh(this);
        // the saved size was applied before this hook; keep it instead of snapping back to the widget minimum
        this.setSize([Math.max(width, this.size[0]), Math.max(height, this.size[1])]);
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

        // what the prompt writer was given, to read it yourself: mode, settings, pictures, your text and every instruction
        const given = el("textarea", { width: "100%", height: "100%", resize: "none", boxSizing: "border-box", padding: "6px 8px", fontSize: "11px", borderRadius: "4px", border: "1px solid var(--border-color, #555)", background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #ddd)", whiteSpace: "pre-wrap", overflowX: "hidden", overflowWrap: "anywhere" },
          { readOnly: true, placeholder: "What the prompt writer was given appears here: settings, pictures, your text and every instruction." });
        given.addEventListener("wheel", (event) => event.stopPropagation());
        given.addEventListener("pointerdown", (event) => event.stopPropagation());
        const givenWrapper = el("div", { width: "100%", height: "100%" });
        givenWrapper.append(given);
        const request = node.addDOMWidget("writer_input", "promptforge_writer_input", givenWrapper, { serialize: false, getMinHeight: () => 220 });
        Object.defineProperty(request, "width", { get: () => node.size[0], set: () => {}, configurable: true });
        node.promptforgeShowRequest = ({ index, text }) => {
          given.value = (index === 0 ? "" : given.value + "\n\n\n") + `##### PART ${index + 1} #####\n` + text;
        };

        // every part's prompt, added as soon as it has been written
        const box = el("textarea", { width: "100%", height: "100%", resize: "none", boxSizing: "border-box", padding: "6px 8px", fontSize: "12px", borderRadius: "4px", border: "1px solid var(--border-color, #555)", background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #ddd)", whiteSpace: "pre-wrap", overflowX: "hidden", overflowWrap: "anywhere" },
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
      const SAVED = "h3_idea_size";
      const onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onCreated?.apply(this, arguments);
        const node = this;
        const find = (name) => node.widgets.find((widget) => widget.name === name);
        for (const [name, fallback] of [["general_prompt", 150], ["negative_prompt", 60], ["dialogue", 60]]) makeResizable(node, find(name), name, fallback);
        const general = find("general_prompt");
        addIdeas(node, (added) => { node.widgets.splice(node.widgets.indexOf(added), 1); node.widgets.splice(node.widgets.indexOf(general), 0, added); });
        node.promptforgeRestoreIdeaSize = () => {
          const saved = node.properties?.[SAVED];
          node.setSize([Math.max(saved?.[0] || 560, 560), Math.max(saved?.[1] || node.size[1], node.computeSize()[1])]);
        };
        node.promptforgeRestoreIdeaSize();
      };
      const onResize = nodeType.prototype.onResize;
      nodeType.prototype.onResize = function (size) {
        onResize?.apply(this, arguments);
        this.properties = this.properties || {};
        this.properties[SAVED] = [Math.round(size[0]), Math.round(size[1])];
      };
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        const width = this.size[0], height = this.size[1];
        setTimeout(() => {
          this.promptforgeRestoreIdeaSize?.();
          // the saved size was applied before this hook; keep it instead of snapping back to the widget minimum
          this.setSize([Math.max(width, this.size[0]), Math.max(height, this.size[1])]);
        }, 0);
        return result;
      };
      return;
    }

    if (nodeData.name === ACTION) {
      // the role of every subject slot (woman / man / partner), its picture picker and the recommended frame
      const SAVED = "h3_action_size";
      const onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onCreated?.apply(this, arguments);
        const node = this;
        const find = (name) => node.widgets.find((widget) => widget.name === name);
        for (const name of ["notes", "negative_prompt", "dialogue", "subject_1_look", "subject_2_look", "subject_3_look"]) if (find(name)) makeResizable(node, find(name), name, 48);
        const label = el("div", { fontSize: "12px", padding: "3px 6px", color: "var(--input-text, #ddd)", opacity: ".9", lineHeight: "1.4" });
        const roles = node.addDOMWidget("subject_roles", "promptforge_subject_roles", label, { serialize: false, getMinHeight: () => 72 });
        Object.defineProperty(roles, "width", { get: () => node.size[0], set: () => {}, configurable: true });
        const action = find("action");
        const category = find("category");
        const penetration = find("penetration");
        const cameraWidget = find("camera_angle");
        const allActions = [...(action.options?.values || [])].filter((value) => value !== RANDOM);
        const allCameras = [...(cameraWidget?.options?.values || [])].filter((value) => value !== RANDOM && value !== ANY_CAMERA);
        node.widgets.splice(node.widgets.indexOf(roles), 1);
        node.widgets.splice(node.widgets.indexOf(action) + 1, 0, roles);
        const nameFor = (gender, index, list = []) => {
          if (gender === "female") {
            const women = list.map((item, at) => item.gender === "female" ? at : -1).filter((at) => at >= 0);
            return women.length > 1 ? `Woman ${women.indexOf(index) + 1}` : "Woman";
          }
          return gender === "male" ? "Man" : `Partner ${index + 1}`;
        };
        let known = null;
        const refresh = async () => {
          known = known || await (await api.fetchApi("/promptforge/actions")).json();
          const scattered = category.value === RANDOM || action.value === RANDOM;
          const values = category.value === RANDOM ? allActions : allActions.filter((value) => known[value]?.category === category.value);
          action.options = { ...(action.options || {}), values: [RANDOM, ...values] };
          if (action.value !== RANDOM && !values.includes(action.value) && values.length) action.value = values[0];
          const entry = known[String(action.value)] || { roles: [] };
          const list = entry.roles || [];
          const modes = scattered ? ["vaginal", "anal"] : (entry.penetration || []);
          if (penetration) {
            setShown(penetration, modes.length > 0);
            if (modes.length) {
              penetration.options = { ...(penetration.options || {}), values: scattered ? [RANDOM, ...modes] : modes };
              if (!penetration.options.values.includes(penetration.value)) penetration.value = String(action.value).toLowerCase().includes("anal") ? "anal" : modes[0];
            }
          }
          if (cameraWidget) {
            const values = [RANDOM, ANY_CAMERA, ...(scattered ? allCameras : (entry.cameras || []))];
            cameraWidget.options = { ...(cameraWidget.options || {}), values };
            if (!values.includes(cameraWidget.value)) cameraWidget.value = ANY_CAMERA;
          }
          const shape = entry.aspect === "9:16" ? "portrait" : "landscape";
          label.innerHTML = (scattered ? "<b>Random</b>: the act is rolled when the prompt is written<br>" : "")
            + list.map((item, index) => `<b>${nameFor(item.gender, index, list)}</b>: ${item.role}`).join("<br>")
            + (entry.aspect ? `<br><b>Frame</b>: ${entry.aspect} (${shape})` : "")
            + "<br><b>Extra</b>: a person in the video but not in the act (optional)"
            + (node.promptforgeRandomText ? `<br><b>${node.promptforgeRandomText}</b>` : "");
          for (let number = 1; number <= 3; number++) {
            const used = number <= list.length || number === 3;
            const gender = list[number - 1]?.gender;
            const name = number === 3 && !gender ? "Extra" : nameFor(gender, number - 1, list);
            const from = find(`subject_${number}_from`);
            const look = find(`subject_${number}_look`);
            if (from) {
              from.label = `${name} — picture`;
              from.options = { ...(from.options || {}), label: from.label };
              from.tooltip = `Which picture is the ${name.toLowerCase()}? None = describe only.`;
            }
            if (look) {
              look.label = `${name} — description`;
              look.options = { ...(look.options || {}), label: look.label };
            }
            setShown(from, used);
            setShown(look, used && from?.value !== SUBJECT_DISABLED);
          }
          node.setDirtyCanvas(true, true);
        };
        node.promptforgeRestoreActionSize = () => {
          const saved = node.properties?.[SAVED];
          const minimum = node.computeSize();
          node.setSize([Math.max(saved?.[0] || node.size[0], 620), Math.max(saved?.[1] || node.size[1], minimum[1])]);
        };
        for (const number of [1, 2, 3]) find(`subject_${number}_from`)?.callback;
        const fromCallback = {};
        for (const number of [1, 2, 3]) {
          const from = find(`subject_${number}_from`);
          if (!from) continue;
          fromCallback[number] = from.callback;
          from.callback = (...args) => { fromCallback[number]?.apply(from, args); refresh(); };
        }
        const categoryCallback = category.callback;
        category.callback = (...args) => { categoryCallback?.apply(category, args); refresh(); };
        const callback = action.callback;
        action.callback = (...args) => { callback?.apply(action, args); refresh(); };
        node.promptforgeRefreshAction = refresh;
        node.promptforgeShowRandom = ({ picks }) => { node.promptforgeRandomText = picks.length ? "Rolled: " + picks.map(([name, value]) => `${name} = ${value}`).join(", ") : ""; refresh(); };
        refresh();
        node.promptforgeRestoreActionSize();
      };
      const onResize = nodeType.prototype.onResize;
      nodeType.prototype.onResize = function (size) {
        onResize?.apply(this, arguments);
        this.properties = this.properties || {};
        this.properties[SAVED] = [Math.round(size[0]), Math.round(size[1])];
      };
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        const width = this.size[0], height = this.size[1];
        const category = this.widgets.find((widget) => widget.name === "category");
        const action = this.widgets.find((widget) => widget.name === "action");
        if (category && action) setTimeout(() => api.fetchApi("/promptforge/actions").then((response) => response.json()).then((known) => {
          const found = known[String(action.value)]?.category;
          if (found) category.value = found;
          this.promptforgeRefreshAction?.();
          this.setDirtyCanvas(true, true);
        }), 60);
        this.promptforgeRefreshAction?.();
        // the saved size was applied before this hook; keep it instead of snapping back to the widget minimum
        setTimeout(() => {
          this.promptforgeRestoreActionSize?.();
          this.setSize([Math.max(width, this.size[0]), Math.max(height, this.size[1])]);
        }, 0);
        return result;
      };
      return;
    }

    if (nodeData.name === WRITER) {
      // the style only matters while the template is automatic
      const onCreated = nodeType.prototype.onNodeCreated;
      const refresh = (node) => {
        const template = node.widgets.find((widget) => widget.name === "prompt_template");
        setShown(node.widgets.find((widget) => widget.name === "prompt_style"), String(template.value).startsWith("automatic"));
        node.setSize([node.size[0], node.computeSize()[1]]);
        node.setDirtyCanvas(true, true);
      };
      nodeType.prototype.onNodeCreated = function () {
        onCreated?.apply(this, arguments);
        const template = this.widgets.find((widget) => widget.name === "prompt_template");
        const callback = template.callback;
        template.callback = (...args) => { callback?.apply(template, args); refresh(this); };
        refresh(this);
      };
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        const width = this.size[0], height = this.size[1];
        refresh(this);
        // the saved size was applied before this hook; keep it instead of snapping back to the widget minimum
        this.setSize([Math.max(width, this.size[0]), Math.max(height, this.size[1])]);
        return result;
      };
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

    if (nodeData.name === LIBRARY) {
      const onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onCreated?.apply(this, arguments);
        const node = this;
        const text = node.widgets.find((widget) => widget.name === "text");
        const bar = el("div", { display: "flex", alignItems: "center", gap: "4px", fontSize: "12px", color: "var(--input-text, #ddd)" });
        const select = el("select", { ...INPUT, flex: "1 1 auto", minWidth: "0", height: "24px" });
        const previous = el("button", BUTTON, { textContent: "◀" });
        const next = el("button", BUTTON, { textContent: "▶" });
        const save = el("button", BUTTON, { textContent: "Save" });
        const saveAs = el("button", BUTTON, { textContent: "Save as…" });
        const remove = el("button", BUTTON, { textContent: "Delete" });
        bar.append(el("span", { opacity: ".7" }, { textContent: "Prompts" }), previous, select, next, save, saveAs, remove);
        let prompts = {};
        const refresh = (chosen) => {
          select.replaceChildren(el("option", {}, { value: "", textContent: "— saved prompts —" }), ...Object.keys(prompts).map((name) => el("option", {}, { value: name, textContent: name })));
          select.value = chosen && prompts[chosen] !== undefined ? chosen : "";
        };
        const load = async () => { prompts = await (await api.fetchApi("/promptforge/prompts")).json(); refresh(); };
        const store = async (name) => { prompts = await (await api.fetchApi("/promptforge/prompts", { method: "POST", body: JSON.stringify({ name, text: text.value }) })).json(); refresh(name); };
        select.addEventListener("change", () => { if (prompts[select.value] !== undefined) { text.value = prompts[select.value]; node.setDirtyCanvas(true, true); } });
        previous.addEventListener("click", () => { const names = Object.keys(prompts); if (!names.length) return; const at = (names.indexOf(select.value) - 1 + names.length) % names.length; refresh(names[at]); text.value = prompts[names[at]]; node.setDirtyCanvas(true, true); });
        next.addEventListener("click", () => { const names = Object.keys(prompts); if (!names.length) return; const at = (names.indexOf(select.value) + 1) % names.length; refresh(names[at]); text.value = prompts[names[at]]; node.setDirtyCanvas(true, true); });
        save.addEventListener("click", () => { const name = select.value || window.prompt("Name of this prompt:"); if (name?.trim()) store(name.trim()); });
        saveAs.addEventListener("click", () => { const name = window.prompt("Name of this prompt:", select.value); if (name?.trim()) store(name.trim()); });
        remove.addEventListener("click", async () => {
          if (!select.value || !window.confirm(`Delete the prompt "${select.value}"?`)) return;
          prompts = await (await api.fetchApi("/promptforge/prompts?name=" + encodeURIComponent(select.value), { method: "DELETE" })).json();
          refresh("");
        });
        const widget = domWidget(node, "saved_prompts", bar, () => 30);
        node.widgets.splice(node.widgets.indexOf(widget), 1);
        node.widgets.splice(node.widgets.indexOf(text), 0, widget);
        load();
        node.setSize([Math.max(node.size[0], 520), Math.max(node.size[1], node.computeSize()[1])]);
      };
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        const width = this.size[0], height = this.size[1];
        // the saved size was applied before this hook; keep it instead of snapping back to the widget minimum
        this.setSize([Math.max(width, this.size[0]), Math.max(height, this.size[1])]);
        return result;
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
      const category = find("category");
      const actionWidget = find("action");
      const penetration = find("penetration");
      const cameraWidget = find("camera_angle");
      const allCameras = [...(cameraWidget?.options?.values || [])].filter((value) => value !== RANDOM && value !== ANY_CAMERA);
      let known = null;
      const filter = async () => {
        if (!known) known = await (await api.fetchApi("/promptforge/actions")).json();
        const scattered = category.value === RANDOM || actionWidget.value === RANDOM;
        const values = category.value === RANDOM ? Object.keys(known) : Object.keys(known).filter((value) => known[value].category === category.value);
        actionWidget.options = { ...(actionWidget.options || {}), values: [RANDOM, ...values] };
        if (actionWidget.value !== RANDOM && !values.includes(actionWidget.value) && values.length) actionWidget.value = values[0];
        const modes = scattered ? ["vaginal", "anal"] : (known[actionWidget.value]?.penetration || []);
        if (penetration) {
          setShown(penetration, modes.length > 0);
          if (modes.length) penetration.options = { ...(penetration.options || {}), values: scattered ? [RANDOM, ...modes] : modes };
        }
        if (cameraWidget) {
          const pool = scattered ? allCameras : (known[actionWidget.value]?.cameras || []);
          const cameras = [RANDOM, ANY_CAMERA, ...pool];
          cameraWidget.options = { ...(cameraWidget.options || {}), values: cameras };
          if (!cameras.includes(cameraWidget.value)) cameraWidget.value = ANY_CAMERA;
        }
        node.setDirtyCanvas(true, true);
      };
      category.callback = ((original) => (...args) => { original?.apply(category, args); filter(); })(category.callback);
      actionWidget.callback = ((original) => (...args) => { original?.apply(actionWidget, args); filter(); })(actionWidget.callback);
      filter();

      let texts = [];
      const frame = el("div", { position: "relative", width: "100%", height: "100%", minWidth: "0", overflow: "hidden", contain: "layout paint" });
      const root = el("div", { position: "absolute", inset: "0", display: "flex", flexDirection: "column", gap: "4px", boxSizing: "border-box", overflowY: "auto", fontSize: "12px", color: "var(--input-text, #ddd)" });
      frame.append(root);
      frame.addEventListener("pointerdown", (event) => event.stopPropagation());
      root.addEventListener("wheel", (event) => event.stopPropagation());

      const randomLine = el("div", { opacity: ".85", flex: "0 0 auto" });
      const header = el("div", { display: "flex", alignItems: "center", gap: "6px", flex: "0 0 auto" });
      const minus = el("button", { cursor: "pointer", width: "24px" }, { textContent: "−" });
      const plus = el("button", { cursor: "pointer", width: "24px" }, { textContent: "+" });
      const count = el("input", { width: "52px", textAlign: "center", background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #ddd)", border: "1px solid var(--border-color, #555)", borderRadius: "3px" }, { type: "number", min: 1, max: 200, step: 1 });
      header.append("Parts", minus, count, plus, el("span", { opacity: ".6", marginLeft: "6px" }, { textContent: "What happens in each part. Start a box with (10s) for a longer part, or raw: to use your own finished prompt." }));
      const list = el("div", { display: "flex", flexDirection: "column", gap: "4px" });
      root.append(header, randomLine, list);

      node.properties = node.properties || {};
      const heights = () => (node.properties.partHeights = node.properties.partHeights || []);
      const total = () => Math.max(1, Number(countWidget.value) || texts.length || 3);
      const save = () => {
        texts.length = total();
        store.value = JSON.stringify(Array.from(texts, (text) => text || ""));
      };
      const fit = () => {
        // never below the height the node already has: a measuring pass right after loading must not shrink it
        node.setSize([node.size[0], Math.max(node.size[1], node.computeSize()[1])]);
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

      node.promptforgeShowRandom = ({ picks }) => { randomLine.textContent = picks.length ? "Rolled: " + picks.map(([name, value]) => `${name} = ${value}`).join(", ") : ""; };
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
      const width = this.size[0], height = this.size[1];
      this.promptforgeRestoreParts?.();
      // the saved size was applied before this hook; keep it instead of snapping back to the widget minimum
      this.setSize([Math.max(width, this.size[0]), Math.max(height, this.size[1])]);
      return result;
    };
  },
});
