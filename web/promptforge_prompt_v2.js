import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODES = /^PROMPTFORGE_V2_(IMAGE|H3)_PROMPT$/;
const SOURCES = [["raw", "Raw prompt"], ["idea", "Idea for writer"], ["action", "Action preset"]];
const GENERATIONS = [["image", "Image"], ["minimax", "Minimax H3"]];
const WIDTH = 520;
const MAX_HEIGHT = 620;
const MIN_HEIGHT = 300;
const CHROME = 82;
let catalog;

const widget = (node, name) => node.widgets?.find((entry) => entry.name === name);
function element(tag, props = {}) { return Object.assign(document.createElement(tag), props); }
function button(label, action) {
  const control = element("button", { textContent: label, type: "button" });
  control.addEventListener("click", action);
  return control;
}
function panel(node, name, minHeight, maxHeight) {
  const frame = element("div");
  frame.style.cssText = "position:relative;width:100%;height:calc(100% - 12px);min-width:0;overflow:auto;contain:layout paint";
  const content = element("div");
  content.style.cssText = "display:flex;flex-direction:column;gap:5px;padding:3px 14px 3px 3px;box-sizing:border-box;min-height:100%;color:var(--input-text);font:12px sans-serif";
  frame.append(content);
  frame.addEventListener("pointerdown", (event) => event.stopPropagation());
  frame.addEventListener("wheel", (event) => event.stopPropagation());

  // own grip: the canvas corner sits under a DOM widget, so drag from here instead
  const grip = element("div");
  grip.title = "Drag to resize";
  grip.style.cssText = "position:absolute;right:1px;bottom:1px;width:16px;height:16px;cursor:nwse-resize;z-index:3;"
    + "background:linear-gradient(135deg,transparent 46%,var(--border-color,#888) 46%,var(--border-color,#888) 54%,transparent 54%,transparent 64%,var(--border-color,#888) 64%,var(--border-color,#888) 72%,transparent 72%)";
  frame.append(grip);
  let drag = null;
  grip.addEventListener("pointerdown", (event) => {
    event.stopPropagation();
    event.preventDefault();
    drag = { x: event.clientX, y: event.clientY, w: node.size[0], h: node.size[1] };
    try { grip.setPointerCapture(event.pointerId); } catch { /* synthetic events */ }
  });
  grip.addEventListener("pointermove", (event) => {
    if (!drag) return;
    event.stopPropagation();
    node.setSize([Math.max(320, drag.w + event.clientX - drag.x), Math.max(220, drag.h + event.clientY - drag.y)]);
  });
  const release = () => { drag = null; };
  grip.addEventListener("pointerup", release);
  grip.addEventListener("pointercancel", release);
  node.addDOMWidget(name, "promptforge_v2", frame, { serialize: false, getMinHeight: () => 200 });
  return { frame, content };
}
function change(node, action) {
  node.graph?.beforeChange();
  try { action(); } finally { node.graph?.afterChange(); }
  node.setDirtyCanvas?.(true, true);
}
function modeRow(values, current, pick) {
  const row = element("div");
  row.style.cssText = "display:flex;gap:4px;flex-wrap:wrap";
  const controls = values.map(([value, title]) => {
    const control = button(title, () => pick(value));
    control.dataset.mode = value;
    row.append(control);
    return control;
  });
  return { row, paint: () => {
    for (const control of controls) {
      const selected = control.dataset.mode === current();
      control.style.cssText = `border:1px solid ${selected ? "#78c9ff" : "#65717c"};border-radius:5px;padding:4px 8px;background:${selected ? "#24445a" : "#252b31"};color:#eef5fa`;
      control.setAttribute("aria-pressed", String(selected));
    }
  } };
}
const FIELD = "width:100%;box-sizing:border-box;background:var(--comfy-input-bg,#0f1419);color:var(--input-text,#e7eef5);border:1px solid #3b4a56;border-radius:4px;padding:4px 6px;font:12px sans-serif";

function promptUI(node, fallback) {
  const source = widget(node, "source");
  const generation = widget(node, "generation");
  const action = widget(node, "action");
  const sizes = new Map();
  const hide = (item, hidden) => {
    if (!item) return;
    if (!sizes.has(item)) sizes.set(item, item.computeSize);
    item.hidden = hidden;
    item.computeSize = hidden ? () => [0, -4] : (sizes.get(item) || (() => [0, -4]));
    if (item.inputEl) item.inputEl.style.display = hidden ? "none" : "";
  };
  const generationOf = () => generation?.value || fallback;
  const familyOf = () => generationOf() === "minimax" ? "minimax" : "image";
  const owned = ["source", "generation", "action", "text", "text_in", "negative", "notes",
    "subject_1", "subject_2", "subject_3", "camera", "soundscape", "music"]
    .map((name) => widget(node, name)).filter(Boolean);

  let panelHeight = MIN_HEIGHT;
  const { content: root } = panel(node, "prompt_controls", () => panelHeight, MAX_HEIGHT);
  const status = element("div");
  status.style.cssText = "color:#f0b6a0;overflow-wrap:anywhere";
  const saved = element("select", { ariaLabel: "Saved PromptForge ideas" });
  saved.style.cssText = "flex:1;min-width:100px";
  const saveName = element("input", { type: "text", placeholder: "Idea name" });
  saveName.style.cssText = "flex:1;min-width:80px";
  const saveBar = element("div");
  saveBar.style.cssText = "display:flex;gap:4px;align-items:center;flex-wrap:wrap";
  const chosen = element("div");
  chosen.style.cssText = "display:flex;gap:5px;align-items:center;overflow-wrap:anywhere";
  const search = element("input", { type: "search", placeholder: "Search presets: name, description, gender" });
  search.style.cssText = FIELD;
  const tabs = element("div");
  tabs.style.cssText = "display:flex;gap:3px;flex-wrap:wrap";
  const rows = element("div");
  rows.style.cssText = "display:flex;flex-direction:column;gap:3px;max-height:200px;overflow:auto;padding-right:2px";
  let tab = "All";
  let ideas = {};

  const section = (title) => {
    const label = element("div", { textContent: title });
    label.style.cssText = "font-size:11px;font-weight:600;letter-spacing:.03em;opacity:.7";
    return label;
  };
  const areas = new Map();
  function field(name, minHeight, placeholder) {
    const item = widget(node, name);
    const area = element("textarea", { placeholder, spellcheck: false });
    area.style.cssText = `${FIELD};min-height:${minHeight}px;resize:vertical;line-height:1.35`;
    area.value = item?.value || "";
    area.addEventListener("input", () => {
      if (!item) return;
      item.value = area.value;
      item.callback?.(area.value);
    });
    areas.set(name, area);
    // dragging the textarea grip grows the node instead of scrolling the panel
    area.addEventListener("pointerup", () => {
      const needed = node.computeSize?.()[1] || 0;
      if (needed > (node.size?.[1] || 0)) node.setSize?.([node.size[0], needed]);
    });
    return area;
  }
  function line(name, placeholder) {
    const item = widget(node, name);
    const input = element("input", { type: "text", placeholder });
    input.style.cssText = FIELD;
    input.value = item?.value ?? "";
    input.addEventListener("input", () => {
      if (!item) return;
      item.value = input.value;
      item.callback?.(input.value);
    });
    areas.set(name, input);
    return input;
  }
  function syncAreas() {
    for (const [name, area] of areas) {
      const value = widget(node, name)?.value ?? "";
      if (area.value !== value) area.value = value;
    }
  }

  const pick = (label) => {
    if (!action) return;
    change(node, () => {
      action.value = label ? `${familyOf() === "minimax" ? "h3" : "image"}: ${label}` : "";
      action.callback?.(action.value);
    });
  };
  const generationModes = modeRow(GENERATIONS, generationOf, (value) => {
    if (!generation) return;
    generation.value = value;
    generation.callback?.(value);
    change(node, () => {});
    tab = "All";
    load();
    refresh();
  });
  const sourceModes = modeRow(SOURCES, () => source?.value || "raw", (value) => {
    if (!source) return;
    change(node, () => { source.value = value; source.callback?.(value); });
    refresh();
  });

  const familyEntries = () => catalog?.[familyOf()] || {};
  const matches = (query) => (label, entry) => query.every((word) => `${label} ${entry.category} ${entry.description} ${(entry.gender || []).join(" ")}`.toLowerCase().includes(word));
  const paintChosen = () => {
    chosen.replaceChildren(
      element("span", { textContent: action?.value ? `Picked: ${action.value}` : "Picked: none" }),
      button("Clear", () => { pick(""); refresh(); }),
    );
  };
  const paintTabs = () => {
    const categories = [...new Set(Object.values(familyEntries()).map((entry) => entry.category))].sort((a, b) => a.localeCompare(b));
    if (tab !== "All" && !categories.includes(tab)) tab = "All";
    const controls = ["All", ...categories].map((name) => {
      const control = button(name, () => { tab = name; paintTabs(); paintRows(); });
      control.style.cssText = `border:1px solid ${name === tab ? "#78c9ff" : "#48525b"};border-radius:9px;padding:2px 8px;background:${name === tab ? "#24445a" : "#20262c"};color:#dfeaf2`;
      control.setAttribute("aria-pressed", String(name === tab));
      return control;
    });
    tabs.replaceChildren(...controls);
  };
  const paintRows = () => {
    if (source?.value !== "action") { rows.replaceChildren(); return; }
    const query = search.value.toLowerCase().split(/\s+/).filter(Boolean);
    const entries = Object.entries(familyEntries())
      .filter(([label, entry]) => (tab === "All" || entry.category === tab) && matches(query)(label, entry));
    const none = button("(none) — preset free text", () => { pick(""); refresh(); });
    none.style.cssText = rowsStyle(!action?.value);
    const list = entries.map(([label, entry]) => {
      const row = button("", () => { pick(label); refresh(); });
      row.setAttribute("aria-label", label);
      row.style.cssText = rowsStyle(action?.value === `${familyOf() === "minimax" ? "h3" : "image"}: ${label}`);
      const title = element("div", { textContent: label });
      title.style.cssText = "font-weight:600;color:#eef5fa";
      const description = element("div", { textContent: entry.description });
      description.style.cssText = "opacity:.8;line-height:1.25";
      const meta = element("div", { textContent: [entry.category, ...(entry.gender || []), entry.aspect].filter(Boolean).join(" · ") });
      meta.style.cssText = "font-size:11px;opacity:.7";
      row.append(title, description, meta);
      return row;
    });
    rows.replaceChildren(none, ...list);
  };
  function rowsStyle(selected) {
    return `display:flex;flex-direction:column;gap:2px;text-align:left;border:1px solid ${selected ? "#78c9ff" : "#333c44"};border-radius:5px;padding:4px 7px;background:${selected ? "#24445a" : "#1c2227"};color:#dce8ef`;
  }

  const promptHeading = section("Prompt");
  const promptArea = field("text", 60, "Describe the scene");
  const negativeArea = field("negative", 40, "What to keep out");

  const actionBlock = element("div");
  actionBlock.style.cssText = "display:flex;flex-direction:column;gap:5px";
  actionBlock.append(search, tabs, rows, chosen);

  const subjectBlock = element("div");
  subjectBlock.style.cssText = "display:flex;flex-direction:column;gap:4px";
  const subjectRow = element("div");
  subjectRow.style.cssText = "display:flex;gap:4px";
  const camera = widget(node, "camera");
  const cameraSelect = element("select", { ariaLabel: "Camera" });
  cameraSelect.style.cssText = FIELD;
  for (const value of camera?.options?.values || ["auto"]) cameraSelect.append(new Option(value, value));
  cameraSelect.value = camera?.value || "auto";
  cameraSelect.addEventListener("change", () => {
    if (!camera) return;
    camera.value = cameraSelect.value;
    camera.callback?.(camera.value);
  });
  for (const [index, name] of ["subject_1", "subject_2", "subject_3"].entries()) {
    const input = line(name, `Subject ${index + 1}`);
    input.style.flex = "1";
    input.style.minWidth = "0";
    subjectRow.append(input);
  }
  subjectBlock.append(section("Subjects & camera"), subjectRow, cameraSelect);

  const notesBlock = element("div");
  notesBlock.style.cssText = "display:flex;flex-direction:column;gap:4px";
  notesBlock.append(section("Notes"), field("notes", 40, "Extra direction folded into the action brief"));

  const soundBlock = element("div");
  soundBlock.style.cssText = "display:flex;flex-direction:column;gap:4px";
  soundBlock.append(section("Soundscape"), field("soundscape", 40, "Room tone, dialogue, effects"), line("music", "Music direction, N/A for none"));

  const savedBlock = element("div");
  savedBlock.style.cssText = "display:flex;flex-direction:column;gap:4px";
  savedBlock.append(section("Saved ideas"), saveBar);

  // saved ideas keep every field of the node, generation included
  const dataFields = ["text", "negative", "action", "generation", "subject_1", "subject_2", "subject_3", "camera", "notes"];
  const fieldName = (key) => key === "text" ? "general_prompt" : key === "negative" ? "negative_prompt" : key;
  const refreshIdeas = (selected = "") => {
    saved.replaceChildren(new Option("Saved prompts and ideas", ""), ...Object.keys(ideas).sort((a, b) => a.localeCompare(b)).map((key) => new Option(key, key)));
    saved.value = selected && ideas[selected] ? selected : "";
    saveName.value = saved.value || saveName.value;
  };
  const applyFields = (data) => {
    change(node, () => {
      for (const key of dataFields) {
        const item = widget(node, key);
        if (item && typeof data?.[fieldName(key)] === "string") { item.value = data[fieldName(key)]; item.callback?.(item.value); }
      }
      if (data?.source && SOURCES.some(([value]) => value === data.source)) { source.value = data.source; source.callback?.(source.value); }
    });
    refresh();
  };
  const reloadIdeas = async (selected = "") => {
    try { const response = await api.fetchApi("/promptforge/ideas"); if (!response.ok) throw new Error(`Saved prompt list unavailable (${response.status})`); ideas = await response.json(); refreshIdeas(selected); status.textContent = `${Object.keys(ideas).length} saved`; }
    catch (error) { status.textContent = error.message; }
  };
  const storeIdea = async (overwrite) => {
    const name = (overwrite ? (saved.value || saveName.value) : saveName.value).trim();
    if (!name) { status.textContent = "Enter a name to save."; return; }
    const current = Object.fromEntries([...dataFields.map((key) => [fieldName(key), widget(node, key)?.value ?? ""]), ["source", source?.value || "raw"]]);
    const response = await api.fetchApi("/promptforge/ideas", { method: "POST", body: JSON.stringify({ name, data: current }) });
    if (!response.ok) throw new Error(`Could not save prompt (${response.status})`);
    ideas = await response.json(); refreshIdeas(name); status.textContent = `Saved “${name}”.`;
  };
  saved.onchange = () => { if (saved.value) { applyFields(ideas[saved.value]); saveName.value = saved.value; status.textContent = `Loaded “${saved.value}”.`; } };
  saveBar.append(saved, saveName,
    button("Save", async () => { try { await storeIdea(true); } catch (error) { status.textContent = error.message; } }),
    button("Save as", async () => { try { await storeIdea(false); } catch (error) { status.textContent = error.message; } }),
    button("Delete", async () => { try { if (!saved.value) throw new Error("Choose a saved prompt first."); const name = saved.value; const response = await api.fetchApi(`/promptforge/ideas?name=${encodeURIComponent(name)}`, { method: "DELETE" }); if (!response.ok) throw new Error(`Could not delete (${response.status})`); ideas = await response.json(); refreshIdeas(); status.textContent = `Deleted “${name}”.`; } catch (error) { status.textContent = error.message; } }),
  );

  search.oninput = paintRows;

  function paint() {
    generationModes.paint();
    sourceModes.paint();
    paintChosen();
    paintTabs();
    paintRows();
  }

  const visible = (block, show) => { block.style.display = show ? "" : "none"; };
  const measurable = () => {
    const height = root.scrollHeight || root.offsetHeight || 0;
    return Number.isFinite(height) && height > 0 ? height + 8 : 0;
  };
  const estimate = () => {
    const audio = generationOf() === "minimax";
    const actionMode = source?.value === "action";
    return 340
      + (actionMode ? 330 : 0)
      + (audio ? 100 : 0);
  };
  let targetHeight = MIN_HEIGHT;
  let sizedOnce = false;
  // `grow` is only true when the caller wants the node re-measured (mode/source change).
  // A plain refresh leaves whatever height the user dragged.
  const fit = (grow = false) => {
    const current = node.size?.[1] || 0;
    if (sizedOnce && !grow && current >= MIN_HEIGHT) {
      panelHeight = Math.max(120, current - CHROME);
      return;
    }
    const measured = measurable() || Math.max(120, estimate() - CHROME);
    panelHeight = Math.max(120, measured);
    const full = (node.computeSize?.()[1] || panelHeight + CHROME) ;
    targetHeight = Math.max(MIN_HEIGHT, full);
    if (current > targetHeight) targetHeight = current;
    panelHeight = Math.max(120, targetHeight - CHROME);
    const width = Math.max(node.size?.[0] || 0, WIDTH);
    sizedOnce = true;
    if (node.size && Math.abs(node.size[0] - width) < 1 && Math.abs(node.size[1] - targetHeight) < 1) return;
    node.setSize?.([width, targetHeight]);
  };
  let lastShape = "";
  function refresh() {
    const actionMode = source?.value === "action";
    const shape = `${source?.value || ""}|${generationOf()}`;
    const grow = shape !== lastShape;
    lastShape = shape;
    for (const item of owned) hide(item, true);
    visible(actionBlock, actionMode);
    visible(subjectBlock, actionMode);
    visible(notesBlock, actionMode);
    visible(soundBlock, generationOf() === "minimax");
    promptHeading.textContent = actionMode ? "Additional intent" : "Prompt";
    promptArea.placeholder = actionMode ? "Extra intent appended to the action brief" : "Describe the scene";
    negativeArea.placeholder = actionMode ? "Constraints kept out of the brief" : "What to keep out";
    syncAreas();
    paint();
    fit(grow);
    node.setDirtyCanvas?.(true, true);
  }
  async function load() {
    if (catalog) return;
    try {
      const response = await api.fetchApi("/promptforge/v2/actions");
      if (!response.ok) throw new Error(`Preset list unavailable (${response.status})`);
      catalog = await response.json();
      status.textContent = "";
    } catch (error) { status.textContent = error.message; }
    paint();
  }

  const previousResize = node.onResize;
  node.onResize = function (size) {
    size[0] = Math.max(size[0], WIDTH);
    size[1] = Math.max(size[1], MIN_HEIGHT);
    return previousResize?.apply(this, arguments);
  };

  root.append(generationModes.row, sourceModes.row, promptHeading, promptArea,
    section("Negative"), negativeArea, savedBlock, actionBlock, subjectBlock, notesBlock, soundBlock, status);
  for (const [item, callback] of [[source, refresh], [generation, refresh], [action, paint]]) {
    if (!item) continue;
    const original = item.callback;
    item.callback = function () { const result = original?.apply(this, arguments); callback(); return result; };
  }
  for (const item of owned) hide(item, true);
  refresh();
  requestAnimationFrame(fit);
  if (globalThis.ResizeObserver) {
    const observer = new ResizeObserver(() => fit());
    observer.observe(root);
    const removed = node.onRemoved;
    node.onRemoved = function () { observer.disconnect(); return removed?.apply(this, arguments); };
  }
  load();
  reloadIdeas();
  return refresh;
}

app.registerExtension({
  name: "PromptForge.V2Prompt",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.test(nodeData.name)) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      this.resizable = true;
      this.flags = this.flags || {};
      this.flags.resizable = true;
      if (!this.promptforgeV2Prompt) this.promptforgeV2Prompt = promptUI(this, nodeData.name.includes("_H3_") ? "minimax" : "image");
      return result;
    };
    const fallback = nodeData.name.includes("_H3_") ? "minimax" : "image";
    const configure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = configure?.apply(this, arguments);
      // workflows saved before generation existed leave it empty: fall back to the node family, then repaint
      const node = this;
      requestAnimationFrame(() => {
        const generation = widget(node, "generation");
        if (generation && !generation.value) generation.value = fallback;
        node.promptforgeV2Prompt?.();
      });
      return result;
    };
  },
});
