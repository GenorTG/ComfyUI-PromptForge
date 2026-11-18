import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODES = /^PROMPTFORGE_V2_(IMAGE|H3)_PROMPT$/;
const SOURCES = [["raw", "Raw prompt"], ["idea", "Idea for writer"], ["action", "Action preset"]];
const GENERATIONS = [["image", "Image"], ["minimax", "Minimax H3"]];
let catalog;

const widget = (node, name) => node.widgets?.find((entry) => entry.name === name);
function element(tag, props = {}) { return Object.assign(document.createElement(tag), props); }
function button(label, action) {
  const control = element("button", { textContent: label, type: "button" });
  control.addEventListener("click", action);
  return control;
}
function panel(node, name, minHeight = 100) {
  const frame = element("div");
  frame.style.cssText = "position:relative;width:100%;height:100%;min-width:0;overflow:hidden;contain:layout paint";
  const content = element("div");
  content.style.cssText = "position:absolute;inset:0;overflow:auto;display:flex;flex-direction:column;gap:5px;color:var(--input-text);font:12px sans-serif";
  frame.append(content);
  frame.addEventListener("pointerdown", (event) => event.stopPropagation());
  frame.addEventListener("wheel", (event) => event.stopPropagation());
  node.addDOMWidget(name, "promptforge_v2", frame, { serialize: false, getMinHeight: () => minHeight });
  return content;
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
  const hidden = ["action", "subject_1", "subject_2", "subject_3", "camera", "notes"].map((name) => widget(node, name)).filter(Boolean);
  const audio = ["soundscape", "music"].map((name) => widget(node, name)).filter(Boolean);

  const root = panel(node, "prompt_controls", 210);
  root.style.gap = "6px";
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
  const tabs = element("div");
  tabs.style.cssText = "display:flex;gap:3px;flex-wrap:wrap";
  const rows = element("div");
  rows.style.cssText = "display:flex;flex-direction:column;gap:3px;max-height:250px;overflow:auto;padding-right:2px";
  let tab = "All";
  let ideas = {};

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
      const control = button(name, () => { tab = name; paint(); });
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
  function paint() {
    generationModes.paint();
    sourceModes.paint();
    paintChosen();
    paintTabs();
    paintRows();
  }
  function refresh() {
    for (const item of hidden) hide(item, source?.value !== "action");
    for (const item of audio) hide(item, generationOf() !== "minimax");
    paint();
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
  root.append(generationModes.row, sourceModes.row, saveBar, chosen, search, tabs, rows, status);
  for (const [item, callback] of [[source, refresh], [generation, refresh], [action, paint]]) {
    if (!item) continue;
    const original = item.callback;
    item.callback = function () { const result = original?.apply(this, arguments); callback(); return result; };
  }
  hide(source, true);
  hide(generation, true);
  refresh();
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
