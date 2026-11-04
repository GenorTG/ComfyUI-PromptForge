import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "PROMPTFORGE_LoraLoader";
const MAX_ROWS = 120;

let libraryRequest = null;
function loadLibrary(refresh = false) {
  if (!libraryRequest || refresh) {
    libraryRequest = api.fetchApi("/promptforge/loras").then((response) => response.json());
  }
  return libraryRequest;
}

function parseState(text) {
  try {
    const state = JSON.parse(text || "{}");
    return { filter: state.filter || "all", items: Array.isArray(state.items) ? state.items : [] };
  } catch {
    return { filter: "all", items: [] };
  }
}

function el(tag, style = {}, props = {}) {
  const element = Object.assign(document.createElement(tag), props);
  Object.assign(element.style, style);
  return element;
}

function numberInput(value, onChange, { step = 0.05, width = "52px", placeholder = "" } = {}) {
  const input = el("input", { width, background: "var(--comfy-input-bg)", color: "var(--input-text)", border: "1px solid var(--border-color)", borderRadius: "3px", padding: "1px 3px" },
    { type: "number", step, value: value ?? "", placeholder });
  input.addEventListener("change", () => onChange(input.value === "" ? null : Number(input.value)));
  return input;
}

function buildList(node, stateWidget) {
  let state = parseState(stateWidget.value);
  let library = { loras: [], families: [] };
  let search = "";
  let activeOnly = false;

  const root = el("div", { display: "flex", flexDirection: "column", gap: "4px", height: "100%", fontSize: "12px", color: "var(--input-text)" });
  const bar = el("div", { display: "flex", gap: "4px" });
  const searchBox = el("input", { flex: "1", minWidth: "0", background: "var(--comfy-input-bg)", color: "var(--input-text)", border: "1px solid var(--border-color)", borderRadius: "3px", padding: "3px 6px" },
    { type: "search", placeholder: "Search LoRAs (name, folder, trigger words)..." });
  const familyBox = el("select", { background: "var(--comfy-input-bg)", color: "var(--input-text)", border: "1px solid var(--border-color)", borderRadius: "3px", maxWidth: "140px" });
  const refresh = el("button", { cursor: "pointer" }, { textContent: "↻", title: "Rescan the loras folders" });
  const onlyLabel = el("label", { display: "flex", alignItems: "center", gap: "2px", whiteSpace: "nowrap" });
  const onlyBox = el("input", {}, { type: "checkbox" });
  onlyLabel.append(onlyBox, "on only");
  bar.append(searchBox, familyBox, onlyLabel, refresh);
  const list = el("div", { flex: "1", overflowY: "auto", minHeight: "0", border: "1px solid var(--border-color)", borderRadius: "3px" });
  list.addEventListener("wheel", (event) => event.stopPropagation());
  root.append(bar, list);

  const commit = () => {
    state.items = state.items.filter((item) => item.on || item.random || item.group || item.model !== 1 || item.clip != null);
    stateWidget.value = JSON.stringify(state);
    node.setDirtyCanvas?.(true, true);
  };
  const byName = () => new Map(state.items.map((item) => [item.name, item]));
  const itemFor = (name) => {
    let item = state.items.find((entry) => entry.name === name);
    if (!item) {
      item = { name, on: false, model: 1, clip: null, random: false, min: 0.6, max: 1, group: "" };
      state.items.push(item);
    }
    return item;
  };

  function row(entry, item) {
    const line = el("div", { display: "flex", alignItems: "center", gap: "4px", padding: "2px 4px", borderBottom: "1px solid var(--border-color)", flexWrap: "wrap" });
    const box = el("input", {}, { type: "checkbox", checked: !!item?.on });
    box.addEventListener("change", () => { itemFor(entry.name).on = box.checked; commit(); render(); });
    const label = el("span", { flex: "1", minWidth: "120px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", opacity: entry.missing ? "0.5" : "1" }, { textContent: entry.name.replace(/\.[^.]+$/, "") });
    label.title = entry.name + (entry.words?.length ? "\nTrigger: " + entry.words.join(", ") : "");
    line.append(box, label);
    if (entry.family) {
      const tag = library.families.find((family) => family.key === entry.family);
      line.append(el("span", { opacity: "0.6", fontSize: "10px" }, { textContent: tag?.label || entry.family }));
    }
    if (entry.missing) line.append(el("span", { color: "#e66" }, { textContent: "missing" }));
    if (item?.on) {
      const set = (key) => (value) => { item[key] = value; commit(); };
      const group = el("input", { width: "46px", background: "var(--comfy-input-bg)", color: "var(--input-text)", border: "1px solid var(--border-color)", borderRadius: "3px", padding: "1px 3px" },
        { value: item.group || "", placeholder: "group", title: "Items sharing a group name are alternatives: one is picked per run" });
      group.addEventListener("change", () => { item.group = group.value.trim(); commit(); });
      const dice = el("button", { cursor: "pointer", opacity: item.random ? "1" : "0.4" }, { textContent: "🎲", title: "Random strength between min and max" });
      dice.addEventListener("click", () => { item.random = !item.random; commit(); render(); });
      line.append(group, dice);
      if (item.random) {
        line.append(numberInput(item.min, (v) => set("min")(v ?? 0), { width: "46px" }), "–", numberInput(item.max, (v) => set("max")(v ?? 1), { width: "46px" }));
      } else {
        line.append(numberInput(item.model, (v) => set("model")(v ?? 1), { placeholder: "model" }), numberInput(item.clip, set("clip"), { placeholder: "= model" }));
      }
    }
    return line;
  }

  function render() {
    const known = new Map(library.loras.map((entry) => [entry.name, entry]));
    const configured = byName();
    const query = search.toLowerCase().split(/\s+/).filter(Boolean);
    list.replaceChildren();

    const active = state.items.filter((item) => item.on).map((item) => known.get(item.name) || { name: item.name, family: "", words: [], missing: true });
    if (active.length) {
      list.append(el("div", { padding: "2px 4px", fontWeight: "bold", background: "var(--comfy-menu-bg)" }, { textContent: `On (${active.length})` }));
      active.forEach((entry) => list.append(row(entry, configured.get(entry.name))));
    }
    if (activeOnly) return;
    const matches = library.loras.filter((entry) => {
      if (configured.get(entry.name)?.on) return false;
      if (state.filter !== "all" && entry.family !== state.filter) return false;
      const haystack = (entry.name + " " + (entry.words || []).join(" ")).toLowerCase();
      return query.every((word) => haystack.includes(word));
    });
    list.append(el("div", { padding: "2px 4px", fontWeight: "bold", background: "var(--comfy-menu-bg)" }, { textContent: `Library (${matches.length})` }));
    matches.slice(0, MAX_ROWS).forEach((entry) => list.append(row(entry, configured.get(entry.name))));
    if (matches.length > MAX_ROWS) list.append(el("div", { padding: "4px", opacity: "0.6" }, { textContent: `${matches.length - MAX_ROWS} more - narrow the search or pick a family` }));
  }

  function fillFamilies() {
    familyBox.replaceChildren(new Option("All families", "all"), ...library.families.map((family) => new Option(family.label, family.key)));
    familyBox.value = library.families.some((family) => family.key === state.filter) ? state.filter : "all";
    state.filter = familyBox.value;
  }

  async function reload(refreshIndex = false) {
    library = await loadLibrary(refreshIndex);
    fillFamilies();
    render();
  }

  let timer;
  searchBox.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(() => { search = searchBox.value; render(); }, 120); });
  familyBox.addEventListener("change", () => { state.filter = familyBox.value; commit(); render(); });
  onlyBox.addEventListener("change", () => { activeOnly = onlyBox.checked; render(); });
  refresh.addEventListener("click", () => reload(true));

  const widget = node.addDOMWidget("lora_list", "promptforge_lora_list", root, { serialize: false, getMinHeight: () => 380 });
  widget.restore = () => { state = parseState(stateWidget.value); fillFamilies(); render(); };
  reload();
  return widget;
}

app.registerExtension({
  name: "PromptForge.LoraLoader",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const stateWidget = this.widgets.find((widget) => widget.name === "loras");
      stateWidget.hidden = true;
      stateWidget.computeSize = () => [0, -4];
      this.promptforgeList = buildList(this, stateWidget);
      this.setSize([Math.max(this.size[0], 460), Math.max(this.size[1], 560)]);
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      this.promptforgeList?.restore();
      return result;
    };
  },
});
