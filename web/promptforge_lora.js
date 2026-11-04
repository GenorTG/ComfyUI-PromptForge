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

const INPUT_STYLE = { background: "var(--comfy-input-bg)", color: "var(--input-text)", border: "1px solid var(--border-color)", borderRadius: "3px", padding: "1px 4px", boxSizing: "border-box" };

function numberInput(value, onChange, { width = "68px", placeholder = "" } = {}) {
  const input = el("input", { ...INPUT_STYLE, width, flex: "0 0 auto" }, { type: "number", step: 0.05, value: value ?? "", placeholder });
  input.addEventListener("change", () => onChange(input.value === "" ? null : Number(input.value)));
  // the wheel nudges by 0.05 instead of zooming the canvas
  input.addEventListener("wheel", (event) => {
    event.preventDefault();
    event.stopPropagation();
    const next = Math.round(((Number(input.value) || 0) + (event.deltaY < 0 ? 0.05 : -0.05)) * 100) / 100;
    input.value = next;
    onChange(next);
  }, { passive: false });
  return input;
}

const POPUP = { position: "fixed", zIndex: 10000, background: "var(--comfy-menu-bg, #222)", color: "var(--input-text, #ddd)", border: "1px solid var(--border-color, #555)", borderRadius: "6px", boxShadow: "0 4px 16px rgba(0,0,0,.5)", fontSize: "12px" };
let tip = null;
function showTip(event, lines) {
  hideTip();
  tip = el("div", { ...POPUP, padding: "6px 8px", maxWidth: "460px", whiteSpace: "pre-wrap", wordBreak: "break-word", pointerEvents: "none" }, { textContent: lines.join("\n") });
  document.body.append(tip);
  const box = tip.getBoundingClientRect();
  tip.style.left = Math.max(4, Math.min(event.clientX + 12, innerWidth - box.width - 8)) + "px";
  tip.style.top = Math.max(4, Math.min(event.clientY + 16, innerHeight - box.height - 8)) + "px";
}
function hideTip() { tip?.remove(); tip = null; }

async function showInfo(name) {
  const info = await (await api.fetchApi("/promptforge/lora_info?name=" + encodeURIComponent(name))).json();
  const backdrop = el("div", { position: "fixed", inset: "0", zIndex: 10001, background: "rgba(0,0,0,.5)", display: "flex", alignItems: "center", justifyContent: "center" });
  const card = el("div", { ...POPUP, position: "relative", width: "min(640px, 92vw)", maxHeight: "86vh", overflowY: "auto", padding: "14px", display: "flex", flexDirection: "column", gap: "8px" });
  const field = (label, text) => text && card.append(el("div", { whiteSpace: "pre-wrap", wordBreak: "break-word" }, { textContent: `${label}${text}` }));
  card.append(el("div", { fontSize: "15px", fontWeight: "bold" }, { textContent: info.title || name }));
  if (info.preview) card.append(el("img", { maxWidth: "100%", maxHeight: "280px", objectFit: "contain", borderRadius: "4px" }, { src: api.apiURL("/promptforge/lora_preview?name=" + encodeURIComponent(name)) }));
  field("File: ", info.name);
  field("Base model: ", info.base_model);
  field("Trigger words: ", (info.words || []).join(", "));
  field("Usage tips: ", typeof info.tips === "string" ? info.tips : JSON.stringify(info.tips));
  field("Tags: ", (info.tags || []).join(", "));
  field("", info.description);
  if (info.url) card.append(el("a", { color: "#6af" }, { href: info.url, target: "_blank", rel: "noopener", textContent: "Open on CivitAI" }));
  if (!info.words?.length && !info.description && !info.url) field("", "No CivitAI information stored for this LoRA.");
  const close = el("button", { alignSelf: "flex-end", cursor: "pointer" }, { textContent: "Close" });
  close.addEventListener("click", () => backdrop.remove());
  backdrop.addEventListener("click", (event) => { if (event.target === backdrop) backdrop.remove(); });
  card.append(close);
  backdrop.append(card);
  document.body.append(backdrop);
}

function buildList(node, stateWidget) {
  let state = parseState(stateWidget.value);
  let library = { loras: [], families: [] };
  let search = "";
  let activeOnly = false;

  // the frontend sizes `frame`; `root` fills it without ever influencing its size
  const frame = el("div", { position: "relative", width: "100%", height: "100%", minWidth: "0", overflow: "hidden", contain: "layout paint" });
  const root = el("div", { position: "absolute", inset: "0", display: "flex", flexDirection: "column", gap: "4px", boxSizing: "border-box", overflow: "hidden", fontSize: "12px", color: "var(--input-text)" });
  frame.append(root);
  frame.addEventListener("pointerdown", (event) => event.stopPropagation());
  const bar = el("div", { display: "flex", gap: "4px" });
  const searchBox = el("input", { flex: "1", minWidth: "0", background: "var(--comfy-input-bg)", color: "var(--input-text)", border: "1px solid var(--border-color)", borderRadius: "3px", padding: "3px 6px" },
    { type: "search", placeholder: "Search LoRAs (name, folder, trigger words)..." });
  const familyBox = el("select", { background: "var(--comfy-input-bg)", color: "var(--input-text)", border: "1px solid var(--border-color)", borderRadius: "3px", maxWidth: "140px" });
  const counter = el("span", { opacity: "0.7", whiteSpace: "nowrap", alignSelf: "center" });
  const refresh = el("button", { cursor: "pointer" }, { textContent: "↻", title: "Rescan the loras folders" });
  const onlyLabel = el("label", { display: "flex", alignItems: "center", gap: "2px", whiteSpace: "nowrap" });
  const onlyBox = el("input", {}, { type: "checkbox" });
  onlyLabel.append(onlyBox, "on only");
  bar.append(searchBox, familyBox, onlyLabel, counter, refresh);
  const list = el("div", { flex: "1", overflowY: "auto", overflowX: "hidden", minHeight: "0", minWidth: "0", border: "1px solid var(--border-color)", borderRadius: "3px" });
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
    const line = el("div", { display: "flex", alignItems: "center", gap: "4px", padding: "2px 4px", borderBottom: "1px solid var(--border-color)", flexWrap: "nowrap", minWidth: "0", overflow: "hidden" });
    const box = el("input", {}, { type: "checkbox", checked: !!item?.on });
    box.addEventListener("change", () => { itemFor(entry.name).on = box.checked; commit(); render(); });
    const label = el("span", { flex: "1 1 0", minWidth: "40px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", opacity: entry.missing ? "0.5" : "1" }, { textContent: entry.name.replace(/\.[^.]+$/, "") });
    label.addEventListener("mouseenter", (event) => showTip(event, [entry.name, ...(entry.words?.length ? ["Trigger: " + entry.words.join(", ")] : [])]));
    label.addEventListener("mousemove", (event) => showTip(event, [entry.name, ...(entry.words?.length ? ["Trigger: " + entry.words.join(", ")] : [])]));
    label.addEventListener("mouseleave", hideTip);
    const infoButton = el("button", { cursor: "pointer", flex: "0 0 auto", padding: "0 5px", borderRadius: "50%", fontWeight: "bold", fontStyle: "italic" }, { textContent: "i", title: "LoRA information" });
    infoButton.addEventListener("click", () => { hideTip(); showInfo(entry.name); });
    line.append(box, label, infoButton);
    if (entry.family) {
      const tag = library.families.find((family) => family.key === entry.family);
      line.append(el("span", { opacity: "0.6", fontSize: "10px" }, { textContent: tag?.label || entry.family }));
    }
    if (entry.missing) line.append(el("span", { color: "#e66" }, { textContent: "missing" }));
    if (item?.on) {
      const set = (key) => (value) => { item[key] = value; commit(); };
      const group = el("input", { ...INPUT_STYLE, width: "90px", flex: "0 0 auto" },
        { value: item.group || "", placeholder: "group", title: "Items sharing a group name are alternatives: one is picked per run" });
      group.addEventListener("change", () => { item.group = group.value.trim(); commit(); });
      const dice = el("button", { cursor: "pointer", opacity: item.random ? "1" : "0.4" }, { textContent: "🎲", title: "Random strength between min and max" });
      dice.addEventListener("click", () => { item.random = !item.random; commit(); render(); });
      line.append(group, dice);
      if (item.random) {
        line.append(numberInput(item.min, (v) => set("min")(v ?? 0)), "–", numberInput(item.max, (v) => set("max")(v ?? 1)));
      } else {
        line.append(numberInput(item.model, (v) => set("model")(v ?? 1), { placeholder: "model" }), numberInput(item.clip, set("clip"), { placeholder: "= model" }));
      }
    }
    return line;
  }

  const wanted = () => (state.filter === "pipe" ? node.promptforgePipeFamily?.() : state.filter === "all" ? "" : state.filter);

  function render() {
    const scroll = list.scrollTop;
    const known = new Map(library.loras.map((entry) => [entry.name, entry]));
    const configured = byName();
    const query = search.toLowerCase().split(/\s+/).filter(Boolean);
    list.replaceChildren();

    const matches = library.loras.filter((entry) => {
      if (activeOnly && !configured.get(entry.name)?.on) return false;
      if (wanted() && entry.family !== wanted()) return false;
      const haystack = (entry.name + " " + (entry.words || []).join(" ")).toLowerCase();
      return query.every((word) => haystack.includes(word));
    });
    const missing = state.items.filter((item) => item.on && !known.has(item.name)).map((item) => ({ name: item.name, family: "", words: [], missing: true }));
    [...missing, ...matches.slice(0, MAX_ROWS)].forEach((entry) => list.append(row(entry, configured.get(entry.name))));
    if (matches.length > MAX_ROWS) list.append(el("div", { padding: "4px", opacity: "0.6" }, { textContent: `${matches.length - MAX_ROWS} more - narrow the search or pick a family` }));
    counter.textContent = `${state.items.filter((item) => item.on).length} on · ${matches.length} shown`;
    list.scrollTop = scroll;
  }

  function fillFamilies() {
    familyBox.replaceChildren(new Option("All families", "all"), new Option("Follow the connected loader", "pipe"), ...library.families.map((family) => new Option(family.label, family.key)));
    if (!library.families.length && state.filter !== "all" && state.filter !== "pipe") familyBox.append(new Option(state.filter, state.filter));
    familyBox.value = state.filter;
    if (familyBox.value !== state.filter) {
      familyBox.value = "all";
      if (library.families.length) state.filter = "all";
    }
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

  const widget = node.addDOMWidget("lora_list", "promptforge_lora_list", frame, { serialize: false, getMinHeight: () => 380 });
  widget.refreshFilter = () => { if (state.filter === "pipe") render(); };
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
      this.promptforgePipeFamily = () => {
        const link = app.graph.links[this.inputs?.find((input) => input.name === "pipe")?.link];
        return app.graph.getNodeById(link?.origin_id)?.promptforgeFamily?.() ?? "";
      };
      this.promptforgeOnPipe = () => this.promptforgeList.refreshFilter();
      const onResize = this.onResize;
      this.onResize = function (size) {
        size[0] = Math.max(size[0], 520);
        return onResize?.apply(this, arguments);
      };
      this.setSize([Math.max(this.size[0], 620), Math.max(this.size[1], 560)]);
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      this.promptforgeList?.restore();
      return result;
    };
  },
});
