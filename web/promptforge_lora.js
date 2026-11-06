import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "PROMPTFORGE_LoraLoader";
const MAX_ROWS = 120;
const AUDIO_VIDEO = ["minimax_h3", "ltxv"];

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

const INFO_CSS = `
.pf-info { width: min(980px, 94vw); max-height: 90vh; overflow-y: auto; padding: 16px 18px; box-sizing: border-box; font-size: 13px; line-height: 1.4; }
.pf-info h2 { margin: 0 0 8px; font-size: 18px; }
.pf-info .pf-tags { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 12px; padding: 0; list-style: none; }
.pf-info .pf-tags li { padding: 2px 10px; border-radius: 12px; background: rgba(255,255,255,.1); font-size: 12px; }
.pf-info .pf-tags li.pf-base { background: #2a5d9a; color: #fff; }
.pf-info table { width: 100%; border-collapse: collapse; margin-bottom: 14px; }
.pf-info td { padding: 6px 8px; border-bottom: 1px solid rgba(255,255,255,.1); vertical-align: top; word-break: break-word; }
.pf-info td:first-child { width: 130px; opacity: .65; white-space: nowrap; }
.pf-info a { color: #6af; }
.pf-info .pf-words { display: flex; flex-wrap: wrap; gap: 5px; }
.pf-info .pf-word { padding: 1px 9px; border-radius: 4px; background: rgba(255,255,255,.1); cursor: pointer; border: 1px solid transparent; }
.pf-info .pf-word.pf-on { border-color: #6af; background: rgba(80,140,255,.25); }
.pf-info .pf-copy { margin-left: 8px; cursor: pointer; color: #6af; text-decoration: underline; }
.pf-info .pf-text { white-space: pre-wrap; max-height: 150px; overflow-y: auto; }
.pf-info .pf-gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; margin: 0; padding: 0; list-style: none; }
.pf-info figure { margin: 0; display: flex; flex-direction: column; gap: 4px; background: rgba(0,0,0,.25); border-radius: 6px; overflow: hidden; }
.pf-info figure img, .pf-info figure video { width: 100%; max-height: 380px; object-fit: contain; background: #111; }
.pf-info figcaption { padding: 4px 8px 8px; display: flex; flex-direction: column; gap: 2px; font-size: 11.5px; }
.pf-info figcaption label { opacity: .6; margin-right: 4px; }
.pf-info figcaption .pf-long { max-height: 3.9em; overflow: hidden; cursor: pointer; white-space: pre-wrap; word-break: break-word; }
.pf-info figcaption .pf-long.pf-open { max-height: 260px; overflow-y: auto; }
.pf-info .pf-close { position: sticky; float: right; top: 0; cursor: pointer; }
`;

async function showInfo(name) {
  if (!document.getElementById("pf-info-css")) document.head.append(el("style", {}, { id: "pf-info-css", textContent: INFO_CSS }));
  const info = await (await api.fetchApi("/promptforge/lora_info?name=" + encodeURIComponent(name))).json();
  const backdrop = el("div", { position: "fixed", inset: "0", zIndex: 10001, background: "rgba(0,0,0,.55)", display: "flex", alignItems: "center", justifyContent: "center" });
  const card = el("div", { ...POPUP, position: "relative" }, { className: "pf-info" });
  const done = () => { backdrop.remove(); document.removeEventListener("keydown", onKey, true); };
  const onKey = (event) => { if (event.key === "Escape") { event.stopPropagation(); done(); } };
  document.addEventListener("keydown", onKey, true);
  backdrop.addEventListener("pointerdown", (event) => event.stopPropagation());
  backdrop.addEventListener("click", (event) => { if (event.target === backdrop) done(); });

  const close = el("button", {}, { className: "pf-close", textContent: "Close" });
  close.addEventListener("click", done);
  card.append(close, el("h2", {}, { textContent: info.title || name }));

  const tags = el("ul", {}, { className: "pf-tags" });
  const pill = (text, cls = "") => text && tags.append(el("li", {}, { className: cls, textContent: text }));
  pill(info.type); pill(info.base_model, "pf-base"); pill(info.version);
  pill(info.creator && "by " + info.creator);
  pill(info.downloads != null && "↓ " + info.downloads.toLocaleString());
  pill(info.likes != null && "👍 " + info.likes.toLocaleString());
  card.append(tags);

  const table = el("table");
  const row = (label, content) => {
    if (content == null || content === "" || (Array.isArray(content) && !content.length)) return null;
    const cell = el("td");
    cell.append(...(Array.isArray(content) ? content : [content]));
    const line = el("tr");
    line.append(el("td", {}, { textContent: label }), cell);
    table.append(line);
    return line;
  };
  const text = (value) => el("div", {}, { className: "pf-text", textContent: value });
  row("File", info.name);
  row("Hash (sha256)", info.sha256);
  if (info.size) row("Size", (info.size / 1024 ** 2).toFixed(0) + " MB");
  if (info.url) row("Civitai", el("a", {}, { href: info.url, target: "_blank", rel: "noopener", textContent: "View on Civitai" }));
  const chipRow = (label, words, hint) => {
    const box = el("div", {}, { className: "pf-words" });
    const status = el("span");
    const update = () => {
      status.replaceChildren();
      const picked = [...box.querySelectorAll(".pf-on")].map((chip) => chip.dataset.word);
      if (!picked.length) return;
      status.append(`${picked.length} selected`, el("span", {}, { className: "pf-copy", textContent: "Copy" }));
      status.lastChild.addEventListener("click", async () => { await navigator.clipboard.writeText(picked.join(", ")); status.textContent = "Copied ✓"; });
    };
    words.forEach((word) => {
      const chip = el("span", {}, { className: "pf-word", textContent: word, title: hint });
      chip.dataset.word = word;
      chip.addEventListener("click", () => { chip.classList.toggle("pf-on"); update(); });
      box.append(chip);
    });
    const all = el("span", {}, { className: "pf-copy", textContent: "Copy all" });
    all.addEventListener("click", async () => { await navigator.clipboard.writeText(words.join(", ")); all.textContent = "Copied ✓"; });
    row(label, [box, el("div", { marginTop: "4px" }), status, all]);
  };
  if (info.words?.length) chipRow("Trained words", info.words, "Click to select for copying");
  if (info.training?.length) row("Training", text(info.training.map(([label, value]) => `${label}: ${value}`).join("\n")));
  if (info.training_tags?.length) chipRow(info.words?.length ? "Common training tags" : "Training tags", info.training_tags, "A tag that is common in the training data, from the file itself. Click to select for copying");
  row("Usage tips", text(typeof info.tips === "string" ? info.tips : JSON.stringify(info.tips)));
  row("Tags", (info.tags || []).join(", "));
  row("Notes", info.notes && text(info.notes));
  row("Description", info.description && text(info.description));
  if (!table.children.length || (!info.words?.length && !info.description && !info.url && !info.images?.length)) row("Info", "No CivitAI information stored for this LoRA.");
  card.append(table);

  const gallery = el("ul", {}, { className: "pf-gallery" });
  const shots = [...(info.preview ? [{ url: api.apiURL("/promptforge/lora_preview?name=" + encodeURIComponent(name)) }] : []), ...(info.images || [])];
  for (const shot of shots) {
    const figure = el("figure");
    const media = shot.video
      ? el("video", {}, { src: shot.url, loop: true, muted: true, controls: true, preload: "metadata" })
      : el("img", {}, { src: shot.url, loading: "lazy", referrerPolicy: "no-referrer" });
    figure.append(media);
    const caption = el("figcaption");
    for (const [label, key] of [["seed", "seed"], ["steps", "steps"], ["cfg", "cfg"], ["sampler", "sampler"], ["model", "model"], ["positive", "positive"], ["negative", "negative"]]) {
      if (shot[key] == null || shot[key] === "") continue;
      const line = el("div");
      const long = String(shot[key]).length > 80;
      const value = el("span", {}, { className: long ? "pf-long" : "", textContent: String(shot[key]) });
      if (long) value.addEventListener("click", () => value.classList.toggle("pf-open"));
      line.append(el("label", {}, { textContent: label }), value);
      caption.append(line);
    }
    if (caption.children.length) figure.append(caption);
    gallery.append(el("li", {}, {}));
    gallery.lastChild.append(figure);
  }
  if (shots.length) card.append(gallery);
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
  const familyBox = el("select", { flex: "0 0 140px", width: "140px", background: "var(--comfy-input-bg)", color: "var(--input-text)", border: "1px solid var(--border-color)", borderRadius: "3px", maxWidth: "140px" });
  const counter = el("span", { opacity: "0.7", whiteSpace: "nowrap", alignSelf: "center", flex: "0 0 auto", minWidth: "112px", textAlign: "right" });
  const refresh = el("button", { cursor: "pointer" }, { textContent: "↻", title: "Rescan the loras folders" });
  const onlyLabel = el("label", { display: "flex", alignItems: "center", gap: "2px", whiteSpace: "nowrap" });
  const onlyBox = el("input", {}, { type: "checkbox" });
  onlyLabel.append(onlyBox, "on only");
  bar.append(searchBox, familyBox, onlyLabel, counter, refresh);
  const list = el("div", { flex: "1", overflowY: "auto", overflowX: "hidden", minHeight: "0", minWidth: "0", border: "1px solid var(--border-color)", borderRadius: "3px" });
  list.addEventListener("wheel", (event) => event.stopPropagation());
  root.append(bar, list);

  const commit = () => {
    state.items = state.items.filter((item) => item.on || item.random || item.group || item.model !== 1 || item.clip != null || (item.part && item.part !== "all"));
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
      if (AUDIO_VIDEO.includes(entry.family)) {
        const part = el("select", { ...INPUT_STYLE, width: "64px", flex: "0 0 auto" }, { title: "Which layers of this audio+video model the LoRA changes" });
        part.append(new Option("all", "all"), new Option("video", "video"), new Option("audio", "audio"));
        part.value = item.part || "all";
        part.addEventListener("change", () => { item.part = part.value; commit(); });
        line.append(part);
      }
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
  // the frontend narrows widget.width from layout passes; the list should always span the node
  Object.defineProperty(widget, "width", { get: () => node.size[0], set: () => {}, configurable: true });
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
