import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { parseLoras, serializeLoras } from "./promptforge_v2_presets.js";

const NODE = "PROMPTFORGE_V2_LORA_STACK";
const PIPE = "PROMPTFORGE_V2_PIPE";
const MAX_ROWS = 200;

const FAMILIES = { krea2: "Krea 2", moss_image: "MOSS Image", anima: "Anima", minimax_h3: "MiniMax H3" };

const CSS = `
.pflora-panel{display:flex;flex-direction:column;gap:5px;color:var(--input-text,#e7eef5);font:12px system-ui,sans-serif}
.pflora-bar{display:flex;gap:5px;align-items:center;flex-wrap:wrap}
.pflora-chip{align-self:flex-start;padding:4px 9px;border-radius:5px;background:#22303c;border:1px solid #35506a;color:#cfe6fa;font-weight:600}
.pflora-btn{cursor:pointer;padding:3px 8px;border-radius:4px;border:1px solid #46545f;background:#28323a;color:#e7eef5}
.pflora-btn:hover{background:#33414c}
.pflora-btn:disabled{opacity:.45;cursor:default}
.pflora-list{flex:1 1 auto;min-height:40px;max-height:360px;overflow-y:auto;overflow-x:hidden;border:1px solid #33414c;border-radius:5px;background:#13191e}
.pflora-searchbox{flex:0 0 auto;width:100%;box-sizing:border-box;background:#0b1014;color:#e7eef5;border:1px solid #46545f;border-radius:4px;padding:3px 6px}
.pflora-scope[aria-pressed="true"]{border-color:#4a86b8;background:#2e5f86;color:#fff}
.pflora-row{display:flex;gap:5px;align-items:center;padding:4px 5px;border-bottom:1px solid #26323b;background:#1b232a}
.pflora-row.pflora-off .pflora-name{opacity:.5}
.pflora-row.pflora-dragging{opacity:.55}
.pflora-row.pflora-drop-before{box-shadow:inset 0 2px 0 #58a6ff}
.pflora-row.pflora-drop-after{box-shadow:inset 0 -2px 0 #58a6ff}
.pflora-handle{cursor:grab;color:#7d8b98;padding:0 2px;user-select:none}
.pflora-name{flex:1;min-width:50px;overflow-wrap:anywhere;cursor:pointer}
.pflora-num{width:62px;flex:0 0 auto;background:#0f1419;color:#e7eef5;border:1px solid #3b4a56;border-radius:3px;padding:2px 4px;box-sizing:border-box}
.pflora-range-num{width:52px}
.pflora-range{width:80px;text-align:center;cursor:default;opacity:.9}
.pflora-dice{flex:0 0 auto;display:flex;align-items:center;gap:2px;cursor:pointer;opacity:.45;user-select:none}
.pflora-dice input{margin:0}
.pflora-dice-on{opacity:1}
.pflora-part{flex:0 0 auto;background:#0f1419;color:#e7eef5;border:1px solid #3b4a56;border-radius:3px}
.pflora-info-btn{flex:0 0 auto;cursor:pointer;width:19px;height:19px;border-radius:50%;border:1px solid #4b5c69;background:#22303c;color:#bfe0f7;font-style:italic;font-weight:700;line-height:1}
.pflora-remove{flex:0 0 auto;cursor:pointer;border:0;background:none;color:#e88;font-size:13px}
.pflora-status{font-size:11px;opacity:.75;min-height:14px}
.pflora-empty{padding:14px;opacity:.7;text-align:center}
.pflora-pop{position:fixed;z-index:10002;display:flex;flex-direction:column;gap:6px;padding:10px;box-sizing:border-box;background:#11171c;border:1px solid #4a5b69;border-radius:8px;box-shadow:0 12px 34px #000b;color:#e7eef5;font:12px system-ui,sans-serif}
.pflora-pop .pflora-search{flex:0 0 auto;background:#0b1014;color:#e7eef5;border:1px solid #46545f;border-radius:4px;padding:5px 7px}
.pflora-pop .pflora-list{background:#0c1115}
.pflora-pop-row{display:flex;gap:6px;align-items:center;padding:4px 5px;border-bottom:1px solid #212c34;background:#141b21;cursor:pointer}
.pflora-pop-row:hover{background:#1d2833}
.pflora-pop-row .pflora-name{font-weight:600}
.pflora-tag{font-size:10px;padding:1px 5px;border-radius:8px;background:#2c3a45;color:#b9d3e4;white-space:nowrap}
.pflora-overlay{position:fixed;inset:0;z-index:10003;display:flex;align-items:center;justify-content:center;background:#000a}
.pflora-info{position:relative;width:min(560px,calc(100vw - 16px));height:min(520px,calc(100vh - 16px));display:flex;flex-direction:column;padding:12px;box-sizing:border-box;background:#141a20;border:1px solid #4a5b69;border-radius:8px;box-shadow:0 18px 50px #000b;color:#e7eef5;font:12px system-ui,sans-serif}
.pflora-info-head{display:flex;gap:8px;align-items:flex-start}
.pflora-info-title{flex:1;min-width:0;font-size:14px;font-weight:700;overflow-wrap:anywhere}
.pflora-tabs{display:flex;gap:4px;margin-top:8px;border-bottom:1px solid #2b3742;padding-bottom:6px}
.pflora-tab{cursor:pointer;padding:3px 10px;border-radius:4px;border:1px solid #46545f;background:#202933;color:#dbe8f1}
.pflora-tab.pflora-active{background:#2e5f86;border-color:#4a86b8;color:#fff}
.pflora-info-body{flex:1;min-height:0;overflow-y:auto;padding:8px 2px 0}
.pflora-info-body dl{display:grid;grid-template-columns:120px 1fr;gap:6px 10px;margin:0}
.pflora-info-body dt{opacity:.65}
.pflora-info-body dd{margin:0;overflow-wrap:anywhere}
.pflora-info-body a{color:#6cb6f5}
.pflora-info-body img{max-width:100%;max-height:210px;object-fit:contain;background:#0c1115;border-radius:5px}
.pflora-words{display:flex;flex-wrap:wrap;gap:5px}
.pflora-word{padding:1px 8px;border-radius:4px;background:#26343f;cursor:pointer}
.pflora-notes{white-space:pre-wrap;overflow-wrap:anywhere}
`;

function el(tag, props = {}, style = {}) {
  const element = Object.assign(document.createElement(tag), props);
  Object.assign(element.style, style);
  return element;
}
function button(label, action, title) {
  const control = el("button", { type: "button", textContent: label, title: title || label }, {});
  control.className = "pflora-btn";
  control.addEventListener("click", action);
  return control;
}
function style() {
  if (document.getElementById("pflora-css")) return;
  document.head.append(el("style", { id: "pflora-css", textContent: CSS }));
}
const widget = (node, name) => node.widgets?.find((entry) => entry.name === name);
const text = (value) => value == null || value === "" ? "" : String(value);
const numeric = (value, fallback) => value == null || value === "" || !Number.isFinite(Number(value)) ? fallback : Number(value);

function numberInput(value, onChange, placeholder, clamp) {
  const input = el("input", { type: "number", step: 0.05, value: value ?? "", placeholder: placeholder || "", title: placeholder || "" }, {});
  input.className = "pflora-num";
  const commit = (raw) => {
    let next = raw === "" || raw == null ? null : Number(raw);
    if (next !== null && !Number.isFinite(next)) next = null;
    if (next !== null && clamp) next = Math.max(clamp[0], Math.min(clamp[1], next));
    if (next !== null) input.value = next;
    onChange(next);
  };
  input.addEventListener("change", () => commit(input.value));
  input.addEventListener("wheel", (event) => {
    event.preventDefault();
    event.stopPropagation();
    commit(Math.round(((Number(input.value) || 0) + (event.deltaY < 0 ? 0.05 : -0.05)) * 100) / 100);
  }, { passive: false });
  return input;
}

function normalize(row) {
  return {
    ...row,
    enabled: row.enabled ?? row.on ?? true,
    model_strength: row.model_strength ?? row.model ?? 1,
    clip_strength: row.clip_strength ?? row.clip ?? null,
    random: row.random === true,
    min: numeric(row.min, 0.6),
    max: numeric(row.max, 1),
    part: row.part || "all",
  };
}
function canonical(row) {
  const { on, model, clip, random, min, max, ...rest } = row;
  const strength = (value, fallback) => Number.isFinite(Number(value)) ? Number(value) : fallback;
  const edge = (value, fallback) => Math.max(-2, Math.min(2, numeric(value, fallback)));
  const low = edge(min, 0.6);
  const high = edge(max, 1);
  return {
    ...rest,
    name: row.name,
    enabled: row.enabled !== false,
    model_strength: strength(row.model_strength, 1),
    clip_strength: row.clip_strength == null || row.clip_strength === "" ? null : strength(row.clip_strength, 1),
    random: row.random === true,
    min: Math.min(low, high),
    max: Math.max(low, high),
    part: row.part || "all",
  };
}
const blankRow = (name) => ({ name, enabled: true, model_strength: 1, clip_strength: null, random: false, min: 0.6, max: 1, part: "all" });

function readState(node) {
  try {
    const data = JSON.parse(widget(node, "loras")?.value || "{}");
    return (Array.isArray(data) ? data : parseLoras(data)).map(normalize);
  } catch {
    return [];
  }
}

function edit(node, action, refresh = true) {
  const state = widget(node, "loras");
  const rows = readState(node);
  action(rows);
  node.graph?.beforeChange();
  try { state.value = serializeLoras(state.value, rows.map(canonical)); }
  finally { node.graph?.afterChange(); }
  state.callback?.(state.value);
  node.setDirtyCanvas?.(true, true);
  if (refresh) node.promptforgeV2Refresh?.();
}

function upstreamFamily(node) {
  const graph = node.graph;
  if (!graph) return null;
  const seen = new Set([node.id]);
  const queue = [node];
  while (queue.length) {
    const current = queue.shift();
    for (const input of current.inputs || []) {
      if (input.type !== PIPE || input.link == null) continue;
      const link = graph.links instanceof Map ? graph.links.get(input.link) : graph.links?.[input.link];
      if (!link || seen.has(link.origin_id)) continue;
      seen.add(link.origin_id);
      const parent = graph.getNodeById(link.origin_id);
      if (!parent) continue;
      const type = parent.comfyClass || parent.type;
      if (type === "Reroute" || type === "Reroute (rgthree)") { queue.push(parent); continue; }
      if (type === "PROMPTFORGE_V2_H3_MODEL") return "minimax_h3";
      if (type === "PROMPTFORGE_V2_IMAGE_MODEL") {
        return parent.widgets?.find((entry) => entry.name === "family")?.value || "";
      }
      const family = parent.widgets?.find((entry) => entry.name === "family")?.value;
      if (family) return family;
      queue.push(parent);
    }
  }
  return null;
}

let managerReady = false;
let managerItems = null;
let activePicker = null;
let activeInfo = null;

api.fetchApi("/api/lm/loras/folders").then((response) => {
  if (!response.ok) return;
  managerReady = true;
  app.graph?._nodes?.forEach((entry) => entry.promptforgeV2Refresh?.());
}).catch(() => {});

const stem = (path) => path.replace(/\\/g, "/").replace(/\.[^./]+$/, "").toLowerCase();

async function managerEntry(name) {
  if (!managerReady) return null;
  if (!managerItems) {
    const map = new Map();
    for (let page = 1; page <= 20; page += 1) {
      const response = await api.fetchApi(`/api/lm/loras/list?page=${page}&page_size=100`);
      if (!response.ok) break;
      const data = await response.json();
      for (const item of data.items || []) if (item.file_path) map.set(stem(item.file_path), item);
      if (!data.total_pages || page >= data.total_pages) break;
    }
    managerItems = map;
  }
  const key = stem(name);
  for (const [path, item] of managerItems) if (path.endsWith(key)) return item;
  return null;
}

function closePicker() { activePicker?.remove(); activePicker = null; }
function closeInfo() { activeInfo?.remove(); activeInfo = null; }

/* ---------------------------------------------------------------- info window */

function infoRow(list, label, value) {
  if (value == null || value === "" || (Array.isArray(value) && !value.length)) return;
  list.append(el("dt", { textContent: label }));
  const cell = el("dd");
  cell.append(...(Array.isArray(value) ? value : [text(value)]));
  list.append(cell);
}

function chipRow(list, label, words, hint) {
  if (!words?.length) return;
  const box = el("div", {}, {});
  box.className = "pflora-words";
  const picked = [];
  for (const word of words) {
    const chip = el("span", { textContent: word, title: hint || "Click to collect" }, {});
    chip.className = "pflora-word";
    chip.addEventListener("click", () => {
      const at = picked.indexOf(word);
      if (at < 0) picked.push(word); else picked.splice(at, 1);
      chip.style.background = at < 0 ? "#2f5d80" : "";
    });
    box.append(chip);
  }
  const copy = el("a", { href: "#", textContent: "copy all" }, { marginLeft: "6px" });
  copy.addEventListener("click", async (event) => {
    event.preventDefault();
    await navigator.clipboard.writeText(words.join(", "));
    copy.textContent = "copied";
  });
  infoRow(list, label, [box, copy]);
}

function openInfo(name, entry) {
  closeInfo();
  closePicker();
  const backdrop = el("div", {}, {});
  backdrop.className = "pflora-overlay";
  const card = el("div", {}, {});
  card.className = "pflora-info";
  const head = el("div", {}, {});
  head.className = "pflora-info-head";
  const title = el("div", { textContent: name }, {});
  title.className = "pflora-info-title";
  head.append(title, button("Close", () => closeInfo()));
  const tabs = el("div", {}, {});
  tabs.className = "pflora-tabs";
  const body = el("div", {}, {});
  body.className = "pflora-info-body";
  card.append(head, tabs, body);
  backdrop.append(card);
  document.body.append(backdrop);
  activeInfo = backdrop;
  backdrop.addEventListener("pointerdown", (event) => event.stopPropagation());
  backdrop.addEventListener("click", (event) => { if (event.target === backdrop) closeInfo(); });
  const escape = (event) => {
    if (!backdrop.isConnected) return document.removeEventListener("keydown", escape, true);
    if (event.key !== "Escape") return;
    event.stopPropagation();
    closeInfo();
  };
  document.addEventListener("keydown", escape, true);

  const state = { item: undefined };
  let token = 0;
  const show = async (key, tab) => {
    const mine = ++token;
    for (const other of tabs.children) other.classList.toggle("pflora-active", other === tab);
    body.replaceChildren(el("div", { textContent: "Loading…" }, { opacity: ".7" }));
    if (state.item === undefined) {
      try { state.item = await managerEntry(name); }
      catch { state.item = null; }
    }
    if (mine !== token) return;
    const item = state.item;
    body.replaceChildren();
    if (key === "notes") {
      const notes = el("div", { textContent: item?.notes || "No notes for this file." });
      notes.className = "pflora-notes";
      body.append(notes);
      return;
    }
    if (!item) {
      body.append(el("div", { textContent: "The LoRA Manager library has no record for this file yet." }, { opacity: ".7" }));
    }
    const list = el("dl");
    if (key === "details") {
      if (item?.preview_url) {
        const cell = el("dd", {}, { gridColumn: "1 / -1" });
        cell.append(el("img", { src: new URL(item.preview_url, document.baseURI).href, loading: "lazy", alt: name }));
        list.append(el("dt"), cell);
      }
      infoRow(list, "File", name);
      if (entry?.words?.length) infoRow(list, "Trigger", entry.words.join(", "));
      if (entry?.size) infoRow(list, "Size", `${(entry.size / 1024 ** 2).toFixed(0)} MB`);
      if (item) {
        infoRow(list, "Folder", item.folder || "(root)");
        infoRow(list, "Base model", item.base_model);
        if (item.file_size) infoRow(list, "File size", `${(item.file_size / 1024 ** 2).toFixed(0)} MB`);
        infoRow(list, "Hash (sha256)", item.sha256);
        const link = item.civitai?.id
          ? `https://civitai.com/models/${item.civitai.modelId}?modelVersionId=${item.civitai.id}`
          : item.source_url;
        if (link) {
          const anchor = el("a", { href: link, target: "_blank", rel: "noopener", textContent: "Open on Civitai" });
          infoRow(list, "Source", [anchor]);
        }
      }
      if (!list.children.length) list.append(el("dt"), el("dd", { textContent: "No stored details." }));
    } else {
      if (item?.civitai?.name) infoRow(list, "Version", item.civitai.name);
      if (item) {
        chipRow(list, "Tags", item.tags, "A tag stored by LoRA Manager");
        chipRow(list, "Trained words", item.civitai?.trainedWords, "Click to collect words for copying");
        chipRow(list, "Auto tags", item.auto_tags, "Tags found in the file itself");
        const tips = item.usage_tips && item.usage_tips !== "{}" ? item.usage_tips : "";
        if (tips) infoRow(list, "Usage tips", tips);
      }
      if (!list.children.length) list.append(el("dt"), el("dd", { textContent: "No stored metadata." }));
    }
    body.append(list);
  };

  for (const [key, label] of [["details", "Details"], ["metadata", "Metadata"], ["notes", "Notes"]]) {
    const tab = el("button", { type: "button", textContent: label }, {});
    tab.className = "pflora-tab";
    tab.addEventListener("click", () => show(key, tab));
    tabs.append(tab);
    if (key === "details") show(key, tab);
  }
}

/* -------------------------------------------------------------- picker popup */

function openPicker(node, anchor, entriesOf, reload) {
  closePicker();
  closeInfo();
  const popup = el("div", {}, {});
  popup.className = "pflora-pop";
  const search = el("input", { type: "search", placeholder: "Search LoRAs…", value: "" }, {});
  search.className = "pflora-search";
  const list = el("div", {}, {});
  list.className = "pflora-list";
  const bar = el("div", {}, {});
  bar.className = "pflora-bar";
  const count = el("span", {}, { flex: "1", opacity: ".7" });
  let scope = upstreamFamily(node) ? "family" : "all";
  const scopeButton = (label, value, title) => {
    const control = button(label, () => { scope = value; render(); }, title);
    control.className = "pflora-btn pflora-scope";
    return control;
  };
  const allView = scopeButton("View all", "all", "Show every LoRA on disk");
  const familyView = scopeButton("View this family", "family", "Show only LoRAs for the connected model family");
  bar.append(allView, familyView, count, button("Rescan", () => reload(true)));
  popup.append(search, bar, list);
  document.body.append(popup);
  activePicker = popup;

  const place = () => {
    const rect = anchor.getBoundingClientRect();
    const width = Math.min(430, innerWidth - 16);
    const height = Math.min(440, innerHeight - 24);
    popup.style.width = `${width}px`;
    popup.style.height = `${height}px`;
    popup.style.left = `${Math.min(Math.max(8, rect.left), innerWidth - width - 8)}px`;
    const below = rect.bottom + 6;
    popup.style.top = `${below + height > innerHeight - 8 ? Math.max(8, rect.top - height - 6) : below}px`;
  };
  place();
  addEventListener("resize", place);

  const cleanup = () => {
    removeEventListener("resize", place);
    document.removeEventListener("pointerdown", outside, true);
    document.removeEventListener("keydown", escape, true);
  };
  const outside = (event) => {
    if (!popup.isConnected) return cleanup();
    if (popup.contains(event.target) || anchor.contains(event.target)) return;
    cleanup();
    closePicker();
  };
  const escape = (event) => {
    if (event.key !== "Escape" || !popup.isConnected) return;
    cleanup();
    closePicker();
  };
  setTimeout(() => popup.isConnected && document.addEventListener("pointerdown", outside, true), 0);
  document.addEventListener("keydown", escape, true);
  search.focus();

  const render = () => {
    const query = search.value.toLowerCase().split(/\s+/).filter(Boolean);
    const rows = readState(node);
    const known = new Set(rows.map((row) => row.name));
    const family = upstreamFamily(node);
    if (!family) scope = "all";
    for (const [control, value] of [[allView, "all"], [familyView, "family"]]) {
      control.setAttribute("aria-pressed", String(scope === value));
      control.disabled = value === "family" && !family;
    }
    const entries = entriesOf();
    const matches = entries.filter((entry) => {
      if (scope === "family" && entry.family !== family) return false;
      const haystack = `${entry.name} ${(entry.words || []).join(" ")}`.toLowerCase();
      return query.every((word) => haystack.includes(word));
    }).sort((a, b) => Number(b.family === family) - Number(a.family === family));
    list.replaceChildren();
    if (!matches.length) {
      const empty = el("div", { textContent: "No LoRA matches." });
      empty.className = "pflora-empty";
      list.append(empty);
    }
    for (const entry of matches.slice(0, MAX_ROWS)) {
      const row = el("div", {}, {});
      row.className = "pflora-pop-row";
      const tick = el("input", { type: "checkbox", checked: known.has(entry.name) }, {});
      const name = el("span", { textContent: entry.name, title: entry.words?.length ? `Trigger: ${entry.words.join(", ")}` : entry.name }, {});
      name.className = "pflora-name";
      row.append(tick, name);
      if (entry.family) {
        const tag = el("span", { textContent: FAMILIES[entry.family] || entry.family }, {});
        tag.className = "pflora-tag";
        row.append(tag);
      }
      row.addEventListener("click", (event) => {
        if (event.target.closest("input")) return;
        const present = readState(node).some((item) => item.name === entry.name);
        edit(node, (items) => {
          if (present) items.splice(items.findIndex((item) => item.name === entry.name), 1);
          else items.push(blankRow(entry.name));
        });
        render();
      });
      tick.addEventListener("change", () => {
        edit(node, (items) => {
          const at = items.findIndex((item) => item.name === entry.name);
          if (tick.checked && at < 0) items.push(blankRow(entry.name));
          if (!tick.checked && at >= 0) items.splice(at, 1);
        });
        render();
      });
      list.append(row);
    }
    count.textContent = `${matches.length} show · ${known.size} in stack${scope === "family" && family ? ` · ${FAMILIES[family] || family} only` : ""}`;
  };
  search.addEventListener("input", render);
  popup.addEventListener("pflora-refresh", render);
  render();
}

/* ------------------------------------------------------------------ node UI */

function buildUI(node) {
  if (node.promptforgeLoraBound) return;
  node.promptforgeLoraBound = true;
  style();
  const state = widget(node, "loras");
  if (!state) return;
  const seed = widget(node, "seed");
  if (seed) seed.tooltip = "Seeds the random strengths and the random group picks.";
  state.hidden = true;
  state.computeSize = () => [0, -4];
  if (state.inputEl) state.inputEl.style.display = "none";

  const frame = el("div", {}, {});
  frame.style.cssText = "position:relative;width:100%;height:100%;min-width:0;overflow:auto;contain:layout paint";
  const panel = el("div", {}, { padding: "4px", boxSizing: "border-box" });
  panel.className = "pflora-panel";
  frame.append(panel);
  frame.addEventListener("pointerdown", (event) => event.stopPropagation());
  frame.addEventListener("wheel", (event) => event.stopPropagation());

  const family = el("div", { textContent: "Family: …" }, {});
  family.className = "pflora-chip";
  const bar = el("div", {}, {});
  bar.className = "pflora-bar";
  const mode = widget(node, "layer_controls");
  if (mode) {
    mode.hidden = true;
    mode.computeSize = () => [0, -4];
    if (mode.inputEl) mode.inputEl.style.display = "none";
  }
  const modeSelect = el("select", { ariaLabel: "LoRA target layer controls" }, {});
  modeSelect.className = "pflora-part";
  for (const [value, label] of [["auto", "Auto from model"], ["standard", "Standard model"], ["minimax audiovisual", "MiniMax video + audio"]]) {
    modeSelect.append(new Option(label, value));
  }
  modeSelect.value = mode?.value || "auto";
  const resolvedMode = () => {
    const manual = mode?.value || "auto";
    if (manual !== "auto") return manual;
    return upstreamFamily(node) === "minimax_h3" ? "minimax audiovisual" : "standard";
  };
  modeSelect.addEventListener("change", () => {
    if (!mode) return;
    mode.value = modeSelect.value;
    mode.callback?.(mode.value);
    render();
  });
  const add = button("Add LoRA…", () => openPicker(node, add, () => entries, reload), "Open the LoRA picker");
  const refresh = button("Refresh", () => reload(true), "Rescan the loras folders");
  bar.append(modeSelect, add, refresh);
  const search = el("input", { type: "search", placeholder: "Search the stack…" }, {});
  search.className = "pflora-searchbox";
  let query = "";
  search.addEventListener("input", () => { query = search.value; render(); });
  const list = el("div", {}, {});
  list.className = "pflora-list";
  const status = el("div", {}, {});
  status.className = "pflora-status";
  panel.append(family, bar, search, list, status);
  const MAX_HEIGHT = 640;
  const MIN_HEIGHT = 320;
  const WIDTH = 470;
  let panelHeight = 300;
  let targetHeight = MIN_HEIGHT;
  const frameWidget = node.addDOMWidget("lora_stack_editor", "promptforge_v2", frame, {
    serialize: false, getMinHeight: () => panelHeight, getMaxHeight: () => MAX_HEIGHT });
  const fit = () => {
    const measured = panel.scrollHeight;
    const natural = Number.isFinite(measured) && measured > 0 ? measured + 8 : 300;
    panelHeight = Math.max(120, Math.min(natural, MAX_HEIGHT - 160));
    const full = node.computeSize?.()[1] || panelHeight + 240;
    const overflow = Math.max(0, full - MAX_HEIGHT);
    if (overflow) panelHeight = Math.max(100, panelHeight - overflow);
    targetHeight = Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, full - overflow));
    const width = Math.max(node.size?.[0] || 0, WIDTH);
    if (node.size && Math.abs(node.size[0] - width) < 1 && Math.abs(node.size[1] - targetHeight) < 1) return;
    node.setSize?.([width, targetHeight]);
  };

  let entries = [];

  const refreshFamily = () => {
    const found = upstreamFamily(node);
    const manual = mode?.value !== "auto" ? mode?.value : "";
    if (found === null) family.textContent = "Family: no model upstream";
    else if (found === "") family.textContent = "Family: model without family";
    else family.textContent = `Family: ${FAMILIES[found] || found}${manual ? ` · ${manual}` : ""}`;
  };

  let dragFrom = -1;
  let dragTo = -1;
  const clearDrop = () => {
    for (const row of list.children) row.classList.remove("pflora-drop-before", "pflora-drop-after", "pflora-dragging");
  };

  function row(item, index) {
    const line = el("div", {}, {});
    line.className = item.enabled === false ? "pflora-row pflora-off" : "pflora-row";
    const handle = el("span", { textContent: "⠿", title: "Drag to reorder", draggable: true }, {});
    handle.className = "pflora-handle";
    handle.addEventListener("dragstart", (event) => {
      dragFrom = index;
      line.classList.add("pflora-dragging");
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", String(index));
    });
    handle.addEventListener("dragend", () => { dragFrom = -1; clearDrop(); });
    line.addEventListener("dragover", (event) => {
      if (dragFrom < 0 || dragFrom === index) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
      const box = line.getBoundingClientRect();
      const after = event.clientY > box.top + box.height / 2;
      dragTo = index + (after ? 1 : 0);
      if (dragFrom < dragTo) dragTo -= 1;
      clearDrop();
      line.classList.add(after ? "pflora-drop-after" : "pflora-drop-before");
    });
    line.addEventListener("drop", (event) => {
      event.preventDefault();
      const from = dragFrom;
      const to = dragTo;
      dragFrom = -1;
      clearDrop();
      if (from < 0 || to < 0 || from === to) return;
      edit(node, (items) => { const [moved] = items.splice(from, 1); items.splice(to, 0, moved); });
    });

    const tick = el("input", { type: "checkbox", checked: item.enabled !== false, title: item.enabled === false ? "Disabled" : "Enabled" }, {});
    tick.addEventListener("change", () => edit(node, (items) => { items[index].enabled = tick.checked; }));
    const dice = el("label", { title: "Random strength between min and max" }, {});
    dice.className = item.random ? "pflora-dice pflora-dice-on" : "pflora-dice";
    dice.setAttribute("aria-pressed", String(item.random));
    dice.addEventListener("click", (event) => {
      event.preventDefault();
      edit(node, (items) => { items[index].random = !items[index].random; });
    });
    dice.append(el("input", { type: "checkbox", checked: item.random }), el("span", { textContent: "🎲" }));
    const name = el("span", { textContent: item.name, title: item.name }, {});
    name.className = "pflora-name";
    name.addEventListener("click", () => edit(node, (items) => { items[index].enabled = items[index].enabled === false; }));
    line.append(handle, tick, dice, name);

    if (item.part && resolvedMode() !== "standard") {
      const part = el("select", { title: "Target layers (audiovisual models)" }, {});
      part.className = "pflora-part";
      for (const value of ["all", "video", "audio"]) part.append(new Option(value, value));
      part.value = item.part;
      part.addEventListener("change", () => edit(node, (items) => { items[index].part = part.value; }));
      line.append(part);
    }

    if (item.random) {
      const values = { min: item.min, max: item.max };
      const range = el("input", { type: "text", readOnly: true, value: `${values.min} – ${values.max}`,
        title: "Strength is drawn from this range for every run; clip follows the model" }, {});
      range.className = "pflora-num pflora-range";
      const sync = () => { range.value = `${values.min} – ${values.max}`; };
      const set = (key) => (value) => {
        values[key] = Math.max(-2, Math.min(2, value ?? (key === "min" ? 0.6 : 1)));
        edit(node, (items) => { items[index][key] = values[key]; }, false);
        sync();
      };
      const low = numberInput(values.min, set("min"), "min", [-2, 2]);
      const high = numberInput(values.max, set("max"), "max", [-2, 2]);
      low.className = "pflora-num pflora-range-num";
      high.className = "pflora-num pflora-range-num";
      const order = () => {
        if (values.min <= values.max) return;
        [values.min, values.max] = [values.max, values.min];
        low.value = values.min;
        high.value = values.max;
        edit(node, (items) => { items[index].min = values.min; items[index].max = values.max; }, false);
        sync();
      };
      low.addEventListener("blur", order);
      high.addEventListener("blur", order);
      line.append(low, high, range);
    } else {
      line.append(
        numberInput(item.model_strength, (value) => edit(node, (items) => { items[index].model_strength = value ?? 1; }), "model"),
        numberInput(item.clip_strength, (value) => edit(node, (items) => { items[index].clip_strength = value; }), "= model"),
      );
    }
    if (managerReady) {
      const info = el("button", { type: "button", textContent: "i", title: "LoRA information" }, {});
      info.className = "pflora-info-btn";
      const entry = entries.find((candidate) => candidate.name === item.name);
      info.addEventListener("click", () => openInfo(item.name, entry));
      line.append(info);
    }
    const remove = el("button", { type: "button", textContent: "×", title: "Remove from the stack" }, {});
    remove.className = "pflora-remove";
    remove.addEventListener("click", () => edit(node, (items) => { items.splice(index, 1); }));
    line.append(remove);
    return line;
  }

  function render() {
    const scroll = list.scrollTop;
    const all = readState(node);
    const words = query.toLowerCase().split(/\s+/).filter(Boolean);
    const rows = all
      .map((item, index) => [item, index])
      .filter(([item]) => words.every((word) => item.name.toLowerCase().includes(word)));
    list.replaceChildren();
    if (!all.length) {
      const empty = el("div", { textContent: "No LoRAs in the stack. Use Add LoRA… to pick one." });
      empty.className = "pflora-empty";
      list.append(empty);
    } else if (!rows.length) {
      const empty = el("div", { textContent: `No stack entry matches “${query}”.` });
      empty.className = "pflora-empty";
      list.append(empty);
    }
    for (const [item, index] of rows) list.append(row(item, index));
    list.scrollTop = scroll;
    const on = all.filter((item) => item.enabled !== false).length;
    status.textContent = `${on} enabled · ${all.length} in stack${query ? ` · ${rows.length} shown` : ""} · layer: ${resolvedMode()}`;
    refreshFamily();
    fit();
  }

  async function reload(rescan = false) {
    try {
      const response = await api.fetchApi(rescan ? "/promptforge/v2/loras?refresh=1" : "/promptforge/v2/loras");
      if (!response.ok) throw new Error(`LoRA list unavailable (${response.status})`);
      entries = (await response.json()).loras || [];
      status.textContent = `${entries.length} LoRAs on disk`;
      render();
      if (activePicker && activePicker.isConnected) {
        activePicker.dispatchEvent(new CustomEvent("pflora-refresh"));
      }
    } catch (error) {
      status.textContent = error.message;
    }
  }

  node.promptforgeV2Refresh = render;
  node.promptforgeFamilyLabel = refreshFamily;

  if (mode) {
    const modeCallback = mode.callback;
    mode.callback = function () {
      const result = modeCallback?.apply(this, arguments);
      modeSelect.value = mode.value;
      render();
      return result;
    };
  }

  const previousConnections = node.onConnectionsChange;
  node.onConnectionsChange = function () {
    const result = previousConnections?.apply(this, arguments);
    render();
    return result;
  };
  const previousRemoved = node.onRemoved;
  node.onRemoved = function () {
    closePicker();
    closeInfo();
    return previousRemoved?.apply(this, arguments);
  };

  let familyTick = 0;
  const previousDraw = node.onDrawForeground;
  node.onDrawForeground = function () {
    const result = previousDraw?.apply(this, arguments);
    const now = performance.now();
    if (now - familyTick > 400) { familyTick = now; refreshFamily(); }
    return result;
  };

  const previousResize = node.onResize;
  node.onResize = function (size) {
    size[0] = Math.max(size[0], WIDTH);
    size[1] = targetHeight;
    return previousResize?.apply(this, arguments);
  };
  render();
  requestAnimationFrame(fit);
  if (globalThis.ResizeObserver) {
    const observer = new ResizeObserver(() => fit());
    observer.observe(panel);
    const stop = node.onRemoved;
    node.onRemoved = function () { observer.disconnect(); return stop?.apply(this, arguments); };
  }
  reload();
  // While promptforge_v2.js still carries the old editor, drop its duplicate panel.
  requestAnimationFrame(() => {
    const dupes = (node.widgets || []).filter((entry) => entry !== frameWidget && entry.name?.startsWith("lora_stack_editor"));
    for (const entry of dupes) {
      entry.element?.remove();
      const at = node.widgets.indexOf(entry);
      if (at >= 0) node.widgets.splice(at, 1);
    }
    if (!dupes.length) return;
    frameWidget.name = "lora_stack_editor";
    node.promptforgeV2Refresh = render;
    node.setDirtyCanvas?.(true, true);
  });
}

app.registerExtension({
  name: "PromptForge.V2LoraStack",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "PROMPTFORGE_V2_IMAGE_MODEL") {
      const created = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = created?.apply(this, arguments);
        const node = this;
        const family = this.widgets?.find((entry) => entry.name === "family");
        if (family) {
          const callback = family.callback;
          family.callback = function () {
            const value = callback?.apply(this, arguments);
            node.graph?._nodes?.forEach((entry) => entry.promptforgeFamilyLabel?.());
            return value;
          };
        }
        return result;
      };
      return;
    }
    if (nodeData.name !== NODE) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      buildUI(this);
      return result;
    };
    const configure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = configure?.apply(this, arguments);
      this.promptforgeV2Refresh?.();
      return result;
    };
  },
});
