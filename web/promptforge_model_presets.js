import { app } from "../../scripts/app.js";
import { parsePresets } from "./promptforge_v2_presets.js";

const NODE = "PROMPTFORGE_V2_IMAGE_MODEL";
const PRESET_NODE = "PROMPTFORGE_V2_PRESETS";
const SAMPLING = "PROMPTFORGE_V2_IMAGE_SAMPLING_SETTINGS";
const ENCODE = "PROMPTFORGE_V2_IMAGE_ENCODE";
const FORMAT = "PROMPTFORGE_V2_IMAGE_FORMAT";
const TAG = /\[[a-z]+\]\s+/gi;

// Only these snapshot fields are written. Format keeps `sizing` and `preset`
// first because the preset widget rewrites the size widgets from its own table.
const FIELDS = new Map([
  [SAMPLING, ["steps", "cfg", "sampler_name", "scheduler"]],
  [ENCODE, ["clip_skip"]],
  [FORMAT, ["sizing", "preset", "aspect", "megapixels", "orientation", "width", "height"]],
]);

const CSS = `
.pfmp-tools{display:flex;gap:4px;align-items:center;flex:0 0 auto}
.pfmp-tools[hidden]{display:none}
.pfmp-use{cursor:pointer;padding:3px 6px;border-radius:4px;border:1px solid #4b6b84;background:#23384a;color:#cfe8fb;font:11px system-ui,sans-serif;white-space:nowrap}
.pfmp-pick{max-width:150px;background:var(--comfy-input-bg,#1b2026);color:var(--input-text,#e7eef5);border:1px solid #46545f;border-radius:4px;padding:2px 3px;font:11px system-ui,sans-serif}
.pfmp-status{flex:0 0 100%;font:11px/1.25 system-ui,sans-serif;color:#9fc7e4;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pfmp-status[hidden]{display:none}
`;

const typeOf = (node) => node.comfyClass || node.type;
const widget = (node, name) => node.widgets?.find((entry) => entry.name === name);
const tagless = (value) => String(value || "").replace(TAG, "").trim();
const pathOf = (value) => tagless(value).replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
const baseOf = (value) => tagless(value).split("/").pop() || tagless(value);
const connected = (node, name) => node.inputs?.some((input) => (input.name === name || input.widget?.name === name) && input.link != null);

function element(tag, props = {}) {
  return Object.assign(document.createElement(tag), props);
}
function style() {
  if (document.getElementById("pfmp-css")) return;
  document.head.append(element("style", { id: "pfmp-css", textContent: CSS }));
}

// A snapshot names its checkpoint in the captured IMAGE_MODEL widgets, which
// is the shape capturePreset writes. settings.models is the pipe's field name,
// so hand-written stores that use it are read too.
function modelOf(preset) {
  const listed = preset.settings?.models?.model;
  if (typeof listed === "string" && listed.trim()) return listed;
  const saved = preset.nodes.find((entry) => entry.type === NODE)?.widgets?.model;
  return typeof saved === "string" ? saved : "";
}

// Match on the normalized file path suffix: the store may keep a full path
// while the widget keeps the folder-relative one, or the reverse.
function sameModel(saved, current) {
  const a = pathOf(saved);
  const b = pathOf(current);
  if (!a || !b) return false;
  return a === b || b.endsWith(`/${a}`) || a.endsWith(`/${b}`);
}

function matchesFor(store, current) {
  let data;
  try { data = parsePresets(store); }
  catch { return []; }
  return Object.entries(data.presets).filter(([, preset]) => sameModel(modelOf(preset), current));
}

// Prefer the capture role so two encodes or two sampling nodes stay apart;
// fall back to the only node of that type, and skip ambiguous ones.
function targetOf(nodes, saved) {
  const typed = nodes.filter((entry) => typeOf(entry) === saved.type);
  if (saved.role) {
    const roled = typed.filter((entry) => entry.properties?.promptforge_role === saved.role);
    if (roled.length === 1) return roled[0];
  }
  return typed.length === 1 ? typed[0] : null;
}

function applySettings(node, name, preset) {
  const graph = node.graph;
  const nodes = graph?._nodes || [];
  const model = widget(node, "model");
  let written = 0;
  graph?.beforeChange?.();
  try {
    for (const [type, fields] of FIELDS) {
      const saved = preset.nodes.find((entry) => entry.type === type);
      if (!saved) continue;
      const target = targetOf(nodes, saved);
      if (!target) continue;
      for (const field of fields) {
        if (!Object.hasOwn(saved.widgets, field)) continue;
        const value = saved.widgets[field];
        if (connected(target, field)) continue;
        const control = target.widgets?.find((entry) => entry.name === field && entry.options?.serialize !== false);
        if (!control) continue;
        if (control.type === "combo" && Array.isArray(control.options?.values) && !control.options.values.includes(value)) continue;
        control.value = value;
        control.callback?.(value);
        written++;
      }
      target.promptforgeV2Format?.();
    }
  } finally { graph?.afterChange?.(); }

  node.setDirtyCanvas?.(true, true);
  graph?.setDirtyCanvas?.(true, true);
  const status = node.promptforgeModelPresets?.status;
  if (status) {
    status.hidden = false;
    status.textContent = written
      ? `applied ${new Date().toLocaleString()} for ${baseOf(model?.value)}`
      : `no replaceable settings in "${name}"`;
    status.title = status.textContent;
  }
}

function refresh(node) {
  const controls = node.promptforgeModelPresets;
  if (!controls) return;
  const presetsNode = node.graph?._nodes?.find((entry) => typeOf(entry) === PRESET_NODE);
  const current = widget(node, "model")?.value || "";
  const entry = presetsNode && widget(presetsNode, "presets");
  // typing in the presets JSON is a plain textarea edit, hook its callback once
  if (entry && !entry.promptforgeModelPresetsBound) {
    entry.promptforgeModelPresetsBound = true;
    const callback = entry.callback;
    entry.callback = function () {
      const result = callback?.apply(this, arguments);
      refresh(node);
      return result;
    };
  }
  const store = entry?.value;
  const matches = store && current && !current.startsWith("(") ? matchesFor(store, current) : [];
  controls.matches = matches;
  controls.tools.hidden = !matches.length;
  controls.use.hidden = matches.length !== 1;
  controls.pick.hidden = matches.length < 2;
  if (matches.length > 1) {
    controls.pick.replaceChildren(new Option("Choose a saved set…", ""), ...matches.map(([key]) => new Option(key, key)));
    controls.pick.value = "";
  }
  // keep a fresh confirmation visible, drop a stale one
  if (!matches.length) controls.status.hidden = true;
}

// Widget callbacks refresh directly; a redraw also catches node adds/removes,
// so the cheap signature compare keeps the last resort in sync.
function refreshIfChanged(node) {
  const controls = node.promptforgeModelPresets;
  if (!controls) return;
  const presetsNode = node.graph?._nodes?.find((entry) => typeOf(entry) === PRESET_NODE);
  const store = presetsNode ? widget(presetsNode, "presets")?.value : null;
  const model = widget(node, "model")?.value;
  const last = controls.signature;
  if (last && last.node === presetsNode && last.store === store && last.model === model) return;
  controls.signature = { node: presetsNode, store, model };
  refresh(node);
}

function unbind(node) {
  node.graph?.promptforgeModelPresetWatchers?.delete(node);
}

// Node add/remove is the only reliable signal for the presets node appearing
// or leaving, including when the model node itself is off screen.
function watchGraph(node) {
  const graph = node.graph;
  if (!graph) return;
  graph.promptforgeModelPresetWatchers ||= new Set();
  graph.promptforgeModelPresetWatchers.add(node);
  if (graph.promptforgeModelPresetHooked) return;
  graph.promptforgeModelPresetHooked = true;
  for (const key of ["onNodeAdded", "onNodeRemoved"]) {
    const callback = graph[key];
    graph[key] = function () {
      const result = callback?.apply(this, arguments);
      for (const watcher of this.promptforgeModelPresetWatchers) if (watcher.graph) refresh(watcher);
      return result;
    };
  }
}

function bind(node) {
  if (node.promptforgeModelPresets) return;
  const row = node.widgets?.find((entry) => entry.name === "model_picker")?.element;
  if (!row) return;
  style();

  const tools = element("span", { className: "pfmp-tools" });
  const use = element("button", { type: "button", textContent: "Use saved settings", title: "Write the saved generation settings for this model into this workflow" });
  use.className = "pfmp-use";
  const pick = element("select", { ariaLabel: "Saved settings for this model", title: "Several saved settings match this model" });
  pick.className = "pfmp-pick";
  pick.hidden = true;
  tools.append(use, pick);
  const status = element("div", { className: "pfmp-status", hidden: true });

  // the row's own 48px leaves room for the confirmation line
  row.style.flexWrap = "wrap";
  row.style.rowGap = "2px";
  row.append(tools, status);

  node.promptforgeModelPresets = { tools, use, pick, status, matches: [], signature: null };

  use.addEventListener("click", () => {
    const [name, preset] = node.promptforgeModelPresets.matches[0] || [];
    if (preset) applySettings(node, name, preset);
  });
  pick.addEventListener("change", () => {
    const found = node.promptforgeModelPresets.matches.find(([name]) => name === pick.value);
    if (found) applySettings(node, ...found);
  });
  // the model select lives in this row and bubbles its change
  row.addEventListener("change", () => refresh(node));

  // programmatic model writes (family filter, configure) do not touch the DOM
  const model = widget(node, "model");
  const callback = model.callback;
  model.callback = function () {
    const result = callback?.apply(this, arguments);
    refresh(node);
    return result;
  };

  watchGraph(node);
  refresh(node);
}

app.registerExtension({
  name: "PromptForge.V2ImageModelPresets",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      const node = this;
      // the picker row comes from the model UI, which may not have run yet
      let tries = 0;
      const attach = () => {
        bind(node);
        if (!node.promptforgeModelPresets && ++tries < 40) requestAnimationFrame(attach);
      };
      attach();
      return result;
    };
    // off-screen nodes never draw, so the draw hook is only a late fallback
    const draw = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function () {
      bind(this);
      refreshIfChanged(this);
      return draw?.apply(this, arguments);
    };
    const removed = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      unbind(this);
      return removed?.apply(this, arguments);
    };
  },
});
