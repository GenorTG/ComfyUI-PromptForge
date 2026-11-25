import { app } from "../../scripts/app.js";
import { upstreamNodes } from "./promptforge_v2_presets.js";

const NODE = "PROMPTFORGE_V2_IMAGE_FORMAT";
const MODEL = "PROMPTFORGE_V2_IMAGE_MODEL";
const MODES = [["presets", "Presets"], ["generic", "Generic"], ["megapixels", "Megapixels"], ["custom", "Custom"]];
const FAMILY_PREFIX = { krea2: "krea ", anima: "anima ", moss_image: "moss " };
const FAMILY_DEFAULT = "krea2";
const H3_NODE = "PROMPTFORGE_V2_H3_FORMAT";
const H3_UPSCALE = "PROMPTFORGE_V2_H3_UPSCALE_SETTINGS";
const H3_LATENT_UPSCALE = "PROMPTFORGE_V2_H3_LATENT_UPSCALE";
const H3_REFERENCES = "PROMPTFORGE_V2_H3_REFERENCES";

// name: width, height, aspect, orientation. The backend carries the same table; preset maths derives the megapixels.
const table = (rows) => Object.fromEntries(rows.map(([name, width, height, aspect, orientation]) =>
  [name, { width, height, aspect, orientation, megapixels: +(width * height / 1_000_000).toFixed(6) }]));
const PRESETS = table([
  ["krea square 1024x1024", 1024, 1024, "1:1", "landscape"],
  ["krea landscape 4:3 1152x864", 1152, 864, "4:3", "landscape"],
  ["krea portrait 4:3 864x1152", 864, 1152, "4:3", "portrait"],
  ["krea landscape 3:2 1200x800", 1200, 800, "3:2", "landscape"],
  ["krea portrait 3:2 800x1200", 800, 1200, "3:2", "portrait"],
  ["krea landscape 16:9 1344x768", 1344, 768, "16:9", "landscape"],
  ["krea portrait 16:9 768x1344", 768, 1344, "16:9", "portrait"],
  ["krea landscape 21:9 1456x624", 1456, 624, "21:9", "landscape"],
  ["anima square 1024x1024", 1024, 1024, "1:1", "landscape"],
  ["anima portrait 4:3 864x1152", 864, 1152, "4:3", "portrait"],
  ["anima landscape 4:3 1152x864", 1152, 864, "4:3", "landscape"],
  ["anima portrait 3:2 832x1248", 832, 1248, "3:2", "portrait"],
  ["anima landscape 3:2 1248x832", 1248, 832, "3:2", "landscape"],
  ["anima portrait 16:9 768x1344", 768, 1344, "16:9", "portrait"],
  ["anima landscape 16:9 1344x768", 1344, 768, "16:9", "landscape"],
  ["moss cover 1024x1536", 1024, 1536, "3:2", "portrait"],
  ["moss cover 4:3 960x1280", 960, 1280, "4:3", "portrait"],
  ["moss square 1024x1024", 1024, 1024, "1:1", "landscape"],
  ["moss spread 1536x1024", 1536, 1024, "3:2", "landscape"],
  ["moss spread 4:3 1280x960", 1280, 960, "4:3", "landscape"],
  ["moss cinema 16:9 1344x768", 1344, 768, "16:9", "landscape"],
  ["moss banner 21:9 1680x720", 1680, 720, "21:9", "landscape"],
]);
const GENERIC = table([
  ["generic square 512x512", 512, 512, "1:1", "landscape"],
  ["generic square 768x768", 768, 768, "1:1", "landscape"],
  ["generic square 1024x1024", 1024, 1024, "1:1", "landscape"],
  ["generic hd 1280x720", 1280, 720, "16:9", "landscape"],
  ["generic portrait hd 720x1280", 720, 1280, "16:9", "portrait"],
  ["generic large 1536x864", 1536, 864, "16:9", "landscape"],
  ["generic portrait large 864x1536", 864, 1536, "16:9", "portrait"],
]);

const widget = (node, name) => node.widgets?.find((entry) => entry.name === name);
const typeOf = (node) => node.comfyClass || node.type;
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

let changing = 0;
function change(node, action) {
  // nested changes (mode pick -> preset apply) keep a single undo transaction
  if (!changing++) node.graph?.beforeChange();
  try { action(); } finally { if (!--changing) node.graph?.afterChange(); }
  node.setDirtyCanvas?.(true, true);
}
function modeRow(values, current, pick) {
  const row = element("div");
  // the row stays on one line so no mode button can wrap out of the short panel
  row.style.cssText = "display:flex;gap:4px;flex-wrap:wrap";
  const controls = values.map(([value, title]) => {
    const control = button(title, () => pick(value));
    control.dataset.mode = value;
    control.style.cssText = "flex:0 0 auto;white-space:nowrap;cursor:pointer;font:11px/1.4 sans-serif;padding:3px 7px";
    row.append(control);
    return control;
  });
  return { row, paint: () => {
    for (const control of controls) {
      const selected = control.dataset.mode === current();
      control.style.cssText = `flex:0 0 auto;white-space:nowrap;cursor:pointer;font:11px/1.4 sans-serif;padding:3px 7px;border:1px solid ${selected ? "#78c9ff" : "#65717c"};border-radius:5px;background:${selected ? "#24445a" : "#252b31"};color:#eef5fa`;
      control.setAttribute("aria-pressed", String(selected));
    }
  } };
}

const sizes = new WeakMap();
function hide(item, hidden) {
  if (!item) return;
  if (!sizes.has(item)) sizes.set(item, item.computeSize);
  item.hidden = hidden;
  if (hidden) item.computeSize = () => [0, -4];
  else if (sizes.get(item)) item.computeSize = sizes.get(item);
  else delete item.computeSize;
  if (item.inputEl) item.inputEl.style.display = hidden ? "none" : "";
}

// the size maths below mirrors the backend, including Python's round-half-to-even
const round = (value) => value % 1 === 0.5 && Math.floor(value) % 2 === 0 ? Math.floor(value) : Math.round(value);
const snap = (value, step, least) => Math.max(least, round(value / step) * step);
const linked = (node, names) => node.inputs?.some((input) => input.link != null && names.includes(input.widget?.name || input.name));
const value = (node, name) => widget(node, name)?.value;
const describe = (width, height) => `${width}\u00d7${height} (${(width * height / 1_000_000).toFixed(2)} MP)`;

function downstream(start, type) {
  const seen = new Set();
  const queue = [start];
  while (queue.length) {
    const node = queue.shift();
    for (const output of node.outputs || []) for (const id of output.links || []) {
      const link = node.graph?.links instanceof Map ? node.graph.links.get(id) : node.graph?.links?.[id];
      const next = link && node.graph.getNodeById(link.target_id);
      if (!next || seen.has(next.id)) continue;
      if (typeOf(next) === type) return next;
      seen.add(next.id);
      queue.push(next);
    }
  }
  return null;
}

function imageSize(node, mode) {
  if (linked(node, ["sizing", "preset", "aspect", "megapixels", "orientation", "width", "height"])) return "Size comes from a connected input";
  const entry = (mode === "generic" ? GENERIC : PRESETS)[value(node, "preset")];
  if ((mode === "presets" || mode === "generic") && entry) return `Generates ${describe(entry.width, entry.height)}`;
  const aspect = value(node, "aspect");
  if (mode !== "custom" && aspect !== "custom") {
    const [a, b] = aspect.split(":").map(Number);
    const ratio = value(node, "orientation") === "portrait" ? b / a : a / b;
    const area = value(node, "megapixels") * 1_000_000;
    return `Generates ${describe(snap(Math.sqrt(area * ratio), 16, 16), snap(Math.sqrt(area / ratio), 16, 16))}`;
  }
  return `Generates ${describe(snap(value(node, "width"), 16, 16), snap(value(node, "height"), 16, 16))}`;
}

function h3Size(node) {
  if (linked(node, ["width", "height", "seconds", "aspect", "megapixels", "orientation"])) return "Size comes from a connected input";
  let width = value(node, "width"), height = value(node, "height");
  const aspect = value(node, "aspect"), orientation = value(node, "orientation");
  if (aspect !== "custom") {
    const [a, b] = aspect.split(":").map(Number);
    height = Math.sqrt(value(node, "megapixels") * 1024 * 1024 / (a / b));
    width = height * (a / b);
  }
  if ((orientation === "landscape" && width < height) || (orientation === "portrait" && width > height)) [width, height] = [height, width];
  width = snap(width, 32, 32);
  height = snap(height, 32, 32);
  let frames = Math.max(5, Math.ceil(value(node, "seconds") * 24));
  while (frames % 17 !== 5) frames++;
  const lines = [`Generates ${describe(width, height)}`, `${frames} frames, ${(frames / 24).toFixed(2)} s`];
  const upscale = downstream(node, H3_UPSCALE);
  const latent = downstream(node, H3_LATENT_UPSCALE);
  if (upscale && value(upscale, "enabled") && (!latent || value(latent, "enabled"))) {
    const mode = value(upscale, "mode");
    const percent = value(upscale, "percent") / 100;
    const scale = mode === "percent per side" ? percent : Math.sqrt(mode === "percent simplified" ? percent : value(upscale, "megapixels") * 1_000_000 / (width * height));
    lines.push(`Second pass ${describe(snap(width * scale, 32, 64), snap(height * scale, 32, 64))}`);
  } else if (upscale) lines.push("Second pass off");
  const references = downstream(node, H3_REFERENCES);
  if (references && ["i2v", "l2v", "fl2v"].includes(value(references, "mode")) && value(references, "frame_fit") === "match first frame") {
    lines.push("The input frame replaces this aspect (frame_fit)");
  }
  return lines.join("\n");
}

// widgets here and downstream change the result, so the line is re-read on a short timer
function sizeLine(node, text) {
  const line = element("div");
  line.style.cssText = "white-space:pre-line;line-height:1.35;opacity:.9";
  const update = () => {
    if (!node.graph) return;
    const next = text();
    if (line.textContent !== next) line.textContent = next;
  };
  const timer = setInterval(update, 500);
  const removed = node.onRemoved;
  node.onRemoved = function () { clearInterval(timer); return removed?.apply(this, arguments); };
  return { line, update };
}

function h3FormatUI(node) {
  const aspect = widget(node, "aspect");
  let lastAspect = "16:9";
  const mode = () => aspect.value === "custom" ? "custom" : "megapixels";
  const root = panel(node, "format_controls", 108);
  const modes = modeRow([["megapixels", "Megapixels"], ["custom", "Custom"]], mode, (picked) => {
    if (picked === mode()) return;
    change(node, () => {
      aspect.value = picked === "custom" ? "custom" : lastAspect;
      aspect.callback?.(aspect.value);
    });
  });
  const size = sizeLine(node, () => h3Size(node));
  function refresh() {
    const custom = mode() === "custom";
    if (!custom) lastAspect = aspect.value;
    hide(aspect, custom);
    hide(widget(node, "megapixels"), custom);
    hide(widget(node, "width"), !custom);
    hide(widget(node, "height"), !custom);
    modes.paint();
    size.update();
    node.setDirtyCanvas?.(true, true);
  }
  const callback = aspect.callback;
  aspect.callback = function () {
    const result = callback?.apply(this, arguments);
    refresh();
    return result;
  };
  root.append(modes.row, size.line);
  refresh();
  return refresh;
}

function formatUI(node) {
  const sizing = widget(node, "sizing");
  const preset = widget(node, "preset");
  const mode = () => (MODES.some(([value]) => value === sizing?.value) ? sizing.value : "megapixels");
  const family = () => {
    const model = upstreamNodes(node).find((entry) => typeOf(entry) === MODEL);
    const value = model && widget(model, "family")?.value;
    return Object.hasOwn(FAMILY_PREFIX, value) ? value : FAMILY_DEFAULT;
  };
  const options = () => mode() === "generic" ? Object.keys(GENERIC) : Object.keys(PRESETS).filter((name) => name.startsWith(FAMILY_PREFIX[family()]));
  const apply = (name) => {
    const entry = PRESETS[name] || GENERIC[name];
    if (!entry) return;
    change(node, () => {
      for (const [key, value] of Object.entries(entry)) {
        const target = widget(node, key);
        if (!target) continue;
        target.value = value;
        target.callback?.(value);
      }
    });
  };

  const root = panel(node, "format_controls", 54);
  const size = sizeLine(node, () => imageSize(node, mode()));
  const modes = modeRow(MODES, mode, (value) => {
    if (!sizing) return;
    // always write: a hidden combo can hold a stale or invalid value
    change(node, () => {
      sizing.value = value;
      sizing.callback?.(value);
      refresh();
      if (value !== "presets" && value !== "generic") return;
      if (preset && !options().includes(preset.value)) preset.value = options()[0];
      apply(preset?.value);
    });
  });

  function refresh() {
    // heal a missing or stale sizing value so the row and the backend never disagree
    if (sizing && !MODES.some(([value]) => value === sizing.value)) sizing.value = "megapixels";
    const active = mode();
    hide(sizing, true);
    hide(preset, active !== "presets" && active !== "generic");
    hide(widget(node, "aspect"), active !== "megapixels");
    hide(widget(node, "megapixels"), active !== "megapixels");
    hide(widget(node, "orientation"), active !== "megapixels");
    hide(widget(node, "width"), active !== "custom");
    hide(widget(node, "height"), active !== "custom");
    // a family change only changes the list: the picked size stays until the user picks another
    if (preset?.options) preset.options.values = options();
    modes.paint();
    size.update();
    node.setDirtyCanvas?.(true, true);
  }
  function bindFamily() {
    const model = upstreamNodes(node).find((entry) => typeOf(entry) === MODEL);
    const target = model && widget(model, "family");
    if (!target || target.promptforgeFormatBound) return;
    target.promptforgeFormatBound = true;
    const callback = target.callback;
    target.callback = function () {
      const result = callback?.apply(this, arguments);
      refresh();
      return result;
    };
  }

  root.append(modes.row, size.line);
  const callback = preset?.callback;
  if (preset) preset.callback = function (value) {
    apply(value);
    return callback?.apply(this, arguments);
  };
  refresh();
  bindFamily();
  return () => { bindFamily(); refresh(); };
}

app.registerExtension({
  name: "PromptForge.V2Format",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === H3_NODE) {
      const created = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = created?.apply(this, arguments);
        if (!this.promptforgeH3Format) this.promptforgeH3Format = h3FormatUI(this);
        return result;
      };
      const configure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const result = configure?.apply(this, arguments);
        requestAnimationFrame(() => this.promptforgeH3Format?.());
        return result;
      };
      return;
    }
    if (nodeData.name !== NODE) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      if (!this.promptforgeV2Format) this.promptforgeV2Format = formatUI(this);
      return result;
    };
    const configure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      // workflows saved before the sizing widget hold [aspect, megapixels, orientation, width, height, batch_size]
      const values = info?.widgets_values;
      const legacy = Array.isArray(values) && values.length === 6 && !MODES.some(([value]) => value === values[0]);
      const result = configure?.apply(this, arguments);
      requestAnimationFrame(() => {
        if (legacy) {
          ["aspect", "megapixels", "orientation", "width", "height", "batch_size"].forEach((name, index) => {
            const target = widget(this, name);
            if (target) target.value = values[index];
          });
          const sizing = widget(this, "sizing");
          if (sizing) sizing.value = "megapixels";
        }
        this.promptforgeV2Format?.();
      });
      return result;
    };
    const connections = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const result = connections?.apply(this, arguments);
      this.promptforgeV2Format?.();
      return result;
    };
  },
});
