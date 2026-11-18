import { app } from "../../scripts/app.js";
import { upstreamNodes } from "./promptforge_v2_presets.js";

const NODE = "PROMPTFORGE_V2_IMAGE_FORMAT";
const MODEL = "PROMPTFORGE_V2_IMAGE_MODEL";
const MODES = [["presets", "Presets"], ["generic", "Generic"], ["megapixels", "Megapixels"], ["custom", "Custom"]];
const FAMILY_PREFIX = { krea2: "krea ", anima: "anima ", moss_image: "moss " };
const FAMILY_DEFAULT = "krea2";

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

function formatUI(node) {
  const sizing = widget(node, "sizing");
  const preset = widget(node, "preset");
  const sizes = new Map();
  const hide = (item, hidden) => {
    if (!item) return;
    if (!sizes.has(item)) sizes.set(item, item.computeSize);
    item.hidden = hidden;
    if (hidden) item.computeSize = () => [0, -4];
    else if (sizes.get(item)) item.computeSize = sizes.get(item);
    else delete item.computeSize;
    if (item.inputEl) item.inputEl.style.display = hidden ? "none" : "";
  };
  const mode = () => (MODES.some(([value]) => value === sizing?.value) ? sizing.value : "megapixels");
  const family = () => {
    const model = upstreamNodes(node).find((entry) => typeOf(entry) === MODEL);
    const value = model && widget(model, "family")?.value;
    return Object.hasOwn(FAMILY_PREFIX, value) ? value : FAMILY_DEFAULT;
  };
  const options = () => mode() === "generic" ? Object.keys(GENERIC) : Object.keys(PRESETS).filter((name) => name.startsWith(FAMILY_PREFIX[family()]));
  const chosen = () => {
    const names = options();
    if (preset?.options) preset.options.values = names;
    if (names.includes(preset?.value)) return;
    if (!preset) return;
    preset.value = names[0];
    // only write the preset into the size widgets while it is the active source
    if (mode() === "presets" || mode() === "generic") preset.callback?.(preset.value);
  };
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

  const root = panel(node, "format_controls", 30);
  const modes = modeRow(MODES, mode, (value) => {
    if (!sizing || mode() === value) return;
    change(node, () => { sizing.value = value; sizing.callback?.(value); });
    refresh();
    if (value === "presets" || value === "generic") apply(preset?.value);
  });

  function refresh() {
    const active = mode();
    hide(sizing, true);
    hide(preset, active !== "presets" && active !== "generic");
    hide(widget(node, "aspect"), active !== "megapixels");
    hide(widget(node, "megapixels"), active !== "megapixels");
    hide(widget(node, "orientation"), active !== "megapixels");
    hide(widget(node, "width"), active !== "custom");
    hide(widget(node, "height"), active !== "custom");
    chosen();
    modes.paint();
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

  root.append(modes.row);
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
