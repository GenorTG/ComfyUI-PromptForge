export const PIPE = "PROMPTFORGE_V2_PIPE";
export const PREFIX = "PROMPTFORGE_V2_";
export const typeOf = (node) => node.comfyClass || node.type;
const clone = (value) => JSON.parse(JSON.stringify(value));
const object = (value) => value && typeof value === "object" && !Array.isArray(value);

export function createRoleId() {
  const cryptoApi = globalThis.crypto;
  if (typeof cryptoApi?.randomUUID === "function") return cryptoApi.randomUUID();
  if (typeof cryptoApi?.getRandomValues === "function") {
    const bytes = cryptoApi.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = [...bytes].map((value) => value.toString(16).padStart(2, "0")).join("");
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (slot) => {
    const value = Math.floor(Math.random() * 16);
    return (slot === "x" ? value : (value & 0x3) | 0x8).toString(16);
  });
}

export function parsePresets(value) {
  const data = typeof value === "string" ? JSON.parse(value) : clone(value);
  if (!object(data) || data.schema_version !== 2 || !object(data.presets)) throw new Error("Expected {schema_version:2,presets:{}}.");
  for (const preset of Object.values(data.presets)) {
    if (!object(preset) || !Array.isArray(preset.nodes)) throw new Error("Preset must contain nodes.");
    const roles = new Set();
    for (const node of preset.nodes) {
      if (!object(node) || typeof node.role !== "string" || !node.role || roles.has(node.role) || typeof node.type !== "string" || !node.type.startsWith(PREFIX) || !object(node.widgets)) throw new Error("Invalid or duplicate preset role/type/widgets.");
      roles.add(node.role);
      for (const [name, value] of Object.entries(node.widgets)) {
        if (name === "loras") {
          if (!Array.isArray(value)) throw new Error("Preset loras must be an array.");
          parseLoras(value);
        }
        else if (!["string", "number", "boolean"].includes(typeof value)) throw new Error("Preset widget values must be strings, numbers or booleans.");
      }
    }
  }
  return data;
}

const OWNED_SOCKETS = new Set([PIPE, "PROMPTFORGE_V2_WRITER_MODEL", "PROMPTFORGE_V2_H3_REFERENCE_BANK"]);
const MEDIA_SOURCES = new Set(["PROMPTFORGE_V2_LOAD_IMAGE", "PROMPTFORGE_V2_MASK_PREPARE", "PROMPTFORGE_V2_IMAGE_SCALE", "PROMPTFORGE_V2_IMAGE_COMPOSITE"]);

export function upstreamNodes(start) {
  const graph = start.graph;
  if (!graph) return [];
  const seen = new Set([start.id]);
  const result = [];
  function visit(node, throughType) {
    for (const input of node.inputs || []) {
      const socket = throughType || input.type;
      if (!OWNED_SOCKETS.has(socket) && socket !== "IMAGE" && socket !== "MASK") continue;
      const link = graph.links instanceof Map ? graph.links.get(input.link) : graph.links?.[input.link];
      if (!link || seen.has(link.origin_id)) continue;
      const parent = graph.getNodeById(link.origin_id);
      if (!parent) continue;
      const type = typeOf(parent);
      if (type === "Reroute" || type === "Reroute (rgthree)") {
        seen.add(parent.id);
        visit(parent, socket);
      } else if (type?.startsWith(PREFIX) && parent.outputs?.[link.origin_slot]?.type === socket
        && (OWNED_SOCKETS.has(socket) || MEDIA_SOURCES.has(type))) {
        seen.add(parent.id);
        result.push(parent);
        visit(parent);
      }
    }
  }
  visit(start);
  return result;
}

export function ensureRole(node, uuid = createRoleId) {
  node.properties ||= {};
  return node.properties.promptforge_role ||= uuid();
}

function excluded(name, includeText, widget, value) {
  if (["presets", "preset_name", "include_text"].includes(name)) return true;
  if (typeof value === "string" && /^(data:|blob:)/i.test(value)) return true;
  const prose = typeof value === "string" && widget?.type !== "combo" && !Array.isArray(widget?.options?.values)
    && /(^|_)(prompt|prompts|positive|negative|text|description|descriptions|caption|captions|notes|instructions|entries|subject_[123])$/.test(name);
  return !includeText && prose;
}
const connected = (node, name) => node.inputs?.some((input) => (input.name === name || input.widget?.name === name) && input.link != null);

export function capturePreset(nodes, includeText = false, uuid) {
  const roles = new Set();
  return { nodes: nodes.map((node) => {
    let role = ensureRole(node, uuid);
    if (roles.has(role)) node.properties.promptforge_role = role = (uuid || createRoleId)();
    roles.add(role);
    const widgets = {};
    for (const widget of node.widgets || []) {
      if (!widget.name || widget.options?.serialize === false || excluded(widget.name, includeText, widget, widget.value) || connected(node, widget.name)) continue;
      if (widget.name === "loras") {
        const loras = typeof widget.value === "string" ? JSON.parse(widget.value) : widget.value;
        widgets.loras = parseLoras(loras);
      } else if (["string", "number", "boolean"].includes(typeof widget.value)) widgets[widget.name] = widget.value;
    }
    return { role, type: typeOf(node), widgets };
  }) };
}

export function applyPreset(graph, nodes, preset, includeText = false) {
  parsePresets({ schema_version: 2, presets: { apply: preset } });
  const report = [];
  graph.beforeChange();
  try {
    for (const saved of preset.nodes) {
      const matches = nodes.filter((node) => node.properties?.promptforge_role === saved.role && typeOf(node) === saved.type);
      if (matches.length !== 1) { report.push(`Skipped ${saved.role}: ${matches.length ? "ambiguous role" : "role/type not found"}.`); continue; }
      const node = matches[0];
      for (const [name, value] of Object.entries(saved.widgets)) {
        if (connected(node, name)) { report.push(`Skipped ${name}: connected input.`); continue; }
        const widget = node.widgets?.find((entry) => entry.name === name && entry.options?.serialize !== false);
        if (!widget) { report.push(`Skipped ${name}: widget missing.`); continue; }
        if (excluded(name, includeText, widget, value)) { report.push(`Skipped ${name}: prose excluded or media payload.`); continue; }
        const next = name === "loras" ? serializeLoras(widget.value, value) : clone(value);
        if (widget.type === "combo" && Array.isArray(widget.options?.values) && !widget.options.values.includes(next)) { report.push(`Skipped ${name}: unavailable option.`); continue; }
        widget.value = next;
        widget.callback?.(next);
        report.push(`Applied ${saved.type}.${name}.`);
      }
      node.promptforgeV2Refresh?.();
    }
  } finally { graph.afterChange(); }
  graph.setDirtyCanvas?.(true, true);
  return report;
}

export function parseLoras(value) {
  const data = typeof value === "string" ? JSON.parse(value) : clone(value);
  const rows = Array.isArray(data) ? data : data?.items;
  if (!Array.isArray(rows) || rows.some((row) => !object(row) || typeof row.name !== "string")) throw new Error("loras must be an array of entries with a name.");
  return rows;
}

export function serializeLoras(current, rows) {
  const data = typeof current === "string" ? JSON.parse(current) : current;
  return JSON.stringify(Array.isArray(data) ? rows : { ...data, items: rows });
}
