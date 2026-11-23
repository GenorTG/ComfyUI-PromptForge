import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NS = "PromptForge.Resources";
const POLL = 2000;
const METERS = [["cpu", "CPU"], ["cores", "Cores"], ["ram", "RAM"], ["swap", "Swap"],
                ["gpu", "GPU"], ["vram", "VRAM"], ["disk", "Disk"]];
const IDS = METERS.map(([id]) => id);
const LABELS = new Map(METERS);
const DEFAULT_ORDER = IDS.join(",");
const TONE = (value) => value >= 85 ? "#d05555" : value >= 60 ? "#d6a13c" : "#4caf7d";
const GIB = 1024 ** 3;

const CSS = `
#promptforge-resources{display:flex;align-items:center;height:32px;padding:0 14px;font:11px/1 ui-monospace,monospace;
 color:var(--input-text,#ccc);background:var(--comfy-menu-bg,#202020);border-bottom:1px solid var(--border-color,#3a3a3a);
 user-select:none;overflow:hidden;white-space:nowrap}
#promptforge-resources[hidden]{display:none}
#promptforge-resources.pf-fixed{position:fixed;top:0;left:0;right:0;z-index:1003}
.pf-content{display:flex;flex:1 1 auto;align-items:center;justify-content:center;gap:18px;height:100%;min-width:0;overflow:hidden;white-space:nowrap}
.pf-meter{display:flex;flex:0 0 auto;align-items:center;gap:6px;height:100%;padding:0 7px;border:1px solid #ffffff1a;
 border-radius:6px;background:#ffffff0f}
.pf-label{flex:0 0 auto;font-size:9px;opacity:.55;letter-spacing:.05em;text-transform:uppercase}
.pf-value{flex:0 0 auto;min-width:5.4em;font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.pf-track{position:relative;display:block;flex:0 0 32px;width:32px;height:4px;border-radius:2px;background:#0006;overflow:hidden}
.pf-fill{display:block;height:100%;width:0;background:#4caf7d;transition:width .25s linear}
.pf-cores{flex:1 1 130px;min-width:96px}
.pf-cores .pf-value{min-width:auto}
.pf-cores .pf-track{display:flex;flex:1 1 auto;width:auto;height:5px;background:none;overflow:visible;border-radius:0;gap:1px}
.pf-core{position:relative;flex:1 1 4px;min-width:1px;height:5px;border-radius:1px;background:#0006;overflow:hidden}
.pf-core .pf-fill{position:absolute;left:0;top:0;width:0;height:100%;transition:width .25s linear}
`;

const state = { enabled: true, meters: {}, order: [], data: {} };
let timer = 0;
let attempts = 0;

function element(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text) node.textContent = text;
  return node;
}

const percent = (used, total) => total > 0 ? Math.min(100, used * 100 / total) : 0;
const gib = (bytes) => (bytes / GIB).toFixed(1);

const bar = element("div");
bar.id = "promptforge-resources";
bar.hidden = true;
const content = element("div", "pf-content");
bar.append(content);

// widest text each meter can show, in characters: a fixed width keeps the bar from shifting as the numbers change
const WIDTH = { cpu: 4, gpu: 12, ram: 14, swap: 14, vram: 14, disk: 14, cores: 8 };
const views = new Map();

function meter(labelText, width) {
  const wrap = element("span", "pf-meter");
  const value = element("span", "pf-value");
  value.style.cssText = `flex:0 0 ${width}ch;width:${width}ch;min-width:${width}ch`;
  const track = element("span", "pf-track");
  const fill = element("span", "pf-fill");
  wrap.append(element("span", "pf-label", labelText), value, track);
  track.append(fill);
  return { wrap, fill, value, width };
}

function view(key, labelText, id) {
  if (!views.has(key)) views.set(key, meter(labelText, WIDTH[id]));
  return views.get(key);
}

function paint(target, load, text, title) {
  target.fill.style.width = `${load}%`;
  target.fill.style.background = TONE(load);
  target.value.textContent = text.padStart(target.width, "\u00a0");
  if (title) target.wrap.title = title;
}

function drawCores(cores) {
  if (!views.has("cores")) {
    const wrap = element("span", "pf-meter pf-cores");
    const value = element("span", "pf-value");
    value.style.cssText = `flex:0 0 ${WIDTH.cores}ch;width:${WIDTH.cores}ch;min-width:${WIDTH.cores}ch`;
    const track = element("span", "pf-track");
    wrap.append(element("span", "pf-label", "Cores"), value, track);
    views.set("cores", { wrap, value, track, fills: [] });
  }
  const target = views.get("cores");
  if (target.fills.length !== cores.length) {
    target.track.replaceChildren();
    target.fills = cores.map(() => {
      const core = element("span", "pf-core");
      const fill = element("span", "pf-fill");
      core.append(fill);
      target.track.append(core);
      return fill;
    });
  }
  let sum = 0;
  let peak = 0;
  cores.forEach((load, index) => {
    sum += load;
    peak = Math.max(peak, load);
    target.fills[index].style.width = `${load}%`;
    target.fills[index].style.background = TONE(load);
  });
  target.value.textContent = `${Math.round(sum / cores.length)}% avg`.padStart(WIDTH.cores, "\u00a0");
  target.wrap.title = `${cores.length} cores, ${peak.toFixed(0)}% peak`;
  return target.wrap;
}

function meterNodes() {
  const nodes = [];
  const { cpu, ram, swap, gpus = [], disk } = state.data;
  for (const id of state.order) {
    if (!state.meters[id]) continue;
    if (id === "cores" && cpu?.cores?.length) {
      nodes.push(drawCores(cpu.cores));
    } else if (id === "cpu" && cpu) {
      const target = view("cpu", "CPU", "cpu");
      paint(target, cpu.percent, `${cpu.percent.toFixed(0)}%`);
      nodes.push(target.wrap);
    } else if (id === "ram" && ram) {
      const target = view("ram", "RAM", "ram");
      paint(target, percent(ram.used, ram.total), `${gib(ram.used)}/${gib(ram.total)} GB`, `${gib(ram.used)} GB of ${gib(ram.total)} GB in use`);
      nodes.push(target.wrap);
    } else if (id === "swap" && swap) {
      const target = view("swap", "Swap", "swap");
      paint(target, percent(swap.used, swap.total), `${gib(swap.used)}/${gib(swap.total)} GB`);
      nodes.push(target.wrap);
    } else if (id === "gpu" && gpus.length) {
      gpus.forEach((gpu, index) => {
        const target = view(`gpu${index}`, `GPU${gpus.length > 1 ? index : ""}`, "gpu");
        paint(target, gpu.utilisation, `${gpu.utilisation}% · ${gpu.temperature}°C`, gpu.name);
        nodes.push(target.wrap);
      });
    } else if (id === "vram" && gpus.length) {
      gpus.forEach((gpu, index) => {
        const target = view(`vram${index}`, `VRAM${gpus.length > 1 ? index : ""}`, "vram");
        paint(target, percent(gpu.used, gpu.total), `${gib(gpu.used)}/${gib(gpu.total)} GB`, gpu.name);
        nodes.push(target.wrap);
      });
    } else if (id === "disk" && disk) {
      const target = view("disk", "Disk", "disk");
      paint(target, percent(disk.total - disk.free, disk.total), `${gib(disk.total - disk.free)}/${gib(disk.total)} GB`, `${gib(disk.free)} GB free`);
      nodes.push(target.wrap);
    }
  }
  return nodes;
}

function draw() {
  const nodes = state.enabled ? meterNodes() : [];
  const same = nodes.length === content.children.length && nodes.every((node, index) => content.children[index] === node);
  if (!same) content.replaceChildren(...nodes);
}

function stop() {
  if (timer) clearInterval(timer);
  timer = 0;
}

async function tick() {
  if (!bar.isConnected) {
    stop();
    return;
  }
  try {
    const response = await api.fetchApi("/promptforge/resources");
    if (!response.ok) return;
    state.data = await response.json();
    draw();
  } catch {
    // keep the last reading; a later tick retries without logging
  }
}

function start() {
  stop();
  if (!state.enabled || !bar.isConnected) return;
  tick();
  timer = setInterval(tick, POLL);
}

function normaliseOrder(value) {
  const wanted = String(value || "").split(/[,\s]+/).map((id) => id.trim().toLowerCase()).filter((id) => LABELS.has(id));
  return [...new Set(wanted), ...IDS.filter((id) => !wanted.includes(id))];
}

function setting(id) {
  return app.ui.settings.getSettingValue(`${NS}.${id}`);
}

function apply() {
  state.enabled = setting("Enabled") !== false;
  state.meters = Object.fromEntries(IDS.map((id) => [id, setting(`meter.${id}`) ?? id !== "cores"]));
  state.order = normaliseOrder(setting("Order"));
  bar.hidden = !state.enabled;
  draw();
  if (state.enabled && !document.hidden) start();
  else stop();
}

function mount() {
  const target = document.getElementById("comfyui-body-top") || document.querySelector(".comfyui-menu");
  if (target) {
    target.append(bar);
    start();
    return;
  }
  if (attempts++ > 60) {
    bar.classList.add("pf-fixed");
    document.body.append(bar);
    start();
    return;
  }
  requestAnimationFrame(mount);
}

function registerSetting(definition) {
  app.ui.settings.addSetting({ onChange: apply, ...definition });
}

app.registerExtension({
  name: "PromptForge.Resources",
  setup() {
    if (!document.getElementById("promptforge-resources-style")) {
      const style = element("style");
      style.id = "promptforge-resources-style";
      style.textContent = CSS;
      document.head.append(style);
    }
    registerSetting({ id: `${NS}.Enabled`, category: ["PromptForge", "Resource readout"], name: "Show system readout",
                      type: "boolean", defaultValue: true, tooltip: "CPU, memory, GPU and disk meters in the top bar." });
    for (const [id, label] of METERS) {
      registerSetting({ id: `${NS}.meter.${id}`, category: ["PromptForge", "Resource readout"], name: `Show ${label} meter`,
                        type: "boolean", defaultValue: id !== "cores", tooltip: id === "cores" ? "One bar per CPU core; off by default." : undefined });
    }
    registerSetting({ id: `${NS}.Order`, category: ["PromptForge", "Resource readout"], name: "Meter order",
                      type: "text", defaultValue: DEFAULT_ORDER, tooltip: `Comma separated: ${IDS.join(", ")}` });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) stop();
      else if (state.enabled) start();
    });
    apply();
    mount();
  },
});
