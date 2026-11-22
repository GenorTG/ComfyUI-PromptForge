import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "PROMPTFORGE_Sampler";
const PRESET_FIELDS = ["steps", "cfg", "sampler_name", "scheduler", "denoise", "runs", "step_cache", "cache_threshold"];
const NO_PRESET = "(no preset)";

app.registerExtension({
  name: "PromptForge.Sampler",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const node = this;
      const widget = (name) => node.widgets.find((item) => item.name === name);

      const apply = (settings) => {
        for (const [name, value] of Object.entries(settings)) {
          const target = widget(name);
          if (!target || value === undefined) continue;
          if (target.options?.values && !target.options.values.includes(value)) continue;
          target.value = value;
        }
        node.setDirtyCanvas(true, true);
      };

      const loaderNode = () => {
        const link = app.graph.links[node.inputs?.find((input) => input.name === "pipe")?.link];
        return app.graph.getNodeById(link?.origin_id);
      };
      const model = () => loaderNode()?.promptforgeModelInfo?.();
      let showAll = false;
      const presetWidget = node.addWidget("combo", "preset", NO_PRESET, (name) => name !== NO_PRESET && apply(presets[name] ?? {}), { values: [NO_PRESET] });
      presetWidget.serialize = false;
      let presets = {};
      const fits = (entry) => {
        const current = model();
        if (!current || !entry.family) return true;
        return entry.family === current.family && (!entry.model || entry.model === current.model);
      };
      const reload = (data) => {
        presets = data;
        const names = Object.keys(presets).filter((name) => showAll || fits(presets[name]));
        presetWidget.options.values = [NO_PRESET, ...names];
        if (presetWidget.value !== NO_PRESET && !names.includes(presetWidget.value)) presetWidget.value = NO_PRESET;
      };
      api.fetchApi("/promptforge/sampler_presets").then((response) => response.json()).then(reload);
      const all = node.addWidget("toggle", "show all presets", false, (value) => { showAll = value; reload(presets); }, {});
      node.promptforgeOnPipe = () => { showAll = false; all.value = false; api.fetchApi("/promptforge/sampler_presets").then((response) => response.json()).then(reload); };
      all.serialize = false;

      const save = node.addWidget("button", "Save as preset", null, async () => {
        const name = window.prompt("Preset name", presetWidget.value !== NO_PRESET ? presetWidget.value : "");
        if (!name) return;
        const current = model() || {};
        await send({ name, family: current.family || "", model: current.model || "",
          settings: Object.fromEntries(PRESET_FIELDS.map((field) => [field, widget(field).value])) });
        presetWidget.value = name;
      });
      const remove = node.addWidget("button", "Delete preset", null, async () => {
        if (presetWidget.value === NO_PRESET) return;
        await send({ name: presetWidget.value, delete: true });
        presetWidget.value = NO_PRESET;
      });
      // only ever runs when the button is pressed; the loader is whatever is wired to the pipe input right now
      const loader = loaderNode;
      const say = (summary, detail) => {
        const toast = app.extensionManager?.toast;
        if (toast) toast.add({ severity: "info", summary, detail, life: 4000 });
        else window.alert(`${summary}\n${detail}`);
      };
      const recommend = async () => {
        const info = loader()?.promptforgeModelInfo?.();
        if (!info) return say("PromptForge Sampler", "Connect a PromptForge Loader to the pipe input first.");
        const { settings, source } = await (await api.fetchApi("/promptforge/recommend?" + new URLSearchParams(info))).json();
        apply(settings);
        say("Recommended settings applied", Object.entries(settings).map(([key, value]) => `${key}: ${value} (${source[key]})`).join("\n"));
      };
      const recommendButton = node.addWidget("button", "Use recommended settings", null, recommend);
      [save, remove, recommendButton].forEach((button) => { button.serialize = false; });
    };
  },
});
