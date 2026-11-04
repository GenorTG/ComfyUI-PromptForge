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
      let presets = {};
      let origin = null;

      const apply = (settings) => {
        for (const [name, value] of Object.entries(settings)) {
          const target = widget(name);
          if (!target || value === undefined) continue;
          if (target.options?.values && !target.options.values.includes(value)) continue;
          target.value = value;
        }
        node.setDirtyCanvas(true, true);
      };

      const presetWidget = node.addWidget("combo", "preset", NO_PRESET, (name) => name !== NO_PRESET && apply(presets[name] ?? {}), { values: [NO_PRESET] });
      presetWidget.serialize = false;
      const reload = (data) => {
        presets = data;
        presetWidget.options.values = [NO_PRESET, ...Object.keys(presets)];
      };
      api.fetchApi("/promptforge/sampler_presets").then((response) => response.json()).then(reload);

      const send = async (body) => reload(await (await api.fetchApi("/promptforge/sampler_presets", { method: "POST", body: JSON.stringify(body) })).json());
      const save = node.addWidget("button", "Save as preset", null, async () => {
        const name = window.prompt("Preset name", presetWidget.value !== NO_PRESET ? presetWidget.value : "");
        if (!name) return;
        await send({ name, settings: Object.fromEntries(PRESET_FIELDS.map((field) => [field, widget(field).value])) });
        presetWidget.value = name;
      });
      const remove = node.addWidget("button", "Delete preset", null, async () => {
        if (presetWidget.value === NO_PRESET) return;
        await send({ name: presetWidget.value, delete: true });
        presetWidget.value = NO_PRESET;
      });
      const recommend = async () => {
        const info = origin?.promptforgeModelInfo?.();
        if (!info) return;
        const query = new URLSearchParams(info).toString();
        const { settings } = await (await api.fetchApi("/promptforge/recommend?" + query)).json();
        apply(settings);
      };
      const recommendButton = node.addWidget("button", "Use recommended settings", null, recommend);
      [save, remove, recommendButton].forEach((button) => { button.serialize = false; });

      // wiring a PromptForge Loader fills in the settings recommended for its model
      node.promptforgeOnPipe = (loader) => {
        origin = loader;
        recommend();
      };
    };
  },
});
