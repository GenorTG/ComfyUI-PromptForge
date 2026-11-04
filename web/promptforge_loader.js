import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "PROMPTFORGE_Loader";
const CLIP_SLOTS = ["clip_1", "clip_2", "clip_3", "clip_4"];

function setHidden(widget, hidden) {
  if (widget.promptforgeSize === undefined) widget.promptforgeSize = widget.computeSize ?? null;
  widget.hidden = hidden;
  widget.computeSize = hidden ? () => [0, -4] : widget.promptforgeSize ?? undefined;
}

app.registerExtension({
  name: "PromptForge.Loader",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const node = this;
      const widget = (name) => node.widgets.find((item) => item.name === name);
      const modelWidget = widget("model");
      const familyWidget = widget("family");
      const allModels = [...modelWidget.options.values];
      let catalog = null;

      const familyOfModel = (value) => catalog?.models.find((entry) => entry.value === value)?.family || "";
      const familyByLabel = (label) => catalog?.families.find((entry) => entry.label === label);

      node.promptforgeFamily = () => (familyByLabel(familyWidget.value) || catalog?.families.find((entry) => entry.key === familyOfModel(modelWidget.value)))?.key || "";

      node.promptforgeFamilyLabel = () => catalog?.families.find((entry) => entry.key === node.promptforgeFamily())?.label ?? "";
      const FOLDERS = { ckpt: "checkpoints", unet: "diffusion_models", gguf: "unet_gguf" };
      node.promptforgeModelInfo = () => {
        const match = /^\[(\w+)\] (.+)$/.exec(modelWidget.value);
        return { family: node.promptforgeFamily(), model: match?.[2] ?? "", folder: FOLDERS[match?.[1]] ?? "" };
      };

      // tell nodes wired to PIPE what changed, so a LoRA list or sampler can follow the model
      function notifyLinked() {
        const slot = node.outputs?.find((output) => output.name === "PIPE");
        for (const id of slot?.links ?? []) {
          app.graph.getNodeById(app.graph.links[id]?.target_id)?.promptforgeOnPipe?.(node);
        }
      }

      function refresh({ keepSize = false, notify = false } = {}) {
        if (!catalog) return;
        const chosen = familyByLabel(familyWidget.value);
        const family = chosen || catalog.families.find((entry) => entry.key === familyOfModel(modelWidget.value));
        if (chosen) {
          const allowed = catalog.models.filter((entry) => entry.family === chosen.key).map((entry) => entry.value);
          modelWidget.options.values = allowed.length ? allowed : allModels;
          if (allowed.length && !allowed.includes(modelWidget.value)) modelWidget.value = allowed[0];
        } else {
          modelWidget.options.values = allModels;
        }
        const slots = Math.max(1, family?.clips.length ?? 1);
        CLIP_SLOTS.forEach((name, index) => {
          const slot = widget(name);
          setHidden(slot, index >= slots && (slot.value === "(none)" || slot.value === "(baked-in)"));
          slot.label = family?.clips[index] ? `${name} (${family.clips[index]})` : name;
        });
        setHidden(widget("audio_vae"), !(family?.vaes.includes("audio")) && widget("audio_vae").value === "(none)");
        if (notify) notifyLinked();
        if (!keepSize) node.setSize([node.size[0], node.computeSize()[1]]);
        node.setDirtyCanvas(true, true);
      }

      const wrap = (target) => {
        const original = target.callback;
        target.callback = function () {
          const result = original?.apply(this, arguments);
          refresh({ notify: true });
          return result;
        };
      };
      wrap(modelWidget);
      wrap(familyWidget);
      api.fetchApi("/promptforge/catalog").then((response) => response.json()).then((data) => {
        catalog = data;
        refresh();
      });
      node.promptforgeRefresh = refresh;
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      this.promptforgeRefresh?.({ keepSize: true });
      return result;
    };
  },
});
