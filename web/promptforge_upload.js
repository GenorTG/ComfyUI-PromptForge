import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// an Upload button for the Load Video / Load Audio nodes: the file goes to the input folder and is chosen in the list
const FILES = { PROMPTFORGE_LoadVideo: ["video", "video/*"], PROMPTFORGE_LoadAudio: ["audio", "audio/*,video/*"] };

app.registerExtension({
  name: "PromptForge.Upload",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!FILES[nodeData.name]) return;
    const [field, accept] = FILES[nodeData.name];
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const node = this;
      const picker = Object.assign(document.createElement("input"), { type: "file", accept });
      picker.addEventListener("change", async () => {
        const file = picker.files[0];
        picker.value = "";
        if (!file) return;
        const body = new FormData();
        body.append("image", file);
        body.append("type", "input");
        const response = await api.fetchApi("/upload/image", { method: "POST", body });
        if (response.status !== 200) return window.alert(`Upload failed: ${response.status} ${response.statusText}`);
        const data = await response.json();
        const name = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
        const combo = node.widgets.find((widget) => widget.name === field);
        if (!combo.options.values.includes(name)) combo.options.values.push(name);
        combo.value = name;
        combo.callback?.(name);
        node.setDirtyCanvas(true, true);
      });
      const button = node.addWidget("button", `upload ${field}`, null, () => picker.click());
      button.serialize = false;
    };
  },
});
