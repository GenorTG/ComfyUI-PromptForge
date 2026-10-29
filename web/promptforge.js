import { app } from "../../scripts/app.js";

const NODE_NAME = "PROMPTFORGE_PromptEnhancer";

// Which model widgets are relevant for each source.
const VISIBILITY = {
    "safetensors": {
        show: ["safetensors_model", "quantization", "custom_repo_id", "local_model_path"],
        hide: ["gguf_model", "mmproj_path"],
    },
    "GGUF": {
        show: ["gguf_model", "local_model_path", "mmproj_path"],
        hide: ["safetensors_model", "quantization", "custom_repo_id"],
    },
    "connected clip": {
        show: [],
        hide: ["safetensors_model", "gguf_model", "quantization", "custom_repo_id", "local_model_path", "mmproj_path"],
    },
};

function setHidden(widget, hidden) {
    if (!widget) return;
    if (hidden) {
        if (!widget.__pfOriginal) {
            widget.__pfOriginal = { type: widget.type, computeSize: widget.computeSize };
        }
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
    } else if (widget.__pfOriginal) {
        widget.type = widget.__pfOriginal.type;
        widget.computeSize = widget.__pfOriginal.computeSize;
        delete widget.__pfOriginal;
    }
}

function applyVisibility(node) {
    const source = node.widgets?.find((w) => w.name === "source");
    if (!source) return;

    const rule = VISIBILITY[source.value] ?? VISIBILITY["safetensors"];
    for (const widget of node.widgets ?? []) {
        if (rule.hide.includes(widget.name)) setHidden(widget, true);
    }
    for (const widget of node.widgets ?? []) {
        if (rule.show.includes(widget.name)) setHidden(widget, false);
    }

    node.setSize([node.size[0], node.computeSize()[1]]);
    app.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "GenorTG.PromptForge.Visibility",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const source = this.widgets?.find((w) => w.name === "source");
            if (source) {
                const callback = source.callback;
                source.callback = (...args) => {
                    const callbackResult = callback?.apply(source, args);
                    applyVisibility(this);
                    return callbackResult;
                };
            }
            requestAnimationFrame(() => applyVisibility(this));
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            requestAnimationFrame(() => applyVisibility(this));
            return result;
        };
    },
});
