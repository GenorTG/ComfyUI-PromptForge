"""Action preset editor: writes, duplicates and deletes presets of the user's extra file.

The node itself only chains the pipe; the widgets carry the form the frontend shows. The
save and delete ticks let a plain queue run write without the buttons.
"""
from comfy_api.latest import io

from . import action_store
from .contracts import schema as nodes_schema, PIPE, ensure_pipe

Pipe = io.Custom(PIPE)
PIPE_FAMILY = {"image": "image", "h3": "minimax_h3"}


class ActionPreset(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return nodes_schema("", "ACTION_PRESET", "Action Preset", [
            io.Combo.Input("family", options=list(action_store.FAMILIES)),
            io.String.Input("label", default=""),
            io.Combo.Input("category", options=[*action_store.CATEGORIES["image"], *action_store.CATEGORIES["h3"]]),
            io.Int.Input("people", default=1, min=1, max=3),
            io.String.Input("genders", default="female", tooltip="Comma list, one per person: female,male."),
            io.String.Input("aspect", default="16:9", tooltip="H3 only: recommended frame shape."),
            io.String.Input("brief", default="", multiline=True, tooltip="{A} and {B} are the people, in order."),
            io.Combo.Input("penetration", options=list(action_store.PENETRATIONS), default="auto",
                           tooltip="H3 only: the opening the act uses, auto reads it from the label."),
            io.Boolean.Input("save", default=False),
            io.Boolean.Input("delete", default=False),
            Pipe.Input("pipe", optional=True)],
            [Pipe.Output("pipe"), io.String.Output("label")])

    @classmethod
    def execute(cls, family="image", label="", category="", people=1, genders="female", aspect="16:9",
                brief="", penetration="auto", save=False, delete=False, pipe=None):
        pipe = ensure_pipe(pipe, family=PIPE_FAMILY[family])
        label = label.strip()
        if delete:
            if not label:
                raise ValueError("An action preset needs a label to delete.")
            action_store.delete(family, label)
        if save:
            action_store.save(family, label, {"category": category, "people": people, "genders": genders,
                                              "aspect": aspect, "brief": brief, "penetration": penetration})
        return io.NodeOutput(pipe, label, ui={"label": [label]})


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_V2_ACTION_PRESET": ActionPreset}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_V2_ACTION_PRESET": "PromptForge V2 · Action Preset"}
