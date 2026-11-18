#!/usr/bin/env python3
"""Build blank UI/API pairs from registered ComfyUI object_info, without model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PIPE = "PROMPTFORGE_V2_PIPE"
PREFIX = "PROMPTFORGE_V2_"
GROUPS = ["01 Models & LoRAs", "02 Intent & References", "03 Prompt & Review",
          "04 Format & Sampling", "06 Encode & Sample", "07 Refine & Decode", "08 Optional RIFE", "09 Inspect & Save",
          "05 Performance & Preview"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def inputs(info):
    for section in ("required", "optional"):
        values = info.get("input", {}).get(section, {})
        order = info.get("input_order", {}).get(section, list(values))
        for name in order:
            yield name, values[name], section == "optional"


def is_widget(spec):
    return not spec[1].get("forceInput", False) and (isinstance(spec[0], list) or spec[0] in ("STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"))


def socket_type(spec):
    return "COMBO" if isinstance(spec[0], list) else spec[0]


def default(spec):
    kind, options = spec
    if "default" in options:
        return options["default"]
    if isinstance(kind, list):
        return kind[0] if kind else "<SELECT_ARTIFACT>"
    if kind == "COMBO":
        return options.get("options", ["<SELECT_ARTIFACT>"])[0]
    return {"STRING": "", "INT": 0, "FLOAT": 0.0, "BOOLEAN": False}[kind]


class Graph:
    def __init__(self, catalogue, name, family, mode):
        self.catalogue, self.name, self.family, self.mode = catalogue, name, family, mode
        self.nodes, self.api, self.links, self.placeholders = [], {}, [], []
        self.pipe = None
        self.roles = {}

    def add(self, key, role, group, values=None, connections=None, chain=True):
        key = key if key.startswith(PREFIX) or key == "SaveImage" else PREFIX + key
        info = self.catalogue[key]
        values, connections = dict(values or {}), dict(connections or {})
        declared = {name for name, _, _ in inputs(info)}
        unknown = (values.keys() | connections.keys()) - declared
        if unknown:
            raise ValueError(f"{key}: unknown inputs {unknown}")
        if chain and "pipe" in declared and "pipe" not in connections and self.pipe is not None:
            connections["pipe"] = self.pipe
        node_id = len(self.nodes) + 1
        named, widgets, sockets, api_inputs = {}, [], [], {}
        for name, spec, optional in inputs(info):
            # Older object_info schemas omit the options dict on sockets.
            spec = (spec[0], spec[1] if len(spec) > 1 else {})
            if spec[0] == "COMFY_AUTOGROW_V3" and optional and name not in connections:
                continue
            widget = is_widget(spec)
            if widget:
                value = values.get(name, default(spec))
                named[name] = value
                widgets.append(value)
                if spec[1].get("control_after_generate"):
                    widgets.append("fixed")
                elif name == "seed" and spec[0] == "INT":
                    # Current ComfyUI V3 frontend adds this control widget to
                    # seed inputs even when V3 object_info omits the legacy flag.
                    widgets.append(False)
                api_inputs[name] = value
                if isinstance(value, str) and value.startswith("<"):
                    self.placeholders.append({"node": str(node_id), "role": role, "input": name, "value": value})
            if not widget or name in connections:
                slot = {"name": name, "type": socket_type(spec), "link": None}
                if widget:
                    slot["widget"] = {"name": name}
                if optional:
                    slot["shape"] = 7
                sockets.append(slot)
            if name in connections:
                source, output = connections[name]
                link_id = len(self.links) + 1
                source_node = self.nodes[source - 1]
                source_type = source_node["outputs"][output]["type"]
                target_type = socket_type(spec)
                if source_type != target_type and "*" not in (source_type, target_type):
                    raise ValueError(f"{role}.{name}: {source_type} != {target_type}")
                sockets[-1]["link"] = link_id
                source_node["outputs"][output]["links"].append(link_id)
                self.links.append([link_id, source, output, node_id, len(sockets)-1, source_type])
                api_inputs[name] = [str(source), output]
            elif not widget and not optional:
                raise ValueError(f"{role}.{name}: required socket has no link")
        output_names = info.get("output_name", info.get("output", []))
        outputs = [{"name": output_names[i], "type": kind, "links": [], "slot_index": i}
                   for i, kind in enumerate(info.get("output", []))]
        node = {"id": node_id, "type": key, "pos": [0, 0], "size": [430, 200], "flags": {},
                "order": node_id - 1, "mode": 0, "inputs": sockets, "outputs": outputs,
                "title": role.replace("_", " ").title(), "properties": {
                    "Node name for S&R": key, "promptforge_role": role,
                    "promptforge_v2_widget_values": named}, "widgets_values": widgets}
        self.nodes.append(node)
        self.roles[role] = (node_id, group)
        self.api[str(node_id)] = {"class_type": key, "inputs": api_inputs, "_meta": {"title": node["title"]}}
        if chain and PIPE in info.get("output", []):
            self.pipe = (node_id, info["output"].index(PIPE))
        return node_id

    # canvas sizes mirror web/promptforge_v2.js SIZES so the builder and the frontend agree
    ROLE_SIZE = {"loras": (560, 620), "edit_lora": (400, 260), "prompt": (440, 300), "prompt_writer": (420, 250),
                 "prompt_review": (440, 260), "preview": (420, 320), "first_pass_preview": (420, 300),
                 "save_video": (440, 400), "save_image": (440, 360), "presets": (420, 380),
                 "run_inspect": (440, 280), "reference_1": (440, 460), "reference_2": (440, 460),
                 "reference_3": (440, 460), "reference_4": (440, 460), "reference_scale": (400, 200),
                 "first_frame": (440, 460), "last_frame": (440, 460), "reference_image": (440, 460),
                 "source_video": (420, 300), "bank_image": (440, 460), "bank_video": (420, 300),
                 "bank_audio": (420, 240), "subject_mask": (420, 240), "edit_mask": (380, 220)}
    HIDDEN_INPUTS = {"source", "loras", "layer_controls", "action", "subject_1", "subject_2", "subject_3", "camera", "notes"}

    def node_size(self, node):
        role = node["properties"]["promptforge_role"]
        width, min_height = self.ROLE_SIZE.get(role, (400, 0))
        named = node["properties"]["promptforge_v2_widget_values"]
        height = 96
        for name, spec, _ in inputs(self.catalogue[node["type"]]):
            if name not in named or name in self.HIDDEN_INPUTS:
                continue
            options = spec[1] if len(spec) > 1 else {}
            height += (86 if name in ("text", "negative") else 58) if options.get("multiline") else 25
        return width, max(190, min_height, height)

    def layout(self):
        groups, plans = [], []
        ordered_groups = [GROUPS[i] for i in (0, 1, 2, 3, 8, 4, 5, 6, 7)]
        for index, label in enumerate(ordered_groups):
            members = [self.nodes[n - 1] for n, group in self.roles.values() if group == label]
            if not members:
                continue
            plans.append((index, label, members, [self.node_size(node) for node in members]))

        columns = [0, 0, 0, 0, 0]
        for index, _, _, sizes in plans:
            columns[index % 5] = max(columns[index % 5], max(width for width, _ in sizes))
        offsets, x = {}, 44
        for column, width in enumerate(columns):
            offsets[column] = x
            x += width + 70
        tallest = max((sum(height + 56 for _, height in sizes) for _, _, _, sizes in plans if True), default=1400)
        row_height = tallest + 170

        for index, label, members, sizes in plans:
            column, row = index % 5, index // 5
            x, top = offsets[column], row * row_height
            y = top + 76
            for node, (width, height) in zip(members, sizes):
                node["pos"], node["size"] = [x, y], [width, height]
                y += height + 56
            groups.append({"title": label, "bounding": [x - 16, top, columns[column] + 32, y - top - 24],
                           "color": "#35566b", "font_size": 24, "flags": {}})
        return groups

    def validate(self):
        by_id = {n["id"]: n for n in self.nodes}
        for link, source, output, target, slot, kind in self.links:
            assert by_id[source]["outputs"][output]["type"] == kind
            assert link in by_id[source]["outputs"][output]["links"]
            assert by_id[target]["inputs"][slot]["link"] == link
        preset = next(n for n in self.nodes if n["type"] == PREFIX + "PRESETS")
        reachable, pending = set(), [preset["id"]]
        while pending:
            current = pending.pop()
            if current in reachable:
                continue
            reachable.add(current)
            pending.extend(link[1] for link in self.links if link[3] == current and link[5] == PIPE)
        pipe_nodes = {n["id"] for n in self.nodes if n["id"] <= preset["id"] and any(o["type"] == PIPE for o in n["outputs"])}
        if not pipe_nodes <= reachable:
            raise ValueError(f"{self.name}: presets cannot reach pipe nodes {pipe_nodes - reachable}")
        for n in self.nodes:
            assert n["mode"] == 0
            assert n["type"] in self.catalogue
            assert n["type"].startswith(PREFIX) or n["type"] == "SaveImage"
            api_inputs = self.api[str(n["id"])]["inputs"]
            for name, value in n["properties"]["promptforge_v2_widget_values"].items():
                assert api_inputs[name] == value
            for name, spec, _ in inputs(self.catalogue[n["type"]]):
                if len(spec) < 2:
                    spec = (spec[0], {})
                if not is_widget(spec) or name not in api_inputs:
                    continue
                value = api_inputs[name]
                if isinstance(value, str) and value.startswith("<"):
                    continue
                options = spec[0] if isinstance(spec[0], list) else spec[1].get("options")
                if options is not None:
                    assert value in options, (n["type"], name, value)
                if name in ("text", "negative", "subject_1", "subject_2", "subject_3", "notes", "local_model_path", "mmproj_path"):
                    assert value == "", (n["type"], name, "template must be blank")
        for left in self.nodes:
            x, y = left["pos"]
            w, h = left["size"]
            for right in self.nodes[left["id"]:]:
                rx, ry = right["pos"]
                rw, rh = right["size"]
                assert x + w <= rx or rx + rw <= x or y + h <= ry or ry + rh <= y, "overlapping nodes"
        return {"registration": "passed", "links_and_types": "passed", "preset_upstream_reach": "passed",
                "ui_api_parity": "passed", "blank_content": "passed", "layout_overlap": "passed",
                "execution": "not run: blank inputs and model artifacts required"}

    def save(self, destination, template_menu):
        groups = self.layout()
        checks = self.validate()
        ui = {"last_node_id": len(self.nodes), "last_link_id": len(self.links), "nodes": self.nodes,
              "links": self.links, "groups": groups, "config": {}, "extra": {
                  "ds": {"scale": .32, "offset": [30, 30]}, "promptforge_v2": {
                      "template": self.name, "blank": True, "family": self.family, "mode": self.mode}}, "version": .4}
        write_json(destination / f"{self.name}.json", ui)
        write_json(template_menu / f"V2_{self.name}.json", ui)
        write_json(destination / f"{self.name}.api.json", self.api)
        artifacts = [{"role": item["role"], "input": item["input"], "placeholder": item["value"]}
                     for item in self.placeholders]
        notes = ["Fill blank prompt text before queueing.", "Raw mode skips writer execution; idea/action needs Writer Model."]
        if self.mode in ("edit", "transfer"):
            artifacts.append({"role": "loras", "input": "loras", "requirement": "Add the matching Krea identity/edit LoRA; stack starts empty."})
            notes.append("SAM3 automatic mask inference is untested with weights; a manual/alpha mask can replace Subject Mask.")
        if "multipart" in self.name:
            notes.append("Fill both blank parts_json entries; each part must have explicit text and positive duration.")
        if self.family == "h3":
            notes.append("RIFE is disabled by its enabled widget; enabling it requires a selected interpolation checkpoint.")
        baselines = {"krea2": "Krea 2 turbo baseline: euler/simple, 10 steps, CFG 1, NAG off; verify against the selected checkpoint.",
                     "moss_image": "MOSS baseline from the legacy workflow: er_sde/bong_tangent, 8 steps, CFG 1 for text to image.",
                     "anima": "Anima baseline from the model card: er_sde/simple, 30 steps, CFG 4.5.",
                     "h3": "H3 baseline from the legacy workflow and quality notes: euler/beta57, 8 steps, CFG 1, shifts 0/0; second pass er_sde, 3 steps, start sigma 0.9035; the second pass is the Upscale Settings switch."}
        notes.append(baselines[self.family])
        manifest = {"template": self.name, "family": self.family, "task": self.mode,
                    "ui": f"{self.name}.json", "api": f"{self.name}.api.json", "blank": True,
                    "native_template_menu": {"file": f"V2_{self.name}.json", "intentional_ui_duplicate": True},
                    "custom_packs": ["ComfyUI-PromptForge"], "node_types": sorted({n["type"] for n in self.nodes}),
                    "roles": {role: str(value[0]) for role, value in self.roles.items()},
                    "placeholders": self.placeholders, "validation": checks, "artifacts": artifacts, "notes": notes,
                    "schema_sha256": {key: hashlib.sha256(json.dumps(self.catalogue[key], sort_keys=True).encode()).hexdigest()
                                      for key in sorted({n["type"] for n in self.nodes})}}
        write_json(destination / f"{self.name}.manifest.json", manifest)
        return manifest



def add_writer(graph, h3=False):
    writer = graph.add("WRITER_MODEL", "writer_model", GROUPS[0], chain=False)
    prompt = graph.add("H3_PROMPT" if h3 else "IMAGE_PROMPT", "prompt", GROUPS[1], {"source": "raw", "text": "", "negative": ""})
    writer_node = graph.add("H3_PROMPT_WRITER" if h3 else "IMAGE_PROMPT_WRITER", "prompt_writer", GROUPS[2],
                            connections={"writer_model": (writer, 0), "text": (prompt, 1)})
    graph.add("PROMPT_INSPECT", "prompt_review", GROUPS[2], {}, {"text": (writer_node, 1)})
    return writer, prompt


def add_end(graph, video=False, images=None, audio=None):
    prefix = f"video/PromptForge/{graph.name}" if video else f"PromptForge/{graph.name}"
    if video:
        wires = {"images": images} if images else {}
        if images and audio:
            wires["audio"] = audio
        graph.add("H3_PREVIEW_SAVE", "save_video", GROUPS[7],
                  {"save": True, "filename_prefix": prefix}, wires)
    else:
        graph.add("IMAGE_SAVE", "save_image", GROUPS[7], {"filename_prefix": prefix},
                  {"images": images} if images else None)
    graph.add("PRESETS", "presets", GROUPS[7], {
        "preset_name": "", "presets": '{"schema_version":2,"presets":{}}', "include_text": False})
    graph.add("RUN_INSPECT", "run_inspect", GROUPS[7])
    return graph


def image_graph(catalogue, name, family, task="t2i"):
    graph = Graph(catalogue, name, family, task)
    model = graph.add("IMAGE_MODEL", "model", GROUPS[0], {"family": family,
        "model": "<SELECT_DIFFUSION_MODEL>", "clip": "<SELECT_TEXT_ENCODER>", "vae": "<SELECT_IMAGE_VAE>"})
    loras = graph.add("LORA_STACK", "loras", GROUPS[0],
                      connections={"model": (model, 1), "clip": (model, 2)})
    model_out, clip_out = (loras, 1), (loras, 2)
    if task in ("edit", "transfer", "multi"):
        single = graph.add("SINGLE_LORA", "edit_lora", GROUPS[0],
                           {"lora_name": "Krea 2/tool/krea2_identity_edit_v2.safetensors"},
                           connections={"model": (loras, 1), "clip": (loras, 2)})
        model_out, clip_out = (single, 1), (single, 2)
    # Format precedes Edit Prepare; every stage remains on one upstream pipe.
    format_node = graph.add("IMAGE_FORMAT", "format", GROUPS[3],
                            {"aspect": "4:3", "orientation": "portrait", "megapixels": 1.5},
                            {"model": model_out})
    baseline = {"krea2": {"steps": 10, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"},
                "anima": {"steps": 30, "cfg": 4.5, "sampler_name": "er_sde", "scheduler": "simple"}}.get(family, {})
    if family == "moss_image":
        baseline = {"steps": 20, "cfg": 2.0, "sampler_name": "euler", "scheduler": "normal"} if task == "edit" else \
                   {"steps": 8, "cfg": 1.0, "sampler_name": "er_sde", "scheduler": "bong_tangent"}
    graph.add("IMAGE_SAMPLING_SETTINGS", "sampling_first", GROUPS[3], baseline)
    loaded = []
    for index in range(2 if task in ("transfer", "multi") else 1 if task == "edit" else 0):
        loaded.append(graph.add("LOAD_IMAGE", f"reference_{index+1}", GROUPS[1],
            {"filename": f"<INPUT_REFERENCE_{index+1}.png>", "index": index+1}))
    if loaded:
        if task == "edit":
            # prepare scales the reference to the pipe format itself
            wires = {"image": (loaded[0], 1)}
            mask_image = (loaded[0], 1)
        elif task == "multi":
            wires = {"image": (loaded[0], 1), "image_b": (loaded[1], 1)}
            mask_image = (loaded[0], 1)
        else:
            scale = graph.add("IMAGE_SCALE", "reference_scale", GROUPS[1], {"use_format": True, "crop": "center"},
                              {"image": (loaded[0], 1)})
            wires = {"image": (scale, 0), "image_b": (scale, 0)}
            mask_image = (scale, 0)
        subject = graph.add("SUBJECT_MASK", "subject_mask", GROUPS[1],
                            {"checkpoint": "<SELECT_SAM3_CHECKPOINT>", "subject": "person"},
                            {"image": mask_image})
        mask = graph.add("MASK_PREPARE", "edit_mask", GROUPS[1],
                         connections={"mask": (subject, 1)}, chain=False)
        if task == "transfer":
            wires["mask"] = (mask, 0)
        graph.add("KREA_EDIT_PREPARE", "edit_prepare", GROUPS[4], {"fit_mode": "fit"}, connections=wires)
    _, prompt = add_writer(graph)
    encode = graph.add("IMAGE_ENCODE", "encode", GROUPS[4], {"task": "edit" if loaded else "t2i"},
                       connections={"clip": clip_out, "vae": (model, 3), "model": model_out, "text": (prompt, 1),
                                  "latent": (format_node, 1)})
    sample = graph.add("IMAGE_SAMPLE", "sample_first", GROUPS[4],
                       connections={"model": model_out, "positive": (encode, 1), "negative": (encode, 2), "latent": (encode, 3)})
    decoded = graph.add("IMAGE_DECODE", "decode", GROUPS[5],
                        connections={"vae": (model, 3), "latent": (sample, 1)})
    if task == "transfer":
        composite = graph.add("IMAGE_COMPOSITE", "transfer_composite", GROUPS[5], connections={
            "destination": (loaded[1], 1), "source": (decoded, 1), "mask": (mask, 0)}, chain=False)
        composited = graph.add("IMAGE_RESULT", "composite_result", GROUPS[5], connections={"images": (composite, 0)})
        return add_end(graph, images=(composited, 1))
    add_end(graph, images=(decoded, 1))
    return graph


def h3_graph(catalogue, name, mode, multipart=False):
    graph = Graph(catalogue, name, "h3", mode)
    model = graph.add("H3_MODEL", "model", GROUPS[0], {
        "model_name": "<SELECT_H3_DIFFUSION_MODEL>", "clip_name": "<SELECT_H3_TEXT_ENCODER>",
        "vae_name": "<SELECT_H3_VIDEO_VAE>", "audio_vae_name": "<SELECT_H3_AUDIO_VAE>"})
    loras = graph.add("LORA_STACK", "loras", GROUPS[0], {"layer_controls": "auto"},
                      {"model": (model, 1), "clip": (model, 2)})
    model_out, clip_out = (loras, 1), (loras, 2)
    graph.add("PERFORMANCE", "performance", GROUPS[8])
    graph.add("PREVIEW_SETTINGS", "preview", GROUPS[8], {"enabled": True})
    graph.add("H3_FORMAT", "format", GROUPS[3], {"width": 848, "height": 480, "seconds": 5.0})
    if mode in ("i2v", "fl2v"):
        graph.add("LOAD_IMAGE", "first_frame", GROUPS[1], {"filename": "<INPUT_FIRST_FRAME.png>", "role": "first_frame"})
    if mode in ("l2v", "fl2v"):
        graph.add("LOAD_IMAGE", "last_frame", GROUPS[1], {"filename": "<INPUT_LAST_FRAME.png>", "role": "last_frame"})
    reference_image = None
    if mode == "ref2v" and name != "h3_ref2v_reference_bank":
        reference_image = graph.add("LOAD_IMAGE", "reference_image", GROUPS[1],
                                    {"filename": "<INPUT_REFERENCE.png>", "role": "reference"})
    if mode == "ref2v" and name == "h3_ref2v_reference_bank":
        image = graph.add("LOAD_IMAGE", "bank_image", GROUPS[1], {"filename": "<OPTIONAL_BANK_IMAGE.png>", "role": "reference", "index": 2})
        video = graph.add("LOAD_VIDEO", "bank_video", GROUPS[1], {"filename": "<OPTIONAL_BANK_VIDEO.mp4>", "role": "reference"})
        audio = graph.add("LOAD_AUDIO", "bank_audio", GROUPS[1], {"filename": "<OPTIONAL_BANK_AUDIO.wav>"})
        bank_image = graph.add("H3_REFERENCE_BANK", "bank_image_entry", GROUPS[1],
                               {"name": "subject", "kind": "image"}, {"images": (image, 1)}, chain=False)
        bank_video = graph.add("H3_REFERENCE_BANK", "bank_video_entry", GROUPS[1],
                               {"name": "motion", "kind": "video"},
                               {"bank": (bank_image, 0), "images": (video, 1), "audio": (video, 2)}, chain=False)
        bank_audio = graph.add("H3_REFERENCE_BANK", "bank_audio_entry", GROUPS[1],
                               {"name": "sound", "kind": "audio"},
                               {"bank": (bank_video, 0), "audio": (audio, 1)}, chain=False)
        graph.add("H3_REFERENCE_SELECT", "reference_selection", GROUPS[1],
                  {"names": "subject,motion,sound"}, {"bank": (bank_audio, 0)})
    if mode == "continue":
        source = graph.add("LOAD_VIDEO", "source_video", GROUPS[1], {"filename": "<INPUT_SOURCE_VIDEO.mp4>", "role": "continue"})
        source_clip = graph.add("H3_SOURCE_CLIP", "source_clip", GROUPS[5], connections={"pipe": (source, 0)}, chain=False)
    graph.add("H3_REFERENCES", "references", GROUPS[1], {"mode": mode},
              connections={"reference_images": (reference_image, 1)} if reference_image else None)
    writer, prompt = add_writer(graph, h3=True)
    graph.add("H3_SAMPLING_SETTINGS", "sampling_first_and_refine", GROUPS[3])
    upscale_model = "<SELECT_H3_FP32_LATENT_UPSCALER>"
    graph.add("H3_UPSCALE_SETTINGS", "upscale_settings", GROUPS[3],
              {"upscale_model": upscale_model, "enabled": False, "mode": "megapixels", "percent": 200.0,
               "megapixels": 1.0})
    result = None
    if multipart:
        graph.add("H3_PARTS", "parts", GROUPS[3], {"parts_json": '[{"text":"","seconds":5},{"text":"","seconds":5}]'})
        graph.add("H3_CONTINUITY", "continuity", GROUPS[3], {"mode": "independent" if mode == "ref2v" else "continue"})
        graph.add("H3_SEQUENCE", "sequence", GROUPS[4], connections={"writer_model": (writer, 0)})
    else:
        if mode == "continue":
            graph.add("H3_CONTINUITY", "continuity", GROUPS[3])
        encode = graph.add("H3_ENCODE", "encode", GROUPS[4],
                           connections={"clip": clip_out, "vae": (model, 3), "model": model_out, "text": (prompt, 1)})
        sample_node = graph.add("H3_SAMPLE", "sample_first", GROUPS[4],
                                connections={"model": model_out, "positive": (encode, 1),
                                             "negative": (encode, 2), "latent": (encode, 3)})
        graph.add("H3_FIRST_PASS_PREVIEW", "first_pass_preview", GROUPS[5], {"enabled": True},
                  connections={"pipe": (sample_node, 0)})
        upscaled = graph.add("H3_LATENT_UPSCALE", "latent_upscale", GROUPS[5],
                             connections={"latent": (sample_node, 1)})
        refined = graph.add("H3_REFINE", "sample_refine", GROUPS[5],
                            connections={"model": model_out, "latent": (upscaled, 1)})
        decoded = graph.add("H3_DECODE", "decode", GROUPS[5],
                            connections={"vae": (model, 3), "latent": (refined, 1)})
        result = decoded
        if mode == "continue":
            result = graph.add("H3_JOIN", "join_source", GROUPS[5], {"overlap_frames": 22},
                               {"previous": (source_clip, 0), "images": (decoded, 1), "audio": (decoded, 2)})
    interpolated = graph.add("FRAME_INTERPOLATE", "optional_rife", GROUPS[6], {"enabled": False})
    cleaned = graph.add("VIDEO_CLEANUP", "optional_cleanup", GROUPS[6],
                        {"enabled": False, "denoise_quality": "Ultra", "deblur_quality": "Low"},
                        connections={"images": (interpolated, 1)})
    add_end(graph, video=True, images=(cleaned, 1), audio=(result, 2) if result else None)
    return graph


def build_catalogue(catalogue):
    graphs = [image_graph(catalogue, "krea_t2i", "krea2"),
              image_graph(catalogue, "krea_edit", "krea2", "edit"),
              image_graph(catalogue, "krea_ref2i_multi", "krea2", "multi"),
              image_graph(catalogue, "krea_transfer", "krea2", "transfer"),
              image_graph(catalogue, "moss_t2i", "moss_image"),
              image_graph(catalogue, "anima_t2i", "anima")]
    for mode in ("t2v", "i2v", "l2v", "fl2v", "ref2v"):
        graphs.append(h3_graph(catalogue, f"h3_{mode}", mode))
    graphs.extend([h3_graph(catalogue, "h3_ref2v_reference_bank", "ref2v"),
                   h3_graph(catalogue, "h3_multipart_t2v", "t2v", multipart=True),
                   h3_graph(catalogue, "h3_multipart_references", "ref2v", multipart=True),
                   h3_graph(catalogue, "h3_continue_existing", "continue")])
    return graphs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-info", type=Path, help="Registered ComfyUI /object_info JSON snapshot")
    parser.add_argument("--url", default="http://127.0.0.1:8199")
    parser.add_argument("--output", type=Path, default=ROOT / "example_workflows/v2")
    parser.add_argument("--template-menu-dir", type=Path, default=ROOT / "example_workflows",
                        help="UI-only copies for ComfyUI's nonrecursive native template discovery")
    args = parser.parse_args()
    if args.object_info:
        catalogue = json.loads(args.object_info.read_text())
    else:
        with urllib.request.urlopen(args.url.rstrip("/") + "/object_info", timeout=60) as response:
            catalogue = json.load(response)
    # Keep generated workflows aligned with the current source schemas even if
    # object_info came from the running service before it was restarted.
    stack = catalogue.get(PREFIX + "LORA_STACK", {})
    if "layer_controls" in stack.get("input", {}).get("required", {}):
        stack["input"]["required"]["layer_controls"] = [["auto", "standard", "minimax audiovisual"], {"default": "auto"}]
    save = catalogue.get(PREFIX + "H3_PREVIEW_SAVE", {})
    required = save.get("input", {}).get("required", {})
    order = save.setdefault("input_order", {}).setdefault("required", list(required))
    additions = {
        "pingpong": ["BOOLEAN", {"default": False}], "save_metadata": ["BOOLEAN", {"default": True}],
        "show_preview": ["BOOLEAN", {"default": True}], "loop_preview": ["BOOLEAN", {"default": True}],
        "autoplay_preview": ["BOOLEAN", {"default": True}],
    }
    for name, spec in additions.items():
        required.setdefault(name, spec)
        if name not in order:
            order.append(name)
    if PREFIX + "H3_FIRST_PASS_PREVIEW" in catalogue:
        catalogue[PREFIX + "H3_FIRST_PASS_PREVIEW"]["input"]["required"]["enabled"][1]["default"] = True
    if PREFIX + "PREVIEW_SETTINGS" in catalogue:
        catalogue[PREFIX + "PREVIEW_SETTINGS"]["input"]["required"]["enabled"][1]["default"] = True
    # This optional V2 node may be absent from a snapshot made before its latest
    # registration. Keep the builder usable with a captured object_info file.
    catalogue.setdefault(PREFIX + "H3_FIRST_PASS_PREVIEW", {
        "input": {"required": {"pipe": [PIPE, {}], "enabled": ["BOOLEAN", {"default": True}],
                                "filename_prefix": ["STRING", {"default": "video/PromptForge_H3_first_pass", "multiline": False}]}},
        "input_order": {"required": ["pipe", "enabled", "filename_prefix"]},
        "output": [PIPE], "output_name": ["pipe"]})
    args.output.mkdir(parents=True, exist_ok=True)
    args.template_menu_dir.mkdir(parents=True, exist_ok=True)
    manifests = [graph.save(args.output, args.template_menu_dir) for graph in build_catalogue(catalogue)]
    write_json(args.output / "manifest.json", {"schema_version": 2, "count": len(manifests), "templates": manifests})
    print(f"Validated and generated {len(manifests)} blank UI/API workflow pairs in {args.output}")


if __name__ == "__main__":
    main()
