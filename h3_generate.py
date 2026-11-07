"""PromptForge H3 Endless Video: the generator. It makes a long MiniMax H3 video part by part and returns it; saving is up to you.

It does only the generating. What to make and how comes from small settings nodes (h3_settings.py): the Story (a prompt
per part), References (characters), Video Format, Continuity, Sampling, and the optional Second Stage, Live Preview and
Speed and Memory. Outputs are the frames, the sound, the prompts that were written and a short report, so they go into
any save or post-processing nodes.

For every part:
  1. the prompt writer writes the part's prompt from the story, the references and the end of the previous part,
  2. the prompt is encoded (with the references and the carried-forward frames and sound),
  3. the first stage is sampled; with a Second Stage connected the latent is upscaled and sampled again,
  4. the latents are decoded with the tiled H3 decoder, the join is smoothed and the part is previewed in the node.
The models are released between these steps, so the language model, the diffusion model and the decoders never sit in
VRAM together.
"""
import base64
import gc
import importlib
import json
import os
import re
import threading
import time

import psutil
import torch
import torchaudio
from aiohttp import web
from server import PromptServer

import comfy.model_management as model_management
import comfy.model_prefetch
import comfy.nested_tensor
import comfy.ops
import comfy.utils
import node_helpers
from comfy_extras import nodes_audio, nodes_minimax_h3
from comfy_extras.nodes_sparse_attention import BlockSparseAttention
from comfy_extras.nodes_model_advanced import ModelAttentionBackend

from . import empty_latent, h3_settings, h3_upscaler, minimax_decode, preview_override, sampler

SPARSE_DENSE_BLOCKS = "0, 1"
FPS = 24
SILENCE_RATE = 32000
GRID = 17
PRESETS = {"T2V": "MiniMax H3 T2VA", "I2V": "MiniMax H3 I2VA", "L2V": "MiniMax H3 L2VA", "FL2V": "MiniMax H3 FL2VA", "REF2V": "MiniMax H3 Ref2VA"}
SEAM_FLOOR = 25.0
SEAM_RATIO = 6.0
TRANSITION_TEXT = {
    h3_settings.STORY_STYLES[0]: "Stay in the same continuous shot: same place, subjects, outfits, lighting and camera, with the action moving on. No cut in the first shot of this part.",
    h3_settings.STORY_STYLES[1]: "Continue naturally. If the story calls for it you may cut to a new angle, shot or place, but keep the same subjects, outfits and look.",
    h3_settings.STORY_STYLES[2]: "Open this part with a cut to a new shot (a new angle, framing or place that fits the story). Keep the same subjects, outfits and look.",
}


def _package():
    return importlib.import_module(__package__)


_pending = {}


@PromptServer.instance.routes.post("/promptforge/review")
async def _review_route(request):
    body = await request.json()
    slot = _pending.get(str(body.get("node")))
    if slot is None:
        return web.json_response({"error": "nothing is waiting for a review"}, status=404)
    slot["action"], slot["text"] = str(body.get("action", "approve")), str(body.get("text", ""))
    slot["event"].set()
    return web.json_response({})


def nearest_grid(frames):
    """The frame count on H3's 17k + 5 grid that is closest to `frames`."""
    return GRID * max(0, round((frames - 5) / GRID)) + 5


def chunk_feed_forward(model, chunks, threshold=4096):
    """A copy of the MiniMax H3 model whose feed-forward layers run over the packed tokens in `chunks` pieces (exact: rows are independent)."""
    patched = model.clone()
    blocks = getattr(patched.get_model_object("diffusion_model"), "blocks", None)
    if not blocks or not hasattr(blocks[0], "mlp") or not hasattr(blocks[0].mlp, "fc1"):
        return model

    def forward_of(mlp):
        def forward(x):
            if x.shape[0] <= threshold:
                return comfy.ops.linear_input_act(mlp.fc2, mlp.fc1(x), "swiglu")
            out = torch.empty_like(x)
            offset = 0
            for part in torch.chunk(x, chunks, dim=0):
                out[offset:offset + part.shape[0]] = comfy.ops.linear_input_act(mlp.fc2, mlp.fc1(part), "swiglu")
                offset += part.shape[0]
            return out
        return forward

    for index, block in enumerate(blocks):
        patched.add_object_patch(f"diffusion_model.blocks.{index}.mlp.forward", forward_of(block.mlp))
    return patched


def stage2_sigmas(steps, start):
    """The second-stage schedule: `start`, then a straight line from just above it down to 0. For 3 steps from 0.9035 this is 0.9035, 0.6316, 0.3158, 0 (the schedule used by the usual MiniMax H3 two-pass workflows)."""
    top = start * 1.0486
    return [start] + [top * (steps - index) / steps for index in range(1, steps)] + [0.0]


def part_boxes(text):
    """The parts written in the node's part boxes (a JSON list of strings). A box may start with (10s) to give that part its own length."""
    try:
        values = json.loads(text) if text and text.strip() else []
    except ValueError:
        return []
    plan = []
    for value in values if isinstance(values, list) else []:
        value = str(value).strip()
        match = re.match(r"\(\s*(\d+(?:\.\d+)?)\s*s(?:ec(?:onds?)?)?\s*\)\s*", value, re.I)
        plan.append({"text": value[match.end():].strip() if match else value, "seconds": float(match.group(1)) if match else None})
    return plan


def part_text(index, total, brief, reference_lines, plan, seconds):
    """What the Enhancer is asked for one part: the brief as context, this part's idea as the thing to show, the other parts as things to keep out."""
    lines = ([f"Whole-video context (do not show all of it in this part): {brief}"] if brief else []) + list(reference_lines)
    idea = plan[index]["text"] if index < len(plan) else ""
    if total == 1:
        lines.append(f"The video is {seconds:g} seconds long. " + (f"The idea: {idea}" if idea else "The user gave no instruction: invent something fitting."))
        return "\n".join(lines)
    if idea:
        lines.append(f"PART {index + 1} OF {total} ({seconds:g} seconds). This part shows exactly the following, and nothing that belongs to another part: {idea}")
        lines.append("Show it completely, paced across the whole part. Do not skip ahead to, repeat or summarise the other parts.")
    else:
        lines.append(f"PART {index + 1} OF {total} ({seconds:g} seconds). The user gave no instruction for this part: continue the story from the brief naturally.")
    others = [f"Part {number + 1}: {part['text'][:300]}" for number, part in enumerate(plan) if number != index and part["text"]]
    if others:
        lines.append("Other parts, made separately (never show their events here):\n" + "\n".join(others))
    return "\n".join(lines)


def _section(text, name):
    match = re.search(rf"^{name}:\s*\n(.*?)(?=\n\s*\n[a-z_]+:\s*\n|\Z)", text, re.S | re.M)
    return match.group(1).strip() if match else ""


def story_note(prompt):
    """(summary, last shot) of a finished H3 prompt, falling back to its beginning and end."""
    summary = _section(prompt, "summary") or prompt[:500]
    shots = re.findall(r"\[Shot \d+\].*?(?=\[Shot \d+\]|\Z)", _section(prompt, "detailed_description") or prompt, re.S)
    return summary[:600], (shots[-1].strip() if shots else prompt[-500:])[:700]


def continuation_brief(index, total, seconds, notes, last_shot, mode, prefix_seconds, transition):
    lines = [
        f"# Continuation job (part {index + 1} of {total})",
        f"You are writing the next part of ONE long video that is generated {seconds:g} seconds at a time. Every part starts from the ending of the previous one, so this part must carry on from it, not start over.",
    ]
    if notes:
        lines.append("Story so far:\n" + "\n".join(f"- Part {number}: {text}" for number, text in notes))
    lines.append("The previous part ended with: " + last_shot)
    if mode == "anchor tail":
        lines.append(
            f"The target video opens with the existing ending of the previous part ({prefix_seconds:.2f} seconds of footage that is already made). The context images show it. "
            f"Describe the first shot as starting from exactly what those frames show (same place, subjects, outfits, lighting and camera, mid-motion), and put the new action after {prefix_seconds:.2f} seconds. "
            "Never cite the context images as <Video 1> or <Picture N>; they are not labelled references.")
    elif mode == "reference video":
        lines.append("<Video 1> is the ending of the previous part. The task type is [video continuation]: the target video continues <Video 1> without restarting, keeping what it shows, and the new action follows on from its last moment.")
    lines.append(TRANSITION_TEXT[transition])
    lines.append("This is the final part: bring the story to a natural close." if index + 1 == total else "This is not the last part: do not conclude the story; leave the action open so the next part can continue.")
    lines.append("The instructions for this part in the user's text are binding: show them, and only them.")
    return "\n".join(lines)


def small(frames, longest=512):
    """Frames shrunk for the language model: the context only has to show what is happening, and every full-size frame costs a thousand tokens."""
    scale = longest / max(frames.shape[1], frames.shape[2])
    if scale >= 1:
        return frames
    resized = comfy.utils.common_upscale(frames.movedim(-1, 1), round(frames.shape[2] * scale), round(frames.shape[1] * scale), "area", "disabled")
    return resized.movedim(1, -1)


def seam_jump(previous_last, previous_before, first):
    """(jump, motion): mean picture change from the previous part's last frame to this part's first frame, and the change between the previous part's last two frames."""
    def gray(frame):
        return small(frame[None], 160).mean(dim=-1)[0] * 255
    return float((gray(first) - gray(previous_last)).abs().mean()), float((gray(previous_last) - gray(previous_before)).abs().mean())


def blend_seam(previous, images, limit=0.08, frames=8):
    """Evens out the brightness/colour flash at a join: the first `frames` frames get their average colour moved onto a straight line from the previous part's last frame to where the new part settles. A difference above `limit` is a real change and stays."""
    count = min(frames, images.shape[0] - 1)
    means = images[:count + 1].mean(dim=(1, 2))
    start, settled = previous[-1].mean(dim=(0, 1)), means[count]
    if (settled - start).abs().max() > limit:
        return images, settled - start
    steps = torch.arange(1, count + 1, dtype=images.dtype).view(-1, 1) / (count + 1)
    correction = start + (settled - start) * steps - means[:count]
    images[:count] = (images[:count] + correction.view(count, 1, 1, 3)).clamp_(0, 1)
    return images, correction.abs().max(dim=0).values


def tail_audio(waveform, sample_rate, frames):
    """The last `frames` frames of a part's audio, as an audio dict for the H3 guide."""
    count = min(waveform.shape[-1], round(frames / FPS * sample_rate))
    return {"waveform": waveform[None, :, -count:].contiguous(), "sample_rate": sample_rate}


class PROMPTFORGE_H3Generate:
    CATEGORY = h3_settings.CATEGORY
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING", "IMAGE", "AUDIO")
    RETURN_NAMES = ("frames", "audio", "prompts", "report", "first_pass_frames", "first_pass_audio")
    DESCRIPTION = ("Makes a long MiniMax H3 video part by part and returns the frames and the sound; it does not save anything, so plug it into a save node. "
                   "What to make comes from the Idea node and the Modes node (and optionally Parts, References, Video Format, Continuity, Sampling, Second Stage, Live Preview, Speed and Memory); "
                   "an unconnected settings node means its defaults. The last two outputs give the first pass (before the upscale); it is only decoded when they are connected.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The MiniMax H3 diffusion model, with the LoRAs and sigma shift you want already applied."}),
                "clip": ("CLIP", {"tooltip": "The MiniMax H3 text encoder."}),
                "vae": ("VAE", {"tooltip": "The MiniMax H3 video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "The MiniMax H3 audio VAE."}),
                "writer": ("PROMPTFORGE_MODEL", {"tooltip": "From a PromptForge Model node: the language model that writes the prompt of every part. A vision model sees the references and the end of the previous part."}),
                "idea": ("PROMPTFORGE_H3_IDEA", {"tooltip": "From a PromptForge H3 Idea node: what the video is about."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 32 - 1, "tooltip": "Part n uses seed + n, for the prompt writer and the sampler."}),
            },
            "optional": {
                "references": ("PROMPTFORGE_H3_REFERENCES", {"tooltip": "The first / last frame (I2V, L2V, FL2V) and the reference pictures, videos and sounds (REF2V). Without it the video comes from the idea alone."}),
                "format": ("PROMPTFORGE_H3_FORMAT", {"tooltip": "Size and length. Default: 16:9, 0.5 megapixels, 5 seconds."}),
                "continuity": ("PROMPTFORGE_H3_CONTINUITY", {"tooltip": "How the parts join. Default: the last 22 frames and the sound are carried forward, the joins are smoothed."}),
                "sampling": ("PROMPTFORGE_H3_SAMPLING", {"tooltip": "Sampler, steps and cfg of the first stage. Default: euler, simple, 8 steps, cfg 1."}),
                "second_stage": ("PROMPTFORGE_H3_SECOND_STAGE", {"tooltip": "Connect to upscale every part and sample it again. Not connected: one stage."}),
                "preview": ("PROMPTFORGE_H3_PREVIEW", {"tooltip": "Settings of the live preview. Default: a 4-frame loop."}),
                "performance": ("PROMPTFORGE_H3_PERFORMANCE", {"tooltip": "Attention backend and feed-forward chunking. Default: the fast settings."}),
                "modes": ("PROMPTFORGE_H3_MODES", {"tooltip": "T2V / I2V / L2V / FL2V / REF2V, one clip or parts, extending a video, the second stage. Default: T2V, one clip."}),
                "parts": ("PROMPTFORGE_H3_PARTS", {"tooltip": "From a PromptForge H3 Parts node: what happens in each part of a long video. Used when the Modes node says several parts."}),
                "continue_from": ("PROMPTFORGE_H3_CONTINUE", {"tooltip": "An existing video to extend: the first part continues from its ending and the result starts with that video."}),
                "decode_vae": ("VAE", {"tooltip": "Decode with this VAE instead of the video VAE, for example the MiniMax H3 2x detail decoder (the Loader's extra VAE). The frames then come out twice as large."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"},
        }

    def generate(self, model, clip, vae, audio_vae, writer, idea, seed, references=None, format=None, continuity=None, sampling=None, second_stage=None,
                 preview=None, performance=None, continue_from=None, modes=None, parts=None, decode_vae=None, unique_id=None, prompt=None):
        story = h3_settings.settings("idea", idea)
        refs = h3_settings.settings("references", references)
        fmt = h3_settings.settings("format", format)
        join = h3_settings.settings("continuity", continuity)
        sampling = h3_settings.settings("sampling", sampling)
        modes = h3_settings.settings("modes", modes)
        second = h3_settings.settings("second_stage", second_stage) if second_stage is not None and modes["two_stage_upscale"] else None
        if not modes["extend_existing_video"] or (continue_from is not None and continue_from["video"] is None):
            continue_from = None
        wants_first_pass = any(isinstance(link, list) and link[0] == str(unique_id) and link[1] in (4, 5) for node in (prompt or {}).values() for link in node["inputs"].values())
        preview = h3_settings.settings("preview", preview)
        perf = h3_settings.settings("performance", performance)

        kind = modes["mode"].split(" ")[0]
        several = modes["generation"] == h3_settings.GENERATIONS[1]
        if several and parts is None:
            raise ValueError("[PromptForge] several parts needs a PromptForge H3 Parts node: connect one, or set the Modes node to one clip.")
        mode = "none" if join["story_style"] == h3_settings.STORY_STYLES[2] else h3_settings.JOIN_MODES[join["how_parts_join"]]
        if kind != "REF2V" and mode == "reference video":
            mode = "anchor tail"  # only reference-to-video can take the previous ending as a reference video
        context = int(join["carried_frames"])
        if mode == "reference video":
            context = max(5, context)
        width, height = empty_latent.resolve_size("aspect + megapixels", 0, 0, "", fmt["aspect_ratio"], fmt["megapixels"], fmt["orientation"], 32)
        if perf["attention"] != "as connected":
            model = ModelAttentionBackend.execute(model, perf["attention"]).args[0]
        if int(perf["ffn_chunks"]) > 1:
            model = chunk_feed_forward(model, int(perf["ffn_chunks"]))
        model = nodes_minimax_h3.MiniMaxH3SigmaShift.execute(model, sampling["shift_video"], sampling["shift_audio"]).args[0]
        if perf["sparse_attention"]:
            model = BlockSparseAttention.execute(model, {"selection": "vsa", "keep_percent": perf["keep_percent"]}, 0.0, 1.0, SPARSE_DENSE_BLOCKS).args[0]

        if several:
            total = max(1, int(parts["parts"]))
            plan = (part_boxes(parts["part_prompts"]) + [{"text": "", "seconds": None}] * total)[:total]
            brief = story["general_prompt"].strip()
        else:
            total, plan, brief = 1, [{"text": story["general_prompt"].strip(), "seconds": None}], ""
        lengths = [float(part["seconds"] or fmt["seconds"]) for part in plan]

        text_of = lambda key: refs.get(key, "").strip()
        first, last = refs.get("first_frame"), refs.get("last_frame")
        if kind in ("I2V", "FL2V") and first is None:
            raise ValueError(f"[PromptForge] {kind} needs a first frame: connect a picture to first_frame of the References node.")
        if kind in ("L2V", "FL2V") and last is None:
            raise ValueError(f"[PromptForge] {kind} needs a last frame: connect a picture to last_frame of the References node.")
        images = [(number, refs.get(f"image_{number}")) for number in range(1, 10) if kind == "REF2V" and refs.get(f"image_{number}") is not None]
        pictures = ([first] if kind == "REF2V" and first is not None else []) + [image for _, image in images]
        picture_lines = ["<Picture 1> is the exact first frame of the video."] if kind == "REF2V" and first is not None else []
        for position, (number, _) in enumerate(images, start=len(picture_lines) + 1):
            picture_lines.append(f"<Picture {position}>: {text_of(f'description_{number}')}" if text_of(f"description_{number}") else f"<Picture {position}> is a reference picture.")
        extras = {"ref_videos": {}, "ref_video_audios": {}, "ref_audios": {}}
        video_lines, audio_lines = [], []
        for number in range(1, 4):
            video = refs.get(f"video_{number}")
            if video is None or kind != "REF2V":
                continue
            position = len(extras["ref_videos"]) + 1
            extras["ref_videos"][f"ref_video_{position}"] = video
            video_lines.append(f"<Video {position}>: {text_of(f'video_description_{number}') or 'a reference video.'}")
            soundtrack = refs.get(f"video_audio_{number}")
            if soundtrack is not None:
                extras["ref_video_audios"][f"ref_video_audio_{position}"] = soundtrack
                audio_lines.append(f"<Audio {len(extras['ref_video_audios'])}> is the soundtrack of <Video {position}>.")
        for number in range(1, 4):
            sound = refs.get(f"audio_{number}")
            if sound is None or kind != "REF2V":
                continue
            extras["ref_audios"][f"ref_audio_{len(extras['ref_audios']) + 1}"] = sound
            audio_lines.append(f"<Audio {len(extras['ref_video_audios']) + len(extras['ref_audios'])}>: {text_of(f'audio_description_{number}') or 'a sound reference.'}")
        if kind == "REF2V" and not (pictures or extras["ref_videos"] or extras["ref_audios"]):
            raise ValueError("[PromptForge] REF2V needs at least one reference: connect pictures, videos or sounds to the References node (and switch their Load nodes on).")

        def anchors(index):
            """The first frame belongs to the first part, the last frame to the last part."""
            return (first if kind in ("I2V", "FL2V", "REF2V") and index == 0 else None), (last if kind in ("L2V", "FL2V") and index == total - 1 else None)

        def writer_inputs(index):
            """The pictures the prompt writer sees for this part and what each one is."""
            if kind == "REF2V":
                return pictures, picture_lines + video_lines + audio_lines
            frames = [(frame, name) for frame, name in zip(anchors(index), ("first", "last")) if frame is not None]
            return [frame for frame, _ in frames], [f"<Picture {number}> is the exact {name} frame of the video." for number, (_, name) in enumerate(frames, start=1)]

        preset = story["prompt_template"] if story["prompt_template"] != h3_settings.AUTOMATIC_TEMPLATE else ("NSFW › " if story["prompt_style"] == "NSFW" else "") + PRESETS[kind]

        parts_frames, parts_audio, prompts, notes, report = [], [], [], [], []
        first_parts, first_waves, first_rate = [], [], 0
        tail, tail_wave, last_shot, sample_rate, previous = None, None, "", 0, None
        review = {"on": bool(story["review_prompts"]) and unique_id is not None and PromptServer.instance.client_id is not None}
        if story["review_prompts"] and not review["on"]:
            print("[PromptForge] endless video: review_prompts needs the ComfyUI page (run it from the browser); continuing without review.")
        source = None
        if continue_from is not None:
            source = continue_from["video"][..., :3]
            tail = source[-context:].clone() if context > 1 else source[-1:].clone()
            previous = source[-2:].clone()
            old = continue_from.get("audio")
            sample_rate = int(old["sample_rate"]) if old is not None else SILENCE_RATE
            tail_wave = old["waveform"][0].float().cpu() if old is not None else torch.zeros(2, round(source.shape[0] / FPS * sample_rate))
        self._status(unique_id, "Starting")
        try:
            for index in range(total):
                model_management.throw_exception_if_processing_interrupted()
                seconds = lengths[index]
                clock = time.time()
                marks = {}
                continuing = (index > 0 or source is not None) and mode != "none"
                prefix = context if continuing and mode == "anchor tail" else 0
                self._status(unique_id, f"Part {index + 1}/{total}: writing the prompt")
                idea = plan[index]["text"]
                shown, lines = writer_inputs(index)
                if idea.lower().startswith("raw:") or not story["use_prompt_writer"]:
                    prompt = idea[4:].strip() if idea.lower().startswith("raw:") else idea
                else:
                    ask = lambda attempt: self._prompt(writer, story, mode, join["story_style"], continuing, index, total, seconds, seed + index + 500 * attempt,
                                                       part_text(index, total, brief, lines, plan, seconds), shown, preset, tail if continuing else None, notes, last_shot, prefix)
                    prompt = ask(0)
                    if review["on"]:
                        clock_wait = time.time()
                        prompt = self._review(unique_id, index, total, prompt, ask, review)
                        clock += time.time() - clock_wait  # waiting for you is not generation time
                marks["prompt"] = time.time() - clock
                prompts.append(prompt)
                self._prompt_preview(unique_id, index, prompt)
                notes_entry = story_note(prompt)
                notes.append((index + 1, notes_entry[0]))
                last_shot = notes_entry[1]

                self._status(unique_id, f"Part {index + 1}/{total}: encoding the prompt")
                length = nearest_grid(prefix + seconds * FPS)
                carried = (tail if continuing else None, (tail_wave, sample_rate) if continuing else None)
                positive, latent = self._guard("encoding", lambda: self._conditioning(clip, vae, audio_vae, prompt, width, height, length, kind, pictures, anchors(index), *carried, mode, extras, continuing))
                negative = positive if sampling["cfg"] == 1.0 else clip.encode_from_tokens_scheduled(clip.tokenize(""))

                marks["encode"] = time.time() - clock - marks["prompt"]
                check = join["check_joins"] and continuing and join["story_style"] == h3_settings.STORY_STYLES[0] and previous is not None
                for attempt in range(1 + (int(join["join_retries"]) if check else 0)):
                    self._status(unique_id, f"Part {index + 1}/{total}: sampling" + (f" (try {attempt + 1})" if attempt else ""))
                    self._release()
                    clock_sample = time.time()
                    latent_out = self._guard("sampling", lambda: self._stage(model, positive, negative, latent, seed + index + 1000 * attempt, sampling["steps"], sampling, preview, unique_id, f"part {index + 1}, first stage"))
                    marks["sample"] = time.time() - clock_sample
                    first_pass = None
                    if second is not None and wants_first_pass:
                        self._status(unique_id, f"Part {index + 1}/{total}: decoding the first pass")
                        self._release()
                        first_pass = self._decode(latent_out, decode_vae, vae, audio_vae)
                    if second is not None:
                        latent_out = self._second_stage(model, vae, audio_vae, positive, negative, latent_out, second, sampling, preview, unique_id, index, total, seed + index + 1000 * attempt + 7,
                                                        kind, anchors(index), carried, mode, continuing, marks)
                    self._status(unique_id, f"Part {index + 1}/{total}: decoding")
                    self._release()
                    clock_decode = time.time()
                    frames, audio = self._guard("decoding", lambda: self._decode(latent_out, decode_vae, vae, audio_vae))
                    del latent_out
                    if not check:
                        break
                    jump, motion = seam_jump(previous[-1], previous[-2] if previous.shape[0] > 1 else previous[-1], frames[prefix])
                    ok = jump <= max(SEAM_FLOOR, SEAM_RATIO * motion)
                    print(f"[PromptForge] endless video: part {index + 1} join change {jump:.1f} (previous motion {motion:.1f}) -> {'ok' if ok else 'jump'}")
                    if ok:
                        break
                    self._status(unique_id, f"Part {index + 1}/{total}: the picture jumps at the join; trying again" if attempt < int(join["join_retries"]) else f"Part {index + 1}/{total}: still jumping, keeping it")
                del positive, negative, latent
                self._release()

                marks["decode"] = time.time() - clock_decode
                sample_rate = int(audio["sample_rate"])
                wave = audio["waveform"][0].float().cpu()
                # the decoded frames can be larger than the generation (2x decoder); the frames cut off here are the carried-forward ending
                frames = frames[prefix:]
                if join["smooth_joins"] and previous is not None and continuing:
                    frames, correction = blend_seam(previous, frames)
                    print(f"[PromptForge] endless video: part {index + 1} join colour correction up to {[round(float(value) * 255, 1) for value in correction]} (0-255)")
                wave = wave[:, round(prefix / FPS * sample_rate):round(prefix / FPS * sample_rate) + round(frames.shape[0] / FPS * sample_rate)]
                parts_frames.append(frames.half())
                parts_audio.append(wave)
                if first_pass is not None:
                    first_frames, first_audio = first_pass[0][prefix:], first_pass[1]["waveform"][0].float().cpu()
                    start = round(prefix / FPS * int(first_pass[1]["sample_rate"]))
                    first_parts.append(first_frames.half())
                    first_waves.append(first_audio[:, start:start + round(first_frames.shape[0] / FPS * int(first_pass[1]["sample_rate"]))])
                    first_rate = int(first_pass[1]["sample_rate"])
                    first_pass = None
                self._show_part(unique_id, index, frames, int(preview["size"]))
                marks["total"] = time.time() - clock
                line = f"part {index + 1}/{total} ({frames.shape[2]}x{frames.shape[1]}, {frames.shape[0]} frames): " + ", ".join(f"{name} {value:.0f}s" for name, value in marks.items())
                report.append(line)
                self._status(unique_id, f"Part {index + 1}/{total} done in {marks['total']:.0f}s ({', '.join(f'{name} {value:.0f}s' for name, value in marks.items() if name != 'total')})")

                previous = frames[-2:].clone()
                tail = frames[-context:].clone() if context > 1 else frames[-1:].clone()
                tail_wave = wave
                del frames
                gc.collect()
        finally:
            self._release()

        self._status(unique_id, "Done")
        if source is not None:
            parts_frames.insert(0, self._fit(source, parts_frames[0]).half())
            parts_audio.insert(0, self._resample(old, parts_audio[0].shape[0], sample_rate, source.shape[0]))
        video = self._assemble(parts_frames)
        text = "\n\n".join(f"--- Part {number + 1} ---\n{prompt}" for number, prompt in enumerate(prompts))
        audio = {"waveform": torch.cat(parts_audio, dim=1)[None], "sample_rate": sample_rate}
        if second is None:
            first_video, first_audio = (video, audio) if wants_first_pass else (None, None)  # without a second stage the video is the first pass
        else:
            first_video = self._assemble(first_parts) if first_parts else None
            first_audio = {"waveform": torch.cat(first_waves, dim=1)[None], "sample_rate": first_rate} if first_waves else None
        return (video, audio, text, "\n".join(report), first_video, first_audio)

    def _guard(self, label, work):
        """Runs `work`; when the GPU runs out of memory everything is unloaded and it runs once more."""
        try:
            return work()
        except model_management.OOM_EXCEPTION:
            print(f"[PromptForge] endless video: out of memory while {label}; freeing everything and trying once more.")
            free = self._release()
            try:
                return work()
            except model_management.OOM_EXCEPTION:
                print(f"[PromptForge] endless video: out of memory again while {label} ({free:.1f} GB free after unloading, {torch.cuda.memory_allocated() / 1024 ** 3:.1f} GB held by this process).")
                raise

    @staticmethod
    def _decode(latent_out, decode_vae, vae, audio_vae):
        """(frames, audio) of an AV latent."""
        frames = minimax_decode.PROMPTFORGE_MiniMaxDecode().decode(latent_out, decode_vae if decode_vae is not None else vae, 0, 0, "cpu", True)[0]
        return frames, nodes_audio.vae_decode_audio(audio_vae, {"samples": latent_out["samples"].unbind()[1]})

    @staticmethod
    def _fit(frames, like):
        """The frames of an existing video at the size of the new parts."""
        if frames.shape[1:3] == like.shape[1:3]:
            return frames
        return comfy.utils.common_upscale(frames.movedim(-1, 1), like.shape[2], like.shape[1], "lanczos", "center").movedim(1, -1)

    @staticmethod
    def _resample(audio, channels, rate, frames):
        """The sound of an existing video (silence without any) in the format of the new parts, as long as its picture."""
        count = round(frames / FPS * rate)
        if audio is None:
            return torch.zeros(channels, count)
        wave = audio["waveform"][0].float().cpu()
        if audio["sample_rate"] != rate:
            wave = torchaudio.functional.resample(wave, audio["sample_rate"], rate)
        wave = wave.mean(0, keepdim=True).expand(channels, -1) if wave.shape[0] != channels else wave
        return torch.nn.functional.pad(wave[:, :count], (0, max(0, count - wave.shape[1])))

    @staticmethod
    def _assemble(parts):
        """All parts as one IMAGE batch: float32 when the RAM allows it, else float16."""
        count = sum(part.shape[0] for part in parts)
        size = count * parts[0].shape[1] * parts[0].shape[2] * 3
        dtype = torch.float32 if size * 4 < 0.4 * psutil.virtual_memory().available else torch.float16
        if dtype == torch.float16:
            print("[PromptForge] endless video: the video is large, so the frames are returned as float16 to save RAM.")
        video = torch.empty((count, *parts[0].shape[1:]), dtype=dtype)
        position = 0
        while parts:
            part = parts.pop(0)
            video[position:position + part.shape[0]] = part
            position += part.shape[0]
        return video

    @staticmethod
    def _status(unique_id, text):
        print(f"[PromptForge] endless video: {text}")
        if unique_id is not None:
            PromptServer.instance.send_progress_text(text, unique_id)

    @staticmethod
    def _release():
        """Everything ComfyUI holds goes, so the next stage (language model, sampler, decoder) has the whole GPU. Returns the free VRAM in GB."""
        # ComfyUI drops the model compiler's CUDA graphs when a node finishes; this node runs for a whole video, so it does it between the stages
        comfy.model_prefetch.cleanup_prefetch_queues()
        model_management.unload_all_models()
        gc.collect()
        model_management.soft_empty_cache()
        if not torch.cuda.is_available():
            return 0.0
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated() / 1024 ** 3
        if held > 1.0:
            listed = [type(loaded.model.model).__name__ for loaded in model_management.current_loaded_models if not loaded.is_dead()]
            print(f"[PromptForge] endless video: {held:.1f} GB of VRAM is still held after unloading (ComfyUI still lists: {', '.join(listed) or 'no models'}); freeing again.")
            model_management.cleanup_models_gc()
            gc.collect()
            torch.cuda.empty_cache()
        return torch.cuda.mem_get_info()[0] / 1024 ** 3

    @staticmethod
    def _show_part(unique_id, index, frames, size):
        """The finished part as a short clip in the node (not saved anywhere)."""
        if unique_id is None:
            return
        scale = size / max(frames.shape[1], frames.shape[2])
        shown = frames if scale >= 1 else comfy.utils.common_upscale(frames.movedim(-1, 1), round(frames.shape[2] * scale), round(frames.shape[1] * scale), "area", "disabled").movedim(1, -1)
        mime, data = preview_override.encode((shown.clamp(0, 1) * 255).byte().numpy(), FPS, 80)
        PromptServer.instance.send_sync("promptforge.part", {"node": str(unique_id), "index": index, "mime": mime, "image": base64.b64encode(data).decode()}, PromptServer.instance.client_id)

    def _stage(self, model, positive, negative, latent, seed, steps, sampling, preview, unique_id, label, custom_sigmas="", sampler_name=None):
        """One sampling stage, previewed in this node by PromptForge's preview override unless the preview is off (or left to ComfyUI)."""
        if preview["mode"] == "live":
            model = preview_override.apply(model, unique_id, label=label, max_resolution=int(preview["size"]), frames=int(preview["frames"]), fps=int(preview["fps"]), tiny_vae=preview["decoder"])
        return sampler.PROMPTFORGE_Sampler().sample(model, positive, negative, latent, seed, steps, sampling["cfg"], sampler_name or sampling["sampler_name"], sampling["scheduler"], 1.0, 1,
                                                    custom_sigmas=custom_sigmas, preview="default" if preview["mode"] == "ComfyUI's own" else "none")[0]

    def _second_stage(self, model, vae, audio_vae, positive, negative, latent_out, second, sampling, preview, unique_id, index, total, seed, kind, anchors, carried, mode, continuing, marks):
        """Upscales the first stage's video latent and samples it again; the sound latent is kept. The conditioning of the first stage (the text
        and the references, encoded once) is used again; only the keyframes (first / last frame, the carried ending) are encoded again at the new size."""
        self._status(unique_id, f"Part {index + 1}/{total}: upscaling the latent")
        self._release()
        clock = time.time()
        video, audio_latent = latent_out["samples"].unbind()
        video = self._guard("upscaling", lambda: h3_upscaler.upscale(video, second["upscale_model"], second["size_mode"], second["scale"], second["megapixels"], second["width"], second["height"], second["align"]))
        marks["upscale"] = time.time() - clock
        latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio_latent))}
        positive = node_helpers.conditioning_set_values(positive, {"minimax_keyframes": []})
        positive = self._guard("encoding", lambda: self._guides(positive, latent, vae, audio_vae, kind, anchors, *carried, mode, continuing, True))
        if sampling["cfg"] == 1.0:
            negative = positive
        steps = int(second["steps"])
        sigmas = ", ".join(f"{value:.4f}" for value in stage2_sigmas(steps, float(second["start_sigma"])))
        self._status(unique_id, f"Part {index + 1}/{total}: sampling the second stage")
        self._release()
        clock = time.time()
        result = self._guard("sampling", lambda: self._stage(model, positive, negative, latent, seed, steps, sampling, preview, unique_id, f"part {index + 1}, second stage", sigmas, second["sampler_name"]))
        marks["sample2"] = time.time() - clock
        return result

    def _review(self, unique_id, index, total, prompt, rewrite, review):
        """Waits for the page: approve the prompt (edited or not), regenerate it, or approve this and all that follow."""
        attempt = 0
        while True:
            slot = {"event": threading.Event(), "action": None, "text": prompt}
            _pending[str(unique_id)] = slot
            PromptServer.instance.send_sync("promptforge.review", {"node": str(unique_id), "index": index, "total": total, "text": prompt, "attempt": attempt}, PromptServer.instance.client_id)
            self._status(unique_id, f"Part {index + 1}/{total}: waiting for you to approve the prompt")
            try:
                while not slot["event"].wait(0.5):
                    model_management.throw_exception_if_processing_interrupted()
            finally:
                _pending.pop(str(unique_id), None)
            if slot["action"] == "regenerate":
                attempt += 1
                self._status(unique_id, f"Part {index + 1}/{total}: writing the prompt again")
                prompt = rewrite(attempt)
                continue
            if slot["action"] == "approve_all":
                review["on"] = False
            return slot["text"].strip() or prompt

    @staticmethod
    def _prompt_preview(unique_id, index, prompt):
        if unique_id is not None:
            PromptServer.instance.send_sync("promptforge.prompt", {"node": str(unique_id), "index": index, "text": prompt}, PromptServer.instance.client_id)

    def _prompt(self, writer, story, mode, story_style, continuing, index, total, seconds, seed, text, pictures, preset, tail, notes, last_shot, prefix):
        package = _package()
        refs = (list(pictures) + [None] * 3)[:3]
        continuation = continuation_brief(index, total, seconds, notes[-4:], last_shot, mode, prefix / FPS, story_style) if continuing else ""
        context = small(tail) if (continuing and tail is not None) else None
        for attempt in range(2):
            free = self._release()
            print(f"[PromptForge] endless video: {free:.1f} GB of VRAM free for the prompt writer.")
            try:
                # a CUDA error inside llama.cpp cannot be caught and ends ComfyUI, so with little free VRAM the writer starts smaller
                settings = writer
                if attempt or free < 10:
                    settings = dict(writer, kv_cache_type="q8_0", **({"context_size": 12288} if attempt or free < 8.5 else {}))
                answer, _ = package.PROMPTFORGE_PromptEnhancer().enhance(
                    settings, text, "inputs" if (pictures or context is not None) else "text_only", preset, seed + attempt * 1000, story["writer_max_tokens"], story["writer_creativity"], story["writer_top_p"], story["writer_top_k"], 1, story["writer_repetition_penalty"], "",
                    video_seconds=max(1, round(seconds + prefix / FPS)), image=refs[0], image2=refs[1], image3=refs[2],
                    video=context if mode == "reference video" else None, frame_count=6 - 2 * attempt,
                    negative_prompt=story["negative_prompt"], auto_retry=story["check_prompts"], dialogue=story["dialogue"], dialogue_language=story["dialogue_language"],
                    dialogue_language_other=story["dialogue_language_other"], continuation=continuation,
                    context_video=context if mode == "anchor tail" else None)
                return answer
            except ValueError as error:
                if attempt or "llama context" not in str(error):
                    raise
                print("[PromptForge] endless video: the prompt writer could not start (out of VRAM?); freeing memory and trying again with a smaller context.")
                time.sleep(3)

    def _conditioning(self, clip, vae, audio_vae, prompt, width, height, length, kind, pictures, anchors, tail, tail_wave, mode, extras, continuing):
        """The positive conditioning and the empty AV latent. How they are made depends on the mode: T2V, I2V, L2V and FL2V encode the
        first / last frame as keyframes of the image-to-video node (a center-cropped copy, so nothing is stretched); REF2V encodes the references."""
        first, last = anchors
        if kind == "REF2V":
            refs = {f"ref_image_{number + 1}": image for number, image in enumerate(pictures)}
            videos = dict(extras["ref_videos"])
            if continuing and mode == "reference video" and len(videos) < 3:
                videos[f"ref_video_{len(videos) + 1}"] = tail
            positive, latent = nodes_minimax_h3.MiniMaxH3ReferenceToVideo.execute(
                clip, prompt, width, height, length, "match", vae=vae, audio_vae=audio_vae, ref_images=refs, ref_videos=videos, ref_video_audios=extras["ref_video_audios"], ref_audios=extras["ref_audios"]).args
        else:
            fit = lambda image: None if image is None else comfy.utils.common_upscale(image[:1, ..., :3].movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
            positive, latent = nodes_minimax_h3.MiniMaxH3ImageToVideo.execute(clip, vae, prompt, width, height, length, first_frame=fit(first), last_frame=fit(last)).args
        return self._guides(positive, latent, vae, audio_vae, kind, anchors, tail, tail_wave, mode, continuing, kind == "REF2V"), latent

    @staticmethod
    def _guides(positive, latent, vae, audio_vae, kind, anchors, tail, tail_wave, mode, continuing, frames):
        """Keyframes of the latent's size: the first and the last frame (`frames`; I2V, L2V and FL2V already have them from their encoder) and the carried ending of the previous part."""
        first, last = anchors
        if frames and first is not None:
            positive = nodes_minimax_h3.MiniMaxH3AddGuide.execute(positive, latent, 0, vae=vae, image=first[:1]).args[0]
        if frames and last is not None:
            positive = nodes_minimax_h3.MiniMaxH3AddGuide.execute(positive, latent, -1, vae=vae, image=last[:1]).args[0]
        if continuing and mode == "anchor tail":
            positive = nodes_minimax_h3.MiniMaxH3AddGuide.execute(positive, latent, 0, vae=vae, audio_vae=audio_vae, image=tail, audio=tail_audio(tail_wave[0], tail_wave[1], tail.shape[0])).args[0]
        return positive


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_H3Generate": PROMPTFORGE_H3Generate}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_H3Generate": "PromptForge H3 Endless Video"}
