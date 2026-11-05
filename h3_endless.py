"""PromptForge MiniMax H3 Endless Video: one node that makes a long video part by part.

Every part is a 5 or 10 second MiniMax H3 generation. The first one follows the user's instructions (text only,
reference images, or a first frame). Each following part starts from the ending of the previous one:

* anchor tail (default): the last frames and the last audio of the previous part are anchored at the start of the
  new part with H3's own guide mechanism, so the new part continues frame-exactly; the anchored frames are cut off
  when the parts are joined,
* reference video: the previous ending is given to H3 as a reference video (the model's native continuation),
* none: every part is generated independently from the references (cuts).

For each part the Enhancer writes a fresh prompt from the whole-video brief, the story so far and the frames it is
continuing; the part is sampled, decoded with the tiled H3 decoder, previewed, and the models are released
between the stages so the language model, the diffusion model and the decoders never sit in VRAM together.
Parts are encoded as they finish and joined at the end with one audio track, so the length is only limited by disk.
"""
import gc
import importlib
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from server import PromptServer

import comfy.model_management as model_management
import comfy.samplers
import comfy.utils
import folder_paths
from comfy_extras import nodes_audio, nodes_minimax_h3

from . import empty_latent, minimax_decode, sampler, video_save

FPS = 24
GRID = 17
SEAM_FLOOR = 25.0
SEAM_RATIO = 6.0
CONTINUITY = ("anchor tail", "reference video", "none")
TRANSITIONS = ("continue the same shot", "let the story decide", "cut to a new shot")
CONTEXTS = ("1", "5", "22", "39", "56", "73")
SECONDS = ("5", "10")
PRESET_STYLES = ("standard", "NSFW")
DIALOGUE_AUTO = "auto (from the instructions)"
DEFAULT_OPTIONS = {
    "parts": 3, "seconds": 5, "aspect_ratio": "16:9", "orientation": "landscape", "megapixels": 0.5, "transition": TRANSITIONS[0],
    "continuity": CONTINUITY[0], "context_frames": 22, "part_ideas": "", "preset_style": "standard", "negative_prompt": "",
    "dialogue": "", "dialogue_language": DIALOGUE_AUTO, "dialogue_language_other": "", "temperature": 0.8, "seam_blend": True, "seam_check": False, "seam_retries": 1, "steps": 8, "cfg": 1.0,
    "sampler_name": "euler", "scheduler": "simple", "codec": "h264 (nvenc)", "quality": 19, "keep_parts": False,
}
TRANSITION_TEXT = {
    TRANSITIONS[0]: "Stay in the same continuous shot: same place, subjects, outfits, lighting and camera, with the action moving on. No cut in the first shot of this part.",
    TRANSITIONS[1]: "Continue naturally. If the story calls for it you may cut to a new angle, shot or place, but keep the same subjects, outfits and look.",
    TRANSITIONS[2]: "Open this part with a cut to a new shot (a new angle, framing or place that fits the story). Keep the same subjects, outfits and look.",
}


def _package():
    return importlib.import_module(__package__)


def nearest_grid(frames):
    """The frame count on H3's 17k + 5 grid that is closest to `frames`."""
    return GRID * max(0, round((frames - 5) / GRID)) + 5


PART_MARKERS = (
    re.compile(r"^\s*(?:part|segment|scene|shot)\s*(\d+)\s*(?:\(\s*(\d+(?:\.\d+)?)\s*s(?:ec(?:onds?)?)?\s*\))?\s*[:.\-)]\s*(.*)$", re.I),
    re.compile(r"^\s*(\d+)\s*[.)]\s+()(.*)$"),
    re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*:\s*(.*)$", re.I),
)


def parse_plan(text, leading_is_part):
    """(brief, parts) from a text that splits into parts with 'Part 2: ...', '2. ...', '0-5s: ...' or lines of only ---.

    Each part is {"text", "seconds"} (seconds is None unless given as 'Part 2 (10s):' or '5-15s:'). Text before the first
    marker is the brief, or the first part when `leading_is_part`.
    """
    brief, parts, marked = [], [], False
    leading = {"text": [], "seconds": None} if leading_is_part else None
    current = leading
    for line in (text or "").splitlines():
        if re.fullmatch(r"\s*-{3,}\s*", line):
            current = {"text": [], "seconds": None}
            parts.append(current)
            continue
        match = next((m for m in (pattern.match(line) for pattern in PART_MARKERS) if m), None)
        if match:
            marked = True
            seconds = None
            if match.re is PART_MARKERS[0] and match.group(2):
                seconds = float(match.group(2))
            elif match.re is PART_MARKERS[2]:
                seconds = float(match.group(2)) - float(match.group(1))
            current = {"text": [match.group(3)], "seconds": seconds if seconds and seconds > 0 else None}
            parts.append(current)
        elif current is not None:
            current["text"].append(line)
        else:
            brief.append(line)
    if leading is not None and not marked and any(piece.strip() for piece in leading["text"]):
        parts.insert(0, leading)
    return "\n".join(brief).strip(), [{"text": " ".join(piece.strip() for piece in part["text"] if piece.strip()), "seconds": part["seconds"]} for part in parts]


def part_text(index, total, brief, reference_lines, plan, seconds):
    """What the Enhancer is asked for one part: the brief as context, this part's idea as the thing to show, the other parts as things to keep out."""
    lines = ([f"Whole-video context (do not show all of it in this part): {brief}"] if brief else []) + list(reference_lines)
    idea = plan[index]["text"] if index < len(plan) else ""
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


class PROMPTFORGE_H3Options:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "parts": ("INT", {"default": DEFAULT_OPTIONS["parts"], "min": 1, "max": 200, "tooltip": "How many parts the video is made of."}),
                "seconds_per_part": (list(SECONDS), {"default": "5", "tooltip": "New seconds of video per part."}),
                "aspect_ratio": (list(empty_latent.RATIOS), {"default": "16:9"}),
                "orientation": (["landscape", "portrait"], {"default": "landscape"}),
                "megapixels": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05, "tooltip": "Size of every part. 0.5 is a good first-pass size."}),
                "transition": (list(TRANSITIONS), {"default": TRANSITIONS[0], "tooltip": "How a part follows the previous one: carry on in the same shot, let the story decide, or cut to a new shot each time (cuts do not carry the previous frames forward)."}),
                "continuity": (list(CONTINUITY), {"default": CONTINUITY[0], "tooltip": "anchor tail: the previous ending (frames and audio) is anchored at the start of the new part, frame-exact. reference video: H3's own continuation from a reference video. none: every part is independent."}),
                "context_frames": (list(CONTEXTS), {"default": "22", "tooltip": "Frames of the previous part carried forward (24 per second). More context follows the motion better but costs a little time per part. 1 carries only the last frame."}),
                "part_ideas": ("STRING", {"default": "", "multiline": True, "placeholder": "Part 1: what happens in the first part\nPart 2: what happens in the second part\nPart 3 (10s): a longer part", "tooltip": "What each part shows. Start lines with 'Part 2:', '2.' or '0-5s:' (add (10s) to give one part its own length), or separate parts with a line containing only ---. Start a part with raw: to use your own finished prompt for it. The same markers also work inside the instructions text of the Endless node."}),
            },
            "optional": {
                "preset_style": (list(PRESET_STYLES), {"default": "standard", "tooltip": "Which family of MiniMax H3 prompt presets the Enhancer uses."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "dialogue": ("STRING", {"default": "", "multiline": True, "tooltip": "What the dialogue should be about and how it sounds. Written fresh for every part, sized to its length."}),
                "dialogue_language": ([DIALOGUE_AUTO, "Arabic", "Chinese", "English", "French", "German", "Italian", "Japanese", "Korean", "Portuguese", "Russian", "Spanish", "other (type below)"], {"default": DIALOGUE_AUTO}),
                "dialogue_language_other": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Creativity of the prompt writer. Lower sticks closer to your text."}),
                "seam_blend": ("BOOLEAN", {"default": True, "tooltip": "Evens out the brief light flash at the join: the first third of a second of the new part is brought onto a straight brightness/colour line from the previous part's last frame. Skipped when the change is large enough to be a real cut."}),
                "seam_check": ("BOOLEAN", {"default": False, "tooltip": "When a part is meant to continue the same shot, compare its first frame with the previous part's last frame and generate it again with another seed if the picture jumps. Not used when cuts are allowed."}),
                "seam_retries": ("INT", {"default": 1, "min": 0, "max": 5, "tooltip": "How many times a jumping part is generated again."}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 200, "tooltip": "6-8 with a turbo LoRA, more without."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "codec": (list(video_save.CODECS), {"default": "h264 (nvenc)"}),
                "quality": ("INT", {"default": 19, "min": 0, "max": 63}),
                "keep_parts": ("BOOLEAN", {"default": False, "tooltip": "Also keep every part (with its audio) as its own file in the output folder."}),
            },
        }

    RETURN_TYPES = ("PROMPTFORGE_H3_OPTIONS",)
    RETURN_NAMES = ("OPTIONS",)
    FUNCTION = "build"
    CATEGORY = "🔮 PromptForge"
    DESCRIPTION = "Settings for the PromptForge H3 Endless Video node: parts, length, size, how parts continue, an idea per part, dialogue and sampling."

    def build(self, parts, seconds_per_part, aspect_ratio, orientation, megapixels, transition, continuity, context_frames, part_ideas, **optional):
        options = dict(DEFAULT_OPTIONS)
        options.update(optional, parts=parts, seconds=int(seconds_per_part), aspect_ratio=aspect_ratio, orientation=orientation, megapixels=megapixels,
                       transition=transition, continuity=continuity, context_frames=int(context_frames), part_ideas=part_ideas)
        return (options,)


class PROMPTFORGE_H3Endless:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The MiniMax H3 model, with LoRAs, shift and the preview override already applied."}),
                "clip": ("CLIP",),
                "vae": ("VAE", {"tooltip": "The MiniMax H3 video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "The MiniMax H3 audio VAE."}),
                "enhancer": ("PROMPTFORGE_MODEL", {"tooltip": "From a PromptForge Model node: the language model that writes every part's prompt."}),
                "instructions": ("STRING", {"default": "", "multiline": True, "tooltip": "What the whole video is about. For reference images, say who or what each one is."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 32 - 1, "tooltip": "Part n uses seed + n for the prompt and for the sampler."}),
                "filename_prefix": ("STRING", {"default": "PromptForge/endless"}),
            },
            "optional": {
                "options": ("PROMPTFORGE_H3_OPTIONS", {"tooltip": "From a PromptForge H3 Options node. Without it, 3 parts of 5 seconds."}),
                "decode_vae": ("VAE", {"tooltip": "Optional: decode the video with this VAE instead of the main one, for example the MiniMax H3 2x detail VAE (the Loader's extra_vae). Frames then come out at twice the generation size."}),
                "first_frame": ("IMAGE", {"tooltip": "Optional first frame of the first part."}),
                "ref_image_1": ("IMAGE", {"tooltip": "Reference images are used in every part, so the characters stay the same."}),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
                "ref_text_1": ("STRING", {"forceInput": True, "tooltip": "Description of reference image 1, for example the DESCRIPTION output of a PromptForge Load Image node."}),
                "ref_text_2": ("STRING", {"forceInput": True}),
                "ref_text_3": ("STRING", {"forceInput": True}),
                "voice_audio": ("AUDIO", {"tooltip": "Optional voice reference, used in every part (<Audio 1>)."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("VIDEO_PATH", "PROMPTS")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "🔮 PromptForge"
    DESCRIPTION = "Makes a long MiniMax H3 video part by part: writes each part's prompt, continues from the previous ending, previews every part and joins them into one video."

    def generate(self, model, clip, vae, audio_vae, enhancer, instructions, seed, filename_prefix, options=None, decode_vae=None, first_frame=None,
                 ref_image_1=None, ref_image_2=None, ref_image_3=None, ref_text_1="", ref_text_2="", ref_text_3="", voice_audio=None, unique_id=None):
        opts = {**DEFAULT_OPTIONS, **(options or {})}
        default_seconds = int(opts["seconds"])
        mode = "none" if opts["transition"] == TRANSITIONS[2] else opts["continuity"]
        context = int(opts["context_frames"])
        if mode == "reference video":
            context = max(5, context)
        width, height = empty_latent.resolve_size("aspect + megapixels", 0, 0, "", opts["aspect_ratio"], opts["megapixels"], opts["orientation"], 32)

        brief, plan = parse_plan(instructions, False)
        listed = parse_plan(opts["part_ideas"], True)[1]
        plan = [part for part in (listed or plan)]
        brief = brief if plan and not listed else instructions.strip()
        total = max(int(opts["parts"]), len(plan))
        if len(plan) > int(opts["parts"]):
            print(f"[PromptForge] endless video: {len(plan)} parts were described, so {len(plan)} parts will be made.")
        plan += [{"text": "", "seconds": None}] * (total - len(plan))
        lengths = [float(part["seconds"] or default_seconds) for part in plan]

        pictures = [image for image in (first_frame, ref_image_1, ref_image_2, ref_image_3) if image is not None][:3]
        texts = [text for image, text in zip((ref_image_1, ref_image_2, ref_image_3), (ref_text_1, ref_text_2, ref_text_3)) if image is not None]
        offset = 1 if first_frame is not None else 0
        reference_lines = []
        if first_frame is not None:
            reference_lines.append("<Picture 1> is the exact first frame of the video.")
        for number, text in enumerate(texts):
            if text and text.strip() and number + offset < len(pictures):
                reference_lines.append(f"<Picture {number + offset + 1}>: {text.strip()}")
        if voice_audio is not None:
            reference_lines.append("<Audio 1> is the voice-timbre reference for the main speaker (S1).")

        scratch = tempfile.mkdtemp(prefix="promptforge_endless_")
        parts_video, parts_audio, prompts, notes = [], [], [], []
        tail, tail_wave, last_shot, sample_rate, previous = None, None, "", 0, None
        self._status(unique_id, "Starting")
        try:
            for index in range(total):
                model_management.throw_exception_if_processing_interrupted()
                seconds = lengths[index]
                clock = time.time()
                marks = {}
                continuing = index > 0 and mode != "none"
                prefix = context if continuing and mode == "anchor tail" else 0
                self._status(unique_id, f"Part {index + 1}/{total}: writing the prompt")
                idea = plan[index]["text"]
                if idea.lower().startswith("raw:"):
                    prompt = idea[4:].strip()
                else:
                    prompt = self._prompt(enhancer, opts, mode, continuing, index, total, seconds, seed + index, part_text(index, total, brief, reference_lines, plan, seconds),
                                          pictures, tail if continuing else None, notes, last_shot, prefix)
                marks["prompt"] = time.time() - clock
                prompts.append(prompt)
                self._prompt_preview(unique_id, index, prompt)
                notes_entry = story_note(prompt)
                notes.append((index + 1, notes_entry[0]))
                last_shot = notes_entry[1]

                self._status(unique_id, f"Part {index + 1}/{total}: encoding the prompt")
                length = nearest_grid(prefix + seconds * FPS)
                positive, latent = self._conditioning(clip, vae, audio_vae, prompt, width, height, length, pictures, first_frame if index == 0 else None,
                                                      tail if continuing else None, (tail_wave, sample_rate) if continuing else None, mode, voice_audio, continuing)
                negative = positive if opts["cfg"] == 1.0 else clip.encode_from_tokens_scheduled(clip.tokenize(""))

                marks["encode"] = time.time() - clock - marks["prompt"]
                check = opts["seam_check"] and continuing and opts["transition"] == TRANSITIONS[0] and previous is not None
                for attempt in range(1 + (int(opts["seam_retries"]) if check else 0)):
                    self._status(unique_id, f"Part {index + 1}/{total}: sampling" + (f" (try {attempt + 1})" if attempt else ""))
                    self._release()
                    clock_sample = time.time()
                    latent_out = sampler.PROMPTFORGE_Sampler().sample(model, positive, negative, latent, seed + index + 1000 * attempt, opts["steps"], opts["cfg"], opts["sampler_name"], opts["scheduler"], 1.0, 1)[0]
                    marks["sample"] = time.time() - clock_sample
                    self._status(unique_id, f"Part {index + 1}/{total}: decoding")
                    self._release()
                    clock_decode = time.time()
                    images = minimax_decode.PROMPTFORGE_MiniMaxDecode().decode(latent_out, decode_vae if decode_vae is not None else vae, 0, 0, "cpu", True)[0]
                    audio = nodes_audio.vae_decode_audio(audio_vae, {"samples": latent_out["samples"].unbind()[1]})
                    del latent_out
                    if not check:
                        break
                    jump, motion = seam_jump(previous[-1], previous[-2] if previous.shape[0] > 1 else previous[-1], images[prefix])
                    ok = jump <= max(SEAM_FLOOR, SEAM_RATIO * motion)
                    print(f"[PromptForge] endless video: part {index + 1} seam change {jump:.1f} (previous motion {motion:.1f}) -> {'ok' if ok else 'jump'}")
                    if ok:
                        break
                    self._status(unique_id, f"Part {index + 1}/{total}: the picture jumps at the join; trying again" if attempt < int(opts["seam_retries"]) else f"Part {index + 1}/{total}: still jumping, keeping it")
                del positive, negative, latent
                self._release()

                marks["decode"] = time.time() - clock_decode
                sample_rate = int(audio["sample_rate"])
                wave = audio["waveform"][0].float().cpu()
                # the decoded frames can be larger than the generation (2x decoder); the frames cut off here are the anchored ending
                images = images[prefix:]
                if opts["seam_blend"] and previous is not None and continuing:
                    images, offset = blend_seam(previous, images)
                    print(f"[PromptForge] endless video: part {index + 1} join colour correction up to {[round(float(value) * 255, 1) for value in offset]} (0-255)")
                wave = wave[:, round(prefix / FPS * sample_rate):round(prefix / FPS * sample_rate) + round(images.shape[0] / FPS * sample_rate)]
                preview = self._save_part(images, wave, sample_rate, filename_prefix, opts, index, scratch, parts_video, opts["keep_parts"])
                parts_audio.append(wave)
                self._preview(unique_id, preview)
                marks["total"] = time.time() - clock
                self._status(unique_id, f"Part {index + 1}/{total} done in {marks['total']:.0f}s (" + ", ".join(f"{name} {value:.0f}s" for name, value in marks.items() if name != "total") + ")")

                previous = images[-2:].clone()
                tail = images[-context:].clone() if context > 1 else images[-1:].clone()
                tail_wave = wave
                del images
                gc.collect()

            self._status(unique_id, "Joining the parts")
            final = self._join(parts_video, parts_audio, sample_rate, filename_prefix, scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
            self._release()

        self._status(unique_id, "Done")
        text = "\n\n".join(f"--- Part {number + 1} ---\n{prompt}" for number, prompt in enumerate(prompts))
        return {"ui": {"promptforge_video": [final[1]]}, "result": (str(final[0]), text)}

    @staticmethod
    def _status(unique_id, text):
        print(f"[PromptForge] endless video: {text}")
        if unique_id is not None:
            PromptServer.instance.send_progress_text(text, unique_id)

    @staticmethod
    def _release():
        """Everything ComfyUI holds goes, so the next stage (language model, sampler, decoder) has the whole GPU. Returns the free VRAM in GB."""
        model_management.unload_all_models()
        gc.collect()
        model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            if os.environ.get("PROMPTFORGE_DEBUG_VRAM"):
                live = sorted(((obj.numel() * obj.element_size(), tuple(obj.shape), str(obj.dtype)) for obj in gc.get_objects() if torch.is_tensor(obj) and obj.is_cuda), reverse=True)[:6]
                print(f"[PromptForge] VRAM debug: allocated {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB, reserved {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB, largest live tensors {live}")
            return torch.cuda.mem_get_info()[0] / 1024 ** 3
        return 0.0

    @staticmethod
    def _preview(unique_id, preview):
        if unique_id is not None:
            PromptServer.instance.send_sync("executed", {"node": str(unique_id), "display_node": str(unique_id), "output": {"promptforge_video": [preview]},
                                                         "prompt_id": next((item[1] for item in PromptServer.instance.prompt_queue.currently_running.values()), None)}, PromptServer.instance.client_id)

    @staticmethod
    def _prompt_preview(unique_id, index, prompt):
        if unique_id is not None:
            PromptServer.instance.send_sync("executed", {"node": str(unique_id), "display_node": str(unique_id), "output": {"promptforge_prompt": [{"index": index, "text": prompt}]},
                                                         "prompt_id": next((item[1] for item in PromptServer.instance.prompt_queue.currently_running.values()), None)}, PromptServer.instance.client_id)

    def _prompt(self, enhancer, opts, mode, continuing, index, total, seconds, seed, text, pictures, tail, notes, last_shot, prefix):
        package = _package()
        refs = list(pictures) + [None] * (3 - len(pictures))
        have_refs = bool(pictures) or (continuing and mode == "reference video")
        preset = ("NSFW › " if opts["preset_style"] == "NSFW" else "") + ("MiniMax H3 Ref2VA" if have_refs else "MiniMax H3 T2VA")
        continuation = continuation_brief(index, total, seconds, notes[-4:], last_shot, mode, prefix / FPS, opts["transition"]) if continuing else ""
        context = small(tail) if (continuing and tail is not None) else None
        for attempt in range(2):
            free = self._release()
            print(f"[PromptForge] endless video: {free:.1f} GB of VRAM free for the prompt writer.")
            try:
                settings = dict(enhancer, context_size=8192, kv_cache_type="q8_0") if attempt else enhancer
                answer, _ = package.PROMPTFORGE_PromptEnhancer().enhance(
                    settings, text, "inputs" if (pictures or context is not None) else "text_only", preset, seed + attempt * 1000, 2048, opts["temperature"], 0.95, 40, 1, 1.1, "",
                    video_seconds=max(1, round(seconds + prefix / FPS)), image=refs[0], image2=refs[1], image3=refs[2],
                    video=context if mode == "reference video" else None, frame_count=6 - 2 * attempt,
                    negative_prompt=opts["negative_prompt"], auto_retry=True, dialogue=opts["dialogue"], dialogue_language=opts["dialogue_language"],
                    dialogue_language_other=opts["dialogue_language_other"], continuation=continuation,
                    context_video=context if mode == "anchor tail" else None)
                return answer
            except ValueError as error:
                if attempt or "llama context" not in str(error):
                    raise
                print("[PromptForge] endless video: the prompt writer could not start (out of VRAM?); freeing memory and trying again with a smaller context.")
                time.sleep(3)

    def _conditioning(self, clip, vae, audio_vae, prompt, width, height, length, pictures, first_frame, tail, tail_wave, mode, voice_audio, continuing):
        refs = {f"ref_image_{number + 1}": image for number, image in enumerate(pictures)}
        videos, soundtracks = {}, {}
        if continuing and mode == "reference video":
            videos["ref_video_1"] = tail
        voices = {"ref_audio_1": voice_audio} if voice_audio is not None else {}
        if refs or videos or voices:
            positive, latent = nodes_minimax_h3.MiniMaxH3ReferenceToVideo.execute(
                clip, prompt, width, height, length, "match", vae=vae, audio_vae=audio_vae, ref_images=refs, ref_videos=videos, ref_video_audios=soundtracks, ref_audios=voices).args
        else:
            positive, latent = nodes_minimax_h3.MiniMaxH3ImageToVideo.execute(clip, vae, prompt, width, height, length).args
        if first_frame is not None:
            positive = nodes_minimax_h3.MiniMaxH3AddGuide.execute(positive, latent, 0, vae=vae, image=first_frame[:1]).args[0]
        if continuing and mode == "anchor tail":
            guide_audio = tail_audio(tail_wave[0], tail_wave[1], tail.shape[0])
            positive = nodes_minimax_h3.MiniMaxH3AddGuide.execute(positive, latent, 0, vae=vae, audio_vae=audio_vae, image=tail, audio=guide_audio).args[0]
        return positive, latent

    def _save_part(self, images, wave, sample_rate, filename_prefix, opts, index, scratch, parts_video, keep):
        saver = video_save.PROMPTFORGE_SaveVideo()
        quality = dict(frame_rate=float(FPS), codec=opts["codec"], speed="fast", rate_control="quality", quality=opts["quality"], bitrate_mbit=15, bit_depth="8", pingpong=False, save_metadata=False)
        silent = saver.save(images, filename_prefix=f"promptforge_endless_{os.path.basename(scratch)}/part_{index + 1:03}", save_output=False, **quality)["ui"]["promptforge_video"][0]
        parts_video.append(Path(folder_paths.get_temp_directory()) / silent["subfolder"] / silent["filename"])
        sound = {"waveform": wave[None], "sample_rate": sample_rate}
        name = f"{filename_prefix}_part_{index + 1:03}" if keep else f"promptforge_endless_{os.path.basename(scratch)}/preview_{index + 1:03}"
        return saver.save(images, filename_prefix=name, save_output=keep, audio=sound, **quality)["ui"]["promptforge_video"][0]

    def _join(self, parts_video, parts_audio, sample_rate, filename_prefix, scratch):
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        os.makedirs(folder, exist_ok=True)
        filename = f"{name}_{counter:05}.mp4"
        target = Path(folder) / filename
        listing = Path(scratch) / "parts.txt"
        listing.write_text("".join(f"file '{path.as_posix()}'\n" for path in parts_video), encoding="utf-8")
        wave = torch.cat(parts_audio, dim=1)
        audio_path = Path(scratch) / "audio.f32"
        audio_path.write_bytes(wave.t().contiguous().numpy().tobytes())
        args = [video_save._ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                "-f", "f32le", "-ar", str(sample_rate), "-ac", str(wave.shape[0]), "-i", str(audio_path),
                "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(target)]
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode:
            target.unlink(missing_ok=True)
            raise RuntimeError("[PromptForge] joining the parts failed: " + result.stderr[-300:])
        for path in parts_video:
            path.unlink(missing_ok=True)
        return target, {"filename": filename, "subfolder": subfolder, "type": "output"}


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_H3Options": PROMPTFORGE_H3Options, "PROMPTFORGE_H3Endless": PROMPTFORGE_H3Endless}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_H3Options": "PromptForge H3 Endless Options", "PROMPTFORGE_H3Endless": "PromptForge H3 Endless Video"}
