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
from pathlib import Path

import torch
from server import PromptServer

import comfy.model_management as model_management
import comfy.samplers
import folder_paths
from comfy_extras import nodes_audio, nodes_minimax_h3

from . import empty_latent, minimax_decode, sampler, video_save

FPS = 24
GRID = 17
CONTINUITY = ("anchor tail", "reference video", "none")
TRANSITIONS = ("continue the same shot", "let the story decide", "cut to a new shot")
CONTEXTS = ("1", "5", "22", "39", "56", "73")
SECONDS = ("5", "10")
PRESET_STYLES = ("standard", "NSFW")
DIALOGUE_AUTO = "auto (from the instructions)"
DEFAULT_OPTIONS = {
    "parts": 3, "seconds": 5, "aspect_ratio": "16:9", "orientation": "landscape", "megapixels": 0.5, "transition": TRANSITIONS[0],
    "continuity": CONTINUITY[0], "context_frames": 22, "part_ideas": "", "preset_style": "standard", "negative_prompt": "",
    "dialogue": "", "dialogue_language": DIALOGUE_AUTO, "dialogue_language_other": "", "steps": 8, "cfg": 1.0,
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


def split_ideas(text, parts):
    """One idea per part from text separated by lines with only ---; missing ones are empty."""
    ideas = [piece.strip() for piece in re.split(r"^\s*---+\s*$", text or "", flags=re.M)]
    return (ideas + [""] * parts)[:parts]


def _section(text, name):
    match = re.search(rf"^{name}:\s*\n(.*?)(?=\n\s*\n[a-z_]+:\s*\n|\Z)", text, re.S | re.M)
    return match.group(1).strip() if match else ""


def story_note(prompt):
    """(summary, last shot) of a finished H3 prompt, falling back to its beginning and end."""
    summary = _section(prompt, "summary") or prompt[:500]
    shots = re.findall(r"\[Shot \d+\].*?(?=\[Shot \d+\]|\Z)", _section(prompt, "detailed_description") or prompt, re.S)
    return summary[:600], (shots[-1].strip() if shots else prompt[-500:])[:700]


def continuation_brief(index, total, seconds, instructions, notes, last_shot, mode, prefix_seconds, transition, idea):
    lines = [
        f"# Continuation job (part {index + 1} of {total})",
        f"You are writing the next part of ONE long video that is generated {seconds} seconds at a time. Every part starts from the ending of the previous one, so this part must carry on from it, not start over.",
        "Whole-video brief: " + instructions.strip(),
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
    if idea:
        lines.append("What this part must show (from the user): " + idea)
    return "\n".join(lines)


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
                "part_ideas": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional: what each part should show. Separate the parts with a line containing only ---. Parts without an idea are continued from the story automatically."}),
            },
            "optional": {
                "preset_style": (list(PRESET_STYLES), {"default": "standard", "tooltip": "Which family of MiniMax H3 prompt presets the Enhancer uses."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "dialogue": ("STRING", {"default": "", "multiline": True, "tooltip": "What the dialogue should be about and how it sounds. Written fresh for every part, sized to its length."}),
                "dialogue_language": ([DIALOGUE_AUTO, "Arabic", "Chinese", "English", "French", "German", "Italian", "Japanese", "Korean", "Portuguese", "Russian", "Spanish", "other (type below)"], {"default": DIALOGUE_AUTO}),
                "dialogue_language_other": ("STRING", {"default": ""}),
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

    def generate(self, model, clip, vae, audio_vae, enhancer, instructions, seed, filename_prefix, options=None, first_frame=None,
                 ref_image_1=None, ref_image_2=None, ref_image_3=None, ref_text_1="", ref_text_2="", ref_text_3="", voice_audio=None, unique_id=None):
        opts = {**DEFAULT_OPTIONS, **(options or {})}
        total, seconds = int(opts["parts"]), int(opts["seconds"])
        mode = "none" if opts["transition"] == TRANSITIONS[2] else opts["continuity"]
        context = int(opts["context_frames"])
        if mode == "reference video":
            context = max(5, context)
        width, height = empty_latent.resolve_size("aspect + megapixels", 0, 0, "", opts["aspect_ratio"], opts["megapixels"], opts["orientation"], 32)
        ideas = split_ideas(opts["part_ideas"], total)

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
        tail, tail_wave, last_shot, sample_rate = None, None, "", 0
        self._status(unique_id, "Starting")
        try:
            for index in range(total):
                model_management.throw_exception_if_processing_interrupted()
                continuing = index > 0 and mode != "none"
                prefix = context if continuing and mode == "anchor tail" else 0
                self._status(unique_id, f"Part {index + 1}/{total}: writing the prompt")
                prompt = self._prompt(enhancer, opts, mode, continuing, index, total, seconds, seed + index, instructions, reference_lines, ideas[index],
                                      pictures, tail if continuing else None, notes, last_shot, prefix, voice_audio is not None)
                prompts.append(prompt)
                notes_entry = story_note(prompt)
                notes.append((index + 1, notes_entry[0]))
                last_shot = notes_entry[1]

                self._status(unique_id, f"Part {index + 1}/{total}: encoding the prompt")
                length = nearest_grid(prefix + seconds * FPS)
                positive, latent = self._conditioning(clip, vae, audio_vae, prompt, width, height, length, pictures, first_frame if index == 0 else None,
                                                      tail if continuing else None, (tail_wave, sample_rate) if continuing else None, mode, voice_audio, continuing)
                negative = positive if opts["cfg"] == 1.0 else clip.encode_from_tokens_scheduled(clip.tokenize(""))

                self._status(unique_id, f"Part {index + 1}/{total}: sampling")
                self._release()
                latent_out = sampler.PROMPTFORGE_Sampler().sample(model, positive, negative, latent, seed + index, opts["steps"], opts["cfg"], opts["sampler_name"], opts["scheduler"], 1.0, 1)[0]
                del positive, negative, latent

                self._status(unique_id, f"Part {index + 1}/{total}: decoding")
                self._release()
                images = minimax_decode.PROMPTFORGE_MiniMaxDecode().decode(latent_out, vae, 0, 0, "cpu", True)[0]
                audio = nodes_audio.vae_decode_audio(audio_vae, {"samples": latent_out["samples"].unbind()[1]})
                del latent_out
                self._release()

                sample_rate = int(audio["sample_rate"])
                wave = audio["waveform"][0].float().cpu()
                images = images[prefix:]
                wave = wave[:, round(prefix / FPS * sample_rate):round(prefix / FPS * sample_rate) + round(images.shape[0] / FPS * sample_rate)]
                preview = self._save_part(images, wave, sample_rate, filename_prefix, opts, index, scratch, parts_video, opts["keep_parts"])
                parts_audio.append(wave)
                self._preview(unique_id, preview)

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
        """Everything ComfyUI holds goes, so the next stage (language model, sampler, decoder) has the whole GPU."""
        model_management.unload_all_models()
        gc.collect()
        model_management.soft_empty_cache()

    @staticmethod
    def _preview(unique_id, preview):
        if unique_id is not None:
            PromptServer.instance.send_sync("executed", {"node": str(unique_id), "display_node": str(unique_id), "output": {"promptforge_video": [preview]},
                                                         "prompt_id": next((item[1] for item in PromptServer.instance.prompt_queue.currently_running.values()), None)}, PromptServer.instance.client_id)

    def _prompt(self, enhancer, opts, mode, continuing, index, total, seconds, seed, instructions, reference_lines, idea, pictures, tail, notes, last_shot, prefix, has_voice):
        package = _package()
        refs = list(pictures) + [None] * (3 - len(pictures))
        have_refs = bool(pictures) or (continuing and mode == "reference video")
        name = "MiniMax H3 Ref2VA" if have_refs else "MiniMax H3 T2VA"
        preset = ("NSFW › " if opts["preset_style"] == "NSFW" else "") + name
        lines = [f"Whole video: {total} parts of {seconds} seconds. Brief: {instructions.strip()}"]
        lines += reference_lines
        if idea and not continuing:
            lines.append(f"What this first part must show: {idea}")
        brief = "\n".join(lines)
        total_seconds = seconds + prefix / FPS
        continuation = ""
        if continuing:
            continuation = continuation_brief(index, total, seconds, instructions, notes[-4:], last_shot, mode, prefix / FPS, opts["transition"], idea)
        self._release()
        answer, _ = package.PROMPTFORGE_PromptEnhancer().enhance(
            enhancer, brief, "inputs" if (pictures or tail is not None) else "text_only", preset, seed, 2048, 0.8, 0.95, 40, 1, 1.1, "",
            video_seconds=max(1, round(total_seconds)), image=refs[0], image2=refs[1], image3=refs[2],
            video=tail if (continuing and mode == "reference video") else None, frame_count=8,
            negative_prompt=opts["negative_prompt"], auto_retry=True, dialogue=opts["dialogue"], dialogue_language=opts["dialogue_language"],
            dialogue_language_other=opts["dialogue_language_other"], continuation=continuation,
            context_video=tail if (continuing and mode == "anchor tail") else None)
        return answer

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
