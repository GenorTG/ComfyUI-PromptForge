"""PromptForge Load Video: a video from the input folder as 24 fps frames, its sound and a description; nothing at all when no file is chosen or the node is switched off."""
import os

import torch

import folder_paths
from comfy_api.latest import InputImpl

FPS = 24


class PROMPTFORGE_LoadVideo:
    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.filter_files_content_types(os.listdir(folder_paths.get_input_directory()), ["video"])
        return {
            "required": {
                "video": ([""] + sorted(files), {"tooltip": "A video from the input folder (put the files there). Empty = no video."}),
                "description": ("STRING", {"default": "", "multiline": True, "placeholder": "What this video shows. It goes to the prompt writer.", "tooltip": "Output as DESCRIPTION."}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Off = the node outputs nothing, as if it were bypassed, so it can stay in a workflow and be switched on when needed."}),
                "max_seconds": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 3600.0, "step": 0.5, "tooltip": "Only the first seconds are loaded. 0 = the whole video. Reference videos work best at 2 to 15 seconds."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("IMAGES", "AUDIO", "DESCRIPTION")
    FUNCTION = "load"
    CATEGORY = "\U0001f52e PromptForge/Load"
    DESCRIPTION = "Load a video from the input folder as 24 fps frames with its sound and a description. With no file chosen, or switched off, the outputs are empty."

    @classmethod
    def IS_CHANGED(cls, video, description, enabled, max_seconds):
        return f"{os.path.getmtime(folder_paths.get_annotated_filepath(video)) if video and enabled else 0}|{description}|{enabled}|{max_seconds}"

    @classmethod
    def VALIDATE_INPUTS(cls, video, enabled=True):
        return not video or not enabled or folder_paths.exists_annotated_filepath(video) or f"Invalid video file: {video}"

    def load(self, video, description, enabled, max_seconds):
        if not video or not enabled:
            return (None, None, "")
        components = InputImpl.VideoFromFile(folder_paths.get_annotated_filepath(video)).get_components()
        frames, rate = components.images, float(components.frame_rate)
        count = round(frames.shape[0] * FPS / rate)
        if max_seconds:
            count = min(count, round(max_seconds * FPS))
        frames = frames[torch.arange(count).mul(rate / FPS).floor().long().clamp(max=frames.shape[0] - 1)]
        audio = components.audio
        if audio is not None and max_seconds:
            audio = {"waveform": audio["waveform"][..., :round(max_seconds * audio["sample_rate"])], "sample_rate": audio["sample_rate"]}
        return (frames, audio, description.strip())


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_LoadVideo": PROMPTFORGE_LoadVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_LoadVideo": "PromptForge Load Video"}
