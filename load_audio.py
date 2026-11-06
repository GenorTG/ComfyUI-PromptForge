"""PromptForge Load Audio: a sound from the input folder with a description of it, and nothing at all when no file is chosen."""
import os

import folder_paths
from comfy_extras import nodes_audio


class PROMPTFORGE_LoadAudio:
    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.filter_files_content_types(os.listdir(folder_paths.get_input_directory()), ["audio", "video"])
        return {
            "required": {
                "audio": ([""] + sorted(files), {"tooltip": "A sound or video file from the input folder (put the files there). Empty = no sound: the output stays empty, so the node can stay in a workflow unused."}),
                "description": ("STRING", {"default": "", "multiline": True, "placeholder": "What this sound is, for example the voice of the woman (S1). It goes to the prompt writer.", "tooltip": "Output as DESCRIPTION."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("AUDIO", "DESCRIPTION")
    FUNCTION = "load"
    CATEGORY = "🔮 PromptForge/Load"
    DESCRIPTION = "Load a sound from the input folder together with a description of it. With no file chosen the audio output is empty."

    @classmethod
    def IS_CHANGED(cls, audio, description):
        return f"{os.path.getmtime(folder_paths.get_annotated_filepath(audio)) if audio else 0}|{description}"

    @classmethod
    def VALIDATE_INPUTS(cls, audio):
        return not audio or folder_paths.exists_annotated_filepath(audio) or f"Invalid audio file: {audio}"

    def load(self, audio, description):
        if not audio:
            return (None, description.strip())
        waveform, sample_rate = nodes_audio.load(folder_paths.get_annotated_filepath(audio))
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}, description.strip())


NODE_CLASS_MAPPINGS = {"PROMPTFORGE_LoadAudio": PROMPTFORGE_LoadAudio}
NODE_DISPLAY_NAME_MAPPINGS = {"PROMPTFORGE_LoadAudio": "PromptForge Load Audio"}
