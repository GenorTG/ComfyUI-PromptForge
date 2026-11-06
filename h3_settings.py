"""The small settings nodes of the MiniMax H3 Endless Video generator.

Each one answers a single question and plugs into the generator (PromptForge H3 Endless Video):

    Story ............ what happens (general prompt, a box per part, dialogue, prompt style)
    References ....... who is in it (character pictures and their descriptions, a voice)
    Video Format ..... how big and how long (aspect ratio, megapixels, seconds per part)
    Continuity ....... how the parts join (carry the ending forward, cuts, smoothing)
    Sampling ......... how the first stage is sampled (sampler, steps, cfg)
    Second Stage ..... optional: upscale every part and sample it again
    Live Preview ..... optional: what you see while it samples
    Speed and Memory . optional: attention backend and feed-forward chunking

Every node is optional except the Story: an unconnected one means its defaults. Connecting the Second Stage node is
what turns the second stage on, so none of them has an on/off switch for a whole section.
"""
import comfy.samplers

from . import h3_upscaler, preview_decoder
from . import empty_latent

CATEGORY = "🔮 PromptForge/Endless Video"
FPS = 24

# how the parts join: what the user sees -> what the generator does
JOIN_MODES = {
    "carry the last frames forward": "anchor tail",
    "give the ending as a reference video": "reference video",
    "independent parts (no carry-over)": "none",
}
STORY_STYLES = ("continue the same shot", "let the story decide", "cut to a new shot")
CARRIED_FRAMES = ("1", "5", "22", "39", "56", "73")
SECONDS = ("5", "10")
PROMPT_STYLES = ("standard", "NSFW")
DIALOGUE_AUTO = "auto (from the prompts)"
DIALOGUE_LANGUAGES = (DIALOGUE_AUTO, "Arabic", "Chinese", "English", "French", "German", "Italian", "Japanese", "Korean", "Portuguese", "Russian", "Spanish", "other (type below)")
PREVIEW_MODES = ("live", "none", "ComfyUI's own")
ATTENTION = ("comfy kitchen attention", "pytorch attention", "as connected")

DEFAULTS = {
    "story": {"general_prompt": "", "parts": 3, "part_prompts": "", "negative_prompt": "", "dialogue": "", "dialogue_language": DIALOGUE_AUTO,
              "dialogue_language_other": "", "prompt_style": "standard", "writer_creativity": 0.8, "review_prompts": False},
    "references": {"description_1": "", "description_2": "", "description_3": ""},
    "format": {"aspect_ratio": "16:9", "orientation": "landscape", "megapixels": 0.5, "seconds_per_part": "5"},
    "continuity": {"how_parts_join": next(iter(JOIN_MODES)), "story_style": STORY_STYLES[0], "carried_frames": "22", "smooth_joins": True, "check_joins": False, "join_retries": 1},
    "sampling": {"sampler_name": "er_sde", "scheduler": "sgm_uniform", "steps": 8, "cfg": 1.0, "shift_video": 12.0, "shift_audio": 3.0},
    "second_stage": {"upscale_model": h3_upscaler.default_model(), "size_mode": h3_upscaler.MODES[0], "scale": 2.0, "megapixels": 1.0, "width": 1280, "height": 704, "align": 32,
                    "precision": "fp16", "steps": 3, "start_sigma": 0.9035, "sampler_name": "er_sde"},
    "preview": {"mode": PREVIEW_MODES[0], "frames": 4, "fps": 8, "size": 512, "decoder": preview_decoder.default_decoder()},
    "performance": {"attention": ATTENTION[0], "ffn_chunks": 3, "sparse_attention": True, "keep_percent": 10.0},
    "continue_from": {},
}


def settings(name, given):
    """The defaults of one settings node, overridden by what the connected node gave."""
    return {**DEFAULTS[name], **(given or {})}


class _Settings:
    """Base of the settings nodes: the widgets go into one dict, which is the node's only output."""
    CATEGORY = CATEGORY
    FUNCTION = "build"
    KEY = ""

    def build(self, **values):
        return (values,)


class PROMPTFORGE_H3Story(_Settings):
    KEY = "story"
    RETURN_TYPES = ("PROMPTFORGE_H3_STORY",)
    RETURN_NAMES = ("story",)
    DESCRIPTION = "What the video is about and what happens in each part. Write the general prompt, press + to add a box for every part, and describe that part in its box. Plug into the H3 Endless Video node."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "general_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "What the whole video is about, who is in it, the style. What happens in each part goes in the boxes below.",
                                              "tooltip": "Context for every part: the idea of the whole video, the look, who is in it. It is not shown in full in each part; each part shows what its own box says."}),
            },
            "optional": {
                "parts": ("INT", {"default": 3, "min": 1, "max": 200, "tooltip": "Number of parts. Press + / - on the node."}),
                "part_prompts": ("STRING", {"default": "", "multiline": True, "tooltip": "The part boxes (kept by the node)."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Things the video must not contain."}),
                "dialogue": ("STRING", {"default": "", "multiline": True, "tooltip": "What the characters talk about and how it sounds. Written fresh for every part and sized to its length. Empty = no dialogue instructions."}),
                "dialogue_language": (list(DIALOGUE_LANGUAGES), {"default": DIALOGUE_AUTO, "tooltip": "Language of the spoken lines. The list is what MiniMax H3 speaks well."}),
                "dialogue_language_other": ("STRING", {"default": "", "tooltip": "Used when the language is 'other', for example Polish. Languages outside the list work only partly."}),
                "prompt_style": (list(PROMPT_STYLES), {"default": "standard", "tooltip": "Which family of MiniMax H3 prompt presets the prompt writer uses."}),
                "writer_creativity": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "How freely the prompt writer words things. Lower sticks closer to your text."}),
                "review_prompts": ("BOOLEAN", {"default": False, "tooltip": "Pause after each part's prompt is written: a panel lets you edit it, approve it, or have it written again. Needs the ComfyUI page open."}),
            },
        }


class PROMPTFORGE_H3References(_Settings):
    KEY = "references"
    RETURN_TYPES = ("PROMPTFORGE_H3_REFERENCES",)
    RETURN_NAMES = ("references",)
    DESCRIPTION = "Who is in the video: up to three character pictures with a description of each, an optional exact first frame and an optional voice. Used in every part so the characters stay the same."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "first_frame": ("IMAGE", {"tooltip": "The exact first frame of the video (the first part only)."}),
                "image_1": ("IMAGE", {"tooltip": "A character, place or object that must look the same in every part."}),
                "description_1": ("STRING", {"default": "", "multiline": True, "tooltip": "Who or what image_1 is. Can come from the DESCRIPTION output of a Load Image node."}),
                "image_2": ("IMAGE",),
                "description_2": ("STRING", {"default": "", "multiline": True}),
                "image_3": ("IMAGE",),
                "description_3": ("STRING", {"default": "", "multiline": True}),
                "voice": ("AUDIO", {"tooltip": "A voice sample: the main speaker keeps its sound in every part."}),
            },
        }


class PROMPTFORGE_H3Format(_Settings):
    KEY = "format"
    RETURN_TYPES = ("PROMPTFORGE_H3_FORMAT",)
    RETURN_NAMES = ("format",)
    DESCRIPTION = "How big and how long: aspect ratio, size in megapixels and the length of each part. The video is 24 frames per second."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (list(empty_latent.RATIOS), {"default": "16:9"}),
                "orientation": (["landscape", "portrait"], {"default": "landscape"}),
                "megapixels": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05, "tooltip": "Size of every part when it is sampled. 0.5 is a good size for a first stage; with a Second Stage node connected the final size is larger."}),
                "seconds_per_part": (list(SECONDS), {"default": "5", "tooltip": "New seconds of video in each part. A part box in the Story node can start with (8s) to give that part its own length."}),
            },
        }


class PROMPTFORGE_H3Continuity(_Settings):
    KEY = "continuity"
    RETURN_TYPES = ("PROMPTFORGE_H3_CONTINUITY",)
    RETURN_NAMES = ("continuity",)
    DESCRIPTION = "How each part continues from the previous one: what is carried forward, whether the story may cut, and how the joins are smoothed."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "how_parts_join": (list(JOIN_MODES), {"default": next(iter(JOIN_MODES)), "tooltip": "carry the last frames forward: the end of the previous part (frames and sound) is placed at the start of the next one, so it carries on exactly where it stopped. give the ending as a reference video: MiniMax's own way to continue a video. independent parts: every part starts from the references only."}),
                "story_style": (list(STORY_STYLES), {"default": STORY_STYLES[0], "tooltip": "Whether the story stays in one shot, may cut when it suits, or cuts to a new shot at the start of every part. Cutting needs no carried frames."}),
                "carried_frames": (list(CARRIED_FRAMES), {"default": "22", "tooltip": "How many frames of the previous part are carried forward (24 per second). More follows the motion better and costs a little time. 1 carries only the last frame."}),
                "smooth_joins": ("BOOLEAN", {"default": True, "tooltip": "Evens out the short flash of brightness that can appear where two parts meet. A real scene change is left alone."}),
                "check_joins": ("BOOLEAN", {"default": False, "tooltip": "When a part is meant to continue the same shot but its first picture jumps, generate it again with another seed. Every retry costs a whole part of time, so it is off by default."}),
                "join_retries": ("INT", {"default": 1, "min": 0, "max": 5, "tooltip": "How many times a jumping part is generated again (used with check_joins)."}),
            },
        }


class PROMPTFORGE_H3Sampling(_Settings):
    KEY = "sampling"
    RETURN_TYPES = ("PROMPTFORGE_H3_SAMPLING",)
    RETURN_NAMES = ("sampling",)
    DESCRIPTION = "How every part is sampled in the first stage: sampler, schedule, steps and cfg. A turbo LoRA wants 6 to 8 steps and cfg 1."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "er_sde", "tooltip": "er_sde with sgm_uniform or euler with simple are the reliable pairs for 8-step turbo H3 models."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "sgm_uniform"}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 200, "tooltip": "Steps of the first stage."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "1 for turbo and distilled models."}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01, "tooltip": "Noise schedule shift of the video. 12 for MiniMax H3 turbo models; it is applied to the model, no shift node is needed."}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01, "tooltip": "Noise schedule shift of the audio. 3 for MiniMax H3."}),
            },
        }


class PROMPTFORGE_H3SecondStage(_Settings):
    KEY = "second_stage"
    RETURN_TYPES = ("PROMPTFORGE_H3_SECOND_STAGE",)
    RETURN_NAMES = ("second_stage",)
    DESCRIPTION = "Connect this to make every part two-stage: the first stage is sampled small, its latent is upscaled with the MiniMax H3 latent upscaler, and a few more steps sample it again at the larger size. 6 first-stage steps plus 3 here is 9 steps per part."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": (h3_upscaler.model_names(), {"default": h3_upscaler.default_model(), "tooltip": "The H3 latent upscaler, from models/latent_upscale_models."}),
                "size_mode": (list(h3_upscaler.MODES), {"default": h3_upscaler.MODES[0], "tooltip": "How the size of the second stage is chosen: a multiplier of the first stage, a number of megapixels, or exact pixel dimensions. Only the matching size setting is shown."}),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05, "tooltip": "Multiplier of the first stage's width and height."}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "width": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 704, "min": 64, "max": 8192, "step": 8}),
                "align": ("INT", {"default": 32, "min": 1, "max": 512, "tooltip": "Rounds the size to a multiple of this many pixels. 32 is recommended."}),
                "precision": (list(h3_upscaler.PRECISIONS), {"default": "fp16", "tooltip": "Number format of the upscaler. fp16 is fine on RTX cards."}),
                "steps": ("INT", {"default": 3, "min": 1, "max": 100, "tooltip": "Steps of the second stage."}),
                "start_sigma": ("FLOAT", {"default": 0.9035, "min": 0.05, "max": 1.0, "step": 0.0005, "tooltip": "How much noise the upscaled latent gets before the second stage; the rest of the steps fall in a straight line to 0. 3 steps from 0.9035 are 0.9035, 0.6316, 0.3158, 0. Lower keeps more of the first stage."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "er_sde", "tooltip": "Sampler of the second stage. er_sde adds a little fresh noise at every step, which cleans up the upscaled latent; euler does not."}),
            },
        }


class PROMPTFORGE_H3Preview(_Settings):
    KEY = "preview"
    RETURN_TYPES = ("PROMPTFORGE_H3_PREVIEW",)
    RETURN_NAMES = ("preview",)
    DESCRIPTION = "What you see while a part is sampled: a short loop of the latent frames, drawn in the generator node. Without this node the preview is a 4-frame loop."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (list(PREVIEW_MODES), {"default": PREVIEW_MODES[0], "tooltip": "live: PromptForge's own preview in the generator node. none: no preview. ComfyUI's own: whatever ComfyUI is set to show."}),
                "frames": ("INT", {"default": 4, "min": 1, "max": 99999, "tooltip": "How many latent frames the loop shows. 99999 shows every frame."}),
                "fps": ("INT", {"default": 8, "min": 1, "max": 60, "tooltip": "Playback speed of the loop."}),
                "size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 16, "tooltip": "Longest side of the preview in pixels."}),
                "decoder": (preview_decoder.decoder_names(), {"default": preview_decoder.default_decoder(), "tooltip": "What draws the frames: latent2rgb is instant and needs no file; a tiny decoder from models/vae_approx (taeh3 for MiniMax H3) gives real colours and detail."}),
            },
        }


class PROMPTFORGE_H3Performance(_Settings):
    KEY = "performance"
    RETURN_TYPES = ("PROMPTFORGE_H3_PERFORMANCE",)
    RETURN_NAMES = ("performance",)
    DESCRIPTION = "Speed and memory settings of the diffusion model. The defaults (INT8 attention, feed-forward in 3 chunks) are the fast ones."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "attention": (list(ATTENTION), {"default": ATTENTION[0], "tooltip": "comfy kitchen attention is INT8 attention: about 40% faster for MiniMax H3 with a tiny precision cost (it falls back to pytorch where it is not available). as connected keeps what the model already has."}),
                "ffn_chunks": ("INT", {"default": 3, "min": 1, "max": 64, "tooltip": "Splits the model's feed-forward layers into this many chunks: lower peak VRAM, same result. 1 = off."}),
                "sparse_attention": ("BOOLEAN", {"default": True, "tooltip": "Video sparse attention (VSA): faster, and the FastH3-based models (10Eros Max betas) are trained for it - without it they flicker. Turn off for other H3 checkpoints. The first blocks and short sequences stay dense."}),
                "keep_percent": ("FLOAT", {"default": 10.0, "min": 0.5, "max": 95.0, "step": 0.5, "tooltip": "Share of the video each part of the picture attends to. FastH3 models are trained at 10 (90% sparse)."}),
            },
        }


class PROMPTFORGE_H3ContinueFrom(_Settings):
    KEY = "continue_from"
    RETURN_TYPES = ("PROMPTFORGE_H3_CONTINUE",)
    RETURN_NAMES = ("continue_from",)
    DESCRIPTION = "Extend a video you already have: the first part continues from its ending (frames and sound), and the result is the old video followed by the new parts."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"video": ("IMAGE", {"tooltip": "The video to extend, as frames (from a Load Video node, or a previous Endless Video result)."})},
            "optional": {"audio": ("AUDIO", {"tooltip": "Its sound. Without it the old part is silent."})},
        }


NODES = {
    "PROMPTFORGE_H3Story": ("PromptForge H3 Story", PROMPTFORGE_H3Story),
    "PROMPTFORGE_H3References": ("PromptForge H3 References", PROMPTFORGE_H3References),
    "PROMPTFORGE_H3ContinueFrom": ("PromptForge H3 Continue Existing Video", PROMPTFORGE_H3ContinueFrom),
    "PROMPTFORGE_H3Format": ("PromptForge H3 Video Format", PROMPTFORGE_H3Format),
    "PROMPTFORGE_H3Continuity": ("PromptForge H3 Continuity", PROMPTFORGE_H3Continuity),
    "PROMPTFORGE_H3Sampling": ("PromptForge H3 Sampling", PROMPTFORGE_H3Sampling),
    "PROMPTFORGE_H3SecondStage": ("PromptForge H3 Second Stage (upscale)", PROMPTFORGE_H3SecondStage),
    "PROMPTFORGE_H3Preview": ("PromptForge H3 Live Preview", PROMPTFORGE_H3Preview),
    "PROMPTFORGE_H3Performance": ("PromptForge H3 Speed and Memory", PROMPTFORGE_H3Performance),
}
NODE_CLASS_MAPPINGS = {key: node for key, (_, node) in NODES.items()}
NODE_DISPLAY_NAME_MAPPINGS = {key: name for key, (name, _) in NODES.items()}
