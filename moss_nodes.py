"""PromptForge wrappers for the core nodes the image workflows need."""
import json

import nodes as comfy_nodes
import node_helpers
from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple
from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels
from comfy_extras.nodes_moss_image import TextEncodeMOSSImageEdit


class PROMPTFORGE_TextEncode:
    """Positive and negative text in one node, so a workflow needs one encode instead of two."""
    CATEGORY = "🔮 PromptForge/Prompts"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("POSITIVE", "NEGATIVE")
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"default": "", "multiline": True, "tooltip": "The prompt."}),
                "negative": ("STRING", {"default": "worst quality, bad anatomy, watermark, jpeg artifact", "multiline": True, "tooltip": "What the image should avoid."}),
            }
        }

    def encode(self, clip, text, negative):
        encoder = comfy_nodes.CLIPTextEncode()
        return encoder.encode(clip, text)[0], encoder.encode(clip, negative)[0]


class PROMPTFORGE_MossEditEncode:
    """MOSS edit conditioning for both CFG passes: the prompt and the negative share the reference images."""
    CATEGORY = "🔮 PromptForge/Prompts"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("POSITIVE", "NEGATIVE")
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "The edit instruction: what should change."}),
                "negative": ("STRING", {"default": "nsfw, worst quality, bad anatomy, watermark, jpeg artifact", "multiline": True, "tooltip": "What the image should avoid. The reference images are attached to it too."}),
                "edit_format": (["ai-toolkit", "ai-toolkit (kv_cache off)", "Moss-Image-Edit"], {"default": "Moss-Image-Edit"}),
                "image": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.05, "tooltip": "How hard the prompt listens to the images."}),
                "negative_image_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.05}),
                "vae": ("VAE",),
            },
        }

    def encode(self, clip, prompt, negative, edit_format, image, image2=None, image3=None, image4=None,
               image_strength=1.0, negative_image_strength=1.0, vae=None):
        images = {f"image_{index}": value for index, value in enumerate((image, image2, image3, image4)) if value is not None}
        positive = TextEncodeMOSSImageEdit.execute(clip, prompt, edit_format, images, vae=vae, image_strength=image_strength).result[0]
        negative_out = TextEncodeMOSSImageEdit.execute(clip, negative, edit_format, images, vae=vae, image_strength=negative_image_strength).result[0]
        return positive, negative_out


class PROMPTFORGE_KreaEditEncode:
    """Grounded Krea edit positive and negative in one node; both see the source image."""
    CATEGORY = "🔮 PromptForge/Prompts"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("POSITIVE", "NEGATIVE")
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "The edit instruction: what should change."}),
                "negative": ("STRING", {"default": "deformed, blurry, bad anatomy, bad hands, low quality", "multiline": True}),
                "image": ("IMAGE",),
            },
            "optional": {
                "grounding_px": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 64}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def encode(self, clip, prompt, negative, image, grounding_px=0, system_prompt=""):
        grounded = comfy_nodes.NODE_CLASS_MAPPINGS["Krea2EditGroundedEncode"]()
        positive = grounded.encode(clip, prompt, image=image, grounding_px=grounding_px, system_prompt=system_prompt)[0]
        negative_out = comfy_nodes.NODE_CLASS_MAPPINGS["Krea2EditGroundedEncode"]().encode(
            clip, negative, image=image, grounding_px=grounding_px, system_prompt=system_prompt)[0]
        return positive, negative_out


class PROMPTFORGE_PromptLibrary:
    """A saved prompt you can switch quickly; the bar comes from the browser side."""
    CATEGORY = "🔮 PromptForge/Prompts"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"default": "", "multiline": True, "tooltip": "Use the bar to load, save or delete prompts."})}}

    def get(self, text):
        return (text,)


EDIT_LORAS = {
    "Krea 2": "Krea 2/tool/krea2_identity_edit_v2.safetensors",
    "Krea 2 (general edit)": "Krea 2/tool/krea2_general_edit_v2.safetensors",
    "MOSS Image": "MOSS Image/tool/moss_image_edit_lora_rank_64_bf16.safetensors",
}


class PROMPTFORGE_EditLora:
    """The one edit LoRA of a family, fixed by name; only strength and an off switch can change."""
    CATEGORY = "🔮 PromptForge/Prompts"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "family": (list(EDIT_LORAS), {"default": "Krea 2", "tooltip": "Picks the edit LoRA for this family. The path is fixed."}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    def apply(self, model, family, strength, enabled):
        if not enabled or strength == 0:
            return (model,)
        item = {"name": EDIT_LORAS[family], "on": True, "model": strength, "clip": None, "random": False,
                "min": 0.6, "max": 1, "group": ""}
        return (comfy_nodes.NODE_CLASS_MAPPINGS["PROMPTFORGE_LoraLoader"]().load(
            model, json.dumps({"filter": "", "items": [item]}, separators=(",", ":")), 0, True)[0],)


class PROMPTFORGE_KreaNAG:
    """Krea2 text-to-image NAG as a model patch, with an off switch."""
    CATEGORY = "🔮 PromptForge/Sampling"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "use_nag": ("BOOLEAN", {"default": False, "tooltip": "Apply normalized attention guidance from the negative prompt. Works at cfg 1 and does not change cfg."}),
            },
            "optional": {
                "nag_negative": ("CONDITIONING",),
                "ref_position": (["aligned (trained)", "beside", "bag of features"], {"default": "aligned (trained)", "advanced": True}),
                "ref_position_a": (["aligned (trained)", "beside", "bag of features"], {"default": "aligned (trained)", "advanced": True}),
                "phi": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1, "advanced": True}),
                "tau": ("FLOAT", {"default": 2.5, "min": 0.01, "max": 20.0, "step": 0.05, "advanced": True}),
                "alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True}),
            },
        }

    def patch(self, model, use_nag, nag_negative=None, phi=4.0, tau=2.5, alpha=0.25):
        if not use_nag or nag_negative is None:
            return (model,)
        return comfy_nodes.NODE_CLASS_MAPPINGS["Krea2NormalizedAttentionGuidance"]().patch(
            model, nag_negative, phi, tau, alpha, 1000.0, 0.0)


class PROMPTFORGE_KreaEditModelPatch:
    """Krea2 edit source patch; the same node can also apply NAG to the negative prompt."""
    CATEGORY = "🔮 PromptForge/Sampling"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "source_latent": ("LATENT",),
                "use_nag": ("BOOLEAN", {"default": True, "tooltip": "Apply normalized attention guidance from the negative prompt. Works at cfg 1 and does not change cfg."}),
            },
            "optional": {
                "nag_negative": ("CONDITIONING",),
                "source_latent_b": ("LATENT",),
                "ref_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "ref_boost_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "ref_boost_mask": ("MASK",),
                "fit_mode": (["fit", "crop (legacy)"], {"default": "fit"}),
                "vae": ("VAE",),
                "source_image": ("IMAGE",),
                "target_latent": ("LATENT",),
                "ref_position": (["aligned (trained)", "beside", "bag of features"], {"default": "aligned (trained)", "advanced": True}),
                "ref_position_a": (["aligned (trained)", "beside", "bag of features"], {"default": "aligned (trained)", "advanced": True}),
                "phi": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1, "advanced": True}),
                "tau": ("FLOAT", {"default": 2.5, "min": 0.01, "max": 20.0, "step": 0.05, "advanced": True}),
                "alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True}),
            },
        }

    def patch(self, model, source_latent, use_nag, nag_negative=None, source_latent_b=None, ref_boost=1.0,
              ref_boost_a=1.0, fit_mode="fit", vae=None, source_image=None, target_latent=None,
              phi=4.0, tau=2.5, alpha=0.25, ref_position="aligned (trained)", ref_position_a="aligned (trained)", ref_boost_mask=None):
        common = dict(source_latent=source_latent, source_latent_b=source_latent_b, ref_boost=ref_boost,
                      ref_boost_a=ref_boost_a, fit_mode=fit_mode, vae=vae, source_image=source_image,
                      target_latent=target_latent, ref_position=ref_position, ref_position_a=ref_position_a,
                      ref_boost_mask=ref_boost_mask)
        rest = {key: value for key, value in common.items() if key != "source_latent"}
        if use_nag and nag_negative is not None:
            # the NAG patch predates ref_position; both refs use the trained aligned position
            nag_rest = {key: value for key, value in rest.items() if key not in ("ref_position", "ref_position_a")}
            return comfy_nodes.NODE_CLASS_MAPPINGS["Krea2EditNormalizedAttentionGuidance"]().patch(
                model, nag_negative, source_latent, phi, tau, alpha, 1000.0, 0.0, **nag_rest)
        return comfy_nodes.NODE_CLASS_MAPPINGS["Krea2EditModelPatch"]().patch(model, source_latent, **rest)


class PROMPTFORGE_SkipLayerGuidance:
    CATEGORY = "🔮 PromptForge/Sampling"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "single_layers": ("STRING", {"default": "22"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    def patch(self, model, single_layers, start_percent, end_percent):
        return SkipLayerGuidanceDiTSimple.execute(model, start_percent, end_percent, "", single_layers).result


class PROMPTFORGE_ImageScale:
    CATEGORY = "🔮 PromptForge/Load"
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "WIDTH", "HEIGHT")
    FUNCTION = "scale"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01}),
                "method": (["lanczos", "area", "bilinear", "bicubic", "nearest-exact"], {"default": "lanczos"}),
                "step": ("INT", {"default": 16, "min": 1, "max": 256}),
            }
        }

    def scale(self, image, megapixels, method, step):
        out = ImageScaleToTotalPixels.execute(image, method, megapixels, step).result[0]
        return (out, int(out.shape[2]), int(out.shape[1]))


class PROMPTFORGE_VAEDecode:
    CATEGORY = "🔮 PromptForge/Load"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}

    def decode(self, samples, vae):
        return comfy_nodes.VAEDecode().decode(vae, samples)


class PROMPTFORGE_SaveImage:
    CATEGORY = "🔮 PromptForge/Load"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "filename_prefix": ("STRING", {"default": "PromptForge/image"})}}

    def save(self, images, filename_prefix="PromptForge/image"):
        return comfy_nodes.NODE_CLASS_MAPPINGS["SaveImage"]().save_images(images, filename_prefix=filename_prefix)


NODE_CLASS_MAPPINGS = {
    "PROMPTFORGE_TextEncode": PROMPTFORGE_TextEncode,
    "PROMPTFORGE_MossEditEncode": PROMPTFORGE_MossEditEncode,
    "PROMPTFORGE_KreaEditEncode": PROMPTFORGE_KreaEditEncode,
    "PROMPTFORGE_PromptLibrary": PROMPTFORGE_PromptLibrary,
    "PROMPTFORGE_EditLora": PROMPTFORGE_EditLora,
    "PROMPTFORGE_KreaNAG": PROMPTFORGE_KreaNAG,
    "PROMPTFORGE_KreaEditModelPatch": PROMPTFORGE_KreaEditModelPatch,
    "PROMPTFORGE_SkipLayerGuidance": PROMPTFORGE_SkipLayerGuidance,
    "PROMPTFORGE_ImageScale": PROMPTFORGE_ImageScale,
    "PROMPTFORGE_VAEDecode": PROMPTFORGE_VAEDecode,
    "PROMPTFORGE_SaveImage": PROMPTFORGE_SaveImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PROMPTFORGE_TextEncode": "PromptForge Text Encode",
    "PROMPTFORGE_MossEditEncode": "PromptForge MOSS Edit Encode",
    "PROMPTFORGE_KreaEditEncode": "PromptForge Krea Edit Encode",
    "PROMPTFORGE_PromptLibrary": "PromptForge Prompt Library",
    "PROMPTFORGE_EditLora": "PromptForge Edit LoRA",
    "PROMPTFORGE_KreaNAG": "PromptForge Krea NAG",
    "PROMPTFORGE_KreaEditModelPatch": "PromptForge Krea Edit Model Patch",
    "PROMPTFORGE_SkipLayerGuidance": "PromptForge Skip Layer Guidance",
    "PROMPTFORGE_ImageScale": "PromptForge Image Scale",
    "PROMPTFORGE_VAEDecode": "PromptForge VAE Decode",
    "PROMPTFORGE_SaveImage": "PromptForge Save Image",
}
