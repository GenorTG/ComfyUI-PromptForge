"""Freeze of the 1.30 node set.

Every node registered here is legacy: same behaviour, but the all-in-one names and configs are
replaced by per-family nodes. Workflow compatibility is kept (class keys never change), so this
module only decorates the display names and descriptions, at import time. See LEGACY.md.
"""

LEGACY = {
    "PROMPTFORGE_Model": "the per-family model node",
    "PROMPTFORGE_PromptEnhancer": "the per-family enhancer node",
    "PROMPTFORGE_ConfigFinder": "the per-family settings report",
    "PROMPTFORGE_ImageAction": "the per-family image action node",
    "PROMPTFORGE_TextEncode": "the per-family encode node",
    "PROMPTFORGE_MossEditEncode": "PromptForge MOSS Image Encode (new)",
    "PROMPTFORGE_KreaEditEncode": "PromptForge Krea 2 Edit Encode (new)",
    "PROMPTFORGE_PromptLibrary": "PromptForge Prompt Library (new key)",
    "PROMPTFORGE_EditLora": "the per-family edit LoRA node",
    "PROMPTFORGE_KreaNAG": "the Krea 2 sampler",
    "PROMPTFORGE_KreaEditModelPatch": "PromptForge Krea 2 Edit (new)",
    "PROMPTFORGE_SkipLayerGuidance": "the per-family sampler",
    "PROMPTFORGE_Loader": "the per-family loader",
    "PROMPTFORGE_LoraLoader": "the per-family LoRA loader",
    "PROMPTFORGE_EmptyLatent": "the per-family latent node",
    "PROMPTFORGE_Sampler": "the per-family sampler",
    "PROMPTFORGE_VAEDecode": "the per-family decoder",
    "PROMPTFORGE_ImageScale": "PromptForge Image Scale (new key)",
    "PROMPTFORGE_LoadImage": "PromptForge Load Image (new key)",
    "PROMPTFORGE_LoadVideo": "PromptForge Load Video (new key)",
    "PROMPTFORGE_LoadAudio": "PromptForge Load Audio (new key)",
    "PROMPTFORGE_SaveImage": "PromptForge Save Image (new key)",
    "PROMPTFORGE_SaveVideo": "PromptForge MiniMax H3 Save Video (new)",
    "PROMPTFORGE_FrameInterpolate": "PromptForge Frame Interpolate (new key)",
    "PROMPTFORGE_PreviewOverride": "PromptForge Preview Override (new key)",
    "PROMPTFORGE_QuantizeModel": "PromptForge Quantize Model (new key)",
    "PROMPTFORGE_QuantizeTextEncoder": "PromptForge Quantize Text Encoder (new key)",
    "PROMPTFORGE_MiniMaxDecode": "PromptForge MiniMax H3 Decode (new)",
    "PROMPTFORGE_H3Generate": "PromptForge MiniMax H3 Generate (new)",
    "PROMPTFORGE_H3Modes": "PromptForge MiniMax H3 Generate (new)",
    "PROMPTFORGE_H3Idea": "PromptForge MiniMax H3 Prompt (new)",
    "PROMPTFORGE_H3Action": "PromptForge MiniMax H3 Action (new)",
    "PROMPTFORGE_H3Writer": "PromptForge MiniMax H3 Prompt (new)",
    "PROMPTFORGE_H3Parts": "PromptForge MiniMax H3 Parts (new)",
    "PROMPTFORGE_H3References": "PromptForge MiniMax H3 References (new)",
    "PROMPTFORGE_H3ContinueFrom": "PromptForge MiniMax H3 Continue (new)",
    "PROMPTFORGE_H3Format": "PromptForge MiniMax H3 Format (new)",
    "PROMPTFORGE_H3Continuity": "PromptForge MiniMax H3 Joins (new)",
    "PROMPTFORGE_H3Sampling": "PromptForge MiniMax H3 Sampler (new)",
    "PROMPTFORGE_H3SecondStage": "PromptForge MiniMax H3 Upscale (new)",
    "PROMPTFORGE_H3Preview": "PromptForge MiniMax H3 Preview (new)",
    "PROMPTFORGE_H3Performance": "PromptForge MiniMax H3 Speed (new)",
}


def mark(class_mappings, display_names):
    """Prefix every legacy node, so the picker shows the old set apart from the new one."""
    for key, replacement in LEGACY.items():
        name = display_names.get(key)
        if name:
            display_names[key] = name.replace("PromptForge ", "PromptForge Legacy · ")
        node = class_mappings.get(key)
        if node is not None:
            old = getattr(node, "DESCRIPTION", "") or ""
            node.DESCRIPTION = f"Legacy node, replaced by {replacement}." + (f" {old}" if old else "")
    return class_mappings, display_names
