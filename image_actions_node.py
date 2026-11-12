"""PromptForge Action node: a pose or act plus place, camera, light and outfit for still-image prompts."""
from . import image_actions


class PROMPTFORGE_ImageAction:
    CATEGORY = "🔮 PromptForge/Prompts"
    RETURN_TYPES = ("PROMPTFORGE_ACTION", "STRING")
    RETURN_NAMES = ("ACTION", "BRIEF")
    FUNCTION = "build"
    DESCRIPTION = ("A pose, place and camera as a binding brief for the prompt writer, for any image model. "
                   "Works from an input image or from words, for one person or two.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "category": (list(image_actions.CATEGORIES), {"default": image_actions.CATEGORIES[0]}),
                "action": ([f"{group} · {label}" for label, (_, group, _) in image_actions.ACTIONS.items()],
                           {"default": f"{next(iter(image_actions.ACTIONS.values()))[1]} · {next(iter(image_actions.ACTIONS))}"}),
                "source": (list(image_actions.SUBJECT_SOURCES), {"default": "input image",
                                                                 "tooltip": "input image: keep the person from the connected image. text description: the writer invents the person from the text below."}),
                "place": (list(image_actions.PLACES), {"default": "auto"}),
                "camera": (list(image_actions.CAMERAS), {"default": "auto"}),
                "lighting": (list(image_actions.LIGHTING), {"default": "auto"}),
                "outfit": (list(image_actions.OUTFITS), {"default": "as in the image"}),
                "style": (list(image_actions.STYLES), {"default": "auto"}),
            },
            "optional": {
                "subject_1": ("STRING", {"default": "", "multiline": True, "placeholder": "Who the first person is. Empty keeps the person from the input image.", "tooltip": "Identity text for the first person. With an image source this adds to the image; without one it is all the writer knows."}),
                "subject_2": ("STRING", {"default": "", "multiline": True, "placeholder": "Who the second person is. Empty = the other person in the image, or an invented man.", "tooltip": "Identity text for the second person in a couple act."}),
                "notes": ("STRING", {"default": "", "multiline": True, "placeholder": "Anything else: mood, props, weather, exact framing.", "tooltip": "Extra binding detail for this shot."}),
                "overrides": ("STRING", {"default": "", "multiline": True, "placeholder": "Hard overrides, for example: red dress, no glasses.", "tooltip": "Passed last and wins over the rest of the brief."}),
                "instructions": ("STRING", {"default": "", "multiline": True, "placeholder": "Extra instructions for this shot: mood, prop, exact framing. Replaces the enhancer prompt while the action is on.", "tooltip": "Custom instructions appended to the action brief."}),
            },
        }

    def build(self, category, action, source, place, camera, lighting, outfit, style,
              subject_1="", subject_2="", notes="", overrides="", instructions=""):
        action = action.split(" · ", 1)[-1]
        data = {"action": action, "source": source, "place": place, "camera": camera, "lighting": lighting,
                "outfit": outfit, "style": style, "subject_1": subject_1,
                "subject_2": subject_2, "notes": notes, "overrides": overrides, "instructions": instructions}
        text = image_actions.brief(action, (subject_1, subject_2), camera, place, lighting, outfit, style, notes, overrides, source, instructions)
        return (data, text)
