"""V2 nodes are registered separately from the frozen legacy keys."""
from . import shared, media, refs, masking, performance, image, krea, prompting, h3

MODULES = (shared, media, refs, masking, performance, image, krea, prompting, h3)
NODE_CLASS_MAPPINGS = {key: value for module in MODULES for key, value in module.NODE_CLASS_MAPPINGS.items()}
NODE_DISPLAY_NAME_MAPPINGS = {key: value for module in MODULES for key, value in module.NODE_DISPLAY_NAME_MAPPINGS.items()}
