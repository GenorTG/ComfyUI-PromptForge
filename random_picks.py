"""Random choice for the action pickers: one concrete value per run, announced to the node UI."""
import random

RANDOM = "random"


def pick(values):
    return random.choice(list(values))


def announce(node_id, picks):
    """Show what was rolled on the node; `picks` is [(field, value), ...]."""
    if node_id is None:
        return
    from server import PromptServer
    PromptServer.instance.send_sync("promptforge.random", {"node": str(node_id), "picks": picks}, PromptServer.instance.client_id)
