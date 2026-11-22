"""The action store splits shipped from user presets and validates writes."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import aiohttp.web

server = types.ModuleType("server")
server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=aiohttp.web.RouteTableDef()))
sys.modules.setdefault("server", server)

PACK = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("action_store_test", PACK / "v2" / "action_store.py")
store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(store)

SHIPPED = {
    "Portrait - neutral": {"label": "Portrait - neutral", "category": "SFW - portrait and expression", "people": 1, "brief": "{A} faces the camera."},
    "Position - cowgirl": {"label": "Position - cowgirl", "category": "Positions", "aspect": "9:16", "penetration": "auto",
                           "roles": [{"gender": "female", "role": "the woman on top"}, {"gender": "male", "role": "the man lies back"}],
                           "brief": "{A} rides {B}."},
}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.previous = store.DIR
        store.DIR = self.directory
        for family in store.FAMILIES:
            entries = dict(SHIPPED) if family == "h3" else {"Portrait - neutral": SHIPPED["Portrait - neutral"]}
            (self.directory / f"{family}_actions.json").write_text(json.dumps(entries))

    def tearDown(self):
        store.DIR = self.previous
        self.temporary.cleanup()

    def test_shipped_entries_load_and_extra_is_absent(self):
        self.assertEqual(store.labels("image"), ["Portrait - neutral"])
        self.assertEqual(store.get("h3", "Position - cowgirl")["aspect"], "9:16")
        self.assertFalse((self.directory / "image_actions_extra.json").exists())

    def test_save_writes_the_extra_file_and_shadows(self):
        store.save("image", "Portrait - neutral", {"category": "SFW - portrait and expression", "people": 1, "brief": "edited"})
        self.assertEqual(store.get("image", "Portrait - neutral")["brief"], "edited")
        self.assertEqual(json.loads((self.directory / "image_actions_extra.json").read_text())["Portrait - neutral"]["brief"], "edited")
        self.assertFalse(store.nsfw("image", "Portrait - neutral"))

    def test_delete_removes_an_override_and_hides_a_shipped_preset(self):
        store.save("image", "Portrait - neutral", {"category": "SFW - portrait and expression", "people": 1, "brief": "edited"})
        self.assertTrue(store.delete("image", "Portrait - neutral"))
        self.assertEqual(store.get("image", "Portrait - neutral")["brief"], "{A} faces the camera.")
        self.assertTrue(store.delete("image", "Portrait - neutral"))
        self.assertEqual(store.labels("image"), [])
        self.assertFalse(store.delete("image", "Portrait - neutral"))

    def test_genders_build_roles_and_keep_written_texts(self):
        saved = store.save("h3", "Position - cowgirl", {"category": "Positions", "people": 2, "genders": "female,male", "brief": "changed"})
        self.assertEqual([role["role"] for role in saved["roles"]], ["the woman on top", "the man lies back"])
        fresh = store.save("h3", "Oral - test", {"category": "Oral", "people": 2, "genders": "female,male", "brief": "x"})
        self.assertEqual([role["role"] for role in fresh["roles"]], ["the woman", "the man"])
        pair = store.save("h3", "Oral - pair", {"category": "Oral", "roles": [["female", "she kneels"]], "brief": "x"})
        self.assertEqual(pair["roles"], [{"gender": "female", "role": "she kneels"}])

    def test_h3_is_always_nsfw_and_extra_only_image_presets_are_too(self):
        store.save("image", "Portrait - extra", {"category": "SFW - portrait and expression", "people": 1, "brief": "x"})
        self.assertTrue(store.nsfw("image", "Portrait - extra"))
        self.assertTrue(store.nsfw("h3", "Position - cowgirl"))

    def test_validation(self):
        for payload in ({}, {"category": "Nope", "brief": "x"}, {"category": "SFW - solo pose", "brief": ""}):
            with self.assertRaises(ValueError):
                store.save("image", "Bad", payload)
        with self.assertRaises(ValueError):
            store.save("h3", "Bad", {"category": "Positions", "brief": "x", "people": 2, "genders": "female"})
        with self.assertRaises(ValueError):
            store.save("h3", "Bad", {"category": "Positions", "brief": "x", "people": 1, "aspect": "4:3"})
        with self.assertRaises(ValueError):
            store.save("h3", "Bad", {"category": "Positions", "brief": "x", "roles": ["female"], "penetration": "mouth"})
        with self.assertRaises(ValueError):
            store.save("nope", "Bad", {"category": "Positions", "brief": "x"})

    def test_broken_shipped_entry_is_repaired(self):
        (self.directory / "image_actions.json").write_text(json.dumps({"Broken": {"brief": "x", "people": "many"}}))
        entry = store.get("image", "Broken")
        self.assertEqual((entry["category"], entry["people"]), ("SFW - portrait and expression", 1))


if __name__ == "__main__":
    unittest.main()
