import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("pf_contracts", Path(__file__).parents[1] / "v2" / "contracts.py")
contracts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contracts)


class PipeTests(unittest.TestCase):
    def test_branches_do_not_share_configuration(self):
        base = contracts.evolve(contracts.new_pipe("krea2"), "format", {"size": [1024, 1024]})
        branch = contracts.evolve(base, "sampling", {"steps": 8})
        branch["settings"]["format"]["size"][0] = 512
        self.assertEqual(base["settings"]["format"]["size"], [1024, 1024])

    def test_artifacts_are_shared_without_copying_or_invalidating_other_branch(self):
        model, latent = object(), object()
        base = contracts.evolve(contracts.new_pipe("minimax_h3"), artifacts={"model": model, "latent": latent})
        branch = contracts.evolve(base, invalidate=("latent",))
        self.assertIs(contracts.require(branch, "model"), model)
        self.assertIs(contracts.require(base, "latent"), latent)
        with self.assertRaisesRegex(ValueError, "latent"):
            contracts.require(branch, "latent")

    def test_metadata_excludes_live_artifacts(self):
        pipe = contracts.evolve(contracts.new_pipe("moss_image"), artifacts={"model": object()})
        self.assertNotIn("artifacts", contracts.metadata(pipe))

    def test_legacy_pipe_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "legacy"):
            contracts.validate({"family": "krea2"})


if __name__ == "__main__":
    unittest.main()
