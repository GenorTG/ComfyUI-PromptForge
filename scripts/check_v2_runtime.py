#!/usr/bin/env python3
"""CPU smoke using real Comfy node initialization and an isolated temporary base.

Run with ComfyUI's Python: python scripts/check_v2_runtime.py [--comfy-root PATH]
No server socket, model downloads, or user model directories are used. Tiny random
models and dispatch spies check plumbing, not checkpoint quality.
"""
import argparse
import asyncio
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

PACK = Path(__file__).resolve().parents[1]


def bootstrap(comfy_root=None):
    root = Path(comfy_root or os.environ.get("COMFYUI_ROOT", PACK.parents[1])).resolve()
    if not (root / "nodes.py").is_file():
        raise RuntimeError(f"ComfyUI nodes.py not found in {root}; use --comfy-root.")
    temporary = tempfile.TemporaryDirectory(prefix="promptforge-v2-runtime-")
    base = Path(temporary.name)
    os.environ["NUMBA_CACHE_DIR"] = str(base / "numba-cache")
    for name in ("input", "output", "temp", "user", "web", "models", "custom_nodes"):
        (base / name).mkdir()
    (base / "custom_nodes" / PACK.name).symlink_to(PACK, target_is_directory=True)
    sys.path.insert(0, str(root))
    # Must precede the first comfy.cli_args/model_management import.
    sys.argv = [str(PACK / "scripts/check_v2_runtime.py"), "--cpu", "--offline", "--disable-assets",
                "--base-directory", str(base), "--user-directory", str(base / "user"),
                "--front-end-root", str(base / "web"), "--disable-all-custom-nodes",
                "--whitelist-custom-nodes", PACK.name]
    import comfy.options
    comfy.options.enable_args_parsing()
    from comfy.cli_args import args
    if not args.cpu:
        raise RuntimeError("ComfyUI was imported before CPU arguments; run this check in a fresh process.")
    import folder_paths
    for key, (_, extensions) in list(folder_paths.folder_names_and_paths.items()):
        directory = base / ("custom_nodes" if key == "custom_nodes" else f"models/{key}")
        directory.mkdir(parents=True, exist_ok=True)
        folder_paths.folder_names_and_paths[key] = ([str(directory)], extensions)
    folder_paths.set_input_directory(str(base / "input"))
    folder_paths.set_output_directory(str(base / "output"))
    folder_paths.set_temp_directory(str(base / "temp"))
    folder_paths.set_user_directory(str(base / "user"))
    import nodes
    import server
    from app.assets.manager import default_asset_manager
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    prompt_server = server.PromptServer(loop, default_asset_manager())
    failed = loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
    package_path = str(base / "custom_nodes" / PACK.name).replace(".", "_x_")
    package = sys.modules.get(package_path)
    if package is None or not hasattr(package, "NODE_CLASS_MAPPINGS"):
        raise RuntimeError("PromptForge failed real custom-node initialization; see import diagnostics above.")
    print(f"Isolated CPU init: {len(package.v2.NODE_CLASS_MAPPINGS)} V2 nodes; optional core import failures: {failed}")
    return SimpleNamespace(pack=package, nodes=nodes, folder_paths=folder_paths, base=base,
                           temporary=temporary, loop=loop, server=prompt_server, core_import_failures=failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path)
    parser.add_argument("--test", action="append", help="Run one RuntimeTests method; repeat to select several.")
    options = parser.parse_args()
    os.environ["COMFYUI_ROOT"] = str(options.comfy_root or PACK.parents[1])
    spec = importlib.util.spec_from_file_location("pf_runtime_tests", PACK / "tests/test_v2_runtime.py")
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    suite = (unittest.defaultTestLoader.loadTestsFromNames(["RuntimeTests." + name for name in options.test], tests)
             if options.test else unittest.defaultTestLoader.loadTestsFromModule(tests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
