"""System readout for the top bar: CPU, memory, GPUs and the ComfyUI disk.

psutil covers CPU, RAM and swap, pynvml the GPUs, and the disk figures come from the
filesystem holding the ComfyUI base directory. Every source is optional; a missing
library or driver drops its group from the payload instead of failing the route.
"""
import shutil
from pathlib import Path

from aiohttp import web
from server import PromptServer

import folder_paths

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
except ImportError:
    pynvml = None

routes = PromptServer.instance.routes

_devices = []
if pynvml is not None:
    try:
        pynvml.nvmlInit()
        _devices = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(pynvml.nvmlDeviceGetCount())]
    except pynvml.NVMLError:
        pynvml = None

if psutil is not None:
    # cpu_percent compares against its previous call, so seed the baseline once.
    psutil.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None, percpu=True)


def _cpu():
    if psutil is None:
        return None
    try:
        return {"percent": psutil.cpu_percent(interval=None),
                "cores": psutil.cpu_percent(interval=None, percpu=True)}
    except (OSError, RuntimeError):
        return None


def _memory():
    if psutil is None:
        return None, None
    try:
        ram = psutil.virtual_memory()
        swap = psutil.swap_memory()
    except (OSError, RuntimeError):
        return None, None
    return {"used": ram.total - ram.available, "total": ram.total}, {"used": swap.used, "total": swap.total}


def _gpus():
    gpus = []
    for device in _devices:
        try:
            memory = pynvml.nvmlDeviceGetMemoryInfo(device)
            usage = pynvml.nvmlDeviceGetUtilizationRates(device)
            name = pynvml.nvmlDeviceGetName(device)
            gpus.append({"name": name.decode() if isinstance(name, bytes) else name,
                         "used": memory.used, "total": memory.total,
                         "utilisation": usage.gpu,
                         "temperature": pynvml.nvmlDeviceGetTemperature(device, pynvml.NVML_TEMPERATURE_GPU)})
        except pynvml.NVMLError:
            continue
    return gpus


def _disk():
    usage = shutil.disk_usage(Path(folder_paths.base_path))
    return {"free": usage.free, "total": usage.total}


@routes.get("/promptforge/resources")
async def _resources(request):
    payload = {}
    cpu = _cpu()
    if cpu:
        payload["cpu"] = cpu
    ram, swap = _memory()
    if ram:
        payload["ram"] = ram
        payload["swap"] = swap
    gpus = _gpus()
    if gpus:
        payload["gpus"] = gpus
    try:
        payload["disk"] = _disk()
    except OSError:
        pass
    return web.json_response(payload)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
