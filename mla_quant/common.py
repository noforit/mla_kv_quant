"""Portable paths and one common method definition for both evaluation suites."""

from pathlib import Path
import random
import numpy as np
import torch
from . import cache
from .numerics import configure_reference_numerics

MODEL_IDS = {
    "deepseek": "deepseek-ai/DeepSeek-V2-Lite",
    "moonlight": "moonshotai/Moonlight-16B-A3B",
    "minicpm": "openbmb/MiniCPM3-4B",
}
METHODS = ("bf16", "int2", "h128", "kvarn", "oscar", "rolaq", "ours")
WINDOWS = {"nowin": (0, 0), "s4r8": (4, 8), "s4r16": (4, 16)}


def resolve_model(name):
    path = Path(name).expanduser()
    if path.is_dir():
        return str(path.resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_IDS.get(name, name))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    configure_reference_numerics("highest")


def install_method(method, model_type, artifact=None, window="nowin"):
    configure_reference_numerics("highest")
    cache.unpatch_transformers_mla_kv_cache()
    if method == "bf16":
        return
    sink, recent = WINDOWS[window]
    native = {
        "int2": "naive",
        "h128": "quarot",
        "rolaq": "learned_rotation",
        "ours": "learned_rotation",
    }.get(method, method)
    if method in ("oscar", "rolaq", "ours") and not artifact:
        raise ValueError(f"{method} needs --artifact")
    cache.patch_transformers_mla_kv_cache(
        method=native,
        rotation_dir=artifact,
        model_types=(model_type,),
        int2_mode="lloyd_up" if method == "oscar" else "affine",
        hadamard_order=128,
        sink_tokens=sink,
        recent_tokens=recent,
    )
