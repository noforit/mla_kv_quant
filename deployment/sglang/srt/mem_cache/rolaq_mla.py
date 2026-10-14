"""Frozen RoLaQ parameters for the native OSCAR MLA packed cache.

The cache ABI stays unchanged: INT2 codes plus FP32 (scale, zero).
Only the writer's thresholds and gain differ from the OSCAR baseline.
"""

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=8)
def load_manifest(path):
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("recipe") != "learned_shared_latent_rotation_quantizer_v2":
        raise ValueError("RoLaQ requires a frozen v2 learned artifact")
    if manifest.get("family") != "oscar_lloyd":
        raise ValueError("The packed RoLaQ writer currently supports oscar_lloyd only")
    if manifest.get("components") != ["nope"] or manifest.get("group_size") != 128:
        raise ValueError("Expected content-only, group-128 RoLaQ artifact")
    count = manifest["model"]["num_hidden_layers"]
    if set(manifest["layers"]) != {str(i) for i in range(count)}:
        raise ValueError("RoLaQ artifact must cover every decoder layer")
    for layer, item in manifest["layers"].items():
        if item["file"] != f"layer_{int(layer)}.pt":
            raise ValueError("Unexpected rotation filename")
        data = (root / item["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"RoLaQ rotation hash mismatch: layer {layer}")
        t, gain = item["param0"], item["param1"]
        if not (math.isfinite(t) and math.isfinite(gain) and t > 0 and gain > 0):
            raise ValueError(f"Invalid quantizer parameters at layer {layer}")
    return manifest


def load_layer_quantizers(path, rotation_path, group_size, width, layer_ids):
    if not path:
        return [None for _ in layer_ids]
    if Path(path).resolve() != Path(rotation_path).resolve():
        raise ValueError("Learned quantizer and rotations must come from the same artifact")
    manifest = load_manifest(path)
    if manifest["group_size"] != group_size or manifest["model"]["kv_lora_rank"] != width:
        raise ValueError("RoLaQ artifact does not match the cache dimensions")
    result = []
    for layer in layer_ids:
        item = manifest["layers"][str(layer)]
        result.append((float(item["param0"]), float(item["param1"])))
    return result


def fold_read_rotation(weight, layer_id, rotation_path):
    """Return W_b @ R for derived MLA readers; leave kv_b_proj unchanged.

    W_b is [heads * (key_dim + value_dim), latent_dim]. Splitting this
    temporary yields q @ (W_k R) and o_rot @ (R.T W_v.T). The original
    kv_b_proj continues to consume unrotated latents in MHA prefill.
    """
    if weight.dtype != torch.bfloat16:
        raise ValueError("Read-rotation absorption is validated for BF16 weights only")
    path = Path(rotation_path) / f"layer_{layer_id}.pt"
    rotation = torch.load(path, map_location="cpu", weights_only=True)
    if rotation.shape != (weight.shape[-1], weight.shape[-1]):
        raise ValueError("Read-rotation shape mismatch")
    # Match the packed writer: artifact FP32 -> BF16 -> FP32 effective R.
    effective = rotation.to(device=weight.device, dtype=weight.dtype).float()
    return (weight.float() @ effective).to(weight.dtype)
