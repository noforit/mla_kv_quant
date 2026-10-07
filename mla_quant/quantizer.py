"""Hard-forward INT2 calibration, orthogonal corrections, and exported inference."""

from __future__ import annotations
import json
import math
import os
from pathlib import Path
from typing import Any
import torch
from torch.autograd import Function
from .oscar import (
    LLOYD_MAX_CENTROIDS,
    LLOYD_MAX_THRESHOLDS,
    LLOYD_MAX_STRETCH,
    OSCAR_LLOYD_CENTERS,
    checkpoint_identity,
    oscar_lloyd_int2_qdq,
    sha256_file,
)

FORMAT_VERSION = 2
RECIPE = "learned_shared_latent_rotation_quantizer_v2"
LEGACY_RECIPE = "learned_shared_latent_rotation_quantizer_v1"
GRADIENT_RECIPE = "affine_full_round_ste_lloyd_hard_gain_ordered_soft_threshold_v2"
NUMERICS = "fp32_rotation_bf16_roundtrip_fp32_qdq_input_dtype_roundtrip_fp32_inverse_input_dtype"
OSCAR_LLOYD_INIT = (float(LLOYD_MAX_THRESHOLDS[-1]), float(LLOYD_MAX_STRETCH))
LLOYD_BOUNDS = ((0.5, 1.5), (0.8, 1.3))
QUANT_FAMILIES = ("oscar_lloyd",)
OFFICIAL_BASELINE_MODE = "lloyd_up"


def oscar_lloyd_hard(values, group_size, threshold, gain):
    t, g = (float(threshold.detach()), float(gain.detach()))
    if g == float(torch.tensor(LLOYD_MAX_STRETCH, dtype=torch.float32)):
        g = LLOYD_MAX_STRETCH
    return oscar_lloyd_int2_qdq(values, group_size, threshold=t, gain=g)


class _LloydSTE(Function):

    @staticmethod
    def forward(ctx, values, group_size, threshold, gain, temperature, official):
        hard_fn = oscar_lloyd_hard
        hard = hard_fn(values, int(group_size), threshold, gain)
        ctx.save_for_backward(values.float(), threshold.float(), gain.float(), temperature.float())
        ctx.group_size = int(group_size)
        ctx.official = bool(official)
        return hard

    @staticmethod
    def backward(ctx, grad):
        values, threshold, gain, temperature = ctx.saved_tensors
        g = int(ctx.group_size)
        groups = values.reshape(-1, g)
        mean = groups.mean(-1, keepdim=True)
        std = ((groups - mean).square().mean(-1, keepdim=True) + 1e-08).sqrt()
        normalized = (groups - mean) / std
        grad_groups = grad.reshape_as(groups).float()
        centers = OSCAR_LLOYD_CENTERS
        centroids = torch.tensor(centers, dtype=groups.dtype, device=groups.device)
        edges = torch.stack((-threshold, torch.zeros_like(threshold), threshold))
        soft = torch.sigmoid(
            (normalized.unsqueeze(-1) - edges.view(1, 1, -1)) / temperature.clamp_min(0.0001)
        )
        codes = (normalized.unsqueeze(-1) >= edges.view(1, 1, -1)).sum(-1)
        grad_gain = (grad_groups * std * centroids[codes]).sum().reshape_as(gain)
        dsoft = soft * (1 - soft) / temperature.clamp_min(0.0001)
        c0, c1, c2, c3 = centroids
        d_expected = dsoft[..., 0] * (c1 - c0) - dsoft[..., 2] * (c3 - c2)
        grad_threshold = (grad_groups * (std * gain) * d_expected).sum().reshape_as(threshold)
        return (grad.reshape_as(values), None, grad_threshold, grad_gain, None, None)


def quantize_learned(values, group_size, family, first, second, temperature=None):
    if family != "oscar_lloyd":
        raise ValueError("only the official OSCAR affine-grid family is supported")
    if temperature is None:
        temperature = values.new_tensor(0.1, dtype=torch.float32)
    if torch.is_grad_enabled() and (
        first.requires_grad or second.requires_grad or values.requires_grad
    ):
        return _LloydSTE.apply(values, group_size, first, second, temperature, True)
    return oscar_lloyd_hard(values, group_size, first, second)


def _normalize_high_precision_mask(mask: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    """Accept [seq], [batch, seq] or exactly latent.shape[:-1] (batch, 1, seq)."""
    mask = mask.to(device=latent.device, dtype=torch.bool)
    expected = tuple(latent.shape[:-1])
    if mask.ndim == 1 and len(expected) > 1:
        mask = mask.reshape(*[1] * (len(expected) - 1), mask.shape[0])
    elif mask.ndim == 2 and len(expected) == 3:
        mask = mask.unsqueeze(1)
    try:
        mask = torch.broadcast_to(mask, expected)
    except RuntimeError as error:
        raise ValueError(
            f"high precision mask shape {tuple(mask.shape)} does not match {expected}"
        ) from error
    return mask


def rotate_quantize(
    latent: torch.Tensor,
    rotation: torch.Tensor,
    group_size: int,
    family: str,
    first: torch.Tensor,
    second: torch.Tensor,
    temperature: torch.Tensor | None = None,
    high_precision_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """OSCAR cast order, with optional rows returned completely unquantized."""
    if high_precision_mask is not None:
        high_precision_mask = _normalize_high_precision_mask(high_precision_mask, latent)
        if bool(high_precision_mask.all()):
            return latent
    effective = rotation.to(dtype=latent.dtype).float()
    rotated = latent.float() @ effective
    quantized = quantize_learned(rotated, group_size, family, first, second, temperature)
    dequantized = quantized.to(dtype=latent.dtype)
    restored = (dequantized.float() @ effective.T).to(dtype=latent.dtype)
    if high_precision_mask is None:
        return restored
    return torch.where(high_precision_mask.unsqueeze(-1), latent, restored)


def skew_exp_rotation(base: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    skew = raw - raw.T
    delta = torch.matrix_exp(skew.float())
    identity = torch.eye(delta.shape[-1], device=delta.device, dtype=torch.float32)
    for _ in range(2):
        delta = delta @ (1.5 * identity - 0.5 * (delta.T @ delta))
    return base.float() @ delta


def skew_cayley_rotation(base: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    """Cayley transform of a skew-symmetric matrix: exactly orthogonal.

    ``R = (I - A) (I + A)^{-1}`` with ``A = raw - raw.T``.  The transpose is
    solved directly so the forward only needs one linear solve, and the
    result is orthogonal by construction for any finite skew ``A``.
    """
    skew = (raw - raw.T).float()
    identity = torch.eye(skew.shape[-1], device=skew.device, dtype=torch.float32)
    left = (identity + skew).transpose(-1, -2)
    right = (identity - skew).transpose(-1, -2)
    delta = torch.linalg.solve(left, right).transpose(-1, -2)
    return base.float() @ delta


def project_quant_params(family: str, first: torch.Tensor, second: torch.Tensor) -> None:
    if family not in QUANT_FAMILIES:
        raise ValueError(f"unknown quant family {family!r}")
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError("non-finite quantizer parameters")
    bounds = LLOYD_BOUNDS
    with torch.no_grad():
        first.clamp_(*bounds[0])
        second.clamp_(*bounds[1])


def _validate_parameters(family, first, second):
    if family not in QUANT_FAMILIES:
        raise ValueError(f"unknown quant family {family!r}")
    bounds = LLOYD_BOUNDS
    for value, (lo, hi) in zip((first, second), bounds):
        if not math.isfinite(value) or not lo - 1e-06 <= value <= hi + 1e-06:
            raise ValueError("quantizer parameter non-finite or out of bounds")


def _validate_rotation(rotation, width, layer):
    if rotation.dtype != torch.float32 or rotation.shape != (width, width):
        raise ValueError(f"layer {layer} requires a square FP32 rotation")
    if not torch.isfinite(rotation).all():
        raise ValueError(f"layer {layer} rotation contains non-finite values")
    residual = (rotation.double().T @ rotation.double() - torch.eye(width)).abs().max().item()
    if residual > 1e-05:
        raise ValueError(f"layer {layer} rotation not orthogonal, residual={residual}")


def write_learned_artifact(
    output_dir: str | Path,
    rotations: dict[int, torch.Tensor],
    params: dict[int, tuple[float, float]],
    *,
    model_path: str | Path,
    base_rotation_dir: str | Path,
    family: str,
    train_part: str,
    mla_output_weight: float,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite learned artifact: {root}")
    from .artifacts import local_checkpoint_fingerprint, atomic_write_json

    identity = local_checkpoint_fingerprint(model_path)
    width = int(identity["kv_lora_rank"])
    expected = set(range(int(identity["num_hidden_layers"])))
    if set(rotations) != expected or set(params) != expected:
        raise ValueError("learned artifact must contain every layer")
    if train_part not in ("rotation", "quantizer", "joint"):
        raise ValueError("unknown train part")
    if not math.isfinite(mla_output_weight) or mla_output_weight < 0:
        raise ValueError("MLA output weight must be finite and nonnegative")
    for layer in expected:
        _validate_rotation(rotations[layer].detach().cpu(), width, layer)
        _validate_parameters(family, *params[layer])
    root.mkdir(parents=True, exist_ok=False)
    layers = {}
    for layer in range(len(expected)):
        rotation = rotations[layer].detach().cpu().float().contiguous()
        path = root / f"layer_{layer}.pt"
        torch.save(rotation, path)
        first, second = params[layer]
        layers[str(layer)] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "param0": float(first),
            "param1": float(second),
        }
    manifest = {
        "format_version": FORMAT_VERSION,
        "recipe": RECIPE,
        "numerics": NUMERICS,
        "family": family,
        "train_part": train_part,
        "mla_output_weight": mla_output_weight,
        "group_size": 128,
        "components": ["nope"],
        "windows": False,
        "model": identity,
        "base_rotation_dir": str(Path(base_rotation_dir).resolve()),
        "layers": layers,
        "calibration": calibration,
        "builder_sha256": sha256_file(__file__),
        "gradient_recipe": calibration.get("gradient_recipe", GRADIENT_RECIPE),
    }
    atomic_write_json(root / "manifest.json", manifest)
    return manifest


class LearnedRotationQuantizer:
    """Inference quantizer. Loads the exported matrix; no matrix exponential."""

    def __init__(self, artifact_dir: str | Path):
        self.root = Path(artifact_dir).expanduser().resolve()
        path = self.root / "manifest.json"
        self.manifest = json.loads(path.read_text())
        self.manifest_sha256 = sha256_file(path)
        if (self.manifest.get("format_version"), self.manifest.get("recipe")) not in (
            (FORMAT_VERSION, RECIPE),
            (1, LEGACY_RECIPE),
        ):
            raise ValueError("not a learned rotation artifact")
        if self.manifest.get("numerics") != NUMERICS:
            raise ValueError("learned artifact numerical recipe mismatch")
        if self.manifest.get("group_size") != 128 or self.manifest.get("components") != ["nope"]:
            raise ValueError("learned artifact must be nope-only group 128")
        self.family = self.manifest["family"]
        self.width = int(self.manifest["model"]["kv_lora_rank"])
        self.group_size = 128
        if self.width <= 0 or self.width % self.group_size:
            raise ValueError("latent width is incompatible with group size")
        count = int(self.manifest["model"]["num_hidden_layers"])
        if set(self.manifest["layers"]) != {str(i) for i in range(count)}:
            raise ValueError("learned artifact has missing or unexpected layers")
        if self.manifest.get("windows") is not False:
            raise ValueError("window lifecycle is not supported by this artifact")
        self.rotations = {}
        self.params = {}
        for layer, record in self.manifest["layers"].items():
            if record["file"] != f"layer_{int(layer)}.pt":
                raise ValueError("unexpected learned rotation filename")
            file = self.root / record["file"]
            if sha256_file(file) != record["sha256"]:
                raise ValueError(f"learned rotation hash mismatch at layer {layer}")
            rotation = torch.load(file, map_location="cpu", weights_only=True)
            _validate_rotation(rotation, self.width, layer)
            _validate_parameters(self.family, record["param0"], record["param1"])
            self.rotations[int(layer)] = rotation
            self.params[int(layer)] = (float(record["param0"]), float(record["param1"]))
        self._cache: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
        self.high_precision_mask: torch.Tensor | None = None
        self.layer_calls = {str(i): {"prefill": 0, "decode": 0} for i in self.rotations}
        self.call_count = 0
        self.element_count = 0
        self.first_shapes: list[tuple[int, ...]] = []
        self._validated_configs: list[Any] = []

    def validate_config(self, config: Any) -> None:
        from .numerics import validate_artifact_numerics

        validate_artifact_numerics(self.manifest)
        if any((config is previous for previous in self._validated_configs)):
            return
        expected = self.manifest["model"]
        for key in ("model_type", "num_hidden_layers", "kv_lora_rank", "qk_rope_head_dim"):
            if getattr(config, key, None) != expected[key]:
                raise ValueError(f"learned artifact model mismatch: {key}")
        from .artifacts import local_checkpoint_fingerprint

        model_path = getattr(config, "_name_or_path", "")
        if not model_path:
            raise ValueError("learned artifact validation needs the model checkpoint path")
        actual = local_checkpoint_fingerprint(model_path)
        identity_fields = ["config_sha256", "weight_index_sha256"]
        if "weight_files" in expected:
            sizes = lambda record: {name: value["size"] for name, value in record.items()}
            if sizes(actual["weight_files"]) != sizes(expected["weight_files"]):
                raise ValueError("learned artifact shard names/sizes differ from the checkpoint")
        if any((actual.get(key) != expected.get(key) for key in identity_fields)):
            raise ValueError("learned artifact belongs to a different checkpoint")
        self._validated_configs.append(config)

    def quantize_layer(self, tensor: torch.Tensor, layer_idx: int, component: str) -> torch.Tensor:
        if component != "nope":
            raise ValueError("learned rotation quantizes nope only")
        if tensor.ndim < 2 or tensor.shape[-1] != self.width or (not tensor.is_floating_point()):
            raise ValueError("invalid MLA latent shape or dtype")
        if layer_idx not in self.rotations:
            raise ValueError(f"missing learned rotation for layer {layer_idx}")
        key = (layer_idx, str(tensor.device), tensor.dtype)
        if key not in self._cache:
            self._cache[key] = self.rotations[layer_idx].to(tensor.device, tensor.dtype).float()
        phase = "decode" if tensor.shape[-2] == 1 else "prefill"
        self.layer_calls[str(layer_idx)][phase] += 1
        self.call_count += 1
        self.element_count += tensor.numel()
        if len(self.first_shapes) < 4:
            self.first_shapes.append(tuple(tensor.shape))
        first, second = self.params[layer_idx]
        p0 = torch.tensor(first, dtype=torch.float32, device=tensor.device)
        p1 = torch.tensor(second, dtype=torch.float32, device=tensor.device)
        mask = self.high_precision_mask
        if mask is not None and mask.shape[-1] != tensor.shape[-2]:
            mask = mask[..., -tensor.shape[-2] :]
        return rotate_quantize(
            tensor, self._cache[key], self.group_size, self.family, p0, p1, high_precision_mask=mask
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "recipe": self.manifest["recipe"],
            "family": self.family,
            "train_part": self.manifest.get("train_part"),
            "mla_output_weight": self.manifest.get("mla_output_weight"),
            "rotation_dir": str(self.root),
            "manifest_sha256": self.manifest_sha256,
            "rotation_hashes": {
                layer: record["sha256"] for layer, record in self.manifest["layers"].items()
            },
            "calibration": self.manifest.get("calibration", {}),
            "numerics": self.manifest.get("numerics"),
            "components": ["nope"],
        }
