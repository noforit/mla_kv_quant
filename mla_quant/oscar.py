"""OSCAR MLA covariance rotation and official affine-grid INT2 baseline."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import torch

FORMAT_VERSION = 1
RECIPE = "oscar_mla_uniform_no_window"
SOURCE_COMMIT = "19f85e13059de3da60686af4cbdd778b5671d9ff"
NUMERICS = "rotation_input_dtype_roundtrip_fp32_rotate_uniform_qdq_input_dtype_roundtrip_fp32_inverse_input_dtype"
CONFIG_FIELDS = ("model_type", "num_hidden_layers", "kv_lora_rank", "qk_rope_head_dim")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def checkpoint_identity(model_path: str | Path) -> dict[str, Any]:
    """Identify the local checkpoint, allowing filesystem aliases, not other weights.

    Config/index hashes detect changed configuration and shard maps. This is not
    a hash of all weight bytes; the checkpoint must remain immutable after fit.
    """
    root = Path(model_path).expanduser().resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError(f"OSCAR requires a local model directory with config.json: {root}")
    config = json.loads(config_path.read_text())
    index = root / "model.safetensors.index.json"
    return {
        "path": str(root),
        "config_sha256": sha256_file(config_path),
        "weight_index_sha256": sha256_file(index) if index.exists() else None,
        **{key: config.get(key) for key in CONFIG_FIELDS},
    }


def normalized_hadamard(size: int) -> torch.Tensor:
    """FP64 recursion matches OSCAR's build_hadamard, including normalization."""
    if size < 1 or size & size - 1:
        raise ValueError(f"Hadamard size must be a positive power of two: {size}")
    if size == 1:
        return torch.ones(1, 1, dtype=torch.float64)
    h = normalized_hadamard(size // 2)
    return torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0) / 2**0.5


def make_br_perm_matrix(eigenvalues: torch.Tensor) -> torch.Tensor:
    """Official descending-eigenvalue, bit-reversal placement (do not sort U)."""
    width = len(eigenvalues)
    if width < 1 or width & width - 1:
        raise ValueError("OSCAR MLA latent width must be a power of two")
    bits = (width - 1).bit_length()
    order = [int(bin(i)[2:].zfill(bits)[::-1], 2) for i in range(width)]
    sorted_idx = torch.argsort(eigenvalues, descending=True)
    perm = torch.zeros(width, dtype=torch.long)
    for i in range(width):
        perm[order[i]] = sorted_idx[i]
    return torch.eye(width, dtype=torch.float64)[:, perm]


def fit_oscar_rotation(latents: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Fit U_(C.T C/N) @ P @ H_block on CPU in FP64; return FP32."""
    if latents.ndim != 2 or latents.shape[0] < 2:
        raise ValueError("Expected at least two calibration rows [tokens, latent_width]")
    width = latents.shape[1]
    if group_size <= 0 or width % group_size:
        raise ValueError("latent width must be divisible by group_size")
    if width & width - 1:
        raise ValueError("latent width must be a power of two")
    c = latents.detach().cpu().float().double()
    if not torch.isfinite(c).all():
        raise ValueError("Calibration latents contain NaN or infinity")
    second_moment = c.T @ c / c.shape[0]
    second_moment = (second_moment + second_moment.T) / 2
    eigenvalues, basis = torch.linalg.eigh(second_moment)
    permutation = make_br_perm_matrix(eigenvalues)
    h = normalized_hadamard(group_size)
    hblock = torch.block_diag(*[h for _ in range(width // group_size)])
    return (basis @ permutation @ hblock).float().contiguous()


def uniform_int2_qdq(values: torch.Tensor, group_size: int) -> torch.Tensor:
    """Exact torch uniform branch of OSCAR's _fake_quant_int2_groupwise."""
    shape = values.shape
    groups = values.reshape(-1, group_size).float()
    minimum = groups.amin(dim=-1, keepdim=True)
    maximum = groups.amax(dim=-1, keepdim=True)
    scale = torch.where(
        (maximum - minimum).abs() > 1e-08, (maximum - minimum) / 3.0, torch.ones_like(minimum)
    )
    codes = ((groups - minimum) / scale).round().clamp(0.0, 3.0)
    return (codes * scale + minimum).reshape(shape)


LLOYD_MAX_THRESHOLDS = (-0.9810652732849121, 0.0, 0.9810652732849121)
LLOYD_MAX_CENTROIDS = (
    -1.5095585584640503,
    -0.4527800381183624,
    0.4527800381183624,
    1.5095585584640503,
)
LLOYD_MAX_STRETCH = 1.16
OSCAR_LLOYD_CENTERS = tuple(
    (
        LLOYD_MAX_CENTROIDS[0] + i * (LLOYD_MAX_CENTROIDS[3] - LLOYD_MAX_CENTROIDS[0]) / 3
        for i in range(4)
    )
)


def oscar_lloyd_int2_qdq(
    values: torch.Tensor,
    group_size: int,
    threshold: float = LLOYD_MAX_THRESHOLDS[-1],
    gain: float = LLOYD_MAX_STRETCH,
) -> torch.Tensor:
    """Pinned upstream MLA Lloyd branch, including its AFFINE reconstruction.

    Source: commit SOURCE_COMMIT, mla_int2_kv_pool._fake_quant_int2_groupwise.
    The historical project `lloyd_up` reconstructed nonuniform centroids; that
    implementation is preserved under `lloyd_up_legacy` after the baseline fix.
    """
    shape = values.shape
    groups = values.reshape(-1, group_size).float()
    mean = groups.mean(-1, keepdim=True)
    diff = groups - mean
    std = (diff.pow(2).mean(-1, keepdim=True) + 1e-08).sqrt()
    normalized = diff / std
    codes = (
        (normalized >= -threshold).to(torch.uint8)
        + (normalized >= 0.0).to(torch.uint8)
        + (normalized >= threshold).to(torch.uint8)
    )
    span = LLOYD_MAX_CENTROIDS[3] - LLOYD_MAX_CENTROIDS[0]
    scale = span / 3.0 * gain * std
    zero = -LLOYD_MAX_CENTROIDS[0] / (span / 3.0) - mean / scale
    return ((codes.float() - zero) * scale).reshape(shape)


INT2_MODES = ("affine", "lloyd_up")


def oscar_qdq(
    tensor: torch.Tensor, rotation: torch.Tensor, group_size: int, int2_mode: str = "affine"
) -> torch.Tensor:
    rotated = tensor.float() @ rotation
    if int2_mode == "affine":
        quantized = uniform_int2_qdq(rotated, group_size)
    elif int2_mode == "lloyd_up":
        quantized = oscar_lloyd_int2_qdq(rotated, group_size)
    else:
        raise ValueError(f"unsupported INT2 mode: {int2_mode}")
    dequantized = quantized.to(tensor.dtype)
    return (dequantized.float() @ rotation.T).to(tensor.dtype)


def write_rotation_artifact(
    output_dir: str | Path,
    rotations: dict[int, torch.Tensor],
    *,
    model_path: str | Path,
    group_size: int,
    calibration: dict[str, Any],
    recipe: str = RECIPE,
    composition: str = "Rc@P@blockdiag(H_group)",
    numerics: str = NUMERICS,
    manifest_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write new matrices and manifest; refuse to overwrite any prior fit."""
    reserved = {
        "format_version",
        "recipe",
        "source_commit",
        "composition",
        "numerics",
        "bits",
        "group_size",
        "components",
        "windows",
        "lloyd_max",
        "hp_subspace",
        "model",
        "layers",
        "calibration",
        "fitter_sha256",
    }
    if manifest_extra and reserved.intersection(manifest_extra):
        raise ValueError("manifest_extra cannot overwrite reserved fields")
    root = Path(output_dir)
    identity = checkpoint_identity(model_path)
    width = identity["kv_lora_rank"]
    if group_size <= 0 or group_size & group_size - 1 or width % group_size:
        raise ValueError("OSCAR group_size must be a power of two dividing latent width")
    expected = set(range(identity["num_hidden_layers"]))
    if set(rotations) != expected:
        raise ValueError(f"Expected all layer IDs {sorted(expected)}, got {sorted(rotations)}")
    for layer_idx, rotation in rotations.items():
        _validate_matrix(rotation.detach().cpu().float(), width, layer_idx)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Rotation output directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for layer_idx, rotation in sorted(rotations.items()):
        rotation = rotation.detach().cpu().float().contiguous()
        _validate_matrix(rotation, identity["kv_lora_rank"], layer_idx)
        path = root / f"layer_{layer_idx}.pt"
        torch.save(rotation, path)
        files[str(layer_idx)] = {"file": path.name, "sha256": sha256_file(path)}
    manifest = {
        "format_version": FORMAT_VERSION,
        "recipe": recipe,
        "source_commit": SOURCE_COMMIT,
        "composition": composition,
        "numerics": numerics,
        "bits": 2,
        "group_size": group_size,
        "components": ["nope"],
        "windows": False,
        "lloyd_max": False,
        "hp_subspace": False,
        "model": identity,
        "layers": files,
        "calibration": calibration,
        "fitter_sha256": sha256_file(__file__),
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    temp = root / "manifest.json.tmp"
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    temp.replace(root / "manifest.json")
    return manifest


def _validate_matrix(rotation: torch.Tensor, width: int, layer_idx: int) -> None:
    if not isinstance(rotation, torch.Tensor) or rotation.shape != (width, width):
        raise ValueError(f"Layer {layer_idx}: expected [{width}, {width}] rotation")
    if rotation.dtype != torch.float32 or not torch.isfinite(rotation).all():
        raise ValueError(f"Layer {layer_idx}: rotation must be finite FP32")
    residual = (rotation.double().T @ rotation.double() - torch.eye(width)).abs().max().item()
    if residual > 1e-05:
        raise ValueError(f"Layer {layer_idx}: rotation is not orthogonal (residual={residual})")


class OscarMLAQuantizer:
    """Per-layer OSCAR reference quantizer, all tokens INT2, rope untouched."""

    artifact_recipe = RECIPE
    artifact_composition = "Rc@P@blockdiag(H_group)"
    artifact_numerics = NUMERICS

    def __init__(
        self,
        rotation_dir: str | Path,
        bits: int = 2,
        group_size: int = 128,
        int2_mode: str = "affine",
    ):
        if bits != 2:
            raise ValueError("OSCAR-MLA reference supports bits=2 only")
        if int2_mode not in INT2_MODES:
            raise ValueError(f"int2_mode must be one of {INT2_MODES}, got {int2_mode!r}")
        self.int2_mode = int2_mode
        self.root = Path(rotation_dir).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        self.manifest_sha256 = sha256_file(manifest_path)
        m = self.manifest
        if (m.get("format_version"), m.get("recipe"), m.get("numerics")) != (
            FORMAT_VERSION,
            self.artifact_recipe,
            self.artifact_numerics,
        ):
            raise ValueError("Unsupported OSCAR rotation manifest or numerical recipe")
        composition = m.get("composition")
        if self.artifact_composition is None:
            allowed = getattr(self, "supported_compositions", ())
            if composition not in allowed:
                raise ValueError(
                    f"Unsupported OSCAR rotation artifact composition: {composition!r}"
                )
            self.artifact_composition = composition
        elif composition != self.artifact_composition:
            raise ValueError(
                f"OSCAR rotation artifact composition mismatch: got {composition!r}, expected {self.artifact_composition!r}"
            )
        if m.get("bits") != 2 or m.get("group_size") != group_size:
            raise ValueError("OSCAR rotation artifact bits/group_size mismatch")
        if m.get("components") != ["nope"] or any(
            (m.get(key) is not False for key in ("windows", "lloyd_max", "hp_subspace"))
        ):
            raise ValueError(
                "OSCAR artifact must be nope-only, uniform, without windows/HP subspace"
            )
        self.bits = bits
        self.group_size = group_size
        self.width = m["model"]["kv_lora_rank"]
        if group_size <= 0 or group_size & group_size - 1 or self.width % group_size:
            raise ValueError("OSCAR group_size must be a power of two dividing latent width")
        count = m["model"]["num_hidden_layers"]
        if set(m["layers"]) != {str(i) for i in range(count)}:
            raise ValueError("OSCAR artifact has missing or unexpected layer IDs")
        self.rotations: dict[int, torch.Tensor] = {}
        for layer_idx in range(count):
            record = m["layers"][str(layer_idx)]
            if record["file"] != f"layer_{layer_idx}.pt":
                raise ValueError("Unexpected OSCAR layer filename")
            path = self.root / record["file"]
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"Layer {layer_idx}: rotation file hash mismatch")
            rotation = torch.load(path, map_location="cpu", weights_only=True)
            _validate_matrix(rotation, self.width, layer_idx)
            self.rotations[layer_idx] = rotation
        self._device_rotations: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
        self.call_count = self.element_count = 0
        self.first_shapes: list[tuple[int, ...]] = []
        self.layer_calls = {str(i): {"prefill": 0, "decode": 0} for i in range(count)}
        self._validated_configs: list[Any] = []

    def validate_config(self, config: Any) -> None:
        if any((config is previous for previous in self._validated_configs)):
            return
        expected = self.manifest["model"]
        for key in CONFIG_FIELDS:
            if getattr(config, key, None) != expected[key]:
                raise ValueError(
                    f"OSCAR model mismatch: {key}={getattr(config, key, None)!r}, expected {expected[key]!r}"
                )
        model_path = getattr(config, "_name_or_path", "")
        actual = checkpoint_identity(model_path)
        if any((actual[k] != expected[k] for k in ("config_sha256", "weight_index_sha256"))):
            raise ValueError("OSCAR rotations belong to a different or changed local checkpoint")
        self._validated_configs.append(config)

    def quantize_layer(self, tensor: torch.Tensor, layer_idx: int, component: str) -> torch.Tensor:
        if component != "nope":
            raise ValueError("OSCAR-MLA must not quantize rope")
        if layer_idx not in self.rotations:
            raise ValueError(f"No OSCAR rotation for layer {layer_idx}")
        if tensor.ndim < 2 or not tensor.is_floating_point() or tensor.shape[-1] != self.width:
            raise ValueError(f"Expected floating MLA tensor ending in width={self.width}")
        if tensor.numel() == 0:
            return tensor.clone()
        key = (layer_idx, str(tensor.device), tensor.dtype)
        if key not in self._device_rotations:
            self._device_rotations[key] = (
                self.rotations[layer_idx].to(device=tensor.device, dtype=tensor.dtype).float()
            )
        self.call_count += 1
        self.element_count += tensor.numel()
        if len(self.first_shapes) < 4:
            self.first_shapes.append(tuple(tensor.shape))
        phase = "decode" if tensor.shape[-2] == 1 else "prefill"
        self.layer_calls[str(layer_idx)][phase] += 1
        return oscar_qdq(
            tensor, self._device_rotations[key], self.group_size, int2_mode=self.int2_mode
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "recipe": self.artifact_recipe,
            "int2_mode": self.int2_mode,
            "int2_recipe_version": 2,
            "lloyd_reconstruction": (
                "official_mla_affine_ratio_1.16"
                if self.int2_mode in {"lloyd_up", "oscar_lloyd_up"}
                else (
                    "legacy_centroid_ratio_1.16"
                    if self.int2_mode == "lloyd_up_legacy"
                    else "not_upstream_lloyd_up"
                )
            ),
            "rotation_dir": str(self.root),
            "manifest_sha256": self.manifest_sha256,
            "rotation_hashes": {k: v["sha256"] for k, v in self.manifest["layers"].items()},
            "calibration": self.manifest["calibration"],
            "source_commit": self.manifest["source_commit"],
            "composition": self.artifact_composition,
            "numerics": self.artifact_numerics,
            "hadamard_order": self.group_size,
        }
